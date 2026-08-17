"""
Fetch raw, timestamped transcripts for the episodes listed in episodes.yaml.

Primary path: youtube-transcript-api (fast, free, works for any video that
has captions -- auto-generated or uploaded).

Fallback path: download the audio with yt-dlp and transcribe it locally with
faster-whisper. Set WHISPER_FALLBACK=1 to enable it. It covers two cases:
videos with no captions at all, and -- the common one -- YouTube blocking the
caption endpoint for this IP, which it does independently of video delivery.
Downloads keep working when /api/timedtext returns 429, so this path is the
only one that does not depend on IP reputation.

It is opt-in because it is expensive: roughly 0.6x realtime per episode on
CPU with the `small` model, i.e. longer than the episode itself. See
whisper_model_size() for the speed/accuracy tradeoff.

Speaker diarization via pyannote-audio can be layered on top of the Whisper
segments if you want speaker labels for multi-host episodes -- left as a
documented extension point rather than wired in by default, since it requires
a HuggingFace token and a GPU for reasonable speed.
"""
import json
import os
import re
import shutil
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml
from dotenv import load_dotenv
# Guaranteed present: youtube-transcript-api itself is built on requests.
from requests.exceptions import RetryError
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    RequestBlocked,
)
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

# This module reads several settings from the environment (the delay below,
# and the proxy credentials in build_proxy_config), so it has to load .env
# itself. It used to rely on ingest_pipeline's load_dotenv(), which does not
# work: running this module standalone never imports ingest_pipeline at all,
# and flow.py imports THIS module first -- so in both cases the proxy vars
# were invisible and build_proxy_config() silently returned None.
load_dotenv()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_transcripts"
RAW_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_DIR = Path(__file__).parent.parent / "data" / "audio"

DEFAULT_FETCH_DELAY_SECONDS = 10.0
DEFAULT_WHISPER_MODEL = "base"

# Tried in order. YouTube gates its high-bitrate audio behind an "n challenge"
# that yt-dlp can only answer with a solver script it downloads at runtime
# (--remote-components ejs). Without it those streams return 403 while the
# lowest-bitrate stream is served normally -- so falling back to it turns a
# dead episode into a live one. The quality loss is irrelevant: everything is
# downmixed to 16 kHz mono for Whisper regardless of source bitrate.
AUDIO_FORMAT_CHAIN = ("bestaudio/best", "worstaudio/worst")


def fetch_delay_seconds() -> float:
    """Seconds to wait between fetches. YouTube rate-limits bursts from one IP.

    Read per call rather than bound at import: as a module-level constant it
    was evaluated before load_dotenv() had run, pinning it to the built-in
    default no matter what .env said.
    """
    return float(os.environ.get("FETCH_DELAY_SECONDS", DEFAULT_FETCH_DELAY_SECONDS))


class TranscriptsUnavailable(RuntimeError):
    """No transcripts could be fetched, so there is nothing to ingest.

    Its own type so the flow can report it as a plain one-line failure --
    it's an expected, actionable condition, not a crash. Anything else still
    propagates with a full traceback.
    """

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be", "www.youtu.be",
}


def normalize_video_id(value: str) -> str:
    """Accept either a bare 11-character video ID or any common YouTube URL.

    Worth normalizing rather than just documenting: video_id isn't only used
    to fetch the transcript, it's substituted straight into build_deeplink().
    A URL slipping through would yield `watch?v=https://youtube.com/...` on
    every source card -- i.e. a broken "jump to moment" link everywhere.
    """
    value = value.strip()
    if _VIDEO_ID_RE.match(value):
        return value

    parsed = urlparse(value if "//" in value else f"https://{value}")
    host = parsed.netloc.lower()
    if host not in _YOUTUBE_HOSTS:
        raise ValueError(f"Not a recognised YouTube video ID or URL: {value!r}")

    if host.endswith("youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif parsed.path.startswith(("/shorts/", "/embed/", "/live/", "/v/")):
        candidate = parsed.path.split("/")[2]
    else:
        candidate = (parse_qs(parsed.query).get("v") or [""])[0]

    if not _VIDEO_ID_RE.match(candidate):
        raise ValueError(f"Could not extract a YouTube video ID from {value!r}")
    return candidate


def load_episode_list(path: str = "episodes.yaml") -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["episodes"]


def build_proxy_config():
    """Optional, opt-in. Returns None unless proxy env vars are set.

    YouTube blocks IPs that make too many transcript requests, and blocks
    most cloud-provider ranges outright. Routing through a residential proxy
    is the library's documented workaround; see .env.example.
    """
    ws_user = os.environ.get("WEBSHARE_PROXY_USERNAME")
    ws_pass = os.environ.get("WEBSHARE_PROXY_PASSWORD")
    if ws_user and ws_pass:
        return WebshareProxyConfig(proxy_username=ws_user, proxy_password=ws_pass)

    http_url = os.environ.get("PROXY_HTTP_URL")
    https_url = os.environ.get("PROXY_HTTPS_URL")
    if http_url or https_url:
        return GenericProxyConfig(http_url=http_url, https_url=https_url)

    return None


def fetch_transcript(video_id: str) -> list[dict]:
    """Returns a list of {text, start, duration} segments, in seconds.

    youtube-transcript-api 1.x replaced the old static
    `YouTubeTranscriptApi.get_transcript(...)` with an instance `.fetch(...)`
    returning a FetchedTranscript. `.to_raw_data()` converts it back to the
    plain list-of-dicts shape the chunking step already expects, so nothing
    downstream had to change.

    RequestBlocked/IpBlocked deliberately propagate -- the caller decides
    whether to fall back to Whisper or stop the run.
    """
    api = YouTubeTranscriptApi(proxy_config=build_proxy_config())
    try:
        return api.fetch(video_id).to_raw_data()
    except (TranscriptsDisabled, NoTranscriptFound):
        return []
    except RetryError as e:
        # A block reaches the caller as one of two unrelated exception types,
        # depending on whether a proxy is configured:
        #   no proxy -> the library inspects the 429 itself and raises IpBlocked
        #   proxy    -> it mounts a urllib3 Retry(status_forcelist=[429]) first,
        #               and once that is exhausted urllib3/requests raises
        #               RetryError before the library ever sees the response.
        # Normalised here so callers only have to handle RequestBlocked.
        raise RequestBlocked(video_id) from e


def whisper_fallback_enabled() -> bool:
    return os.environ.get("WHISPER_FALLBACK") == "1"


def whisper_model_size() -> str:
    """Whisper model used for the fallback. Measured on a 16-core CPU:

        tiny   2.7x realtime, noticeably worse wording
        base   1.6x realtime, matches `small` on clean speech
        small  0.7x realtime

    `base` is the default because transcript quality only has to be good
    enough for retrieval and quoting, and `small` more than doubles the
    runtime of an already long job.
    """
    return os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)


def available_js_runtimes() -> dict:
    """Enable whichever JavaScript runtimes are actually installed.

    YouTube requires deciphering a signature computed in JS, which yt-dlp
    delegates to an external runtime. It enables ONLY `deno` by default, so on
    a machine with node or bun but no deno every download is unsigned -- which
    YouTube rejects as `HTTP Error 403: Forbidden` on the media URL. The
    connection to JavaScript is invisible from the error, and it hits only some
    videos, so it looks like a per-video quirk rather than a missing tool.

    Ordered as yt-dlp prioritises them (deno > node > bun); it picks the
    highest-priority runtime that is both enabled and available.
    """
    return {name: {} for name in ("deno", "node", "bun") if shutil.which(name)}


def download_audio(video_id: str, attempts: int = 3) -> Path:
    """Download the audio-only stream and extract it to 16 kHz mono wav.

    Whisper resamples to 16 kHz mono internally, so doing it here costs
    nothing and keeps a 3-hour episode near 350 MB instead of ~2 GB.

    Retried because YouTube's 403s here are intermittent, not deterministic:
    an episode that downloaded fine minutes earlier can 403 on the next
    attempt and succeed again after a pause. Unlike the caption endpoint's
    429s, backing off briefly does help, so it is worth a few tries before
    giving up on the episode.
    """
    import yt_dlp
    from yt_dlp.utils import DownloadError

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    stem = AUDIO_DIR / video_id
    opts = {
        "outtmpl": f"{stem}.%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": {"extractaudio": ["-ac", "1", "-ar", "16000"]},
        "quiet": True,
        "noprogress": True,
    }
    # Only set when something was found: an empty dict would *disable* the
    # default rather than fall back to it.
    runtimes = available_js_runtimes()
    if runtimes:
        opts["js_runtimes"] = runtimes

    path = stem.with_suffix(".wav")

    # Reuse an audio file left behind by an interrupted run. Safe to trust:
    # yt-dlp downloads to a .part file and ffmpeg only writes the .wav once
    # extraction finishes, so the .wav existing means it is complete. Without
    # this, a run killed during transcription re-downloads ~300 MB it already
    # has -- and transcription is exactly when a long job tends to be killed.
    if path.exists():
        print(f"  reusing already-downloaded audio ({path.stat().st_size / 1e6:.0f} MB)",
              flush=True)
        return path

    url = f"https://www.youtube.com/watch?v={video_id}"
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        for fmt in AUDIO_FORMAT_CHAIN:
            try:
                with yt_dlp.YoutubeDL({**opts, "format": fmt}) as ydl:
                    ydl.download([url])
                if path.exists():
                    return path
            except DownloadError as e:
                last_error = e
                # Clear any partial output so the next try starts clean.
                path.unlink(missing_ok=True)
                if fmt != AUDIO_FORMAT_CHAIN[-1]:
                    print(f"  format '{fmt}' refused; trying a lower-bitrate "
                          f"stream", flush=True)

        if attempt < attempts:
            backoff = 30 * attempt
            print(f"  download attempt {attempt}/{attempts} failed on every "
                  f"format; retrying in {backoff}s", flush=True)
            time.sleep(backoff)

    raise last_error if last_error else RuntimeError(
        f"yt-dlp produced no output for {video_id}")


def transcribe_with_whisper(video_id: str, audio_path: str | Path) -> list[dict]:
    """Transcribe a downloaded audio file into {text, start, duration} segments,
    the same shape fetch_transcript() returns, so chunking is unaffected."""
    # ctranslate2 and onnxruntime each bundle their own OpenMP runtime, and
    # loading both aborts the process on Windows ("OMP: Error #15"). Verified
    # that the documented override transcribes a known clip correctly rather
    # than silently corrupting it. Must be set before faster_whisper imports.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from faster_whisper import WhisperModel

    model = WhisperModel(
        whisper_model_size(),
        device="cpu",
        compute_type="int8",
        cpu_threads=os.cpu_count() or 4,
    )
    segments, info = model.transcribe(str(audio_path), word_timestamps=False)

    # `segments` is a generator: nothing is transcribed until it is consumed.
    # Consuming it in a loop rather than a list comprehension lets us report
    # progress as it goes -- on a 2.5-hour episode the alternative is a silent
    # hour and a half with no way to tell work from a hang.
    started = time.time()
    out = []
    last_report = 0.0
    for seg in segments:
        out.append({
            "text": seg.text.strip(),
            "start": seg.start,
            "duration": seg.end - seg.start,
        })
        elapsed = time.time() - started
        # Throttled to once a minute: per-segment output would be thousands of
        # lines, and this has to stay readable in a log file too.
        if elapsed - last_report >= 60:
            last_report = elapsed
            speed = seg.end / elapsed if elapsed else 0
            remaining = (info.duration - seg.end) / speed / 60 if speed else 0
            print(f"    {seg.end / info.duration * 100:5.1f}%  "
                  f"({seg.end / 60:.0f}/{info.duration / 60:.0f} min audio, "
                  f"{speed:.2f}x realtime, ~{remaining:.0f} min left)",
                  flush=True)

    elapsed = time.time() - started
    print(f"  transcribed {info.duration / 60:.0f} min of audio in "
          f"{elapsed / 60:.0f} min ({info.duration / elapsed:.2f}x realtime)",
          flush=True)
    return out


def fetch_via_whisper(video_id: str) -> list[dict]:
    """Download + transcribe, removing the audio afterwards.

    The wav is deleted because it is pure intermediate state -- the JSON
    transcript is what the pipeline consumes, and six episodes of retained
    audio would be a couple of GB for nothing.
    """
    print(f"  downloading audio for {video_id} (yt-dlp)...", flush=True)
    audio_path = download_audio(video_id)
    size_mb = audio_path.stat().st_size / 1e6
    print(f"  got {size_mb:.0f} MB; transcribing with Whisper "
          f"'{whisper_model_size()}' -- this takes a while", flush=True)
    try:
        return transcribe_with_whisper(video_id, audio_path)
    finally:
        audio_path.unlink(missing_ok=True)


def try_fetch_via_whisper(video_id: str) -> list[dict] | None:
    """fetch_via_whisper, but returning None instead of raising on a download
    failure, so one bad episode does not abandon the rest of the run.

    Worth the extra layer: a single 403 previously propagated out of the
    Prefect task and killed a multi-hour job with three episodes still to go,
    discarding work that had nothing to do with the failure.
    """
    from yt_dlp.utils import DownloadError

    try:
        return fetch_via_whisper(video_id)
    except DownloadError as e:
        print(f"  yt-dlp could not download {video_id}: {e}", flush=True)
        print("  skipping this episode; re-run to retry just this one", flush=True)
        return None


def write_transcript(out_path: Path, video_id: str, ep: dict, segments: list[dict]) -> None:
    """Shared by the caption and Whisper paths so both produce identical JSON."""
    with open(out_path, "w") as f:
        json.dump({"video_id": video_id, "meta": ep, "segments": segments}, f, indent=2)
    print(f"  wrote {len(segments)} segments -> {out_path}")


def main():
    episodes = load_episode_list()
    delay = fetch_delay_seconds()
    proxy = build_proxy_config()
    fetched = 0
    skipped: list[str] = []

    # Stated up front: a mistyped proxy variable otherwise looks exactly like
    # a working one until the run dies on the same block as before.
    print(f"[fetch] proxy: {type(proxy).__name__ if proxy else 'none (direct connection)'}")
    if whisper_fallback_enabled():
        print(f"[fetch] Whisper fallback: enabled (model '{whisper_model_size()}')")

    for i, ep in enumerate(episodes):
        vid = ep["video_id"]
        out_path = RAW_DIR / f"{vid}.json"
        if out_path.exists():
            print(f"[skip] {vid} already fetched")
            continue

        # Space out requests. Fetching several long episodes back-to-back is
        # what gets an IP rate-limited by YouTube in the first place.
        if i > 0:
            time.sleep(delay)

        print(f"[fetch] {vid} -- {ep.get('title', '')}")
        try:
            segments = fetch_transcript(vid)
        except RequestBlocked:
            # Covers IpBlocked too. The caption endpoint being blocked says
            # nothing about video delivery, so when the Whisper fallback is
            # enabled this is recoverable per-episode rather than fatal.
            if whisper_fallback_enabled():
                print("  caption endpoint blocked for this IP -- falling back to Whisper")
                segments = try_fetch_via_whisper(vid)
                if segments is None:
                    skipped.append(vid)
                    continue
                write_transcript(out_path, vid, ep, segments)
                fetched += 1
                continue

            # Without the fallback there is nothing else to try. Retrying
            # immediately extends the block, so stop here -- everything
            # fetched so far is on disk and the next run skips it.
            remaining = len(episodes) - i
            # The advice differs sharply depending on whether a proxy is in
            # play: with one configured the library has already retried
            # through several exit IPs, so "wait for the block to lapse" is
            # wrong -- the account or its IP pool is the thing at fault.
            if proxy:
                remedy = (
                    "  The proxy is configured but its exit IPs are blocked too. Check the "
                    "account has residential (not datacenter) proxies and available "
                    "bandwidth -- YouTube blocks datacenter ranges outright."
                )
            else:
                remedy = (
                    "  Wait for the block to lapse (usually minutes to hours), raise "
                    "FETCH_DELAY_SECONDS, or configure a proxy -- see .env.example."
                )
            print(
                f"\n  YouTube is rate-limiting this IP -- stopping with {remaining} "
                f"episode(s) still to fetch.\n"
                f"  {fetched} episode(s) saved to {RAW_DIR}; re-running resumes "
                f"from where this left off.\n"
                f"{remedy}"
            )
            break

        if not segments:
            # Captions are genuinely absent (disabled or none published),
            # which is the other case the Whisper fallback exists for.
            if not whisper_fallback_enabled():
                print(f"  no captions found for {vid}; set WHISPER_FALLBACK=1 to "
                      f"transcribe it locally instead")
                continue
            print(f"  no captions for {vid} -- falling back to Whisper")
            segments = try_fetch_via_whisper(vid)
            if segments is None:
                skipped.append(vid)
                continue

        write_transcript(out_path, vid, ep, segments)
        fetched += 1

    on_disk = len(list(RAW_DIR.glob("*.json")))
    print(f"\n[fetch] {fetched} newly fetched; {on_disk}/{len(episodes)} episodes now on disk.")
    if skipped:
        print(f"[fetch] {len(skipped)} skipped after download failures: {', '.join(skipped)}")

    # Fail loudly when there is nothing at all to ingest. A partial fetch is
    # fine -- the next run resumes -- but a run that produced no corpus must
    # not report success, or a scheduled pipeline reports green while doing
    # nothing at all.
    if on_disk == 0:
        raise TranscriptsUnavailable(
            "No transcripts could be fetched, so there is nothing to ingest. "
            "If this was an IP block, wait for it to lapse and re-run, raise "
            "FETCH_DELAY_SECONDS, configure a proxy, or set WHISPER_FALLBACK=1 "
            "to transcribe the audio locally instead (see .env.example)."
        )

    return on_disk


if __name__ == "__main__":
    main()
