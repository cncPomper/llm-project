"""
Fetch raw, timestamped transcripts for the episodes listed in episodes.yaml.

Primary path: youtube-transcript-api (fast, free, works for any video that
has captions -- auto-generated or uploaded).

Fallback path: if no captions are available, download audio and run
faster-whisper locally (see `transcribe_with_whisper`). Speaker diarization
via pyannote-audio can be layered on top of the Whisper segments if you
want speaker labels for multi-host episodes -- left as a documented
extension point below rather than wired in by default, since it requires
a HuggingFace token and a GPU for reasonable speed.
"""
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    RequestBlocked,
)
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_transcripts"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Seconds to wait between fetches. YouTube rate-limits bursts from one IP.
FETCH_DELAY_SECONDS = float(os.environ.get("FETCH_DELAY_SECONDS", "3"))


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

    RequestBlocked/IpBlocked deliberately propagate -- the caller stops the
    run rather than burning through the remaining episodes against a block.
    """
    api = YouTubeTranscriptApi(proxy_config=build_proxy_config())
    try:
        return api.fetch(video_id).to_raw_data()
    except (TranscriptsDisabled, NoTranscriptFound):
        return []


def transcribe_with_whisper(video_id: str, audio_path: str) -> list[dict]:
    """
    Fallback for videos without captions. Requires the audio to already be
    downloaded to `audio_path` (e.g. via yt-dlp, not included here to keep
    this script dependency-light and ToS-friendly -- plug in your own
    downloader).
    """
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=False)
    return [
        {"text": seg.text.strip(), "start": seg.start, "duration": seg.end - seg.start}
        for seg in segments
    ]


def main():
    episodes = load_episode_list()
    fetched = 0

    for i, ep in enumerate(episodes):
        vid = ep["video_id"]
        out_path = RAW_DIR / f"{vid}.json"
        if out_path.exists():
            print(f"[skip] {vid} already fetched")
            continue

        # Space out requests. Fetching several long episodes back-to-back is
        # what gets an IP rate-limited by YouTube in the first place.
        if i > 0:
            time.sleep(FETCH_DELAY_SECONDS)

        print(f"[fetch] {vid} -- {ep.get('title', '')}")
        try:
            segments = fetch_transcript(vid)
        except RequestBlocked:
            # Covers IpBlocked too. Retrying immediately extends the block,
            # so stop here -- everything fetched so far is already on disk
            # and the next run skips it.
            remaining = len(episodes) - i
            print(
                f"\n  YouTube is rate-limiting this IP -- stopping with {remaining} "
                f"episode(s) still to fetch.\n"
                f"  {fetched} episode(s) saved to {RAW_DIR}; re-running resumes "
                f"from where this left off.\n"
                f"  Wait for the block to lapse (usually minutes to hours), raise "
                f"FETCH_DELAY_SECONDS, or configure a proxy -- see .env.example."
            )
            break

        if not segments:
            print(f"  no captions found for {vid}; use transcribe_with_whisper() "
                  f"with a downloaded audio file as a fallback")
            continue

        with open(out_path, "w") as f:
            json.dump({"video_id": vid, "meta": ep, "segments": segments}, f, indent=2)
        fetched += 1
        print(f"  wrote {len(segments)} segments -> {out_path}")

    on_disk = len(list(RAW_DIR.glob("*.json")))
    print(f"\n[fetch] {fetched} newly fetched; {on_disk}/{len(episodes)} episodes now on disk.")

    # Fail loudly when there is nothing at all to ingest. A partial fetch is
    # fine -- the next run resumes -- but a run that produced no corpus must
    # not report success, or a scheduled pipeline reports green while doing
    # nothing at all.
    if on_disk == 0:
        raise TranscriptsUnavailable(
            "No transcripts could be fetched, so there is nothing to ingest. "
            "If this was an IP block, wait for it to lapse and re-run, raise "
            "FETCH_DELAY_SECONDS, or configure a proxy (see .env.example)."
        )

    return on_disk


if __name__ == "__main__":
    main()
