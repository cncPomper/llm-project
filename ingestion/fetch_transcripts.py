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
from pathlib import Path

import yaml
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_transcripts"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_episode_list(path: str = "episodes.yaml") -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["episodes"]


def fetch_transcript(video_id: str) -> list[dict]:
    """Returns a list of {text, start, duration} segments, in seconds."""
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
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
    for ep in episodes:
        vid = ep["video_id"]
        out_path = RAW_DIR / f"{vid}.json"
        if out_path.exists():
            print(f"[skip] {vid} already fetched")
            continue

        print(f"[fetch] {vid} -- {ep.get('title', '')}")
        segments = fetch_transcript(vid)

        if not segments:
            print(f"  no captions found for {vid}; use transcribe_with_whisper() "
                  f"with a downloaded audio file as a fallback")
            continue

        with open(out_path, "w") as f:
            json.dump({"video_id": vid, "meta": ep, "segments": segments}, f, indent=2)
        print(f"  wrote {len(segments)} segments -> {out_path}")


if __name__ == "__main__":
    main()
