"""
Small deterministic tools the app/agent can call. These are the pieces
that shouldn't be left to LLM generation (exact URLs, exact timestamps).
"""
import os

import psycopg2


def build_deeplink(video_id: str, start_seconds: float) -> str:
    """https://www.youtube.com/watch?v=<id>&t=<seconds>s -- jumps straight
    to the moment in the video the answer was drawn from."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start_seconds)}s"


def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def list_episode_topics(video_id: str, limit: int = 10) -> list[str]:
    """Rough topic list for an episode, for a 'browse this episode' UI
    affordance -- pulls distinct parent-window excerpts as topic seeds."""
    pg = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", 5432),
        dbname=os.environ.get("POSTGRES_DB", "podcast_explorer"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )
    cur = pg.cursor()
    cur.execute(
        "SELECT text FROM parents WHERE video_id = %s ORDER BY start_sec LIMIT %s",
        (video_id, limit),
    )
    rows = [r[0][:120] + "..." for r in cur.fetchall()]
    cur.close()
    pg.close()
    return rows
