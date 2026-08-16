"""
Streamlit chat UI.

- Ask a question -> retrieves transcript context -> generates a grounded
  answer -> shows source cards with episode title, timestamp, a clickable
  "jump to moment" deep link, and the excerpt text.
- 👍/👎 feedback per answer, logged to Postgres for the monitoring
  dashboard.

Note on structure: every turn is rendered from `st.session_state.history`,
including the one just answered. A button click triggers a rerun in which
`st.chat_input` returns None, so anything rendered inside an `if question:`
block would vanish before the click could be handled -- which is why the
feedback buttons live in the history loop with `on_click` callbacks.
"""
import os
import time

import psycopg2
import streamlit as st
from dotenv import load_dotenv

from rag.prompts import generate_answer, rewrite_query
from rag.retrieval import retrieve
from rag.tools import build_deeplink, format_timestamp

load_dotenv()

st.set_page_config(page_title="Podcast Knowledge Explorer", page_icon="🎙️", layout="centered")
st.title("🎙️ Podcast Knowledge Explorer")
st.caption("Ask a question about the ingested episodes — get a grounded answer with exact timestamps.")


def get_pg():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", 5432),
        dbname=os.environ.get("POSTGRES_DB", "podcast_explorer"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )


def log_query(question: str, rewritten: str, answer: str, sources: list[dict], latency_ms: float, query_id: int):
    pg = get_pg()
    cur = pg.cursor()
    cur.execute(
        """INSERT INTO query_log (id, question, rewritten_question, answer, num_sources, latency_ms)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (query_id, question, rewritten, answer, len(sources), latency_ms),
    )
    pg.commit()
    cur.close()
    pg.close()


def log_feedback(query_id: int, is_positive: bool):
    pg = get_pg()
    cur = pg.cursor()
    cur.execute(
        "INSERT INTO feedback (query_id, is_positive) VALUES (%s, %s)",
        (query_id, is_positive),
    )
    pg.commit()
    cur.close()
    pg.close()


def submit_feedback(turn_idx: int, is_positive: bool):
    """on_click callback -- runs at the start of the rerun the click causes,
    so the write happens before the page is redrawn."""
    turn = st.session_state.history[turn_idx]
    try:
        log_feedback(turn["query_id"], is_positive)
        turn["feedback"] = is_positive
    except Exception as e:  # e.g. Postgres down, or the query row never landed
        turn["feedback_error"] = str(e)


def render_sources(sources: list[dict]):
    for src in sources:
        link = build_deeplink(src["video_id"], src["start"])
        ts = format_timestamp(src["start"])
        with st.expander(f"▶ {src.get('title', src['video_id'])} — {ts}"):
            st.markdown(f"[Jump to this moment]({link})")
            st.write(src["text"])


def render_feedback(turn_idx: int, turn: dict):
    if turn.get("feedback") is not None:
        st.caption(
            "👍 Thanks for the feedback!" if turn["feedback"]
            else "👎 Thanks — we'll use this to improve retrieval."
        )
        return

    col1, col2 = st.columns(2)
    col1.button("👍 Helpful", key=f"up_{turn['query_id']}",
                on_click=submit_feedback, args=(turn_idx, True))
    col2.button("👎 Not helpful", key=f"down_{turn['query_id']}",
                on_click=submit_feedback, args=(turn_idx, False))

    if turn.get("feedback_error"):
        st.caption(f"(feedback not logged: {turn['feedback_error']})")


if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Settings")
    strategy = st.selectbox(
        "Retrieval strategy",
        ["hybrid_rerank", "hybrid", "parent_document", "flat"],
        index=0,
        help="Controls how transcript context is retrieved. See eval/results.md for a comparison.",
    )
    use_rewrite = st.checkbox("Rewrite query before retrieval", value=True)

question = st.chat_input("e.g. What protocol does he recommend for morning sunlight?")

if question:
    with st.spinner("Searching transcripts..."):
        start = time.time()
        search_query = rewrite_query(question) if use_rewrite else question
        sources = retrieve(search_query, strategy=strategy)
        answer = generate_answer(question, sources)
        latency_ms = (time.time() - start) * 1000

    query_id = int(time.time() * 1000)
    turn = {
        "query_id": query_id,
        "question": question,
        "answer": answer,
        "sources": sources,
        "feedback": None,
    }

    try:
        log_query(question, search_query, answer, sources, latency_ms, query_id)
    except Exception as e:
        turn["log_error"] = str(e)

    st.session_state.history.append(turn)

for idx, turn in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        render_sources(turn["sources"])
        if turn.get("log_error"):
            st.caption(f"(monitoring log skipped: {turn['log_error']})")
        render_feedback(idx, turn)
