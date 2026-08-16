"""
Streamlit chat UI.

- Ask a question -> retrieves transcript context -> generates a grounded
  answer -> shows source cards with episode title, timestamp, a clickable
  "jump to moment" deep link, and the excerpt text.
- 👍/👎 feedback per answer, logged to Postgres for the monitoring
  dashboard.
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

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        for src in turn["sources"]:
            link = build_deeplink(src["video_id"], src["start"])
            ts = format_timestamp(src["start"])
            with st.expander(f"▶ {src.get('title', src['video_id'])} — {ts}"):
                st.markdown(f"[Jump to this moment]({link})")
                st.write(src["text"])

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching transcripts..."):
            start = time.time()
            search_query = rewrite_query(question) if use_rewrite else question
            sources = retrieve(search_query, strategy=strategy)
            answer = generate_answer(question, sources)
            latency_ms = (time.time() - start) * 1000

        st.write(answer)
        for src in sources:
            link = build_deeplink(src["video_id"], src["start"])
            ts = format_timestamp(src["start"])
            with st.expander(f"▶ {src.get('title', src['video_id'])} — {ts}"):
                st.markdown(f"[Jump to this moment]({link})")
                st.write(src["text"])

        query_id = int(time.time() * 1000)
        try:
            log_query(question, search_query, answer, sources, latency_ms, query_id)
        except Exception as e:
            st.caption(f"(monitoring log skipped: {e})")

        col1, col2 = st.columns(2)
        if col1.button("👍 Helpful", key=f"up_{query_id}"):
            log_feedback(query_id, True)
            st.toast("Thanks for the feedback!")
        if col2.button("👎 Not helpful", key=f"down_{query_id}"):
            log_feedback(query_id, False)
            st.toast("Thanks — we'll use this to improve retrieval.")

    st.session_state.history.append({"question": question, "answer": answer, "sources": sources})
