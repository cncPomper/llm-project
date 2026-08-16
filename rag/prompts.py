"""
Prompt construction, including the query-rewriting step used as one of
the compared LLM strategies in eval/llm_eval.py.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

# Loaded here, at import, because importers (streamlit_app, llm_eval) call
# load_dotenv() *after* importing this module -- so anything read at module
# level would miss .env entirely on a local run.
load_dotenv()

MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

_client = None


def get_client() -> OpenAI:
    """Built lazily: constructing OpenAI() at import time raises when
    OPENAI_API_KEY is absent, which would break `import rag.prompts` for
    callers that never touch the LLM (e.g. retrieval-only eval runs)."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client

REWRITE_SYSTEM_PROMPT = """You rewrite casual, conversational podcast-listener \
questions into concise, keyword-rich search queries optimized for retrieval \
over podcast transcripts. Keep proper nouns (people, protocols, supplements) \
intact. Return ONLY the rewritten query, nothing else."""

ANSWER_SYSTEM_PROMPT = """You are a podcast knowledge assistant. You answer \
questions ONLY using the provided transcript excerpts. Every claim in your \
answer must be traceable to one of the excerpts.

Rules:
- If the excerpts don't contain the answer, say so plainly -- do not guess.
- After each claim, cite the excerpt it came from using its [N] label.
- Be concise and specific (e.g. actual protocols/numbers), not vague.
- Do not invent timestamps, numbers, or quotes not present in the excerpts.
"""


def rewrite_query(raw_query: str) -> str:
    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": raw_query},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def format_excerpts(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        mm, ss = divmod(int(c["start"]), 60)
        lines.append(f"[{i}] ({c.get('title', c['video_id'])} @ {mm:02d}:{ss:02d})\n{c['text']}")
    return "\n\n".join(lines)


def build_answer_prompt(question: str, chunks: list[dict]) -> list[dict]:
    excerpts = format_excerpts(chunks)
    user_prompt = f"Transcript excerpts:\n\n{excerpts}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(question: str, chunks: list[dict]) -> str:
    messages = build_answer_prompt(question, chunks)
    resp = get_client().chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()
