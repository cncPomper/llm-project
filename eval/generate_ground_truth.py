"""
Generates a synthetic ground-truth set for retrieval + LLM evaluation:
for each flat chunk, ask an LLM to write a question that chunk answers.
This gives us (question -> known source chunk) pairs, so we can measure
whether retrieval actually finds the right chunk (Hit Rate / MRR) and
whether the generated answer references the right content.

Output: eval/ground_truth.jsonl
  {"question": ..., "video_id": ..., "chunk_id": ..., "chunk_text": ...}
"""
import json
import os
import random
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

OUT_PATH = Path(__file__).parent / "ground_truth.jsonl"

QUESTION_PROMPT = """Given this excerpt from a podcast transcript, write ONE \
specific question a listener might ask that this excerpt directly answers. \
Return ONLY the question, no preamble.

Excerpt:
{text}"""


def get_pg():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", 5432),
        dbname=os.environ.get("POSTGRES_DB", "podcast_explorer"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )


def main(sample_size: int = 100, seed: int = 42):
    pg = get_pg()
    cur = pg.cursor()
    cur.execute("SELECT id, video_id, text FROM flat_chunks")
    rows = cur.fetchall()
    cur.close()
    pg.close()

    random.seed(seed)
    sample = random.sample(rows, min(sample_size, len(rows)))

    with open(OUT_PATH, "w") as f:
        for chunk_id, video_id, text in sample:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": QUESTION_PROMPT.format(text=text)}],
                temperature=0.7,
            )
            question = resp.choices[0].message.content.strip()
            f.write(json.dumps({
                "question": question,
                "video_id": video_id,
                "chunk_id": chunk_id,
                "chunk_text": text,
            }) + "\n")

    print(f"Wrote {len(sample)} ground-truth pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
