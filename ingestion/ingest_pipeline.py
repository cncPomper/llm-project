"""
Chunking + embedding + loading step of the ingestion pipeline.

Builds TWO parallel indexes from the same raw transcripts, so retrieval
strategies can be evaluated against each other later:

1. FLAT chunking
   Fixed ~300-token sliding-window chunks. Simple, single granularity.
   Stored in Qdrant collection `transcripts_flat`.

2. PARENT-DOCUMENT chunking
   Small ~100-token "child" chunks are embedded and indexed for precise
   matching. Each child stores a reference to a larger ~600-token "parent"
   window (built by merging several consecutive raw segments). At query
   time we search over children but return/generate from the parent, which
   tends to give better recall with better generation context.
   Children -> Qdrant collection `transcripts_child`.
   Parents   -> Postgres table `parents` (looked up by id at query time).

All chunks keep the source timestamp range so the app can build a YouTube
deep link (`&t=<seconds>s`) next to every answer.
"""
import json
import os
import uuid
from pathlib import Path

import psycopg2
import tiktoken
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_transcripts"
ENCODER = tiktoken.get_encoding("cl100k_base")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # swap for OpenAI embeddings if preferred
VECTOR_SIZE = 384

FLAT_CHUNK_TOKENS = 300
FLAT_OVERLAP_TOKENS = 50
CHILD_CHUNK_TOKENS = 100
PARENT_WINDOW_TOKENS = 600


def n_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def load_raw_transcripts() -> list[dict]:
    docs = []
    for path in RAW_DIR.glob("*.json"):
        with open(path) as f:
            docs.append(json.load(f))
    return docs


def sliding_window_chunks(segments: list[dict], window_tokens: int, overlap_tokens: int):
    """Yield (text, start_time, end_time) tuples from raw transcript segments."""
    buf_text, buf_tokens, start_time = [], 0, None
    for seg in segments:
        if start_time is None:
            start_time = seg["start"]
        buf_text.append(seg["text"])
        buf_tokens += n_tokens(seg["text"])
        end_time = seg["start"] + seg.get("duration", 0)

        if buf_tokens >= window_tokens:
            yield " ".join(buf_text), start_time, end_time
            # keep the tail as overlap for the next chunk
            tail = " ".join(buf_text)
            tail_tokens = ENCODER.encode(tail)[-overlap_tokens:]
            buf_text = [ENCODER.decode(tail_tokens)]
            buf_tokens = len(tail_tokens)
            start_time = end_time

    if buf_text:
        yield " ".join(buf_text), start_time, segments[-1]["start"] + segments[-1].get("duration", 0)


def build_flat_chunks(doc: dict) -> list[dict]:
    chunks = []
    for text, start, end in sliding_window_chunks(doc["segments"], FLAT_CHUNK_TOKENS, FLAT_OVERLAP_TOKENS):
        chunks.append({
            "id": str(uuid.uuid4()),
            "video_id": doc["video_id"],
            "title": doc["meta"].get("title", ""),
            "text": text,
            "start": start,
            "end": end,
        })
    return chunks


def build_parent_document_chunks(doc: dict) -> tuple[list[dict], list[dict]]:
    """Returns (children, parents). Each child has a parent_id pointing into parents."""
    parents = []
    for text, start, end in sliding_window_chunks(doc["segments"], PARENT_WINDOW_TOKENS, overlap_tokens=0):
        parents.append({
            "id": str(uuid.uuid4()),
            "video_id": doc["video_id"],
            "title": doc["meta"].get("title", ""),
            "text": text,
            "start": start,
            "end": end,
        })

    children = []
    for parent in parents:
        # re-split the parent window into smaller child chunks for indexing
        fake_segments = [{"text": parent["text"], "start": parent["start"], "duration": parent["end"] - parent["start"]}]
        for text, start, end in sliding_window_chunks(fake_segments, CHILD_CHUNK_TOKENS, overlap_tokens=20):
            children.append({
                "id": str(uuid.uuid4()),
                "parent_id": parent["id"],
                "video_id": doc["video_id"],
                "text": text,
                "start": start,
                "end": end,
            })
    return children, parents


def get_qdrant() -> QdrantClient:
    return QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))


def get_pg():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", 5432),
        dbname=os.environ.get("POSTGRES_DB", "podcast_explorer"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )


def ensure_qdrant_collections(client: QdrantClient):
    for name in [os.environ.get("QDRANT_COLLECTION_FLAT", "transcripts_flat"),
                 os.environ.get("QDRANT_COLLECTION_CHILD", "transcripts_child")]:
        if not client.collection_exists(name):
            client.create_collection(name, vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE))


def run():
    docs = load_raw_transcripts()
    if not docs:
        print("No raw transcripts found -- run `python -m ingestion.fetch_transcripts` first.")
        return

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    qdrant = get_qdrant()
    ensure_qdrant_collections(qdrant)
    pg = get_pg()
    cur = pg.cursor()

    flat_points, child_points, all_parents = [], [], []

    for doc in docs:
        flat_chunks = build_flat_chunks(doc)
        children, parents = build_parent_document_chunks(doc)
        all_parents.extend(parents)

        for c in flat_chunks:
            vec = embedder.encode(c["text"]).tolist()
            flat_points.append(PointStruct(id=c["id"], vector=vec, payload=c))

        for c in children:
            vec = embedder.encode(c["text"]).tolist()
            child_points.append(PointStruct(id=c["id"], vector=vec, payload=c))

        # also store flat chunk text in Postgres for BM25 (tsvector) search
        for c in flat_chunks:
            cur.execute(
                """INSERT INTO flat_chunks (id, video_id, title, text, start_sec, end_sec)
                   VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING""",
                (c["id"], c["video_id"], c["title"], c["text"], c["start"], c["end"]),
            )

        for p in parents:
            cur.execute(
                """INSERT INTO parents (id, video_id, title, text, start_sec, end_sec)
                   VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING""",
                (p["id"], p["video_id"], p["title"], p["text"], p["start"], p["end"]),
            )

        print(f"[ingest] {doc['video_id']}: {len(flat_chunks)} flat chunks, "
              f"{len(children)} child chunks, {len(parents)} parent windows")

    if flat_points:
        qdrant.upsert(os.environ.get("QDRANT_COLLECTION_FLAT", "transcripts_flat"), points=flat_points)
    if child_points:
        qdrant.upsert(os.environ.get("QDRANT_COLLECTION_CHILD", "transcripts_child"), points=child_points)

    pg.commit()
    cur.close()
    pg.close()
    print("Ingestion complete.")


if __name__ == "__main__":
    run()
