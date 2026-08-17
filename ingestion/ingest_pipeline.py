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
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, FilterSelector,
)
from sentence_transformers import SentenceTransformer

from ingestion.ad_filter import strip_ads, strip_ads_enabled

load_dotenv()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_transcripts"
ENCODER = tiktoken.get_encoding("cl100k_base")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # swap for OpenAI embeddings if preferred
VECTOR_SIZE = 384

FLAT_CHUNK_TOKENS = 300
FLAT_OVERLAP_TOKENS = 50
CHILD_CHUNK_TOKENS = 100
PARENT_WINDOW_TOKENS = 600

# Chunks are embedded in batches rather than one call per chunk; on CPU this
# is the difference between minutes and hours for a multi-hour episode.
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))


def chunk_uuid(video_id: str, kind: str, index: int) -> str:
    """Deterministic chunk ID, derived from (video, kind, position).

    Random uuid4s would make every re-run insert a fresh copy of the same
    chunk into Qdrant. Deriving the ID means re-ingesting an episode
    overwrites its own points instead -- which is what lets an interrupted
    run be resumed by simply running it again.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}:{kind}:{index}"))


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
            if overlap_tokens > 0:
                # keep the tail as overlap for the next chunk
                tail = " ".join(buf_text)
                tail_tokens = ENCODER.encode(tail)[-overlap_tokens:]
                buf_text = [ENCODER.decode(tail_tokens)]
                buf_tokens = len(tail_tokens)
            else:
                # Must be handled separately: `tokens[-0:]` is the WHOLE list,
                # not an empty one, so the zero-overlap path would carry the
                # entire buffer forward and re-emit it on every later segment.
                buf_text, buf_tokens = [], 0
            start_time = end_time

    if buf_text:
        yield " ".join(buf_text), start_time, segments[-1]["start"] + segments[-1].get("duration", 0)


def token_window_chunks(text: str, window_tokens: int, overlap_tokens: int):
    """Split a single block of text into overlapping token windows.

    Needed because sliding_window_chunks only breaks *between* segments: fed
    one long block it just hands the whole block back. Child chunks have to
    be split *within* the parent window, so they need a token-level split.

    Yields (text, start_fraction, end_fraction), the fractions giving each
    window's position within the block so the caller can interpolate
    timestamps across the parent's span.
    """
    tokens = ENCODER.encode(text)
    if not tokens:
        return

    step = max(1, window_tokens - overlap_tokens)
    total = len(tokens)
    for i in range(0, total, step):
        window = tokens[i:i + window_tokens]
        if not window:
            break
        end = min(i + window_tokens, total)
        yield ENCODER.decode(window), i / total, end / total
        if end >= total:
            break


def build_flat_chunks(doc: dict) -> list[dict]:
    chunks = []
    for i, (text, start, end) in enumerate(
        sliding_window_chunks(doc["segments"], FLAT_CHUNK_TOKENS, FLAT_OVERLAP_TOKENS)
    ):
        chunks.append({
            "id": chunk_uuid(doc["video_id"], "flat", i),
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
    for i, (text, start, end) in enumerate(
        sliding_window_chunks(doc["segments"], PARENT_WINDOW_TOKENS, overlap_tokens=0)
    ):
        parents.append({
            "id": chunk_uuid(doc["video_id"], "parent", i),
            "video_id": doc["video_id"],
            "title": doc["meta"].get("title", ""),
            "text": text,
            "start": start,
            "end": end,
        })

    children = []
    for parent in parents:
        # re-split the parent window into smaller child chunks for indexing
        span = parent["end"] - parent["start"]
        for text, frac_start, frac_end in token_window_chunks(
            parent["text"], CHILD_CHUNK_TOKENS, overlap_tokens=20
        ):
            children.append({
                "id": chunk_uuid(doc["video_id"], "child", len(children)),
                "parent_id": parent["id"],
                "video_id": doc["video_id"],
                "text": text,
                # interpolated across the parent's span by token position --
                # the transcript's own timings don't survive the re-split
                "start": parent["start"] + span * frac_start,
                "end": parent["start"] + span * frac_end,
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


def embed_points(embedder, chunks: list[dict]) -> list[PointStruct]:
    """Embed a whole chunk list in batches, not one call per chunk."""
    if not chunks:
        return []
    vectors = embedder.encode(
        [c["text"] for c in chunks],
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
    )
    return [
        PointStruct(id=c["id"], vector=v.tolist(), payload=c)
        for c, v in zip(chunks, vectors)
    ]


def purge_document(video_id: str, qdrant: QdrantClient, cur) -> None:
    """Remove every trace of an episode from both stores before re-ingesting.

    Needed because a re-run is not otherwise idempotent in the way it looks:
    the Postgres inserts are ON CONFLICT DO NOTHING, so an existing row keeps
    its OLD text while Qdrant's upsert replaces the vector -- leaving the two
    stores describing different chunks under the same id. And anything that
    changes how many chunks an episode yields (new chunking constants, or
    stripping sponsor reads) leaves the surplus tail of the previous run
    behind as orphans, since deterministic ids only overwrite index-for-index.
    """
    for collection in [os.environ.get("QDRANT_COLLECTION_FLAT", "transcripts_flat"),
                       os.environ.get("QDRANT_COLLECTION_CHILD", "transcripts_child")]:
        qdrant.delete(
            collection_name=collection,
            points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="video_id", match=MatchValue(value=video_id))
            ])),
        )
    cur.execute("DELETE FROM flat_chunks WHERE video_id = %s", (video_id,))
    cur.execute("DELETE FROM parents WHERE video_id = %s", (video_id,))


def already_ingested(cur, video_id: str) -> bool:
    cur.execute("SELECT 1 FROM flat_chunks WHERE video_id = %s LIMIT 1", (video_id,))
    return cur.fetchone() is not None


def ingest_document(doc: dict, embedder, qdrant: QdrantClient, cur, pg) -> None:
    """Ingest one episode and commit it.

    Qdrant is written before Postgres, and Postgres is what marks an episode
    as done. If this dies midway the episode looks un-ingested and is redone
    on the next run -- which is safe because chunk IDs are deterministic, so
    the re-run overwrites the same Qdrant points rather than duplicating them.
    """
    video_id = doc["video_id"]

    # Strip sponsor reads before chunking, not after: this is the one point
    # that feeds both indexes, so filtering here keeps the flat and
    # parent-document views of the corpus identical. Timestamps are untouched,
    # so deep links built from surviving segments still land correctly.
    if strip_ads_enabled():
        kept, removed = strip_ads(doc["segments"])
        if removed:
            ad_seconds = sum(s.get("duration", 0.0) for s in removed)
            print(f"[ads] {video_id}: dropped {len(removed)} segments "
                  f"({ad_seconds / 60:.1f} min) of sponsor reads")
            doc = {**doc, "segments": kept}

    flat_chunks = build_flat_chunks(doc)
    children, parents = build_parent_document_chunks(doc)

    flat_points = embed_points(embedder, flat_chunks)
    child_points = embed_points(embedder, children)

    if flat_points:
        qdrant.upsert(os.environ.get("QDRANT_COLLECTION_FLAT", "transcripts_flat"), points=flat_points)
    if child_points:
        qdrant.upsert(os.environ.get("QDRANT_COLLECTION_CHILD", "transcripts_child"), points=child_points)

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

    pg.commit()
    print(f"[ingest] {video_id}: {len(flat_chunks)} flat chunks, "
          f"{len(children)} child chunks, {len(parents)} parent windows -- committed")


def run(force: bool = False):
    """Ingest every fetched transcript, one episode per transaction.

    force=True (or REINGEST=1) re-ingests episodes already in Postgres --
    needed after changing the chunking constants, since otherwise the
    already-ingested check skips them.
    """
    force = force or os.environ.get("REINGEST") == "1"

    docs = load_raw_transcripts()
    if not docs:
        print("No raw transcripts found -- run `python -m ingestion.fetch_transcripts` first.")
        return

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    qdrant = get_qdrant()
    ensure_qdrant_collections(qdrant)
    pg = get_pg()
    cur = pg.cursor()

    done = skipped = 0
    try:
        for doc in docs:
            if not force and already_ingested(cur, doc["video_id"]):
                print(f"[skip] {doc['video_id']} already ingested")
                skipped += 1
                continue

            if force and already_ingested(cur, doc["video_id"]):
                purge_document(doc["video_id"], qdrant, cur)

            ingest_document(doc, embedder, qdrant, cur, pg)
            done += 1
    finally:
        cur.close()
        pg.close()

    print(f"\nIngestion complete: {done} episode(s) ingested, {skipped} already present.")


if __name__ == "__main__":
    run()
