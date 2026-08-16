"""
Retrieval strategies, selectable via RETRIEVAL_STRATEGY env var:

  flat             -- vector search over flat (fixed-window) chunks
  parent_document   -- vector search over small child chunks, resolved to
                        their larger parent window for generation
  hybrid            -- vector + BM25 keyword search, fused with
                        Reciprocal Rank Fusion (RRF)
  hybrid_rerank     -- hybrid, then a cross-encoder reranks the fused
                        top-N before truncating to TOP_K

Each function returns a list of dicts:
  {video_id, title, text, start, end, score}
so the caller can build both the LLM context and the UI's timestamped
source cards from the same objects.
"""
import os

import psycopg2
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_embedder = None
_reranker = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


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


def vector_search(query: str, collection: str, top_k: int) -> list[dict]:
    client = get_qdrant()
    vec = get_embedder().encode(query).tolist()
    hits = client.search(collection_name=collection, query_vector=vec, limit=top_k)
    results = []
    for h in hits:
        payload = dict(h.payload)
        payload["score"] = h.score
        results.append(payload)
    return results


def bm25_search_flat(query: str, top_k: int) -> list[dict]:
    """Simple in-process BM25 over flat_chunks (fine for a project-scale corpus;
    for a larger one, use Postgres tsvector/tsquery directly)."""
    pg = get_pg()
    cur = pg.cursor()
    cur.execute("SELECT id, video_id, title, text, start_sec, end_sec FROM flat_chunks")
    rows = cur.fetchall()
    cur.close()
    pg.close()

    if not rows:
        return []

    corpus = [r[3].split() for r in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.split())
    ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {"video_id": r[1], "title": r[2], "text": r[3], "start": r[4], "end": r[5], "score": float(s)}
        for r, s in ranked
    ]


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Fuse multiple ranked lists (by 'text' as the dedup key) using RRF."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            key = item["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            items[key] = item
    fused = sorted(items.values(), key=lambda it: scores[it["text"]], reverse=True)
    return fused


def resolve_children_to_parents(children: list[dict]) -> list[dict]:
    pg = get_pg()
    cur = pg.cursor()
    resolved = []
    for c in children:
        cur.execute("SELECT video_id, title, text, start_sec, end_sec FROM parents WHERE id = %s", (c["parent_id"],))
        row = cur.fetchone()
        if row:
            resolved.append({
                "video_id": row[0], "title": row[1], "text": row[2],
                "start": row[3], "end": row[4], "score": c["score"],
            })
    cur.close()
    pg.close()
    return resolved


def retrieve(query: str, strategy: str | None = None, top_k: int | None = None) -> list[dict]:
    strategy = strategy or os.environ.get("RETRIEVAL_STRATEGY", "hybrid_rerank")
    top_k = top_k or int(os.environ.get("TOP_K", 5))

    if strategy == "flat":
        return vector_search(query, os.environ.get("QDRANT_COLLECTION_FLAT", "transcripts_flat"), top_k)

    if strategy == "parent_document":
        children = vector_search(query, os.environ.get("QDRANT_COLLECTION_CHILD", "transcripts_child"), top_k * 2)
        return resolve_children_to_parents(children)[:top_k]

    if strategy in ("hybrid", "hybrid_rerank"):
        vec_results = vector_search(query, os.environ.get("QDRANT_COLLECTION_FLAT", "transcripts_flat"), top_k * 3)
        bm25_results = bm25_search_flat(query, top_k * 3)
        fused = reciprocal_rank_fusion([vec_results, bm25_results])[: top_k * 3]

        if strategy == "hybrid":
            return fused[:top_k]

        # hybrid_rerank: cross-encoder rerank the fused candidates
        reranker = get_reranker()
        pairs = [(query, item["text"]) for item in fused]
        scores = reranker.predict(pairs)
        reranked = [item for item, _ in sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)]
        return reranked[:top_k]

    raise ValueError(f"Unknown retrieval strategy: {strategy}")
