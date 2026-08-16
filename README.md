# Podcast Knowledge Explorer

**Multi-speaker podcast transcript RAG for Huberman Lab / Lex Fridman style long-form episodes.**

Ask a question like *"What does Huberman recommend for morning light exposure?"* and get a
direct answer plus the exact video timestamp(s) and deep links to jump straight to that
moment — no scrubbing through 2–3 hour episodes.

---

## 1. Problem Statement

Long-form podcasts (Huberman Lab, Lex Fridman, etc.) pack dozens of specific,
actionable claims — protocols, dosages, study references, opinions — into
episodes that run 2–4 hours. Listeners who want a specific recommendation
("what's the caffeine timing protocol?") have no easy way to find it other
than scrubbing the video or reading a flat, unsearchable transcript.

This project builds a **RAG + light agent** application that:

1. Ingests YouTube transcripts (with timestamps, and speaker labels where
   available) for a chosen set of episodes.
2. Indexes them for both **flat chunk retrieval** and **parent-document
   retrieval** (small child chunks for search precision, larger parent
   windows returned for context).
3. Answers user questions by retrieving relevant transcript windows,
   building a grounded prompt, and generating an answer.
4. Surfaces the **exact timestamp and clickable deep link**
   (`https://youtube.com/watch?v=<id>&t=<seconds>s`) next to every claim in
   the answer.
5. Is evaluated on retrieval quality (flat vs. parent-document) and on
   final answer quality across prompting strategies.
6. Collects user feedback and exposes a monitoring dashboard.

## 2. Data Source

- **YouTube Transcript API** (`youtube-transcript-api`) for auto-generated
  or creator-uploaded captions — includes per-line start times.
- Optional fallback: **Whisper** transcription (`faster-whisper`) for
  episodes without captions, which can also produce **speaker diarization**
  when combined with `pyannote-audio`.
- A curated `episodes.yaml` file lists the video IDs to ingest (starter list
  included — swap in whichever channel/episodes you like).

> This is a different corpus from the DTC Zoomcamp FAQ dataset used in
> course modules, per the project rules.

## 3. Architecture

```
YouTube video IDs (episodes.yaml)
        │
        ▼
ingestion/fetch_transcripts.py   ── pulls raw transcript + timestamps
        │
        ▼
ingestion/ingest_pipeline.py     ── chunks two ways:
        │                              1. flat: fixed-size sliding window
        │                              2. parent-document: small child
        │                                 chunks (search) mapped to a
        │                                 larger parent window (context)
        ▼
Qdrant (vector index)  +  Postgres (BM25/text search via tsvector,
                            parent-document store, feedback log)
        │
        ▼
rag/retrieval.py   ── hybrid search (BM25 + vector) → optional rerank
        │              → resolves child hits to parent windows
        ▼
rag/tools.py       ── build_deeplink(video_id, seconds)
                       list_episode_topics(video_id)
        │
        ▼
LLM (prompt in rag/prompts.py) ── grounded answer + citations w/ timestamps
        │
        ▼
app/streamlit_app.py  ── chat UI, shows answer + clickable timestamped
                          sources, collects 👍/👎 feedback
        │
        ▼
Postgres feedback/query log  →  Grafana dashboard (monitoring/)
```

## 4. Retrieval Strategies Compared (Retrieval Evaluation)

`eval/retrieval_eval.py` builds a synthetic ground-truth set (LLM-generated
questions with known source chunk/timestamp) and reports **Hit Rate** and
**MRR** for:

| Strategy | Description |
|---|---|
| Flat chunking | Fixed ~300-token sliding window chunks, indexed and returned as-is |
| Parent-document retrieval | Small ~100-token child chunks used for matching; the containing ~600-token parent window is returned for generation |
| Hybrid (vector + BM25) | Both of the above combined with keyword search, reciprocal rank fusion |
| + Rerank | Cross-encoder rerank of the fused top-K before passing to the LLM |

The best-performing combination (tracked in `eval/results.md`) is what the
app uses by default — configurable via `.env` (`RETRIEVAL_STRATEGY=`).

## 5. LLM Answer Evaluation

`eval/llm_eval.py` compares:

1. Plain RAG (stuff retrieved context into prompt)
2. RAG + query rewriting (rewrite vague/conversational questions into
   retrieval-friendly queries before search)
3. RAG + query rewriting + rerank

using an LLM-as-judge rubric (faithfulness to transcript, timestamp
accuracy, relevance) over a held-out question set. Results in
`eval/results.md`.

## 6. Interface

Streamlit chat app (`app/streamlit_app.py`):

- Chat-style Q&A
- Every answer shows **source cards**: episode title, speaker (if
  diarized), timestamp range, a "▶ jump to moment" deep link, and the
  quoted transcript excerpt
- 👍 / 👎 feedback buttons per answer, logged to Postgres

## 7. Ingestion Pipeline

`ingestion/ingest_pipeline.py` is a script orchestrated by **Prefect**
(`ingestion/flow.py`) so it's a repeatable, automated pipeline rather than
a one-off notebook: fetch → chunk (both strategies) → embed → load.

## 8. Monitoring

All queries, retrieved sources, generated answers, latency, and feedback
are logged to Postgres (`monitoring/schema.sql`). `monitoring/grafana/`
contains a dashboard JSON with 5 charts:

1. Query volume over time
2. 👍/👎 feedback ratio over time
3. Average end-to-end latency
4. Most-asked-about episodes/topics
5. Retrieval hit-rate (sampled) over time

## 9. Containerization

`docker-compose.yml` runs everything: `app` (Streamlit), `qdrant`,
`postgres`, `grafana`. `Dockerfile` builds the app image.

### Quickstart

**Step 0 — put real episodes in `episodes.yaml`.** It ships with
`REPLACE_ME_*` placeholders; until they're replaced with real YouTube video
IDs there is nothing to ingest and the app has nothing to answer from.

```bash
cp .env.example .env    # fill in OPENAI_API_KEY
```

**1. Start the infrastructure.** `monitoring/schema.sql` runs automatically
on Postgres's first boot.

```bash
docker compose up --build -d qdrant postgres grafana
```

**2. Ingest.** Run from the **repo root** — `episodes.yaml` is resolved
relative to the working directory. Use `python -m ...` rather than
`python path/to/script.py`: running a script by path puts *its own*
directory on `sys.path` instead of the repo root, so the `ingestion.` /
`rag.` / `eval.` package imports won't resolve.

The `.env` hostnames (`postgres`, `qdrant`) only resolve inside the Compose
network, so override them for host-side runs:

```bash
uv sync                       # creates .venv from uv.lock, exactly pinned

export POSTGRES_HOST=localhost QDRANT_URL=http://localhost:6333
# Windows PowerShell:
#   $env:POSTGRES_HOST="localhost"; $env:QDRANT_URL="http://localhost:6333"

uv run python -m ingestion.flow    # fetch -> chunk -> embed -> load (Prefect)
```

`uv run` executes inside the project venv, so there is no activate step.

**3. Evaluate** (same shell, same env overrides — costs OpenAI tokens):

```bash
uv run python -m eval.generate_ground_truth
uv run python -m eval.retrieval_eval
uv run python -m eval.llm_eval
```

**4. Start the app:**

```bash
docker compose up --build -d app
```

- App: http://localhost:8501
- Grafana: http://localhost:3000 (default admin/admin)
- Qdrant dashboard: http://localhost:6333/dashboard

Grafana isn't provisioned automatically: add Postgres as a datasource
(host `postgres:5432`, matching your `.env` credentials), then import
`monitoring/grafana/dashboard.json`.

## 10. Reproducibility

- Dependencies are managed with [uv](https://docs.astral.sh/uv/). Direct
  pins live in `pyproject.toml`; `uv.lock` pins the full transitive tree
  (162 packages) with hashes and is committed, so `uv sync` reproduces the
  exact environment. The container installs from the same lock via
  `uv sync --frozen`, which fails rather than silently re-resolving if the
  lock and `pyproject.toml` drift apart.
- Python is pinned to 3.11 (`.python-version`), matching the container.
- `episodes.yaml` lists the exact video IDs used — transcripts are fetched
  live via the YouTube API so no large data files need to be committed.
- `.env.example` documents every required environment variable.

## 11. Best Practices Implemented

- [x] Hybrid search (BM25 + vector, reciprocal rank fusion) — `rag/retrieval.py`
- [x] Document re-ranking (cross-encoder) — `rag/retrieval.py`
- [x] Query rewriting — `rag/prompts.py::rewrite_query`

## 12. Evaluation Criteria Mapping

| Criterion | Where addressed |
|---|---|
| Problem description | Section 1 above |
| Retrieval flow | Sections 3–4, `rag/retrieval.py` |
| Retrieval evaluation | Section 4, `eval/retrieval_eval.py`, `eval/results.md` |
| LLM evaluation | Section 5, `eval/llm_eval.py`, `eval/results.md` |
| Interface | Section 6, `app/streamlit_app.py` |
| Ingestion pipeline | Section 7, `ingestion/flow.py` (Prefect) |
| Monitoring | Section 8, `monitoring/` |
| Containerization | Section 9, `docker-compose.yml` |
| Reproducibility | Section 10 |
| Best practices | Section 11 |

## 13. Project Structure

```
podcast-explorer/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── .env.example
├── episodes.yaml
├── ingestion/
│   ├── fetch_transcripts.py
│   ├── ingest_pipeline.py
│   └── flow.py
├── rag/
│   ├── retrieval.py
│   ├── prompts.py
│   └── tools.py
├── eval/
│   ├── generate_ground_truth.py
│   ├── retrieval_eval.py
│   ├── llm_eval.py
│   └── results.md
├── app/
│   └── streamlit_app.py
└── monitoring/
    ├── schema.sql
    └── grafana/
        └── dashboard.json
```

## 14. Next Steps / TODO

- [ ] Pick final episode list in `episodes.yaml`
- [ ] Run ingestion end-to-end and populate Qdrant + Postgres
- [ ] Generate ground-truth eval set and fill in `eval/results.md`
- [ ] Record a short Streamlit screen-capture demo and embed it here
- [ ] Add screenshots of the Grafana dashboard
