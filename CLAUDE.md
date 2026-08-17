# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with [uv]; `uv run` executes inside `.venv`, so there is no activate step.

```bash
uv sync                              # recreate .venv exactly from uv.lock

docker compose up --build -d qdrant postgres grafana   # infra (schema.sql runs on first pg boot)
docker compose up --build -d app                        # Streamlit on :8501

uv run python -m ingestion.flow            # Prefect flow: fetch -> chunk -> embed -> load
uv run python -m ingestion.fetch_transcripts   # fetch step alone
uv run python -m ingestion.ingest_pipeline     # chunk/embed/load step alone

uv run python -m eval.generate_ground_truth    # writes eval/ground_truth.jsonl (costs OpenAI tokens)
uv run python -m eval.retrieval_eval           # appends a table to eval/results.md
uv run python -m eval.llm_eval                 # appends a table to eval/results.md

uv run streamlit run app/streamlit_app.py      # app on the host instead of in Compose
```

There is no test suite and no linter configured — don't invent commands for them.

Two things break runs and are easy to trip over:

- **Always run from the repo root with `python -m`.** Running a script by path puts *its own* directory on `sys.path` instead of the repo root, so `from ingestion...` / `from rag...` fail. `episodes.yaml` is also resolved relative to the working directory. (The Dockerfile sets `PYTHONPATH=/app` for the same reason — `streamlit run app/...` would otherwise put `/app/app` on the path.)
- **Host-side runs need env overrides.** `.env` holds the Compose hostnames (`postgres`, `qdrant`), which only resolve inside the Compose network:
  ```powershell
  $env:POSTGRES_HOST="localhost"; $env:QDRANT_URL="http://localhost:6333"
  ```

## Architecture

Corpus: YouTube transcripts for the video IDs in `episodes.yaml`. Ingestion builds **two parallel indexes from the same raw transcripts** so retrieval strategies can be compared head-to-head:

| Index | Vectors | Text/lookup store |
|---|---|---|
| Flat (~300-token sliding window) | Qdrant `transcripts_flat` | Postgres `flat_chunks` (also feeds BM25) |
| Parent-document (~100-token children → ~600-token parents) | Qdrant `transcripts_child` | Postgres `parents` (resolved by `parent_id` at query time) |

Flow of a query: `app/streamlit_app.py` → `rag.prompts.rewrite_query` (optional) → `rag.retrieval.retrieve` (strategy from the sidebar or `RETRIEVAL_STRATEGY`) → `rag.prompts.generate_answer` → source cards built by `rag.tools.build_deeplink`. Every turn is logged to Postgres `query_log`, feedback to `feedback`, both read by the Grafana dashboard.

### Contracts worth preserving

- **Retrieval result shape.** Every strategy in `rag/retrieval.py` returns `{video_id, title, text, start, end, score}`. Note the rename: Postgres columns are `start_sec`/`end_sec`, the dicts use `start`/`end`. The UI source cards, the LLM prompt's `[N]` excerpts, and both eval scripts all consume this same shape — changing chunk payloads means changing all three consumers.
- **`video_id` must be a bare 11-char ID.** It is substituted directly into `build_deeplink()`, so a full URL yields a broken "jump to moment" link on every card. `fetch_transcripts.normalize_video_id` exists for this.
- **Chunk IDs are deterministic** (`uuid5` over `video_id:kind:index`), which is what makes ingestion resumable: a re-run overwrites its own Qdrant points instead of duplicating them. Presence in Postgres `flat_chunks` is the "already ingested" marker, and Qdrant is written *before* Postgres so a crash mid-episode leaves it looking un-ingested.
- **After changing any chunking constant** (`FLAT_CHUNK_TOKENS`, `PARENT_WINDOW_TOKENS`, …) existing episodes are silently skipped. Set `REINGEST=1` to rebuild them.
- **The embedding model is hardcoded in two places** — `EMBED_MODEL_NAME`/`VECTOR_SIZE` in `ingestion/ingest_pipeline.py` and `EMBED_MODEL_NAME` in `rag/retrieval.py`. They must match, and switching models requires deleting/recreating the Qdrant collections. The `EMBEDDING_MODEL` var in `.env` is not read by any code path today.
- A Postgres connection helper is duplicated across `rag/retrieval.py`, `rag/tools.py`, `ingestion/ingest_pipeline.py`, `app/streamlit_app.py`, and `eval/generate_ground_truth.py` — change one connection default and check the others.
- `query_log.id` is **not** a sequence; the app generates it as `int(time.time() * 1000)` before inserting, so it can wire the feedback buttons to a row it hasn't written yet.

### Dependency pinning is deliberate

`pyproject.toml` sets `exclude-newer = "2024-08-15"`. The direct pins are mid-2024 releases with loose transitive bounds, and an unconstrained resolve pairs them with 2026 packages that break at import (documented in the file). Two packages are exempted via `exclude-newer-package` because they talk to live YouTube endpoints that have since changed: `youtube-transcript-api` and `yt-dlp` (the latter is effectively broken within months of release, so it needs re-locking whenever extraction breaks). When adding a dependency, add it under that cap rather than raising the date; `uv.lock` is committed and the container installs with `uv sync --frozen`, which fails loudly on drift.

### Ingestion failure modes

`ingestion/fetch_transcripts.py` handles YouTube rate-limiting as an expected condition, not a crash: `RequestBlocked` is recoverable per-episode when the Whisper fallback is on, and otherwise stops the run cleanly (retrying extends the block — hence `retries=0` on `fetch_task`), keeping what's on disk for the next run. A run that fetched *nothing* raises `TranscriptsUnavailable`, which `ingestion/flow.py` converts into a one-line Prefect `Failed` state and skips ingest.

**Captions are currently blocked from this machine.** YouTube returns 429 on `GET /api/timedtext` for this IP, so `youtube-transcript-api` raises `IpBlocked` for every video. Verified that the watch page and `/youtubei/v1/player` return 200 and **video downloads still work** — only the caption endpoint is blocked. Raising `FETCH_DELAY_SECONDS` does not help (the 429 lands on the first request of a run), and a datacenter proxy does not either, since YouTube blocks those ranges wholesale.

Hence the fallback: `WHISPER_FALLBACK=1` downloads audio with yt-dlp and transcribes locally with faster-whisper. It triggers on `RequestBlocked` *or* genuinely absent captions, and produces the same `{text, start, duration}` segments as the caption path, so nothing downstream changes. Notes:

- `WHISPER_MODEL` defaults to `base`. Measured on 16 cores: `tiny` 2.7x realtime, `base` 1.6x, `small` 0.7x — `small` takes longer than the episode itself. The full 6-episode corpus is ~12h of audio ≈ 7.5h at `base`.
- Requires **ffmpeg on PATH** for the audio extraction postprocessor.
- Audio is extracted to 16 kHz mono (what Whisper resamples to anyway) and deleted after transcription — it's intermediate state, and retaining six episodes would be GBs for nothing.
- `transcribe_with_whisper` sets `KMP_DUPLICATE_LIB_OK` before importing faster_whisper: ctranslate2 and onnxruntime each bundle an OpenMP runtime and loading both aborts the process on Windows. Verified the override transcribes a known clip correctly rather than silently corrupting output.
- yt-dlp warns about a missing JS runtime (deno). Extraction still works for all six episodes, but if a video starts failing, that warning is the first thing to look at.
