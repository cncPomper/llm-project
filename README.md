# Podcast Knowledge Explorer

![Podcast Knowledge Explorer](imgs/podcaster.jpg)

**Multi-speaker podcast transcript RAG for long-form episodes.**

Ask a question like *"What does Cesar Millan mean by calm assertive energy?"* and get a
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
- **Local transcription fallback** (`yt-dlp` + `faster-whisper`), enabled
  with `WHISPER_FALLBACK=1`. It covers episodes without captions and — the
  case that actually matters in practice — YouTube blocking the caption
  endpoint for your IP. See [Troubleshooting](#15-troubleshooting-youtube-blocks).
  Speaker diarization via `pyannote-audio` can be layered on top of the
  Whisper segments, but is not wired in (needs a HuggingFace token and a GPU).
- A curated `episodes.yaml` file lists the video IDs to ingest — six
  long-form episodes (~12 hours of audio) across five channels.

> This is a different corpus from the DTC Zoomcamp FAQ dataset used in
> course modules, per the project rules.

## 3. Architecture

```
YouTube video IDs (episodes.yaml)
        │
        ▼
ingestion/fetch_transcripts.py   ── pulls raw transcript + timestamps
        │                            ├─ captions via youtube-transcript-api
        │                            └─ if blocked/absent and WHISPER_FALLBACK=1:
        │                               yt-dlp audio → faster-whisper
        │                               (same {text, start, duration} shape)
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

Measured over 100 synthetic questions (`eval/results.md`):

| Strategy | Hit Rate@5 | MRR@5 |
|---|---|---|
| flat | 0.84 | 0.697 |
| parent_document | 0.90 | 0.791 |
| hybrid | 0.89 | 0.752 |
| **hybrid_rerank** | **0.92** | **0.862** |

`hybrid_rerank` wins on both metrics and is the default
(`RETRIEVAL_STRATEGY` in `.env`, switchable per query in the app sidebar).
The rerank is what separates it from plain `hybrid` — same candidates, but
MRR climbs from 0.752 to 0.862, i.e. the right chunk moves up the list
rather than merely appearing in it.

## 5. LLM Answer Evaluation

`eval/llm_eval.py` compares:

1. Plain RAG (stuff retrieved context into prompt)
2. RAG + query rewriting (rewrite vague/conversational questions into
   retrieval-friendly queries before search)
3. RAG + query rewriting + rerank

using an LLM-as-judge rubric (faithfulness to transcript, relevance,
specificity), scored 1–5 over 30 questions:

| Strategy | Faithfulness | Relevance | Specificity |
|---|---|---|---|
| plain_rag | 4.73 | 4.87 | 4.10 |
| rewrite_rag | 4.77 | 4.90 | 3.93 |
| rewrite_rerank_rag | 4.57 | 4.73 | 4.03 |

**All three are indistinguishable.** The spread is ~0.2 on a 1–5 scale at
N=30, which is noise, not a result — the rubric does not separate strategies
that share a corpus and a generator. Query rewriting neither helps nor hurts
here, so it stays available as a sidebar toggle.

The one consistent signal is that **specificity is the weakest dimension**
across the board: answers are faithful and relevant but stay general. That
is the thing worth attacking next, and it points at retrieval granularity
rather than prompting.

Two caveats on how much weight these numbers carry:

- The questions are LLM-generated *from* a chunk, so they are keyword-rich
  and already retrieval-friendly. None resemble the vague, conversational
  question rewriting exists to fix.
- An earlier run of this table reported rewriting losing by a full point.
  That was a bug, not a finding — the judge scored unparseable replies as
  0/0/0, and the failures clustered on one strategy. See
  `eval/results.md` for the superseded numbers and the cause.

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

It is built to survive a long run being interrupted, which matters because
transcribing ~12 hours of audio takes hours:

- **Already-fetched episodes are skipped** (a transcript JSON on disk) and
  **already-ingested ones too** (a row in Postgres), so re-running resumes
  rather than redoing.
- **Chunk IDs are deterministic** (`uuid5` of video/kind/position), so a
  re-ingest overwrites its own Qdrant points instead of duplicating them.
- **Downloaded audio is reused** after an interruption — a killed run does
  not re-download ~300 MB it already has.
- **One episode failing does not kill the run**; it is skipped, reported in
  the summary, and retried on the next run.

Set `REINGEST=1` to force re-ingestion after changing the chunking constants,
which the already-ingested check would otherwise skip.

## 8. Monitoring

All queries, retrieved sources, generated answers, latency, and feedback
are logged to Postgres (`monitoring/schema.sql`). `monitoring/grafana/`
contains a provisioned dashboard with 6 panels:

1. Query volume over time
2. 👍/👎 feedback ratio over time
3. Average end-to-end latency
4. Recent questions (with source count and latency)
5. Negative feedback count, last 7 days
6. Most-cited episodes, by indexed chunk count

## 9. Containerization

`docker-compose.yml` runs everything: `app` (Streamlit), `qdrant`,
`postgres`, `grafana`. `Dockerfile` builds the app image.

### Quickstart

**Prerequisites:** [uv](https://docs.astral.sh/uv/), Docker, and — if you
will use the Whisper fallback — **ffmpeg on PATH** (used to extract audio to
16 kHz mono) and optionally **Node** or **Deno** (yt-dlp needs a JavaScript
runtime to decipher YouTube signatures; without one some downloads fail with
`HTTP Error 403`).

`episodes.yaml` already lists six real episodes, so you can run this as-is;
edit it to point at whichever channel or episodes you prefer.

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

`POSTGRES_HOST` / `QDRANT_URL` must point at `localhost` for host-side runs —
the Compose service names (`postgres`, `qdrant`) only resolve inside the
Compose network. `.env.example` ships the Compose values because the app
container reads that file; set them to `localhost` in your own `.env`, or
override per shell. This does not affect the container, which gets both from
the `environment:` block in `docker-compose.yml`.

```bash
uv sync                       # creates .venv from uv.lock, exactly pinned

uv run python -m ingestion.flow    # fetch -> chunk -> embed -> load (Prefect)
```

`uv run` executes inside the project venv, so there is no activate step.

If the caption endpoint is blocked for your IP, set `WHISPER_FALLBACK=1`
first — otherwise the run stops with nothing fetched. Transcribing all six
episodes takes a few hours; the log reports progress once a minute, and
`-u` keeps it streaming when you redirect it:

```bash
uv run python -u -m ingestion.flow 2>&1 | tee ingest.log
```

**3. Evaluate** (costs OpenAI tokens):

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

Grafana provisions itself at startup — `monitoring/grafana/provisioning/`
defines the Postgres datasource and a dashboard provider, and
`monitoring/grafana/dashboard.json` is mounted into the path it scans. No
manual datasource setup or dashboard import: the dashboard is under the
"Podcast Explorer" folder as soon as the container is up.

## 10. Reproducibility

- Dependencies are managed with [uv](https://docs.astral.sh/uv/). Direct
  pins live in `pyproject.toml`; `uv.lock` pins the full transitive tree
  (162 packages) with hashes and is committed, so `uv sync` reproduces the
  exact environment. The container installs from the same lock via
  `uv sync --frozen`, which fails rather than silently re-resolving if the
  lock and `pyproject.toml` drift apart.
- Pins are resolved as of `exclude-newer = 2024-08-15`, contemporaneous with
  the direct pins, because several declare loose bounds and break when paired
  with 2026 transitive packages. Two exceptions are exempted via
  `exclude-newer-package`: `youtube-transcript-api` and `yt-dlp` both talk to
  live YouTube endpoints that have since changed, so they must track upstream.
  `yt-dlp` especially needs re-locking whenever YouTube breaks extraction.
- ffmpeg is an external, unpinned dependency of the Whisper fallback.
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
├── CLAUDE.md                 # repo guidance for Claude Code
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
├── data/                     # gitignored
│   ├── raw_transcripts/      # fetched transcripts (JSON)
│   └── audio/                # Whisper intermediates, deleted after use
└── monitoring/
    ├── schema.sql
    └── grafana/
        └── dashboard.json
```

## 14. Next Steps / TODO

- [x] Pick final episode list in `episodes.yaml` — 6 episodes, ~12 h of audio
- [x] Run ingestion end-to-end and populate Qdrant + Postgres — 562 flat
      chunks, 241 parent windows, 1900 child vectors
- [x] Generate the ground-truth set (100 questions) and run both evaluations
      — results in `eval/results.md`
- [x] Confirm `RETRIEVAL_STRATEGY=hybrid_rerank` against the numbers
      (best on both Hit Rate and MRR)
- [ ] Re-run the LLM eval against hand-written, conversational questions —
      the synthetic set is biased toward already-well-formed queries, so it
      cannot show whether query rewriting earns its keep
- [ ] Attack specificity, the weakest judged dimension — answers are
      faithful but general, which points at retrieval granularity
- [ ] Record a short Streamlit screen-capture demo and embed it here
- [ ] Add screenshots of the Grafana dashboard

Known limitation: sponsor reads are indexed as ordinary transcript content,
so ad copy can surface as a retrieval hit and will depress eval scores.

## 15. Troubleshooting: YouTube blocks

Transcript fetching and audio download are two different endpoints with
independent failure modes. Both are expected conditions the pipeline handles
rather than crashes.

**`IpBlocked` / `RequestBlocked` on every video.** YouTube is returning 429
on `/api/timedtext` for your IP. The watch page and audio downloads keep
working — only captions are blocked, so this does not mean you are offline
or banned. Raising `FETCH_DELAY_SECONDS` does not help if the 429 lands on
the first request of a run, since the limit predates the run. Options, in
order of reliability:

1. `WHISPER_FALLBACK=1` — transcribe locally, independent of IP reputation.
2. A **residential** proxy (`WEBSHARE_PROXY_USERNAME` / `_PASSWORD`, or
   `PROXY_HTTP_URL`). Datacenter proxies do not work: YouTube blocks those
   ranges wholesale, so the exit IP 429s exactly like your own. Check what
   you are actually getting before paying — a rotating residential pool
   returns a different IP per request.
3. Wait for the block to lapse (minutes to hours for a home IP).

**`HTTP Error 403: Forbidden` from yt-dlp.** Two distinct causes:

- *No JavaScript runtime.* yt-dlp enables only `deno` by default; install
  Deno or Node so signatures can be deciphered. `download_audio()` enables
  whichever of deno/node/bun it finds.
- *The "n challenge".* High-bitrate audio is gated behind a challenge that
  needs a solver script yt-dlp fetches at runtime. Rather than enabling
  `--remote-components ejs`, which downloads and executes remote code, the
  pipeline falls back to the lowest-bitrate stream — which is ungated, and
  costs nothing here since audio is downmixed to 16 kHz mono for Whisper.

**Whisper is slow.** `WHISPER_MODEL` defaults to `base`. Measured on 16 CPU
cores: `tiny` ~2.7x realtime (noticeably worse wording), `base` ~1.6–7x
depending on load, `small` ~0.7x — i.e. `small` takes longer than the episode
itself. On Windows, `KMP_DUPLICATE_LIB_OK` is set before importing
faster-whisper because ctranslate2 and onnxruntime each bundle an OpenMP
runtime and loading both aborts the process.
