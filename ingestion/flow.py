"""
Prefect flow that automates the full ingestion pipeline:

    fetch_transcripts  ->  ingest_pipeline (chunk + embed + load)

Run once, from the repo root (the `-m` form matters -- running this file by
path puts ingestion/ on sys.path instead of the repo root, so the
`from ingestion import ...` below would fail):
    python -m ingestion.flow

Or deploy it to run on a schedule (e.g. nightly, to pick up new episodes
added to episodes.yaml) with:
    prefect deployment build ingestion/flow.py:ingestion_flow -n "podcast-ingest" --cron "0 3 * * *"
    prefect deployment apply ingestion_flow-deployment.yaml
"""
import sys

from prefect import flow, task
from prefect.states import Completed, Failed

from ingestion import fetch_transcripts, ingest_pipeline
from ingestion.fetch_transcripts import TranscriptsUnavailable


@task(retries=0)
def fetch_task():
    """No retries on purpose. The dominant failure here is YouTube
    rate-limiting the IP, and retrying 30s later just extends the block.
    fetch_transcripts.main() handles it instead: it stops cleanly, keeps
    what it already wrote, and resumes on the next run.

    Returns an explicit Failed state rather than raising, so an expected,
    actionable condition reads as one line instead of burying it under
    several screens of Prefect internals. Real bugs still raise normally.
    """
    try:
        on_disk = fetch_transcripts.main()
    except TranscriptsUnavailable as e:
        return Failed(message=str(e))
    return Completed(message=f"{on_disk} episode(s) on disk")


@task(retries=1)
def ingest_task():
    ingest_pipeline.run()


@flow(name="podcast-knowledge-ingestion")
def ingestion_flow():
    fetch_state = fetch_task(return_state=True)
    if fetch_state.is_failed():
        # Skip ingest: there is nothing on disk to chunk.
        return fetch_state
    ingest_task()


if __name__ == "__main__":
    state = ingestion_flow(return_state=True)
    if state.is_failed():
        # The flow's own message is just "1/1 states failed"; the useful text
        # is on the state the failing task returned.
        inner = state.result(raise_on_failure=False)
        message = getattr(inner, "message", None) or state.message
        print(f"\nIngestion failed: {message}", file=sys.stderr)
        raise SystemExit(1)
