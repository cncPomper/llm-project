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
from prefect import flow, task

from ingestion import fetch_transcripts, ingest_pipeline


@task(retries=0)
def fetch_task():
    """No retries on purpose. The dominant failure here is YouTube
    rate-limiting the IP, and retrying 30s later just extends the block.
    fetch_transcripts.main() handles it instead: it stops cleanly, keeps
    what it already wrote, and resumes on the next run."""
    fetch_transcripts.main()


@task(retries=1)
def ingest_task():
    ingest_pipeline.run()


@flow(name="podcast-knowledge-ingestion")
def ingestion_flow():
    fetch_task()
    ingest_task()


if __name__ == "__main__":
    ingestion_flow()
