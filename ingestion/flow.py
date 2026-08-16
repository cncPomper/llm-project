"""
Prefect flow that automates the full ingestion pipeline:

    fetch_transcripts  ->  ingest_pipeline (chunk + embed + load)

Run once:
    python ingestion/flow.py

Or deploy it to run on a schedule (e.g. nightly, to pick up new episodes
added to episodes.yaml) with:
    prefect deployment build ingestion/flow.py:ingestion_flow -n "podcast-ingest" --cron "0 3 * * *"
    prefect deployment apply ingestion_flow-deployment.yaml
"""
from prefect import flow, task

from ingestion import fetch_transcripts, ingest_pipeline


@task(retries=2, retry_delay_seconds=30)
def fetch_task():
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
