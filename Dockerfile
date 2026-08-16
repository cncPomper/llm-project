FROM python:3.11-slim

# Pinned rather than floating, to keep builds reproducible (README §10).
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/

WORKDIR /app

# `streamlit run app/streamlit_app.py` puts /app/app on sys.path, not /app,
# so `from rag...` / `from ingestion...` would not resolve without this.
ENV PYTHONPATH=/app \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies first, in their own layer: this only re-runs when
# pyproject.toml / uv.lock change, not on every source edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

# Use the synced venv directly, so CMD needs no `uv run` wrapper.
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
