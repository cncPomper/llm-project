FROM python:3.11-slim

WORKDIR /app

# `streamlit run app/streamlit_app.py` puts /app/app on sys.path, not /app,
# so `from rag...` / `from ingestion...` would not resolve without this.
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
