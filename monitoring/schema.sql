-- Run automatically on first Postgres container start (mounted via
-- docker-compose into /docker-entrypoint-initdb.d/).

CREATE TABLE IF NOT EXISTS flat_chunks (
    id UUID PRIMARY KEY,
    video_id TEXT NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    start_sec REAL,
    end_sec REAL
);

CREATE TABLE IF NOT EXISTS parents (
    id UUID PRIMARY KEY,
    video_id TEXT NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    start_sec REAL,
    end_sec REAL
);

-- Full-text search support for BM25-style keyword retrieval
ALTER TABLE flat_chunks ADD COLUMN IF NOT EXISTS text_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS flat_chunks_tsv_idx ON flat_chunks USING GIN (text_tsv);

-- Monitoring / observability
CREATE TABLE IF NOT EXISTS query_log (
    id BIGINT PRIMARY KEY,
    question TEXT NOT NULL,
    rewritten_question TEXT,
    answer TEXT,
    num_sources INT,
    latency_ms REAL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    query_id BIGINT REFERENCES query_log(id),
    is_positive BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
