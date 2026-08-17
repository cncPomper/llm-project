# Evaluation Results

This file is appended to automatically by `retrieval_eval.py` and
`llm_eval.py`. Run them after ingestion + `generate_ground_truth.py`:

```bash
python -m eval.generate_ground_truth
python -m eval.retrieval_eval
python -m eval.llm_eval
```

Results will appear below as they're generated.

## Retrieval Evaluation

| Strategy | Hit Rate@5 | MRR@5 | N |
|---|---|---|---|
| flat | 0.84 | 0.697 | 100 |
| parent_document | 0.9 | 0.791 | 100 |
| hybrid | 0.89 | 0.752 | 100 |
| hybrid_rerank | 0.92 | 0.862 | 100 |

## LLM Answer Evaluation

| Strategy | Faithfulness | Relevance | Specificity | N |
|---|---|---|---|---|
| plain_rag | 4.67 | 4.83 | 3.97 | 30 |
| rewrite_rag | 3.73 | 3.87 | 3.07 | 30 |
| rewrite_rerank_rag | 4.23 | 4.23 | 3.57 | 30 |
