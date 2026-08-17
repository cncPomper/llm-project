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
| plain_rag | 4.73 | 4.87 | 4.10 | 30 |
| rewrite_rag | 4.77 | 4.90 | 3.93 | 30 |
| rewrite_rerank_rag | 4.57 | 4.73 | 4.03 | 30 |

All three are within ~0.2 of each other on a 1–5 scale at N=30, which is
noise rather than a difference. The rubric does not separate them: every
strategy retrieves from the same corpus with the same generator, and the
judge rates them all as faithful and relevant. Specificity is consistently
the weakest dimension, which is the more useful signal — answers stay
grounded but stay general.

### Superseded: first LLM evaluation run

The initial run produced these numbers, which were **wrong** and are kept
only to explain the correction:

| Strategy | Faithfulness | Relevance | Specificity | N |
|---|---|---|---|---|
| plain_rag | 4.67 | 4.83 | 3.97 | 30 |
| rewrite_rag | 3.73 | 3.87 | 3.07 | 30 |
| rewrite_rerank_rag | 4.23 | 4.23 | 3.57 | 30 |

`judge()` parsed the model's reply with `json.loads` and, on failure,
returned `{"faithfulness": 0, "relevance": 0, "specificity": 0}`. For longer
prompts the model wraps its JSON in a ```` ```json ```` fence, which
`json.loads` rejects — so a perfectly good answer was scored zero. The
failures were not evenly distributed across strategies, so `rewrite_rag`
absorbed most of them and appeared to lose by a full point. Fixed by
stripping the fence and dropping unparseable samples instead of scoring
them zero.

## LLM Answer Evaluation -- hand-written conversational questions

| Strategy | Faithfulness | Relevance | Specificity | N |
|---|---|---|---|---|
| plain_rag | 4.17 | 4.46 | 3.54 | 24 |
| rewrite_rag | 4.08 | 4.42 | 3.33 | 24 |
| rewrite_rerank_rag | 3.79 | 4.25 | 3.25 | 24 |

What query rewriting did to the first few questions:

| Asked | Rewritten to |
|---|---|
| my dog goes nuts every time someone rings the doorbell, what do i do | how to stop dog barking at doorbell |
| whats that calm energy thing the dog guy keeps going on about | calm energy dog training technique |
| is walking the dog every day actually that important or is it overrated | importance of daily dog walking |
| how are you supposed to say hi to a dog you dont know | how to greet an unfamiliar dog |
| my rescue is super anxious all the time, any idea whats going on there | rescue dog anxiety causes and solutions |
| what did elon actually end up cutting with that government efficiency thing | Elon Musk government efficiency cuts details |
