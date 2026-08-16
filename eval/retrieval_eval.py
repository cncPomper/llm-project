"""
Evaluates each retrieval strategy against eval/ground_truth.jsonl using:

  Hit Rate @ K -- fraction of questions where the known source text
                  overlaps with at least one retrieved chunk in the top K
  MRR @ K      -- mean reciprocal rank of the first retrieved chunk that
                  overlaps the known source text

"Overlap" is checked by substring/containment between the ground-truth
chunk text and retrieved chunk/parent text, since parent-document
retrieval returns a different (larger) span than the original flat chunk.

Run: python eval/retrieval_eval.py
Writes a Markdown table appended to eval/results.md
"""
import json
from pathlib import Path

from rag.retrieval import retrieve

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.md"
STRATEGIES = ["flat", "parent_document", "hybrid", "hybrid_rerank"]
TOP_K = 5


def load_ground_truth():
    with open(GROUND_TRUTH_PATH) as f:
        return [json.loads(line) for line in f]


def is_hit(gt_text: str, retrieved_text: str) -> bool:
    # crude but robust overlap check across differing chunk granularities
    gt_snippet = gt_text[:80].strip()
    return gt_snippet[:40] in retrieved_text or retrieved_text[:80] in gt_text


def evaluate_strategy(strategy: str, ground_truth: list[dict]) -> dict:
    hits, reciprocal_ranks = 0, []
    for item in ground_truth:
        results = retrieve(item["question"], strategy=strategy, top_k=TOP_K)
        rank = None
        for i, r in enumerate(results, start=1):
            if r["video_id"] == item["video_id"] and is_hit(item["chunk_text"], r["text"]):
                rank = i
                break
        if rank:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(ground_truth)
    return {
        "strategy": strategy,
        "hit_rate": round(hits / n, 3) if n else 0.0,
        "mrr": round(sum(reciprocal_ranks) / n, 3) if n else 0.0,
        "n": n,
    }


def main():
    ground_truth = load_ground_truth()
    if not ground_truth:
        print("No ground truth found -- run eval/generate_ground_truth.py first.")
        return

    rows = [evaluate_strategy(s, ground_truth) for s in STRATEGIES]

    table = "| Strategy | Hit Rate@{k} | MRR@{k} | N |\n|---|---|---|---|\n".format(k=TOP_K)
    for r in rows:
        table += f"| {r['strategy']} | {r['hit_rate']} | {r['mrr']} | {r['n']} |\n"

    print(table)
    with open(RESULTS_PATH, "a") as f:
        f.write("\n## Retrieval Evaluation\n\n" + table)


if __name__ == "__main__":
    main()
