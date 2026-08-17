"""
Compares final-answer quality across 3 LLM strategies using an
LLM-as-judge rubric:

  A. plain_rag            -- retrieve(raw question) -> generate
  B. rewrite_rag           -- rewrite_query(question) -> retrieve -> generate
  C. rewrite_rerank_rag     -- rewrite_query(question) -> retrieve w/ hybrid_rerank -> generate

Judge scores each answer 1-5 on: faithfulness (grounded in excerpts, no
hallucinated timestamps/numbers), relevance, and specificity.

Run from the repo root:

  python -m eval.llm_eval
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag.prompts import generate_answer, rewrite_query
from rag.retrieval import retrieve

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.md"

JUDGE_PROMPT = """You are grading a podcast Q&A assistant's answer.

Question: {question}

Answer: {answer}

Source excerpts the answer should be grounded in:
{excerpts}

Score the answer from 1 (bad) to 5 (excellent) on each dimension. \
Return ONLY a JSON object like:
{{"faithfulness": <1-5>, "relevance": <1-5>, "specificity": <1-5>}}"""


def parse_scores(raw: str) -> dict | None:
    """Parse the judge's JSON, tolerating a Markdown code fence.

    Despite "Return ONLY a JSON object", the model wraps its answer in
    ```json ... ``` for longer prompts, which json.loads rejects. Returns None
    when the response genuinely cannot be parsed, so the caller can drop the
    sample rather than score it.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip the opening fence (with optional language tag) and closing one.
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def judge(question: str, answer: str, chunks: list[dict]) -> dict | None:
    """Returns None if the judge's reply could not be parsed.

    Deliberately NOT a zero score: scoring an unparseable reply 0/0/0 silently
    turns a formatting quirk into a damning result, and pulls a strategy's
    average down hard for reasons that have nothing to do with its answers.
    """
    excerpts = "\n\n".join(c["text"] for c in chunks)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, answer=answer, excerpts=excerpts)}],
        temperature=0,
    )
    return parse_scores(resp.choices[0].message.content)


def run_strategy(name: str, question: str) -> dict:
    if name == "plain_rag":
        chunks = retrieve(question, strategy="hybrid")
    elif name == "rewrite_rag":
        chunks = retrieve(rewrite_query(question), strategy="hybrid")
    elif name == "rewrite_rerank_rag":
        chunks = retrieve(rewrite_query(question), strategy="hybrid_rerank")
    else:
        raise ValueError(name)

    answer = generate_answer(question, chunks)
    return {"answer": answer, "scores": judge(question, answer, chunks)}


def main(sample_size: int = 30):
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = [json.loads(line) for line in f][:sample_size]

    if not ground_truth:
        print("No ground truth found -- run eval/generate_ground_truth.py first.")
        return

    strategies = ["plain_rag", "rewrite_rag", "rewrite_rerank_rag"]
    dims = ("faithfulness", "relevance", "specificity")
    totals = {s: dict.fromkeys(dims, 0) for s in strategies}
    # Counted per strategy, because averaging over the full sample when some
    # samples were dropped would understate every score it dropped one from.
    scored = dict.fromkeys(strategies, 0)

    for item in ground_truth:
        for s in strategies:
            result = run_strategy(s, item["question"])
            if result["scores"] is None:
                continue
            scored[s] += 1
            for dim in dims:
                totals[s][dim] += result["scores"].get(dim, 0)

    table = "| Strategy | Faithfulness | Relevance | Specificity | N |\n|---|---|---|---|---|\n"
    for s in strategies:
        n = scored[s]
        if not n:
            table += f"| {s} | - | - | - | 0 |\n"
            continue
        f_, r_, sp_ = (totals[s][d] / n for d in dims)
        table += f"| {s} | {f_:.2f} | {r_:.2f} | {sp_:.2f} | {n} |\n"

    dropped = {s: len(ground_truth) - scored[s] for s in strategies if scored[s] < len(ground_truth)}
    if dropped:
        table += ("\nUnparseable judge replies, excluded from the averages: "
                  + ", ".join(f"{s} {n}" for s, n in dropped.items()) + "\n")

    print(table)
    with open(RESULTS_PATH, "a") as f:
        f.write("\n## LLM Answer Evaluation\n\n" + table)


if __name__ == "__main__":
    main()
