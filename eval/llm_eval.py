"""
Compares final-answer quality across 3 LLM strategies using an
LLM-as-judge rubric:

  A. plain_rag            -- retrieve(raw question) -> generate
  B. rewrite_rag           -- rewrite_query(question) -> retrieve -> generate
  C. rewrite_rerank_rag     -- rewrite_query(question) -> retrieve w/ hybrid_rerank -> generate

Judge scores each answer 1-5 on: faithfulness (grounded in excerpts, no
hallucinated timestamps/numbers), relevance, and specificity. Run:

  python eval/llm_eval.py
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


def judge(question: str, answer: str, chunks: list[dict]) -> dict:
    excerpts = "\n\n".join(c["text"] for c in chunks)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, answer=answer, excerpts=excerpts)}],
        temperature=0,
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {"faithfulness": 0, "relevance": 0, "specificity": 0}


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
    scores = judge(question, answer, chunks)
    return {"answer": answer, **scores}


def main(sample_size: int = 30):
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = [json.loads(line) for line in f][:sample_size]

    if not ground_truth:
        print("No ground truth found -- run eval/generate_ground_truth.py first.")
        return

    strategies = ["plain_rag", "rewrite_rag", "rewrite_rerank_rag"]
    totals = {s: {"faithfulness": 0, "relevance": 0, "specificity": 0} for s in strategies}

    for item in ground_truth:
        for s in strategies:
            result = run_strategy(s, item["question"])
            for dim in ("faithfulness", "relevance", "specificity"):
                totals[s][dim] += result[dim]

    n = len(ground_truth)
    table = "| Strategy | Faithfulness | Relevance | Specificity | N |\n|---|---|---|---|---|\n"
    for s in strategies:
        f_, r_, sp_ = (totals[s][d] / n for d in ("faithfulness", "relevance", "specificity"))
        table += f"| {s} | {f_:.2f} | {r_:.2f} | {sp_:.2f} | {n} |\n"

    print(table)
    with open(RESULTS_PATH, "a") as f:
        f.write("\n## LLM Answer Evaluation\n\n" + table)


if __name__ == "__main__":
    main()
