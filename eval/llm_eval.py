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

The default question set is eval/ground_truth.jsonl, whose questions are
LLM-generated *from* a chunk and therefore already keyword-rich -- which is
exactly the shape query rewriting is supposed to produce, so that set cannot
show whether rewriting earns its keep. Point --questions at a hand-written,
conversational set to test that:

  python -m eval.llm_eval --questions eval/questions_conversational.jsonl \
      --label "LLM Answer Evaluation -- conversational questions"

Any JSONL file with a "question" field works; nothing else is read.
"""
import argparse
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
        search_query = question
        chunks = retrieve(search_query, strategy="hybrid")
    elif name == "rewrite_rag":
        search_query = rewrite_query(question)
        chunks = retrieve(search_query, strategy="hybrid")
    elif name == "rewrite_rerank_rag":
        search_query = rewrite_query(question)
        chunks = retrieve(search_query, strategy="hybrid_rerank")
    else:
        raise ValueError(name)

    answer = generate_answer(question, chunks)
    return {
        "answer": answer,
        "search_query": search_query,
        "scores": judge(question, answer, chunks),
    }


def main(sample_size: int = 30,
         questions_path: Path = GROUND_TRUTH_PATH,
         label: str = "LLM Answer Evaluation"):
    with open(questions_path, encoding="utf-8") as f:
        ground_truth = [json.loads(line) for line in f if line.strip()][:sample_size]

    if not ground_truth:
        print(f"No questions found in {questions_path} -- for the default set, "
              f"run eval/generate_ground_truth.py first.")
        return

    print(f"{len(ground_truth)} questions from {questions_path.name}\n", flush=True)

    strategies = ["plain_rag", "rewrite_rag", "rewrite_rerank_rag"]
    dims = ("faithfulness", "relevance", "specificity")
    totals = {s: dict.fromkeys(dims, 0) for s in strategies}
    # Counted per strategy, because averaging over the full sample when some
    # samples were dropped would understate every score it dropped one from.
    scored = dict.fromkeys(strategies, 0)

    # Kept so the write-up can show what rewriting actually did to a
    # conversational question, which the score table alone cannot convey.
    rewrites: list[tuple[str, str]] = []

    for i, item in enumerate(ground_truth, 1):
        for s in strategies:
            result = run_strategy(s, item["question"])
            if s == "rewrite_rag" and len(rewrites) < 6:
                rewrites.append((item["question"], result["search_query"]))
            if result["scores"] is None:
                continue
            scored[s] += 1
            for dim in dims:
                totals[s][dim] += result["scores"].get(dim, 0)
        print(f"[{i:2d}/{len(ground_truth)}] {item['question'][:60]}", flush=True)

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

    if rewrites:
        table += "\nWhat query rewriting did to the first few questions:\n\n"
        table += "| Asked | Rewritten to |\n|---|---|\n"
        for raw, rewritten in rewrites:
            table += f"| {raw} | {rewritten} |\n"

    print(table)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n## {label}\n\n" + table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=GROUND_TRUTH_PATH,
                        help="JSONL file with a 'question' field per line")
    parser.add_argument("--label", default="LLM Answer Evaluation",
                        help="Heading written to eval/results.md")
    parser.add_argument("-n", "--sample-size", type=int, default=30)
    args = parser.parse_args()
    main(sample_size=args.sample_size, questions_path=args.questions, label=args.label)
