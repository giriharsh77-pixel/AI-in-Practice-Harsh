#!/usr/bin/env python3
"""Lab 4 evaluation. Scaffolding provided; the judges are yours.

    python labs/lab4/evaluate.py --full --save reports/lab4.json
    python labs/lab4/evaluate.py --gold-context
    python labs/lab4/evaluate.py --calibrate      # writes the hand-label sheet
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.evals import (  # noqa: E402
    JUDGE_RUBRIC_CORRECTNESS,
    JUDGE_RUBRIC_FAITHFULNESS,
    judge_agreement,
    llm_judge,
)
from aip.retrieval import DenseRetriever, format_context  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.rag import REFUSAL, answer_question, answer_with_gold_context  # noqa: E402

GOLDEN = ROOT / "data/eval/rag_golden.jsonl"
LABEL_SHEET = ROOT / "labs/lab4/calibration_labels.jsonl"


def build_retriever():
    """Lab 3 winning configuration: markdown-aware chunking at 800 chars, dense retrieval."""
    corpus = load_corpus()
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=800)]
    return DenseRetriever(chunks)


IMPROVED_FAITHFULNESS_RUBRIC = """\
You are grading whether an ANSWER is fully supported by the provided CONTEXT.

Rules:
- Judge ONLY whether the answer is supported by the context. Do NOT judge
  helpfulness, writing quality, or your own knowledge.
- An answer is UNSUPPORTED (score 0) if it states anything the context does
  not contain, even if that statement is true in the real world.
- Refusing to answer when the context is genuinely insufficient is SUPPORTED
  (score 1).
- A partial refusal (answering part, refusing part) is supported IF the
  answered part is in the context.
- An answer that cites correctly but paraphrases a claim into a STRONGER
  claim than the context warrants is UNSUPPORTED.
- Focus on factual claims, not framing or transition language.

CONTEXT:
{context}

ANSWER:
{answer}

Reply as JSON: {{"score": 0 or 1, "unsupported_claims": [list of unsupported claims if any], "reason": "one sentence"}}
"""

IMPROVED_CORRECTNESS_RUBRIC = """\
Compare a CANDIDATE answer to a REFERENCE answer for the same question.

Scoring:
  Score 2 = same substantive content as the reference (wording may differ).
            If the reference says the correct behaviour is to REFUSE (the
            question is unanswerable), then a refusal scores 2 and a
            confident answer scores 0.
  Score 1 = partially correct: contains some correct content, but omits
            something the reference states, or adds something the reference
            contradicts. A partial refusal that correctly answers the
            answerable part but refuses the unanswerable part scores 1 or 2.
  Score 0 = wrong, contradicts the reference, or confidently answers when the
            reference says to refuse.

QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {candidate}

Reply as JSON: {{"score": 0|1|2, "reason": "one sentence"}}
"""


def judge_faithfulness(answer_text: str, context: str) -> int:
    """Improved faithfulness judge. Returns 0 or 1."""
    verdict = llm_judge(IMPROVED_FAITHFULNESS_RUBRIC.format(
        context=context[:8000], answer=answer_text), tier="LARGE", max_tokens=2048)
    if verdict.get("parse_error"):
        return -1
    return int(verdict.get("score", 0))


def judge_correctness(question: str, candidate: str, reference: str) -> int:
    """Improved correctness judge with explicit refusal handling. Returns 0, 1 or 2."""
    verdict = llm_judge(IMPROVED_CORRECTNESS_RUBRIC.format(
        question=question, reference=reference, candidate=candidate),
        tier="LARGE", max_tokens=2048)
    if verdict.get("parse_error"):
        return -1
    return int(verdict.get("score", 0))


def run_full(save: str = "") -> None:
    questions = load_questions(include_unanswerable=True)
    retriever = build_retriever()
    rows = []

    with Budget(limit_usd=1.00, label="lab4-full") as b:
        for q in questions:
            a = answer_question(q["question"], retriever)
            ctx = format_context(a.hits)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"

            faith = judge_faithfulness(a.text, ctx)
            correct = judge_correctness(q["question"], a.text, q["gold_answer"])

            rows.append({
                "id": q["id"], "kind": q["kind"], "unanswerable": unanswerable,
                "answer": a.text, "refused": a.refused,
                "citations_valid": a.citations_valid,
                "invalid_citations": a.invalid_citations,
                "faithfulness": faith if faith >= 0 else None,
                "correctness": correct if correct >= 0 else None,
                "retrieved": [h.doc_id for h in a.hits],
                "relevant": q["relevant_docs"],
            })

    valid_faith = [r for r in rows if r["faithfulness"] is not None]
    valid_correct_ans = [r for r in rows
                         if not r["unanswerable"] and r["correctness"] is not None]

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]

    print(f"\nn = {len(rows)}  ({len(ans)} answerable, {len(una)} unanswerable)")
    print(f"citation validity   {statistics.fmean(r['citations_valid'] for r in rows):.3f}"
          "   (target 1.000)")
    if valid_faith:
        print(f"faithfulness        {statistics.fmean(r['faithfulness'] for r in valid_faith):.3f}"
              f"   (n={len(valid_faith)}, excluded {len(rows)-len(valid_faith)} parse errors)")
    if valid_correct_ans:
        mean_c = statistics.fmean(r['correctness'] for r in valid_correct_ans)
        print(f"correctness (0-2)   {mean_c:.3f}"
              f"  normalised {mean_c / 2:.3f}")
    rec = (sum(1 for r in una if r["refused"]) / len(una)) if una else 0.0
    prec = (sum(1 for r in refusals if r["unanswerable"]) / len(refusals)) if refusals else 1.0
    print(f"refusal recall      {rec:.3f}   ({sum(1 for r in una if r['refused'])}/{len(una)})")
    print(f"refusal precision   {prec:.3f}   ({len(refusals)} refusals total)")
    print("\n" + b.report())

    print("\nby question kind (mean correctness / 2):")
    kinds = sorted({r["kind"] for r in ans})
    for kind in kinds:
        sub = [r for r in ans if r["kind"] == kind and r["correctness"] is not None]
        if sub:
            print(f"  {kind:<16} {statistics.fmean(r['correctness'] for r in sub)/2:.3f}"
                  f"  n={len(sub)}")

    if save:
        for r in rows:
            if r["faithfulness"] is None:
                r["faithfulness"] = 0
            if r["correctness"] is None:
                r["correctness"] = 0
        p = ROOT / save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved -> {p}   (Lab 5 reads this file)")


def run_gold_context() -> None:
    """E2: the decomposition."""
    questions = [q for q in load_questions() if q["relevant_docs"]]
    retriever = build_retriever()
    corpus = load_corpus()

    retrieved_scores, gold_scores = [], []
    with Budget(limit_usd=1.00, label="lab4-decomposition"):
        for q in questions:
            a = answer_question(q["question"], retriever)
            c = judge_correctness(q["question"], a.text, q["gold_answer"])
            retrieved_scores.append((c if c >= 0 else 0) / 2)

            g = answer_with_gold_context(
                q["question"], [corpus[d] for d in q["relevant_docs"] if d in corpus])
            gc = judge_correctness(q["question"], g.text, q["gold_answer"])
            gold_scores.append((gc if gc >= 0 else 0) / 2)

    A, B = statistics.fmean(gold_scores), statistics.fmean(retrieved_scores)
    print(f"\ncorrectness with GOLD context       A = {A:.3f}   <- generation ceiling")
    print(f"correctness with RETRIEVED context  B = {B:.3f}   <- your system")
    print(f"retrieval-attributable loss   A - B = {A - B:.3f}")
    print(f"generation-attributable loss  1 - A = {1 - A:.3f}")
    print("\nWhichever is larger is where Lab 5 goes.")


def make_calibration_sheet() -> None:
    rows = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    sample = rows[:20]
    LABEL_SHEET.write_text("\n".join(json.dumps({
        "id": r["id"], "answer": r["answer"],
        "human_faithfulness": None, "human_correctness": None,
    }, ensure_ascii=False) for r in sample) + "\n", encoding="utf-8")
    print(f"wrote {LABEL_SHEET}")
    print("Fill in human_faithfulness (0/1) and human_correctness (0/1/2), then:")
    print("  python labs/lab4/evaluate.py --kappa")


def report_kappa() -> None:
    human = [json.loads(l) for l in LABEL_SHEET.open(encoding="utf-8")]
    machine = {r["id"]: r for r in
               json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))}
    for field in ("faithfulness", "correctness"):
        h = [r[f"human_{field}"] for r in human if r[f"human_{field}"] is not None]
        m = [machine[r["id"]][field] for r in human if r[f"human_{field}"] is not None]
        if not h:
            print(f"{field}: no human labels yet")
            continue
        print(f"{field}: {judge_agreement(m, h)}")
    print("\nkappa < 0.4 -> fix the rubric, not the model. Read your disagreements.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gold-context", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    if a.full:
        run_full(a.save)
    if a.gold_context:
        run_gold_context()
    if a.calibrate:
        make_calibration_sheet()
    if a.kappa:
        report_kappa()
    if not any([a.full, a.gold_context, a.calibrate, a.kappa]):
        ap.print_help()
