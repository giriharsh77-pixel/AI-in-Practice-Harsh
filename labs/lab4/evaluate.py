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
LABEL_SHEET = ROOT / "reports/calibration_labels.jsonl"
FAITHFULNESS_RUBRIC = """\
You are grading ONE thing: whether every claim in the ANSWER is supported by
the CONTEXT. You are not grading helpfulness, style, or whether the answer is
true in the real world.

Rules:
1. A claim is SUPPORTED only if the context states it. Knowledge from outside
   the context makes a claim UNSUPPORTED, even if it is true.
2. A claim that is STRONGER than the context is UNSUPPORTED. Examples: the
   context says "up to 12 sessions" and the answer says "12 sessions"; the
   context attaches a condition, waiting period or exclusion and the answer
   drops it; the context says "may" and the answer says "will".
3. Support may come from any part of the context, or from combining parts of
   it. Do not check whether the [n] citation points to the right source; check
   only whether the claim is in the context.
4. A full refusal ("I don't have enough information in the provided sources
   to answer that.") makes no claims, so it is SUPPORTED.
5. A partial answer (some claims, then that refusal sentence) is judged only
   on the claims it makes. The refusal sentence itself is not a claim.
6. Saying that sources conflict is SUPPORTED if the context does contain the
   conflicting statements.

Score 1 if every claim is supported. Score 0 if any claim is unsupported.

CONTEXT:
{context}

ANSWER:
{answer}

Reply as JSON only:
{{"unsupported_claims": ["..."], "reason": "one sentence", "score": 0 or 1}}
"""

CORRECTNESS_RUBRIC = """\
You are grading ONE thing: whether the CANDIDATE answer gives the same
substantive answer as the REFERENCE. Ignore wording, length, citations, and
extra detail that does not contradict the reference.

If the REFERENCE says the question cannot be answered (the information is not
available, or the correct behaviour is to decline):
  2 = the candidate declines, OR states only what the reference says is
      supported and declines the rest
  1 = the candidate declines but also asserts something the reference does
      not support
  0 = the candidate gives a confident answer, e.g. invents a figure, limit or
      deadline

Otherwise (the REFERENCE gives an answer):
  2 = same substantive content. Every figure, limit, deadline, yes/no and
      eligibility condition the reference states is present and matches
  1 = partially correct: the central fact is right but a secondary detail from
      the reference is missing, OR the candidate answers part of the question
      and declines the rest
  0 = the central fact is wrong (a wrong number, limit, deadline or yes/no),
      OR the candidate fully refuses

Sources that conflict: if the candidate gives the reference's value, score it
as above even if it also mentions an archived or superseded value, as long as
it marks that one as superseded. If it presents both values as equally valid
without saying which is current, score 1. If it gives only the other value,
score 0.

QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {candidate}

Reply as JSON only:
{{"reason": "one sentence", "score": 0, 1 or 2}}
"""


def _mean(xs):
    """Average that skips None (judge parse failures are missing data)."""
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else float("nan")


def build_retriever():
    """Lab 3 winning configuration: markdown-aware chunking at size=400 with
    the heading prefix, dense exact search, no reranker.
    (nDCG@10 0.8527, recall@5 0.9028, hit_rate@1 0.7857 on the Lab 3 set.)"""
    corpus = load_corpus()
    corpus = {d: t for d, t in corpus.items()
              if "ARCHIVED" not in d and not d.startswith(("motor-", "life-", "travel-"))}
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=400)]
    return DenseRetriever(chunks)


# ---------------------------------------------------------------------------
# judges (yours)
# ---------------------------------------------------------------------------
def judge_faithfulness(answer_text: str, context: str) -> int:
    """TODO D1: improve JUDGE_RUBRIC_FAITHFULNESS and return 0 or 1.

    Things the shipped rubric does not yet handle well:
      - a partial refusal (answers part, refuses part)
      - an answer that cites correctly but paraphrases into a stronger claim
      - an answer that is right about the world and wrong about the context
    """
    verdict = llm_judge(FAITHFULNESS_RUBRIC.format(
        context=context[:8000], answer=answer_text), tier="LARGE")
    if verdict.get("parse_error") or "score" not in verdict:
        return None
    return int(verdict["score"])


def judge_correctness(question: str, candidate: str, reference: str) -> int:
    """TODO D1: returns 0, 1 or 2. Handle refusal cases explicitly --
    a correct refusal on an unanswerable question must score 2, and the
    shipped rubric does not say so."""
    verdict = llm_judge(CORRECTNESS_RUBRIC.format(
        question=question, reference=reference, candidate=candidate), tier="LARGE")
    if verdict.get("parse_error") or "score" not in verdict:
        return None
    return int(verdict["score"])


# ---------------------------------------------------------------------------
def run_full(save: str = "") -> None:
    questions = load_questions(include_unanswerable=True)
    retriever = build_retriever()
    rows = []

    with Budget(limit_usd=1.00, label="lab4-full") as b:
        for q in questions:
            a = answer_question(q["question"], retriever)
            ctx = format_context(a.hits)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
            rows.append({
                "id": q["id"], "kind": q["kind"], "unanswerable": unanswerable,
                "answer": a.text, "refused": a.refused,
                "citations_valid": a.citations_valid,
                "invalid_citations": a.invalid_citations,
                "faithfulness": judge_faithfulness(a.text, ctx),
                "correctness": judge_correctness(q["question"], a.text, q["gold_answer"]),
                "retrieved": [h.doc_id for h in a.hits],
                "relevant": q["relevant_docs"],
            })

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]

    print(f"\nn = {len(rows)}  ({len(ans)} answerable, {len(una)} unanswerable)")
    print(f"citation validity   {statistics.fmean(r['citations_valid'] for r in rows):.3f}"
          "   (target 1.000)")
    print(f"faithfulness        {_mean(r['faithfulness'] for r in rows):.3f}")
    print(f"correctness (0-2)   {_mean(r['correctness'] for r in ans):.3f}"
          f"  normalised {_mean(r['correctness'] for r in ans) / 2:.3f}")
    n_pe = sum(1 for r in rows if r["faithfulness"] is None or r["correctness"] is None)
    print(f"judge parse errors  {n_pe} excluded")
    rec = (sum(1 for r in una if r["refused"]) / len(una)) if una else 0.0
    prec = (sum(1 for r in refusals if r["unanswerable"]) / len(refusals)) if refusals else 1.0
    print(f"refusal recall      {rec:.3f}   ({sum(1 for r in una if r['refused'])}/{len(una)})")
    print(f"refusal precision   {prec:.3f}   ({len(refusals)} refusals total)")
    print("\n" + b.report())

    print("\nby question kind (mean correctness / 2):")
    kinds = sorted({r["kind"] for r in ans})
    for kind in kinds:
        sub = [r for r in ans if r["kind"] == kind]
        print(f"  {kind:<16} {_mean(r['correctness'] for r in sub)/2:.3f}"
              f"  n={len(sub)}")

    if save:
        p = ROOT / save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved -> {p}   (Lab 5 reads this file)")


def run_gold_context() -> None:
    """E2: the decomposition. This is the highest-value 10 minutes in the lab."""
    questions = [q for q in load_questions() if q["relevant_docs"]]
    retriever = build_retriever()
    corpus = load_corpus()

    retrieved_scores, gold_scores = [], []
    with Budget(limit_usd=1.00, label="lab4-decomposition"):
        for q in questions:
            a = answer_question(q["question"], retriever)
            retrieved_scores.append(
                judge_correctness(q["question"], a.text, q["gold_answer"]))
            g = answer_with_gold_context(
                q["question"], [corpus[d] for d in q["relevant_docs"] if d in corpus])
            gold_scores.append(
                judge_correctness(q["question"], g.text, q["gold_answer"]))

    A, B = _mean(gold_scores) / 2, _mean(retrieved_scores) / 2
    print(f"\ncorrectness with GOLD context       A = {A:.3f}   <- generation ceiling")
    print(f"correctness with RETRIEVED context  B = {B:.3f}   <- your system")
    print(f"retrieval-attributable loss   A - B = {A - B:.3f}")
    print(f"generation-attributable loss  1 - A = {1 - A:.3f}")
    print("\nWhichever is larger is where Lab 5 goes.")


def make_calibration_sheet() -> None:
    """D2: writes 20 answers for you to hand-label BEFORE seeing the judge."""
    rows = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    refused = [r for r in rows if r["refused"]]
    answered = [r for r in rows if not r["refused"]]
    sample = (refused + answered[::3])[:20]
    LABEL_SHEET.write_text("\n".join(json.dumps({
        "id": r["id"], "answer": r["answer"],
        "human_faithfulness": None, "human_correctness": None,
    }, ensure_ascii=False) for r in sample) + "\n", encoding="utf-8")
    qs = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    retriever = build_retriever()
    view = []
    for r in sample:
        q = qs[r["id"]]
        a = answer_question(q["question"], retriever)
        view += ["=" * 80, f"{r['id']} | {q['kind']}",
                 f"QUESTION: {q['question']}", f"GOLD    : {q['gold_answer']}",
                 f"ANSWER  : {r['answer']}", "CONTEXT :", format_context(a.hits)]
    view_path = LABEL_SHEET.with_name("labelling_view.txt")
    view_path.write_text("\n".join(view) + "\n", encoding="utf-8")

    print(f"wrote {LABEL_SHEET}")
    print(f"wrote {view_path}   <- read this while labelling")
    print("Fill in human_faithfulness (0/1) and human_correctness (0/1/2), then:")
    print("  python labs/lab4/evaluate.py --kappa")


def report_kappa() -> None:
    human = [json.loads(l) for l in LABEL_SHEET.open(encoding="utf-8")]
    machine = {r["id"]: r for r in
               json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))}
    for field in ("faithfulness", "correctness"):
        keep = [r for r in human
                if r[f"human_{field}"] is not None and machine[r["id"]][field] is not None]
        h = [r[f"human_{field}"] for r in keep]
        m = [machine[r["id"]][field] for r in keep]
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