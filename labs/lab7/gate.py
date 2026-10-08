#!/usr/bin/env python3
"""Lab 7 — the regression gate. Exits non-zero when a threshold is breached.

    python labs/lab7/gate.py --config labs/lab7/thresholds.yml
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def measure() -> dict[str, float]:
    """TODO D1: run the golden set and return the metric dict.

    Keys must match thresholds.yml.
    """
    from aip.chunking import markdown_chunks
    from aip.cost import Budget
    from aip.evals import retrieval_metrics
    from aip.retrieval import DenseRetriever, format_context
    from labs.lab3.search import load_corpus, load_questions
    from labs.lab4.evaluate import judge_correctness, judge_faithfulness
    from labs.lab4.rag import REFUSAL, answer_question

    corpus = load_corpus()
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=800)]
    retriever = DenseRetriever(chunks)

    questions = load_questions(include_unanswerable=True)
    rows = []
    latencies: list[float] = []
    costs: list[float] = []
    hit_rates: list[float] = []

    with Budget(limit_usd=1.50, label="gate-measure") as b:
        for q in questions:
            t0 = time.perf_counter()
            cost_before = b.spent_usd
            a = answer_question(q["question"], retriever)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            cost_after = b.spent_usd

            latencies.append(elapsed_ms)
            costs.append(cost_after - cost_before)

            ctx = format_context(a.hits)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"

            faith = judge_faithfulness(a.text, ctx)
            correct = judge_correctness(q["question"], a.text, q["gold_answer"])

            seen, ranked = set(), []
            for h in a.hits:
                if h.doc_id not in seen:
                    seen.add(h.doc_id)
                    ranked.append(h.doc_id)
            if q["relevant_docs"]:
                rm = retrieval_metrics(ranked, q["relevant_docs"], ks=(5,))
                hit_rates.append(rm["hit_rate@5"])

            rows.append({
                "id": q["id"],
                "kind": q["kind"],
                "unanswerable": unanswerable,
                "refused": a.refused,
                "citations_valid": a.citations_valid,
                "faithfulness": faith if faith >= 0 else None,
                "correctness": correct if correct >= 0 else None,
            })

    valid_faith = [r for r in rows if r["faithfulness"] is not None]
    valid_correct_ans = [r for r in rows
                         if not r["unanswerable"] and r["correctness"] is not None]
    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]

    faithfulness = statistics.fmean(r["faithfulness"] for r in valid_faith) if valid_faith else 0.0
    correctness = (statistics.fmean(r["correctness"] for r in valid_correct_ans) / 2
                   if valid_correct_ans else 0.0)
    citation_validity = statistics.fmean(r["citations_valid"] for r in rows)
    refusal_recall = (sum(1 for r in una if r["refused"]) / len(una)) if una else 0.0
    refusal_precision = ((sum(1 for r in refusals if r["unanswerable"]) / len(refusals))
                         if refusals else 1.0)
    hit_rate_at_5 = statistics.fmean(hit_rates) if hit_rates else 0.0
    cost_per_query = statistics.fmean(costs) if costs else 0.0
    p95_latency = sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "citation_validity": citation_validity,
        "refusal_recall": refusal_recall,
        "refusal_precision": refusal_precision,
        "hit_rate_at_5": hit_rate_at_5,
        "cost_per_query_usd": cost_per_query,
        "p95_latency_ms": p95_latency,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="labs/lab7/thresholds.yml")
    args = ap.parse_args()

    thresholds = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    metrics = measure()

    failures = []
    width = max(len(k) for k in thresholds)
    print(f"{'metric':<{width}}  {'value':>10}  {'gate':>14}  status")
    print("-" * (width + 40))
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not measured")
            print(f"{name:<{width}}  {'—':>10}  {'':>14}  MISSING")
            continue
        ok, gate = True, ""
        if "min" in rule:
            gate, ok = f">= {rule['min']}", value >= rule["min"]
        if "max" in rule and ok:
            gate, ok = f"<= {rule['max']}", value <= rule["max"]
        if not ok:
            failures.append(f"{name}: {value} violates {gate}")
        print(f"{name:<{width}}  {value:>10.4f}  {gate:>14}  {'ok' if ok else 'FAIL'}")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
