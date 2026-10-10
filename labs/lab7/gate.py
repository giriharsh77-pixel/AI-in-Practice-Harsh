#!/usr/bin/env python3
"""Lab 7 — the regression gate. Exits non-zero when a threshold is breached.

    python labs/lab7/gate.py --config labs/lab7/thresholds.yml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache  # noqa: E402
from aip.retrieval import format_context  # noqa: E402
from labs.lab3.search import load_questions  # noqa: E402
from labs.lab4 import rag  # noqa: E402
from labs.lab4.evaluate import build_retriever, judge_correctness, judge_faithfulness  # noqa: E402


def measure() -> dict[str, float]:
    calls = []                                   
    real_chat = rag.chat

    def spy(*args, **kwargs):
        out = real_chat(*args, **kwargs)
        calls.append(out["usage"])
        return out

    rag.chat = spy
    rows = []
    try:
        retriever = build_retriever()
        for q in load_questions(include_unanswerable=True):
            calls.clear()
            a = rag.answer_question(q["question"], retriever)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
            rows.append({
                "id": q["id"], "unanswerable": unanswerable, "refused": a.refused,
                "citations_valid": a.citations_valid,
                "faithfulness": judge_faithfulness(a.text, format_context(a.hits)),
                "correctness": None if unanswerable else
                judge_correctness(q["question"], a.text, q["gold_answer"]),
                "hit_at_5": any(h.doc_id in q["relevant_docs"] for h in a.hits[:5]),
                "cost_usd": sum(u["cost_usd"] for u in calls),
                "latency_ms": sum(u["latency_ms"] for u in calls),
            })
    except cache.CacheMiss as exc:
        raise SystemExit("GATE ERROR: a call is not in the committed cache, so the pipeline or a "
                         "prompt changed.\nRun the gate once online with AIP_CACHE_DIR=ci_cache "
                         f"to record it, then commit ci_cache/.\n\n{exc}") from exc
    finally:
        rag.chat = real_chat

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]

    def mean(xs):
        return float(np.mean([x for x in xs if x is not None]))   # None = judge parse failure

    metrics = {
        "correctness": mean(r["correctness"] for r in ans) / 2,
        "faithfulness": mean(r["faithfulness"] for r in rows),
        "citation_validity": mean(r["citations_valid"] for r in rows),
        "refusal_recall": sum(r["refused"] for r in una) / len(una),
        "refusal_precision": sum(r["unanswerable"] for r in refusals) / len(refusals) if refusals else 1.0,
        "hit_rate_at_5": mean(r["hit_at_5"] for r in ans),
        "cost_per_query_usd": mean(r["cost_usd"] for r in rows),
        "p95_latency_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
    }
    out = ROOT / "reports/gate.json"                 
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2), encoding="utf-8")
    return metrics


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