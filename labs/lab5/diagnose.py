#!/usr/bin/env python3
"""Lab 5 — the failure classifier.

    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labs.lab3.search import load_corpus, load_questions  # noqa: E402

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _extract_key_terms(gold_answer: str) -> list[str]:
    terms = []
    numbers = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%?", gold_answer)
    terms.extend(numbers)
    words = [w for w in re.findall(r"[a-zA-Z]+", gold_answer.lower()) if len(w) > 4]
    terms.extend(words)
    return terms


def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
                     relevant_docs: list[str]) -> bool:
    """Mode 1 test. Improved: checks key numbers and significant terms."""
    text = " ".join(corpus.get(d, "") for d in relevant_docs).lower()
    if not text:
        return False

    key_terms = _extract_key_terms(gold_answer)
    if not key_terms:
        norm_gold = _normalize(gold_answer)
        norm_text = _normalize(text)
        tokens = [t for t in norm_gold.split() if len(t) > 3]
        if not tokens:
            return True
        return sum(1 for t in tokens if t in norm_text) / len(tokens) > 0.3

    numbers = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%?", gold_answer)
    if numbers:
        found = sum(1 for n in numbers if n in text)
        if found >= max(1, len(numbers) * 0.5):
            return True

    words = [w for w in key_terms if not re.match(r"\d", w)]
    if words:
        found = sum(1 for w in words if w in text)
        return found / len(words) > 0.3

    return True


def classify(row: dict, q: dict, corpus: dict[str, str], *,
             gold_context_fixes_it: bool | None = None,
             in_top_30: bool | None = None,
             dropped_by_reranker: bool | None = None) -> tuple[int, str]:
    """Walk the T4 diagnostic tree. Returns (mode, evidence)."""

    if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
        return 7, f"correct answer, invalid citations {row.get('invalid_citations')}"

    if not answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
        return 1, "gold answer content not found in the relevant documents"

    if gold_context_fixes_it is not None:
        if not gold_context_fixes_it:
            return 6, "generation failure: gold context did NOT fix the answer"

    if in_top_30 is not None and in_top_30:
        if dropped_by_reranker:
            return 5, "reranker dropped the gold document from the final set"
        return 4, "gold doc in top 30 but buried below final_k"

    if in_top_30 is not None and not in_top_30:
        return 3, "gold doc not in top 30 -- embedding mismatch or query phrasing"

    if gold_context_fixes_it is True:
        retrieved = row.get("retrieved", [])
        relevant = q.get("relevant_docs", [])
        found_in_retrieved = any(r in relevant for r in retrieved)
        if found_in_retrieved:
            return 4, "retrieval issue: gold doc retrieved but likely not ranked high enough"
        return 3, "retrieval issue: gold doc not found in retrieved set"

    return 2, "needs_human_check: open the chunks around the gold answer"


def pareto(tally: Counter) -> str:
    total = sum(tally.values()) or 1
    lines, cum = ["failure mode          n    share   cumulative"], 0
    for mode, n in tally.most_common():
        cum += n
        bar = "█" * round(30 * n / total)
        lines.append(f"{MODES[mode]:<20} {n:>3}   {n/total:>5.1%}   "
                     f"{cum/total:>5.1%}  {bar}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/lab4.json")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--save", default="reports/lab5_diagnosis.json")
    args = ap.parse_args()

    rows = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    corpus = load_corpus()

    failures = [r for r in rows
                if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
    print(f"{len(failures)} failures out of {len(rows)}\n")

    out, tally = [], Counter()
    for r in failures:
        q = questions[r["id"]]

        retrieved_set = set(r.get("retrieved", []))
        relevant_set = set(q.get("relevant_docs", []))

        if relevant_set and not relevant_set & retrieved_set:
            gold_context_fixes_it = True
            in_top_30 = False
        elif relevant_set and relevant_set & retrieved_set:
            gold_context_fixes_it = True
            in_top_30 = True
        else:
            gold_context_fixes_it = None
            in_top_30 = None

        mode, evidence = classify(
            r, q, corpus,
            gold_context_fixes_it=gold_context_fixes_it,
            in_top_30=in_top_30,
            dropped_by_reranker=False,
        )
        tally[mode] += 1
        out.append({"id": r["id"], "kind": q["kind"], "mode": mode,
                    "mode_name": MODES[mode], "evidence": evidence,
                    "question": q["question"], "answer": r["answer"][:300]})
        print(f"  {r['id']:<5} {MODES[mode]:<20} {evidence}")

    print("\n" + pareto(tally))
    print("\nCases marked needs_human_check are Part A2. Open them.")

    p = ROOT / args.save
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
