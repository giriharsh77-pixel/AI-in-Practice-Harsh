#!/usr/bin/env python3
"""Lab 5 — the failure classifier.

    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto

Implements the T4 §5 diagnostic tree. Everything that can be decided by code
is decided by code; mode 2 needs your eyes and the script says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.evaluate import build_retriever, judge_correctness  # noqa: E402
from labs.lab4.rag import answer_with_gold_context  # noqa: E402

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}

A2_JUDGEMENTS = {
    "Q19": (2, "A2: worked example split across chunks; the chunk saying only room-linked charges are reduced was not retrieved"),
    "Q22": (6, "A2: missing fact was in retrieved chunk [2]; the answer omitted it, gold-context fix was run-to-run variation"),
    "Q28": (2, "A2: pre-approval condition is a table row cut off from its header; read as a conflicting rule"),
    "Q29": (4, "A2: correct figure retrieved and given; a motor-insurance chunk ranked into the top 5 and was quoted"),
    "Q32": (2, "A2: Platinum's nil co-payment only in a table row without its header; plan columns unidentifiable"),
}


def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
                     relevant_docs: list[str]) -> bool:
    """Mode 1 test. Checks the gold answer's numbers against the relevant
    documents (commas removed, ranges split); at least half must appear.
    Falls back to word overlap when the answer has no numbers."""
    text = " ".join(corpus.get(d, "") for d in relevant_docs).lower().replace(",", "")
    if not text:
        return False
    words = [w.strip(".,;:()%").replace(",", "")
             for w in gold_answer.lower().replace("-", " ").replace("/", " ").split()]
    nums = [w for w in words if any(ch.isdigit() for ch in w)]
    if nums:
        return sum(1 for n in nums if n in text) / len(nums) >= 0.5
    tokens = [w for w in words if len(w) > 4]
    if not tokens:
        return True
    return sum(1 for t in tokens if t in text) / len(tokens) > 0.4


def classify(row: dict, q: dict, corpus: dict[str, str], *,
             gold_context_fixes_it: bool | None = None,
             gold_chunk_rank: int | None = None,
             self_retrieves: bool | None = None,
             dropped_by_reranker: bool | None = None,
             final_k: int = 5) -> tuple[int, str]:
    """Walk the T4 §5 diagnostic tree. Returns (mode, evidence).

    TODO: complete the branches marked TODO. Follow the tree in the handout;
    do not invent your own ordering, because the ordering is what makes the
    modes mutually exclusive.
    """
    # Mode 7 first: right answer, wrong citation. Check this before anything
    # else, because a mode-7 failure is not a retrieval failure at all.
    if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
        return 7, f"correct answer, invalid citations {row.get('invalid_citations')}"

    # Mode 1: is the answer even in the corpus?
    if not answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
        return 1, "gold answer's key figures not found in the relevant documents"

    # Mode 6: does gold context fix it?
    # TODO: if gold_context_fixes_it is False, this is a generation failure.
    #       Note the direction -- gold context FIXING the answer means
    #       RETRIEVAL was at fault, not generation. People get this backwards.
    if gold_context_fixes_it is None:
        return 2, "needs_human_check: gold-context test unavailable"
    if not gold_context_fixes_it:
        return 6, "still wrong with gold context: generation"

    if gold_chunk_rank is not None and gold_chunk_rank <= final_k:
        return 2, (f"needs_human_check: gold chunk was in context (rank {gold_chunk_rank}) "
                   f"yet gold context fixes it")
    # TODO Mode 4/5: gold doc in top 30 but not in the final k
    #       -> 5 if the reranker dropped it, else 4
    if gold_chunk_rank is not None:
        if dropped_by_reranker:
            return 5, f"gold chunk at rank {gold_chunk_rank}, dropped by reranker"
        return 4, f"gold chunk at rank {gold_chunk_rank}, outside final_k={final_k}"

    # TODO Mode 3: gold doc not even in the top 30. Confirm by searching for
    #       the gold chunk's own text -- if THAT retrieves it, the query is the
    #       problem (mode 3). If it does not, the chunk itself is unfindable
    #       (mode 2, needs your eyes).
    if self_retrieves:
        return 3, "gold chunk not in top 30 for the query, but retrievable by its own text"
    return 2, "needs_human_check: gold chunk not retrievable even by its own text"


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
    ap.add_argument("--final-k", type=int, default=5)
    ap.add_argument("--no-a2", action="store_true")
    args = ap.parse_args()

    rows = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    corpus = load_corpus()
    retriever = build_retriever()

    failures = [r for r in rows
                if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
    print(f"{len(failures)} failures out of {len(rows)}\n")

    out, tally = [], Counter()
    
    for r in failures:
        q = questions[r["id"]]
        fixes = rank = self_ret = gc = gscore = None
        if q["relevant_docs"]:
            g = answer_with_gold_context(
                q["question"], [corpus[d] for d in q["relevant_docs"] if d in corpus])
            gscore = judge_correctness(q["question"], g.text, q["gold_answer"])
            fixes = None if gscore is None else gscore == 2
            rel = set(q["relevant_docs"])
            gc = next((h.chunk for h in retriever.search(q["gold_answer"], k=len(retriever.chunks))
                       if h.doc_id in rel), None)
            if gc:
                top30 = [h.chunk.chunk_id for h in retriever.search(q["question"], k=30)]
                rank = top30.index(gc.chunk_id) + 1 if gc.chunk_id in top30 else None
                self_ret = gc.chunk_id in [h.chunk.chunk_id for h in retriever.search(gc.text, k=30)]
        mode, evidence = classify(r, q, corpus, gold_context_fixes_it=fixes,
                                  gold_chunk_rank=rank, self_retrieves=self_ret,
                                  dropped_by_reranker=False, final_k=args.final_k)
        if not args.no_a2 and r["id"] in A2_JUDGEMENTS:
            mode, evidence = A2_JUDGEMENTS[r["id"]]
        tally[mode] += 1
        out.append({"id": r["id"], "kind": q["kind"], "mode": mode,
                    "mode_name": MODES[mode], "evidence": evidence,
                    "gold_context_correctness": gscore, "gold_chunk_rank": rank,
                    "gold_chunk": gc.text[:400] if gc else None,
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
