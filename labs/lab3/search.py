#!/usr/bin/env python3
"""Lab 3 — retrieval sweeps.

The scaffolding (corpus loading, metric computation, table printing) is
written for you. The sweeps are yours.

    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import STRATEGIES, Chunk  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import Bm25Retriever, DenseRetriever, HybridRetriever, Retriever  # noqa: E402

CORPUS_DIR = ROOT / "data/corpus"
GOLDEN = ROOT / "data/eval/rag_golden.jsonl"


# ---------------------------------------------------------------------------
# scaffolding (provided)
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    if include_unanswerable:
        return rows
    # THREE questions (Q36, Q38, Q39) have no relevant document, so recall and
    # nDCG are undefined for them -- you cannot rank correctly against an empty
    # relevant set. Dropping them leaves n = 42.
    #
    # Do not confuse that with the FIVE questions of kind 'unanswerable'
    # (Q36-Q40): two of those do keep relevant documents, because part of what
    # they ask is supported. All five are measured properly in Lab 4, as
    # refusal precision and recall.
    #
    # Excluding the three is correct -- but say so in your report rather than
    # letting an unexplained n = 42 pass for a stated 45.
    return [r for r in rows if r["relevant_docs"]]


def build_chunks(corpus: dict[str, str], strategy: str = "sliding",
                 size: int = 800, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for doc_id, text in corpus.items():
        try:
            out.extend(fn(text, doc_id, size=size, **kw))
        except TypeError:                       # chunker without that kwarg
            out.extend(fn(text, doc_id, size=size))
    return out


def evaluate(retriever: Retriever, questions: list[dict], k: int = 10,
             reranker=None, final_k: int = 5) -> dict:
    """Run every question, return aggregate metrics + per-kind breakdown."""
    agg: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_q: dict[str, float] = {}
    per_q_mrr: dict[str, float] = {}

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k)
        if reranker is not None:
            hits = reranker.rerank(q["question"], hits, k=final_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # A document counts as retrieved at rank r if any of its chunks does.
        seen, ranked = set(), []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
        per_q[q["id"]] = m["hit_rate@5"]
        per_q_mrr[q["id"]] = m["mrr"]
        for key, val in m.items():
            agg[key].append(val)
            by_kind[q["kind"]][key].append(val)

    out = {k2: statistics.fmean(v) for k2, v in agg.items()}
    out["latency_p50_ms"] = statistics.median(latencies)
    out["latency_p95_ms"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    out["_by_kind"] = {kind: {k2: statistics.fmean(v) for k2, v in d.items()}
                       for kind, d in by_kind.items()}
    out["_per_question"] = per_q            # hit_rate@5 -- saturated, see kind_table
    out["_per_question_mrr"] = per_q_mrr    # use this one for Part B
    out["_kind_n"] = {kind: len(d["mrr"]) for kind, d in by_kind.items()}
    return out


def table(rows: dict[str, dict], cols: tuple[str, ...] =
          ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10",
           "latency_p95_ms")) -> str:
    name_w = max(len(n) for n in rows) + 2
    head = f"{'config':<{name_w}}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<{name_w}}" + "".join(f"{m.get(c, 0):>15.4f}" for c in cols))
    return "\n".join(lines)


def kind_table(metrics: dict, col: str = "hit_rate@5") -> str:
    """Break a result down by question kind.

    NOTE the default column. `hit_rate@5` is saturated on this corpus -- every
    retriever scores 0.93-0.98 -- so this table will look flat and tell you
    nothing. Pass col='mrr' or col='ndcg@10' for Part B. The default is left
    saturated on purpose.
    """
    bk, counts = metrics["_by_kind"], metrics.get("_kind_n", {})
    w = max(len(k) for k in bk) + 2
    lines = [f"{'kind':<{w}}{col:>12}{'n':>6}", "-" * (w + 18)]
    for kind, m in sorted(bk.items()):
        lines.append(f"{kind:<{w}}{m.get(col, 0):>12.4f}{counts.get(kind, 0):>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sweeps (yours)
# ---------------------------------------------------------------------------
def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")


def sweep_chunking() -> None:
    """A1: all four strategies at size=800.
    A2: the winner at sizes 400 / 800 / 1600.
    A3: markdown WITH and WITHOUT the '[heading > path]' prefix.
    """
    corpus = load_corpus()
    questions = load_questions()

    # --- A1: All four strategies at size 800 ---
    print("=== A1: Strategies at size 800 ===")
    strategies = ["fixed", "sliding", "recursive", "markdown"]
    a1_results = {}

    for strat in strategies:
        t0 = time.perf_counter()
        chunks = build_chunks(corpus, strategy=strat, size=800)
        retriever = DenseRetriever(chunks)
        build_time = time.perf_counter() - t0

        metrics = evaluate(retriever, questions)
        metrics["chunks"] = len(chunks)
        metrics["build_s"] = build_time
        a1_results[f"{strat}-800"] = metrics

    cols = ("hit_rate@1", "recall@5", "mrr", "ndcg@10", "chunks", "build_s")
    print(table(a1_results, cols=cols))
    print()

    # --- A2: Winner across sizes {400, 800, 1600} ---
    print("=== A2: Size sweep for winner (markdown) ===")
    sizes = [400, 800, 1600]
    a2_results = {}

    for sz in sizes:
        t0 = time.perf_counter()
        chunks = build_chunks(corpus, strategy="markdown", size=sz)
        retriever = DenseRetriever(chunks)
        build_time = time.perf_counter() - t0

        metrics = evaluate(retriever, questions)
        metrics["chunks"] = len(chunks)
        metrics["build_s"] = build_time
        a2_results[f"markdown-{sz}"] = metrics

    print(table(a2_results, cols=cols))
    print()

# --- A3: Heading-path prefix ablation ---
    print("=== A3: Markdown heading prefix ablation ===")
    
    # 1. With prefix (default markdown_chunks behavior)
    chunks_with = build_chunks(corpus, strategy="markdown", size=800)
    retriever_with = DenseRetriever(chunks_with)
    m_with = evaluate(retriever_with, questions)

    # 2. Without prefix: strip the leading '[heading > path]' prefix
    chunks_without = []
    for c in chunks_with:
        text = c.text
        if text.startswith("[") and "]" in text:
            text = text.split("]", 1)[1].lstrip()
        chunks_without.append(
            Chunk(
                chunk_id=c.chunk_id,
                text=text,
                doc_id=c.doc_id,
                meta=c.meta,
            )
        )

    retriever_without = DenseRetriever(chunks_without)
    m_without = evaluate(retriever_without, questions)

    a3_results = {
        "markdown-800 WITH prefix": m_with,
        "markdown-800 WITHOUT prefix": m_without,
    }
    
    a3_cols = ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10")
    print(table(a3_results, cols=a3_cols))

    # --- A3: Heading-path prefix ablation ---
    # print("=== A3: Markdown heading prefix ablation ===")
    
    # # 1. With prefix (default markdown_chunks behavior)
    # chunks_with = build_chunks(corpus, strategy="markdown", size=800)
    # retriever_with = DenseRetriever(chunks_with)
    # m_with = evaluate(retriever_with, questions)

    # # 2. Without prefix: strip the leading '[heading > path]' prefix
    # # Do NOT modify aip/chunking.py; use a list comprehension on the chunks.
    # chunks_without = []
    # for c in chunks_with:
    #     text = c.text
    #     if text.startswith("[") and "]" in text:
    #         text = text.split("]", 1)[1].lstrip()
    #     chunks_without.append(Chunk(text=text, doc_id=c.doc_id, meta=c.meta))

    # retriever_without = DenseRetriever(chunks_without)
    # m_without = evaluate(retriever_without, questions)

    # a3_results = {
    #     "markdown-800 WITH prefix": m_with,
    #     "markdown-800 WITHOUT prefix": m_without,
    # }
    
    # a3_cols = ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10")
    # print(table(a3_results, cols=a3_cols))


def sweep_retrieval() -> None:
    """B1-B4: retriever type, RRF k, and fusion weights."""
    corpus = load_corpus()
    questions = load_questions()
    chunks = build_chunks(corpus, "markdown", 800)

    dense = DenseRetriever(chunks)
    bm25 = Bm25Retriever(chunks)
    hybrid = HybridRetriever([dense, bm25])

    # B1: dense / bm25 / hybrid
    print("=== B1: Retriever comparison (mrr) ===")
    results = {}
    for name, r in [("dense", dense), ("bm25", bm25), ("hybrid", hybrid)]:
        results[name] = evaluate(r, questions)
    print(table(results))
    print()

    # B2: kind breakdown + Q44 and Q41
    print("=== B2: Per-kind MRR breakdown ===")
    for name, m in results.items():
        print(f"\n--- {name} ---")
        print(kind_table(m, col="mrr"))
        q44 = m["_per_question_mrr"].get("Q44", 0)
        q41 = m["_per_question_mrr"].get("Q41", 0)
        print(f"  Q44 mrr={q44:.4f}  Q41 mrr={q41:.4f}")

    # B3: RRF k sweep
    print("\n=== B3: RRF k sweep ===")
    rrf_results = {}
    for k_val in [10, 30, 60, 100]:
        h = HybridRetriever([dense, bm25], rrf_k=k_val)
        rrf_results[f"rrf_k={k_val}"] = evaluate(h, questions)
    print(table(rrf_results))

    # B4: unequal fusion weights
    print("\n=== B4: Fusion weights ===")
    wt_results = {}
    for wts, label in [([1.0, 1.0], "1:1"), ([2.0, 1.0], "2:1 dense"), ([1.0, 2.0], "1:2 bm25")]:
        h = HybridRetriever([dense, bm25], weights=wts)
        wt_results[label] = evaluate(h, questions)
    print(table(wt_results))


def sweep_rerank() -> None:
    """C1-C4: cross-encoder vs LLM reranker."""
    from aip.retrieval import CrossEncoderReranker, LLMReranker

    corpus = load_corpus()
    questions = load_questions()
    chunks = build_chunks(corpus, "markdown", 800)
    dense = DenseRetriever(chunks)

    # Baseline without reranking
    m_base = evaluate(dense, questions, k=30, final_k=5)

    # C1: CrossEncoderReranker
    print("=== C1: CrossEncoderReranker ===")
    ce = CrossEncoderReranker()
    m_ce = evaluate(dense, questions, k=30, reranker=ce, final_k=5)

    # C2: LLMReranker
    print("=== C2: LLMReranker ===")
    llm_rr = LLMReranker()
    m_llm = evaluate(dense, questions, k=30, reranker=llm_rr, final_k=5)

    results = {"no-rerank": m_base, "cross-encoder": m_ce, "llm-reranker": m_llm}
    cols = ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10", "latency_p95_ms")
    print(table(results, cols=cols))

    # C3: decision table
    print("\n=== C3: Decision table ===")
    print(f"{'':20s} {'cross-encoder':>15} {'llm-reranker':>15}")
    print(f"{'mrr':20s} {m_ce['mrr']:>15.4f} {m_llm['mrr']:>15.4f}")
    print(f"{'p95 latency ms':20s} {m_ce['latency_p95_ms']:>15.1f} {m_llm['latency_p95_ms']:>15.1f}")
    print("Interactive search: cross-encoder (fast, good quality)")
    print("Overnight batch:    llm-reranker  (higher quality, cost OK at scale)")

    # C4: query reranking made worse
    print("\n=== C4: Queries where reranking hurt ===")
    for qid in sorted(m_base["_per_question_mrr"]):
        before = m_base["_per_question_mrr"][qid]
        after = m_ce["_per_question_mrr"][qid]
        if after < before:
            print(f"  {qid}: mrr {before:.4f} -> {after:.4f} (cross-encoder hurt)")


def sweep_index() -> None:
    """D1-D3: ChromaRetriever vs DenseRetriever, metadata filtering."""
    from aip.retrieval import ChromaRetriever

    corpus = load_corpus()
    questions = load_questions()
    chunks = build_chunks(corpus, "markdown", 800)

    # D1/D2: DenseRetriever vs ChromaRetriever
    print("=== D1/D2: Dense vs Chroma ===")
    dense = DenseRetriever(chunks)
    chroma = ChromaRetriever(chunks, collection="lab3_sweep", reset=True)

    m_dense = evaluate(dense, questions)
    m_chroma = evaluate(chroma, questions)
    results = {"dense (exact)": m_dense, "chroma (HNSW)": m_chroma}
    cols = ("hit_rate@1", "recall@5", "mrr", "ndcg@10", "latency_p95_ms")
    print(table(results, cols=cols))

    # D3: metadata filtering
    print("\n=== D3: Metadata filtering (status=current) ===")
    for c in chunks:
        c.meta["status"] = "archived" if "ARCHIVED" in c.doc_id else "current"
    chroma_meta = ChromaRetriever(chunks, collection="lab3_meta", reset=True)

    # Evaluate Q29, Q30, Q31 before and after filtering
    target_qs = [q for q in questions if q["id"] in ("Q29", "Q30", "Q31")]
    print(f"\nQ29/Q30/Q31 hit_rate@1:")
    for label, where in [("unfiltered", None), ("status=current", {"status": "current"})]:
        hits = {}
        for q in target_qs:
            h = chroma_meta.search(q["question"], k=10, where=where)
            seen = []
            for hit in h:
                if hit.doc_id not in seen:
                    seen.append(hit.doc_id)
            m = retrieval_metrics(seen, q["relevant_docs"], ks=(1,))
            hits[q["id"]] = m["hit_rate@1"]
        print(f"  {label:20s}  Q29={hits.get('Q29',0):.0f}  Q30={hits.get('Q30',0):.0f}  Q31={hits.get('Q31',0):.0f}")


SWEEPS = {
    "chunking": sweep_chunking,
    "retrieval": sweep_retrieval,
    "rerank": sweep_rerank,
    "index": sweep_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    args = ap.parse_args()
    if args.baseline or not args.sweep:
        sweep_baseline()
    if args.sweep:
        SWEEPS[args.sweep]()


if __name__ == "__main__":
    main()
