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
    """TODO A1-A3.

    A1: all four strategies at size=800.
    
    A2: the winner at sizes 400 / 800 / 1600. Plot or tabulate the curve.
    A3: markdown WITH and WITHOUT the '[heading > path]' prefix.
        (Strip it with a list comprehension over the chunks -- do not modify
         aip/chunking.py; other labs depend on it.)

    Report chunk count and index build time alongside quality. A configuration
    that is 1 point better and takes 4x as long to build is a real trade-off.
    """

    corpus, questions = load_corpus(), load_questions()
    results = {}

    for strategy in STRATEGIES:
        chunks = build_chunks(corpus, strategy, 800)
        r = DenseRetriever(chunks)
        m = evaluate(r, questions)
        m["chunk_count"] = len(chunks)
        results[f"{strategy}-800"] = m
    print(table(results))
    for name, m in results.items():
        if "_chunk_count" in m:
            print(f"{name}: {m['_chunk_count']} chunks")


    winner = "markdown"  
    for size in (400, 800, 1600):
        chunks = build_chunks(corpus, winner, size)
        r = DenseRetriever(chunks)
        m = evaluate(r, questions)
        results[f"{winner}-{size}"] = m
    print(table({k: v for k, v in results.items() if winner in k}))


    # A3: prefix ablation
    with_prefix = build_chunks(corpus, "markdown", 400)
    without_prefix = [
        Chunk(c.text.split("]\n", 1)[-1] if c.text.startswith("[") else c.text,
              c.doc_id, c.chunk_id, c.meta)
        for c in with_prefix
    ]
    r_with = DenseRetriever(with_prefix)
    r_without = DenseRetriever(without_prefix)
    m_with = evaluate(r_with, questions)
    m_without = evaluate(r_without, questions)
    print(table({"markdown-400-with-prefix": m_with,
                 "markdown-400-without-prefix": m_without}))


    #raise NotImplementedError


def sweep_retrieval() -> None:
    """TODO B1-B4.

    B1: dense / bm25 / hybrid on your best chunking.
    B2: print kind_table(m, col='mrr') for each, and pull out Q44 and Q41
        individually from metrics['_per_question_mrr'].

        USE MRR, NOT hit_rate@5. Every retriever here scores 0.93-0.98 on
        hit_rate@5, so it is saturated and shows you nothing -- which is why
        kind_table() and metrics['_per_question'] both default to it. That
        default is the trap, and noticing it is part of the lab.

    B3: RRF k in {10, 30, 60, 100} -- HybridRetriever(..., rrf_k=k).
    B4: unequal fusion weights -- HybridRetriever(..., weights=[2.0, 1.0]).
    """
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "markdown", 400)  # your Part A winner

    dense = DenseRetriever(chunks)
    bm25 = Bm25Retriever(chunks)
    hybrid = HybridRetriever([dense, bm25])

    results = {
        "dense": evaluate(dense, questions),
        "bm25": evaluate(bm25, questions),
        "hybrid": evaluate(hybrid, questions),
    }
    print(table(results))
    print()

    # B2: per-kind breakdown, using MRR not hit_rate@5 (hit_rate@5 is saturated)
    for name, m in results.items():
        print(f"--- {name} ---")
        print(kind_table(m, col="mrr"))
        print()

    # B2 cont'd: pull out Q44 (exact identifier -> BM25's turf)
    # and Q41 (paraphrase, zero lexical overlap -> dense's turf) individually
    for name, m in results.items():
        q44 = m["_per_question_mrr"].get("Q44")
        q41 = m["_per_question_mrr"].get("Q41")
        print(f"{name}: Q44 mrr={q44}  Q41 mrr={q41}")


    # B3: RRF k sweep
    for k in (10, 30, 60, 100):
        h = HybridRetriever([dense, bm25], rrf_k=k)
        m = evaluate(h, questions)
        results[f"hybrid-k{k}"] = m
    print(table({k2: v for k2, v in results.items() if k2.startswith("hybrid-k")}))

    # B4: unequal weights, favoring the stronger retriever (dense)
    h_weighted = HybridRetriever([dense, bm25], weights=[2.0, 1.0])
    results["hybrid-weighted-2-1"] = evaluate(h_weighted, questions)
    print(table({"hybrid": results["hybrid"],
                 "hybrid-weighted-2-1": results["hybrid-weighted-2-1"]}))
    #raise NotImplementedError


def sweep_rerank() -> None:
    """TODO C1-C4.

    Retrieve k=30, rerank to 5: evaluate(r, questions, k=30, reranker=rr,
    final_k=5).

    C1: CrossEncoderReranker. First run downloads ~90 MB.
    C2: LLMReranker -- report cost as well as latency.
    C3: the decision table, and TWO different deployment answers
        (interactive search box vs overnight batch). They should differ.
    C4: find a query reranking made worse, using
        metrics['_per_question_mrr'] before and after.
    """
    
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "markdown", 400)
    dense = DenseRetriever(chunks)

    from aip.retrieval import CrossEncoderReranker, LLMReranker

    # C1: cross-encoder. First call downloads ~90MB, one-time.
    ce = CrossEncoderReranker()
    m_ce = evaluate(dense, questions, k=30, reranker=ce, final_k=5)

    # C2: LLM reranker — costs money per query, unlike C1
    llm_rr = LLMReranker()
    m_llm = evaluate(dense, questions, k=30, reranker=llm_rr, final_k=5)

    m_base = evaluate(dense, questions)  # no rerank, k=10 default — your baseline

    # find a question the cross-encoder made worse
    print("\nquestions where cross-encoder hurt MRR:")
    for qid, base_mrr in m_base["_per_question_mrr"].items():
        ce_mrr = m_ce["_per_question_mrr"].get(qid)
        if ce_mrr is not None and ce_mrr < base_mrr:
            print(f"  {qid}: {base_mrr} -> {ce_mrr}")

    print(table({"dense-no-rerank": m_base,
                 "dense+cross-encoder": m_ce,
                 "dense+llm-rerank": m_llm}))


    
    #raise NotImplementedError


def inspect_rerank_failure(question_id: str) -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "markdown", 400)
    dense = DenseRetriever(chunks)
    from aip.retrieval import CrossEncoderReranker
    ce = CrossEncoderReranker()

    q = next(x for x in questions if x["id"] == question_id)
    before = dense.search(q["question"], k=5)
    after = ce.rerank(q["question"], dense.search(q["question"], k=30), k=5)

    print("Q:", q["question"])
    print("gold:", q["relevant_docs"])
    print("\nBEFORE (dense only):")
    for h in before:
        print(f"  [{h.doc_id}] score={h.score:.3f}  {h.text[:100]}")
    print("\nAFTER (cross-encoder reranked):")
    for h in after:
        print(f"  [{h.doc_id}] score={h.score:.3f}  {h.text[:100]}")


def sweep_index() -> None:
    """TODO D1-D3.

    D1/D2: ChromaRetriever vs DenseRetriever -- recall gap and latency.
    D3: pass status metadata into the chunks and filter at query time.

        Set chunk.meta['status'] = 'archived' if 'ARCHIVED' in doc_id else 'current'
        then ChromaRetriever.search(..., where={"status": "current"}).

        Report hit_rate@1 on Q29/Q30/Q31 before and after (hit_rate@1, not
        @5 -- @5 is saturated here and will hide the whole effect).
    """
    
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "markdown", 400)  # your Part A winner

    from aip.retrieval import ChromaRetriever

    # D1/D2: exact (DenseRetriever) vs ANN (ChromaRetriever)
    dense = DenseRetriever(chunks)
    # reset=True is important: ChromaRetriever persists to disk at .chroma/
    # by default. Without reset=True, a stale index from an earlier run
    # (different chunking) silently gets reused -- same "cached index" trap
    # as Part A's checkpoint warning.
    chroma = ChromaRetriever(chunks, path=".chroma", collection="lab3", reset=True)

    m_dense = evaluate(dense, questions)
    m_chroma = evaluate(chroma, questions)
    print(table({"dense-exact": m_dense, "chroma-hnsw": m_chroma}))
    print()
    print(f"dense p95: {m_dense['latency_p95_ms']:.2f} ms")
    print(f"chroma p95: {m_chroma['latency_p95_ms']:.2f} ms")
    print("-> at this scale (few hundred chunks), exact should be as good "
          "or faster than HNSW. Run scripts/expand_corpus.py --docs 4000 "
          "and re-test to find the crossover (see README D2).")
    print()

    # D3: the archived-doc trap. Q29/Q30/Q31 have a correct current answer
    # and a wrong answer sitting in an ARCHIVED doc that competes lexically.
    for c in chunks:
        c.meta["status"] = "archived" if "ARCHIVED" in c.doc_id else "current"

    chroma_meta = ChromaRetriever(chunks, path=".chroma", collection="lab3-meta",
                                  reset=True)
    trap_ids = {"Q29", "Q30", "Q31"}
    trap_qs = [q for q in questions if q["id"] in trap_ids]

    def hit_at_1(retriever, qs, where=None) -> dict[str, bool]:
        out = {}
        for q in qs:
            hits = retriever.search(q["question"], k=5, where=where) \
                   if where else retriever.search(q["question"], k=5)
            out[q["id"]] = bool(hits) and hits[0].doc_id in q["relevant_docs"]
        return out

    before = hit_at_1(chroma_meta, trap_qs)
    after = hit_at_1(chroma_meta, trap_qs, where={"status": "current"})

    print("D3 -- hit_rate@1 on the archived-doc trap questions:")
    print(f"{'':6}{'before filter':>16}{'after filter':>16}")
    for qid in sorted(trap_ids):
        print(f"{qid:6}{str(before.get(qid, '-')):>16}{str(after.get(qid, '-')):>16}")
    b_rate = sum(before.values()) / len(before)
    a_rate = sum(after.values()) / len(after)
    print(f"\noverall: {b_rate:.3f} -> {a_rate:.3f}")
    #raise NotImplementedError


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
