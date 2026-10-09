# Lab 3 — Semantic Search That Actually Works — Report

**Corpus:** 30 docs (16 Aurora Health policy + 14 distractors) · **n = 42** golden
questions (3 of the 45 have no relevant document and are excluded from
retrieval metrics — they belong to Lab 4's refusal measurement instead).

---

## Part A — Chunking

**A1/A2 — strategy and size sweep**

| config | hit_rate@1 | hit_rate@5 | recall@5 | mrr | ndcg@10 |
|---|---|---|---|---|---|
| fixed-800 | 0.7381 | 0.9524 | 0.8373 | 0.8387 | 0.7952 |
| sliding-800 | 0.7857 | 0.9286 | 0.8452 | 0.8451 | 0.8053 |
| recursive-800 | 0.7619 | 0.9524 | 0.8750 | 0.8611 | 0.8251 |
| markdown-800 | 0.7619 | 0.9762 | 0.8988 | 0.8720 | 0.8458 |
| markdown-400 | **0.7857** | 0.9762 | **0.9028** | **0.8800** | **0.8527** |
| markdown-1600 | 0.7143 | 0.9524 | 0.8750 | 0.8262 | 0.8075 |

Markdown-aware chunking beats fixed-800 by 0.05 nDCG@10, confirming the
heading-path prefix carries real signal. The size curve is **non-monotonic**:
400 > 800 > 1600. This is the dilution effect — at 1600 characters a chunk
routinely spans several sub-topics, so its embedding becomes an average that
matches none of them precisely. Smaller chunks stay topically pure and rank
better, at the cost of more chunks to index.

**A3 — heading-path prefix ablation** (markdown-400)

| config | hit_rate@1 | hit_rate@5 | recall@5 | mrr | ndcg@10 |
|---|---|---|---|---|---|
| with prefix | **0.7857** | 0.9762 | 0.9028 | **0.8800** | **0.8527** |
| without prefix | 0.6905 | 1.0000 | 0.9107 | 0.8131 | 0.8204 |

The prefix's value is concentrated in **ranking, not recall** — hit_rate@5 is
actually marginally lower with the prefix (0.976 vs 1.000), but hit_rate@1
jumps +0.096 and MRR +0.067. Without the heading context, a correct chunk
often still lands in the top 5 but rarely at rank 1; the prefix gives the
embedding enough specificity to win the top spot outright.

**A4 — chunking failure** *(fill in your own example: pick a question with
`mrr == 0.0` on markdown-400 and paste the retrieved-vs-correct chunk here —
see `inspect_failure()`)*

**Winner: markdown-400, with heading-path prefix.**

---

## Part B — Dense vs BM25 vs Hybrid

**B1 — overall**

| config | hit_rate@1 | hit_rate@5 | recall@5 | mrr | ndcg@10 | p95 latency |
|---|---|---|---|---|---|---|
| dense | **0.7857** | 0.9762 | **0.9028** | **0.8800** | **0.8527** | 4.95 ms |
| bm25 | 0.4762 | 0.9286 | 0.7956 | 0.6698 | 0.6978 | 3.02 ms |
| hybrid | 0.6667 | 0.9762 | 0.8631 | 0.7976 | 0.7949 | 8.10 ms |

**Hybrid loses to plain dense** (nDCG@10 0.7949 vs 0.8527) and is ~1.6×
slower, since it runs both retrievers per query. This is a corpus-specific
finding, not a general rule: BM25 alone (0.6978) is substantially weaker than
dense (0.8527), so Reciprocal Rank Fusion is combining a strong retriever
with a much weaker one, and drags good rankings down more often than it
rescues bad ones.

**B2 — per-kind (MRR, not hit_rate@5 — hit_rate@5 is saturated at 0.93–0.98
across every retriever and hides the effect entirely)**

| kind | dense | bm25 | hybrid |
|---|---|---|---|
| aggregation | 0.8750 | 0.3750 | 0.5833 |
| multi_hop | 1.0000 | 0.6500 | 0.8167 |
| paraphrase | 0.8000 | 0.4867 | 0.6500 |
| single_hop | 0.9074 | 0.8519 | 0.9444 |
| trap_archived | 0.8333 | 0.5111 | 0.6667 |
| unanswerable | 0.3125 | 0.4167 | 0.3750 |

**Mechanism, Q44** (exact identifier `AUR-HI-SIL-2026`): dense mrr=0.50,
bm25 mrr=1.00, hybrid mrr=1.00. An alphanumeric code carries no distinct
semantic meaning, so dense finds it but doesn't rank it first; BM25's exact
string match wins outright, and its strong rank survives fusion.

**Mechanism, Q41** ("if I skip paying on time, how long before I lose
everything I've built up?" — zero lexical overlap with "grace period"):
dense mrr=1.00, bm25 mrr=0.00, hybrid mrr=0.25. BM25 misses completely
(no shared vocabulary), and RRF cannot rescue a document one retriever ranks
nowhere — fusion actively **degrades** dense's otherwise-perfect result
rather than merely failing to improve it.

**B3 — RRF k sweep**

| k | ndcg@10 |
|---|---|
| 10 | 0.8156 |
| 30 | 0.7949 |
| 60 | 0.7949 |
| 100 | 0.7901 |

Small effect, declining as k grows. At small k, `1/(k+rank)` is more
sensitive to rank differences, so dense's confident top ranks dominate the
fusion; at large k, scores compress and BM25's noisier rankings get more
equal say — consistent with the strong/weak imbalance diagnosed in B2.

**B4 — weighted fusion**

| config | hit_rate@1 | ndcg@10 |
|---|---|---|
| hybrid (equal) | 0.6667 | 0.7949 |
| hybrid, weight 2:1 toward dense | 0.6905 | 0.8068 |

Weighting toward the stronger retriever partially recovers the loss,
confirming the diagnosis is imbalance, not a flaw in RRF itself.

---

## Part C — Reranking

| config | hit_rate@1 | ndcg@10 | p95 latency | cost |
|---|---|---|---|---|
| dense, no rerank | **0.7857** | 0.8527 | 11.2 ms | $0 |
| + cross-encoder | 0.7619 | 0.8174 | 905 ms | $0 |
| + LLM rerank | **0.8333** | **0.8577** | 31,134 ms | small, cached |

The cross-encoder (`ms-marco-MiniLM`) **hurts** quality — it was trained on
web search click data, not insurance policy prose, and is out of domain here.
The LLM reranker **helps** — best hit_rate@1 and nDCG@10 of all three — but
at ~2,800× the latency of no reranking.

**C4 — a query the cross-encoder made worse (Q41):** dense correctly ranked
the grace-period passage first (score 0.673). After cross-encoder reranking,
that passage dropped to rank 4, and completely unrelated chunks (refund
policy, pre-existing-condition waiting periods) were promoted to the top.
The reranker appears to match on generic "policy admin" surface language
rather than the specific concept asked about — the same paraphrase weakness
B2 already exposed in BM25, now showing up in a different retrieval
component for a different reason.

**Two deployment answers (C3):**
- **Interactive search box:** plain dense, no rerank. It is both the fastest
  option and the best-quality option among the three here — no trade-off to
  make.
- **Overnight batch job:** the LLM reranker. Latency is irrelevant offline,
  and it delivers the best absolute quality; cost is a few cents for the
  golden set, an acceptable trade for a batch context.
- The cross-encoder is dominated in this evaluation: slower than no-rerank
  and lower-quality than either alternative.

---

## Part D — Index and metadata

**D1/D2 — exact vs ANN, at ~150 chunks**

| config | ndcg@10 | p95 latency |
|---|---|---|
| dense (exact) | 0.8527 | 6.56 ms |
| chroma (HNSW) | 0.8527 | 9.04 ms |

Identical quality; exact search is faster at this scale, since HNSW's
approximation overhead isn't repaid until the index is much larger.

**D3 — the archived-document trap (Q29–Q31)**

| question | before filter | after filter (`status: current`) |
|---|---|---|
| Q29 | True | True |
| Q30 | **False** | **True** |
| Q31 | True | True |
| **overall hit_rate@1** | **0.667** | **1.000** |

Filtering on `status` metadata fixes Q30 outright: without it, the retriever
surfaces the superseded `claims-timelines-2024-ARCHIVED.md` doc, which
states stale figures (e.g. 48-hour notification, 30-day settlement) instead
of the current ones (72 hours, 45 days). A one-line metadata filter removes
the trap entirely.

---

## Final recommended configuration

**Markdown chunking (size 400, with heading-path prefix) + plain dense
retrieval (no hybrid, no rerank) + `status: current` metadata filter.**

- nDCG@10: 0.8527 (target ≥ 0.80, met)
- hit_rate@1: 0.7857
- p95 latency: ~7–11 ms
- Cost: $0 per query (embeddings only, cached)
- Hybrid fusion and the local cross-encoder are both dominated on this
  corpus; reserve LLM reranking for offline/batch use only.

---

## One thing that surprised me

Hybrid fusion didn't just fail to beat its component retrievers on Q41 — it
scored *worse than the average of dense (1.0) and BM25 (0.0) would suggest*,
landing at 0.25. A retriever that totally misses a query can actively pull a
otherwise-perfect result down after fusion, rather than simply being ignored.
Reciprocal Rank Fusion has no mechanism to detect "this retriever found
nothing relevant" and discount it — it only sees ranks, not confidence.
