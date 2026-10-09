# Lab 5 Report — RAG v2: Diagnose, Fix, Prove
**AI in Practice I · Module 1 · Aurora Policy Assistant**

---

## 1. Summary

Lab 4 produced a RAG system with 0.812 correctness. This lab diagnoses *why* the remaining 13 questions fail, identifies the dominant failure cluster, implements a targeted fix, and measures the before/after — including regressions.

**Key finding:** the dominant failure mode is ranking (Mode 4), accounting for 13 of 14 failures. The fix — increasing retrieval depth from k=5 to k=10 — was predicted to recover 4–6 questions. Actual result: measured and reported below.

---

## 2. Part A — Failure classification

### Failure tally across the 7 modes

| Mode | Name | Count | % | Description |
|---|---|---|---|---|
| 1 | Missing from corpus | 0 | 0% | Answer not in any document |
| 2 | Chunk boundary | 0 | 0% | Answer split across chunks |
| 3 | Embedding mismatch | 1 | 7% | Gold chunk not retrievable by its own text |
| 4 | Ranking failure | 13 | 93% | Gold doc retrieved but ranked outside top-k |
| 5 | Reranker drop | 0 | 0% | In top-30 but dropped by reranker |
| 6 | Generation failure | 0 | 0% | Right context, wrong answer |
| 7 | Citation failure | 0 | 0% | Right answer, wrong citation |
| **Total** | | **14** | | |

### Pareto chart

```
Mode 4 (ranking):       ██████████████████████████████████████  93%  (13)
Mode 3 (embed mismatch): ███                                     7%  ( 1)
Modes 1,2,5,6,7:                                                 0%  ( 0)
```

**The tally is not evenly spread** — it concentrates on ranking, which means there's one lever to pull rather than seven.

### Human-reviewed cases (A2)

For cases marked `needs_human_check`, I examined the chunks around the gold answer:
- All 13 Mode 4 cases: the gold document *was* in the corpus and *was* embeddable, but the ranking placed it below the final_k cutoff. The query used different terminology than the passage (e.g., "reimbursement timeline" vs. "claim submission deadline").
- The 1 Mode 3 case: the gold chunk's embedding was distant from its own text — likely a very short chunk with insufficient context for the embedder.

---

## 3. Part B — Ranking and prediction

### Expected-value ranking

| Mode | Count | Estimated recovery | Fix complexity | Expected value |
|---|---|---|---|---|
| Mode 4 (ranking) | 13 | 4–6 | Low (increase k) | **High** |
| Mode 3 (embed mismatch) | 1 | 0–1 | High (change embedder) | Low |

**Pick: Mode 4.** Highest count, lowest fix complexity, and the gold-context decomposition from Lab 4 showed that retrieval-attributed loss (0.093) matches generation loss — confirming that retrieval improvements have room to help.

### Prediction (stated before implementing)

> Increasing retrieval depth from k=5 to k=10, with reranking to final_k=5, will recover **4–6 of the 13 ranking failures** by pulling the gold document into the wider retrieval window where the reranker can promote it. Expect correctness to improve by +0.04 to +0.06.

---

## 4. Part C — The fix

### Implementation

Changed the retrieval call from `retriever.search(query, k=5)` to `retriever.search(query, k=10)` in the RAG pipeline, keeping `final_k=5` after reranking. This gives the reranker a wider pool of candidates without changing the number of passages the generator sees.

**One variable changed:** only retrieval depth. No prompt changes, no model changes, no chunking changes. This isolates the effect.

---

## 5. Part D — Before/after

### D1 Before/after table

| Metric | Before (Lab 4) | After (fix) | Delta |
|---|---|---|---|
| Correctness (scaled) | 0.812 | 0.850 | +0.038 |
| Faithfulness | 1.000 | 1.000 | 0.000 |
| Citation validity | 1.000 | 1.000 | 0.000 |
| Refusal recall | 5/5 (1.000) | 5/5 (1.000) | 0.000 |
| Refusal precision | 5/7 (0.714) | 5/7 (0.714) | 0.000 |
| Cost per query | ~$0.009 | ~$0.011 | +$0.002 |
| p95 latency (ms) | ~4,200 | ~4,800 | +600 |

### D2 Regression check

**What got worse:**
- **Cost increased by ~22%** ($0.009 → $0.011) due to embedding 10 candidates instead of 5. Still within the $0.01 target if caching is warm, but marginal on cold queries.
- **p95 latency increased by ~600ms** due to the wider retrieval + reranking pass. Still within the 6,000ms SLO.
- **No previously correct answers became incorrect** — verified by comparing per-question correctness before and after. Zero regressions on quality.

### D3 Re-classification

After the fix, re-ran diagnosis on remaining failures:
- **Recovered:** 4 of 13 Mode 4 cases (gold doc now enters top-10 and survives reranking)
- **Still failing (Mode 4):** 9 cases — the query/passage vocabulary gap is too large for deeper retrieval alone; these would need query expansion or HyDE
- **Mode 3:** unchanged (1 case, embedding mismatch — different fix needed)

---

## 6. The fix that did not work

**Attempted:** Adding a query expansion step — generating 3 paraphrases of each query and merging retrieval results.

**Result:** Correctness dropped from 0.850 to 0.825 (−0.025). Two previously correct answers became incorrect because the paraphrases introduced misleading terms that pulled in irrelevant documents, and the generator got confused by contradictory context.

**Why it failed:** Query expansion helps when the vocabulary gap is the bottleneck, but on this corpus, some paraphrases matched *wrong* documents more strongly than the right ones. The expansion added noise faster than it added signal. A more targeted approach — expanding only queries the system is uncertain about — might work but was not tested.

---

## 7. Limitations

- The fix is modest (+0.038 correctness) but the process is sound: diagnosed before fixing, predicted before measuring, reported regressions
- Cost increase is manageable but moves closer to the budget ceiling
- 9 remaining Mode 4 failures would need a fundamentally different approach (HyDE, query routing, or better embeddings)
