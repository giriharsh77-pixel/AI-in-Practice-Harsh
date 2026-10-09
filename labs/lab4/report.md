# Lab 4 Report — RAG v1: Grounded Answers with Citations
**AI in Practice I · Module 1 · Aurora Policy Assistant**

---

## 1. Summary

Built a RAG pipeline that retrieves policy passages and generates grounded answers with enforced citations. Every claim carries a `[n]` citation index validated in code. When sources don't cover the question, the system refuses with an exact machine-detectable string rather than guessing.

### Headline metrics on the 45-question golden set

| Metric | Target | Measured |
|---|---|---|
| Citation validity | 1.00 | **1.000** |
| Faithfulness | ≥ 0.90 | **1.000** |
| Correctness (scaled 0–1) | ≥ 0.75 | **0.812** |
| Refusal recall | ≥ 4/5 | **5/5 (1.000)** |
| Refusal precision | ≥ 0.70 | **5/7 (0.714)** |
| Cost per query | ≤ $0.01 | ~$0.009 |

---

## 2. Part A — The generation prompt

### ANSWER_SYSTEM differences from reference

My prompt included all six required elements (T4 §6.1):
1. Answer only from numbered sources; no general knowledge
2. Cite by index — `[1]`, `[2][5]`
3. Never cite a number not supplied
4. Exact refusal string: `"I cannot answer this from the provided sources."`
5. When sources disagree, surface the disagreement explicitly
6. Length discipline: 2–3 sentences unless the question demands more

**Key difference from `aip/rag.py::ANSWER_SYSTEM`:** the reference prompt includes an explicit instruction about partial coverage ("if only part of the question is answerable, answer that part and refuse the rest"). My initial prompt missed this, which caused Q37 (partially answerable) to fully refuse instead of partially answering. Added after Part C testing.

---

## 3. Part B — Citation enforcement

### What happens on failure

Citation validation (`validate_answer`) checks three conditions:
1. Every `[n]` index is in range `[1, len(sources)]`
2. The answer is non-empty and not truncated (`finish_reason != "length"`)
3. Non-refusal answers contain at least one citation

**On failure:** the system retries with a corrective message appended: *"Your previous answer contained invalid citation [n]. Only sources [1] through [k] exist. Rewrite with valid citations only."* After 2 failed retries, it falls back to refusal — better to decline than to serve a hallucinated citation.

**Citation validity: 1.000** — the validate-and-repair loop catches and fixes every invalid citation before it reaches the user. The repair rate was ~8.9% (4 of 45 answers needed one retry).

---

## 4. Part C — Refusal

### Refusal at two strictness settings

| Setting | Refusal recall | Refusal precision | Total refusals |
|---|---|---|---|
| Strict (default) | **5/5 = 1.000** | **5/7 = 0.714** | 7 |
| Relaxed (partial answers allowed) | 4/5 = 0.800 | 4/5 = 0.800 | 5 |

**Both numbers are extremely noisy.** With only 5 unanswerable questions and 7 total refusals, moving one case shifts precision by ±0.12 and recall by ±0.20. These are directions, not measurements.

**The two false-positive refusals** (strict mode): Q37 (partially answerable — sources cover part of the question) and one edge case where the relevant passage used different terminology than the question. The strict setting is correct for production: a false refusal sends the user to a human agent, while a false answer sends them wrong information.

**Product recommendation:** Deploy with strict refusal. The 2 false positives route to a human, which costs ~$2 each. A false answer (missed refusal) costs a compliance incident. The asymmetry favours strict.

---

## 5. Part D — The judge

### Judge agreement (Cohen's κ)

| Rubric | κ | Interpretation |
|---|---|---|
| Faithfulness (binary: supported or not) | **0.72** | Substantial agreement |
| Correctness (3-point: wrong / partial / correct) | **0.58** | Moderate agreement |

Both exceed the κ ≥ 0.4 threshold. The faithfulness rubric is easier because it's binary and the evidence is right there in the context. Correctness is harder because partial credit requires judgement about "how much" of the gold answer was captured.

**Self-preference check:** ran 10 answers through a second judge call with swapped order. No systematic bias detected (5 agreed, 5 unchanged).

---

## 6. Part E — Decomposition and failure analysis

### E2 Decomposition table

The gold-context decomposition isolates retrieval from generation failures:

| Condition | Correctness | Interpretation |
|---|---|---|
| A: Retrieved context (full pipeline) | 0.812 | End-to-end quality |
| B: Gold context (perfect retrieval) | 0.905 | Generator ceiling |
| A − B (retrieval-attributed loss) | −0.093 | What retrieval costs us |
| 1.0 − B (generation-attributed loss) | −0.095 | What generation costs us |

**Conclusion:** retrieval and generation contribute roughly equally to the remaining errors. Retrieval-attributed loss (0.093) is slightly less than generation-attributed loss (0.095), meaning improving either stage has similar expected value.

### E3 Failure-mode tally (13 failures)

| Mode | Count | Description |
|---|---|---|
| Mode 4: Ranking failure | 8 | Gold doc retrieved but ranked too low |
| Mode 6: Generation failure | 3 | Right context provided, wrong answer generated |
| Mode 2: Chunk boundary | 1 | Gold answer split across two chunks |
| Mode 3: Embedding mismatch | 1 | Gold chunk not retrievable even by its own text |

**Dominant cluster:** Mode 4 (ranking) accounts for 62% of failures. This is the target for Lab 5.

---

## 7. Limitations

- Refusal metrics computed over only 5 unanswerable questions — too noisy for confident claims
- Judge correctness κ (0.58) is moderate; a human-validated subset would strengthen claims
- No streaming implemented (stretch goal)
- Cost per query is near the $0.01 ceiling; a larger corpus would exceed it without caching
