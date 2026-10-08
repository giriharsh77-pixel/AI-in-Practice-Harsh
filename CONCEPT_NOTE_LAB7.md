# Concept Note — Lab 7: Ship It
## Aurora Policy Assistant: From Components to Production Service

**Author:** Harsh Giri
**Date:** October 2026
**Module:** AI in Practice — Module 1 Capstone

---

## 1. Problem Statement

Labs 1–6 produced a set of working components: a retrieval pipeline, a RAG
generator with citation enforcement, a failure diagnostic framework, and a
tool-using agent with security guards. None of these are usable by anyone
outside a Python notebook. Lab 7 bridges this gap — wrapping the entire
pipeline into a production-grade HTTP service with caching, streaming,
observability, a regression gate, and a two-page evaluation report that
defends its behaviour with data.

## 2. System Architecture

The Aurora Policy Assistant is a FastAPI service that answers questions about
insurance policy documents using retrieval-augmented generation.

### Architecture Layers

```
┌─────────────────────────────────────────────────┐
│  Streamlit UI (ui.py)                           │
├─────────────────────────────────────────────────┤
│  HTTP Layer — FastAPI (service.py)              │
│  POST /ask  ·  POST /ask/stream  ·  GET /health│
│  GET /metrics                                   │
├─────────────────────────────────────────────────┤
│  Error Handling                                 │
│  422 validation · 429 budget/rate · 503 outage  │
├─────────────────────────────────────────────────┤
│  Cache Layer (aip.cache)                        │
│  Exact response cache (hash of normalised req)  │
├─────────────────────────────────────────────────┤
│  RAG Pipeline (Lab 4)                           │
│  Retrieve → Generate → Validate → Repair        │
├─────────────────────────────────────────────────┤
│  Retrieval (Lab 3)                              │
│  DenseRetriever · markdown chunks · 800 chars   │
├─────────────────────────────────────────────────┤
│  Security Guards (Lab 6)                        │
│  ToolGuard: allowlist, call budget, schema check│
├─────────────────────────────────────────────────┤
│  Observability (aip.tracing + aip.cost)         │
│  JSONL traces · cost accounting · budget ceiling│
├─────────────────────────────────────────────────┤
│  Regression Gate (gate.py + thresholds.yml)     │
│  CI offline replay · golden set · exit non-zero │
└─────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Pipeline built once at startup** — not per request. Building the
   DenseRetriever embeds the entire corpus (~2,000 chunks). Doing this
   per request would add ~30 seconds to every call. The pipeline is
   initialised lazily on the first request and cached in a module-level
   global.

2. **Guards wired at service level** — Lab 6's ToolGuard is instantiated
   alongside the pipeline with an explicit allowlist (`search_policy`,
   `get_policy_details`, `compute_premium`) and confirmation required for
   `issue_refund`. A service without guards is not shippable.

3. **Generate-then-stream for SSE** — The streaming endpoint generates the
   full answer first, validates citations, then streams word-by-word. This
   trades slightly higher time-to-first-token for guaranteed citation
   validity. The alternative (stream raw, then correct) risks the user
   reading unvalidated text before a correction event arrives.

## 3. API Contract

| Endpoint          | Method | Purpose                              |
|-------------------|--------|--------------------------------------|
| `/ask`            | POST   | Grounded answer with citations       |
| `/ask/stream`     | POST   | Same, with server-sent events        |
| `/health`         | GET    | Index size, model profile, uptime    |
| `/metrics`        | GET    | Cost, latency percentiles, hit rate  |

### POST /ask — Request/Response

**Request:** `{question, top_k?, mode?}` where mode is `rag` (default) or
`tools`.

**Response:** `{answer, refused, citations[], latency_ms, cost_usd, cached,
trace_id}` — cost and trace ID are in the response body deliberately. Cost
is how the person paying for it sees what a query costs; trace ID is how
anyone debugging finds the trace.

### Error Semantics

| Code | Meaning              | When                           |
|------|----------------------|--------------------------------|
| 422  | Validation error     | Malformed request (FastAPI)    |
| 429  | Too many requests    | Budget exceeded or rate limited |
| 503  | Service unavailable  | Upstream model outage          |

503 includes a `Retry-After` header. Returning 500 for a rate limit makes
every client retry immediately — exactly wrong.

## 4. Evaluation Results

Evaluated on 45 golden-set questions (40 answerable, 5 unanswerable):

| Metric              | Value | Gate Threshold | Status |
|----------------------|-------|----------------|--------|
| Citation validity    | 1.000 | ≥ 0.98         | PASS   |
| Faithfulness         | 1.000 | ≥ 0.90         | PASS   |
| Correctness          | 0.812 | ≥ 0.75         | PASS   |
| Refusal recall       | 1.000 | ≥ 0.80         | PASS   |
| Refusal precision    | 0.714 | ≥ 0.70         | PASS   |
| Hit rate@5           | 0.952 | ≥ 0.85         | PASS   |
| Cost per query       | ~$0.003 | ≤ $0.010     | PASS   |
| p95 latency          | ~4,000 ms | ≤ 6,000 ms | PASS   |

### Failure Analysis

14 of 45 questions scored below full correctness:
- **13 ranking failures** (92.9%) — gold document exists in the top 30 but is
  buried below `final_k=5`. Concentrated in multi-hop and aggregation questions.
- **1 embedding mismatch** (7.1%) — a paraphrase the embedding model does not
  recognise as semantically equivalent.
- **2 false refusals** — Q23 (multi-hop about adding a parent to a policy) and
  Q44 (paraphrase), reducing refusal precision to 0.714.

## 5. Observability

Every request writes a structured JSONL trace via `aip.tracing`. Each span
records name, duration, cost, and parent span ID. This answers operational
questions like "why did request X take 9 seconds?" by breaking the request
into retrieval, generation, and validation stages.

The Streamlit dashboard (`dashboard.py`) reads these traces and displays:
- Latency by stage (p50/p95 per span name)
- Cumulative cost over time
- Error log
- **Alert condition:** p95 latency of recent queries exceeding the 6,000 ms
  SLO, with a red banner when triggered. Action: check provider status and
  retrieval index health.

## 6. Regression Gate

`gate.py` runs the full golden set, computes all eight metrics, and exits
non-zero if any threshold is breached. It runs with `AIP_OFFLINE=1` against
the committed cache, so CI needs no API key and costs nothing.

The gate is demonstrated failing by deliberately breaking the pipeline
(e.g., setting `final_k=1`), which causes correctness and hit_rate to drop
below threshold.

**A gate you have not seen fail is a gate you do not have.**

## 7. Cost Analysis

| Scope                | Cost      |
|----------------------|-----------|
| Per query (uncached) | ~$0.003   |
| Per 1,000 queries    | ~$3.00    |
| Per year at 10k/day  | ~$10,950  |

Cost is dominated by the MAIN-tier generation call. Cache hits are free.
On the Gemini free tier, actual dollar cost is $0 up to rate limits.

## 8. Limitations

1. **Not safe for coverage decisions without human review** — 13 ranking
   failures mean the system can miss relevant policy terms. A customer
   relying solely on this system could get an incomplete answer.
2. **No staleness detection** — answers from a static corpus snapshot.
   Policy amendments after the last index build are invisible.
3. **Numerical claims unverified in RAG mode** — the agent mode uses
   `compute_premium` for arithmetic, but RAG mode does not verify numbers.
4. **Guard coverage at 71%** — 29% of red-team attacks bypassed the guards.

## 9. What We Would Do Next

1. **Cross-encoder reranker** (+5–8% correctness, ~$0.001/query additional) —
   addresses 13 of 14 failures directly.
2. **Semantic cache with measured threshold** (30–50% latency reduction) —
   must sweep to find the threshold where wrong hits begin (~0.92).
3. **Feedback-driven golden set expansion** — thumbs-down button feeds a
   review queue; confirmed failures become new golden-set entries.
