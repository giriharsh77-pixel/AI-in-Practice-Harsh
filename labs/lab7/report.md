# Lab 7 Report — Ship It: From Components to Production Service
**AI in Practice I · Module 1 · Aurora Policy Assistant — Capstone**

---

## 1. Summary

Wrapped the Labs 3–6 pipeline into a production-grade FastAPI service with four endpoints, a Streamlit UI with expandable citations, an observability dashboard, and a regression gate. The system runs live, serves grounded answers with streaming, tracks cost and latency, and fails the gate on cue when thresholds are breached.

### Service endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/ask` | POST | Synchronous RAG answer with citations, cost, trace_id |
| `/ask/stream` | POST | SSE streaming — generate-then-stream for citation safety |
| `/health` | GET | Index size, model profile, cache stats, uptime |
| `/metrics` | GET | Cost today, cost/query, cache hit rate, p99 latency |

---

## 2. Architecture decisions

### A. Retrieval pipeline (wired in `pipeline()`)

- **Chunking:** `markdown_chunks(size=800)` — winner from Lab 3
- **Retriever:** `DenseRetriever` for in-memory exact cosine search
- **Guard:** `ToolGuard` from Lab 6 with allowlist, call budget (8), and `requires_confirmation` on `issue_refund`

### B. Streaming strategy (B3)

Chose **generate-then-stream** over raw streaming:
1. Generate the full answer and validate citations
2. Stream the validated answer word-by-word as SSE events
3. Send a final `done` event with metadata (citations, cost, trace_id)

**Trade-off:** slightly higher time-to-first-token, but citations are guaranteed valid before the user reads anything. The alternative (stream raw, send correction event) risks the user reading and acting on an unvalidated citation before the correction arrives.

### C. Error handling (A3)

Distinguishes three error classes:
- **Rate limiting** (429 from upstream) → HTTP 429 with `Retry-After: 30`
- **Provider outage** (APIError, Timeout, ServiceUnavailable) → HTTP 503 with `Retry-After: 10`
- **Budget exceeded** → HTTP 429 with budget details
- **Unexpected errors** → HTTP 503 (safe fallback, not a 500 with stack trace)

---

## 3. Observability

### Dashboard (`dashboard.py`)

Reads JSONL traces from `.aip_traces/` and displays:
- Span count, total cost, model calls, cache hit rate, error count
- Latency by stage (p50/p95 per span name)
- Cumulative cost over time
- Error log with timestamps

### Alert condition (C4)

SLO: p95 latency ≤ 6,000ms on recent 20 queries. When breached, the dashboard fires an alert recommending:
1. Check provider status (generation slowdown)
2. Check retrieval index health (corpus growth, index degradation)

### Metrics endpoint

```json
{
  "cost_usd": 0.0423,
  "calls": 5,
  "cost_per_query_usd": 0.00846,
  "cache_hit_rate": 0.400,
  "p99_latency_ms": 4891.2
}
```

---

## 4. Regression gate

### How it works (`gate.py`)

1. Runs all 45 golden-set questions through the full pipeline
2. Measures 8 metrics: correctness, faithfulness, citation validity, refusal recall/precision, hit_rate@5, cost per query, p95 latency
3. Compares each against thresholds in `thresholds.yml`
4. Exits non-zero if any threshold is breached

### Thresholds (`thresholds.yml`)

| Metric | Gate |
|---|---|
| Correctness | ≥ 0.75 |
| Faithfulness | ≥ 0.90 |
| Citation validity | ≥ 0.95 |
| Refusal recall | ≥ 0.80 |
| Refusal precision | ≥ 0.70 |
| Hit rate@5 | ≥ 0.80 |
| Cost per query | ≤ $0.015 |
| p95 latency | ≤ 6000ms |

### Demonstrating gate failure

To make the gate fail on cue, lower `refusal_precision` threshold to 0.90 (our measured 0.714 will breach it) or raise `correctness` threshold to 0.95. The gate prints a clear `GATE FAILED` with the breached metric and exits with code 1.

---

## 5. UI (`ui.py`)

### Features
- Text input for questions, with a primary "Ask" button
- Answer display with **expandable citations** — each citation is a Streamlit expander showing the source doc_id and the excerpt text. This is the non-negotiable requirement: grounding the user cannot check is decoration.
- Metrics bar: latency, cost, cached status, source count
- Trace ID for debugging
- "Flag for review" button that appends to a JSONL review queue for golden set curation

---

## 6. Live metrics

From a test run of 5 queries:

| Metric | Value |
|---|---|
| p95 latency | ~4,200 ms |
| Cost per query | ~$0.009 |
| Cache hit rate | 40% (warm cache) |
| Citation validity | 1.000 |
| Errors | 0 |

All within SLO targets.

---

## 7. The worst remaining failure

**Q37 (partially answerable):** The system refuses entirely when it should answer the part that's covered and refuse the rest. This is the same issue from Lab 4 — the refusal prompt is binary ("answer or refuse") when it should support partial coverage. A fix would add a third mode to the system prompt: "answer what you can, and state what you cannot."

**Next step:** Add partial-refusal support to `ANSWER_SYSTEM` and re-measure refusal precision (currently 0.714 — the two false-positive refusals are both partial-coverage questions).

---

## 8. Limitations

- No authentication or rate limiting — production would need API keys and per-user quotas
- The generate-then-stream approach adds ~500ms to TTFT vs raw streaming
- No horizontal scaling — single-process uvicorn with an in-memory index
- Dashboard reads local JSONL files — production would use a proper observability backend (Langfuse, Datadog)
- The regression gate takes ~3 minutes to run all 45 questions — too slow for a pre-commit hook, appropriate for CI
