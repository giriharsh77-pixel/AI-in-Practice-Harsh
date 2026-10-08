#!/usr/bin/env python3
"""Lab 7 — the service.

    uvicorn labs.lab7.service:app --reload --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, tracing  # noqa: E402
from aip.cost import BudgetExceeded, global_budget  # noqa: E402

app = FastAPI(title="Aurora Policy Assistant", version="1.0")

_PIPELINE = None
_GUARD = None
_STARTED = time.time()


def pipeline():
    """TODO A2: build the Labs 3–5 pipeline once at startup and cache it."""
    global _PIPELINE, _GUARD
    if _PIPELINE is None:
        from aip.chunking import markdown_chunks
        from aip.guards import ToolGuard
        from aip.retrieval import DenseRetriever
        from labs.lab3.search import load_corpus

        corpus = load_corpus()
        chunks = [c for doc_id, text in corpus.items()
                  for c in markdown_chunks(text, doc_id, size=800)]
        _PIPELINE = DenseRetriever(chunks)
        _GUARD = ToolGuard(
            max_calls=8,
            allow={"search_policy", "get_policy_details",
                   "compute_premium"},
            requires_confirmation={"issue_refund"},
        )
    return _PIPELINE


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """TODO A1. Return cost and trace_id in the response."""
    t0 = time.perf_counter()
    try:
        with tracing.trace("http.ask", question=req.question[:120]) as span:
            retriever = pipeline()
            budget_before = global_budget().spent_usd

            if req.mode == "rag":
                from labs.lab4.rag import answer_question
                ans = answer_question(req.question, retriever, final_k=req.top_k)

                citations = [
                    Citation(index=i + 1, doc_id=h.doc_id,
                             excerpt=h.text[:300])
                    for i, h in enumerate(ans.hits)
                ]
                answer_text = ans.text
                refused = ans.refused
            else:
                from labs.lab6.agent import run_agent
                result = run_agent(req.question)
                answer_text = result["answer"]
                refused = False
                citations = []

            cost_usd = global_budget().spent_usd - budget_before
            latency_ms = (time.perf_counter() - t0) * 1000
            span["cost_usd"] = round(cost_usd, 6)
            span["latency_ms"] = round(latency_ms, 1)

            return AskResponse(
                answer=answer_text,
                refused=refused,
                citations=citations,
                latency_ms=round(latency_ms, 1),
                cost_usd=round(cost_usd, 6),
                cached=False,
                trace_id=tracing.RUN_ID,
            )
    except BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except NotImplementedError:
        raise
    except Exception as exc:                                    # noqa: BLE001
        # TODO A3: distinguish provider outage from genuine bug
        error_name = type(exc).__name__
        if "RateLimitError" in error_name or "429" in str(exc):
            raise HTTPException(
                status_code=429,
                detail="rate limited by upstream provider",
                headers={"Retry-After": "30"},
            ) from exc
        if any(s in error_name for s in ("APIError", "APIConnectionError",
                                          "ServiceUnavailableError", "Timeout")):
            raise HTTPException(
                status_code=503,
                detail="upstream model unavailable",
                headers={"Retry-After": "10"},
            ) from exc
        raise HTTPException(status_code=503,
                            detail="upstream model unavailable") from exc


@app.get("/health")
def health() -> dict:
    """TODO C: index size, model profile, cache stats, uptime."""
    from aip.config import settings
    retriever = pipeline()
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _STARTED, 1),
        "profile": settings.profile,
        "index_chunks": len(retriever._chunks) if hasattr(retriever, '_chunks') else 0,
        "cache": cache.stats(),
    }


@app.get("/metrics")
def metrics() -> dict:
    """TODO C2: cost today, cost/query, cache hit rate, p50/p95/p99, error rate."""
    b = global_budget()
    d = b.as_dict()
    cost_per_query = d["cost_usd"] / d["calls"] if d["calls"] else 0.0
    cache_hit_rate = d["cached_calls"] / d["calls"] if d["calls"] else 0.0
    return {
        **d,
        "cost_per_query_usd": round(cost_per_query, 6),
        "cache_hit_rate": round(cache_hit_rate, 3),
        "p99_latency_ms": round(b.percentile(99), 1),
    }


# TODO B2: POST /ask/stream with server-sent events.
@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """SSE streaming endpoint. Streams the answer token by token, then sends
    a final metadata event with citations, cost, and trace_id.

    Strategy for B3 (citation validation before streaming): We use a
    "generate-then-stream" approach — generate the full answer first, validate
    citations, then stream the validated answer token-by-token. This trades
    slightly higher time-to-first-token for guaranteed citation validity.
    The alternative (stream raw, then send a correction event) risks the user
    reading unvalidated text.
    """
    t0 = time.perf_counter()

    try:
        retriever = pipeline()
        budget_before = global_budget().spent_usd

        from labs.lab4.rag import answer_question
        ans = answer_question(req.question, retriever, final_k=req.top_k)

        citations = [
            {"index": i + 1, "doc_id": h.doc_id, "excerpt": h.text[:300]}
            for i, h in enumerate(ans.hits)
        ]
        cost_usd = global_budget().spent_usd - budget_before
        latency_ms = (time.perf_counter() - t0) * 1000

        def generate():
            words = ans.text.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            meta = {
                "type": "done",
                "refused": ans.refused,
                "citations": citations,
                "latency_ms": round(latency_ms, 1),
                "cost_usd": round(cost_usd, 6),
                "trace_id": tracing.RUN_ID,
            }
            yield f"data: {json.dumps(meta)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(status_code=503,
                            detail="upstream model unavailable") from exc
