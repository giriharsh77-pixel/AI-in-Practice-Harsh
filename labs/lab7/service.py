#!/usr/bin/env python3
"""Lab 7 — the service.

    uvicorn labs.lab7.service:app --reload --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq
"""
from __future__ import annotations

import sys
import time
import re, threading
import json, uuid
from collections import Counter
import numpy as np

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, cost, tracing  # noqa: E402
from aip.config import resolve_model, settings  # noqa: E402
from aip.guards import ToolGuard, delimit_untrusted, detect_injection  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context # noqa: E402
from aip.cost import Budget, BudgetExceeded, global_budget  # noqa: E402
from aip.embed import embed  # noqa: E402
from aip.llm import _is_retryable  # noqa: E402 

REQUEST_BUDGET_USD = 0.05
RETRY_AFTER_S = 30
READ_ONLY_TOOLS = {"search_policy", "get_policy_details", "compute_premium"}
_CITE = re.compile(r"\[(\d+)\]")


_PIPELINE = None
_BUILD_LOCK = threading.Lock()
_STARTED = time.time()
_RESPONSE_CACHE: dict[tuple, dict] = {} 
_SEMANTIC: list[tuple] = []  
SEMANTIC_THRESHOLD = 0.92

class InjectionFilteredRetriever(Retriever):
    """Layer 2: drop retrieved chunks that look like injected instructions.

    Wraps the real retriever, so labs/lab4/rag.py is unchanged. It scans
    retrieved text only, never the user's own question (Lab 6 D3).
    """

    name = "dense+layer2"

    def __init__(self, inner: Retriever):
        self.inner = inner
        self.chunks = getattr(inner, "chunks", [])

    def search(self, query: str, k: int = 8) -> list[Hit]:
        hits = self.inner.search(query, k=k)
        kept = [h for h in hits if not detect_injection(h.text).flagged]
        if len(kept) < len(hits):
            tracing.event("guard.layer2_dropped", n_dropped=len(hits) - len(kept),
                          dropped=[h.doc_id for h in hits if h not in kept])
        return kept

@dataclass
class Pipeline:
    retriever: Retriever        # Layer 2 wrapped, for RAG mode
    base_retriever: Retriever   # plain index, shared with the agent
    n_chunks: int
    model: str
    built_s: float

def pipeline() -> Pipeline:
    """Build the v3 pipeline once and cache it."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    with _BUILD_LOCK:                       # two first requests must not build twice
        if _PIPELINE is None:
            from labs.lab4.evaluate import build_retriever
            from labs.lab6 import agent

            t0 = time.perf_counter()
            base = build_retriever()        # v3: markdown-400, 214 chunks
            agent._RETRIEVER = base         # share the index with the Lab 6 agent
            _PIPELINE = Pipeline(
                retriever=InjectionFilteredRetriever(base),
                base_retriever=base,
                n_chunks=len(base.chunks),
                model=resolve_model("MAIN"),
                built_s=round(time.perf_counter() - t0, 2),
            )
    return _PIPELINE


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pipeline()          # build before the first request, not during it
    import litellm  # noqa: F401
    yield


app = FastAPI(title="Aurora Policy Assistant", version="1.0", lifespan=lifespan)

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    sources: list[str]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str

def _ask_rag(req: AskRequest, span: dict) -> dict:
    from labs.lab4.rag import ANSWER_SYSTEM, REFUSAL, answer_question
    from labs.lab6.agent import filter_output

    p = pipeline()
    a = answer_question(req.question, p.retriever,
                        k=max(12, req.top_k), final_k=req.top_k)

    # Layer 5: check what is about to leave.
    text, filtered = filter_output(a.text, ANSWER_SYSTEM.replace(REFUSAL, ""))

    cited = sorted({int(n) for n in _CITE.findall(text)})
    citations = [] if "prompt_leak" in filtered else [
        Citation(index=n, doc_id=a.hits[n - 1].doc_id, excerpt=a.hits[n - 1].text.strip())
        for n in cited if 1 <= n <= len(a.hits)
    ]
    span.update(refused=a.refused, n_citations=len(citations),
                n_sources=len(a.hits), filtered=filtered)
    return {"answer": text, "refused": a.refused, "citations": citations,
            "sources": list(dict.fromkeys(h.doc_id for h in a.hits))}


def _ask_tools(req: AskRequest, span: dict) -> dict:
    from labs.lab6.agent import run_agent

    pipeline()                              # make sure the agent shares the index
    guard = ToolGuard(max_calls=6, allow=READ_ONLY_TOOLS,
                      requires_confirmation={"issue_refund"},
                      confirm_fn=lambda name, args: False)   # refunds never auto-confirmed
    r = run_agent(req.question, guard=guard, layers={1, 2, 3, 4, 5})

    if r["stopped_because"] == "budget":
        # run_agent swallows BudgetExceeded; surface it so both modes return 429.
        raise BudgetExceeded("tool loop stopped at its spend limit")

    answer = r["answer"]
    refused = r["stopped_because"] != "answered" or not answer.strip()
    if refused:
        answer = ("I couldn't complete this request within my limits. "
                  "Please rephrase it or contact Aurora support.")
    span.update(refused=refused, stopped_because=r["stopped_because"],
                tool_calls=[e["tool"] for e in r["tool_log"]],
                tools_denied=sum(1 for e in r["tool_log"] if not e.get("ok")),
                filtered=r.get("filtered", []))
    return {"answer": answer, "refused": refused, "citations": [], "sources": []}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    trace_id = None
    try:
        with tracing.trace("http.ask", question=req.question[:120], mode=req.mode) as span:
            trace_id = f"{tracing.RUN_ID}:{span['span_id']}"
            key = (" ".join(req.question.lower().split()).rstrip("?.! "), req.top_k)
            layer = "exact" if req.mode == "rag" and key in _RESPONSE_CACHE else None
            with Budget(limit_usd=REQUEST_BUDGET_USD, label="http.ask") as b:
                if req.mode == "rag" and layer is None:
                    qv = embed(req.question)        # the retriever reuses this vector from aip's cache
                    best, best_key = max(((float(qv @ v), k) for v, k in _SEMANTIC if k[1] == req.top_k),
                                         default=(0.0, None))
                    span.update(semantic_best=round(best, 3))
                    if best >= SEMANTIC_THRESHOLD:
                        layer, key = "semantic", best_key
                if layer:
                    body = _RESPONSE_CACHE[key]
                else:
                    body = (_ask_tools if req.mode == "tools" else _ask_rag)(req, span)
                    if req.mode == "rag":
                        _RESPONSE_CACHE[key] = body
                        _SEMANTIC.append((qv, key))
            cost = round(b.spent_usd, 6)
            if layer is None and cost == 0:
                layer = "aip"                       # pipeline ran, model call replayed from aip's cache
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            span.update(request_cost_usd=cost, request_latency_ms=latency_ms,
                        cached=layer is not None, cache_layer=layer)
            return AskResponse(**body, latency_ms=latency_ms, cost_usd=cost,
                               cached=layer is not None, trace_id=trace_id)
    except BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail={
            "error": "budget_exhausted", "message": str(exc), "trace_id": trace_id}) from exc
    except cache.CacheMiss as exc:
        raise HTTPException(status_code=503, headers={"Retry-After": str(RETRY_AFTER_S)}, detail={
            "error": "model_unavailable_offline", "trace_id": trace_id}) from exc
    except Exception as exc:                                    # noqa: BLE001
        if _is_retryable(exc):
            raise HTTPException(status_code=503, headers={"Retry-After": str(RETRY_AFTER_S)}, detail={
                "error": "upstream_unavailable",
                "message": "The language model provider is unavailable. Please retry shortly.",
                "trace_id": trace_id}) from exc
        raise HTTPException(status_code=500, detail={
            "error": "internal_error",
            "message": "Unexpected error. Quote the trace_id when reporting it.",
            "trace_id": trace_id}) from exc

@app.get("/health")
def health() -> dict:
    p = pipeline()
    return {"status": "ok", "uptime_s": round(time.time() - _STARTED, 1),
            "index_chunks": p.n_chunks, "model": p.model,
            "index_build_s": p.built_s, "cache": cache.stats()}


@app.get("/metrics")
def metrics() -> dict:
    midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    reqs = [r for f in settings.trace_dir.glob("*.jsonl") if f.stat().st_mtime >= midnight
            for r in map(json.loads, f.open(encoding="utf-8"))
            if r.get("name") in ("http.ask", "http.ask_stream") and r["ts"] >= midnight]
    ok = [r for r in reqs if r.get("status") == "ok"]
    cached = [r for r in ok if r.get("cached")]
    uncached = [r for r in ok if not r.get("cached")]
    errors = Counter(r.get("error", "").split(":")[0] for r in reqs if r.get("status") == "error")
    cost = sum(r.get("request_cost_usd", 0) for r in ok)

    def pcts(rs):
        xs = [r["request_latency_ms"] for r in rs if r.get("request_latency_ms") is not None]
        return {f"p{p}": round(float(np.percentile(xs, p)), 1) for p in (50, 95, 99)} if xs else {}

    return {
        "requests_today": len(reqs),
        "cost_today_usd": round(cost, 6),
        "cost_per_query_usd": round(cost / len(ok), 6) if ok else 0.0,
        "cost_per_uncached_query_usd": round(cost / len(uncached), 6) if uncached else 0.0,
        "cache_hit_rate": round(len(cached) / len(ok), 3) if ok else 0.0,
        "cache_hits_by_layer": dict(Counter(r.get("cache_layer") for r in cached)),
        "latency_ms": {"all": pcts(ok), "uncached": pcts(uncached), "cached": pcts(cached)},
        "error_rate": round(sum(errors.values()) / len(reqs), 3) if reqs else 0.0,
        "errors_by_type": dict(errors),        # BudgetExceeded -> 429, CacheMiss/provider errors -> 503
        "refusal_rate": round(sum(1 for r in ok if r.get("refused")) / len(ok), 3) if ok else 0.0,
        "tool_calls": dict(Counter(t for r in ok for t in r.get("tool_calls", []))),
        "process_budget": global_budget().as_dict(),   # this process only; the budget behind the 429s
    }

@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    from litellm import completion, stream_chunk_builder
    from labs.lab4.rag import ANSWER_SYSTEM, REFUSAL, validate_answer
    from labs.lab6.agent import filter_output

    t0 = time.perf_counter()
    rid = uuid.uuid4().hex[:12]
    try:    # everything before the first byte, so an outage is still a proper 503
        p = pipeline()
        # The same prompt answer_question() builds: layer-2 retrieval, delimited sources, same system prompt.
        hits = p.retriever.search(req.question, k=max(12, req.top_k))[:req.top_k]
        context = format_context(hits, max_chars=8000)
        n_sources = len(re.findall(r"^\[\d+\] \(source:", context, flags=re.M))
        hits = hits[:n_sources]
        messages = [{"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user", "content": f"{delimit_untrusted(context)}\n\n"
                                                f"Question: {req.question}\n\nAnswer with citations:"}]
        retrieve_ms = round((time.perf_counter() - t0) * 1000, 1)
        stream = completion(model=p.model, messages=messages, temperature=0.0, max_tokens=1500,
                            stream=True, timeout=settings.timeout_s)
    except Exception as exc:  # noqa: BLE001
        if _is_retryable(exc):
            raise HTTPException(status_code=503, headers={"Retry-After": str(RETRY_AFTER_S)},
                                detail={"error": "upstream_unavailable", "trace_id": rid}) from exc
        raise HTTPException(status_code=500, detail={"error": "internal_error", "trace_id": rid}) from exc

    def events():
        chunks, text, ttft_ms = [], "", None
        try:
            for ch in stream:
                chunks.append(ch)
                delta = ch.choices[0].delta.content if ch.choices else None
                if delta:
                    ttft_ms = ttft_ms or round((time.perf_counter() - t0) * 1000, 1)
                    text += delta
                    yield {"event": "token", "data": json.dumps({"text": delta})}
        except Exception as exc:  # noqa: BLE001  -- the 200 is already sent; an error can only be an event now
            yield {"event": "error", "data": json.dumps(
                {"error": "upstream_unavailable" if _is_retryable(exc) else "internal_error", "trace_id": rid})}
            return
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        full = stream_chunk_builder(chunks, messages=messages)
        pt = getattr(full.usage, "prompt_tokens", 0) or 0
        ct = getattr(full.usage, "completion_tokens", 0) or 0
        usd = cost.price_of(p.model, pt, ct)
        try:
            cost.record(cost.Usage(p.model, pt, ct, usd, latency_ms, cached=False, calls=1,
                                   priced=cost.is_priced(p.model)))
        except BudgetExceeded:
            pass    # already spent; the next request is refused with 429

        check = validate_answer(text, n_sources, full.choices[0].finish_reason)
        final, filtered = filter_output(text.strip(), ANSWER_SYSTEM.replace(REFUSAL, ""))
        if not check["valid"]:
            final = REFUSAL
        retracted = final != text.strip()
        cited = sorted({int(n) for n in _CITE.findall(final)})
        citations = [] if "prompt_leak" in filtered else [
            Citation(index=n, doc_id=hits[n - 1].doc_id, excerpt=hits[n - 1].text.strip()).model_dump()
            for n in cited if 1 <= n <= len(hits)]
        refused = check["refused"] or not check["valid"]

        tracing.event("http.ask_stream", span_id=rid, question=req.question[:120],
                      retrieve_ms=retrieve_ms, ttft_ms=ttft_ms, request_latency_ms=latency_ms,
                      request_cost_usd=round(usd, 6), completion_tokens=ct, refused=refused,
                      validation=check["reason"], filtered=filtered, retracted=retracted)
        yield {"event": "done", "data": json.dumps({
            "replace": final if retracted else None, "refused": refused, "citations": citations,
            "sources": list(dict.fromkeys(h.doc_id for h in hits)), "ttft_ms": ttft_ms,
            "latency_ms": latency_ms, "cost_usd": round(usd, 6), "cached": False,
            "trace_id": f"{tracing.RUN_ID}:{rid}"})}

    return EventSourceResponse(events())