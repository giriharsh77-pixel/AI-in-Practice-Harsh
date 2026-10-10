#!/usr/bin/env python3
"""Lab 7 B4 — where does the time go? Latency by stage, read back from .aip_traces.

    python labs/lab7/scratch3.py

A full-pipeline RAG request (an http.ask span with at least one paid llm.call
under it) is split into:
    query embed   embed.batch spans anywhere under the request
    search        retrieve.dense minus the embedding inside it
    generate      the llm.call spans (more than one = a repair retry)
    other         everything else: Layer 2/5 guards, validation, FastAPI
Cached requests and streamed requests are reported separately.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aip.config import settings  # noqa: E402

spans = [json.loads(line) for f in sorted(settings.trace_dir.glob("*.jsonl"))
         for line in f.open(encoding="utf-8") if line.strip()]
children = defaultdict(list)
for s in spans:
    children[s.get("parent_id")].append(s)


def kids(span, name):
    return [c for c in children[span["span_id"]] if c["name"] == name]


def pct(xs, p):
    return np.percentile(xs, p) if xs else float("nan")


stages = defaultdict(list)          # full-pipeline requests: stage -> [ms per request]
cached = defaultdict(list)          # served without generating: layer -> [ms]
for r in spans:
    if r["name"] != "http.ask" or r.get("mode") != "rag" or r.get("status") != "ok":
        continue
    llm = [c for c in kids(r, "llm.call") if not c.get("cached")]
    if not llm:
        cached[r.get("cache_layer") or "aip"].append(r["duration_ms"])
        continue
    dense = kids(r, "retrieve.dense")
    embeds = kids(r, "embed.batch") + [e for d in dense for e in kids(d, "embed.batch")]
    embed_ms = sum(e["duration_ms"] for e in embeds)
    search_ms = sum(d["duration_ms"] for d in dense) - sum(e["duration_ms"] for d in dense
                                                            for e in kids(d, "embed.batch"))
    gen_ms = sum(c["duration_ms"] for c in llm)
    stages["query embed"].append(embed_ms)
    stages["search"].append(search_ms)
    stages["generate (LLM)"].append(gen_ms)
    stages["other (guards, validation)"].append(r["duration_ms"] - embed_ms - search_ms - gen_ms)
    stages["total"].append(r["duration_ms"])
    stages["llm calls"].append(len(llm))
    stages["paid embeds"].append(sum(1 for e in embeds if e.get("n_uncached")))

n = len(stages["total"])
print(f"\nFULL-PIPELINE RAG REQUESTS (n={n})")
print(f"  {'stage':<28}{'p50 ms':>9}{'p95 ms':>9}{'share':>8}")
for name in ("query embed", "search", "generate (LLM)", "other (guards, validation)", "total"):
    xs = stages[name]
    share = sum(xs) / sum(stages["total"]) if n else float("nan")
    print(f"  {name:<28}{pct(xs, 50):>9.0f}{pct(xs, 95):>9.0f}{share:>8.0%}")
if n:
    print(f"  requests with a repair retry: {sum(c > 1 for c in stages['llm calls'])}/{n}"
          f"   with a paid query embedding: {sum(e > 0 for e in stages['paid embeds'])}/{n}")

print("\nSERVED WITHOUT GENERATING")
print(f"  {'layer':<28}{'n':>5}{'p50 ms':>9}{'p95 ms':>9}")
for layer, xs in sorted(cached.items()):
    print(f"  {layer:<28}{len(xs):>5}{pct(xs, 50):>9.0f}{pct(xs, 95):>9.0f}")

st = [s for s in spans if s["name"] == "http.ask_stream"]
if st:
    print(f"\nSTREAMED (n={len(st)})")
    for key in ("retrieve_ms", "ttft_ms", "request_latency_ms"):
        xs = [s[key] for s in st if s.get(key) is not None]
        print(f"  {key:<28}p50 {pct(xs, 50):>7.0f}   p95 {pct(xs, 95):>7.0f}")
    ratio = [s["ttft_ms"] / s["request_latency_ms"] for s in st if s.get("ttft_ms")]
    print(f"  first token as share of total: {np.mean(ratio):.0%}")

print("\nNote: with few requests, p95 is close to the maximum.")