#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings  # noqa: E402

BASELINE_REFUSAL = 0.20
ALERT_WINDOW = 20                 # judge the last 20 answers ...
ALERT_MIN = 10                    # ... and never fewer than 10, so 1 refusal in 2 cannot fire it
STAGES = ["embed", "search", "generate", "other"]

st.set_page_config(page_title="Aurora Assistant — Ops", layout="wide")
st.title("Aurora Policy Assistant — operations")

runs = sorted(settings.trace_dir.glob("*.jsonl"), reverse=True)
if not runs:
    st.info(f"No traces yet in {settings.trace_dir}. Run some queries first.")
    st.stop()

# Every server restart starts a new run file, so default to all of today's runs.
today = time.strftime("%Y%m%d")
chosen = st.sidebar.multiselect("runs", [p.stem for p in runs],
                                default=[p.stem for p in runs if p.stem.startswith(today)] or [runs[0].stem])
spans = [json.loads(line) for p in runs if p.stem in chosen
         for line in p.open(encoding="utf-8") if line.strip()]
children = defaultdict(list)
for s in spans:
    children[s.get("parent_id")].append(s)


def kids(span, name=None):
    return [c for c in children[span["span_id"]] if name is None or c["name"] == name]


rows = []
for r in spans:
    if r["name"] not in ("http.ask", "http.ask_stream"):
        continue
    row = {"ts": r["ts"], "trace_id": f"{r['run_id']}:{r['span_id']}", "question": r.get("question"),
           "mode": r.get("mode", "rag"), "status": r.get("status"),
           "error": (r.get("error") or "").split(":")[0] or None,
           "latency_ms": r.get("request_latency_ms", r["duration_ms"]),
           "cost_usd": r.get("request_cost_usd", 0.0), "cached": bool(r.get("cached")),
           "cache_layer": r.get("cache_layer"), "refused": bool(r.get("refused"))}
    llm = [c for c in kids(r, "llm.call") if not c.get("cached")]
    if llm and row["mode"] == "rag":
        dense = kids(r, "retrieve.dense")
        inner = sum(e["duration_ms"] for d in dense for e in kids(d, "embed.batch"))
        row["embed"] = sum(e["duration_ms"] for e in kids(r, "embed.batch")) + inner
        row["search"] = sum(d["duration_ms"] for d in dense) - inner
        row["generate"] = sum(c["duration_ms"] for c in llm)
        row["other"] = row["latency_ms"] - row["embed"] - row["search"] - row["generate"]
    rows.append(row)

if not rows:
    st.info("No requests in the selected runs.")
    st.stop()
req = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
req["ts"] = pd.to_datetime(req["ts"], unit="s")
ok = req[req["status"] == "ok"]

c = st.columns(5)
c[0].metric("requests", len(req))
c[1].metric("total cost", f"${ok['cost_usd'].sum():.4f}")
c[2].metric("cache hit rate", f"{ok['cached'].mean():.0%}")
c[3].metric("error rate", f"{(req['status'] == 'error').mean():.0%}")
c[4].metric("refusal rate", f"{ok['refused'].mean():.0%}")

# --- C4: the alert ------------------------------------------------------------
# A broken index does not raise errors; it quietly turns answers into refusals.
recent = ok.tail(ALERT_WINDOW)["refused"]
rate = recent.mean() if len(recent) else 0.0
if len(recent) >= ALERT_MIN and rate >= 2 * BASELINE_REFUSAL:
    st.error(f"ALERT: {rate:.0%} of the last {len(recent)} answers were refusals "
             f"(baseline {BASELINE_REFUSAL:.0%}). This usually means retrieval broke, not that users changed. "
             "1) /health: did index_chunks change?  2) traces: any guard.layer2_dropped events?  "
             "3) diff the corpus against the last good version and roll back the changed document.")
else:
    st.success(f"Refusal alert quiet: {rate:.0%} of the last {len(recent)} answers were refusals "
               f"(fires at {2 * BASELINE_REFUSAL:.0%} over at least {ALERT_MIN}).")

# --- C3: latency by stage over time --------------------------------------------
st.subheader("Latency by stage, uncached RAG requests (ms)")
full = ok.dropna(subset=["generate"]) if "generate" in ok else ok.iloc[0:0]
if len(full):
    st.bar_chart(full[STAGES].reset_index(drop=True))     # one stacked bar per request, in order
    st.dataframe(full[STAGES + ["latency_ms"]].quantile([0.5, 0.95]).round(0)
                 .rename(index={0.5: "p50", 0.95: "p95"}))

left, right = st.columns(2)
with left:
    st.subheader("Cumulative cost (USD)")
    st.line_chart(ok.set_index("ts")["cost_usd"].cumsum())
with right:
    st.subheader("Cache hit and error rate (last 10 requests)")
    st.line_chart(pd.DataFrame({
        "cache hit rate": req["cached"].astype(float).rolling(10, min_periods=1).mean(),
        "error rate": (req["status"] == "error").astype(float).rolling(10, min_periods=1).mean(),
    }).set_index(req["ts"]))

st.subheader("Errors")
errs = req[req["status"] == "error"]
st.dataframe(errs[["ts", "error", "question", "trace_id"]] if len(errs) else pd.DataFrame(),
             hide_index=True)

# --- C1: why did request X take 9 seconds? ----------------------------------------
st.subheader("Look up a request")
st.caption("Slowest requests in the selection:")
st.dataframe(ok.nlargest(5, "latency_ms")[["trace_id", "latency_ms", "cached", "refused", "question"]],
             hide_index=True)
tid = st.text_input("trace_id (shown under every answer in the UI)").strip()
if tid:
    root = next((s for s in spans if f"{s['run_id']}:{s['span_id']}" == tid), None)
    if root is None:
        st.warning("That trace_id is not in the selected runs.")
    else:
        st.json({k: v for k, v in root.items() if k not in ("run_id", "span_id", "parent_id")})
        tree, todo = [], [(root, 0)]
        while todo:                                          
            s, depth = todo.pop()
            tree.append({"span": "· " * depth + s["name"], "ms": s.get("duration_ms"),
                         **{k: s.get(k) for k in ("model", "cached", "n_uncached", "prompt_tokens",
                                                  "completion_tokens", "cost_usd", "finish_reason",
                                                  "status")}})
            todo += [(k, depth + 1) for k in sorted(kids(s), key=lambda k: k["ts"], reverse=True)]
        st.dataframe(pd.DataFrame(tree), hide_index=True)