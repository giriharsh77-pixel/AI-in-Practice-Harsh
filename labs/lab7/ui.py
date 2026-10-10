#!/usr/bin/env python3
"""Lab 7 — Streamlit front end.

    streamlit run labs/lab7/ui.py

Requires the service to be running:
    uvicorn labs.lab7.service:app --port 8000

The one non-negotiable UI requirement: **citations must be expandable to show
the source text.** Grounding the user cannot check is decoration.
"""
from __future__ import annotations

import json

import requests
import streamlit as st

API = st.sidebar.text_input("Service URL", "http://localhost:8000")
mode = st.sidebar.radio("Mode", ["rag", "tools"])
stream = st.sidebar.checkbox("Stream the answer (rag mode)")

st.title("Aurora Policy Assistant")
st.caption("Answers come only from Aurora's policy documents. "
           "Every claim is cited. When the documents do not cover a question, "
           "the assistant says so instead of guessing.")

q = st.text_input("Ask a question",
                  placeholder="How long do I have to file a reimbursement claim?")

if st.button("Ask", type="primary") and q:
    box = st.empty()                      # the answer area; streaming writes into it as tokens arrive
    with st.spinner("thinking"):
        try:
            if stream and mode == "rag":
                # B2: server-sent events. Tokens arrive as "token" events; citations and
                # the validation verdict arrive once, in the final "done" event (B3).
                data, text, event = {}, "", None
                with requests.post(f"{API}/ask/stream", json={"question": q},
                                   stream=True, timeout=90) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:") and event == "token":
                            text += json.loads(line[5:])["text"]
                            box.markdown(text)
                        elif line.startswith("data:"):
                            data = json.loads(line[5:])
                if event == "error":
                    st.error("The answer was interrupted. Please retry.")
                    st.caption(f"trace: `{data.get('trace_id')}`")
                    st.stop()
                data["answer"] = data.get("replace") or text   # B3: a retraction replaces what was shown
            else:
                r = requests.post(f"{API}/ask", json={"question": q, "mode": mode}, timeout=90)
                r.raise_for_status()
                data = r.json()
        except requests.HTTPError as exc:
            code = exc.response.status_code
            st.error({422: "Please ask a question between 3 and 1000 characters.",
                      429: "The assistant has reached its spending limit. Please try again later.",
                      503: f"The language model is unavailable. Please retry in "
                           f"{exc.response.headers.get('Retry-After', 'a few')} seconds.",
                      }.get(code, "Something went wrong."))
            detail = exc.response.json().get("detail")
            if isinstance(detail, dict):
                st.caption(f"HTTP {code} · trace: `{detail.get('trace_id')}`")
            st.stop()
        except requests.RequestException as exc:
            st.error(f"service unreachable: {exc}")
            st.stop()

    if data.get("refused"):
        box.warning(data["answer"])
    else:
        box.markdown(data["answer"])

    for c in data.get("citations", []):
        with st.expander(f"[{c['index']}] {c['doc_id']}"):
            st.markdown(c["excerpt"])
    if mode == "tools":
        st.caption("Tools mode answers from tool calls, so it has no citations.")

    cols = st.columns(4)
    if data.get("ttft_ms"):
        cols[0].metric("first token", f"{data['ttft_ms']:.0f} ms",
                       f"total {data['latency_ms']:.0f} ms", delta_color="off")
    else:
        cols[0].metric("latency", f"{data.get('latency_ms', 0):.0f} ms")
    cols[1].metric("cost", f"${data.get('cost_usd', 0):.5f}")
    cols[2].metric("cached", "yes" if data.get("cached") else "no")
    cited = {c["doc_id"] for c in data.get("citations", [])}
    cols[3].metric("cited", f"{len(cited)} of {len(data.get('sources', []))} docs")
    uncited = [s for s in data.get("sources", []) if s not in cited]
    if uncited:
        st.caption("Also retrieved, not cited: " + ", ".join(uncited))
    st.caption(f"trace: `{data.get('trace_id', '')}`")

# TODO stretch: a thumbs-down button that appends the case to a review queue.
# That queue is how real golden sets get built.