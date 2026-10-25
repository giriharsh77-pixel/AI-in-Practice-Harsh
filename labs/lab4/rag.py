#!/usr/bin/env python3
"""Lab 4 — your RAG pipeline.

Write this yourself. `aip/rag.py` is the reference implementation; look at it
after Part A, not before. Labs 5-7 build on whichever of the two you prefer,
but you must be able to explain every line of the one you use.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402
from aip.chunking import markdown_chunks  # noqa: E402

# The exact string the system must emit when it cannot answer. Exact, because
# downstream code detects refusal by matching it -- a paraphrase is a bug.
REFUSAL = "I don't have enough information in the provided sources to answer that."
_CITE = re.compile(r"\[(\d+)\]") 

# TODO A: write this before you read aip/rag.py::ANSWER_SYSTEM.
ANSWER_SYSTEM = f"""\
You are a helpdesk assistant for Aurora health insurance. Follow these rules,
in priority order:

1. If the sources do not answer the question, reply with exactly:
   {REFUSAL}
   If they answer only part of it, answer that part with citations, then end
   with that exact sentence for the rest.
2. Answer ONLY from the numbered sources provided. Do not use general knowledge
   or assumptions, and never infer numbers, limits or deadlines the sources do
   not state.
3. End every factual sentence with the source index that supports it, e.g. [1]
   or [2][5].
4. Only cite source numbers that were actually supplied. Never invent one.
5. If sources disagree, do not pick one silently. State each version with its
   citation and say they conflict. Mention if a source is archived.
6. Keep it to two or three sentences unless the question needs more. No
   preamble.

{UNTRUSTED_SYSTEM_CLAUSE}
"""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    citations_valid: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    n_citations: int = 0
    truncated: bool = False


def validate_answer(text: str, n_sources: int, finish_reason: str | None = None) -> dict:
    """TODO B2. Return a dict with at least:

        {"valid": bool, "refused": bool, "invalid_citations": [ints],
         "n_citations": int, "truncated": bool, "reason": str}

    Checks:
      - every [n] is between 1 and n_sources
      - not truncated (finish_reason == "length" means the answer was cut off,
        and a cut-off prose answer LOOKS FINE -- this is T1 failure mode 4 and
        it is the dangerous one)
      - a non-refusal answer contains at least one citation
    """
    stripped = (text or "").strip()
    truncated = finish_reason == "length"
    full_refusal = stripped == REFUSAL
    partial = not full_refusal and stripped.endswith(REFUSAL)

    cited = sorted({int(m) for m in _CITE.findall(stripped)})
    invalid = [c for c in cited if not 1 <= c <= n_sources]
    n_citations = len(cited)

    if not stripped:
        reason = "empty"
    elif truncated:
        reason = "truncated (finish_reason=length)"
    elif invalid:
        reason = f"out-of-range citations {invalid} with {n_sources} sources"
    elif full_refusal:
        reason = "refusal"
    elif n_citations == 0:
        reason = "no citations in a non-refusal answer"
    else:
        reason = "partial" if partial else "ok"

    return {
        "valid": reason in ("ok", "partial", "refusal"),
        "refused": full_refusal or partial,
        "partial": partial,
        "invalid_citations": invalid,
        "n_citations": n_citations,
        "truncated": truncated,
        "reason": reason,
    }
    
    #raise NotImplementedError


def answer_question(question: str, retriever: Retriever, *, k: int = 12,
                    final_k: int = 8, reranker=None, tier: str = "MAIN",max_chars: int = 8000) -> Answer:
    """TODO: retrieve -> (rerank) -> generate -> validate -> maybe repair.

    B3: on validation failure, retry ONCE with a corrective message (or a
    bigger token budget if truncated). If that also fails, return REFUSAL.
    We never strip bad citations: an invented index usually means an invented
    claim, and stripping hides the evidence while keeping the claim.
    """
    hits = retriever.search(question, k=k)
    final = reranker.rerank(question, hits, k=final_k) if reranker else list(hits)[:final_k]

    raw_context = format_context(final, max_chars=max_chars)
    n_sources = len(re.findall(r"^\[\d+\] \(source:", raw_context, flags=re.M))
    messages = [{"role": "user", "content":
                 f"{delimit_untrusted(raw_context)}\n\nQuestion: {question}\n\nAnswer with citations:"}]
    
    out = chat(messages, system=ANSWER_SYSTEM, tier=tier, temperature=0.0,
               max_tokens=1500, return_full=True)
    text = out["text"].strip()
    check = validate_answer(text, n_sources, out.get("finish_reason"))

    if not check["valid"]:
        if check["truncated"]:
            retry, budget = messages, 3000          
        else:
            retry, budget = [*messages,
                             {"role": "assistant", "content": text},
                             {"role": "user", "content":
                              f"Your answer failed validation: {check['reason']}. "
                              f"Only sources [1] to [{n_sources}] exist. Rewrite it so every "
                              f"factual sentence cites one of them, or reply with exactly:\n{REFUSAL}"}], 600
        out = chat(retry, system=ANSWER_SYSTEM, tier=tier, temperature=0.0,
                   max_tokens=budget, return_full=True)
        text = out["text"].strip()
        check = validate_answer(text, n_sources, out.get("finish_reason"))
    
    if not check["valid"]:
        text = REFUSAL
        check = validate_answer(text, n_sources)

    return Answer(
        question=question,
        text=text,
        hits=final[:n_sources],
        refused=check["refused"],
        citations_valid=check["valid"],
        invalid_citations=check["invalid_citations"],
        n_citations=check["n_citations"],
        truncated=check["truncated"],
    )


    #raise NotImplementedError


def answer_with_gold_context(question: str, gold_docs: list[str], *,
                             tier: str = "MAIN") -> Answer:
    """TODO E2: same generator, but the context is the gold documents.

    No retrieval at all -- read data/corpus/<doc_id>.md for each gold doc,
    chunk it or pass it whole, and generate. The difference between this and
    answer_question() is the damage your retriever is doing.
    """
    chunks = [c for j, doc in enumerate(gold_docs, 1)
              for c in markdown_chunks(doc, f"gold-{j}", size=400)]
    gold = [Hit(c, 1.0, "gold", i) for i, c in enumerate(chunks)]

    class _GoldRetriever(Retriever):
        def search(self, query, k=8):
            return gold

    return answer_question(question, _GoldRetriever(), k=len(gold),
                           final_k=len(gold), tier=tier, max_chars=40000)
    
    #raise NotImplementedError