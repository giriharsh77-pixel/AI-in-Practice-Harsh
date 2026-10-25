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

<<<<<<< HEAD
from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted, enforce_citations  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402

REFUSAL = "I don't have enough information in the provided sources to answer that."

ANSWER_SYSTEM = f"""\
You answer questions using ONLY the numbered sources provided below.

Rules, in priority order:
1. If the sources do not contain the answer, reply exactly:
   "I don't have enough information in the provided sources to answer that."
   Do not guess, speculate, or fall back on your own knowledge.
2. Every factual sentence must end with a citation to the source(s) that
   support it, using the format [1] or [2][5]. Multiple citations may follow
   one sentence.
3. Never cite a source number that was not given to you.
4. If sources disagree with each other, say so explicitly and cite both sides.
5. Be concise. Two or three sentences unless the question requires more detail.
6. If you can partially answer but not fully, answer the part you can support
   and refuse the rest.
=======
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
>>>>>>> source/main

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
<<<<<<< HEAD
    """Return validation results for a generated answer."""
    refused = text.strip().startswith(REFUSAL[:40])
    truncated = finish_reason == "length"

    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
    invalid = [c for c in cited if c < 1 or c > n_sources]
    n_citations = len(cited)

    if truncated:
        return {
            "valid": False, "refused": refused, "invalid_citations": invalid,
            "n_citations": n_citations, "truncated": True,
            "reason": "answer was truncated (finish_reason=length)",
        }

    if invalid:
        return {
            "valid": False, "refused": refused, "invalid_citations": invalid,
            "n_citations": n_citations, "truncated": False,
            "reason": f"out-of-range citations: {invalid}",
        }

    if not text.strip():
        return {
            "valid": False, "refused": refused, "invalid_citations": [],
            "n_citations": 0, "truncated": False,
            "reason": "empty answer",
        }

    if not refused and n_citations == 0:
        return {
            "valid": False, "refused": False, "invalid_citations": [],
            "n_citations": 0, "truncated": False,
            "reason": "non-refusal answer has no citations",
        }

    return {
        "valid": True, "refused": refused, "invalid_citations": [],
        "n_citations": n_citations, "truncated": False, "reason": "",
    }


def answer_question(question: str, retriever: Retriever, *, k: int = 12,
                    final_k: int = 5, reranker=None, tier: str = "MAIN") -> Answer:
    """Retrieve -> (rerank) -> generate -> validate -> maybe repair."""
    hits = retriever.search(question, k=k)

    if reranker is not None:
        final_hits = reranker.rerank(question, hits, k=final_k)
    else:
        final_hits = hits[:final_k]

    context_str = format_context(final_hits)
    context_block = delimit_untrusted(context_str)
    prompt = f"{context_block}\n\nQuestion: {question}\n\nAnswer with citations:"

    result = chat(prompt, system=ANSWER_SYSTEM, tier=tier,
                  temperature=0.0, max_tokens=600, return_full=True)
    text = result["text"].strip()
    finish_reason = result.get("finish_reason")

    val = validate_answer(text, len(final_hits), finish_reason)

    if not val["valid"]:
        repair_msg = (
            f"Your previous answer had a problem: {val['reason']}. "
            f"Please answer the question again. Remember: cite only source numbers "
            f"[1] through [{len(final_hits)}]. If you cannot answer from the sources, "
            f'reply exactly: "{REFUSAL}"'
        )
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
            {"role": "user", "content": repair_msg},
        ]
        result2 = chat(messages, tier=tier, temperature=0.0,
                       max_tokens=600, return_full=True)
        text2 = result2["text"].strip()
        finish2 = result2.get("finish_reason")
        val2 = validate_answer(text2, len(final_hits), finish2)

        if val2["valid"]:
            text = text2
            val = val2
        else:
            text = REFUSAL
            val = validate_answer(text, len(final_hits))
=======
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
>>>>>>> source/main

    return Answer(
        question=question,
        text=text,
<<<<<<< HEAD
        hits=list(final_hits),
        refused=val["refused"],
        citations_valid=val["valid"],
        invalid_citations=val["invalid_citations"],
        n_citations=val["n_citations"],
        truncated=val["truncated"],
    )


def answer_with_gold_context(question: str, gold_docs: list[str], *,
                             tier: str = "MAIN") -> Answer:
    """Same generator, but context is the gold documents -- no retrieval."""
    from aip.chunking import markdown_chunks

    all_chunks = []
    for i, doc_text in enumerate(gold_docs):
        chunks = markdown_chunks(doc_text, f"gold_{i}", size=1200)
        all_chunks.extend(chunks)

    fake_hits = [Hit(chunk=c, score=1.0, source="gold", rank=i)
                 for i, c in enumerate(all_chunks)]

    context_str = format_context(fake_hits)
    context_block = delimit_untrusted(context_str)
    prompt = f"{context_block}\n\nQuestion: {question}\n\nAnswer with citations:"

    result = chat(prompt, system=ANSWER_SYSTEM, tier=tier,
                  temperature=0.0, max_tokens=600, return_full=True)
    text = result["text"].strip()
    finish_reason = result.get("finish_reason")

    val = validate_answer(text, len(fake_hits), finish_reason)

    if not val["valid"] and val["invalid_citations"]:
        text = REFUSAL
        val = validate_answer(text, len(fake_hits))

    return Answer(
        question=question,
        text=text,
        hits=list(fake_hits),
        refused=val["refused"],
        citations_valid=val["valid"],
        invalid_citations=val["invalid_citations"],
        n_citations=val["n_citations"],
        truncated=val["truncated"],
    )
=======
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
>>>>>>> source/main
