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

    return Answer(
        question=question,
        text=text,
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
