#!/usr/bin/env python3
"""Lab 7 B1 — find the semantic-cache threshold where it starts answering the wrong question.

    python labs/lab7/scratch2.py

SAME pairs have the same answer in different words: the cache SHOULD hit.
TRAPS are near-identical text with a different answer: the cache must NEVER hit.
The highest TRAP similarity is the breaking threshold.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aip.embed import embed_batch  # noqa: E402

SAME = [
    ("How long do I have to file a claim?", "What is the deadline for submitting a reimbursement claim?"),
    ("What is the cataract limit on Silver?", "How much does the Silver plan pay for cataract surgery?"),
    ("Is ambulance covered on the Silver plan?", "Does Aurora Silver pay for an ambulance?"),
    ("What is the waiting period for pre-existing conditions?", "How long before pre-existing diseases are covered?"),
    ("What is the room rent limit on Gold?", "How much room rent does the Gold plan allow per day?"),
    ("Is maternity covered?", "Does my policy pay for childbirth?"),
    ("What is the no-claim bonus on Gold?", "How much NCB do I get on the Gold plan?"),
    ("How do I file a cashless claim?", "What is the process for a cashless claim?"),
    ("What does the Bronze plan cover?", "What is included in Aurora Bronze?"),
    ("What is the cataract limit on Silver?", "what's the silver cataract sublimit"),
]

TRAPS = [
    ("What is the no-claim bonus on Gold?", "What is the no-claim bonus on Silver?"),
    ("What is the waiting period on Gold?", "What is the waiting period on Silver?"),
    ("Is ambulance covered on Silver?", "Is ambulance covered on Bronze?"),
    ("What is the cataract limit on Silver?", "What is the knee replacement limit on Silver?"),
    ("What is the room rent limit on Bronze?", "What is the ICU limit on Bronze?"),
    ("What would a Gold plan cost for a 40-year-old?", "What would a Gold plan cost for a 60-year-old?"),
    ("What was the claim deadline in 2024?", "What is the claim deadline in 2026?"),
    ("What is covered under maternity benefits?", "What is excluded under maternity benefits?"),
    ("How do I file a cashless claim?", "How do I file a reimbursement claim?"),
    ("What is the co-payment on the senior citizen plan?", "What is the co-payment on the Gold plan?"),
]


def sims(pairs):
    v = embed_batch([q for pair in pairs for q in pair], input_type="query")  # same as the retriever
    return [float(v[2 * i] @ v[2 * i + 1]) for i in range(len(pairs))]       # unit vectors: dot = cosine


same, traps = sims(SAME), sims(TRAPS)

for label, pairs, s in (("SHOULD HIT (same answer)", SAME, same), ("MUST NOT HIT (different answer)", TRAPS, traps)):
    print(f"\n{label}")
    for (a, b), x in sorted(zip(pairs, s), key=lambda t: -t[1]):
        print(f"  {x:.3f}  {a}  |  {b}")

print("\nthreshold   paraphrases hit   wrong answers")
for t in (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98):
    print(f"   {t:.2f}         {sum(x >= t for x in same):>2}/{len(same)}             {sum(x >= t for x in traps):>2}/{len(traps)}")

breaking = max(traps)
print(f"\nBreaking threshold (highest trap similarity): {breaking:.3f}")
print(f"Paraphrases above it: {sum(x > breaking for x in same)}/{len(same)}")