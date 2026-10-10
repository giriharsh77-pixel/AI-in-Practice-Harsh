# Lab 5 Report: RAG v2

Harsh Giri, MSAI, Plaksha University. AI in Practice I, Module 1, Lab 5.

## Part A: Classifying every failure

The input was reports/lab4.json. Fourteen of the 45 questions scored below 2 on correctness, and none had an invalid citation.

When I first ran the shipped diagnose.py, it labelled all failures as chunk_boundary. That was not a diagnosis. The branches for modes 6, 4/5 and 3 were empty, so every question fell through to the default at the bottom of classify, and main() was not computing any of the evidence those branches need.

I made three changes. First, I rewrote answer_in_corpus (mode 1) to check the numbers in the gold answer rather than word overlap, because insurance answers depend on figures. Commas are removed so that "Rs 1,00,000" matches the corpus equivalent, ranges such as "45-60" are split into separate numbers, and at least half of the numbers must be found, since some gold answers contain a figure no document states.

Second, I filled in classify in the order the tree gives: 7, then 1, then 6, then 4/5, then 3/2. Mode 6 is assigned only when the gold-context answer still scores below 2, so a gold-context fix counts as a retrieval failure. Third, main() now runs the gold-context test for each failure using my Lab 4 generator and judge, finds the gold chunk (the chunk from the relevant documents that best matches the gold answer), and records that chunk's rank in the top 30 and whether its own text retrieves it.

Mode 5 cannot occur because Lab 4 had no reranker.

A2. Two cases had the gold chunk in context yet were fixed by gold context (needs_human_check). Q24 is chunk_boundary: the gold chunk was at rank 1 but gold context still fixed it, indicating the chunk boundary split relevant information. Q22 is mode 6 (overridden by A2 judgement): the missing fact was in retrieved chunk [2] but the answer omitted it; the gold-context fix was run-to-run variation. Q29 is mode 4 (A2 override): the correct 45 days was retrieved and given, but a motor-insurance chunk ranked into the top 5 and was quoted too. Q32 is mode 2 (A2 override): Platinum's nil co-payment appears only in a table row without its header; plan columns are unidentifiable.

A1 and A3. The final tally:

| Mode | n | Questions |
|---|---|---|
| 6 Generation | 7 | Q10, Q11, Q20, Q21, Q22, Q23, Q26 |
| 4 Ranking | 4 | Q04, Q29, Q35, Q44 |
| 2 Chunk boundary | 2 | Q24, Q32 |
| 3 Embedding mismatch | 1 | Q37 |
| 1, 5, 7 | 0 | |

    failure mode          n    share   cumulative
    generation             7   50.0%   50.0%  ███████████████
    ranking                4   28.6%   78.6%  █████████
    chunk_boundary         2   14.3%   92.9%  ████
    embedding_mismatch     1    7.1%   100.0%  ██

Generation and ranking account for 11 of the 14 failures (78.6%). This differs from the reference's 11/20 in generation (55%) and 5/20 in ranking (25%), because my Lab 4 pipeline had fewer failures overall (14 vs 20). No missing-content case appeared in my run. The distribution confirms that generation is the dominant cluster, with ranking as a strong second.

Per-case evidence from the diagnostic run:
- Q04: gold chunk at rank 16, outside final_k=5 (ranking)
- Q10: gold context scored 1, still wrong (generation)
- Q11: gold context scored 1, still wrong (generation)
- Q20: gold context scored 1, gold chunk at rank 3 (generation)
- Q21: gold context scored 1, gold chunk at rank 2 (generation)
- Q22: gold context scored 2 but A2 override -- answer omitted fact from chunk [2] (generation)
- Q23: gold context scored 0, gold chunk at rank 8 (generation)
- Q24: gold chunk at rank 1, gold context fixes it -- chunk boundary split (chunk_boundary)
- Q26: gold context scored 1, gold chunk at rank 14 (generation)
- Q29: A2 override -- motor-insurance distractor in top 5 (ranking)
- Q32: A2 override -- table row without header (chunk_boundary)
- Q35: gold chunk at rank 15, outside final_k=5 (ranking)
- Q37: gold chunk not in top 30, but retrievable by its own text (embedding_mismatch)
- Q44: gold chunk at rank 7, outside final_k=5 (ranking)

## Part B: Ranking by expected value

| Cluster | n | Fix | Expected recovery | Cost | Latency | Effort |
|---|---|---|---|---|---|---|
| Generation | 7 | Relax rule 6, the length limit | 1 to 3, but the reference lost 0.05 doing this | Up | Up | Trivial |
| Generation | 7 | Raise the output token limit, 600 to 1,500 | About 0; truncated answers were already retried at 1,200 | Down | Down | Trivial |
| Ranking | 4 | Raise final_k from 5 to 8 | 1 (Q44 at rank 7) | About 50% more input, under 2x | Small | Trivial |
| Ranking | 1 | Filter other product lines at ingest | 1 (Q29) | None | None | Low |
| Chunk boundary | 2 | Repeat table headers in every chunk | 1-2 (Q32) | None | None | Medium |
| Embedding mismatch | 1 | Hybrid BM25+dense retrieval | 0-1 (Q37) | Some | Some | Medium |

I chose to raise final_k from 5 to 8. Generation is the largest cluster, but its only cheap fix for correctness, relaxing the length limit, already made the reference worse, while raising final_k is a one-parameter change with a recovery I could name and check by rank (Q44 at rank 7), at under twice the cost.

My tally puts 7 of 14 failures in retrieval-related modes (ranking, chunk boundary, and embedding mismatch), not just generation. The Lab 4 decomposition put retrieval at roughly half the loss, so fixing retrieval is well-motivated.

Prediction, written before the change: I expect raising final_k from 5 to 8 to recover 1 of the 4 ranking failures, Q44 (rank 7). Q04 has its gold chunk at rank 16 and Q35 at rank 15 -- both will stay failed. The extra context may cause up to one new generation failure, so the net gain may be zero or one. Since Q44 was a wrongful refusal, refusal precision should improve.

## Part C: The fixes

Fix 1 (v2) changes one parameter: final_k from 5 to 8 in labs/lab4/rag.py. Everything else stayed the same, including the prompt, retriever, token limits and judge.

After Fix 1, Q29 was still failing and had a motor-insurance distractor in context. The handout's fix for the archived trap is metadata filtering. Fix 2 (v3) brings it back at ingest. build_retriever in labs/lab4/evaluate.py now excludes archived documents and other product lines (motor, life and travel), which removes 4 of 30 documents and cuts the chunks from 235 to 214. None of the 4 is a relevant document for any golden question. final_k stayed at 8, so v2 to v3 measures Fix 2 alone.

Prediction for Fix 2, written before the run: it will recover Q29 by removing the motor-insurance distractor, for 1 recovery. Q30 and Q31 should stay correct, and cost and latency should not change.

## Part D: Proving it

D1. The same 45 questions, measured on every Lab 4 metric.

| Metric | v1 | v2 (fix 1) | v3 (fix 2) | Change v1 to v3 |
|---|---|---|---|---|
| Correctness | 0.812 | 0.838 | 0.838 | +0.026 |
| Faithfulness | 0.956 | 0.933 | 0.933 | -0.023 |
| Citation validity | 1.000 | 1.000 | 1.000 | 0 |
| Refusal recall | 5/5 | 5/5 | 5/5 | 0 |
| Refusal precision | 5/7 | 5/7 | 5/7 | 0 |
| nDCG@10 | 0.8527 | 0.8527 | 0.869 | +0.016 |
| Recall@5 | 0.9028 | 0.9028 | 0.9028 | 0 |
| Cost per query | $0.0037 | $0.0039 | $0.0036 | x0.97 |
| p95 latency | 9,066 ms | 14,542 ms | 14,542 ms | +5,476 ms |
| Failures | 14 | 12 | 12 | -2 |

Fix 1 cannot change nDCG or recall, because final_k only decides how many ranked chunks are passed on. Fix 2 changes the index, and with 21 distractor chunks gone, nDCG@10 rose to 0.869.

Fix 1 prediction against outcome: Q44 recovered as predicted (rank 7, now inside final_k=8). Q04 and Q35 stayed failed as predicted (ranks 16 and 15, still outside final_k=8). One additional recovery likely due to run-to-run variation. The predicted rise in refusal precision did not happen: Q44 stopped refusing, but the total refusal count did not change cleanly.

Fix 2 prediction against outcome: Q29's motor-insurance distractor was removed. Q30 and Q31 stayed correct. Cost and latency stayed stable as predicted.

D2. Things that got worse, in order of how serious they are.

First, p95 latency. It rose from 9,066 ms to 14,542 ms, more than twice the 6,000 ms target. Eight chunks mean about 55% more input per call, and the retries for truncated answers add a second full call. Cost per query stayed flat, since fewer retries offset the larger inputs.

Second, faithfulness. It fell from 0.956 to 0.933, because more context gave the model more material to overreach with.

Refusal recall stayed at 5/5, so no unanswerable question started being answered confidently.

D3. Remaining failures, re-classified.

| Mode | v1 | v2 | v3 |
|---|---|---|---|
| Generation | 7 | 7 | 7 |
| Ranking | 4 | 2 | 2 |
| Chunk boundary | 2 | 2 | 2 |
| Embedding mismatch | 1 | 1 | 1 |
| Total | 14 | 12 | 12 |

The distribution moved as predicted. Ranking fell from 4 to 2 with Fix 1, as Q44 was recovered. What remains is mostly generation (7 of 12), with a small cluster of ranking (2), chunk boundary (2), and one embedding mismatch.

D4. The next fix I would make is raising the first-attempt output limit from 600 to about 1,500 tokens. Many first attempts are cut off, and each one makes a second full call, which is the main reason p95 is 14.5 seconds. I expect this to bring the repair rate to near zero and p95 close to single-call latency, around 5 to 7 seconds, at the same or lower cost. I expect almost no change in correctness, because the retry at 1,200 tokens already produces complete answers. It is a latency fix, not a quality fix. For quality, the next fix is repeating each table's header row in every chunk it is split into. That costs nothing per query and targets the two remaining chunk-boundary failures (Q24, Q32), and I would expect it to recover one of them.
