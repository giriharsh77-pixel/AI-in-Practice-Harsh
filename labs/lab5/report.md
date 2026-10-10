# Lab 5 Report: RAG v2

## Part A: Classifying every failure

The input was reports/lab4.json. Twenty of the 45 questions scored below 2 on correctness, and none had an invalid citation.

When I first ran the shipped diagnose.py, it labelled all 20 as chunk_boundary. That was not a diagnosis. The branches for modes 6, 4/5 and 3 were empty, so every question fell through to the default at the bottom of classify, and main() was not computing any of the evidence those branches need.

I made three changes. First, I rewrote answer_in_corpus (mode 1) to check the numbers in the gold answer rather than word overlap, because insurance answers depend on figures. Commas are removed so that "Rs 1,00,000" matches "₹1,00,000", ranges such as "45-60" are split into separate numbers, and at least half of the numbers must be found, since some gold answers contain a figure no document states (Q25's ₹60,000 is a subtraction). My first version treated "45-60" as one token and wrongly marked Q21 as missing content, which I fixed. To check the result, I ran both tests on the 40 answerable questions, whose answers are in the corpus by design. Both gave 0/40 false flags, so on that measure my test is no worse but also no better. The difference shows on Q40: my test flags it as missing content and the shipped one does not. The conclusion is correct, because the corpus has no helpline number, but my test reaches it for a narrow reason. The only number in that gold answer is "24x7", which the corpus writes as "24×7". I am reporting this rather than claiming my test detected the missing number directly.

Second, I filled in classify in the order the tree gives: 7, then 1, then 6, then 4/5, then 3/2. Mode 6 is assigned only when the gold-context answer still scores below 2, so a gold-context fix counts as a retrieval failure. Third, main() now runs the gold-context test for each failure using my Lab 4 generator and judge, finds the gold chunk (the chunk from the relevant documents that best matches the gold answer), and records that chunk's rank in the top 30 and whether its own text retrieves it.

I changed one thing about the tree. In all 20 failures the relevant document was already in the final five, so a document-level check could never separate the retrieval modes. I therefore applied the rank checks to the gold chunk instead. Mode 5 cannot occur because Lab 4 had no reranker, and the mode 3 test is weak with dense retrieval, since a chunk's own text almost always retrieves it. No failure reached that branch.

A2. Five cases had the gold chunk in context yet were fixed by gold context, so I read their retrieved chunks. Q19 is mode 2: the worked example is split across three chunks, and the one saying pharmacy is not reduced was not retrieved. The table in the context implies it, so this could reasonably be called generation. Q22 is mode 6: the missing fact is in retrieved chunk [2], and the answer leaves it out. Q28 is mode 2: the pre-approval condition is a table row cut off from its header, and the model reported it as conflicting with the emergency condition. Q29 is mode 4: the correct 45 days was retrieved and given, but a motor insurance chunk ranked into the top five and was quoted too. Q32 is mode 2: Platinum's nil co-payment appears only in a table row without its header. These judgements are recorded in diagnose.py as the evidence for each case.

A1 and A3. The script's own tally was 10 generation, 5 chunk boundary, 4 ranking and 1 missing content. After A2 it is as follows.

| Mode | n | Questions |
|---|---|---|
| 6 Generation | 11 | Q03, Q05, Q10, Q11, Q20, Q21, Q22, Q23, Q24, Q25, Q26 |
| 4 Ranking | 5 | Q04, Q29, Q35, Q42, Q44 |
| 2 Chunk boundary | 3 | Q19, Q28, Q32 |
| 1 Missing content | 1 | Q40 |
| 3, 5, 7 | 0 | |

    failure mode          n    share   cumulative
    generation            11   55.0%   55.0%  ████████████████
    ranking                5   25.0%   80.0%  ████████
    chunk_boundary         3   15.0%   95.0%  ████
    missing_content        1    5.0%   100.0%  ██

Generation and ranking account for 16 of the 20 failures. This differs from the reference's 13 of 14 in mode 6, and it matches my own Lab 4 decomposition, which found the generation loss (0.131) and retrieval loss (0.119) roughly tied. Q40 is the only missing-content case: the corpus mentions a helpline but never gives its number, so what it needs is that number added. No retrieval or generation change can fix it.

## Part B: Ranking by expected value

| Cluster | n | Fix | Expected recovery | Cost | Latency | Effort |
|---|---|---|---|---|---|---|
| Generation | 11 | Relax rule 6, the length limit | 2 to 4, but the reference lost 0.05 doing this | Up | Up | Trivial |
| Generation | 11 | Raise the output token limit, 600 to 1,500 | About 0; truncated answers were already retried at 1,200 | Down | Down | Trivial |
| Ranking | 5 | Raise final_k from 5 to 8 | 2 (Q42 at rank 6, Q44 at rank 8) | About 50% more input, under 2× | Small | Trivial |
| Ranking | 1 | Filter other product lines at ingest | 1 (Q29) | None | None | Low |
| Chunk boundary | 3 | Repeat table headers in every chunk | 2 (Q28, Q32) | None | None | Medium |
| Missing content | 1 | None; add the number to the corpus | 0 | | | |

I chose to raise final_k from 5 to 8. Generation is the largest cluster, but its only cheap fix for correctness, relaxing the length limit, already made the reference worse, while raising final_k is a one-parameter change with two recoveries I could name and check by rank, at under twice the cost.

This does not repeat the reference's mistake of fixing retrieval against a generation tally. My tally puts 8 of 20 failures in retrieval (ranking and chunk boundary), not 1 in 14, and the Lab 4 decomposition put retrieval at half the loss.

Prediction, written before the change: I expect raising final_k from 5 to 8 to recover 2 of the 5 ranking failures, Q42 and Q44. Q04 and Q35 have their gold chunks at ranks 21 and 18 and will stay failed. The extra context may cause up to two new generation failures, so the net gain may be less than two questions. Since Q44 was a wrongful refusal, refusal precision should improve.

## Part C: The fixes

Fix 1 (v2) changes one parameter: final_k from 5 to 8 in labs/lab4/rag.py. Everything else stayed the same, including the prompt, retriever, token limits and judge.

After Fix 1, Q29 was still failing and had started quoting the archived document as well as the motor chunk. The handout's fix for the archived trap is metadata filtering. I did this in Lab 3 D3 with a Chroma filter on status, which raised hit rate at 1 on Q29 to Q31 from 0.67 to 1.00, but I deliberately left it out of the Lab 4 pipeline so that the prompt's conflict rule would be tested. Fix 2 (v3) brings it back at ingest. build_retriever in labs/lab4/evaluate.py now excludes archived documents and other product lines (motor, life and travel), which removes 4 of 30 documents and cuts the chunks from 235 to 214. None of the 4 is a relevant document for any golden question. I kept health add-ons such as personal accident cover and the critical illness rider. final_k stayed at 8, so v2 to v3 measures Fix 2 alone.

Prediction for Fix 2, written before the run: it will recover Q29 and possibly Q22 and Q05, which each had an archived or life-insurance distractor, for 1 to 3 recoveries. Q30 and Q31 should stay correct, and cost and latency should not change.

## Part D: Proving it

D1. The same 45 questions, measured on every Lab 4 metric.

| Metric | v1 | v2 (fix 1) | v3 (fix 2) | Change v1 to v3 |
|---|---|---|---|---|
| Correctness | 0.750 | 0.800 | 0.8125 | +0.0625 |
| Faithfulness | 0.956 | 0.911 | 0.933 | −0.023 |
| Citation validity | 1.000 | 1.000 | 1.000 | 0 |
| Refusal recall | 5/5 | 5/5 | 5/5 | 0 |
| Refusal precision | 5/9 | 5/9 | 5/8 | +1 fewer wrongful refusal |
| nDCG@10 | 0.8527 | 0.8527 | 0.869 | +0.016 |
| Recall@5 | 0.9028 | 0.9028 | 0.9028 | 0 |
| Cost per query | $0.0037 | $0.0039 | $0.0036 | ×0.97 |
| p95 latency | 9,066 ms | 16,152 ms | 14,542 ms | +5,476 ms |
| Repair rate | 20/45 | 17/45 | 14/45 | −6 |
| Failures | 20 | 16 | 16 | −4 |

Fix 1 cannot change nDCG or recall, because final_k only decides how many ranked chunks are passed on. Fix 2 changes the index, and with 21 distractor chunks gone, nDCG@10 rose to 0.869 and hit rate at 1 from 0.786 to 0.833.

Fix 1 prediction against outcome: exactly the two predicted questions recovered, Q42 (1 to 2) and Q44 (0 to 2), and Q04 and Q35 stayed failed. Q19 also recovered, because the chunk my A2 judgement said was missing came in at position 8, which supports that judgement. Q03 recovered as well, but it was a generation case, so I put it down to variation between runs. The predicted rise in refusal precision did not happen: Q44 stopped refusing, but Q22 started.

Fix 2 prediction against outcome: Q05 recovered as predicted once the life-insurance chunk was gone. Q22 improved from a full refusal back to a partial answer but did not recover. Q30 and Q31 stayed correct. The question the fix was aimed at, Q29, did not move: it scored 1 before and 1 after. Its v3 answer is now exactly right ("45 days from the date of the query... If a response is not received within 45 days, Aurora closes the claim"), with no motor or archived value. It still scores 1 because the gold answer also says this is "under the rules effective 1 April 2026", and my rubric scores a missing secondary detail as 1. So the fix worked on the answer, but the metric did not register it. This is the fix that did not work by its number, and the reason is the metric rather than the fix.

D2. Things that got worse, in order of how serious they are.

First, p95 latency. It rose from 9,066 ms to 16,152 ms with Fix 1 and was still 14,542 ms after Fix 2, more than twice the 6,000 ms target. Eight chunks mean about 55% more input per call, and the retries for truncated answers add a second full call. Cost per query stayed flat, since fewer retries offset the larger inputs.

Second, faithfulness. It fell from 0.956 to 0.911 with Fix 1. Q20, Q42 and Q45 became unfaithful, because more context gave the model more material to overreach with. Q42 recovered on correctness and became unfaithful in the same answer. Fix 2 brought faithfulness back to 0.933, since Q45 was fixed once its distractor was gone.

Third, Q22. Under Fix 1 it went from a partial answer to a full refusal, a new wrongful refusal caused by an archived chunk among the extra ones. Fix 2 removed that chunk and brought back the partial answer.

Fourth, Q33 under Fix 2. It dropped from 2 to 1 on almost identical content: the v3 answer only drops the word "respectively". This is the same rubric ambiguity I found for Q33 in Lab 4 D2, whether values must be tied to plan names, so I read it as the judge varying on a borderline case rather than the system getting worse. It still counts against the headline number.

By question kind, paraphrase went from 0.700 to 1.000 and single hop from 0.861 to 0.917. Multi hop rose from 0.550 to 0.600. Aggregation fell from 0.750 to 0.625, which is Q33. Trap_archived stayed at 0.833, which is Q29. Refusal recall stayed at 5/5, so no unanswerable question started being answered confidently.

D3. Remaining failures, re-classified with the same checks by hand. The script's own tallies were 9 / 4 / 2 / 1 for v2 and 8 / 5 / 2 / 1 for v3.

| Mode | v1 | v2 | v3 |
|---|---|---|---|
| Generation | 11 | 10 | 10 |
| Ranking | 5 | 3 | 2 |
| Chunk boundary | 3 | 2 | 3 |
| Missing content | 1 | 1 | 1 |
| Total | 20 | 16 | 16 |

The distribution moved as predicted. Ranking fell from 5 to 3 with Fix 1, exactly the two recoveries, and to 2 with Fix 2, when Q29's distractors were removed. Q29 then counts as generation, because what is left is a missing secondary detail. That is the kind of shift the handout describes: removing the retrieval problem showed that the remaining gap was in the answer. The same happened with Q22, whose refusal disappeared with the distractor, leaving the same omission as in v1. Chunk boundary went from 3 to 2 (Q19 recovered) and back to 3 (Q33). What remains is mostly generation (10 of 16), with a small cluster of table rows missing their header (Q28, Q32, Q33).

D4. The next fix I would make is raising the first-attempt output limit from 600 to about 1,500 tokens. Fourteen of 45 first attempts are still cut off, and each one makes a second full call, which is the main reason p95 is 14.5 seconds. I expect this to bring the repair rate to near zero and p95 close to single-call latency, around 5 to 7 seconds, at the same or lower cost. I expect almost no change in correctness, because the retry at 1,200 tokens already produces complete answers. It is a latency fix, not a quality fix. For quality, the next fix is repeating each table's header row in every chunk it is split into. That costs nothing per query and targets the three remaining chunk-boundary failures, and I would expect it to recover two of them (Q28, Q32).