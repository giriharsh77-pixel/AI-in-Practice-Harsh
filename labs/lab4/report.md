# Lab 4 Report: RAG v1

## Part A: The answer prompt

I wrote ANSWER_SYSTEM before opening aip/rag.py and then revised it after comparing the two. The final version is in labs/lab4/rag.py. 

The prompt reads as follows, where {REFUSAL} is the exact refusal string and {UNTRUSTED_SYSTEM_CLAUSE} is the clause from aip/guards.py, both inserted by the f-string:

You are a helpdesk assistant for Aurora health insurance. Follow these rules, in priority order:

1. If the sources do not answer the question, reply with exactly: {REFUSAL} If they answer only part of it, answer that part with citations, then end with that exact sentence for the rest.
2. Answer ONLY from the numbered sources provided. Do not use general knowledge or assumptions, and never infer numbers, limits or deadlines the sources do not state.
3. End every factual sentence with the source index that supports it, e.g. [1] or [2][5].
4. Only cite source numbers that were actually supplied. Never invent one.
5. If sources disagree, do not pick one silently. State each version with its citation and say they conflict. Mention if a source is archived.
6. Keep it to two or three sentences unless the question needs more. No preamble.

{UNTRUSTED_SYSTEM_CLAUSE}


It has six rules in priority order. Rule 1 is the exact refusal string, including how to handle a partial answer. Rule 2 says to answer only from the numbered sources. Rule 3 asks for citations by index, rule 4 forbids citing a number that was never supplied, rule 5 covers sources that disagree, and rule 6 keeps answers to two or three sentences.

Compared with the reference, there are six differences.

1. The reference types the refusal sentence out by hand. My prompt inserts the REFUSAL constant through the f-string, so the prompt and the code that detects refusals always use the same string.
2. The reference has no way to give a partial answer. It either answers everything or refuses everything. My prompt answers the supported part with citations and ends with the refusal sentence. Q37 needs this, because otherwise the model has to either drop a fact it could support or make up the missing limit.
3. Both prompts forbid outside knowledge. My prompt also says not to infer numbers, limits or deadlines, since an invented figure is the most expensive mistake an insurance helpdesk can make.
4. Both prompts ask the model to point out conflicting sources. My prompt also asks it to mention when a source is archived, which targets the claims-timelines (45 days) versus claims-timelines-2024-ARCHIVED (30 days) case.
5. I took the priority ordering from the reference, with refusal as rule 1, so the model knows which rule wins when answering and refusing pull in different directions.
6. Both prompts limit length. My prompt also says no preamble, because output tokens drive most of the cost and latency (T1 §2.2).

Since a partial answer always ends with the refusal sentence, validate_answer sorts every output into one of three kinds. If the text is exactly the refusal string, it is a full refusal and needs no citations. If it ends with the refusal string and has at least one citation, it is a partial answer. Anything else is a normal answer and needs at least one citation. The reference checks for refusals with startswith, which would count my partial answers as normal answers.

## Part B: Citation enforcement

Citation validity was 1.000 (45 out of 45). This comes from the code, not the prompt. validate_answer checks every [n] against the number of sources format_context actually rendered. That number can be smaller than len(hits), because format_context drops any source that does not fit in max_chars, and the reference pipeline passes len(hits) instead. The validator also rejects empty answers, truncated answers (finish_reason of "length"), and any answer that is not a refusal but has no citations.

When validation fails, I retry once and then refuse. A truncated answer is retried with double the token budget. Any other failure is retried with a message that names the problem and the valid range of source numbers. If the retry fails too, the answer is replaced with the refusal string. I decided against stripping bad citations: a made-up source number usually goes with a made-up claim, and removing the number hides the evidence while leaving the claim in place. I allow only one retry to stay inside the cost and latency budget. Every path through the function ends in either a valid answer or a refusal, so an invalid citation never reaches the user.

## Part C: Refusal

C1. All five unanswerable questions (Q36 to Q40) were declined, for a recall of 5/5. Four were full refusals. Q40 was a partial answer: it said the helpline number is printed on the policy schedule [1] and refused the rest. Q37 was a full refusal.

C2. Q37 should have produced a partial answer, since the corpus says a Platinum international benefit exists but does not give its limit. With retrieved context it was refused outright. Before changing the prompt I checked retrieval. Neither relevant document (exclusions or plans-overview) was in the top 5. At final_k = 10, plans-overview came in at rank 9, but the chunk retrieved was a table fragment that had lost its header row and had no row about international cover. The fact Q37 depends on never reached the model at final_k 5, 8 or 10, so given what it saw, the full refusal was the right call. This is failure mode 2 (chunk boundary), and raising final_k does not help. With gold context, the same generator gave the correct partial answer: "Treatment taken outside India is permanently excluded across Aurora indemnity plans, except under the Platinum plan's international emergency benefit [1][5][19]", followed by the refusal sentence for the limit. The partial-answer rule works whenever the supporting fact is present (Q40 shows the same thing), so I left the prompt alone. The whole loss on Q37 comes from retrieval.

C3. Four of the 40 answerable questions were refused. Refusal recall is 5/5 = 1.00 and refusal precision is 5/9 = 0.56. Both numbers are noisy. With only five unanswerable questions, one case moves recall by 0.20 and precision by about 0.1, so I do not treat any difference smaller than about 0.15 as real. Precision falls short of the 0.70 target because of four wrongful refusals: Q23, Q25, Q43 and Q44.

C4. For the strict setting I added this sentence to the prompt: "Refuse unless the sources explicitly and directly state the answer to the whole question. If you are unsure whether the sources fully support an answer, refuse."

| Setting | Recall | Precision | Answerable questions refused |
|---|---|---|---|
| Default | 5/5 = 1.00 | 5/9 = 0.56 | Q23, Q25, Q43, Q44 |
| Strict | 5/5 = 1.00 | 5/13 = 0.38 | the same four, plus Q27, Q28, Q34, Q45 |

Recall was already 5/5, so being stricter could only add wrongful refusals, and it added four with nothing gained. The precision drop of 0.18 is only just above the noise level, so the raw counts (4 wrongful refusals going to 8) are the better evidence.

I would keep the default setting. The two kinds of error cost very different amounts. An unnecessary refusal sends the customer to a human agent, which costs a few minutes and can be fully recovered. An invented answer, such as a wrong claim deadline or coverage limit, gets acted on. The customer may lose a valid claim, and the insurer may face a complaint or a regulatory finding. For that reason I would accept several unnecessary refusals to avoid a single invented answer, and I lean the system toward refusing. On this test set, though, the strict setting added four wrongful refusals without preventing any invented answers, so it is worse than the default on both counts. I would only reconsider it for a deployment with no human agent to fall back on. The four wrongful refusals under the default setting are covered in E3.

## Part D: The judge

D1. I wrote two single-criterion rubrics as new constants in evaluate.py and left the templates in aip/ unchanged.

The faithfulness rubric (0 or 1) makes five changes to the template. First, anything taken from outside the context counts as unsupported, even if it is true. Second, a claim stronger than the context counts as unsupported, for example "up to 12" turned into "12", a dropped condition, waiting period or exclusion, or "may" turned into "will". Third, support can come from combining several parts of the context, and the judge checks the claim itself rather than whether the [n] points at the right source, since the code already checks citation numbers. Fourth, a full refusal makes no claims and counts as supported, and a partial answer is judged only on what it says before the refusal sentence. Fifth, saying that sources conflict counts as supported if the context really does contain the conflict.

The correctness rubric (0, 1 or 2) makes three changes. It has a separate branch for questions whose reference answer says to decline: declining, or answering only the supported part, scores 2, and a confident answer scores 0. Without that branch, all five correct refusals would have scored 0. For answerable questions, a wrong central fact (a number, limit, deadline or yes/no) scores 0, a missing secondary detail scores 1, and a full refusal scores 0. For conflicting sources, giving the current value and marking the archived one as superseded earns full marks, while presenting both as equally valid scores 1.

Both rubrics ask the judge to give its reason before its score. I also changed how parse failures are handled. The provided harness scored an unparseable verdict as 0, which drags the average down without anyone noticing. My judge functions return None when llm_judge reports a parse error or the verdict has no score, and every average skips None values. The full run printed how many cases were excluded, and there were none.

On the full run, faithfulness was 0.956 (43 of 45) and correctness was 0.750 normalised (1.500 out of 2 on the 40 answerable questions).

D2. I hand-labelled 20 answers against each rubric. The sample was every refusal, full or partial (9 in total), plus every third answered question. I picked them using my own refused flag and never looked at the judge's scores while choosing, so that refusal handling would be tested and the labels would not all fall into one class.

| Rubric | Raw agreement | Cohen's kappa | n |
|---|---|---|---|
| Faithfulness | 20/20 = 1.00 | 1.00 | 20 |
| Correctness | 18/20 = 0.90 | 0.81 | 20 |

The faithfulness kappa of 1.00 depends on a single unfaithful answer, Q04. With 19 of 20 answers faithful, chance agreement is very high, and one disagreement would have pushed kappa close to 0. What it shows is that the judge caught the one unfaithful answer, which is less than showing the judge is perfect.

Both correctness disagreements came from the rubric's wording. On Q40 I gave 2 and the judge gave 1: the judge read "asserts something the reference does not support" as "says something the reference does not mention", while I read it as "contradicts the reference". Changing it to "contradicts the reference or invents a figure" would fix this. On Q33 I gave 1 and the judge gave 2, because the rubric never says a comparison has to link each value to its plan, and adding that requirement would fix it. A third unclear case came up while labelling. On Q37 the reference expects a partial answer but the system refused fully, and the decline branch as written scores that 2. I gave it 2 to follow the rubric, although I think it deserves 1. Both kappa values are above 0.4, so I did not revise the rubric, but these three wording fixes are the next changes I would make.

D3. The generator is gemini-3.7-flash (MAIN) and the judge is gemini-3.5-flash (LARGE). They are different models from the same provider and family, which reduces self-preference bias (T3 §4.1) but does not remove it. The bias would push scores up, because a judge trained like the generator is more likely to accept its phrasing as supported and correct. There is a second concern too. Going by the version numbers, the judge looks like an older generation than the generator, not the stronger tier llm_judge is meant to use, and a weaker judge is more likely to miss subtle unsupported claims, which would also inflate faithfulness. My calibration puts some limit on this. On the 20 labelled cases the judge disagreed with me in both directions on correctness, once stricter and once more lenient, so I saw no upward pattern. Twenty cases cannot rule out a small bias, though. The proper fix would be a judge from another provider, which I did not have configured.

## Part E: Evaluation

E1. Results on all 45 questions:

| Metric | Target | Result | Met |
|---|---|---|---|
| Citation validity | 1.00 | 1.000 (45/45) | yes |
| Faithfulness | 0.90 or more | 0.956 (43/45), kappa 1.00 | yes |
| Correctness | 0.75 or more | 0.750, kappa 0.81 | yes, exactly at target |
| Refusal recall | 4/5 or more | 5/5 | yes |
| Refusal precision | 0.70 or more | 5/9 = 0.56 | no |
| Repair rate | reported | 20/45 = 0.444 (reference 0.089) | reported |
| Cost per query | $0.01 or less | $0.0037, generation only | yes |
| p95 end-to-end latency | 6,000 ms or less | 9,066 ms (p50 4,194 ms) | no |

I measured cost and latency per question, adding up every generator call including retries. Judge calls are left out because they are part of evaluating the system, not running it.

Every one of the 20 repairs was a truncation, meaning the first attempt ended with finish_reason "length". The generator is a reasoning model, and its hidden thinking tokens use up most of the 600-token output limit. This is the same problem as the truncated judge described in the handout, only in the generator. The truncation check in validate_answer caught all 20. Without it, 44% of answers would have been sent out cut off mid-sentence, some of them ending on a partial figure that looks complete. The retries also explain the latency miss, since each one is a second full model call, which pushes p95 to 9.1 seconds. Raising the first-attempt limit to around 1,500 tokens should fix this. I did not make that change here, because it would alter every answer and invalidate the C4 and D2 results. It is the first item for Lab 5.

E2. I ran the same generator twice on the 42 questions that have at least one relevant document (Q36, Q38 and Q39 have none). The prompt, model, validation and repair were identical both times. One run used retrieved context. The other used the gold documents, split with the same chunker and passed through answer_question with a stand-in retriever, with the context limit raised so the gold documents were never cut short.

| | Correctness (normalised) |
|---|---|
| A: gold context (generation ceiling) | 0.869 |
| B: retrieved context (my system) | 0.750 |
| Retrieval-attributable loss (A minus B) | 0.119 |
| Generation-attributable loss (1 minus A) | 0.131 |

The two losses are effectively tied. The gap between them is 0.012, which is exactly what one question moving by one score point produces (1 divided by 2, divided by 42), and that is smaller than the judge's own disagreement rate in D2. Neither loss is clearly bigger, and each accounts for about half. Since A is 0.869, even perfect retrieval would only recover 0.119, so improving retrieval alone cannot reach the ceiling.

With the losses tied, I chose where to start Lab 5 based on how well each fix is understood. I will start with generation. Twenty of 45 first attempts were truncated (E1), the cause is known, and a one-line change should fix it while also addressing the latency miss. Retrieval comes second. I have identified specific chunk-level failures, listed below, but there is no single change that fixes them all.

E3. Of the 20 answers that scored below 2, gold context fixed 10 and did not fix the other 10, which matches the tie in E2. In every case the relevant document itself was retrieved at rank 1 to 3, so the retrieval failures happen at the chunk level: the right document, but the wrong chunk from it. I tagged ten of them.

| Question | Failure mode | What happened |
|---|---|---|
| Q44 | 2, chunk boundary | The product code and the sum-insured options are in separate chunks, and only the chunk with the code was retrieved |
| Q32 | 2, chunk boundary | The co-payment row is in a table chunk without its header, the same split seen in Q33 and Q37 |
| Q28 | 2, chunk boundary | Two conditions arrived in separate chunks and the answer presented them as conflicting |
| Q29 | 4, ranking | A motor insurance chunk from a different product line made the top 5, and the answer quoted it |
| Q04 | 4, ranking | The chunk about the rider that shortens the pre-existing disease wait was not in the top 5 |
| Q35 | 4, ranking | The chunk about the OPD rider's dental check-up was not in the top 5 |
| Q22 | 6, generation | The missing fact was in the retrieved context and the answer left it out |
| Q25 | 6, generation | Prompt rule 2 forbids working out numbers, which blocks the subtraction the question needs, and with gold context the answer was a full refusal |
| Q11 | 6, generation | Gold context did worse and dropped the air ambulance cover it had been given |
| Q03 | 6, generation | The answer is correct but incomplete, as are Q05, Q21 and Q24, because rule 6's length limit cuts secondary details |

That gives three chunk boundary failures, three ranking failures and four generation failures. I left out Q40, because its low score comes from the rubric wording problem found in D2, not from anything the system did wrong.

In order of expected impact, the Lab 5 backlog is as follows. First, raise the output token limit, which should remove the 20 truncated first attempts, the latency miss and the unnecessary refusal sentences in Q23 and Q25. Second, repeat each table's header row in every chunk it is split into, which covers Q32, Q33 and Q37. Third, revisit two prompt rules: rule 2 stops the model doing simple arithmetic (Q25), and rule 6's length limit drops secondary details (Q03, Q05, Q21, Q24). Fourth, filter out other product lines when documents are ingested, the same way Lab 3 filtered archived documents, which would fix Q29.