# Aurora Policy Assistant — Evaluation Report

Sushruta, MSAI, Plaksha University. AI in Practice I, Module 1, Lab 7.

## 1. What it does

The assistant answers customers' questions about Aurora's health insurance policies, such as how long they have to file a claim, what a plan pays for cataract surgery, or whether a newborn is covered. It answers only from Aurora's own policy documents. Every statement in an answer is marked with the document it came from, and the customer can open that passage to check it. When the documents do not contain the answer, it says so rather than guessing. A second mode can calculate a premium using a fixed set of read-only tools. It cannot issue refunds or change a policy.

## 2. How well it works

I measured it on 45 test questions with known answers: 40 answerable (single facts, facts spread across documents, comparisons across plans, reworded questions, and questions that tempt it towards an outdated 2024 rule) and 5 it should refuse. Correctness and faithfulness are scored by an LLM judge whose rubrics I calibrated against my own labels in Lab 4. Every number below is reproduced offline in CI from the committed cache (`labs/lab7/gate.py`, `reports/gate.json`).

| Metric | Value | Gate |
|---|---|---|
| Correctness (0–1, answerable questions) | 0.838 | ≥ 0.80 |
| Faithfulness (every claim supported by the sources) | 0.933 (42/45) | ≥ 0.90 |
| Citation validity | 1.000 (45/45) | ≥ 0.98 |
| Refusal recall (unanswerable questions refused) | 1.000 (5/5) | ≥ 0.95 |
| Refusal precision (refusals that were right) | 0.714 (5/7) | ≥ 0.70 |
| Hit rate at 5 (a relevant document in the top 5) | 1.000 (40/40) | ≥ 0.97 |
| Cost per query | $0.0026 | ≤ $0.0031 |
| p95 generation latency | 7,649 ms | ≤ 8,500 ms |

By question type, correctness is 1.00 for reworded questions (5) and archived-rule traps (3), 0.89 for single facts (18), 0.75 for cross-plan comparisons (4) and 0.65 for facts spread across documents (10). In the Lab 6 red-team, all 17 standard attacks were blocked with no false positives on 4 normal requests.

The gates are set just past today's values, so they fail when the system gets worse. I proved the gate works by passing only one chunk to the model instead of eight: the CI build went red on correctness (0.58), refusal precision (0.24), hit rate and latency. Faithfulness rose to 1.00 and cost fell 27% in the same broken run, because a system that refuses more makes fewer claims and reads less. A gate on either alone would have called the break an improvement.

## 3. Where it fails

No answer in the test set got a central fact wrong. All 13 lost correctness points (out of 80) are partial answers, and they fall into three groups:

- **Omitted a detail that was in front of it: 9 of 13** (Q04, Q06, Q10, Q11, Q20, Q21, Q22, Q24, Q32). The right document was retrieved; the answer left out a secondary condition. Q10 names the escalation steps but drops both deadlines (14 days, 1 year). Q24 explains how a claim reduces the Silver bonus but not how it accrues. Q32, a comparison across plans, misses Platinum and Bronze's 20% co-payment. Multi-document questions suffer most (7 of 10 scored partial).
- **Refused part of a question it could answer: 2 of 13** (Q23, Q25). These are the two false refusals behind the 0.714 precision. Q25 found the ₹1,00,000 caesarean limit but would not subtract it from the ₹1,60,000 bill, because the prompt forbids inferring numbers. I saw the same pattern in live traffic: "Are pre-existing diseases covered on Gold?" was refused although the general 36-month rule was retrieved, because no source said "Gold". I traced it from the request's trace ID to the retrieved documents in under a minute.
- **The needed document was not retrieved: 2 of 13** (Q26, Q35). Q26 needed the Silver and Bronze plan documents for the knee-replacement sub-limits; Q35 needed the out-patient document for the dental check-up.

The judge marked 3 answers unfaithful (Q04, Q20, Q42). Q04 and Q20 state a condition more broadly than the source; on reading, Q42 matches the source and is likely a judge false positive.

## 4. What it costs

Generation is about 95% of the cost; the query embedding is a fraction of a cent. Prices are from the toolkit's price table for gemini-3.7-flash and gemini-embedding-001.

| | Without cache | With the 14% hit rate I measured |
|---|---|---|
| Per query | $0.0026 | $0.0022 |
| Per 1,000 queries | $2.60 | $2.24 |
| Per year at 10,000/day (3.65 M queries) | about $9,500 | about $8,200 |

The hit rate comes from my own test traffic, not real customers, so the cached column is illustrative. Tools mode costs $0.003–0.02 per question and is not included. Refusals are not cheaper: an off-topic question costs about $0.002 because the model still reads eight passages before declining. Hosting and staff costs are excluded.

## 5. How fast it is

Measured from the traces of 34 uncached requests through the service:

| Stage | p50 | p95 | Share of time |
|---|---|---|---|
| Embed the question (API call) | 1,009 ms | 1,857 ms | 16% |
| Search 214 passages | 2 ms | 10 ms | 0% |
| Generate the answer | 3,877 ms | 7,874 ms | 83% |
| Guards and validation | 11 ms | 97 ms | 1% |
| Total | 4,712 ms | 8,321 ms | |

Repeated questions are much faster: 0–1 ms from the exact cache, and about 1.1 s from the semantic cache (it still embeds the new wording). The uncached p95 of 8.3 s misses the 6 s target. The model spends 80–90% of its output on hidden reasoning before it writes a three-sentence answer, so the delay is thinking, not writing. For the same reason streaming did not help: the first word arrived at 97–99% of the total time (3.2 s of 3.3 s; 5.6 s of 5.6 s), against a 1.5 s target.

## 6. What it is not safe for

It should not be relied on for a coverage or claim decision without human review. It has not been seen to state a wrong central fact, but 9 of 40 answers left out a condition that was in the source, and the omitted detail is often the one that matters: a deadline, a co-payment, a waiting period. A customer who acts on an incomplete answer can miss a deadline with a correct-looking, fully cited answer in hand.

It is also not safe for:

- Plan-specific versions of general rules, or any answer that needs arithmetic. It refuses these rather than applying the rule or doing the sum.
- Rules from before 2026. The archived 2024 documents are excluded, so it cannot explain what changed.
- A document store that untrusted people can edit. In Lab 6 a fake but official-looking circular stating a 60-day claim window passed all five guard layers in 3 of 3 runs, and customers were told the wrong deadline. The guards stop instructions, not false facts.
- Premium quotes a customer will act on. Tools-mode answers carry no citation, take 10–13 s, and the policy-lookup tool has no check that the policy belongs to the person asking, so this mode must not face customers without authentication.
- Instant answers. Uncached p95 is 8.3 s.

Known weaknesses of the service itself:

- Semantic cache: its 0.92 threshold rests on 20 hand-written question pairs. A wrong hit returns another plan's answer with valid-looking citations, and nothing downstream can detect it. A cached refusal is also served to every rewording.
- Streaming: when the final validation fails, the text is withdrawn after the customer has already seen it.
- Single process: per-request cost attribution is only correct with one worker, the response cache is unbounded and in memory, there is no authentication, and the 429 message passes internal budget details to the caller.

## 7. What I would do next

1. **Revise the answer prompt for completeness and general rules.** Ask for every limit, condition and deadline the sources attach to the answer, allow rules stated for all policies to apply to a named plan, and allow arithmetic on stated figures. This targets 11 of the 13 partial answers and both false refusals. I expect correctness to rise by about 0.05 (half the omissions recovered) and refusal precision to reach 1.0, at the risk of lower faithfulness. One golden-set run (about $0.15) decides it, and the gate catches a faithfulness drop.
2. **Try the SMALL model tier for generation.** It has no hidden reasoning step, so I expect p95 under 3 s and a first word within the 1.5 s target, at an unknown quality cost (between −0.10 and 0 correctness). A paired run with a McNemar test (about $0.30) decides it. Today the system misses its latency target on every slow question.
3. **Accept documents only from a signed, versioned list.** This closes the false-document attack (N01), the one attack that beat every layer, and makes the refusal-rate alert's remedy (roll back the changed document) a one-step operation. It needs no model change, and it protects the scenario with the highest harm: confidently wrong deadlines.