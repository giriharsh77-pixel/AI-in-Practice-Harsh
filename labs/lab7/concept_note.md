# Lab 7 Concept Note - Nuances of shipping a RAG System

Sushruta Nandy, Roll # 26, MSAI, Plaksha University. 

AI in Practice I

Module 1, Option 2 (Lab 7, individual).

## The problem

By the end of Lab 6 I had a pipeline that answered Aurora policy questions well on my laptop: correctness 0.81, faithfulness 0.93, every citation valid, and every attack in the red-team suite blocked. None of that made it a product. A product has to answer over HTTP, tell a caller what to do when something fails, cost a known amount, explain why any single request was slow or wrong, and refuse to let a change that makes it worse reach users. Lab 7 was about building those properties around a pipeline I mostly did not change. The measured results are in `EVALUATION_REPORT.md`; this note explains the ideas behind the design and the trade-offs I chose.

## The architecture

A question arrives at a FastAPI service. It first meets an exact cache (the normalised question) and then a semantic cache (the question's embedding). On a miss it goes through the same pipeline as Labs 4–6: dense retrieval over 214 chunks, a filter that drops retrieved text that looks like an injected instruction, generation with a delimited, numbered context, citation validation in code, and an output filter that removes leaked prompt text, foreign links and personal data. The response carries the answer, citations the user can open, the cost of this request and a trace ID. Every request writes a trace; `/metrics`, the dashboard and a refusal-rate alert are all computed from those traces. A regression gate replays the golden set from a committed cache in CI on every push.

## Measure first, then fix

My first change was not to the service. The Lab 5 pipeline had a p95 of 14.5 s, and I traced it to a single cause: 14 of 45 answers hit the 600-token limit, because the reasoning model's hidden thinking counts against it, and each one paid for a second full call. Raising the limit removed every retry, cut cost by 28% and p95 by 38%, with no change in quality. The prediction I wrote down before running missed in an instructive way: p95 was still 9 s. With the retries gone, the next bottleneck became visible. Most of each call is hidden reasoning (80–90% of output tokens, for answers of two or three sentences), and provider speed varies threefold for the same work. Breaking latency down by stage later confirmed it: generation is 83% of request time, the query embedding 16%, and everything I wrote, search, guards and validation, about 1%. Latency work on this system is model work. I chose to keep the model the quality was measured on and report the latency miss, rather than switch to a faster model with no quality evidence.

## Caching: cheap when exact, dangerous when semantic

The exact cache is almost free of risk: if the normalised question is the same, the answer is the same, and it returns in under a millisecond. The semantic cache is the interesting one, because it decides that two different questions deserve the same answer. I measured where that breaks using 10 genuine paraphrases and 10 traps, pairs that differ by one word but have different answers. The two groups overlapped completely. The worst trap, "no-claim bonus on Gold" against "on Silver", scored 0.888, higher than 7 of the 10 genuine paraphrases. Embeddings measure how similar the wording is, not whether the answer is the same, and a plan name or a year barely moves the vector. At 0.85, six of ten traps would have been served the wrong plan's answer, with valid citations to the wrong document, so nothing downstream could catch it. I shipped at 0.92, which catches only near-identical rewordings. A missed hit costs a fifth of a cent; a wrong hit gives a customer the wrong plan's terms. I also record the best similarity on every request, so real traffic can show whether the threshold can safely come down.

## Streaming hides writing, not thinking

Streaming is supposed to cut perceived latency, but it created a real design problem: citations can only be validated once the whole answer exists, and by then it has been sent. I chose to stream the prose, hold the citations until the end, and retract the text if validation fails, because first-attempt failures had fallen to 0 of 45. The measurement then made the choice almost moot: the first word arrived at 97–99% of the total time. The model thinks silently and then writes three sentences in about 100 ms, so there was nothing left to hide. Streaming only helps when the visible answer is long relative to the thinking. The cost of my choice is still real: output checks become "retract after showing" instead of "block before showing".

## Observability means answering one question

The test of tracing is whether you can answer "why did request X take 9 seconds?", and "why did it refuse?", from traces alone. In my own test traffic, "Are pre-existing diseases covered on Gold?" was refused although the answer exists. From its trace ID I could see one model call of 6.1 s, a normal stop, nothing filtered and eight sources. The retrieved-sources list showed the general pre-existing-conditions document was there, and the Gold document said nothing on the subject. So it was neither a retrieval failure nor a pipeline fault: the prompt forbids inferring, and no source said "Gold". The alert I chose follows the same idea. A broken index does not raise errors or slow anything down; it quietly turns answers into refusals. So the alert watches the refusal rate, not latency or errors.

## A gate you have not seen fail

The regression gate runs the 45 golden questions offline in CI, replaying about 130 model calls from a 5 MB committed cache in one second with no API key. A replay costs nothing and takes no time, so I computed cost and latency from the values the cache recorded when each call was first made. That catches changes to the requests, though not a slower provider. I set each threshold just past today's value, so the gate fails when the system gets worse rather than restating targets it already misses. To prove it works, I passed one chunk to the model instead of eight. The build went red on correctness (0.84 to 0.58), refusal precision (0.71 to 0.24), retrieval and latency. In the same run, faithfulness rose to 1.00 and cost fell 27%, because a system that refuses more makes fewer claims and reads less. A gate on faithfulness or cost alone would have scored the break as an improvement.

## Personal Takeaways

Every important result in this lab came from measuring something I thought I already understood: the slow tail, the cache threshold, streaming, the gate. The system's honest boundary follows from those measurements. It does not state wrong facts on the test set, but it leaves out conditions (9 of 40 answers), refuses plan-specific versions of general rules, and trusts its documents completely, so it should support a human answering a coverage question, not replace one. My next steps, ranked, are a prompt revision aimed at those omissions and false refusals, a trial of the faster model tier against the gate, and a signed document list to close the one attack that beat every guard layer.