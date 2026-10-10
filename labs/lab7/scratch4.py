#!/usr/bin/env python3
"""Lab 7 C3 — send a realistic mix of traffic through /ask so the dashboard has data.

    python labs/lab7/scratch4.py          (with the service running on :8000)

New questions, exact repeats, a near-identical rewording (semantic cache),
off-topic questions (refusals) and one tools-mode question. About 18 paid
requests, roughly $0.05 and 2 minutes.
"""
import requests

API = "http://localhost:8000/ask"

TRAFFIC = [
    ("rag", "What is the sum insured on the Bronze plan?"),
    ("rag", "Is there a co-payment on the Silver plan?"),
    ("rag", "What is the waiting period for maternity benefits?"),
    ("rag", "Can I get car insurance from Aurora?"),                 # off-topic -> refusal
    ("rag", "How long is the initial waiting period after buying a policy?"),
    ("rag", "what is the sum insured on the bronze plan"),           # exact repeat
    ("rag", "What documents do I need for a reimbursement claim?"),
    ("rag", "Is day-care treatment covered?"),
    ("rag", "What is the ambulance limit on Gold?"),
    ("rag", "What is the capital of France?"),                       # off-topic -> refusal
    ("rag", "Does the senior citizen plan have a co-payment?"),
    ("rag", "How do I renew my policy?"),
    ("rag", "How can I renew my policy?"),                           # near-identical -> semantic?
    ("rag", "What is the waiting period for maternity benefits?"),   # exact repeat
    ("rag", "How is the no-claim bonus applied at renewal?"),
    ("rag", "Can I add my parents to the Gold plan?"),
    ("rag", "What is the free-look period?"),
    ("rag", "Is AYUSH treatment covered?"),
    ("rag", "Does Aurora offer life insurance?"),                    # off-topic -> refusal
    ("rag", "Are pre-existing diseases covered on Gold?"),
    ("rag", "What documents do I need for a reimbursement claim?"),  # exact repeat
    ("tools", "What would a Silver plan cost for a 35-year-old?"),
    ("rag", "Is there a limit on cataract surgery under Gold?"),
    ("rag", "Who do I contact for a cashless pre-authorisation?"),
    ("rag", "Is the free-look period 15 days?"),
]

for i, (mode, q) in enumerate(TRAFFIC, 1):
    r = requests.post(API, json={"question": q, "mode": mode}, timeout=120)
    d = r.json() if r.ok else {}
    print(f"{i:>2} {r.status_code} {d.get('latency_ms', 0):>8.0f} ms  ${d.get('cost_usd', 0):.4f}  "
          f"cached={str(d.get('cached', '-')):<5} refused={str(d.get('refused', '-')):<5} {q}")