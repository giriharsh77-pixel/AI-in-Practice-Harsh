import json, sqlite3, sys
from pathlib import Path

db = Path(sys.argv[1] if len(sys.argv) > 1 else "ci_cache/calls.sqlite3")
rows = sqlite3.connect(db).execute(
    "SELECT request, response FROM calls WHERE kind = 'chat'").fetchall()

gens = []
for req, resp in rows:
    req, resp = json.loads(req), json.loads(resp)
    if req.get("model") != "gemini/gemini-3.7-flash":
        continue                                   # skip judge calls (LARGE tier)
    last = req["messages"][-1]["content"]
    q = last.split("Question:")[-1].split("Answer with citations")[0].strip()
    u = resp["usage"]
    gens.append((u["latency_ms"], u["completion_tokens"], len(resp["text"]), q[:70]))

gens.sort(reverse=True)
print(f"{len(gens)} generation calls\n")
print(f"{'ms':>7} {'out_tok':>8} {'chars':>6}  question")
for ms, tok, chars, q in gens[:10]:
    print(f"{ms:>7.0f} {tok:>8} {chars:>6}  {q}")