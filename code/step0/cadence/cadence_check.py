"""C9 3 cadence-position check (v6.5, 20 Nov 2026, the agent). Takes the evaluation-trigger expression VERBATIM from each trainer's source
(the `if` line after `_evn = os.environ.get("EVAL_N")`), evaluates it for every pass of the Tranche 1 schedules with EVAL_N = 30 and
with the old rule, and asserts: exactly EVAL_N events when EVAL_N <= epochs (else one per pass), the last pass always evaluated, gaps
between events differ by at most 1 pass from epochs / EVAL_N. Writes cadence_result.json."""
import os
import ast, json, os, re
D = "code"
SCHED = {"Q2c 0.5 h": 1836, "Q1 1 h": 972, "Q4 5.4 h full": 60, "Q4 half": 90, "E1 prefix plan 0.5 h": 60, "old 5.4 h": 60, "1 h old": 108}
out = {}
for f, ev_every in (("base6s.py", "B6.EVAL_EVERY"), ("base6.py", "EVAL_EVERY")):
    L = open(f"{D}/{f}").read().splitlines(); i = next(k for k, l in enumerate(L) if '_evn = os.environ.get("EVAL_N")' in l)
    cond = L[i + 1].strip(); assert cond.startswith("if ") and cond.endswith(":"), cond; expr = cond[3:-1]
    ast.parse(expr, mode="eval")
    EE = 6  # EVAL_EVERY in both trainers (asserted below from source)
    assert "EVAL_EVERY = int(os.environ.get(\"EVAL_EVERY\", 6))" in open(f"{D}/base6.py").read(), "EVAL_EVERY default is not 6 (env EVAL_EVERY must be unset in production)"
    res = {}
    for name, epochs in SCHED.items():
        for evn in ("30", None):
            env = dict(_evn=evn, epochs=epochs, int=int, B6=type("B", (), dict(EVAL_EVERY=EE)), EVAL_EVERY=EE)
            pos = [ep + 1 for ep in range(epochs) if eval(expr, env, dict(ep=ep))]
            n = int(evn) if evn else None
            gaps = [b - a for a, b in zip([0] + pos, pos)]
            ok = pos[-1] == epochs and (len(pos) == min(n, epochs) if n else len(pos) == epochs // EE + (epochs % EE != 0))
            if n and n <= epochs: ok &= max(gaps) - min(gaps) <= 1 and abs(max(gaps) - epochs / n) < 1
            res[f"{name} EVAL_N={evn}"] = dict(epochs=epochs, events=len(pos), first=pos[:3], last=pos[-3:], gap_min=min(gaps), gap_max=max(gaps), ok=bool(ok))
    out[f] = dict(expression=expr, results=res, ALL_OK=all(r["ok"] for r in res.values()))
out["ALL_OK"] = all(v["ALL_OK"] for v in out.values() if isinstance(v, dict))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cadence_result.json"), "w"), indent=1)
for f in ("base6s.py", "base6.py"):
    for k, r in out[f]["results"].items(): print(f, k, r["events"], r["gap_min"], r["gap_max"], r["ok"])
print("ALL_OK", out["ALL_OK"])
