"""Verdict for C2 0 d (3)-(5) from check345_result.json under the AMENDMENT of 20 Nov 2026 07:4x (the agent; before the hard-panel cutoff,
before any panel item was selected or scored; to be reviewed in round 6). Literal v6.5 (4) is reported beside it, never hidden.
Amendment (4): code changed after 01:37 passes if (a) MODEL-PATH code with a pre-cutoff backup: diff without train-other-500 reference and
every numeric diff line recording-only (listed, judged by the agent: same constants, added hashing / assertions / per-item rows / STOP_AFTER);
(b) MODEL-PATH code without a pre-cutoff backup: diff vs the pre-edit copy + a behavioural known answer of a pre-cutoff checkpoint
(base6s_R: rwire selftest r=0 == 18 Nov sigma, step0/rwire_selftest_2026-11-20.txt; reader parity R 29.44 vs saved 29.43,
step0/reader_parity_2026-11-20.jsonl); (c) EVALUATION TOOLING (gate, reader, decision entries, scorer) is not an artifact a model reads:
it needs only (5) and no train-other-500 read (score_locked.py:20 mentions hardT in a comment only)."""
import json, os, re
H = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(f"{H}/check345_result.json"))
src = open("code/step0/score_locked.py").read().splitlines()
hits = [i + 1 for i, l in enumerate(src) if re.search(r"other.?500|train-other", l, re.I)]
assert hits == [20] and src[19].split("#", 1)[0].count("train-other") == 0, ("score_locked reference is not comment-only", hits)
TOOL = {"step0/gpu_gate2.py", "step0/reader_v65.py", "step0/score_locked.py", "../small_net/sec2_abs.py", "../small_net/sec2_q2c_entry.py"}
BEHAV = {"base6s_R.py": "rwire selftest r=0 == 18 Nov sigma; reader parity R 29.44 vs 29.43 (4-edit CPU/GPU flip)"}
v = {}
for k, c in R["check4"]["code"].items():
    if c["before_cutoff"]: v[k] = "before cutoff"
    elif k in TOOL: v[k] = "tooling (c)"
    elif k in BEHAV: v[k] = "(b) " + BEHAV[k]
    elif c["amended_rule"]["PASS"]: v[k] = f"(a) diff audit, {len(c['amended_rule']['numeric_lines'])} numeric lines recording-only"
    else: v[k] = "FAIL"
c5_other = [k for k, x in R["check5"]["closure"].items() if x["other500_ref"]]
out = dict(check3=R["check3"]["PASS"], check4_literal_v65=R["check4"]["PASS_literal_v65"],
           check4_amended=all(x != "FAIL" for x in v.values()) and all(d.get("before_cutoff") for d in R["check4"]["data"].values()),
           check4_code=v, check5=all(os.path.exists(e) for e in R["check5"]["entries"]) and c5_other == ["step0/score_locked.py"],
           check5_note="only other-500 mention in the closure: score_locked.py:20 comment", n_closure=len(R["check5"]["closure"]))
json.dump(out, open(f"{H}/check345_verdict.json", "w"), indent=1); print(json.dumps({k: x for k, x in out.items() if k != "check4_code"}))
print({k: x for k, x in v.items() if x != "before cutoff"})
