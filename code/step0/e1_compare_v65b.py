"""v6.5b verdict: the UNCHANGED v6.5 rule (step0/e1_compare_v65.py, sha256 2a2f5ec5...; asserted) on paths S and R, re-run after
patch_q2c.py added recording-only code (hashes of plan, targets, indices, mask; no RNG draw, no training op) to base6s.train. Only the run
directory and result-name prefix differ. Written 20 Nov 07:3x before the runs."""
import hashlib, json, os, sys
P = os.path.dirname(os.path.abspath(__file__)) + "/e1_compare_v65.py"; src = open(P).read()
assert hashlib.sha256(src.encode()).hexdigest().startswith("2a2f5ec5eb483fd0"), "v6.5 rule changed"
for a, b in [('hands-free-mobile/e1v65"', 'hands-free-mobile/e1v65b"'), ('f"{OUT}/e1v65_{p}_{r}_s3.json"', 'f"{OUT}/e1v65b_{p}_{r}_s3.json"'),
             ('if __name__ == "__main__":', 'if False:')]:
    assert src.count(a) == 1, a; src = src.replace(a, b)
g = {"__name__": "v65b"}; exec(compile(src, P, "exec"), g)
R = {p: g["verdict"](p, *g["PATHS"][p]) for p in ("S", "R")}; R["ALL_PASS"] = all(v["PASS"] for v in R.values())
json.dump(R, open(g["E"] + "/e1_verdict_v65b.json", "w"), indent=1); print(json.dumps(R, indent=1)); sys.exit(0 if R["ALL_PASS"] else 1)
