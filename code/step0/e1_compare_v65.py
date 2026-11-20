"""E1 harness verdict v6.5, per path. RULE WRITTEN 20 Nov 04:0x BEFORE any v6.5 run started (runner step0/e1_cpu_v65.sh, fresh dir e1v65).
WHY A NEW TEST: the v6.4 test (e1_compare.py, TS only) is FAIL as written: its condition (3) required the 'e' trace lines to show training
mode, but base6s.train writes the 'e' line right after evaluate() (which calls m.eval()) and before the next pass's m.train() (base6s.py:215),
so every 'e' line reads False in all four runs. All its other conditions held (rc, counts, eval events, A==B, A==C, D diverges at pass 1 and
later, final scores equal, D==C before injection, event order). That FAIL stands; this is a new test on fresh runs, not an amendment.
Change of (3): every 'u' line (each optimizer update) must show all modules in training mode, AND every 'e' line must show False (proves the
evaluation really ran in eval mode); the property that matters (no update happens in eval mode) is kept and the e line is now checked too.
PATHS: TS (base6sy.py TS, 0.5 h; arms A B C D), S (base6s.py S, 0.5 h; A C D), R (base6s_R.py, RWIRE_SEED 3, 0.5 h; A C D),
P1 (base6s.py P, 1 h; A C D), G1 (base6s.py G, 1 h; A C D). k = updates per pass of the dry-built plan (0.5 h: 9, 1 h: 17).
PASS for a path needs ALL: (0) rc 0 every arm; (1) u lines numbered 1..6k in order and the count of u lines before the first p line == k,
p lines 1..6; (2) e events exactly [6] for A (and B) and 1..6 for C, D; (3) every u line True, every e line False; (4) TS only: A == B on
every u and p line and in full event order; (5) A == C on every u and p line; (6) D first differs from A on the p line of pass 1 AND on at
least one u line after update k; (7) final validation results (incl. per-item rows) of A == C (== B for TS); (8) D == C on u lines 1..k;
(9) event order C == D, and A's order == C's order with the e lines of passes 1-5 removed. Exit 1 unless every path PASSES. A PASS covers
that path's CPU code at that data size with the 6-pass prefix only (not CUDA, not the production cadence)."""
import os
import json, os, sys
E = os.environ.get("SCRATCH_DIR", "scratch") + "/e1v65"; OUT = "results/phones_out/base6"
PATHS = dict(TS=("ABCD", 9), S=("ACD", 9), R=("ACD", 9), P1=("ACD", 17), G1=("ACD", 17))


def verdict(p, arms, k):
    st = {}
    for l in open(f"{E}/{p}/status.txt"):
        if " rc " in l: st[l.split()[0]] = l.split()[2]
    T = {r: [l.split() for l in open(f"{E}/{p}/trace_{r}.txt").read().splitlines()] for r in arms}
    U = {r: [l for l in t if l[0] == "u"] for r, t in T.items()}; Pp = {r: [l for l in t if l[0] == "p"] for r, t in T.items()}
    Ev = {r: [int(l[1]) for l in t if l[0] == "e"] for r, t in T.items()}
    seq = {r: [(l[0], l[1]) for l in t if l[0] in "uep"] for r, t in T.items()}
    c = {}
    c["rc"] = all(st.get(r) == "0" for r in arms)
    kk = {r: next((i for i, l in enumerate(T[r]) if l[0] == "p"), None) and sum(1 for l in T[r][:next(i for i, l in enumerate(T[r]) if l[0] == "p")] if l[0] == "u") for r in arms}
    c["counts"] = all([int(l[1]) for l in U[r]] == list(range(1, 6 * k + 1)) and [int(l[1]) for l in Pp[r]] == list(range(1, 7)) and kk[r] == k for r in arms)
    c["eval_events"] = all(Ev[r] == ([6] if r in "AB" else list(range(1, 7))) for r in arms)
    c["u_training_e_eval"] = all(l[-1] == "True" for r in arms for l in U[r]) and all(l[-1] == "False" for r in arms for l in T[r] if l[0] == "e")
    if "B" in arms: c["A_eq_B"] = U["A"] == U["B"] and Pp["A"] == Pp["B"] and seq["A"] == seq["B"]
    c["A_eq_C"] = U["A"] == U["C"] and Pp["A"] == Pp["C"]
    fd = next((i for i, (x, y) in enumerate(zip(Pp["A"], Pp["D"])) if x != y), None)
    c["D_diverges_pass1_and_later"] = fd == 0 and any(a != d for a, d in zip(U["A"][k:], U["D"][k:]))
    def res(r):
        f = f"{OUT}/e1v65_{p}_{r}_s3.json"; return json.load(open(f)).get("sets") if os.path.exists(f) else None
    rs = {r: res(r) for r in arms if r != "D"}
    c["final_equal_incl_items"] = rs["A"] is not None and all(v == rs["A"] for v in rs.values()) and all("items" in v for v in rs["A"].values())
    c["D_eq_C_before_injection"] = len(U["D"]) >= k and U["D"][:k] == U["C"][:k]
    c["event_order"] = seq["C"] == seq["D"] and seq["A"] == [x for x in seq["C"] if not (x[0] == "e" and int(x[1]) < 6)]
    return dict(checks=c, D_first_diff_pass_index=fd, eval_events_seen=Ev, PASS=all(c.values()))


if __name__ == "__main__":
    V = {p: verdict(p, *a) for p, a in PATHS.items()}; V["ALL_PASS"] = all(v["PASS"] for v in V.values())
    json.dump(V, open(f"{E}/e1_verdict_v65.json", "w"), indent=1); print(json.dumps(V, indent=1)); sys.exit(0 if V["ALL_PASS"] else 1)
