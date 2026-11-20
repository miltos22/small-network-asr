"""E1 harness verdict v6.4 (Part C v6.4 C5 4; round 4 C5.4). Rule written 20 Nov 01:59 BEFORE the v6.4 runs started. Traces from step0/e1_cpu.sh.
Trace lines: 'u n loss aux lr all_modules_training', 'e pass all_modules_training' (an evaluation happened after that pass),
'p pass model_hash whole_optimizer_hash scheduler_hash numpy_rng_hash torch_rng_hash'.
PASS needs ALL of: (0) status rc 0 for A, B, C, D; (1) each trace: updates numbered 1..54 in order, passes 1..6 in order; (2) evaluation events
exactly at pass 6 for A and B and at passes 1-6 for C and D (so the tested contrast really happened); (3) every u and e line shows all modules
in training mode; (4) A == B on every u and p line (CPU determinism); (5) A == C on every u and p line (EVAL_N changed only where curves are
taken); (6) D differs from A first on the p line of pass 1 (the injected draw) AND on at least one u line after update 9 (it changed later
training); (7) the final validation scores in the result jsons of A, B, C are identical. Anything else = FAIL; E1 rates are then not used
until fixed and retested (Part C C5 4). A PASS covers the TESTED path only: CPU, base6s.train via base6sy.py arm TS, 0.5 h, a 6-pass prefix.
AMENDMENT 20 Nov 02:44 (round 5 C5.4, written while only run A had started; no trace read; tightens only): (8) D == C on u lines 1-9 (identical
before the injected draw after pass 1); (9) the full event order (sequence of u / e / p line types with their numbers) of A == B and of C == D
(interleaving, not only subsequences); (10) the script exits 1 unless PASS (fail-closed receipt)."""
import os
import json, sys, os
E = os.environ.get("SCRATCH_DIR", "scratch") + "/e1v64"; OUT = "results/phones_out/base6"
st = {l.split()[0]: l.split()[2] for l in open(f"{E}/status.txt") if " rc " in l}
T = {r: [l.split() for l in open(f"{E}/trace_{r}.txt").read().splitlines()] for r in "ABCD"}
U = {r: [l for l in t if l[0] == "u"] for r, t in T.items()}; Pp = {r: [l for l in t if l[0] == "p"] for r, t in T.items()}
Ev = {r: [int(l[1]) for l in t if l[0] == "e"] for r, t in T.items()}
c0 = all(st.get(r) == "0" for r in "ABCD")
c1 = all([int(l[1]) for l in U[r]] == list(range(1, 55)) and [int(l[1]) for l in Pp[r]] == list(range(1, 7)) for r in "ABCD")
c2 = Ev["A"] == [6] and Ev["B"] == [6] and Ev["C"] == list(range(1, 7)) and Ev["D"] == list(range(1, 7))
c3 = all(l[-1] == "True" for r in "ABCD" for l in T[r] if l[0] in "ue")
c4 = U["A"] == U["B"] and Pp["A"] == Pp["B"]; c5 = U["A"] == U["C"] and Pp["A"] == Pp["C"]
fd = next((i for i, (x, y) in enumerate(zip(Pp["A"], Pp["D"])) if x != y), None)
c6 = fd == 0 and any(a != d for a, d in zip(U["A"][9:], U["D"][9:]))
def sets(r):
    p = f"{OUT}/e1v64_{r}_s3.json"; return json.load(open(p)).get("sets") if os.path.exists(p) else None
c7 = sets("A") is not None and sets("A") == sets("B") == sets("C")
c8 = len(U["D"]) >= 9 and len(U["C"]) >= 9 and U["D"][:9] == U["C"][:9]
seq = {r: [(l[0], l[1]) for l in t if l[0] in "uep"] for r, t in T.items()}
c9 = seq["A"] == seq["B"] and seq["C"] == seq["D"]
V = dict(rc=c0, counts=c1, eval_events=c2, eval_events_seen={r: Ev[r] for r in "ABCD"}, training_mode=c3, A_eq_B=c4, A_eq_C=c5,
         D_first_diff_pass_index=fd, D_diverges_later=c6, final_scores_equal=c7, D_eq_C_before_injection=c8, event_order_equal=c9, PASS=all([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]),
         scope="CPU, base6s.train via base6sy.py TS, 0.5 h draw 0, seed 3, 6-pass prefix of the 60-pass schedule")
json.dump(V, open(f"{E}/e1_verdict.json", "w"), indent=1); print(json.dumps(V, indent=1))
sys.exit(0 if V["PASS"] else 1)
