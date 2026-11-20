import os
#!/usr/bin/env python3
"""Per-run GPU ledger for MASTER_PLAN Part C v6.3 (the agent, 20 Nov 2026; review round 3 C5.1 / C5.3 fixes). One row per run; every sum is computed here, never typed.
Replaces ledger_v61.py (one price for every small run; wrong per arm, review round 2 C5.1 / C5.4 and the agent's 01:16 correction C5 4a).

PRICES come from the measured timing table step0/timing_table_2026-11-20.txt (built from every logs_base6/*.log with >= 20 epochs):
  per optimizer step, in two arm classes, median (planning price) and max (worst measured, the sensitivity column):
    plain = base6s arms G, P, S (incl. R, S0, SI, P2, A1-A3 variants), base6.py 5.4 h runs
    tree  = base6s tree variants TS, TN, TP, TSN (the structured readouts; his tree arms)
  The GPU is shared with other agents; the spread inside one arm (TS 0.101-0.229 s/step) is contention, so the median is a planning
  price and the max is shown beside it. Eval cost per curve point: class median of the table's eval_s column. Setup + final scoring:
  396 s/run (MEASURED-DERIVED: 1 h batch 200 min wall / 4 runs vs its loops). Curve points per run: EVAL_N = 30 (E1) for every run.
  ELIGIBLE LOGS (v6.3): the full planned schedule (180 passes at 5.4 h or 972 at 1 h) completed in ONE log (n_eval = 31 or 163; no
  .resumed file, no interrupted part); 324-pass screens excluded. WORST column (v6.3) = max step rate AND max eval time per curve point AND
  2 x setup (the 396 s setup is derived from one batch, so its worst is a stated sensitivity, not a measurement).
  Updates: 1 h = 16,524 (MEASURED 972 passes); 0.5 h = 16,524 at 1,836 passes (DRY-BUILT, step0/batchplan_dry.py: 9 batches per pass,
  known answer 16,524 for the logged 1 h run; v6.2's 1,944 passes would give 17,496);
  5.4 h = 15,091 (MEASURED 180 passes); half schedule = half. 100 h = 3.88 h/run (MEASURED, Part A 4b); 15 % proxy = 0.15 x 3.88
  (INFERRED, linear). CIFAR 100 %: S = 0.162 h (MEASURED max of 532-582 s), P and G_mix = 0.162 x 1.2 =
  0.194 h (untimed, the 1.2 is coded); CIFAR 10 % = 0.04 h (INFERRED). E1 and the locked scorer run on the CPU (0 GPU-h).
Rows: arm, domain, data_size, config_id (filled at the freeze), seed, stage, optimizer_updates, curve_points, price_class,
  expected_gpu_h (median), worst_gpu_h (max), reuse_source, scope (R = recommended, H = headroom, F = deferred, X = spent).
"""
import json, re, statistics as st, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "timing_table_2026-11-20.txt")
PLAIN = {"G", "P", "S", "R", "S0", "SI", "P2", "A1", "A2", "A3", "GT", "G0", ""}; TREE = {"TS", "TN", "TP", "TSN"}
rate = {"plain": [], "tree": []}; evs = {"plain": [], "tree": []}
for line in open(TAB).read().splitlines()[1:]:
    f = [x.strip() for x in line.split("|")]
    if len(f) < 7: continue
    name, arm = f[0], f[1].split()
    var = arm[2] if len(arm) > 2 else ""; script = arm[0]
    th, ep, spe, tot = f[2].split(); pure, sps = map(float, f[3].split()); nev, evc = f[4].split()
    if int(ep) not in (180, 972) or int(nev) != {180: 31, 972: 163}[int(ep)] or "resumed" in name: continue  # v6.3 eligible set
    cls = "tree" if var in TREE else "plain"
    rate[cls].append(sps); e = float(evc)
    if e > 0: evs[cls].append(e)
P = {c: dict(med=st.median(rate[c]), max=max(rate[c]), n=len(rate[c]), eval_med=st.median(evs[c]), eval_max=max(evs[c])) for c in rate}
SETUP_S, EVAL_N = 396, 30
UPD = {"0.5h": 16524, "1h": 16524, "5.4h": 15091}

def small_h(cls, size, frac=1.0, margin=1.0, which="med"):
    ev, su = (P[cls]["eval_med"], SETUP_S) if which == "med" else (P[cls]["eval_max"], 2 * SETUP_S)
    return (UPD[size] * frac * P[cls][which] * margin + EVAL_N * ev + su) / 3600

rows = []
def add(group, arm, domain, size, seeds, stage, cls=None, frac=1.0, margin=1.0, fixed=None, reuse="", scope="R", note=""):
    for sd in seeds:
        if fixed is not None: med = worst = fixed; upd = None; cp = None
        else:
            med, worst = small_h(cls, size, frac, margin, "med"), small_h(cls, size, frac, margin, "max")
            upd = int(UPD[size] * frac); cp = EVAL_N
        rows.append(dict(group=group, arm=arm, domain=domain, data_size=size, config_id="(freeze)", seed=sd, stage=stage,
                         optimizer_updates=upd, curve_points=cp, price_class=cls or "measured/fixed", expected_gpu_h=round(med, 3),
                         worst_gpu_h=round(worst, 3), reuse_source=reuse, scope=scope, note=note))

# ---- Step 0 (authorized allowance 4-5 GPU-h, B 10 row 'Step 0')
add("Step0", "spent: CIFAR S x3 + ResNet-32 s1 + smokes", "both", "-", [None], "spent", fixed=0.85, scope="X")
add("Step0", "E1 harness A/B/C + broken D on the CPU (step0/e1_cpu.sh)", "speech", "0.5h", [3], "calibration", fixed=0.0, note="CPU")
add("Step0", "locked scorer (step0/score_locked.py) on the CPU", "speech", "-", [None], "scoring", fixed=0.0, note="CPU")
add("Step0", "Shared baseline (WN recipe) on the FROZEN geometry + split, run once", "speech", "5.4h", [0, 1, 2], "calibration", "plain")
# ---- Q2 (c): his tree readout S (semantic, base6s.py arm S) vs shuffled wiring R (base6s_R.py, RWIRE_SEED = seed r), 5 paired seeds.
#      v6.3: price class PLAIN (S and R are plain-class arms in the timing table; v6.2 wrongly used the TS/TSN top-only recipe and the tree price)
for a in ("semantic S", "shuffled R"):
    add("Q2c", a, "speech", "0.5h", [0, 1, 2, 3, 4], "primary (alpha .05/3)", "plain")
    add("Q2c", a, "speech", "1h", [0, 1, 2, 3, 4], "robustness", "plain", scope="H2")
# ---- Q4 at 5.4 h on the plain Shared architecture (v6.2 freeze choice, the agent; reprice before the freeze if WH needs specialist heads)
for a in ("R", "D", "W", "WH"):
    add("Q4", a + " full", "speech", "5.4h", [0, 1, 2], "primary", "plain")
    add("Q4", a + " full", "speech", "1h", [0, 1, 2], "secondary", "plain", scope="F")
for a in ("W", "R"):
    add("Q4", a + " compressed half", "speech", "5.4h", [0, 1, 2], "primary (speed)", "plain", frac=0.5)
    add("Q4", a + " compressed half", "speech", "1h", [0, 1, 2], "secondary", "plain", frac=0.5, scope="F")
# ---- Q1 speech 1 h: 3x2 grid at seed 0 per arm; seeds 1-2 on the chosen setting; + replacement seed-0 run if the winner is not reusable
for a in ("P", "Shared"):
    for g in range(6): add("Q1", f"{a} grid cell {g}", "speech", "1h", [0], "co-primary low data (tuning)", "plain")
    add("Q1", f"{a} chosen setting", "speech", "1h", [1, 2], "co-primary low data", "plain", reuse="seed 0 = grid winner if identical in every field")
    add("Q1", f"{a} chosen setting, replacement seed 0", "speech", "1h", [0], "co-primary low data", "plain", scope="R", note="funded: exact reuse not assumed")
for g in range(6): add("Q1", f"speech G_mix grid cell {g}", "speech", "1h", [0], "mechanism (tuning)", "plain", scope="F")
add("Q1", "speech G_mix chosen setting", "speech", "1h", [1, 2], "mechanism", "plain", scope="F")
add("Q1", "speech G_mix chosen, replacement seed 0", "speech", "1h", [0], "mechanism", "plain", scope="F")
# ---- Q1 CIFAR: 10 % 3x2 grid + seeds 1-2 (+ replacement seed 0), 100 % 3 WD + seeds 1-2 (+ replacement seed 0), arms P, Shared, G_mix
for a in ("P", "Shared", "G_mix"):
    for g in range(6): add("Q1", f"CIFAR {a} grid cell {g}", "cifar", "10%", [0], "co-primary low data (tuning)", fixed=0.04)
    add("Q1", f"CIFAR {a} chosen", "cifar", "10%", [0, 1, 2], "co-primary low data", fixed=0.04, note="seed 0 replacement funded")
    for g in range(3): add("Q1", f"CIFAR {a} WD {g}", "cifar", "100%", [0], "co-primary full data (tuning)", fixed=(0.162 if a == "Shared" else 0.194))
    add("Q1", f"CIFAR {a} chosen", "cifar", "100%", [0, 1, 2], "co-primary full data", fixed=0.162 if a == "Shared" else 0.194, note="seed 0 replacement funded")
# ---- Q1 100 h package (deferred)
for a in ("P", "aug Shared"):
    for g in range(6): add("Q1", f"{a} 5.4 h grid cell {g}", "speech", "5.4h", [0], "calibration (proxy candidates)", "plain", scope="F")
    for cand in ("cand1", "cand2"): add("Q1", f"{a} 15 % proxy {cand} (matched seed 0 / order)", "speech", "100h", [0], "calibration", fixed=0.15 * 3.88, scope="F", note="linear extrapolation, not timed")
    add("Q1", f"{a} full", "speech", "100h", [0, 1, 2], "co-primary full data", fixed=3.88, scope="F")
# ---- Q8 screens (exploratory): a-c x 2 seeds on the plain base (x1.2 margin for the added part); ONE shared replacement base pair
for i in ("a cross links", "b accent prototypes + OR gate", "c sequence compartment"):
    add("Q8", i, "speech", "5.4h", [0, 1], "exploration", "plain", margin=1.2)
add("Q8", "shared replacement base pair (if Step 0 baseline is not identical)", "speech", "5.4h", [0, 1], "exploration", "plain", scope="R",
    note="serves a-c: all three add to the same frozen base config")
add("Q8", "c: 5-gram training + decoding calibration", "speech", "-", [None], "exploration", fixed=0.05, note="CPU training; GPU only for logits")
add("Q8", "d named + timing bundle", "speech", "5.4h", [0, 1], "exploration", "plain", margin=1.2, scope="F")
# ---- Outside reference (descriptive), HEADROOM item H1 (v6.2: R at measured prices is over 40 with it): 1 timed smoke + the published
#      recipe's defaults at 2 seeds (x1.5 margin); tuning effort = none on both sides (our Shared baseline is the fixed WN recipe too).
add("REF", "outside recipe smoke", "speech", "5.4h", [None], "descriptive", fixed=0.05, scope="H1")
add("REF", "outside recipe defaults", "speech", "5.4h", [0, 1], "descriptive", "plain", margin=1.5, scope="H1")

def tot(scopes, key="expected_gpu_h", groups=None):
    return round(sum(r[key] for r in rows if r["scope"] in scopes and (groups is None or r["group"] in groups)), 2)
spent = tot("X"); r_new = tot("R"); r_new_w = tot("R", "worst_gpu_h"); h = tot(["H1", "H2"]); f = tot("F")
h1, h2 = tot(["H1"]), tot(["H2"])
cont = 0.10 * r_new
S = dict(prices={c: {k: round(v, 4) if isinstance(v, float) else v for k, v in P[c].items()} for c in P},
         price_small_h={f"{c} {s}": round(small_h(c, s), 3) for c in P for s in UPD},
         spent=spent, R_new=r_new, R_new_worst=r_new_w, contingency_10pc=cont, R_total=round(spent + r_new + cont, 2),
         R_total_worst=round(spent + r_new_w + 0.10 * r_new_w, 2), headroom_H1_REF=h1, headroom_H2_Q2c1h=h2, headroom_H=h,
         R_plus_H1=round(spent + 1.1 * (r_new + h1), 2),
         R_plus_H=round(spent + r_new + h + 0.10 * (r_new + h), 2), deferred_F=f,
         package_100h=round(1.1 * sum(r["expected_gpu_h"] for r in rows if r["scope"] == "F" and r["group"] == "Q1" and ("5.4 h grid" in r["arm"] or r["data_size"] == "100h")), 2),
         R_plus_package=round(spent + 1.1 * (r_new + sum(r["expected_gpu_h"] for r in rows if r["scope"] == "F" and r["group"] == "Q1" and ("5.4 h grid" in r["arm"] or r["data_size"] == "100h"))), 2),
         rounding="every total = unrounded row sum (rows at 3 dp), 10 % applied before the final 2 dp round",
         all_enumerated=round(spent + r_new + h + f + 0.10 * (r_new + h + f), 2),
         by_group_R={g: tot("R", groups=[g]) for g in ("Step0", "Q2c", "Q4", "Q1", "Q8", "REF")},
         step0_allowance_check=dict(spent=spent, planned=tot("R", groups=["Step0"]), total=round(spent + tot("R", groups=["Step0"]), 2), allowance="4-5"),
         runs_R=sum(1 for r in rows if r["scope"] == "R"))
if os.environ.get("FREEZE") == "1":  # v6.3 C5 3: a freeze ledger needs every row's config hash filled in
    empty = [r["arm"] for r in rows if r["config_id"] in ("", "(freeze)", None) and r["stage"] != "spent"]
    if empty: sys.exit(f"FREEZE refused: {len(empty)} rows without a config hash, e.g. {empty[:3]}")
json.dump(dict(summary=S, rows=rows), open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w"), indent=1)
print(json.dumps(S, indent=1))
