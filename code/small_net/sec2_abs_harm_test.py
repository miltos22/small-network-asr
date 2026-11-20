"""v6.4 (round 4 C4) CALIBRATION of the one-sided harm certification of sec2_abs.decide() and of its input checks. Bands written 20 Nov
01:57 BEFORE the first run. Per panel: d[c, r] = Delta_p + a_c + s_r + i_cr, k = 5, panel sizes 40 and 10 (a smaller hard panel allowed by
C2 0 d), the four scenarios of sec2_abs_test.py, 4,000 draws each.
BANDS: (1) false certification (upper bound <= +0.5) when the true panel Delta = +0.5: <= 0.06 in every scenario and size (nominal 0.05);
(2) when true Delta = +0.6: <= 0.05; (3) power reported (true Delta = -0.5), no band. KNOWN ANSWERS: a result that wins on the primary interval
but has one panel at +0.8 must NOT be WIN (harm only); decide() refuses 4 pairs; from_counts() refuses a missing slot, changed items, NaN."""
import json, sys, os, numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sec2_abs import estimate, decide, from_counts, HARM_ALPHA, HARM_BOUND

rng = np.random.default_rng(202611202); K = 5; N = 4000
SCEN = {"speaker": (2.0, 0.2, 0.3), "seed": (0.3, 1.5, 0.3), "interaction": (0.3, 0.2, 2.0), "mixed": (1.0, 0.7, 1.0)}
def cert_rate(delta, C, sa, ss, si):
    n = 0
    for _ in range(N):
        d = delta + sa * rng.standard_normal((C, 1)) + ss * rng.standard_normal((1, K)) + si * rng.standard_normal((C, K))
        e = estimate(d, np.full(C, 1.0 / C)); n += e["Delta"] + stats.t.ppf(1 - HARM_ALPHA, e["df"]) * e["se"] <= HARM_BOUND
    return n / N
res = {}
for C in (40, 10):
    for nm, sc in SCEN.items():
        res[f"C{C}_{nm}"] = dict(false_cert_at_0p5=cert_rate(0.5, C, *sc), false_cert_at_0p6=cert_rate(0.6, C, *sc), power_at_m0p5=cert_rate(-0.5, C, *sc))
b1 = all(v["false_cert_at_0p5"] <= 0.06 and v["false_cert_at_0p6"] <= 0.05 for v in res.values())
mk = lambda delta, C=40: delta + 0.3 * rng.standard_normal((C, 1)) + 0.1 * rng.standard_normal((1, K)) + 0.2 * rng.standard_normal((C, K))
h = decide([mk(-3.0), mk(0.8)], [0.5, 0.5]); ka1 = h["hi"] < 0 and not h["harm_ok"] and h["verdict"] != "WIN"
try: decide([mk(-3.0)[:, :4], mk(-3.0)[:, :4]], [0.5, 0.5]); ka2 = False
except AssertionError: ka2 = True
rows = [dict(panel="clean", arm=a, seed=s, utt=f"u{i}", speaker=f"c{i % 4}", e=float(i % 3), t=10.0) for a in ("S", "R") for s in range(5) for i in range(16)]
ok_adapter = from_counts(rows, ["clean"])[0].shape == (4, 5)
def refuses(rs):
    try: from_counts(rs, ["clean"]); return False
    except AssertionError: return True
ka3 = ok_adapter and refuses(rows[1:] if False else [r for r in rows if not (r["arm"] == "R" and r["seed"] == 4)]) \
      and refuses([dict(r, utt="x" + r["utt"]) if (r["arm"], r["seed"]) == ("R", 2) else r for r in rows]) \
      and refuses([dict(r, e=float("nan")) if r["utt"] == "u3" and r["arm"] == "S" else r for r in rows])
res["known_answers"] = dict(primary_win_harm_fail=bool(ka1), harm_upper=h["harm_upper"], refuses_4_pairs=ka2, adapter_checks=bool(ka3))
res["PASS"] = bool(b1 and ka1 and ka2 and ka3)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/sec2_abs_harm_sims.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=0, default=float))
