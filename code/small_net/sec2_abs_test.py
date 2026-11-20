"""Simulation check of sec2_abs.py (CALIBRATION; bands written 20 Nov 01:4x BEFORE the first run).
Model: d[c, r] = Delta + a_c + s_r + i_cr, panels 40 (clean) + 40 (hard), k = 5 seeds; also clean-only (40). Scenarios: speaker-dominated,
seed-dominated, interaction-dominated, all mixed. 4,000 draws each.
BANDS: coverage of the (1 - 0.05/3) interval >= 0.97 in every scenario (nominal 0.9833); the BROKEN control (speaker variance only, Vc,
df = C_eff - 1) must cover < 0.90 in the seed-dominated scenario. Known answers for decide(): a clear win passes; the same win with one panel
shifted to +1.0 point fails the harm rule; a clear loss is LOSS."""
import json, sys, os, numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sec2_abs import estimate, weights, decide, ALPHA_PRIMARY

rng = np.random.default_rng(20261120); K = 5; N = 4000
SCEN = {"speaker": (2.0, 0.2, 0.3), "seed": (0.3, 1.5, 0.3), "interaction": (0.3, 0.2, 2.0), "mixed": (1.0, 0.7, 1.0)}
res = {}
for panels in ((40, 40), (40,)):
    w = weights(list(panels), [1 / len(panels)] * len(panels)); C = sum(panels)
    for name, (sa, ss, si) in SCEN.items():
        cov = covb = 0
        for _ in range(N):
            d = -0.7 + sa * rng.standard_normal((C, 1)) + ss * rng.standard_normal((1, K)) + si * rng.standard_normal((C, K))
            E = estimate(d, w); q = stats.t.ppf(1 - ALPHA_PRIMARY / 2, E["df"]); cov += abs(E["Delta"] + 0.7) <= q * E["se"]
            qb = stats.t.ppf(1 - ALPHA_PRIMARY / 2, E["Ceff"] - 1); covb += abs(E["Delta"] + 0.7) <= qb * E["Vc"] ** 0.5
        res[f"{'+'.join(map(str, panels))}_{name}"] = dict(coverage=cov / N, broken_coverage=covb / N)
ok = all(v["coverage"] >= 0.97 for v in res.values()) and all(res[k]["broken_coverage"] < 0.90 for k in res if k.endswith("_seed"))
mk = lambda delta, C=40: [delta + 0.3 * rng.standard_normal((C, 1)) + 0.1 * rng.standard_normal((1, K)) + 0.2 * rng.standard_normal((C, K))]
win = decide(mk(-2.0) + mk(-2.0), [0.5, 0.5]); harm = decide(mk(-6.0) + mk(1.0), [0.5, 0.5]); loss = decide(mk(2.0) + mk(2.0), [0.5, 0.5])
ka = win["verdict"] == "WIN" and harm["verdict"] != "WIN" and not harm["harm_ok"] and loss["verdict"] == "LOSS"
res["known_answers"] = dict(win=win["verdict"], harm=[harm["verdict"], harm["harm_upper"]], loss=loss["verdict"])
res["PASS"] = bool(ok and ka)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/sec2_abs_sims.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=0, default=float))
