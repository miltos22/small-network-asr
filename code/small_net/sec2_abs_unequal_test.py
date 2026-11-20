"""CALIBRATION (review round 5 C4): primary-interval coverage for the UNEQUAL panel sizes the hard-panel rule admits (clean 40 + hard 10..39),
panel weights 0.5 / 0.5, plus a heteroscedastic variant (hard panel 2x the speaker and interaction SD). BAND written BEFORE the first run
(20 Nov, same as sec2_abs_test.py): coverage >= 0.97 in every setting (nominal 0.9833). A setting below the band is NOT admitted: the
hard-panel rule then needs that size dropped or the claim made checkpoint-conditional. 4,000 draws each."""
import json, sys, os, numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sec2_abs import estimate, weights, ALPHA_PRIMARY
rng = np.random.default_rng(20261121); K = 5; N = 4000
SCEN = {"speaker": (2.0, 0.2, 0.3), "seed": (0.3, 1.5, 0.3), "interaction": (0.3, 0.2, 2.0), "mixed": (1.0, 0.7, 1.0)}
res = {}
for hard in (10, 15, 20, 30, 39):
    for het in (1.0, 2.0):
        w = weights([40, hard], [0.5, 0.5]); C = 40 + hard; scale = np.r_[np.ones(40), np.full(hard, het)][:, None]
        for name, (sa, ss, si) in SCEN.items():
            cov = 0
            for _ in range(N):
                d = -0.7 + scale * sa * rng.standard_normal((C, 1)) + ss * rng.standard_normal((1, K)) + scale * si * rng.standard_normal((C, K))
                E = estimate(d, w); q = stats.t.ppf(1 - ALPHA_PRIMARY / 2, E["df"]); cov += abs(E["Delta"] + 0.7) <= q * E["se"]
            res[f"40+{hard}_het{het}_{name}"] = cov / N
res["min"] = min(v for v in res.values()); res["PASS"] = bool(res["min"] >= 0.97)
res["below_band"] = [k for k, v in res.items() if isinstance(v, float) and k not in ("min",) and v < 0.97]
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/sec2_abs_unequal_sims.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=0, default=float))
