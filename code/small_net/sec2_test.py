"""Step 0.3 unit tests + coverage sims for sec2_est. Rules written BEFORE running (20 Nov 00:1x): v5.4 gives no numeric band, so the agent sets
it: nominal 95 % two-sided t interval (df = k-1); PASS if Monte-Carlo coverage of V_primary is >= 0.90 in every 'plausible' setting (over-coverage reported as width, not failure)
(seed sd ratio <= 2, interaction <= seed); 'severe undercoverage' = < 0.85 in any plausible setting -> training-procedure inference exploratory.
Tests: (1) equal weights: d_w = (C-1)/C; (2) Vc for k seeds with zero seed/interaction variance tracks the MC variance of Delta_hat over
cluster resampling (ratio in [0.8, 1.25] at C=40); (3) known null: Delta 0 coverage; (4) Delta 4 %."""
import numpy as np, json, sys
from scipy import stats
sys.path.insert(0, __import__("os").path.dirname(__file__)); from sec2_est import estimate
rng = np.random.default_rng(20261120)

def world(C, k, D, sc, ss, si, skew, base=0.45):
    n = np.maximum(5, rng.lognormal(0, skew, C) * 50).round()      # cluster sizes (utterances / items)
    w = n / n.sum(); cb = rng.normal(0, sc, C)                       # cluster difficulty
    sA, sB = rng.normal(0, ss, k), rng.normal(0, ss, k)              # seed effects (independent per arm)
    pB = np.clip(base + cb[:, None] + sB[None, :] + rng.normal(0, si, (C, k)), .02, .98)
    pA = np.clip(base * (1 - D) + cb[:, None] * (1 - D) + sA[None, :] + rng.normal(0, si, (C, k)), .02, .98)
    eA = rng.binomial(n[:, None].astype(int), pA) / n[:, None]; eB = rng.binomial(n[:, None].astype(int), pB) / n[:, None]
    # true procedure Delta: expectation over new seeds and clusters ~ D (clipping is rare at these settings)
    return eA, eB, w

out = {}
w = np.full(10, .1); r = estimate(rng.random((10, 3)), rng.random((10, 3)) + .5, w); assert abs(r["dw"] - .9) < 1e-12; out["t1_dw_equal"] = "PASS"
# t2: no seed noise -> Vc vs MC variance across worlds with fixed seeds
D_, V_ = [], []
for _ in range(2000):
    eA, eB, w = world(40, 3, .04, .05, 0.0, 0.0, .5); r = estimate(eA, eB, w); D_.append(r["Delta"]); V_.append(r["Vc"])
ratio = float(np.mean(V_) / np.var(D_)); out["t2_Vc_over_MCvar"] = round(ratio, 3); assert .8 <= ratio <= 1.25, ratio
def cover(C, k, D, sc, ss, si, skew, N=4000):
    hit = hitq = 0; t = stats.t.ppf(.975, k - 1)
    for _ in range(N):
        eA, eB, w = world(C, k, D, sc, ss, si, skew); r = estimate(eA, eB, w)
        hit += abs(r["Delta"] - D) <= t * np.sqrt(r["Vp"]); hitq += abs(r["Delta"] - D) <= t * np.sqrt(r["Vq"])
    p = hit / N; return dict(cov=round(p, 4), mcse=round(float(np.sqrt(p * (1 - p) / N)), 4), cov_quad=round(hitq / N, 4))
grid = []
for C in (20, 40): 
  for k in (2, 3, 5):
    for ss in (.0, .01, .02):
      for si in (.0, .01):
        for skew in (.3, 1.0):
          for D in (0.0, .04):
            g = dict(C=C, k=k, D=D, ss=ss, si=si, skew=skew); g.update(cover(C, k, D, .05, ss, si, skew, 1500)); grid.append(g)
out["grid"] = grid
pl = [g for g in grid if g["si"] <= max(g["ss"], .0) or g["ss"] == 0]
out["min_cov_all"] = min(g["cov"] for g in grid); out["min_cov_plausible"] = min(g["cov"] for g in pl)
out["max_cov_all"] = max(g["cov"] for g in grid)
out["verdict"] = "PASS" if all(g["cov"] >= .90 for g in pl) else ("SEVERE" if out["min_cov_plausible"] < .85 else "OUTSIDE_BAND")
json.dump(out, open(__import__("os").path.join(__import__("os").path.dirname(__file__), "results", "sec2_sims.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k != "grid"})
for g in grid:
    if g["cov"] < .90: print("under", g)
# ---- broken control (must FAIL): cluster-only variance with checkpoint df (C_eff - 1) when seeds vary (ss .02, k 3, C 40)
hit = 0; N = 3000
for _ in range(N):
    eA, eB, w = world(40, 3, .04, .05, .02, .0, .5); r = estimate(eA, eB, w)
    hit += abs(r["Delta"] - .04) <= stats.t.ppf(.975, r["Ceff"] - 1) * np.sqrt(r["Vc"])
out["broken_cluster_only_cov"] = round(hit / N, 4); assert hit / N < .90, "broken control did not fail"
json.dump(out, open(__import__("os").path.join(__import__("os").path.dirname(__file__), "results", "sec2_sims.json"), "w"), indent=1)
print("broken control coverage", hit / N, "(must be < 0.90): FAILS as required")
