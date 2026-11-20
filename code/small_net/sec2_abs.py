"""Q2(c) analysis (MASTER_PLAN Part C v6.3 C4; review round 3 C4): ABSOLUTE paired PER difference, speaker-equal weights, panels averaged.
d[c, r] = PER_sem[c, r] - PER_shuf[c, r] (points) for speaker c and paired seed r (arm pair r shares data order and seed; shuffle draw r).
Weights: each panel weight p (sum 1; both panels 0.5 / 0.5; clean-only fallback = 1.0 on clean), each speaker of a panel p / C_panel.
Because both arms score the SAME utterances, Delta = sum_c w_c mean_r d[c, r] is linear: no ratio linearisation (unlike sec2_est.py, which is
the relative-error estimator of B 2). Variance: the same three components as sec2_est.py (speakers Vc, seeds Vs, interaction Vi), primary
V = max(Vc, Vs, Vc + Vs - Vi), df = min(k - 1, C_eff - 1) (seed-capped). Decision rules (fixed here, before data):
  WIN  : two-sided (1 - 0.05/3) interval of Delta entirely below 0 (semantic lower PER), AND
  HARM : on EACH panel separately, one-sided 95 % upper bound of that panel's Delta <= +0.5 point.
  LOSS : interval entirely above 0. Otherwise INCONCLUSIVE. No equivalence claim is made from INCONCLUSIVE.
Usage: from sec2_abs import estimate, decide; python -I sec2_abs_test.py runs the simulation check."""
import numpy as np
from scipy import stats

ALPHA_PRIMARY = 0.05 / 3; HARM_BOUND = 0.5; HARM_ALPHA = 0.05


def estimate(d, w):
    d, w = np.asarray(d, float), np.asarray(w, float); C, k = d.shape
    assert w.shape == (C,) and k >= 2 and C >= 2 and abs(w.sum() - 1) < 1e-9 and (w > 0).all()
    dbar = d.mean(1); D = float((w * dbar).sum())
    u = w[:, None] * (d - D)                                             # sum over c of the seed-mean = 0
    assert abs(u.mean(1).sum()) < 1e-9 * max(1.0, np.abs(u).sum())
    dw = 1 - 2 * (w ** 3).sum() / (w ** 2).sum() + (w ** 2).sum()
    ub = u.mean(1); U = u.sum(0)
    Vc = (ub ** 2).sum() / dw; Vs = (U ** 2).sum() / (k * (k - 1))
    Vi = ((u - ub[:, None] - w[:, None] * U[None, :]) ** 2).sum() / (dw * k * (k - 1))
    Vp = max(Vc, Vs, Vc + Vs - Vi); Ceff = 1.0 / (w ** 2).sum(); df = max(1.0, min(k - 1, Ceff - 1))
    return dict(Delta=D, Vc=Vc, Vs=Vs, Vi=Vi, Vp=Vp, se=Vp ** 0.5, df=df, Ceff=Ceff)


def weights(panels, pw):
    # panels: list of speaker counts per panel; pw: panel weights (sum 1)
    assert len(panels) == len(pw) and abs(sum(pw) - 1) < 1e-12
    return np.concatenate([np.full(n, p / n) for n, p in zip(panels, pw)])


K_REQUIRED = 5   # v6.4 (round 4 C4): Q2 (c) is decided only on exactly five complete pairs


def from_counts(rows, panels, arms=("S", "R"), seeds=range(5)):
    """v6.4 live input adapter. rows: dicts {panel, arm, seed, utt, speaker, e, t} from score_locked.py's per-utterance file.
    Checks: exactly the listed arms x seeds per panel, the same utterance ids AND reference counts in every (arm, seed) cell, finite
    non-negative counts, t > 0. Returns d_by_panel (speaker-level PER difference arm0 - arm1 in points, speakers sorted)."""
    out = []
    for pn in panels:
        cell = {}
        for r in rows:
            if r["panel"] != pn: continue
            assert r["arm"] in arms and r["seed"] in seeds, ("unexpected slot", r["arm"], r["seed"])
            assert np.isfinite(r["e"]) and np.isfinite(r["t"]) and r["e"] >= 0 and r["t"] > 0, ("bad counts", r)
            cell.setdefault((r["arm"], r["seed"]), {})[r["utt"]] = (r["speaker"], r["e"], r["t"])
        slots = [(a, s) for a in arms for s in seeds]
        assert sorted(cell) == sorted(slots), ("missing or extra slots in panel", pn, sorted(cell))
        ref = cell[slots[0]]
        for sl in slots: assert {u: (v[0], v[2]) for u, v in cell[sl].items()} == {u: (v[0], v[2]) for u, v in ref.items()}, ("items differ", pn, sl)
        spk = sorted({v[0] for v in ref.values()})
        per = {sl: np.array([100 * sum(v[1] for v in cell[sl].values() if v[0] == c) / sum(v[2] for v in cell[sl].values() if v[0] == c) for c in spk]) for sl in slots}
        out.append(np.stack([per[(arms[0], s)] - per[(arms[1], s)] for s in seeds], 1))
    return out


def decide(d_by_panel, pw, k_required=K_REQUIRED):
    # d_by_panel: list of arrays (C_p, k), same k; returns verdict + all bounds
    assert all(x.shape[1] == k_required for x in d_by_panel), ("Q2 (c) needs exactly", k_required, "complete pairs")
    assert all(np.isfinite(x).all() for x in d_by_panel)
    w = weights([x.shape[0] for x in d_by_panel], pw); E = estimate(np.vstack(d_by_panel), w)
    q = stats.t.ppf(1 - ALPHA_PRIMARY / 2, E["df"]); lo, hi = E["Delta"] - q * E["se"], E["Delta"] + q * E["se"]
    harm = []
    for x in d_by_panel:
        e = estimate(x, np.full(x.shape[0], 1.0 / x.shape[0])); harm.append(e["Delta"] + stats.t.ppf(1 - HARM_ALPHA, e["df"]) * e["se"])
    ok_harm = all(h <= HARM_BOUND for h in harm)
    v = "WIN" if hi < 0 and ok_harm else "LOSS" if lo > 0 else "INCONCLUSIVE"
    return dict(verdict=v, Delta=E["Delta"], lo=lo, hi=hi, df=E["df"], harm_upper=harm, harm_ok=ok_harm, est=E)
