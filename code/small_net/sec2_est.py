"""MASTER_PLAN v5.4 section 2 estimator (Step 0.3). Delta = (eB - eA) / eB, cluster weights w (sum 1), k paired seeds.
e_A, e_B: arrays (C, k) of per-cluster per-seed error rates. Returns Delta_hat, V components, V_primary, df cap, C_eff."""
import numpy as np
from math import isfinite

def estimate(eA, eB, w):
    eA, eB, w = np.asarray(eA, float), np.asarray(eB, float), np.asarray(w, float)
    C, k = eA.shape; assert eB.shape == (C, k) and w.shape == (C,) and k >= 2 and abs(w.sum() - 1) < 1e-9
    hA = (w[:, None] * eA.mean(1, keepdims=True)).sum(); hB = (w[:, None] * eB.mean(1, keepdims=True)).sum()
    D = (hB - hA) / hB
    u = (w[:, None] / hB) * ((1 - D) * (eB - hB) - (eA - hA))           # (C, k)
    assert abs(u.mean(1).sum()) < 1e-9 * max(1.0, np.abs(u).sum())      # sums over c, averages over r = 0
    dw = 1 - 2 * (w ** 3).sum() / (w ** 2).sum() + (w ** 2).sum()
    ub = u.mean(1); U = u.sum(0)
    Vc = (ub ** 2).sum() / dw; Vs = (U ** 2).sum() / (k * (k - 1))
    Vi = ((u - ub[:, None] - w[:, None] * U[None, :]) ** 2).sum() / (dw * k * (k - 1))
    Vp = max(Vc, Vs, Vc + Vs - Vi); Ceff = 1.0 / (w ** 2).sum()
    return dict(Delta=D, Vc=Vc, Vs=Vs, Vi=Vi, Vp=Vp, Vq=Vc + Vs, dw=dw, Ceff=Ceff, df_train=min(k - 1, C - 1), df_ckpt=Ceff - 1)
