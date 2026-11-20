"""Coarticulation as context experts: class means conditioned on the neighbour groups (prev, next), shrunk to the class mean;
shared covariance. Oracle (true neighbours) = ceiling; soft (base-LDA posteriors of the neighbours) = deployable. See NIGHT_REPORT 1e."""
import glob, json
import numpy as np
import af, phones, phones14, phones15, phones16, phones12

GRP = {**{p: 0 for p, m in zip(af.PHONES, phones14.mac_of) if m == 0}, **{p: 1 for p in "B F M P V W".split()},
       **{p: 2 for p in "CH D DH JH L N R S SH T TH Y Z ZH".split()}, **{p: 3 for p in "G HH K NG".split()}}
assert sorted(GRP) == sorted(af.PHONES); G_OF = np.array([GRP[p] for p in af.PHONES]); EDGE = 4; NG = 5; K = phones.NP

def adjacency(S):
    """prev[j] / next[j] = index of the adjacent segment in the same utterance, else -1."""
    sg = S["segs"]; utt = np.searchsorted(np.cumsum(S["lens"]), sg[:, 0], side="right"); n = len(sg)
    ok = (sg[1:, 0] == sg[:-1, 1]) & (utt[1:] == utt[:-1]); prev = np.full(n, -1); nxt = np.full(n, -1)
    prev[1:][ok] = np.arange(n - 1)[ok]; nxt[:-1][ok] = np.arange(1, n)[ok]; return prev, nxt

def fit(X, y, ctx, tau, lam=0.05, tau_p=None, nc=NG * NG, K=K, ms=None, S0=None, alpha=1.0):
    m, s = (X.mean(0), X.std(0) + 1e-9) if ms is None else ms; Z = (X - m) / s; mu = np.stack([Z[y == k].mean(0) for k in range(K)]); R = Z - mu[y]
    Sm = R.T @ R / len(R); Sm = (Sm if S0 is None else alpha * Sm + (1 - alpha) * S0) + lam * np.eye(Z.shape[1]); M_S = Sm - lam * np.eye(Z.shape[1]); pk = np.bincount(y, minlength=K) / len(y)
    sums = np.zeros((nc, K, Z.shape[1])); np.add.at(sums, (ctx, y), Z); cnt = np.zeros((nc, K)); np.add.at(cnt, (ctx, y), 1)
    muc = (sums + tau * mu[None]) / (cnt + tau)[..., None]; tp = tau if tau_p is None else tau_p; pc = (cnt + tp * pk[None]) / (cnt.sum(1, keepdims=True) + tp)  # tau_p: prior shrinkage, separate from the means
    Wc = np.linalg.solve(Sm, muc.reshape(-1, Z.shape[1]).T).T.reshape(nc, K, -1)
    return dict(m=m, s=s, S=M_S, W=Wc, b=-0.5 * np.einsum("ckj,ckj->ck", Wc, muc) + np.log(pc))

def post(M, X, c):
    """P(k | x, c) for one context per row."""
    return post_mix(M, X, np.eye(M["W"].shape[0])[c])  # one-hot mix; loops over contexts (a per-row einsum needs n x K x dims memory)

def post_mix(M, X, w):
    """sum_c w[n, c] P(k | x, c): soft neighbour contexts."""
    Z = (X - M["m"]) / M["s"]; out = np.zeros((len(X), M["W"].shape[1])); kept = np.zeros(len(X))
    for c in range(w.shape[1]):
        sel = w[:, c] > 1e-4
        if not sel.any(): continue
        L = Z[sel] @ M["W"][c].T + M["b"][c]; E = np.exp(L - L.max(1, keepdims=True)); out[sel] += w[sel, c, None] * E / E.sum(1, keepdims=True); kept[sel] += w[sel, c]
    return out / kept[:, None]  # renormalise over the contexts kept (skipped weights <= 1e-4 would otherwise leak mass)

def ctx_true(S):
    g = G_OF[S["segs"][:, 2]]; prev, nxt = adjacency(S); return np.where(prev >= 0, g[prev], EDGE) * NG + np.where(nxt >= 0, g[nxt], EDGE)

def ctx_soft(S, P0):
    gp = np.stack([P0[:, G_OF == g].sum(1) for g in range(4)] + [np.zeros(len(P0))], 1); prev, nxt = adjacency(S); e = np.eye(NG)[EDGE]
    a = np.where((prev >= 0)[:, None], gp[np.maximum(prev, 0)], e); b = np.where((nxt >= 0)[:, None], gp[np.maximum(nxt, 0)], e)
    return (a[:, :, None] * b[:, None, :]).reshape(len(P0), -1)

def acc(P, y, sel):
    mo = phones14.mac_of; pm = np.stack([P[sel][:, mo == k].sum(1) for k in range(3)], 1).argmax(1)
    return round(float(np.mean(pm == mo[y[sel]])), 4), round(float(np.mean(P[sel].argmax(1) == y[sel])), 4)

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC, RD = phones12.build("cal", fc, C["lens"]), phones12.build("dev_other", fd, D["lens"]); yc, y = C["segs"][:, 2], D["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0)
    pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    XC = np.c_[phones15.feats(phones15.causal_norm(RC, fc, C, 200, pm_, pv_), C), phones16.rate(C, fc, 5, pl)]
    XD = np.c_[phones15.feats(phones15.causal_norm(RD, fd, D, 200, pm_, pv_), D), phones16.rate(D, fd, 5, pl)]; del RC, RD
    spk = np.array([f.split("/")[-3] for f in fd]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt])
    cc, cd = ctx_true(C), ctx_true(D); print("dev context counts (prev*5+next):", np.bincount(cd, minlength=25).tolist(), flush=True)
    B0 = fit(XC, yc, np.zeros(len(yc), int), 1.0); P0 = post(B0, XD, np.zeros(len(y), int))
    R = {"base": dict(A=acc(P0, y, A), B=acc(P0, y, ~A))}; print("base (one context)", R["base"], flush=True)
    W = ctx_soft(D, P0); print("soft context: mean prob on the true context", round(float(W[np.arange(len(cd)), cd].mean()), 3), flush=True)
    for tau in (10, 50):
        M = fit(XC, yc, cc, tau); Po, Ps = post(M, XD, cd), post_mix(M, XD, W)
        R[f"oracle_tau{tau}"] = dict(A=acc(Po, y, A), B=acc(Po, y, ~A)); R[f"soft_tau{tau}"] = dict(A=acc(Ps, y, A), B=acc(Ps, y, ~A))
        print(f"oracle_tau{tau}", R[f"oracle_tau{tau}"], f"soft_tau{tau}", R[f"soft_tau{tau}"], flush=True)
    best = max(("soft_tau10", "soft_tau50"), key=lambda k: R[k]["A"][1]); b0, b = R["base"]["B"], R[best]["B"]
    keep = (b[0] - b0[0] >= 0.002 and b[1] >= b0[1]) or (b[1] - b0[1] >= 0.002 and b[0] >= b0[0])
    print("chosen on half A (39-way):", best, "half B:", b, "base half B:", b0, "KEEP" if keep else "DROP", flush=True)
    json.dump(dict(R=R, chosen=best, keep=keep), open(f"{phones.OUT}/phones19.json", "w"), indent=1)
