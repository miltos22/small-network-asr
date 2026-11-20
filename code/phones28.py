"""3c: continuous coarticulatory regression (CCR) in the r76 space: class means conditioned on the neighbours' MEASURED acoustics + prosody,
class-specific slopes shrunk to the category. NIGHT_REPORT 3c."""
import glob, json
import numpy as np
import phones, phones12, phones15, phones16, phones19 as c19, phones23 as t, phones25 as b

N = phones12.NAMES; VA = [N.index(k) for k in ("lF1", "lF2", "lF3", "C1", "C2", "C3", "C4", "C5")]; VR = [N.index("R"), N.index("Elow")]
IE, IF0 = N.index("Erel"), N.index("F0")

def samp(T, segs, cols, q):
    n = segs[:, 1] - segs[:, 0]; return T[segs[:, 0] + np.round(q * (n - 1)).astype(int)][:, cols]

def cond(T, Raw, S, files, pl):
    sg = S["segs"]; prev, nxt = c19.adjacency(S); hp, hn = prev >= 0, nxt >= 0; pi, ni = np.maximum(prev, 0), np.maximum(nxt, 0)
    s20, s50, s80 = (samp(T, sg, VA, q) for q in (0.2, 0.5, 0.8)); vr = samp(T, sg, VR, 0.5)
    tl = np.where(hp[:, None], ((s50 + s80) / 2)[pi], 0); tr = np.where(hn[:, None], ((s20 + s50) / 2)[ni], 0)
    vl, vn = np.where(hp[:, None], vr[pi], 0), np.where(hn[:, None], vr[ni], 0)
    emax = np.array([T[a:c, IE].max() for a, c in sg[:, :2]]); v = (Raw[:, IF0] != 0).astype(np.float64)
    cv, cf = np.r_[0, np.cumsum(v)], np.r_[0, np.cumsum(T[:, IF0] * v)]; nv = cv[sg[:, 1]] - cv[sg[:, 0]]
    f0 = np.where(nv > 0, (cf[sg[:, 1]] - cf[sg[:, 0]]) / np.maximum(nv, 1), 0)
    r = phones16.rate(S, files, 5, pl); ln = np.log(sg[:, 1] - sg[:, 0])
    dE = np.where(hp & hn, emax[ni] - emax[pi], 0); dF = np.where(hp & hn, f0[ni] - f0[pi], 0)
    return np.c_[tl, tr, vl, vn, hp, hn, ln - r, r, dE, dF, emax, f0]

def ccr_fit(Z, U, y, tau=200, lam=1e-3):
    K = phones.NP; d, q = Z.shape[1], U.shape[1]; cnt = np.bincount(y, minlength=K).astype(float)
    mz = np.stack([Z[y == k].mean(0) for k in range(K)]); mu = np.stack([U[y == k].mean(0) for k in range(K)])
    Zc, Uc = Z - mz[y], U - mu[y]; Szu = np.stack([Zc[y == k].T @ Uc[y == k] for k in range(K)]); Suu = np.stack([Uc[y == k].T @ Uc[y == k] for k in range(K)])
    Wk = np.empty((K, d, q))
    for k in range(K):
        cs = t.CAT_OF == t.CAT_OF[k]; nc = cnt[cs].sum(); a = tau / nc
        Wk[k] = np.linalg.solve((Suu[k] + a * Suu[cs].sum(0) + lam * cnt[k] * np.eye(q)).T, (Szu[k] + a * Szu[cs].sum(0)).T).T
    R = Zc - np.einsum("nq,ndq->nd", Uc, Wk[y]) if len(y) < 1 else np.concatenate([Zc[y == k] - Uc[y == k] @ Wk[k].T for k in range(K)])
    S = R.T @ R / len(R) + 1e-6 * np.eye(d); Su = Uc.T @ Uc / len(Uc) + 1e-6 * np.eye(q)
    return dict(mz=mz, mu=mu, W=Wk, Si=np.linalg.inv(S), Sui=np.linalg.inv(Su), lp=np.log(cnt / cnt.sum()))

def ccr_ll(M, Z, U, with_u):
    out = np.empty((len(Z), phones.NP))
    for k in range(phones.NP):
        du = U - M["mu"][k]; r = Z - M["mz"][k] - du @ M["W"][k].T; out[:, k] = -0.5 * np.einsum("nd,de,ne->n", r, M["Si"], r)
        if with_u: out[:, k] += -0.5 * np.einsum("nd,de,ne->n", du, M["Sui"], du)
    return out

def softmax(L): E = np.exp(L - L.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)

def ctx_priors(yc, cc, tau=200):
    K = phones.NP; pk = np.bincount(yc, minlength=K) / len(yc); cnt = np.zeros((25, K)); np.add.at(cnt, (cc, yc), 1)
    return (cnt + tau * pk) / (cnt.sum(1, keepdims=True) + tau)

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC, RD = phones12.build("cal", fc, C["lens"]), phones12.build("dev_other", fd, D["lens"]); yc, y = C["segs"][:, 2], D["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    TC, TD = phones15.causal_norm(RC, fc, C, 200, pm_, pv_), phones15.causal_norm(RD, fd, D, 200, pm_, pv_)
    XC = np.c_[phones15.feats(TC, C), phones16.rate(C, fc, 5, pl)]; XD = np.c_[phones15.feats(TD, D), phones16.rate(D, fd, 5, pl)]
    QC, QD = cond(TC, RC, C, fc, pl), cond(TD, RD, D, fd, pl); qm, qs = QC.mean(0), QC.std(0) + 1e-9; QC, QD = (QC - qm) / qs, (QD - qm) / qs; del RC, RD, TC, TD
    spk = np.array([f.split("/")[-3] for f in fd]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt])
    cc = c19.ctx_true(C); Mfull = c19.fit(XC, yc, cc, 200); V = b.projection(XC, yc, cc, Mfull, 76)
    UC, UD = ((XC - Mfull["m"]) / Mfull["s"]) @ V, ((XD - Mfull["m"]) / Mfull["s"]) @ V; del XC, XD
    R = {}; B0 = c19.fit(UC, yc, np.zeros(len(yc), int), 1.0); P0 = c19.post(B0, UD, np.zeros(len(y), int)); R["plain_r76"] = dict(A=c19.acc(P0, y, A), B=c19.acc(P0, y, ~A))
    M = c19.fit(UC, yc, cc, 200); P = P0
    for _ in (1, 2, 3): P = c19.post_mix(M, UD, c19.ctx_soft(D, P))
    R["discrete_ctx_r76"] = dict(A=c19.acc(P, y, A), B=c19.acc(P, y, ~A)); print({k: R[k] for k in R}, flush=True)
    pc = ctx_priors(yc, cc)
    for name, cols, wu in (("V2_anchors", slice(0, 22), False), ("V3_anchors_prosody", slice(0, 28), False), ("V3u", slice(0, 28), True)):
        Mc = ccr_fit(UC, QC[:, cols], yc); L = ccr_ll(Mc, UD, QD[:, cols], wu); Pc = softmax(L + Mc["lp"])
        R[name] = dict(A=c19.acc(Pc, y, A), B=c19.acc(Pc, y, ~A)); print(name, R[name], flush=True)
        if name == "V3u":
            Pq = Pc
            for _ in (1, 2, 3):
                w = c19.ctx_soft(D, Pq); Pq = softmax(L + np.log(w @ pc))  # p(x|k) * sum_c w_c P(k|c)
            R["V4_V3u+ctx_priors"] = dict(A=c19.acc(Pq, y, A), B=c19.acc(Pq, y, ~A)); print("V4_V3u+ctx_priors", R["V4_V3u+ctx_priors"], flush=True)
    cand = [k for k in R if k.startswith("V")]; best = max(cand, key=lambda k: R[k]["A"][1]); b0, bb = R["discrete_ctx_r76"]["B"], R[best]["B"]
    keep = (bb[0] - b0[0] >= 0.002 and bb[1] >= b0[1]) or (bb[1] - b0[1] >= 0.002 and bb[0] >= b0[0])
    print("chosen on half A (39-way):", best, "half B:", bb, "discrete-context r76 half B:", b0, "KEEP" if keep else "DROP", flush=True)
    json.dump(dict(R=R, chosen=best, keep=keep), open(f"{phones.OUT}/phones28.json", "w"), indent=1)
