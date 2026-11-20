"""Hybrid tree: P(category) from the flat context model, P(phone | category) from leaves with their own covariance shrunk toward
the global one (RDA, alpha). Dual-parent leaves. NIGHT_REPORT 2b."""
import glob, json
import numpy as np
import phones, phones12, phones15, phones16, phones19 as c19, phones23 as t

def hybrid(XC, yc, cc, XD, w, Pflat, G, alpha, tau=200):
    Pc = np.stack([Pflat[:, t.CAT_OF == c].sum(1) for c in range(9)], 1); P = np.zeros_like(Pflat)
    for c, ps in enumerate(t.CATS):
        leaf = [phones.PID[p] for p in ps.split() + t.DUAL.get(c, [])]
        if len(leaf) == 1: P[:, leaf[0]] += Pc[:, c]; continue
        sel = np.isin(yc, leaf); Ml = c19.fit(XC[sel], np.searchsorted(np.sort(leaf), yc[sel]), cc[sel], tau, K=len(leaf), ms=(G["m"], G["s"]), S0=G["S"], alpha=alpha)
        P[:, leaf] += Pc[:, c:c + 1] * c19.post_mix(Ml, XD, w)[:, np.argsort(np.argsort(leaf))]
    assert np.allclose(P.sum(1), 1, atol=1e-6), P.sum(1)[:5]
    return P

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC, RD = phones12.build("cal", fc, C["lens"]), phones12.build("dev_other", fd, D["lens"]); yc, y = C["segs"][:, 2], D["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    XC = np.c_[phones15.feats(phones15.causal_norm(RC, fc, C, 200, pm_, pv_), C), phones16.rate(C, fc, 5, pl)]
    XD = np.c_[phones15.feats(phones15.causal_norm(RD, fd, D, 200, pm_, pv_), D), phones16.rate(D, fd, 5, pl)]; del RC, RD
    spk = np.array([f.split("/")[-3] for f in fd]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt])
    cc = c19.ctx_true(C); B0 = c19.fit(XC, yc, np.zeros(len(yc), int), 1.0); P = c19.post(B0, XD, np.zeros(len(y), int)); M = c19.fit(XC, yc, cc, 200)
    for r in (1, 2, 3): w = c19.ctx_soft(D, P); P = c19.post_mix(M, XD, w)
    w = c19.ctx_soft(D, P); R = {"flat": dict(A=c19.acc(P, y, A), B=c19.acc(P, y, ~A))}; print("flat (phones20 check)", R["flat"], flush=True)
    for alpha in (0.3, 0.7, 1.0):
        Ph = hybrid(XC, yc, cc, XD, w, P, M, alpha); k = f"hybrid_a{alpha}"; R[k] = dict(A=c19.acc(Ph, y, A), B=c19.acc(Ph, y, ~A)); print(k, R[k], flush=True)
    best = max((k for k in R if k != "flat"), key=lambda k: R[k]["A"][1]); b0, b = R["flat"]["B"], R[best]["B"]
    keep = (b[0] - b0[0] >= 0.002 and b[1] >= b0[1]) or (b[1] - b0[1] >= 0.002 and b[0] >= b0[0])
    print("chosen on half A (39-way):", best, "half B:", b, "flat half B:", b0, "KEEP" if keep else "DROP", flush=True)
    json.dump(dict(R=R, chosen=best, keep=keep), open(f"{phones.OUT}/phones24.json", "w"), indent=1)
