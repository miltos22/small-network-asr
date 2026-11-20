"""Budget version: all context models in an r-dim whitened discriminant space (between-(phone,context) vs within-phone). NIGHT_REPORT 2c."""
import glob, json
import numpy as np
from scipy.linalg import eigh
import phones, phones12, phones15, phones16, phones19 as c19, phones23 as t, phones24 as h

def projection(XC, yc, cc, M, r):
    Z = (XC - M["m"]) / M["s"]; Sw = M["S"] + 0.05 * np.eye(Z.shape[1]); key = cc * phones.NP + yc
    cnt = np.bincount(key, minlength=25 * phones.NP); sums = np.zeros((25 * phones.NP, Z.shape[1])); np.add.at(sums, key, Z)
    ok = cnt > 0; mu = sums[ok] / cnt[ok, None]; g = Z.mean(0); Dv = (mu - g) * np.sqrt(cnt[ok] / len(Z))[:, None]
    vals, V = eigh(Dv.T @ Dv, Sw); V = V[:, ::-1][:, :r]  # eigh normalises V' Sw V = I
    return V

def count(M_flat, leaves, r, dims):
    return dims * r + 2 * dims + M_flat["W"].size + M_flat["b"].size + 2 * r + sum(l["W"].size + l["b"].size for l in leaves)

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC, RD = phones12.build("cal", fc, C["lens"]), phones12.build("dev_other", fd, D["lens"]); yc, y = C["segs"][:, 2], D["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    XC = np.c_[phones15.feats(phones15.causal_norm(RC, fc, C, 200, pm_, pv_), C), phones16.rate(C, fc, 5, pl)]
    XD = np.c_[phones15.feats(phones15.causal_norm(RD, fd, D, 200, pm_, pv_), D), phones16.rate(D, fd, 5, pl)]; del RC, RD
    spk = np.array([f.split("/")[-3] for f in fd]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt])
    cc = c19.ctx_true(C); Mfull = c19.fit(XC, yc, cc, 200); dims = XC.shape[1]; R = {}
    for r in (38, 60, 76):
        V = projection(XC, yc, cc, Mfull, r); UC, UD = ((XC - Mfull["m"]) / Mfull["s"]) @ V, ((XD - Mfull["m"]) / Mfull["s"]) @ V
        B0 = c19.fit(UC, yc, np.zeros(len(yc), int), 1.0); P = c19.post(B0, UD, np.zeros(len(y), int)); M = c19.fit(UC, yc, cc, 200)
        for _ in (1, 2, 3): w = c19.ctx_soft(D, P); P = c19.post_mix(M, UD, w)
        w = c19.ctx_soft(D, P); leaves = []
        _fit = c19.fit
        def rec(*a, **k): m = _fit(*a, **k); leaves.append(m); return m
        h.c19.fit = rec; Ph = h.hybrid(UC, yc, cc, UD, w, P, M, 1.0); h.c19.fit = _fit
        n = count(M, leaves, r, dims); R[f"r{r}"] = dict(flat=dict(A=c19.acc(P, y, A), B=c19.acc(P, y, ~A)), hybrid=dict(A=c19.acc(Ph, y, A), B=c19.acc(Ph, y, ~A)), numbers=int(n))
        print(f"r{r}", R[f"r{r}"], flush=True)
    ok = [k for k in R if R[k]["numbers"] < 500_000]; best = max(ok, key=lambda k: R[k]["hybrid"]["A"][1]) if ok else None
    print("chosen on half A (39-way, < 500k numbers):", best, "half B:", R[best]["hybrid"]["B"] if best else None, "vs full-size hybrid half B (0.906, 0.5655)", flush=True)
    json.dump(dict(R=R, chosen=best), open(f"{phones.OUT}/phones25.json", "w"), indent=1)
