"""5d: allophone subclasses (phones8 rules) as classes of the context model; chain emission = logsumexp over subclasses. NIGHT_REPORT 5d."""
import glob, json
import numpy as np
from scipy.special import logsumexp
import phones, phones8, phones12, phones15, phones16, phones19 as c19, phones24 as h, phones25 as b, phones33 as ch

def emis_sub(M, pcs, UD, parent):
    Z = (UD - M["m"]) / M["s"]; out = np.empty((len(UD), 5, phones.NP, 5), np.float32)
    for c in range(25):
        S = Z @ M["W"][c].T + M["b"][c]
        for k in range(phones.NP):
            cols = np.where(parent == k)[0]; out[:, c // 5, k, c % 5] = logsumexp(S[:, cols], 1) - np.log(pcs[c, cols].sum())
    return out

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC, RD = phones12.build("cal", fc, C["lens"]), phones12.build("dev_other", fd, D["lens"]); yc, y = C["segs"][:, 2], D["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    XC = np.c_[phones15.feats(phones15.causal_norm(RC, fc, C, 200, pm_, pv_), C), phones16.rate(C, fc, 5, pl)]
    XD = np.c_[phones15.feats(phones15.causal_norm(RD, fd, D, 200, pm_, pv_), D), phones16.rate(D, fd, 5, pl)]; del RC, RD
    spk = np.array([f.split("/")[-3] for f in fd]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt]); spk_of = spk[utt]
    cc = c19.ctx_true(C); Mfull = c19.fit(XC, yc, cc, 200); V = b.projection(XC, yc, cc, Mfull, 76)
    UC, UD = ((XC - Mfull["m"]) / Mfull["s"]) @ V, ((XD - Mfull["m"]) / Mfull["s"]) @ V; del XC, XD
    st = phones8.stress(fc, C["lens"]); assert len(st) == len(yc); sub_all = phones8.sublabels(C["segs"], C["lens"], st)
    M = c19.fit(UC, yc, cc, 200); logB, lst = ch.bigram(yc, C); rs, re_ = ch.runs(D)
    P = ch.decode(ch.emis(M, ch.m.ctx_priors(yc, cc), UD), rs, re_, logB, lst, 0.5, 2); Ph = h.hybrid(UC, yc, cc, UD, c19.ctx_soft(D, P), P, M, 1.0)
    R = {"chain_hybrid": dict(A=c19.acc(Ph, y, A), B=c19.acc(Ph, y, ~A))}; print(R, flush=True); out = {}
    for name, sub in (("all4", sub_all), ("vowel_stress", np.where(np.isin(yc, sorted(phones8.VOWID)), sub_all, 0))):
        lab = 2 * yc + sub; used, z = np.unique(lab, return_inverse=True); parent = used // 2
        Ms = c19.fit(UC, z, cc, 200, K=len(used)); pcs = np.zeros((25, len(used))); np.add.at(pcs, (cc, z), 1); pk = np.bincount(z, minlength=len(used)) / len(z)
        pcs = (pcs + 200 * pk) / (pcs.sum(1, keepdims=True) + 200)
        Pq = ch.decode(emis_sub(Ms, pcs, UD, parent), rs, re_, logB, lst, 0.5, 2); Pqh = h.hybrid(UC, yc, cc, UD, c19.ctx_soft(D, Pq), Pq, M, 1.0)
        R[name] = dict(subclasses=int(len(used)), chain=dict(A=c19.acc(Pq, y, A), B=c19.acc(Pq, y, ~A)), chain_hybrid=dict(A=c19.acc(Pqh, y, A), B=c19.acc(Pqh, y, ~A)),
                       per_speaker_B=ch.per_speaker(Ph, Pqh, y, spk_of, ~A)); print(name, R[name], flush=True)
    best = max(("all4", "vowel_stress"), key=lambda k: R[k]["chain_hybrid"]["A"][1]); b0, bb = R["chain_hybrid"]["B"], R[best]["chain_hybrid"]["B"]; ps = R[best]["per_speaker_B"]
    keep = ((bb[0] - b0[0] >= 0.002 and bb[1] >= b0[1]) or (bb[1] - b0[1] >= 0.002 and bb[0] >= b0[0])) and ps["improved"] * 3 >= 2 * ps["speakers"]
    print("chosen on half A:", best, "half B:", bb, "chain_hybrid half B:", b0, "KEEP" if keep else "DROP", flush=True)
    json.dump(dict(R=R, chosen=best, keep=keep), open(f"{phones.OUT}/phones38.json", "w"), indent=1)
