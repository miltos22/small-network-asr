"""Soft 3-level tree: manner -> 9 categories -> phones, each node its own context model on its own segments. NIGHT_REPORT step 2."""
import glob, json
import numpy as np
import af, phones, phones12, phones14, phones15, phones16, phones19 as c19

CATS = ["IY IH", "EH AE AH AA", "UW UH OW AO", "ER", "AY AW OY EY", "M N NG", "L R W Y", "P B T D K G CH JH", "F V TH DH S Z SH ZH HH"]
CAT_OF = np.empty(phones.NP, int)
for c, ps in enumerate(CATS):
    for p in ps.split(): CAT_OF[phones.PID[p]] = c
assert sorted(sum((c.split() for c in CATS), [])) == sorted(af.PHONES)
MAN_OF_CAT = np.array([phones14.mac_of[phones.PID[c.split()[0]]] for c in CATS])
assert all((phones14.mac_of[phones.PID[p]] == MAN_OF_CAT[c]) for c, ps in enumerate(CATS) for p in ps.split())
DUAL = {3: ["R"], 0: ["Y"], 2: ["W"], 8: ["CH"]}  # extra leaf members (partner phones)

def tree(XC, yc, cc, XD, w, dual, tau=200):
    def node(sel, lab, K): return c19.post_mix(c19.fit(XC[sel], lab, cc[sel], tau, K=K), XD, w)
    P1 = node(np.ones(len(yc), bool), phones14.mac_of[yc], 3); Pc = np.zeros((len(XD), len(CATS)))
    for m in range(3):
        cs = np.where(MAN_OF_CAT == m)[0]; sel = phones14.mac_of[yc] == m
        Pc[:, cs] = P1[:, m:m + 1] * (node(sel, np.searchsorted(cs, CAT_OF[yc[sel]]), len(cs)) if len(cs) > 1 else 1.0)
    P = np.zeros((len(XD), phones.NP))
    for c, ps in enumerate(CATS):
        leaf = [phones.PID[p] for p in ps.split() + (DUAL.get(c, []) if dual else [])]
        if len(leaf) == 1: P[:, leaf[0]] += Pc[:, c]; continue
        sel = np.isin(yc, leaf); P[:, leaf] += Pc[:, c:c + 1] * node(sel, np.searchsorted(np.sort(leaf), yc[sel]), len(leaf))[:, np.argsort(np.argsort(leaf))]
    assert np.allclose(P.sum(1), 1, atol=1e-6), P.sum(1)[:5]
    return P

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC, RD = phones12.build("cal", fc, C["lens"]), phones12.build("dev_other", fd, D["lens"]); yc, y = C["segs"][:, 2], D["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    XC = np.c_[phones15.feats(phones15.causal_norm(RC, fc, C, 200, pm_, pv_), C), phones16.rate(C, fc, 5, pl)]
    XD = np.c_[phones15.feats(phones15.causal_norm(RD, fd, D, 200, pm_, pv_), D), phones16.rate(D, fd, 5, pl)]; del RC, RD
    print("cal segments", len(yc), "dims", XC.shape[1], "per category", np.bincount(CAT_OF[yc], minlength=9).tolist(), flush=True)
    spk = np.array([f.split("/")[-3] for f in fd]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt])
    cc = c19.ctx_true(C); B0 = c19.fit(XC, yc, np.zeros(len(yc), int), 1.0); P = c19.post(B0, XD, np.zeros(len(y), int)); M = c19.fit(XC, yc, cc, 200)
    for r in (1, 2, 3): w = c19.ctx_soft(D, P); P = c19.post_mix(M, XD, w)
    w = c19.ctx_soft(D, P); R = {"flat": dict(A=c19.acc(P, y, A), B=c19.acc(P, y, ~A))}; print("flat (phones20 check)", R["flat"], flush=True)
    cat_flat = np.stack([P[:, CAT_OF == c].sum(1) for c in range(9)], 1).argmax(1)
    for name, dual in (("tree", False), ("tree_dual", True)):
        Pt = tree(XC, yc, cc, XD, w, dual); ct = np.stack([Pt[:, CAT_OF == c].sum(1) for c in range(9)], 1).argmax(1)
        R[name] = dict(A=c19.acc(Pt, y, A), B=c19.acc(Pt, y, ~A), cat9_B=round(float(np.mean(ct[~A] == CAT_OF[y[~A]])), 4))
        print(name, R[name], "flat cat9_B", round(float(np.mean(cat_flat[~A] == CAT_OF[y[~A]])), 4), flush=True)
    best = max(("tree", "tree_dual"), key=lambda k: R[k]["A"][1]); b0, b = R["flat"]["B"], R[best]["B"]
    keep = (b[0] - b0[0] >= 0.002 and b[1] >= b0[1]) or (b[1] - b0[1] >= 0.002 and b[0] >= b0[0])
    print("chosen on half A (39-way):", best, "half B:", b, "flat half B:", b0, "KEEP" if keep else "DROP", flush=True)
    json.dump(dict(R=R, chosen=best, keep=keep), open(f"{phones.OUT}/phones23.json", "w"), indent=1)
