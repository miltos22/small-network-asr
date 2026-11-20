"""Tree level 1 (reviewer r4 / the author): 3 macro classes vowel | sonorant consonant (nasals + L R W Y) | obstruent, on the DUAL+INT
segment features (includes the syllable-dip measures). Compared with the flat 39-way LDA summed per macro class. Fit on dev-clean cal,
scored on dev-other. Then the same with decoupled anchored post-training (phones10.anchored_lr, lam 1e-3)."""
import numpy as np
import af, phones, phones3, phones10

MAC = {p: 0 for p in af.VOW} | {p: 1 for p in "M N NG L R W Y".split()}
mac_of = np.array([MAC.get(p, 2) for p in af.PHONES])

if __name__ == "__main__":
    S = np.load(f"{phones.CACHE}/seg13.npz"); C, D = phones.load("cal"), phones.load("dev_other")
    XC, XD = np.c_[S["C_DUAL"], S["C_INT"]], np.c_[S["D_DUAL"], S["D_INT"]]; yc, y = C["segs"][:, 2], D["segs"][:, 2]
    mc, md = mac_of[yc], mac_of[y]
    flat = phones3.lda_fit(XC, yc, phones.NP); P = np.exp(((XD - flat["m"]) / flat["s"]) @ flat["W"].T + flat["b"])
    Pm = np.stack([P[:, mac_of == k].sum(1) for k in range(3)], 1).argmax(1)
    def rep(name, pm):
        M = np.zeros((3, 3), int); np.add.at(M, (md, pm), 1)
        print(f"{name:26s} 3-way={np.mean(pm == md):.3f} vowel/consonant={np.mean((pm == 0) == (md == 0)):.3f} sonorant/obstruent={np.mean((pm == 2) == (md == 2)):.3f}  rows true V/S/O:", M.tolist(), flush=True)
    rep("flat 39 summed", Pm)
    c3 = phones3.lda_fit(XC, mc, 3); rep("level-1 LDA (3 classes)", ((((XD - c3["m"]) / c3["s"]) @ c3["W"].T + c3["b"])).argmax(1))
    r3 = phones10.anchored_lr(c3, XC, mc, 1e-3); rep("level-1 LDA + post-trained", ((((XD - r3["m"]) / r3["s"]) @ r3["W"].T + r3["b"])).argmax(1))
