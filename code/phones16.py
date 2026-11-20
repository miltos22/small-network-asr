"""State channel 2: CAUSAL speaking rate on top of the causal 2 s speaker baseline (phones15, kept).
r = EMA of log segment duration over the same speaker's previous segments (corpus order = one phone session per speaker), start =
cal population mean. Variants: +r, +r and r x segment-mean tracks. Spans 5/15/50 segments, chosen on dev-other half A (level-1)."""
import glob, json
import numpy as np
from scipy.signal import lfilter
import phones, phones12, phones15

def rate(S, files, span, pop):
    spk = np.array([f.split("/")[-3] for f in files]); utt = np.searchsorted(np.cumsum(S["lens"]), S["segs"][:, 0], side="right")
    ss = spk[utt]; ln = np.log(S["segs"][:, 1] - S["segs"][:, 0]).astype(np.float64); r = np.empty_like(ln); a = 1.0 / span
    for s in dict.fromkeys(ss):
        i = np.where(ss == s)[0]; e = lfilter([a], [1, -(1 - a)], ln[i] - pop, zi=np.zeros(1))[0] + pop
        r[i] = np.r_[pop, e[:-1]]  # statistics up to the previous segment only
    return r

def segmean(Tr, segs):
    c = np.vstack([np.zeros((1, Tr.shape[1])), np.cumsum(Tr, 0)]); return (c[segs[:, 1]] - c[segs[:, 0]]) / (segs[:, 1] - segs[:, 0])[:, None]

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC, RD = phones12.build("cal", fc, C["lens"]), phones12.build("dev_other", fd, D["lens"]); yc, y = C["segs"][:, 2], D["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pop_m, pop_v = RC[spc].mean(0), RC[spc].var(0)
    TC, TD = phones15.causal_norm(RC, fc, C, 200, pop_m, pop_v), phones15.causal_norm(RD, fd, D, 200, pop_m, pop_v); del RC, RD
    spk = np.array([f.split("/")[-3] for f in fd]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt])
    XC, XD = phones15.feats(TC, C), phones15.feats(TD, D); MC, MD = segmean(TC, C["segs"]), segmean(TD, D["segs"]); del TC, TD
    pop = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    R = {"base_causal_2s": dict(A=phones15.score(XC, yc, XD, y, A), B=phones15.score(XC, yc, XD, y, ~A))}; print("base_causal_2s", R["base_causal_2s"], flush=True)
    for span in (5, 15, 50):
        rc, rd = rate(C, fc, span, pop), rate(D, fd, span, pop)
        print(f"span {span}: rate mean/sd cal {rc.mean():.3f}/{rc.std():.3f} dev {rd.mean():.3f}/{rd.std():.3f}", flush=True)
        for v, (ac, ad) in {"r": (rc[:, None], rd[:, None]), "r+int": (np.c_[rc, (rc - pop)[:, None] * MC], np.c_[rd, (rd - pop)[:, None] * MD])}.items():
            k = f"{v}_span{span}"; R[k] = dict(A=phones15.score(np.c_[XC, ac], yc, np.c_[XD, ad], y, A), B=phones15.score(np.c_[XC, ac], yc, np.c_[XD, ad], y, ~A))
            print(k, R[k], flush=True)
    best = max((k for k in R if k != "base_causal_2s"), key=lambda k: R[k]["A"][0]); b0 = R["base_causal_2s"]["B"]
    keep = R[best]["B"][0] > b0[0] and R[best]["B"][1] >= b0[1]
    print("chosen on half A (level-1):", best, "half B:", R[best]["B"], "base half B:", b0, "KEEP" if keep else "DROP", flush=True)
    json.dump(dict(R=R, chosen=best, keep=keep), open(f"{phones.OUT}/phones16.json", "w"), indent=1)
