"""Segment features read INSIDE the segment from raw 10 ms frames (reviewer round 4, step 1). Same 31 speaker-normalised raw tracks
as phones12 (no window smoothing). Per segment:
  interior samples at 20/50/80 % (clamped to [t1, t2-1]) + interior mean                              (4 x 31)
  context: mean of the 3 frames before t1 and after t2                                               (2 x 31)
  syllable dip on Erel and R: value at mid - mean(max of 120 ms before t1, max of 120 ms after t2)   (2)  peak vs trough
  stationarity: |x(mid)-x(t1)| + |x(t2-1)-x(mid)| over formant + cepstral tracks, per frame           (1)
  burst anchor t* = argmax positive flux step inside the segment: tracks at t*-1, t*, t*+2 + t* position (3 x 31 + 1)
  log duration                                                                                        (1)
Variants: DUAL (phones12 reference 50.5), INT (above), DUAL+INT. LDA fit on dev-clean cal, scored on dev-other."""
import glob
import numpy as np
import frontend as fe
import phones, phones3, phones8, phones11, phones12

N = phones12.NAMES; I_E, I_R, I_FL = N.index("Erel"), N.index("R"), N.index("flux")
SHAPE = [N.index(t) for t in ("C1", "C2", "C3", "C4", "C5", "lF1", "lF2", "lF3")]

def seg_features(X, segs, lens):
    T = len(X); ends = np.cumsum(lens); starts = ends - lens
    utt = np.searchsorted(ends, segs[:, 0], side="right"); out = np.empty((len(segs), 9 * X.shape[1] + 5))
    for j, (t1, t2, _) in enumerate(segs):
        u0, u1 = starts[utt[j]], ends[utt[j]]; n = t2 - t1; last = t2 - 1
        at = lambda q: X[min(last, max(t1, t1 + int(round(q * (n - 1)))))]
        mid = t1 + (n - 1) // 2
        pre = X[max(u0, t1 - 3):t1] if t1 > u0 else X[t1:t1 + 1]; post = X[t2:min(u1, t2 + 3)] if t2 < u1 else X[last:t2]
        dips = []
        for i in (I_E, I_R):
            a = X[max(u0, t1 - 12):t1, i]; b = X[t2:min(u1, t2 + 12), i]
            ref = np.mean([v.max() for v in (a, b) if len(v)]) if (len(a) or len(b)) else X[mid, i]
            dips.append(X[mid, i] - ref)
        stat = (np.abs(X[mid, SHAPE] - X[t1, SHAPE]).sum() + np.abs(X[last, SHAPE] - X[mid, SHAPE]).sum()) / n
        fl = X[t1:t2, I_FL]; ts = t1 + int(np.argmax(np.r_[0, np.diff(fl)])) if n > 1 else t1
        out[j] = np.concatenate([at(0.2), at(0.5), at(0.8), X[t1:t2].mean(0), pre.mean(0), post.mean(0),
                                 X[max(t1, ts - 1)], X[ts], X[min(last, ts + 2)], dips, [stat, (ts - t1) / max(1, n - 1), np.log(n)]])
    return out

if __name__ == "__main__":
    VOW = np.array(sorted(phones8.VOWID))
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    TC = phones11.spk_norm(phones12.build("cal", fc, C["lens"]), fc, C); TD = phones11.spk_norm(phones12.build("dev_other", fd, D["lens"]), fd, D)
    W = [fe._legendre(2, 3), fe._legendre(6, 4)]
    feats = {}
    for nm, Tr, S in (("C", TC, C), ("D", TD, D)):
        Xw = phones12.project(Tr, S["lens"], W)
        feats[nm] = dict(DUAL=phones3.segvec(Xw, S["segs"], len(Xw)), INT=seg_features(Tr, S["segs"], S["lens"])); del Xw
        feats[nm]["DUAL+INT"] = np.c_[feats[nm]["DUAL"], feats[nm]["INT"]]
    y = D["segs"][:, 2]; v = np.isin(y, VOW); yc = C["segs"][:, 2]
    for k in ("DUAL", "INT", "DUAL+INT"):
        c = phones3.lda_fit(feats["C"][k], yc, phones.NP); pr = (((feats["D"][k] - c["m"]) / c["s"]) @ c["W"].T + c["b"]).argmax(1)
        print(f"{k:9s} dims={feats['C'][k].shape[1]:5d} all={np.mean(pr == y):.3f} vowels={np.mean(pr[v] == y[v]):.3f} consonants={np.mean(pr[~v] == y[~v]):.3f}", flush=True)
    np.savez(f"{phones.CACHE}/seg13.npz", C_INT=feats["C"]["INT"], D_INT=feats["D"]["INT"], C_DUAL=feats["C"]["DUAL"], D_DUAL=feats["D"]["DUAL"])
