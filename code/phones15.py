"""State channel 1: CAUSAL speaker baseline (live version of the offline per-speaker normalisation, which gave +3.1).
Per speaker stream (utterances in corpus order, like one session on the phone), running mean and spread of each of the 31 raw tracks,
updated only on speech frames (oracle labels here; the gate later), applied to frame t with the statistics up to t-1.
Start value = population mean/spread from the dev-clean cal speakers. Time constants tau 2 s / 10 s / 60 s of speech, chosen on
dev-other half A (level-1 3-way, then 39-way), reported on half B. References: no normalisation, offline per-speaker normalisation.
Features: DUAL window segvec + INT (phones13). LDA fit on cal (normalised the same way)."""
import glob
import numpy as np
from scipy.signal import lfilter
import frontend as fe
import phones, phones3, phones11, phones12, phones13, phones14

def causal_norm(Tr, files, D, tau_frames, pop_m, pop_v):
    spk = np.array([f.split("/")[-3] for f in files]); fs = np.repeat(spk, D["lens"]); X = Tr.astype(np.float64); out = np.empty_like(X)
    sp = (D["ph"] >= 0) & (D["ph"] != phones.SILID); a = 1.0 / tau_frames
    for s in dict.fromkeys(spk):
        idx = np.where(fs == s)[0]; Xs = X[idx]; m = sp[idx]; Z = Xs[m]
        mz = lfilter([a], [1, -(1 - a)], Z - pop_m, axis=0, zi=np.zeros((1, Z.shape[1])))[0] + pop_m  # EMA from the population start
        vz = lfilter([a], [1, -(1 - a)], (Z - mz) ** 2 - pop_v, axis=0, zi=np.zeros((1, Z.shape[1])))[0] + pop_v
        M = np.full_like(Xs, np.nan); V = np.full_like(Xs, np.nan); M[np.where(m)[0]] = mz; V[np.where(m)[0]] = vz
        prevM = np.vstack([pop_m, M[:-1]]); prevV = np.vstack([pop_v, V[:-1]])  # statistics up to t-1
        for A_, start in ((prevM, pop_m), (prevV, pop_v)):  # forward-fill through non-speech frames
            ok = ~np.isnan(A_[:, 0]); last = np.maximum.accumulate(np.where(ok, np.arange(len(A_)), -1)); A_[:] = np.where((last >= 0)[:, None], A_[np.maximum(last, 0)], start)
        out[idx] = (Xs - prevM) / np.sqrt(prevV + 1e-9)
    return out

def feats(Tr, S):
    Xw = phones12.project(Tr, S["lens"], [fe._legendre(2, 3), fe._legendre(6, 4)])
    return np.c_[phones3.segvec(Xw, S["segs"], len(Xw)), phones13.seg_features(Tr, S["segs"], S["lens"])]

def score(XC, yc, XD, y, sel):
    c = phones3.lda_fit(XC, yc, phones.NP); P = ((XD[sel] - c["m"]) / c["s"]) @ c["W"].T + c["b"]; pr = P.argmax(1)
    mo = phones14.mac_of; E = np.exp(P - P.max(1, keepdims=True)); pm = np.stack([E[:, mo == k].sum(1) for k in range(3)], 1).argmax(1)
    return round(float(np.mean(pm == mo[y[sel]])), 4), round(float(np.mean(pr == y[sel])), 4)

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC, RD = phones12.build("cal", fc, C["lens"]), phones12.build("dev_other", fd, D["lens"]); yc, y = C["segs"][:, 2], D["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pop_m, pop_v = RC[spc].mean(0), RC[spc].var(0)
    spk = np.array([f.split("/")[-3] for f in fd]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt])
    R = {}
    for name, (TC, TD) in {"none": ((RC - pop_m) / np.sqrt(pop_v), (RD - pop_m) / np.sqrt(pop_v)),
                           "offline_per_speaker": (phones11.spk_norm(RC, fc, C), phones11.spk_norm(RD, fd, D))}.items():
        XC, XD = feats(TC, C), feats(TD, D); R[name] = dict(A=score(XC, yc, XD, y, A), B=score(XC, yc, XD, y, ~A)); print(name, R[name], flush=True)
    for tau_s in (2, 10, 60):
        tau = tau_s * 100
        XC, XD = feats(causal_norm(RC, fc, C, tau, pop_m, pop_v), C), feats(causal_norm(RD, fd, D, tau, pop_m, pop_v), D)
        R[f"causal_{tau_s}s"] = dict(A=score(XC, yc, XD, y, A), B=score(XC, yc, XD, y, ~A)); print(f"causal_{tau_s}s", R[f"causal_{tau_s}s"], flush=True)
    best = max((k for k in R if k.startswith("causal")), key=lambda k: R[k]["A"][0]); print("chosen on half A (level-1):", best, "half B:", R[best]["B"], flush=True)
