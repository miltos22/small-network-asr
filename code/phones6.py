"""Phoneme layer 2 (context). Layer 1 = SEGALL segmental LDA (fit on dev-clean cal speakers, unchanged).
Layer 2 job: correct each phone using its NEIGHBOURS' layer-1 probabilities (coarticulation: m/n, d/n, unstressed ɪ/ʌ are decided
by what comes before/after). Inputs per segment: layer-1 log-probs of previous, current, next phone segment in the utterance
(3 x 39), gaps (silence frames) before/after, log duration. Variant L2+: the same plus the current segment's layer-1 inputs.
Closed form (LDA over 39 phones). Calibrated on NEW, messier data: dev-other speakers half A; chosen on half B; test once
(test-clean, test-other). Target set by the author: phoneme top-1 >= 80 %."""
import glob, json
import numpy as np
import phones, phones3, phones5

def l1_inputs(D, M, model):
    L = phones.detector_logodds(D, model)
    return np.concatenate([phones3.segvec(L, D["segs"], len(L)), phones3.segvec(D["X"].astype(np.float64), D["segs"], len(L)), M], 1)

def logp(c, X):
    S = ((X - c["m"]) / c["s"]) @ c["W"].T + c["b"]; return S - np.logaddexp.reduce(S, 1, keepdims=True)

def context(P, segs, lens):
    """prev/cur/next log-probs within the same utterance (edge -> uniform), gaps and duration."""
    n = len(segs); utt = np.searchsorted(np.cumsum(lens), segs[:, 0], side="right")
    U = np.full(P.shape[1], -np.log(P.shape[1]))
    same_prev = np.r_[False, utt[1:] == utt[:-1]]; same_next = np.r_[utt[1:] == utt[:-1], False]
    Pp = np.where(same_prev[:, None], np.vstack([U, P[:-1]]), U); Pn = np.where(same_next[:, None], np.vstack([P[1:], U]), U)
    gp = np.where(same_prev, segs[:, 0] - np.r_[0, segs[:-1, 1]], 30); gn = np.where(same_next, np.r_[segs[1:, 0], 0] - segs[:, 1], 30)
    dur = segs[:, 1] - segs[:, 0]
    return np.concatenate([Pp, P, Pn, np.log1p(np.c_[gp, gn, dur].clip(0))], 1)

def acc(c, X, y):
    return phones3.acc(c, X, y)

if __name__ == "__main__":
    model = np.load(f"{phones.OUT}/model.npy", allow_pickle=True).item()
    C = phones.load("cal"); MC = np.load(f"{phones.CACHE}/segmel_cal.npy")
    l1 = phones3.lda_fit(l1_inputs(C, MC, model), C["segs"][:, 2], phones.NP)
    files = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    D, MD = phones5.build("dev_other", files)
    X1 = l1_inputs(D, MD, model); y = D["segs"][:, 2]; P = logp(l1, X1)
    spk = np.array([f.split("/")[-3] for f in files]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt])
    variants = {"L1_only": X1, "L2": context(P, D["segs"], D["lens"]), "L2plus": np.concatenate([context(P, D["segs"], D["lens"]), X1], 1)}
    R = {}
    for k, X in variants.items():
        if k == "L1_only":
            t1, t3 = phones3.acc(l1, X[~A], y[~A]); R[k] = dict(val_top1=round(t1, 4), val_top3=round(t3, 4))
        else:
            c = phones3.lda_fit(X[A], y[A], phones.NP); t1, t3 = phones3.acc(c, X[~A], y[~A])
            R[k] = dict(val_top1=round(t1, 4), val_top3=round(t3, 4), stored=int(c["n"]))
        print(k, R[k], flush=True)
    best = max(("L2", "L2plus"), key=lambda k: R[k]["val_top1"]); R["chosen"] = best; print("chosen on dev-other half B:", best, flush=True)
    c2 = phones3.lda_fit(variants[best], y, phones.NP)  # refit on all dev-other
    for name in ("test", "test_other"):
        T = phones.load(name); MT = np.load(f"{phones.CACHE}/segmel_{name}.npy"); XT = l1_inputs(T, MT, model); yt = T["segs"][:, 2]
        PT = logp(l1, XT); V = context(PT, T["segs"], T["lens"]) if best == "L2" else np.concatenate([context(PT, T["segs"], T["lens"]), XT], 1)
        a1 = phones3.acc(l1, XT, yt); a2 = phones3.acc(c2, V, yt)
        R[f"TEST_{name}"] = dict(L1_top1=round(a1[0], 4), L1_top3=round(a1[1], 4), L2_top1=round(a2[0], 4), L2_top3=round(a2[1], 4))
        print("TEST", name, R[f"TEST_{name}"], flush=True)
    json.dump(R, open(f"{phones.OUT}/eval_layer2.json", "w"), indent=1)
