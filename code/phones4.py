"""Module 2 diagnosis step 3: do the 27 hand-picked tracks lose information that the plain spectrum has?
Segment vectors (phones3.segvec) from: SEGMEL = 40 log-mel bands (classic segment-classifier input; control),
SEGALL = 22 detector log-odds + 108 raw measurements + 40 log-mel (detectors as named extra channels, not a funnel).
Fit on cal speakers 1-5, validate on 6-10, then ONE test-clean run of both. Same pass marks (65 / 86)."""
import json, os
from multiprocessing import Pool
import numpy as np
import frontend as fe
import phones, phones2, phones3

def job(path):
    x = fe.dither(fe.load(path), np.random.default_rng(abs(hash(path)) % 2 ** 32))
    lm = fe.tracks(x)["logmel"]; T = len(lm)
    lm = lm - lm.mean()  # utterance-level gain removal (shortcut: offline mean; streaming would use a running mean)
    tg = path.replace(f"{phones.DATA}/LibriSpeech", f"{phones.DATA}/align").replace(".flac", ".TextGrid")
    segs = []
    if os.path.exists(tg):
        for i0, i1, p in phones.segments(tg):
            i1 = min(i1, T)
            if i1 > i0 and p in phones.PID:
                segs.append((i0, i1, phones.PID[p]))
    segs = np.array(segs, np.int32).reshape(-1, 3)
    return phones3.segvec(lm, segs, T)

def mel_segs(name):
    p = f"{phones.CACHE}/segmel_{name}.npy"
    if not os.path.exists(p):
        with Pool(4) as pool:
            np.save(p, np.concatenate(pool.map(job, phones.split()[name], chunksize=8)))
    return np.load(p)

if __name__ == "__main__":
    model = np.load(f"{phones.OUT}/model.npy", allow_pickle=True).item()
    R = {}
    C = phones.load("cal"); fr = phones2.split_cal(C); segs = C["segs"]; y = segs[:, 2]; A = fr[segs[:, 0]]
    M = mel_segs("cal"); assert len(M) == len(segs), (len(M), len(segs))
    L = phones.detector_logodds(C, model)
    feats = {"SEGMEL": M, "SEGALL": np.concatenate([phones3.segvec(L, segs, len(L)), phones3.segvec(C["X"].astype(np.float64), segs, len(L)), M], 1)}
    for k, X in feats.items():
        c = phones3.lda_fit(X[A], y[A], phones.NP); t1, t3 = phones3.acc(c, X[~A], y[~A])
        R[k] = dict(val_top1=round(t1, 4), val_top3=round(t3, 4), stored=int(c["n"])); print(k, R[k], flush=True)
    T = phones.load("test"); MT = mel_segs("test"); assert len(MT) == len(T["segs"])
    LT = phones.detector_logodds(T, model); yt = T["segs"][:, 2]
    tf = {"SEGMEL": MT, "SEGALL": np.concatenate([phones3.segvec(LT, T["segs"], len(LT)), phones3.segvec(T["X"].astype(np.float64), T["segs"], len(LT)), MT], 1)}
    for k, X in feats.items():
        c = phones3.lda_fit(X, y, phones.NP); t1, t3 = phones3.acc(c, tf[k], yt)
        R[k].update(TEST_top1=round(t1, 4), TEST_top3=round(t3, 4)); print("TEST", k, R[k], flush=True)
    json.dump(R, open(f"{phones.OUT}/eval_channels.json", "w"), indent=1)
