"""Robustness test: the current best phoneme module (SEGALL = detectors + 108 measurements + 40 mel, segmental LDA; and SEG108
control), calibrated exactly as before (dev-clean only), run ONCE on LibriSpeech test-other (harder speakers / recordings).
Compare to test-clean: SEGALL 61.4 / 87.7, SEG108 57.4 / 85.2."""
import glob, json, os
from multiprocessing import Pool
import numpy as np
import phones, phones3, phones4

def build(name, files):
    p = f"{phones.CACHE}/ph_{name}.npz"
    if not os.path.exists(p):
        with Pool(4) as pool: res = pool.map(phones.job, files, chunksize=8)
        off = np.cumsum([0] + [len(r[1]) for r in res])[:-1]
        segs = np.concatenate([r[5] + np.array([o, o, 0]) for r, o in zip(res, off)])
        np.savez(p, X=np.concatenate([r[0] for r in res]), ph=np.concatenate([r[1] for r in res]), sub=np.concatenate([r[2] for r in res]),
                 half=np.concatenate([r[3] for r in res]), flux=np.concatenate([r[6] for r in res]), segs=segs, lens=np.array([len(r[1]) for r in res]))
    q = f"{phones.CACHE}/segmel_{name}.npy"
    if not os.path.exists(q):
        with Pool(4) as pool: np.save(q, np.concatenate(pool.map(phones4.job, files, chunksize=8)))
    return phones.load(name), np.load(q)

if __name__ == "__main__":
    model = np.load(f"{phones.OUT}/model.npy", allow_pickle=True).item()
    C = phones.load("cal"); MC = np.load(f"{phones.CACHE}/segmel_cal.npy"); L = phones.detector_logodds(C, model); y = C["segs"][:, 2]
    tr = {"SEGALL": np.concatenate([phones3.segvec(L, C["segs"], len(L)), phones3.segvec(C["X"].astype(np.float64), C["segs"], len(L)), MC], 1),
          "SEG108": phones3.segvec(C["X"].astype(np.float64), C["segs"], len(L))}
    files = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/test-other/*/*/*.flac"))
    T, MT = build("test_other", files); assert len(MT) == len(T["segs"])
    LT = phones.detector_logodds(T, model); yt = T["segs"][:, 2]
    te = {"SEGALL": np.concatenate([phones3.segvec(LT, T["segs"], len(LT)), phones3.segvec(T["X"].astype(np.float64), T["segs"], len(LT)), MT], 1),
          "SEG108": phones3.segvec(T["X"].astype(np.float64), T["segs"], len(LT))}
    R = {"n_segments": int(len(yt))}
    keep = T["ph"] >= 0; AF = phones.af_labels(np.where(keep, T["ph"], phones.SILID), T["half"])
    au = [phones.auroc(LT[keep, i], AF[keep, i]) for i in range(LT.shape[1])]; R["detector_auroc_mean"] = float(np.mean(au)); R["detector_auroc_min"] = float(np.min(au))
    for k in tr:
        c = phones3.lda_fit(tr[k], y, phones.NP); t1, t3 = phones3.acc(c, te[k], yt); R[k] = dict(top1=round(t1, 4), top3=round(t3, 4))
    print(json.dumps(R, indent=1)); json.dump(R, open(f"{phones.OUT}/eval_test_other.json", "w"), indent=1)
