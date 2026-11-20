"""Module 2 diagnosis step 2: SEGMENTAL scoring. Frame-sum scoring (v1/phones2) treats frames as independent votes; a phone is a
shape over time. Segment vector = mean of the inputs in each third of the segment + 30 ms before + 30 ms after (coarticulation
context) + log duration. Classifier = LDA over 39 phones (closed form). Inputs compared:
  SEG22 : the 22 detector log-odds (the design)          SEG108: the 108 raw measurements (control)
Fit on cal speakers 1-5, validate on 6-10; then refit on all cal and ONE test-clean run of both."""
import json
import numpy as np
import af, phones, phones2

def segvec(F, segs, T):
    out = np.empty((len(segs), 5 * F.shape[1] + 1))
    for j, (i0, i1, _) in enumerate(segs):
        n = i1 - i0; a, b = i0 + max(1, n // 3), i0 + max(2, 2 * n // 3)
        parts = [F[i0:min(a, i1)], F[min(a, i1 - 1):min(b, i1)], F[min(b, i1 - 1):i1], F[max(0, i0 - 3):i0] if i0 > 0 else F[i0:i0 + 1],
                 F[i1:min(T, i1 + 3)] if i1 < T else F[i1 - 1:i1]]
        out[j] = np.concatenate([p.mean(0) if len(p) else F[i0] for p in parts] + [[np.log(n)]])
    return out

def lda_fit(X, y, K, lam=0.05):
    m, s = X.mean(0), X.std(0) + 1e-9; Z = (X - m) / s
    mu = np.stack([Z[y == k].mean(0) for k in range(K)]); R = Z - mu[y]
    S = R.T @ R / len(R) + lam * np.eye(Z.shape[1]); W = np.linalg.solve(S, mu.T).T
    lp = np.log(np.bincount(y, minlength=K) / len(y))
    return dict(m=m, s=s, W=W, b=-0.5 * np.einsum("kj,kj->k", W, mu) + lp, n=W.size + K + 2 * len(m))

def acc(c, X, y):
    S = ((X - c["m"]) / c["s"]) @ c["W"].T + c["b"]; o = np.argsort(-S, 1)
    return float((o[:, 0] == y).mean()), float((o[:, :3] == y[:, None]).any(1).mean())

if __name__ == "__main__":
    model = np.load(f"{phones.OUT}/model.npy", allow_pickle=True).item()
    C = phones.load("cal"); fr = phones2.split_cal(C); L = phones.detector_logodds(C, model)
    segs = C["segs"]; y = segs[:, 2]; A = fr[segs[:, 0]]
    feats = {"SEG22": segvec(L, segs, len(L)), "SEG108": segvec(C["X"].astype(np.float64), segs, len(L))}
    R = {}
    for k, X in feats.items():
        c = lda_fit(X[A], y[A], phones.NP); t1, t3 = acc(c, X[~A], y[~A])
        R[k] = dict(val_top1=round(t1, 4), val_top3=round(t3, 4), stored=int(c["n"])); print(k, R[k], flush=True)
    T = phones.load("test"); LT = phones.detector_logodds(T, model); yt = T["segs"][:, 2]
    tf = {"SEG22": segvec(LT, T["segs"], len(LT)), "SEG108": segvec(T["X"].astype(np.float64), T["segs"], len(LT))}
    conf = {}
    for k, X in feats.items():
        c = lda_fit(X, y, phones.NP); t1, t3 = acc(c, tf[k], yt)
        R[k].update(TEST_top1=round(t1, 4), TEST_top3=round(t3, 4)); print("TEST", k, R[k], flush=True)
        if k == "SEG22":
            pred = (((tf[k] - c["m"]) / c["s"]) @ c["W"].T + c["b"]).argmax(1)
            M = np.zeros((phones.NP, phones.NP), int); np.add.at(M, (yt, pred), 1)
            R["SEG22_per_phone"] = {af.PHONES[i]: round(M[i, i] / max(1, M[i].sum()), 3) for i in range(phones.NP)}
            R["SEG22_top_conf"] = sorted([(af.PHONES[i], af.PHONES[j], int(M[i, j])) for i in range(phones.NP) for j in range(phones.NP) if i != j], key=lambda t: -t[2])[:12]
    R["PASS_SEG22"] = dict(top1=R["SEG22"]["TEST_top1"] >= 0.65, top3=R["SEG22"]["TEST_top3"] >= 0.86)
    json.dump(R, open(f"{phones.OUT}/eval_segmental.json", "w"), indent=1, default=str); print(json.dumps(R, indent=1, default=str))
