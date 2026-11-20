"""Post-training, decoupled (the author's idea, reviewer round 3 design): start from the closed-form LDA weights and adjust them by
cross-entropy on the layer's OWN phone labels, pulled back toward the designed values (lam * ||W - W_LDA||^2). Detectors frozen
(no gradient crosses a module boundary -> no co-adaptation). Same parameter form as LDA, so size is unchanged.
L1 fit on dev-clean cal; anchor lam chosen on dev-other half A; reported on half B; L2plus (LDA, and anchored-LR variant) on top
fit on half A, scored on half B vs 51.45 %. Test once if the best variant wins."""
import glob, json
import numpy as np
from scipy.optimize import minimize
import phones, phones3, phones5, phones6

def anchored_lr(c, X, y, lam, iters=200):
    Z = (X - c["m"]) / c["s"]; K, d = c["W"].shape; W0 = np.c_[c["W"], c["b"]]; Za = np.c_[Z, np.ones(len(Z))]; Y = np.eye(K)[y]
    def f(w):
        W = w.reshape(K, d + 1); S = Za @ W.T; S -= S.max(1, keepdims=True); P = np.exp(S); P /= P.sum(1, keepdims=True)
        loss = -np.log(P[np.arange(len(y)), y] + 1e-12).mean() + lam * ((W - W0) ** 2).sum()
        return loss, (((P - Y).T @ Za) / len(y) + 2 * lam * (W - W0)).ravel()
    w = minimize(f, W0.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=iters)).x.reshape(K, d + 1)
    return dict(c, W=w[:, :d], b=w[:, d])

def top(P, y):
    o = np.argsort(-P, 1); return round(float((o[:, 0] == y).mean()), 4), round(float((o[:, :3] == y[:, None]).any(1).mean()), 4)

if __name__ == "__main__":
    model = np.load(f"{phones.OUT}/model.npy", allow_pickle=True).item()
    C = phones.load("cal"); MC = np.load(f"{phones.CACHE}/segmel_cal.npy"); XC = phones6.l1_inputs(C, MC, model); yc = C["segs"][:, 2]
    l1 = phones3.lda_fit(XC, yc, phones.NP); del C
    files = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    D, MD = phones5.build("dev_other", files); y = D["segs"][:, 2]; X1 = phones6.l1_inputs(D, MD, model)
    spk = np.array([f.split("/")[-3] for f in files]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt])
    R = {"L1_lda_valB": top(phones6.logp(l1, X1)[~A], y[~A]), "L1_lr_halfA": {}}
    fits = {}
    for lam in (1e-2, 1e-3, 1e-4):
        fits[lam] = anchored_lr(l1, XC, yc, lam); R["L1_lr_halfA"][str(lam)] = top(phones6.logp(fits[lam], X1[A]), y[A]); print(lam, R["L1_lr_halfA"][str(lam)], flush=True)
    lam = max(fits, key=lambda k: R["L1_lr_halfA"][str(k)][0]); l1r = fits[lam]; R["lam1"] = lam
    P = phones6.logp(l1r, X1); R["L1_lr_valB"] = top(P[~A], y[~A])
    V = np.concatenate([phones6.context(P, D["segs"], D["lens"]), X1], 1); VA, VB = V[A], V[~A]
    c2 = phones3.lda_fit(VA, y[A], phones.NP); R["L2plus_lda_valB"] = list(phones3.acc(c2, VB, y[~A]))
    # anchored LR for layer 2: lam chosen on a speaker split INSIDE half A (first/second half of A's speakers), scored on B
    a_spk = sorted(A_spk); A1 = np.array([spk[u] in set(a_spk[: len(a_spk) // 2]) for u in utt])[A]
    c2a = phones3.lda_fit(VA[A1], y[A][A1], phones.NP); inner = {}
    for lam2 in (1e-2, 1e-3, 1e-4):
        inner[lam2] = top(phones6.logp(anchored_lr(c2a, VA[A1], y[A][A1], lam2), VA[~A1]), y[A][~A1])[0]
    lam2 = max(inner, key=inner.get); R["lam2_inner"] = {str(k): v for k, v in inner.items()}; R["lam2"] = lam2
    c2r = anchored_lr(c2, VA, y[A], lam2); R["L2plus_lr_valB"] = list(phones3.acc(c2r, VB, y[~A])); R["ref_L2plus_lda_unpost"] = 0.5145
    print(R, flush=True); json.dump(R, open(f"{phones.OUT}/eval_posttrain.json", "w"), indent=1)
