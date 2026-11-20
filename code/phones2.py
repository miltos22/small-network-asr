"""Module 2 composer diagnosis. v1: detectors pass (AUROC 0.871 mean, 0.787 min) but oracle phone accuracy fails
(top-1 41.7 % / top-3 70.9 % vs 65 / 86). Which composer loses the information? Candidates, all closed form:
  LDA22  : shared covariance over the 22 log-odds at frame t (v1)
  QDA22  : one shrunk covariance per state
  LDA66  : the 22 log-odds' trajectories (Legendre level/slope/curvature over +-80 ms), shared covariance
  QDA66d : same inputs, per-state covariance shrunk 50 % to diagonal
  CTRL108: control, NOT the design - LDA straight on the 108 raw measurements (no detector layer)
Fit on cal speakers 1-5, choose on cal speakers 6-10 (oracle top-1), then ONE test-clean run of the chosen composer + control."""
import json, os
import numpy as np
import frontend as fe
import af, phones

def split_cal(D):
    files = phones.split()["cal"]; spk = np.array([f.split("/")[-3] for f in files])
    order = sorted(set(spk)); A = set(order[:5])
    fr = np.repeat([s in A for s in spk], D["lens"])
    return fr

def ctx(L, lens):
    out = np.empty((len(L), 3 * L.shape[1])); i = 0
    for n in lens:
        out[i:i + n] = np.concatenate([fe.legendre(L[i:i + n, k]) for k in range(L.shape[1])], 1); i += n
    return out

def states(D):
    return np.where(D["ph"] == phones.SILID, 3 * phones.NP, D["ph"].astype(np.int64) * 3 + D["sub"])

def fit_comp(X, st, kind, lam=0.05):
    K = 3 * phones.NP + 1; d = X.shape[1]; m, s = X.mean(0), X.std(0) + 1e-9; Z = (X - m) / s
    mu = np.stack([Z[st == k].mean(0) if (st == k).any() else np.zeros(d) for k in range(K)])
    cnt = np.bincount(st, minlength=K).astype(float); lp = np.log((cnt + 1) / (cnt.sum() + K))
    if kind == "lda":
        R = Z - mu[st]; S = R.T @ R / len(R) + lam * np.eye(d); W = np.linalg.solve(S, mu.T).T
        return dict(kind=kind, m=m, s=s, W=W, b=-0.5 * np.einsum("kj,kj->k", W, mu) + lp, n=W.size + K)
    P, ld = [], []
    for k in range(K):
        Zk = Z[st == k]
        C = np.cov(Zk, rowvar=False) if len(Zk) > d else np.eye(d)
        C = 0.5 * C + 0.5 * np.diag(np.diag(C)) + lam * np.eye(d)
        P.append(np.linalg.inv(C)); ld.append(np.linalg.slogdet(C)[1])
    return dict(kind=kind, m=m, s=s, mu=mu, P=np.stack(P), ld=np.array(ld), lp=lp, n=K * (d + d * (d + 1) // 2) + K)

def score(c, X):
    Z = (X - c["m"]) / c["s"]
    if c["kind"] == "lda":
        S = Z @ c["W"].T + c["b"]
    else:
        S = np.empty((len(Z), len(c["mu"])))
        for k in range(len(c["mu"])):
            D = Z - c["mu"][k]; S[:, k] = -0.5 * np.einsum("ij,jk,ik->i", D, c["P"][k], D) - 0.5 * c["ld"][k] + c["lp"][k]
    return S - np.logaddexp.reduce(S, 1, keepdims=True)

def oracle(S, segs, flux, keep_seg=None):
    top1 = top3 = n = 0
    for j, (i0, i1, p) in enumerate(segs):
        if keep_seg is not None and not keep_seg[j]:
            continue
        sc = np.array([S[np.arange(i0, i1), 3 * k + phones.substates(i0, i1, ph, flux)].sum() for k, ph in enumerate(af.PHONES)])
        o = np.argsort(-sc); top1 += o[0] == p; top3 += p in o[:3]; n += 1
    return top1 / n, top3 / n

if __name__ == "__main__":
    model = np.load(f"{phones.OUT}/model.npy", allow_pickle=True).item()
    C = phones.load("cal"); fr = split_cal(C)
    L = phones.detector_logodds(C, model); Lc = ctx(L, C["lens"]); st = states(C); keep = C["ph"] >= 0
    inputs = {"LDA22": (L, "lda"), "QDA22": (L, "qda"), "LDA66": (Lc, "lda"), "QDA66d": (Lc, "qda"), "CTRL108": (C["X"].astype(np.float64), "lda")}
    segB = fr[C["segs"][:, 0]] == False
    R = {}
    for name, (X, kind) in inputs.items():
        c = fit_comp(X[keep & fr], st[keep & fr], kind)
        t1, t3 = oracle(score(c, X), C["segs"], C["flux"], segB)
        R[name] = dict(val_top1=round(t1, 4), val_top3=round(t3, 4), stored=int(c["n"])); print(name, R[name], flush=True)
    best = max((k for k in R if k != "CTRL108"), key=lambda k: R[k]["val_top1"])
    print("chosen on validation speakers:", best, flush=True)
    T = phones.load("test"); LT = phones.detector_logodds(T, model)
    tin = {"LDA22": LT, "QDA22": LT, "LDA66": ctx(LT, T["lens"]), "QDA66d": ctx(LT, T["lens"]), "CTRL108": T["X"].astype(np.float64)}
    for name in (best, "CTRL108"):
        X, kind = inputs[name]; c = fit_comp(X[keep], st[keep], kind)  # refit on all 10 cal speakers
        t1, t3 = oracle(score(c, tin[name]), T["segs"], T["flux"])
        R[name]["TEST_top1"] = round(t1, 4); R[name]["TEST_top3"] = round(t3, 4); print("TEST", name, R[name], flush=True)
    R["chosen"] = best; R["PASS_chosen"] = dict(top1=R[best]["TEST_top1"] >= 0.65, top3=R[best]["TEST_top3"] >= 0.86)
    json.dump(R, open(f"{phones.OUT}/eval_composer.json", "w"), indent=1); print(json.dumps(R, indent=1))
