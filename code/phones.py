"""Module 2: 22 articulatory-feature detectors (closed-form shrunk Gaussians on masked inputs + Platt) and the
phone-state composer (LDA on the 22 clipped log-odds: w = S^-1 mu, the decorrelated GLS of DESIGN v2).
Data: LibriSpeech dev-clean (30 speakers fit detectors, 10 speakers calibrate + fit composer), test-clean (40 unseen speakers) test.
Pass marks fixed BEFORE results (DESIGN v2, reviewer's estimates):
  2. detector AUROC mean >= 0.85, none < 0.75 (test-clean)
  3. oracle-segmented phone accuracy top-1 >= 65 %, top-3 >= 86 % (test-clean)
Usage: python phones.py build | fit | eval"""
import os
import glob, json, os, re, sys
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
import frontend as fe
import af
from gate import DATA, phone_labels  # noqa: F401 (same data root)

CACHE = os.environ.get("DATA_DIR", "data") + "/cache"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/phones_out"
NP = len(af.PHONES); SILID = NP
PID = {p: i for i, p in enumerate(af.PHONES)}
COLS = {t: slice(4 * i, 4 * i + 4) for i, t in enumerate(af.ALL_TRACKS)}


def segments(tg):
    s = open(tg).read(); tier = s[s.index('name = "phones"'):]
    out = []
    for a, b, txt in re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', tier):
        ph = re.sub(r"\d", "", txt.upper())
        out.append((int(round(float(a) * 100)), int(round(float(b) * 100)), ph))
    return out


def substates(i0, i1, ph, flux):
    """0/1/2 = onset/nucleus/offset. Continuants 25/50/25; stops+affricates split at the burst (closure | burst | VOT)."""
    n = i1 - i0; s = np.ones(n, np.int8)
    if n <= 0:
        return s
    if ph in af.STOPS and n >= 3:
        b = 1 + int(np.argmax(flux[i0 + 1:i1])) if n > 1 else 0
        s[:b] = 0; s[b:b + 2] = 1; s[b + 2:] = 2
    else:
        a = max(1, int(round(0.25 * n))) if n >= 3 else 0
        s[:a] = 0; s[n - a:] = 2 if n >= 3 else 1
    return s


def job(path):
    x = fe.dither(fe.load(path), np.random.default_rng(abs(hash(path)) % 2 ** 32))
    tr = fe.tracks2(x); T = tr["E"].size
    X = np.concatenate([fe.legendre(tr[t], fe.LEG4) for t in af.ALL_TRACKS], 1).astype(np.float32)
    ph = -np.ones(T, np.int16); sub = np.zeros(T, np.int8); half = np.zeros(T, np.int8); seg = -np.ones(T, np.int32)
    tg = path.replace(f"{DATA}/LibriSpeech", f"{DATA}/align").replace(".flac", ".TextGrid")
    segs = []
    if os.path.exists(tg):
        for k, (i0, i1, p) in enumerate(segments(tg)):
            i1 = min(i1, T)
            if i1 <= i0:
                continue
            if p in ("", "SIL", "SP"):
                ph[i0:i1] = SILID
            elif p in PID:
                ph[i0:i1] = PID[p]; sub[i0:i1] = substates(i0, i1, p, tr["flux"])
                half[i0 + (i1 - i0) // 2:i1] = 1; seg[i0:i1] = len(segs); segs.append((i0, i1, PID[p]))
    return X, ph, sub, half, seg, np.array(segs, np.int32).reshape(-1, 3), tr["flux"].astype(np.float32)


def split():
    dev = sorted(glob.glob(f"{DATA}/LibriSpeech/dev-clean/*/*/*.flac"))
    spk = sorted({p.split("/")[-3] for p in dev}); fit_spk = set(spk[:30])
    return dict(fit=[p for p in dev if p.split("/")[-3] in fit_spk], cal=[p for p in dev if p.split("/")[-3] not in fit_spk],
                test=sorted(glob.glob(f"{DATA}/LibriSpeech/test-clean/*/*/*.flac")))


def build():
    for name, files in split().items():
        p = f"{CACHE}/ph_{name}.npz"
        if os.path.exists(p):
            continue
        with Pool(4) as pool:
            res = pool.map(job, files, chunksize=8)
        off = np.cumsum([0] + [len(r[1]) for r in res])[:-1]
        segs = np.concatenate([r[5] + np.array([o, o, 0]) for r, o in zip(res, off)])
        np.savez(p, X=np.concatenate([r[0] for r in res]), ph=np.concatenate([r[1] for r in res]),
                 sub=np.concatenate([r[2] for r in res]), half=np.concatenate([r[3] for r in res]),
                 flux=np.concatenate([r[6] for r in res]), segs=segs, lens=np.array([len(r[1]) for r in res]))
        print("built", name, sum(len(r[1]) for r in res), "frames", len(segs), "phone segments", flush=True)


def af_labels(ph, half):
    V0 = np.stack([af.vec(p, 0) for p in af.PHONES] + [np.zeros(len(af.FEATS), np.int8)])
    V1 = np.stack([af.vec(p, 1) for p in af.PHONES] + [np.zeros(len(af.FEATS), np.int8)])
    return np.where(half[:, None] == 1, V1[ph], V0[ph])


def gauss(X, lam):
    mu = X.mean(0); C = np.cov(X, rowvar=False); C = (1 - lam) * C + lam * np.diag(np.diag(C))
    _, ld = np.linalg.slogdet(C); return mu, np.linalg.inv(C), ld


def ll(X, g):
    mu, Pm, ld = g; D = X - mu; return -0.5 * np.einsum("ij,jk,ik->i", D, Pm, D) - 0.5 * ld


def load(name):
    d = np.load(f"{CACHE}/ph_{name}.npz"); return {k: d[k] for k in d.files}


def detector_logodds(D, model):
    out = np.zeros((len(D["ph"]), len(af.FEATS)))
    for i, f in enumerate(af.FEATS):
        m = model["det"][f]; Z = (D["X"][:, m["cols"]].astype(np.float64) - m["m"]) / m["s"]
        sc = np.clip(ll(Z, m["g1"]) - ll(Z, m["g0"]), -50, 50)
        out[:, i] = 6.0 * np.tanh((m["ab"][0] * sc + m["ab"][1]) / 6.0)  # clipped log-odds: no single detector can veto
    return out


def fit(lam=0.1):
    F, Cc = load("fit"), load("cal")
    det = {}; n_w = 0
    for name, D in (("fit", F), ("cal", Cc)):
        keep = D["ph"] >= 0
        for k in list(D):
            if k not in ("segs", "lens") and len(D[k]) == len(keep):
                D[k] = D[k][keep]
        D["AF"] = af_labels(D["ph"], D["half"])
    for i, f in enumerate(af.FEATS):
        cols = np.concatenate([np.arange(512)[COLS[t]] for t in af.MASK[f]])
        X = F["X"][:, cols].astype(np.float64); y = F["AF"][:, i]
        m, s = X.mean(0), X.std(0) + 1e-9; Z = (X - m) / s
        g1, g0 = gauss(Z[y == 1], lam), gauss(Z[y == 0], lam)
        Zc = (Cc["X"][:, cols].astype(np.float64) - m) / s; yc = Cc["AF"][:, i].astype(np.float64)
        sc = np.clip(ll(Zc, g1) - ll(Zc, g0), -50, 50)
        ab = minimize(lambda ab: np.mean(np.logaddexp(0, -(2 * yc - 1) * (ab[0] * sc + ab[1]))), [0.2, 0.0], method="Nelder-Mead").x
        det[f] = dict(cols=cols, m=m, s=s, g1=g1, g0=g0, ab=ab)
        d = len(cols); n_w += 2 * d + 2 * (d + d * (d + 1) // 2) + 2
        print(f"detector {f:13s} inputs {d:3d} pos {int(y.sum()):7d} platt {ab.round(3)}", flush=True)
    model = dict(det=det)
    # composer: 39 phones x 3 substates + silence, LDA (pooled covariance) on the 22 log-odds, fitted on the CAL speakers
    L = detector_logodds(Cc, model)
    st = np.where(Cc["ph"] == SILID, 3 * NP, Cc["ph"].astype(np.int64) * 3 + Cc["sub"])
    K = 3 * NP + 1; mu = np.zeros((K, L.shape[1])); cnt = np.zeros(K)
    for k in range(K):
        sel = st == k; cnt[k] = sel.sum()
        mu[k] = L[sel].mean(0) if cnt[k] else 0
    R = L - mu[st]; Sg = R.T @ R / len(R) + 0.05 * np.eye(L.shape[1])
    W = np.linalg.solve(Sg, mu.T).T  # decorrelated weights per state
    b = -0.5 * np.einsum("kj,kj->k", W, mu) + np.log((cnt + 1) / (cnt.sum() + K))
    model["comp"] = dict(W=W, b=b)
    n_w += W.size + b.size
    os.makedirs(OUT, exist_ok=True); np.save(f"{OUT}/model.npy", model, allow_pickle=True)
    print("stored numbers", n_w, flush=True)


def auroc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate():
    model = np.load(f"{OUT}/model.npy", allow_pickle=True).item()
    D = load("test"); keep = D["ph"] >= 0
    L = detector_logodds(D, model); AF = af_labels(np.where(keep, D["ph"], SILID), D["half"])
    R = {"auroc": {f: round(auroc(L[keep, i], AF[keep, i]), 4) for i, f in enumerate(af.FEATS)}}
    R["auroc_mean"] = float(np.mean(list(R["auroc"].values()))); R["auroc_min"] = float(min(R["auroc"].values()))
    S = L @ model["comp"]["W"].T + model["comp"]["b"]  # (T, 118) state log-scores
    S = S - np.logaddexp.reduce(S, 1, keepdims=True)
    st = np.where(D["ph"] == SILID, 3 * NP, D["ph"].astype(np.int64) * 3 + D["sub"])
    R["frame_state_acc"] = float((S.argmax(1) == st)[keep].mean())
    fph = np.where(S.argmax(1) == 3 * NP, SILID, S.argmax(1) // 3)
    R["frame_phone_acc"] = float((fph == D["ph"])[keep].mean())
    top1 = top3 = 0; conf = np.zeros((NP, NP), np.int32)
    for i0, i1, p in D["segs"]:
        sc = np.zeros(NP)
        for k, ph in enumerate(af.PHONES):  # score every hypothesis with ITS own substate split
            sub = substates(i0, i1, ph, D["flux"])
            sc[k] = S[np.arange(i0, i1), 3 * k + sub].sum()
        o = np.argsort(-sc); top1 += o[0] == p; top3 += p in o[:3]; conf[p, o[0]] += 1
    n = len(D["segs"]); R["oracle_top1"] = top1 / n; R["oracle_top3"] = top3 / n; R["n_segments"] = int(n)
    pairs = [(af.PHONES[i], af.PHONES[j], int(conf[i, j])) for i in range(NP) for j in range(NP) if i != j]
    R["top_confusions"] = sorted(pairs, key=lambda t: -t[2])[:15]
    R["per_phone_acc"] = {af.PHONES[i]: round(float(conf[i, i] / max(1, conf[i].sum())), 3) for i in range(NP)}
    R["PASS"] = dict(auroc=R["auroc_mean"] >= 0.85 and R["auroc_min"] >= 0.75, top1=R["oracle_top1"] >= 0.65, top3=R["oracle_top3"] >= 0.86)
    json.dump(R, open(f"{OUT}/eval.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in R.items() if k != "per_phone_acc"}, indent=1, default=float))


if __name__ == "__main__":
    {"build": build, "fit": fit, "eval": evaluate}[sys.argv[1]]()
