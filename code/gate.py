"""Module 1: speech gate. Closed-form regularised Gaussian discriminant + Platt scaling + hysteresis/lookback.
Pass marks fixed BEFORE results (DESIGN.md v2 "Experiment 1"):
  A. test-clean (unseen speakers) speech-frame recall >= 98.5 % (after hysteresis + 150 ms lookback)
  B. ESC-50 folds 4-5 (unseen clips) open-frame rate < 1 % of all frames
  C. listening cue: 0 open frames
Primary model = G22 (8 tracks with +-80 ms Legendre context); G8 (instantaneous tracks) is the stated ablation.
Reported without a pass mark: recall in noise (SNR 20/10/5/0 dB), ESC per category and on active frames, his clips.
Usage: python gate.py build | fit | eval"""
import os
import csv, glob, json, os, re, sys
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
import frontend as fe

DATA = os.environ.get("DATA_DIR", "data")
CACHE = os.environ.get("DATA_DIR", "data") + "/cache"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/gate_out"
CUE = "data/raw/listening_cue.wav"
A = "data/audio"
HIS = sorted(glob.glob(f"{A}/hfd/*.wav") + glob.glob(f"{A}/dict_0604/dictations/*.wav") + [f"{A}/hotword_1999.wav"])
SIL = {"", "sil", "sp"}
TRACKS = ["Erel", "H", "R", "C1", "C2", "peaks", "flux"]


def gate_vec(tr, kind="G22"):
    if kind == "G8":
        return np.stack([tr[k] for k in TRACKS + ["mod"]], 1)
    return np.concatenate([fe.legendre(tr[k]) for k in TRACKS] + [tr["mod"][:, None]], 1)


def phone_labels(tg, T):
    """1 speech, 0 silence, -1 ignore (spoken noise / beyond alignment)."""
    s = open(tg).read()
    tier = s[s.index('name = "phones"'):]
    lab = -np.ones(T, np.int8)
    for a, b, txt in re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', tier):
        i0, i1 = int(round(float(a) * 100)), int(round(float(b) * 100))
        txt = txt.lower()
        lab[i0:i1] = -1 if txt == "spn" else (0 if txt in SIL else 1)
    return lab


def esc_meta():
    with open(f"{DATA}/ESC-50-master/meta/esc50.csv") as f:
        return list(csv.DictReader(f))


def mix(x, lab, noise, snr, rng):
    n = np.resize(noise, x.size)
    sp = np.repeat(lab == 1, fe.HOP)[: x.size]
    ps = np.mean(x[: sp.size][sp] ** 2) if sp.any() else np.mean(x ** 2)
    pn = np.mean(n ** 2) + 1e-12
    return x + n * np.sqrt(ps / (pn * 10 ** (snr / 10)))


def audio(args):
    """Rebuild the exact test/fit audio (same seeds) -> (x, labels per 10 ms frame)."""
    kind, path, extra, seed = args
    rng = np.random.default_rng(seed)
    if kind in ("libri", "libri_mix"):
        x = fe.load(path)
        tg = path.replace(f"{DATA}/LibriSpeech", f"{DATA}/align").replace(".flac", ".TextGrid")
        T = 1 + (x.size - fe.WIN) // fe.HOP
        lab = phone_labels(tg, T) if os.path.exists(tg) else -np.ones(T, np.int8)
        if kind == "libri_mix":
            noise_path, snr = extra
            x = mix(x, lab, fe.load(noise_path), snr, rng)
        x = fe.dither(x, rng)
    else:  # plain negative / his clip
        x = fe.dither(fe.load(path), rng)
        T = 1 + (x.size - fe.WIN) // fe.HOP
        lab = np.zeros(T, np.int8)
    return x, lab


def job(args):
    x, lab = audio(args)
    tr = fe.tracks(x)
    return dict(G22=gate_vec(tr).astype(np.float32), G8=gate_vec(tr, "G8").astype(np.float32), lab=lab,
                E=tr["E"].astype(np.float32))


def run_set(name, jobs, procs=4):
    os.makedirs(CACHE, exist_ok=True)
    p = f"{CACHE}/gate_{name}.npz"
    if os.path.exists(p):
        return
    with Pool(procs) as pool:
        res = pool.map(job, jobs, chunksize=8)
    np.savez(p, G22=np.concatenate([r["G22"] for r in res]), G8=np.concatenate([r["G8"] for r in res]),
             lab=np.concatenate([r["lab"] for r in res]), E=np.concatenate([r["E"] for r in res]),
             lens=np.array([len(r["lab"]) for r in res]), names=np.array([j[1] for j in jobs]))
    print("built", name, sum(len(r["lab"]) for r in res), "frames", flush=True)


def cue_variants():
    """Cue as the mic would get it: several gains, alone and over his room-like noise, in 2 s of dither."""
    os.makedirs(CACHE + "/cue", exist_ok=True)
    rng = np.random.default_rng(5); c = fe.load(CUE); paths = []
    for i, g in enumerate([1.0, 0.3, 0.1, 0.03]):
        y = np.zeros(fe.SR * 2, np.float32); y[8000:8000 + c.size] += g * c
        p = f"{CACHE}/cue/cue_{i}.wav"; fe.sf.write(p, y, fe.SR); paths.append(p)
    return paths


def job_lists():
    """All sets in build order (the rng draws must stay in this order: Silero control rebuilds the same audio)."""
    rng = np.random.default_rng(0)
    meta = esc_meta()
    esc = {f: [f"{DATA}/ESC-50-master/audio/{m['filename']}" for m in meta if int(m["fold"]) in f] for f in [(1, 2), (3,), (4, 5)]}
    dev = sorted(glob.glob(f"{DATA}/LibriSpeech/dev-clean/*/*/*.flac"))
    test = sorted(glob.glob(f"{DATA}/LibriSpeech/test-clean/*/*/*.flac"))
    spk = sorted({p.split("/")[-3] for p in dev})
    fit_spk = set(spk[:30])  # 30 speakers fit the Gaussians, 10 calibrate Platt
    dev_fit = [p for p in dev if p.split("/")[-3] in fit_spk]
    dev_cal = [p for p in dev if p.split("/")[-3] not in fit_spk]
    mixj = lambda files, noises, snrs, s0: [("libri_mix", p, (noises[rng.integers(len(noises))], snrs(i)), s0 + i) for i, p in enumerate(files)]
    U = lambda i: float(rng.uniform(0, 20))
    L = {}
    L["fit_libri"] = [("libri", p, None, i) for i, p in enumerate(dev_fit)]
    L["fit_mix"] = mixj(dev_fit, esc[(1, 2)], U, 10000)
    L["fit_esc"] = [("neg", p, None, 20000 + i) for i, p in enumerate(esc[(1, 2)])]
    L["cal_libri"] = [("libri", p, None, 30000 + i) for i, p in enumerate(dev_cal)]
    L["cal_mix"] = mixj(dev_cal, esc[(3,)], U, 40000)
    L["cal_esc"] = [("neg", p, None, 50000 + i) for i, p in enumerate(esc[(3,)])]
    L["test_libri"] = [("libri", p, None, 60000 + i) for i, p in enumerate(test)]
    for snr in (20, 10, 5, 0):
        L[f"test_mix{snr}"] = mixj(test, esc[(4, 5)], lambda i, snr=snr: snr, 70000 + snr * 1000)
    L["test_esc"] = [("neg", p, None, 90000 + i) for i, p in enumerate(esc[(4, 5)])]
    L["test_cue"] = [("neg", p, None, 95000 + i) for i, p in enumerate(cue_variants())]
    L["his"] = [("neg", p, None, 96000 + i) for i, p in enumerate(HIS)]
    return L


def build():
    for name, jobs in job_lists().items():
        run_set(name, jobs, procs=1 if name == "test_cue" else 2 if name == "his" else 4)


def load_set(name, kind):
    d = np.load(f"{CACHE}/gate_{name}.npz")
    return d[kind].astype(np.float64), d["lab"], d["lens"], d["names"], d["E"]


def gauss_fit(X, lam=0.1):
    mu = X.mean(0); S = np.cov(X, rowvar=False)
    S = (1 - lam) * S + lam * np.diag(np.diag(S))  # shrink toward diagonal: stable with correlated tracks
    Pm = np.linalg.inv(S); _, ld = np.linalg.slogdet(S)
    return mu, Pm, ld


def loglik(X, g):
    mu, Pm, ld = g; D = X - mu
    return -0.5 * np.einsum("ij,jk,ik->i", D, Pm, D) - 0.5 * ld


def fit(kind):
    pos, neg = [], []
    for n in ("fit_libri", "fit_mix"):
        X, lab, *_ = load_set(n, kind); pos.append(X[lab == 1]); neg.append(X[lab == 0])
    X, *_ = load_set("fit_esc", kind); neg.append(X)
    pos, neg = np.concatenate(pos), np.concatenate(neg)
    allx = np.concatenate([pos, neg]); m, s = allx.mean(0), allx.std(0) + 1e-9
    g1, g0 = gauss_fit((pos - m) / s), gauss_fit((neg - m) / s)
    # Platt on held-out speakers + ESC fold 3 (convex in a, b)
    cx, cy = [], []
    for n in ("cal_libri", "cal_mix"):
        X, lab, *_ = load_set(n, kind); k = lab >= 0; cx.append(X[k]); cy.append(lab[k])
    X, *_ = load_set("cal_esc", kind); cx.append(X); cy.append(np.zeros(len(X), np.int8))
    cx, cy = np.concatenate(cx), np.concatenate(cy).astype(np.float64)
    sc = loglik((cx - m) / s, g1) - loglik((cx - m) / s, g0)
    sc = np.clip(sc, -50, 50)
    nll = lambda ab: np.mean(np.logaddexp(0, -(2 * cy - 1) * (ab[0] * sc + ab[1])))
    ab = minimize(nll, [0.2, 0.0], method="Nelder-Mead").x
    model = dict(kind=kind, m=m, s=s, g1=g1, g0=g0, ab=ab, n_pos=len(pos), n_neg=len(neg))
    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/model_{kind}.npy", model, allow_pickle=True)
    n_w = 2 * len(m) + 2 * (len(m) + len(m) * (len(m) + 1) // 2) + 2 + 4
    print(kind, "fit: pos", len(pos), "neg", len(neg), "platt", ab, "stored numbers", n_w, flush=True)
    return model


def prob(model, X):
    Z = (X - model["m"]) / model["s"]
    sc = np.clip(loglik(Z, model["g1"]) - loglik(Z, model["g0"]), -50, 50)
    a, b = model["ab"]
    return 1 / (1 + np.exp(-(a * sc + b)))


def hysteresis(p, on=0.6, off=0.4, n_on=3, n_off=30, pre=15):
    """Open after n_on frames > on (then also open the 150 ms before: lookback buffer); close after n_off frames < off."""
    out = np.zeros(p.size, bool); state = False; hi = lo = 0
    for t, v in enumerate(p):
        if not state:
            hi = hi + 1 if v > on else 0
            if hi >= n_on:
                state = True; lo = 0; out[max(0, t - n_on - pre + 1):t + 1] = True
        else:
            out[t] = True
            lo = lo + 1 if v < off else 0
            if lo >= n_off:
                state = False; hi = 0
    return out


def per_utt(p, lens):
    o = np.zeros(p.size, bool); i = 0
    for L in lens:
        o[i:i + L] = hysteresis(p[i:i + L]); i += L
    return o


def evaluate(kind):
    model = np.load(f"{OUT}/model_{kind}.npy", allow_pickle=True).item()
    R = {"kind": kind}
    X, lab, lens, *_ = load_set("test_libri", kind)
    p = prob(model, X); g = per_utt(p, lens)
    R["A_recall_clean"] = float(g[lab == 1].mean()); R["raw_recall_clean_p>0.5"] = float((p[lab == 1] > .5).mean())
    R["open_on_libri_silence"] = float(g[lab == 0].mean())
    for snr in (20, 10, 5, 0):
        X, lab, lens, *_ = load_set(f"test_mix{snr}", kind)
        g = per_utt(prob(model, X), lens); R[f"recall_snr{snr}"] = float(g[lab == 1].mean())
    X, lab, lens, names, E = load_set("test_esc", kind)
    g = per_utt(prob(model, X), lens)
    R["B_esc_open_all_frames"] = float(g.mean())
    # active = within 30 dB of the clip's loudest frame (excludes the silent padding many ESC clips have)
    act = np.zeros(g.size, bool); i = 0; cat = {}
    meta = {m["filename"]: m["category"] for m in esc_meta()}
    for L, nm in zip(lens, names):
        e = E[i:i + L]; act[i:i + L] = e > e.max() - 30
        c = meta[os.path.basename(str(nm))]; cat.setdefault(c, []).append(g[i:i + L].mean()); i += L
    R["esc_open_active_frames"] = float(g[act].mean())
    R["esc_clips_with_any_open"] = float(np.mean([g[a:a + L].any() for a, L in zip(np.cumsum(lens) - lens, lens)]))
    R["esc_by_category_open"] = dict(sorted(((c, round(float(np.mean(v)), 4)) for c, v in cat.items()), key=lambda kv: -kv[1]))
    X, lab, lens, *_ = load_set("test_cue", kind)
    g = per_utt(prob(model, X), lens); R["C_cue_open_frames"] = int(g.sum())
    X, lab, lens, names, E = load_set("his", kind)
    p = prob(model, X); g = per_utt(p, lens); i = 0; his = {}
    for L, nm in zip(lens, names):
        gg = g[i:i + L]; first = int(np.argmax(gg)) if gg.any() else -1
        his[os.path.basename(str(nm))] = dict(open_frac=round(float(gg.mean()), 3), first_open_s=first / 100, dur_s=L / 100)
        i += L
    R["his_clips"] = his
    R["PASS"] = dict(A=R["A_recall_clean"] >= 0.985, B=R["B_esc_open_all_frames"] < 0.01, C=R["C_cue_open_frames"] == 0)
    json.dump(R, open(f"{OUT}/eval_{kind}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k not in ("esc_by_category_open", "his_clips")}, indent=1))
    print("worst ESC categories:", list(R["esc_by_category_open"].items())[:8])
    return R


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "fit":
        for k in ("G22", "G8"): fit(k)
    elif cmd == "eval":
        for k in ("G22", "G8"): evaluate(k)
