"""D1-P programmed units (D1_vs_WSP_2026-11-12.md): weights written from acoustic-phonetic reasoning, NO fitting. Exploratory (not a test).
Each unit is scored on fit utterances against frame truth (ph_fit 'ph', 39 = silence) and, for reference only, against a closed-form
logistic fit on the same inputs (the fitted number is a probe, never the unit). Written 12 Nov 2026 (the agent).
Frame grid: log-mel / band frame t+1 ~ phones12 frame t (exp157 docstring); bands are shifted by one frame and cut to the truth length."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb142 as b, d1_views as v


def roc_auc_score(y, p):  # rank form (Mann-Whitney), ties averaged
    r = np.argsort(np.argsort(p, kind="mergesort")).astype(np.float64) + 1
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True); s = np.bincount(inv, r); r = (s / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def log_loss(y, p): return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logistic_1d(x, y, iters=50):  # Newton on (a, c) for p = sigmoid(a x + c): the PROBE only
    a, c = 0.0, 0.0
    for _ in range(iters):
        z = a * x + c; p = 1 / (1 + np.exp(-z)); w = p * (1 - p) + 1e-9; g = np.array([((p - y) * x).sum(), (p - y).sum()])
        H = np.array([[(w * x * x).sum(), (w * x).sum()], [(w * x).sum(), w.sum()]]); a, c = np.array([a, c]) - np.linalg.solve(H, g)
    return a, c

SIL = 39
LN10 = np.log(10.0)


def utts(n=150, seed=157):
    files, _, lens = b.labelled("fit"); d = np.load(f"{b.C}/ph_fit.npz"); ph = d["ph"]; st = np.r_[0, np.cumsum(lens)]
    idx = np.random.default_rng(seed).choice(len(files), n, replace=False)
    for i in idx:
        E = v.bands(files[i], raw=True)[1:]; y = ph[st[i]:st[i + 1]]; T = min(len(E), len(y))
        yield E[:T], y[:T]


def p1_programmed(E):
    """speech vs silence. Total log energy over 0-8 kHz relative to the utterance noise floor (10th percentile), in dB.
    Speech energy sits 15-40 dB above a recording's noise floor; silence within ~5 dB of it. Programmed: logistic centred at 10 dB above
    the floor, 3 dB per e-fold (score 0.27 at 7 dB, 0.73 at 13 dB)."""
    tot = np.log(np.exp(E).sum(1) + 1e-9); db = 10 / LN10 * (tot - np.percentile(tot, 10))
    return 1 / (1 + np.exp(-(db - 10.0) / 3.0)), db


def score(name, p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return f"{name}: AUROC {roc_auc_score(y, p):.4f}  log-loss {log_loss(y, p):.4f}  acc@0.5 {((p > .5) == y).mean():.4f}"


if __name__ == "__main__":
    P, DB, Y = [], [], []
    for E, y in utts():
        p, db = p1_programmed(E); P.append(p); DB.append(db); Y.append((y != SIL).astype(int))
    P, DB, Y = map(np.concatenate, (P, DB, Y)); print("frames", len(Y), "speech share", round(Y.mean(), 4))
    print(score("P1 programmed (0 fitted numbers)", P, Y))
    half = len(Y) // 2; a, c = fit_logistic_1d(DB[:half], Y[:half])
    print(score("P1 probe: logistic fitted on the same dB input (fit half A, scored half B)", 1 / (1 + np.exp(-(a * DB[half:] + c))), Y[half:]),
          "| fitted centre dB", round(float(-c / a), 2), "e-fold dB", round(float(1 / a), 2))
    print(score("P1 programmed, same half B", P[half:], Y[half:]))


def p1_v2(E):
    """P1 v2 (12 Nov, after the v1 mismatch): v1's error was in my premise "speech sits 15-40 dB above the floor": stop closures (P K T G B)
    and weak fricatives (F TH HH) are speech but sit 11-19 dB median and 17-35 % of their frames < 6 dB, while silence has median 0.2,
    90th pct 7.1 dB. Mechanism: energy alone cannot see a closure; DURATION can: a closure or weak fricative is a short dip (< ~120 ms)
    between speech, a pause is long. Programmed: energy logistic centred at 7 dB (where the two distributions cross), 3 dB e-fold;
    then a dip of < 12 frames (120 ms) whose both neighbours are speech-like (score > 0.5) is given the smaller of its neighbours' scores."""
    p, db = p1_programmed(E); p = 1 / (1 + np.exp(-(db - 7.0) / 3.0)); hi = p > 0.5; T = len(p); q = p.copy(); t = 0
    while t < T:
        if hi[t]: t += 1; continue
        s = t
        while t < T and not hi[t]: t += 1
        if s > 0 and t < T and t - s < 12: q[s:t] = np.maximum(q[s:t], min(p[s - 1], p[t]))
    return q, db


if __name__ == "__main__" and "v2" in sys.argv:
    for name, fn in (("v1", p1_programmed), ("v2", p1_v2)):
        P, Y = [], []
        for E, y in utts(): P.append(fn(E)[0]); Y.append((y != SIL).astype(int))
        print(score(f"P1 {name} programmed", np.concatenate(P), np.concatenate(Y)))


def fit_logistic(X, y, iters=30, l2=1e-3):  # IRLS, multi-d: the PROBE only
    X1 = np.c_[X, np.ones(len(X))]; w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ w, -30, 30))); g = X1.T @ (p - y) + l2 * w; H = (X1 * (p * (1 - p))[:, None]).T @ X1 + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return lambda Z: 1 / (1 + np.exp(-np.clip(np.c_[Z, np.ones(len(Z))] @ w, -30, 30)))


if __name__ == "__main__" and "fair" in sys.argv:
    # equal information: the fitted reader sees the same dB track AND its +-20/60/200 ms time course (10 numbers), as v2 can use duration
    P2, F, Y = [], [], []
    for E, y in utts():
        q, db = p1_v2(E); P2.append(q); F.append(v.time_course(db[:, None].astype(np.float64))[:, 0]); Y.append((y != SIL).astype(int))
    P2, F, Y = np.concatenate(P2), np.concatenate(F), np.concatenate(Y); half = len(Y) // 2
    f = fit_logistic(F[:half], Y[:half]); print(score("probe: logistic on dB + time course (10 inputs), A->B", f(F[half:]), Y[half:]))
    print(score("P1 v2 programmed, half B", P2[half:], Y[half:]))


def p1_v3(E):
    """P1 v3: the equal-information probe showed the decisive input is the MEAN energy over +-60 ms (alone AUROC 0.9778 > v2 0.967).
    Mechanism: a stop closure or weak fricative lasts ~50-100 ms, so a +-60 ms mean lifts it with its neighbours, while a pause stays low.
    Programmed: logistic on the +-60 ms Legendre P0 of the dB track, centre 7 dB (distribution crossing, measured), e-fold 3 dB."""
    _, db = p1_programmed(E); m60 = v.time_course(db[:, None].astype(np.float64))[:, 0, 4]
    return 1 / (1 + np.exp(-(m60 - 7.0) / 3.0)), db


if __name__ == "__main__" and "v3" in sys.argv:
    P3, Y = [], []
    for E, y in utts(): P3.append(p1_v3(E)[0]); Y.append((y != SIL).astype(int))
    P3, Y = np.concatenate(P3), np.concatenate(Y); half = len(Y) // 2
    print(score("P1 v3 programmed, half B", P3[half:], Y[half:])); print(score("P1 v3 programmed, all", P3, Y))


# ---------------- P2 periodicity -> voiced (12 Nov evening) ----------------
import af, phones, phones12
VOICED = np.array([p in af.MEMBERS["voiced"] for p in af.PHONES] + [False])  # index 39 = silence (unvoiced)
TRK = phones12.NAMES


def utts2(n=150, seed=157):
    """as utts() plus the 31 cached tracks (trk31 rows are on the label grid; bands[1:] is too, see module docstring)."""
    files, _, lens = b.labelled("fit"); d = np.load(f"{b.C}/ph_fit.npz"); ph = d["ph"]; st = np.r_[0, np.cumsum(lens)]
    Tr = np.load(f"{phones.CACHE}/trk31_fit.npy", mmap_mode="r"); assert Tr.shape == (int(st[-1]), 31)
    idx = np.random.default_rng(seed).choice(len(files), n, replace=False)
    for i in idx:
        E = v.bands(files[i], raw=True)[1:]; y = ph[st[i]:st[i + 1]]; tr = np.asarray(Tr[st[i]:st[i + 1]], np.float64); T = min(len(E), len(y))
        yield E[:T], tr[:T], y[:T]


def p2_v1(tr):
    """P2 v1, premise written BEFORE looking at any P2 data: voiced speech is periodic (F0 70-400 Hz), so the normalised autocorrelation
    peak R is high (> ~0.6); voiceless fricatives / aspiration are noise (R < ~0.3); silence and closures have little energy and a noisy R.
    Programmed: logistic on R, centre 0.45 (the frontend's own F0 gate, set 5 Nov without this data), e-fold 0.08."""
    R = tr[:, TRK.index("R")]; return 1 / (1 + np.exp(-(R - 0.45) / 0.08))


def p2_inputs(E, tr):
    """the design's P2 inputs (D1_DESIGN 3.2: R, F0, Elow, BANDS 1-3), each as 10 time-course numbers -> (T, 60)."""
    zb = (E - E.mean(0)) / (E.std(0) + 1e-5); X = np.c_[tr[:, [TRK.index("R"), TRK.index("F0"), TRK.index("Elow")]], zb[:, :3]]
    return v.time_course(X.astype(np.float64)).reshape(len(X), -1)


P2_COLS = [f"{c}:{k}" for c in ("R", "F0", "Elow", "band1", "band2", "band3")
           for k in ("now", "P0/20", "P1/20", "P2/20", "P0/60", "P1/60", "P2/60", "P0/200", "P1/200", "P2/200")]


if __name__ == "__main__" and "p2" in sys.argv:
    seed = 4242 if "fresh" in sys.argv else 157
    P, X, Y, PH, R = [], [], [], [], []
    for E, tr, y in utts2(seed=seed):
        P.append(p2_v1(tr)); X.append(p2_inputs(E, tr)); Y.append(VOICED[y].astype(int)); PH.append(y); R.append(tr[:, TRK.index("R")])
    P, X, Y, PH, R = map(np.concatenate, (P, X, Y, PH, R)); sp = PH != SIL; print("seed", seed, "frames", len(Y), "speech", sp.sum(), "voiced share of speech", round(Y[sp].mean(), 4))
    for nm, m in (("voiced speech", sp & (Y == 1)), ("voiceless speech", sp & (Y == 0)), ("silence", ~sp)):
        print(f"R {nm}: p10/50/90", np.round(np.percentile(R[m], [10, 50, 90]), 3))
    med = sorted(((float(np.median(R[PH == k])), af.PHONES[k], int(VOICED[k])) for k in range(39) if (PH == k).sum() > 50))
    print("per-phone median R (low -> high):", " ".join(f"{p}{'+' if vv else '-'}{m:.2f}" for m, p, vv in med))
    Ps, Xs, Ys = P[sp], X[sp], Y[sp]; half = len(Ys) // 2
    print(score("P2 v1 programmed (speech frames, all)", Ps, Ys)); print(score("P2 v1 programmed (speech frames, half B)", Ps[half:], Ys[half:]))
    mu, sd = Xs[:half].mean(0), Xs[:half].std(0) + 1e-9; f = fit_logistic((Xs[:half] - mu) / sd, Ys[:half])
    print(score("probe: logistic on the 60 design inputs, speech frames A->B", f((Xs[half:] - mu) / sd), Ys[half:]))
    if seed == 157:
        au = sorted(((max(roc_auc_score(Ys, Xs[:, j]), 1 - roc_auc_score(Ys, Xs[:, j])), P2_COLS[j]) for j in range(Xs.shape[1])), reverse=True)
        print("single-input AUROC (max of both signs), top 12:", " | ".join(f"{c} {a:.4f}" for a, c in au[:12]))


def p2_cues(E, tr):
    """three named cues, each the +-20 / +-60 ms mean (P0) of a track: R (periodicity, P0/60: best single R input on seed 157),
    LOWdB = energy 250-750 Hz (bands 2-3) in dB above its utterance 10th percentile (voice bar / F1 energy, P0/20),
    ELOW = the frontend's 60-400 Hz energy share in dB (low vs total spectrum tilt, P0/20)."""
    lo = np.log(np.exp(E[:, 1:3]).sum(1) + 1e-9); lodb = 10 / LN10 * (lo - np.percentile(lo, 10))
    X = np.c_[tr[:, TRK.index("R")], lodb, tr[:, TRK.index("Elow")]].astype(np.float64); tc = v.time_course(X)
    return np.c_[tc[:, 0, 4], tc[:, 1, 1], tc[:, 2, 1]]


def gnb(Xa, ya):
    """Gaussian class-conditional per cue (measured means / spreads, 4 numbers per cue) + prior: closed form, no iterative fit.
    logit = log prior odds + sum over cues of the Gaussian log-likelihood ratio (cues treated as independent evidence)."""
    m1, s1, m0, s0 = Xa[ya == 1].mean(0), Xa[ya == 1].std(0), Xa[ya == 0].mean(0), Xa[ya == 0].std(0); lp = np.log(ya.mean() / (1 - ya.mean()))
    def f(Z):
        ll = -0.5 * ((Z - m1) / s1) ** 2 - np.log(s1) + 0.5 * ((Z - m0) / s0) ** 2 + np.log(s0); return 1 / (1 + np.exp(-np.clip(lp + ll.sum(1), -30, 30)))
    return f, dict(m1=m1.round(3).tolist(), s1=s1.round(3).tolist(), m0=m0.round(3).tolist(), s0=s0.round(3).tolist(), prior=round(float(ya.mean()), 4))


if __name__ == "__main__" and "p2b" in sys.argv:
    seed = 4242 if "fresh" in sys.argv else 157
    C, X, Y, PH = [], [], [], []
    for E, tr, y in utts2(seed=seed): C.append(p2_cues(E, tr)); X.append(p2_inputs(E, tr)); Y.append(VOICED[y].astype(int)); PH.append(y)
    C, X, Y, PH = map(np.concatenate, (C, X, Y, PH)); sp = PH != SIL; C, X, Y = C[sp], X[sp], Y[sp]; half = len(Y) // 2
    for j, nm in enumerate(("R P0/60", "LOWdB P0/20", "ELOW P0/20")):
        print(f"{nm}: voiced p10/50/90 {np.round(np.percentile(C[Y == 1, j], [10, 50, 90]), 2)}  voiceless {np.round(np.percentile(C[Y == 0, j], [10, 50, 90]), 2)}  AUROC {roc_auc_score(Y, C[:, j]):.4f}")
    f, prm = gnb(C[:half], Y[:half]); print("GNB numbers (measured on half A):", prm)
    print(score("P2 v2 GNB 3 cues (13 measured numbers), half B", f(C[half:]), Y[half:]))
    mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; g = fit_logistic((X[:half] - mu) / sd, Y[:half])
    print(score("probe: logistic on the 60 design inputs, A->B", g((X[half:] - mu) / sd), Y[half:]))
    h = fit_logistic(C[:half], Y[:half]); print(score("probe: logistic on the same 3 cues (4 fitted numbers), A->B", h(C[half:]), Y[half:]))


if __name__ == "__main__" and "p2c" in sys.argv:
    # where does the 60-input reader beat the 3-cue readers? per-phone accuracy on half B, and the gain from each input GROUP (leave-one-group-in)
    seed = 157; C, X, Y, PH = [], [], [], []
    for E, tr, y in utts2(seed=seed): C.append(p2_cues(E, tr)); X.append(p2_inputs(E, tr)); Y.append(VOICED[y].astype(int)); PH.append(y)
    C, X, Y, PH = map(np.concatenate, (C, X, Y, PH)); sp = PH != SIL; C, X, Y, PH = C[sp], X[sp], Y[sp], PH[sp]; half = len(Y) // 2
    mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd
    g = fit_logistic(Z[:half], Y[:half])(Z[half:]); h = fit_logistic(C[:half], Y[:half])(C[half:]); yb, pb = Y[half:], PH[half:]
    rows = []
    for k in range(39):
        m = pb == k
        if m.sum() > 100: rows.append((((h[m] > .5) == yb[m]).mean() - ((g[m] > .5) == yb[m]).mean(), af.PHONES[k], int(m.sum()), ((g[m] > .5) == yb[m]).mean()))
    rows.sort(); print("per phone (3-cue acc - 60-input acc, n, 60-input acc), worst 10:", " | ".join(f"{p} {d:+.3f} n{n} {a:.3f}" for d, p, n, a in rows[:10]))
    for grp in ("R", "F0", "Elow", "band1", "band2", "band3"):
        cols = [j for j, c in enumerate(P2_COLS) if c.startswith(grp + ":")]; ff = fit_logistic(Z[:half][:, cols], Y[:half])
        print(score(f"probe: {grp} only (10 inputs), A->B", ff(Z[half:][:, cols]), yb))
    for k in ("now", "/20", "/60", "/200"):
        cols = [j for j, c in enumerate(P2_COLS) if c.endswith(k) or (k == "now" and c.endswith(":now"))]; ff = fit_logistic(Z[:half][:, cols], Y[:half])
        print(score(f"probe: window {k} only ({len(cols)} inputs), A->B", ff(Z[half:][:, cols]), yb))


def p2_cues4(E, tr):
    """p2_cues + DIP: low-band (250-750 Hz) dB P0/20 minus its P0/200 (a frame below its own +-200 ms context = a consonant inside voiced
    context: closures and fricatives dip in the voice-bar / F1 band, vowels do not)."""
    c = p2_cues(E, tr); lo = np.log(np.exp(E[:, 1:3]).sum(1) + 1e-9); tc = v.time_course((10 / LN10 * lo)[:, None].astype(np.float64))[:, 0]
    return np.c_[c, tc[:, 1] - tc[:, 7]]


STOPS = np.array([p in set("P B T D K G CH JH".split()) for p in af.PHONES] + [False])

if __name__ == "__main__" and "p2d" in sys.argv:
    seed = 4242 if "fresh" in sys.argv else 157; C, X, Y, PH = [], [], [], []
    for E, tr, y in utts2(seed=seed): C.append(p2_cues4(E, tr)); X.append(p2_inputs(E, tr)); Y.append(VOICED[y].astype(int)); PH.append(y)
    C, X, Y, PH = map(np.concatenate, (C, X, Y, PH)); sp = PH != SIL; C, X, Y, PH = C[sp], X[sp], Y[sp], PH[sp]; half = len(Y) // 2
    mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd; g = fit_logistic(Z[:half], Y[:half])(Z[half:])
    h3 = fit_logistic(C[:half, :3], Y[:half])(C[half:, :3]); h4 = fit_logistic(C[:half], Y[:half])(C[half:]); yb, ns = Y[half:], ~STOPS[PH[half:]]
    print("seed", seed, "DIP AUROC", round(roc_auc_score(Y, C[:, 3]), 4), "stop share of speech frames", round(1 - ns.mean(), 4))
    for nm, p in (("60-input fitted", g), ("3-cue fitted", h3), ("4-cue fitted (+DIP)", h4)):
        print(score(f"{nm}, all speech B", p, yb), "||", score("non-stop B", p[ns], yb[ns]), "||", score("stops B", p[~ns], yb[~ns]))


def p2_prog(Ca, ya, rule):
    """programmed combiner: each cue -> its own equal-variance Gaussian LLR from MEASURED class means and pooled spread (3 numbers per cue,
    closed form), then a combination RULE written from the cue structure: 'sum' (independent evidence), 'mean' (fully redundant),
    'rootk' (sum / sqrt(k): partly redundant, the k cues read one vibration source through different windows)."""
    m1, m0 = Ca[ya == 1].mean(0), Ca[ya == 0].mean(0); s = np.sqrt(0.5 * (Ca[ya == 1].var(0) + Ca[ya == 0].var(0))); lp = np.log(ya.mean() / (1 - ya.mean()))
    k = Ca.shape[1]; div = dict(sum=1.0, mean=float(k), rootk=float(np.sqrt(k)))[rule]
    return lambda Z: 1 / (1 + np.exp(-np.clip(lp + ((m1 - m0) / s ** 2 * (Z - 0.5 * (m1 + m0))).sum(1) / div, -30, 30)))


if __name__ == "__main__" and "p2e" in sys.argv:
    seed = 4242 if "fresh" in sys.argv else 157; C, Y, PH = [], [], []
    for E, tr, y in utts2(seed=seed): C.append(p2_cues4(E, tr)); Y.append(VOICED[y].astype(int)); PH.append(y)
    C, Y, PH = map(np.concatenate, (C, Y, PH)); sp = PH != SIL; C, Y, PH = C[sp], Y[sp], PH[sp]; half = len(Y) // 2; ns = ~STOPS[PH[half:]]
    for rule in ("sum", "mean", "rootk"):
        p = p2_prog(C[:half], Y[:half], rule)(C[half:]); print(score(f"seed {seed} P2 programmed 4 cues, rule {rule}, B", p, Y[half:]), "|| non-stop acc", round(((p[ns] > .5) == Y[half:][ns]).mean(), 4))


# ---------------- P4 aperiodic noise -> frication / aspiration (12 Nov evening) ----------------
NOISY = np.array([p in set("F V TH DH S Z SH ZH HH CH JH".split()) for p in af.PHONES] + [False])  # noise through most of the phone (my truth choice)
PTK = np.array([p in set("P T K".split()) for p in af.PHONES] + [False])  # aspiration only at release: reported apart, excluded from scoring


def p4_v1(tr):
    """P4 v1, premise written BEFORE looking at P4 data: frication / aspiration is turbulent noise, its energy sits high (3.8-7.6 kHz share
    Ehigh near -5 .. -12 dB for S Z SH, lower for F TH HH) while vowels / nasals put almost all energy below 3 kHz (Ehigh ~ -25 dB or less).
    Programmed: logistic on Ehigh, centre -18 dB, e-fold 3 dB."""
    return 1 / (1 + np.exp(-(tr[:, TRK.index("Ehigh")] + 18.0) / 3.0))


def p4_inputs(E, tr):
    """the design's P4 inputs (D1_DESIGN 3.2: Ehigh, BH, WH, R, BANDS 9-12), each x 10 time-course numbers -> (T, 80)."""
    zb = (E - E.mean(0)) / (E.std(0) + 1e-5); X = np.c_[tr[:, [TRK.index(c) for c in ("Ehigh", "BH", "WH", "R")]], zb[:, 8:12]]
    return v.time_course(X.astype(np.float64)).reshape(len(X), -1)


P4_COLS = [f"{c}:{k}" for c in ("Ehigh", "BH", "WH", "R", "band9", "band10", "band11", "band12")
           for k in ("now", "P0/20", "P1/20", "P2/20", "P0/60", "P1/60", "P2/60", "P0/200", "P1/200", "P2/200")]

if __name__ == "__main__" and "p4" in sys.argv:
    seed = 4242 if "fresh" in sys.argv else 157; P, X, Y, PH, EH = [], [], [], [], []
    for E, tr, y in utts2(seed=seed): P.append(p4_v1(tr)); X.append(p4_inputs(E, tr)); Y.append(NOISY[y].astype(int)); PH.append(y); EH.append(tr[:, TRK.index("Ehigh")])
    P, X, Y, PH, EH = map(np.concatenate, (P, X, Y, PH, EH)); keep = (PH != SIL) & ~PTK[PH]
    print("seed", seed, "speech non-PTK frames", keep.sum(), "noisy share", round(Y[keep].mean(), 4))
    for nm, m in (("noisy", keep & (Y == 1)), ("other speech", keep & (Y == 0)), ("PTK", PTK[PH]), ("silence", PH == SIL)):
        print(f"Ehigh {nm}: p10/50/90", np.round(np.percentile(EH[m], [10, 50, 90]), 2))
    med = sorted(((float(np.median(EH[PH == k])), af.PHONES[k], int(NOISY[k])) for k in range(39) if (PH == k).sum() > 50))
    print("per-phone median Ehigh:", " ".join(f"{p}{'+' if q else '-'}{m:.1f}" for m, p, q in med))
    P, X, Y = P[keep], X[keep], Y[keep]; half = len(Y) // 2
    print(score("P4 v1 programmed, half B", P[half:], Y[half:]))
    mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd; g = fit_logistic(Z[:half], Y[:half])
    print(score("probe: logistic on the 80 design inputs, A->B", g(Z[half:]), Y[half:]))
    if seed == 157:
        au = sorted(((max(roc_auc_score(Y, X[:, j]), 1 - roc_auc_score(Y, X[:, j])), P4_COLS[j]) for j in range(X.shape[1])), reverse=True)
        print("single-input AUROC top 12:", " | ".join(f"{c} {a:.4f}" for a, c in au[:12]))
        for grp in ("Ehigh", "BH", "WH", "R", "band9", "band10", "band11", "band12"):
            cols = [j for j, c in enumerate(P4_COLS) if c.startswith(grp + ":")]; print(score(f"probe {grp} only", fit_logistic(Z[:half][:, cols], Y[:half])(Z[half:][:, cols]), Y[half:]))


def p4_cues(E, tr):
    """three views of turbulence, each its +-20 ms mean (P0/20: best window for Ehigh and R on seed 157): Ehigh share (noise sits high),
    R (noise is aperiodic: low R), BH (high-band centroid: where in 3.8-7.6 kHz the noise sits)."""
    X = tr[:, [TRK.index(c) for c in ("Ehigh", "R", "BH")]].astype(np.float64); return v.time_course(X)[:, :, 1]


if __name__ == "__main__" and "p4b" in sys.argv:
    seed = 4242 if "fresh" in sys.argv else 157; C, X, Y, PH = [], [], [], []
    for E, tr, y in utts2(seed=seed): C.append(p4_cues(E, tr)); X.append(p4_inputs(E, tr)); Y.append(NOISY[y].astype(int)); PH.append(y)
    C, X, Y, PH = map(np.concatenate, (C, X, Y, PH)); keep = (PH != SIL) & ~PTK[PH]; C, X, Y, PH = C[keep], X[keep], Y[keep], PH[keep]; half = len(Y) // 2
    mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd; g = fit_logistic(Z[:half], Y[:half])(Z[half:])
    h = fit_logistic(C[:half], Y[:half])(C[half:]); yb, pb = Y[half:], PH[half:]
    print(score(f"seed {seed} probe 80 inputs A->B", g, yb)); print(score("3 cues fitted (4 numbers)", h, yb))
    for k in (2, 3):
        for rule in ("sum", "rootk"):
            p = p2_prog(C[:half, :k], Y[:half], rule)(C[half:, :k]); print(score(f"P4 programmed {k} cues rule {rule}", p, yb))
    p = p2_prog(C[:half], Y[:half], "rootk")(C[half:]); rows = []
    for kk in range(39):
        m = pb == kk
        if m.sum() > 100: rows.append((((p[m] > .5) == yb[m]).mean(), af.PHONES[kk], int(NOISY[kk]), ((g[m] > .5) == yb[m]).mean()))
    rows.sort(); print("per phone acc programmed-3-rootk (fitted-80) worst 8:", " | ".join(f"{q}{'+' if n else '-'} {a:.3f} ({b2:.3f})" for a, q, n, b2 in rows[:8]))


# ---------------- P5 closure (12 Nov night) ----------------
# Truth (my choice, written before P5 data, independent of any cue): MFA marks a stop as ONE segment (closure + release). Positives = the first
# half of each P B T D K G segment (closure is typically 50-70 % of a stop); the second half (burst / aspiration) is excluded from scoring;
# negatives = every other speech frame. Silence excluded (P1's job).
CLOS = np.array([p in set("P B T D K G".split()) for p in af.PHONES] + [False])


def closure_truth(y):
    """-> (truth 0/1, keep mask) per frame."""
    yt = np.zeros(len(y), int); keep = y != SIL; t = 0
    while t < len(y):
        s = t
        while t < len(y) and y[t] == y[s]: t += 1
        if CLOS[y[s]]:
            h = s + max(1, (t - s) // 2); yt[s:h] = 1; keep[h:t] = False
    return yt, keep


def p5_tracks(E):
    """total dB above the utterance floor (as P1), and Erel = that dB minus its own +-200 ms mean (a closure is a hole in its context)."""
    _, db = p1_programmed(E); tc = v.time_course(db[:, None].astype(np.float64))[:, 0]
    return db, tc[:, 0] - tc[:, 7]


def p5_v1(E):
    """P5 v1, premise written BEFORE looking at P5 data: a closure is the vocal tract shut, so energy falls far below its +-200 ms context
    (voiceless closures to near the floor; voiced ones keep only a weak voice bar). Programmed: logistic on Erel P0/20 (the +-20 ms mean of
    dB minus the +-200 ms mean), centre -8 dB, e-fold 3 dB."""
    db, _ = p5_tracks(E); tc = v.time_course(db[:, None].astype(np.float64))[:, 0]; erel20 = tc[:, 1] - tc[:, 7]
    return 1 / (1 + np.exp((erel20 + 8.0) / 3.0)), erel20


def p5_inputs(E):
    """equal-information reference inputs: total dB, Erel, P1 v3 score, each x 10 time-course numbers -> (T, 30)."""
    db, er = p5_tracks(E); p1 = p1_v3(E)[0]; return v.time_course(np.c_[db, er, p1].astype(np.float64)).reshape(len(db), -1)


P5_COLS = [f"{c}:{k}" for c in ("dB", "Erel", "P1") for k in ("now", "P0/20", "P1/20", "P2/20", "P0/60", "P1/60", "P2/60", "P0/200", "P1/200", "P2/200")]

if __name__ == "__main__" and "p5" in sys.argv:
    seed = 4242 if "fresh" in sys.argv else 157; P, ER, X, Y, K, PH = [], [], [], [], [], []
    for E, y in utts(seed=seed):
        p, er = p5_v1(E); yt, kp = closure_truth(y); P.append(p); ER.append(er); X.append(p5_inputs(E)); Y.append(yt); K.append(kp); PH.append(y)
    P, ER, X, Y, K, PH = map(np.concatenate, (P, ER, X, Y, K, PH)); P, ER, X, Y, PH = P[K], ER[K], X[K], Y[K], PH[K]; half = len(Y) // 2
    print("seed", seed, "scored frames", len(Y), "closure share", round(Y.mean(), 4))
    print("Erel P0/20 closure p10/50/90", np.round(np.percentile(ER[Y == 1], [10, 50, 90]), 2), " other speech", np.round(np.percentile(ER[Y == 0], [10, 50, 90]), 2))
    print(score("P5 v1 programmed, half B", P[half:], Y[half:]))
    mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd; g = fit_logistic(Z[:half], Y[:half])(Z[half:])
    print(score("probe: logistic on the 30 inputs, A->B", g, Y[half:]))
    pb, yb = PH[half:], Y[half:]; rows = []
    for kk in range(39):
        m = pb == kk
        if m.sum() > 100: rows.append((((P[half:][m] > .5) == yb[m]).mean(), af.PHONES[kk], int(CLOS[kk]), ((g[m] > .5) == yb[m]).mean()))
    rows.sort(); print("per phone acc programmed (fitted-30) worst 10:", " | ".join(f"{q}{'+' if n else '-'} {a:.3f} ({b2:.3f})" for a, q, n, b2 in rows[:10]))
    if seed == 157:
        au = sorted(((max(roc_auc_score(Y, X[:, j]), 1 - roc_auc_score(Y, X[:, j])), P5_COLS[j]) for j in range(X.shape[1])), reverse=True)
        print("single-input AUROC top 10:", " | ".join(f"{c} {a:.4f}" for a, c in au[:10]))


def p5_cues(E, tr):
    """v2 cues: Erel P0/20 (hole in context), dB P2/60 (dip shape), and two turbulence views from P4 (Ehigh P0/20, R P0/20): a fricative dip is
    noisy and high, a closure is quiet or voice-bar only."""
    db, _ = p5_tracks(E); tc = v.time_course(db[:, None].astype(np.float64))[:, 0]; c4 = p4_cues(E, tr)
    return np.c_[tc[:, 1] - tc[:, 7], tc[:, 6], c4[:, 0], c4[:, 1]]


if __name__ == "__main__" and "p5b" in sys.argv:
    seed = 4242 if "fresh" in sys.argv else 157; C, X, Y, K, PH = [], [], [], [], []
    for E, tr, y in utts2(seed=seed):
        yt, kp = closure_truth(y); C.append(p5_cues(E, tr)); Y.append(yt); K.append(kp); PH.append(y)
        X.append(np.c_[p5_inputs(E), v.time_course(tr[:, [TRK.index("Ehigh"), TRK.index("R")]].astype(np.float64)).reshape(len(y), -1)])
    C, X, Y, K, PH = map(np.concatenate, (C, X, Y, K, PH)); C, X, Y, PH = C[K], X[K], Y[K], PH[K]; half = len(Y) // 2; yb, pb = Y[half:], PH[half:]
    mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd; g = fit_logistic(Z[:half], Y[:half])(Z[half:])
    print("seed", seed, score("probe 50 inputs (dB, Erel, P1, Ehigh, R x10) A->B", g, yb)); print(score("4 cues fitted (5 numbers)", fit_logistic(C[:half], Y[:half])(C[half:]), yb))
    for k, rule in ((1, "sum"), (2, "rootk"), (4, "sum"), (4, "rootk")):
        p = p2_prog(C[:half, :k], Y[:half], rule)(C[half:, :k]); print(score(f"P5 programmed {k} cues rule {rule}", p, yb))
    p = p2_prog(C[:half], Y[:half], "rootk")(C[half:]); rows = []
    for kk in range(39):
        m = pb == kk
        if m.sum() > 100: rows.append((((p[m] > .5) == yb[m]).mean(), af.PHONES[kk], int(CLOS[kk]), ((g[m] > .5) == yb[m]).mean()))
    rows.sort(); print("per phone acc 4-rootk (fitted-50) worst 10:", " | ".join(f"{q}{'+' if n else '-'} {a:.3f} ({b2:.3f})" for a, q, n, b2 in rows[:10]))


def p5_gate(Ca, ya, Cb):
    """P5 v3 (written 12 Nov 22:5x, after v2's seed-157 result, so the fresh set is the check): a WRITTEN GATE, not a sum.
    Stage 1 'dip': v2's 2-cue rootk LLR (Erel P0/20, dB P2/60) + prior -> d = P(closure | dip cues).
    Stage 2 'not noisy, given a dip': equal-variance Gaussian LLRs of Ehigh P0/20 and R P0/20, closure vs other speech, with class means and
    pooled spread measured on half A WEIGHTED by d (so they describe frames that already look like dips: closures vs F / TH / V dips), rootk.
    Combination: logit = logit(d) + d * LLR_noise (the noise evidence counts only as far as the frame is a dip: 'dip AND NOT noisy').
    Primary = soft gate (w = d); the hard gate (w = 1 if d > 0.5 else 0) is reported beside it."""
    f1 = p2_prog(Ca[:, :2], ya, "rootk"); da, db_ = f1(Ca[:, :2]), f1(Cb[:, :2])
    N = Ca[:, 2:4]; w1, w0 = da * ya, da * (1 - ya)
    m1, m0 = (N * w1[:, None]).sum(0) / w1.sum(), (N * w0[:, None]).sum(0) / w0.sum()
    v1, v0 = (((N - m1) ** 2) * w1[:, None]).sum(0) / w1.sum(), (((N - m0) ** 2) * w0[:, None]).sum(0) / w0.sum(); s2 = 0.5 * (v1 + v0)
    llr = lambda Z: ((m1 - m0) / s2 * (Z[:, 2:4] - 0.5 * (m1 + m0))).sum(1) / np.sqrt(2)
    lg = lambda d: np.log(np.clip(d, 1e-12, 1 - 1e-12) / np.clip(1 - d, 1e-12, 1))
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    soft = sig(lg(db_) + db_ * llr(Cb)); hard = sig(lg(db_) + (db_ > 0.5) * llr(Cb))
    return soft, hard, db_, dict(m1=m1.round(3).tolist(), m0=m0.round(3).tolist(), s=np.sqrt(s2).round(3).tolist())


if __name__ == "__main__" and "p5c" in sys.argv:
    seed = 4242 if "fresh" in sys.argv else 157; C, X, Y, K, PH = [], [], [], [], []
    for E, tr, y in utts2(seed=seed):
        yt, kp = closure_truth(y); C.append(p5_cues(E, tr)); Y.append(yt); K.append(kp); PH.append(y)
        X.append(np.c_[p5_inputs(E), v.time_course(tr[:, [TRK.index("Ehigh"), TRK.index("R")]].astype(np.float64)).reshape(len(y), -1)])
    C, X, Y, K, PH = map(np.concatenate, (C, X, Y, K, PH)); C, X, Y, PH = C[K], X[K], Y[K], PH[K]; half = len(Y) // 2; yb, pb = Y[half:], PH[half:]
    assert C.shape[1] == 4 and len(C) == len(Y) == len(X)
    mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd; g = fit_logistic(Z[:half], Y[:half])(Z[half:])
    print("seed", seed, "scored", len(Y), "closure share", round(Y.mean(), 4)); print(score("probe 50 inputs A->B", g, yb))
    soft, hard, d, prm = p5_gate(C[:half], Y[:half], C[half:]); print("noise numbers (d-weighted, half A):", prm)
    print(score("P5 v2 2 cues rootk (stage 1 alone)", d, yb)); print(score("P5 v3 soft gate (primary)", soft, yb)); print(score("P5 v3 hard gate", hard, yb))
    rows = []
    for kk in range(39):
        m = pb == kk
        if m.sum() > 100: rows.append((((soft[m] > .5) == yb[m]).mean(), af.PHONES[kk], int(CLOS[kk]), ((g[m] > .5) == yb[m]).mean(), ((d[m] > .5) == yb[m]).mean()))
    rows.sort(); print("per phone acc v3-soft (fitted-50; stage1) worst 10:", " | ".join(f"{q}{'+' if n else '-'} {a:.3f} ({b2:.3f}; {c2:.3f})" for a, q, n, b2, c2 in rows[:10]))


if __name__ == "__main__" and "p5d" in sys.argv:
    # where do Ehigh / R add information in the 50-input reader? base = the 30 energy inputs; add one Ehigh or R time-course column at a time
    seed = 157; X, Y, K = [], [], []
    for E, tr, y in utts2(seed=seed):
        yt, kp = closure_truth(y); Y.append(yt); K.append(kp)
        X.append(np.c_[p5_inputs(E), v.time_course(tr[:, [TRK.index("Ehigh"), TRK.index("R")]].astype(np.float64)).reshape(len(y), -1)])
    X, Y, K = map(np.concatenate, (X, Y, K)); X, Y = X[K], Y[K]; half = len(Y) // 2; mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd
    cols = P5_COLS + [f"{c}:{k}" for c in ("Ehigh", "R") for k in ("now", "P0/20", "P1/20", "P2/20", "P0/60", "P1/60", "P2/60", "P0/200", "P1/200", "P2/200")]
    base = list(range(30)); a0 = roc_auc_score(Y[half:], fit_logistic(Z[:half][:, base], Y[:half])(Z[half:][:, base])); print("base 30 AUROC", round(a0, 4))
    for nm, grp in (("Ehigh x10", list(range(30, 40))), ("R x10", list(range(40, 50)))):
        c = base + grp; print(nm, "added:", round(roc_auc_score(Y[half:], fit_logistic(Z[:half][:, c], Y[:half])(Z[half:][:, c])) - a0, 4))
    out = []
    for j in range(30, 50):
        c = base + [j]; out.append((roc_auc_score(Y[half:], fit_logistic(Z[:half][:, c], Y[:half])(Z[half:][:, c])) - a0, cols[j]))
    out.sort(reverse=True); print("single added column gain:", " | ".join(f"{c} {g:+.4f}" for g, c in out[:10]))
    j = cols.index("Ehigh:P1/20"); print("Ehigh P1/20 (slope) closure p10/50/90", np.round(np.percentile(X[Y == 1, j], [10, 50, 90]), 3), "other", np.round(np.percentile(X[Y == 0, j], [10, 50, 90]), 3))


if __name__ == "__main__" and "p5e" in sys.argv:
    # P5 v4 (written 12 Nov 22:59 after p5c / p5d on seed 157): add the burst-anticipation cue RISE = Ehigh P1/20 (slope of the high-band share over
    # +-20 ms: a closure is followed within ~20-40 ms by its release burst, so the noise share is rising). Independent evidence -> additive LLR.
    # Primary (chosen before the fresh run): 3 cues (Erel P0/20, dB P2/60, RISE) rule rootk. Run on both seeds; 4242 is the fresh check.
    for seed in (157, 4242):
        C, X, Y, K, PH = [], [], [], [], []
        for E, tr, y in utts2(seed=seed):
            yt, kp = closure_truth(y); eh = v.time_course(tr[:, [TRK.index("Ehigh")]].astype(np.float64))[:, 0]
            C.append(np.c_[p5_cues(E, tr)[:, :2], eh[:, 2]]); Y.append(yt); K.append(kp); PH.append(y)
            X.append(np.c_[p5_inputs(E), v.time_course(tr[:, [TRK.index("Ehigh"), TRK.index("R")]].astype(np.float64)).reshape(len(y), -1)])
        C, X, Y, K, PH = map(np.concatenate, (C, X, Y, K, PH)); C, X, Y, PH = C[K], X[K], Y[K], PH[K]; half = len(Y) // 2; yb, pb = Y[half:], PH[half:]
        mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd; g = fit_logistic(Z[:half], Y[:half])(Z[half:])
        print(f"== seed {seed} scored {len(Y)} closure share {Y.mean():.4f}"); print(score("probe 50 inputs A->B", g, yb))
        print(score("3 cues fitted (4 numbers)", fit_logistic(C[:half], Y[:half])(C[half:]), yb))
        for k, rule in ((2, "rootk"), (3, "sum"), (3, "rootk")):
            p = p2_prog(C[:half, :k], Y[:half], rule)(C[half:, :k]); print(score(f"P5 programmed {k} cues rule {rule}", p, yb))
        p = p2_prog(C[:half], Y[:half], "rootk")(C[half:]); rows = []
        for kk in range(39):
            m = pb == kk
            if m.sum() > 100: rows.append((((p[m] > .5) == yb[m]).mean(), af.PHONES[kk], int(CLOS[kk]), ((g[m] > .5) == yb[m]).mean()))
        rows.sort(); print("per phone acc 3-rootk (fitted-50) worst 8:", " | ".join(f"{q}{'+' if n else '-'} {a:.3f} ({b2:.3f})" for a, q, n, b2 in rows[:8]))


# ---------------- P6 release burst (12 Nov 23:5x) ----------------
# Truth (my choice, written before P6 data; the mirror of P5): positives = the SECOND half of each P B T D K G segment (burst + aspiration);
# the first half (closure) is excluded; negatives = every other speech frame; silence excluded.
def burst_truth(y):
    yt = np.zeros(len(y), int); keep = y != SIL; t = 0
    while t < len(y):
        s = t
        while t < len(y) and y[t] == y[s]: t += 1
        if CLOS[y[s]]:
            h = s + max(1, (t - s) // 2); keep[s:h] = False; yt[h:t] = 1
    return yt, keep


if __name__ == "__main__" and "p6" in sys.argv:
    # premise before data: a release is an abrupt onset after a hole: total energy rising (ONSET = dB P1/20 slope), the noise share high at the
    # frame (EHNOW = Ehigh now: the burst is broadband / high), and a hole just before (Erel P0/20 still low vs +-200 ms).
    # Primary (chosen before running): 3 cues ONSET, EHNOW, Erel P0/20, rule rootk. Both seeds; 4242 = fresh.
    for seed in (157, 4242):
        C, X, Y, K, PH = [], [], [], [], []
        for E, tr, y in utts2(seed=seed):
            yt, kp = burst_truth(y); db, _ = p5_tracks(E); tdb = v.time_course(db[:, None].astype(np.float64))[:, 0]
            eh = v.time_course(tr[:, [TRK.index("Ehigh")]].astype(np.float64))[:, 0]
            C.append(np.c_[tdb[:, 2], eh[:, 0], tdb[:, 1] - tdb[:, 7], eh[:, 2]]); Y.append(yt); K.append(kp); PH.append(y)
            X.append(np.c_[p5_inputs(E), v.time_course(tr[:, [TRK.index("Ehigh"), TRK.index("R")]].astype(np.float64)).reshape(len(y), -1)])
        C, X, Y, K, PH = map(np.concatenate, (C, X, Y, K, PH)); C, X, Y, PH = C[K], X[K], Y[K], PH[K]; half = len(Y) // 2; yb, pb = Y[half:], PH[half:]
        assert C.shape[1] == 4 and len(C) == len(X)
        print(f"== seed {seed} scored {len(Y)} burst share {Y.mean():.4f}")
        if seed == 157:
            for j, nm in enumerate(("ONSET dB P1/20", "EHNOW", "Erel P0/20", "RISE Ehigh P1/20")):
                print(f"  {nm}: burst p10/50/90 {np.round(np.percentile(C[Y == 1, j], [10, 50, 90]), 2)} other {np.round(np.percentile(C[Y == 0, j], [10, 50, 90]), 2)} AUROC {roc_auc_score(Y, C[:, j]):.4f}")
        mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd; g = fit_logistic(Z[:half], Y[:half])(Z[half:])
        print(score("probe 50 inputs A->B", g, yb)); print(score("3 cues fitted (4 numbers)", fit_logistic(C[:half, :3], Y[:half])(C[half:, :3]), yb))
        for k, rule in ((3, "sum"), (3, "rootk"), (4, "rootk")):
            p = p2_prog(C[:half, :k], Y[:half], rule)(C[half:, :k]); print(score(f"P6 programmed {k} cues rule {rule}", p, yb))
        p = p2_prog(C[:half, :3], Y[:half], "rootk")(C[half:, :3]); rows = []
        for kk in range(39):
            m = pb == kk
            if m.sum() > 100: rows.append((((p[m] > .5) == yb[m]).mean(), af.PHONES[kk], int(CLOS[kk]), ((g[m] > .5) == yb[m]).mean()))
        rows.sort(); print("per phone acc 3-rootk (fitted-50) worst 8:", " | ".join(f"{q}{'+' if n else '-'} {a:.3f} ({b2:.3f})" for a, q, n, b2 in rows[:8]))


# ---------------- P7 whole-stop unit (13 Nov 00:4x) ----------------
# Truth (no split): positives = every frame of a P B T D K G segment; negatives = other speech; silence excluded. Chosen because the P5 / P6 half
# split did not separate closure from release (D1P_LOG P6 item 3).
if __name__ == "__main__" and "p7" in sys.argv:
    # premise: a stop is a short hole in its context (Erel P0/20, dB P2/60 dip shape) with an abrupt change at its edge: the noise share and the
    # energy move fast (RISE = Ehigh P1/20, ONSET = dB P1/20; a stop holds BOTH a fall and a rise, so the magnitude |slope| is the cue, not its
    # sign). Primary (chosen before running): 4 cues Erel P0/20, dB P2/60, |RISE|, |ONSET|, rule rootk. Both seeds; 4242 = fresh.
    for seed in (157, 4242):
        C, X, Y, PH = [], [], [], []
        for E, tr, y in utts2(seed=seed):
            sp = y != SIL; db, _ = p5_tracks(E); tdb = v.time_course(db[:, None].astype(np.float64))[:, 0]
            eh = v.time_course(tr[:, [TRK.index("Ehigh")]].astype(np.float64))[:, 0]
            c = np.c_[tdb[:, 1] - tdb[:, 7], tdb[:, 6], np.abs(eh[:, 2]), np.abs(tdb[:, 2])]
            x = np.c_[p5_inputs(E), v.time_course(tr[:, [TRK.index("Ehigh"), TRK.index("R")]].astype(np.float64)).reshape(len(y), -1)]
            C.append(c[sp]); X.append(x[sp]); Y.append(CLOS[y[sp]].astype(int)); PH.append(y[sp])
        C, X, Y, PH = map(np.concatenate, (C, X, Y, PH)); half = len(Y) // 2; yb, pb = Y[half:], PH[half:]
        assert C.shape[1] == 4 and len(C) == len(X) == len(Y)
        print(f"== seed {seed} speech frames {len(Y)} stop share {Y.mean():.4f}")
        if seed == 157:
            for j, nm in enumerate(("Erel P0/20", "dB P2/60", "|RISE|", "|ONSET|")):
                print(f"  {nm}: stop p10/50/90 {np.round(np.percentile(C[Y == 1, j], [10, 50, 90]), 2)} other {np.round(np.percentile(C[Y == 0, j], [10, 50, 90]), 2)} AUROC {roc_auc_score(Y, C[:, j]):.4f}")
        mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd; g = fit_logistic(Z[:half], Y[:half])(Z[half:])
        print(score("probe 50 inputs A->B", g, yb)); print(score("4 cues fitted (5 numbers)", fit_logistic(C[:half], Y[:half])(C[half:]), yb))
        for k, rule in ((2, "rootk"), (4, "sum"), (4, "rootk")):
            p = p2_prog(C[:half, :k], Y[:half], rule)(C[half:, :k]); print(score(f"P7 programmed {k} cues rule {rule}", p, yb))
        p = p2_prog(C[:half], Y[:half], "rootk")(C[half:]); rows = []
        for kk in range(39):
            m = pb == kk
            if m.sum() > 100: rows.append((((p[m] > .5) == yb[m]).mean(), af.PHONES[kk], int(CLOS[kk]), ((g[m] > .5) == yb[m]).mean()))
        rows.sort(); print("per phone acc 4-rootk (fitted-50) worst 8:", " | ".join(f"{q}{'+' if n else '-'} {a:.3f} ({b2:.3f})" for a, q, n, b2 in rows[:8]))


# ---------------- P5 / P6 on the landmark truth (13 Nov 02:2x; D1P_LOG landmark item 4) ----------------
def landmark_truth(y, db):
    """per stop segment [s, t) of >= 2 frames: r = argmax dB rise over [s-1, t+1). closure = [s, r) (P5 positives), release = [r, r+3) clipped to
    t+1 (P6 positives). Returns (yc, kc, yr, kr): truth + keep mask for each unit; the stop's other frames are excluded from that unit."""
    d = np.r_[0, np.diff(db)]; n = len(y); yc, yr = np.zeros(n, int), np.zeros(n, int); kc, kr = y != SIL, y != SIL; t = 0
    while t < n:
        s = t
        while t < n and y[t] == y[s]: t += 1
        if CLOS[y[s]]:
            kc[s:t] = False; kr[s:t] = False
            if t - s >= 2:
                a, b2 = max(1, s - 1), min(n, t + 1); r = a + int(np.argmax(d[a:b2]))
                if r > s: yc[s:r] = 1; kc[s:r] = True
                e = min(r + 3, t + 1, n); yr[r:e] = 1; kr[r:e] = True
    return yc, kc, yr, kr


if __name__ == "__main__" and "p56l" in sys.argv:
    # P5 cues = v4 primary (Erel P0/20, dB P2/60, RISE), rootk. P6 cues WITHOUT ONSET (the landmark is a dB rise): EHNOW, Erel P0/20, RISE, rootk.
    # Primaries chosen before running. Both seeds; 4242 = fresh.
    for seed in (157, 4242):
        D = {u_: ([], [], [], []) for u_ in ("P5", "P6")}
        for E, tr, y in utts2(seed=seed):
            db, _ = p5_tracks(E); tdb = v.time_course(db[:, None].astype(np.float64))[:, 0]; eh = v.time_course(tr[:, [TRK.index("Ehigh")]].astype(np.float64))[:, 0]
            yc, kc, yr, kr = landmark_truth(y, db)
            x = np.c_[p5_inputs(E), v.time_course(tr[:, [TRK.index("Ehigh"), TRK.index("R")]].astype(np.float64)).reshape(len(y), -1)]
            for nm, c, yt, kp in (("P5", np.c_[tdb[:, 1] - tdb[:, 7], tdb[:, 6], eh[:, 2]], yc, kc), ("P6", np.c_[eh[:, 0], tdb[:, 1] - tdb[:, 7], eh[:, 2]], yr, kr)):
                D[nm][0].append(c[kp]); D[nm][1].append(x[kp]); D[nm][2].append(yt[kp]); D[nm][3].append(y[kp])
        for nm in ("P5", "P6"):
            C, X, Y, PH = map(np.concatenate, D[nm]); half = len(Y) // 2; yb = Y[half:]; assert len(C) == len(X) == len(Y)
            mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd; g = fit_logistic(Z[:half], Y[:half])(Z[half:])
            print(f"== seed {seed} {nm} scored {len(Y)} positive share {Y.mean():.4f}"); print(score("  probe 50 inputs A->B", g, yb))
            print(score("  3 cues fitted (4 numbers)", fit_logistic(C[:half], Y[:half])(C[half:]), yb))
            print(score(f"  {nm} programmed 3 cues rootk (primary)", p2_prog(C[:half], Y[:half], "rootk")(C[half:]), yb))


if __name__ == "__main__" and "p6l2" in sys.argv:
    # the P6 landmark truth comes from the dB rise, so the 50-input reader's dB / Erel slope columns (P1/*) see the truth's own definition.
    # Fair reference: the 50 inputs minus dB:P1/* and Erel:P1/* (6 columns) -> 44 inputs. Also the non-circular programmed P6 vs it.
    cols = P5_COLS + [f"{c}:{k}" for c in ("Ehigh", "R") for k in ("now", "P0/20", "P1/20", "P2/20", "P0/60", "P1/60", "P2/60", "P0/200", "P1/200", "P2/200")]
    keepc = [j for j, c in enumerate(cols) if not (c.split(":")[0] in ("dB", "Erel") and ":P1/" in c)]; assert len(keepc) == 44
    for seed in (157, 4242):
        C, X, Y = [], [], []
        for E, tr, y in utts2(seed=seed):
            db, _ = p5_tracks(E); tdb = v.time_course(db[:, None].astype(np.float64))[:, 0]; eh = v.time_course(tr[:, [TRK.index("Ehigh")]].astype(np.float64))[:, 0]
            _, _, yr, kr = landmark_truth(y, db)
            x = np.c_[p5_inputs(E), v.time_course(tr[:, [TRK.index("Ehigh"), TRK.index("R")]].astype(np.float64)).reshape(len(y), -1)][:, keepc]
            C.append(np.c_[eh[:, 0], tdb[:, 1] - tdb[:, 7], eh[:, 2]][kr]); X.append(x[kr]); Y.append(yr[kr])
        C, X, Y = map(np.concatenate, (C, X, Y)); half = len(Y) // 2; yb = Y[half:]
        mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd
        print(f"== seed {seed}", score("probe 44 inputs (no dB / Erel slopes) A->B", fit_logistic(Z[:half], Y[:half])(Z[half:]), yb))
        print(score("  P6 programmed 3 cues rootk", p2_prog(C[:half], Y[:half], "rootk")(C[half:]), yb))


if __name__ == "__main__" and "p6x" in sys.argv:
    # crossed definitions (13 Nov 03:2x; no corpus with closure / release marks on disk: no TIMIT; Buckeye labels have no closures, checked).
    # landmark_E = argmax of the Ehigh first difference inside [s-1, t+1). (1) agreement with the dB landmark. (2) P6 truth from landmark_E,
    # scored with dB-only cues (ONSET dB P1/20, Erel P0/20, dB P2/60; rootk, primary chosen now) vs the 30 energy-only inputs (no Ehigh at all).
    for seed in (157, 4242):
        C, X, Y, dist = [], [], [], []
        for E, tr, y in utts2(seed=seed):
            db, _ = p5_tracks(E); tdb = v.time_course(db[:, None].astype(np.float64))[:, 0]; ehr = tr[:, TRK.index("Ehigh")].astype(np.float64)
            dd, de = np.r_[0, np.diff(db)], np.r_[0, np.diff(ehr)]; n = len(y); yr = np.zeros(n, int); kr = y != SIL; t = 0
            while t < n:
                s = t
                while t < n and y[t] == y[s]: t += 1
                if CLOS[y[s]]:
                    kr[s:t] = False
                    if t - s >= 2:
                        a, b2 = max(1, s - 1), min(n, t + 1); rE = a + int(np.argmax(de[a:b2])); rD = a + int(np.argmax(dd[a:b2])); dist.append(rE - rD)
                        e = min(rE + 3, t + 1, n); yr[rE:e] = 1; kr[rE:e] = True
            C.append(np.c_[tdb[:, 2], tdb[:, 1] - tdb[:, 7], tdb[:, 6]][kr]); X.append(p5_inputs(E)[kr]); Y.append(yr[kr])
        C, X, Y = map(np.concatenate, (C, X, Y)); half = len(Y) // 2; yb = Y[half:]; dist = np.array(dist)
        print(f"== seed {seed} landmark_E - landmark_dB frames: |d|=0 {np.mean(dist == 0):.3f} <=1 {np.mean(np.abs(dist) <= 1):.3f} <=2 {np.mean(np.abs(dist) <= 2):.3f} median {np.median(dist):.0f} (n {len(dist)})")
        mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-9; Z = (X - mu) / sd
        print(score("  probe 30 energy-only inputs A->B", fit_logistic(Z[:half], Y[:half])(Z[half:]), yb), "| positive share", round(Y.mean(), 4))
        print(score("  3 dB cues fitted (4 numbers)", fit_logistic(C[:half], Y[:half])(C[half:]), yb))
        print(score("  P6x programmed 3 dB cues rootk (primary)", p2_prog(C[:half], Y[:half], "rootk")(C[half:]), yb))


# ---------------- generic programmed node (13 Nov 04:1x; D1P_PATH.md link 1) ----------------
def prog_node(Ca, ya, m, rule="rootk"):
    """m-class programmed node: per class c, MEASURED cue means (k numbers) + one pooled within-class spread per cue + prior, closed form;
    score_c = log prior_c - sum_j (z_j - mean_cj)^2 / (2 s_j^2) / div, div from the written rule (sum 1, rootk sqrt k, mean k); softmax.
    For m = 2 it equals p2_prog except that p2_prog pools only the two class variances with equal weight (asserted close in 'nodecheck')."""
    k = Ca.shape[1]; M = np.stack([Ca[ya == c].mean(0) for c in range(m)]); pri = np.array([(ya == c).mean() for c in range(m)])
    s2 = np.mean([Ca[ya == c].var(0) for c in range(m)], 0); div = dict(sum=1.0, mean=float(k), rootk=float(np.sqrt(k)))[rule]
    def f(Z):
        sc = np.log(pri)[None] - (((Z[:, None, :] - M[None]) ** 2) / (2 * s2)).sum(2) / div; sc -= sc.max(1, keepdims=True); e = np.exp(sc); return e / e.sum(1, keepdims=True)
    return f


if __name__ == "__main__" and "nodecheck" in sys.argv:
    # equivalence check on P2 (seed 157): prog_node(m=2) column 1 vs p2_prog, same cues, same rule
    C, Y, PH = [], [], []
    for E, tr, y in utts2(seed=157): C.append(p2_cues4(E, tr)); Y.append(VOICED[y].astype(int)); PH.append(y)
    C, Y, PH = map(np.concatenate, (C, Y, PH)); sp = PH != SIL; C, Y = C[sp], Y[sp]; half = len(Y) // 2
    a = p2_prog(C[:half], Y[:half], "rootk")(C[half:]); b2 = prog_node(C[:half], Y[:half], 2)(C[half:])[:, 1]
    print(score("p2_prog", a, Y[half:])); print(score("prog_node m=2", b2, Y[half:])); print("max |diff|", float(np.abs(a - b2).max()), "corr", float(np.corrcoef(a, b2)[0, 1]))
    assert np.corrcoef(a, b2)[0, 1] > 0.999, "prog_node does not reproduce p2_prog"


# ---------------- broad-class node, flat 7-way (13 Nov 05:0x; D1P_PATH.md item a, node level first) ----------------
BROAD = ["vowel", "nasal", "liquid-glide", "HH", "stop", "fricative", "affricate"]
_bs = dict(nasal="M N NG", **{"liquid-glide": "L R W Y"}, HH="HH", stop="P B T D K G", fricative="F V TH DH S Z SH ZH", affricate="CH JH")
BCLS = np.array([next((BROAD.index(c) for c, ps in _bs.items() if p in ps.split()), 0) for p in af.PHONES] + [-1])


def broad_feats(E, tr):
    """cue SOURCES per frame (unit cues, combined into unit logits later with half-A measured numbers):
    p2 (4 cues), p4 (3 cues), p5s (4 cues), NAS = Elow P0/20 - Emid P0/20 (nasal murmur: low share up, mid down), VOW = Erel P0/60
    (nucleus: loudest vs +-200 ms), LF3 = lF3 P0/20 (R lowers F3), GLIDE = |lF2 P1/60| (glides move F2)."""
    db, _ = p5_tracks(E); tdb = v.time_course(db[:, None].astype(np.float64))[:, 0]; eh = v.time_course(tr[:, [TRK.index("Ehigh")]].astype(np.float64))[:, 0]
    p5s = np.c_[tdb[:, 1] - tdb[:, 7], tdb[:, 6], np.abs(eh[:, 2]), np.abs(tdb[:, 2])]
    T = v.time_course(tr[:, [TRK.index(c) for c in ("Elow", "Emid", "lF3", "lF2")]].astype(np.float64))
    own = np.c_[T[:, 0, 1] - T[:, 1, 1], tdb[:, 4] - tdb[:, 7], T[:, 2, 1], np.abs(T[:, 3, 5])]
    return np.c_[p2_cues4(E, tr), p4_cues(E, tr), p5s, own]


def broad_wide(E, tr):
    """equal-information reference inputs: dB, Erel + 8 tracks the cues draw from, each x 10 time course -> 100."""
    db, er = p5_tracks(E); T = tr[:, [TRK.index(c) for c in ("R", "F0", "Elow", "Emid", "Ehigh", "BH", "lF2", "lF3")]]
    return v.time_course(np.c_[db, er, T].astype(np.float64)).reshape(len(db), -1)


def ovr(Xa, ya, m):
    fs = [fit_logistic(Xa, (ya == c).astype(int)) for c in range(m)]
    return lambda Z: (lambda P: P / P.sum(1, keepdims=True))(np.stack([f(Z) for f in fs], 1))


def mscore(name, P, y, m):
    pr = P.argmax(1); rec = [((pr == c) & (y == c)).sum() / max(1, (y == c).sum()) for c in range(m)]
    ll = -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)))
    return f"{name}: acc {np.mean(pr == y):.4f} macro-recall {np.mean(rec):.4f} log-loss {ll:.4f} | " + " ".join(f"{BROAD[c][:5]} {rec[c]:.3f}" for c in range(m))


if __name__ == "__main__" and "broad" in sys.argv:
    # Primary (chosen before running): unit logits (P2 voicing, P4 frication, P5s stop: each prog rootk on its own truth, measured on half A) +
    # the 4 own cues -> prog_node 7-way rootk (k = 7). References: one-vs-rest logistic on the same 7 node inputs, and on the 100 wide inputs.
    lg = lambda p: np.log(np.clip(p, 1e-9, 1) / np.clip(1 - p, 1e-9, 1))
    for seed in (157, 4242):
        F, W, Y, PH = [], [], [], []
        for E, tr, y in utts2(seed=seed):
            sp = (y != SIL) & (y >= 0); F.append(broad_feats(E, tr)[sp]); W.append(broad_wide(E, tr)[sp]); Y.append(BCLS[y[sp]]); PH.append(y[sp])
        F, W, Y, PH = map(np.concatenate, (F, W, Y, PH)); half = len(Y) // 2; A, B = slice(0, half), slice(half, None); assert (Y >= 0).all()
        u2 = p2_prog(F[A, 0:4], VOICED[PH[A]].astype(int), "rootk"); u4 = p2_prog(F[A, 4:7], NOISY[PH[A]].astype(int), "rootk"); u5 = p2_prog(F[A, 7:11], CLOS[PH[A]].astype(int), "rootk")
        N = np.c_[lg(u2(F[:, 0:4])), lg(u4(F[:, 4:7])), lg(u5(F[:, 7:11])), F[:, 11:15]]
        print(f"== seed {seed} speech frames {len(Y)} class shares", " ".join(f"{BROAD[c][:5]} {np.mean(Y == c):.3f}" for c in range(7)))
        print(mscore("  programmed node 7-way rootk (primary)", prog_node(N[A], Y[A], 7)(N[B]), Y[B], 7))
        print(mscore("  programmed node 7-way sum", prog_node(N[A], Y[A], 7, "sum")(N[B]), Y[B], 7))
        print(mscore("  fitted OvR on the same 7 inputs", ovr(N[A], Y[A], 7)(N[B]), Y[B], 7))
        mu, sd = W[A].mean(0), W[A].std(0) + 1e-9; Z = (W - mu) / sd; print(mscore("  fitted OvR on 100 wide inputs", ovr(Z[A], Y[A], 7)(Z[B]), Y[B], 7))
