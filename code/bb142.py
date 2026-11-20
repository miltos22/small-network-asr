"""exp142 black box: 80 log-mel -> conv (stride 2) -> BiGRU -> 39 phones + blank, CTC, greedy decode (DESIGN_exp142.md).
Same audio, labels and refs as the kept system: fit/cal = phones.split() files with ph_{fit,cal}.npz labels; eval sets = realcut_*.npz
files + segs labels (the exact refs phones46.run_set scores, phones46.py:28). Scorer = p44.edit, PER = sum edits / sum ref length.
Usage: bb142.py feats                                   (build the log-mel caches)
       bb142.py train SHAPE LR SEED EVALSETS [EPOCHS]  (SHAPE wide2 | narrow3; EVALSETS comma list, first = early-stop set)
Env: SHUF=1 shuffles label sequences across training utterances (broken control); PARAMS overrides the budget.
RECIPE=b0 (exp143): standard extras = SpecAugment (2 freq masks <= 15 of 80, 2 time masks <= min(40, 10 % of length)), AdamW wd 1e-2,
average of the 3 best libriA checkpoints (kept only if it is not worse on the early-stop set).
CTX=k (exp144, node P2): the model sees each TRUE segment alone plus k frames each side (k=0: no cross-segment context); outputs are
taken only for the segment's own frames and concatenated per utterance, CTC on the utterance. Uses true cuts at test time too. fp32 only."""
import os
import json, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch, torch.nn as nn
import frontend as fe, phones, phones46 as p46

C = phones.CACHE; O = f"{phones.OUT}/exp142"; NPH = 39; HOP, NFFT, NMEL = 160, 400, 80


def melbank(sr=16000, n=NMEL, nfft=NFFT):
    hz = lambda m: 700 * (10 ** (m / 2595) - 1); mel = lambda f: 2595 * np.log10(1 + f / 700)
    pts = hz(np.linspace(mel(20), mel(sr / 2), n + 2)); fr = np.fft.rfftfreq(nfft, 1 / sr); fb = np.zeros((n, len(fr)), np.float32)
    for i in range(n):
        l, c, r = pts[i:i + 3]; fb[i] = np.clip(np.minimum((fr - l) / (c - l), (r - fr) / (r - c)), 0, None)
    return torch.from_numpy(fb)


FB = melbank()


def logmel(path):
    x = torch.from_numpy(fe.load(path)); S = torch.stft(x, NFFT, HOP, window=torch.hann_window(NFFT), return_complex=True).abs() ** 2
    m = torch.log(FB @ S + 1e-6).T  # (frames, 80)
    return ((m - m.mean(0)) / (m.std(0) + 1e-5)).numpy().astype(np.float16)  # per-utterance norm (the kept lattice is offline too)


def labelled(name):
    """(files, label sequences, frame counts) in the cache's own utterance order."""
    if name in ("fit", "cal"):
        d = np.load(f"{C}/ph_{name}.npz"); files = phones.split()[name]
    else:
        d = np.load(f"{C}/realcut_{name}.npz"); files = list(d["files"])
        if name == "author": import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import asr_compare as ac; files = list(ac.TRUTH)[:10]  # as phones42.py:35
    lens = d["lens"]; assert len(files) == len(lens), (name, len(files), len(lens))
    u0 = np.searchsorted(np.cumsum(lens), d["segs"][:, 0], side="right")
    return files, [d["segs"][u0 == u, 2].astype(np.int64) for u in range(len(lens))], lens


def feats():
    for name in ("fit", "cal", "libriA", "libriB", "testother", "author"):
        p = f"{C}/bb142_{name}.npz"
        if os.path.exists(p): continue
        files, Y, lens = labelled(name); X = [logmel(f) for f in files]
        dev = np.array([len(x) - n for x, n in zip(X, lens)]); print(name, len(X), "frames vs cache lens: max |diff|", np.abs(dev).max(), "values", sorted(set(dev.tolist())), flush=True)
        assert np.abs(dev).max() <= 3, f"{name}: audio does not match the cached utterances"
        np.savez(p, X=np.concatenate(X), lens=np.array([len(x) for x in X]), **{f"y{i}": y for i, y in enumerate(Y)}, n=len(Y))


def load(name):
    d = np.load(f"{C}/bb142_{name}.npz"); X = torch.from_numpy(d["X"].astype(np.float32)); st = np.r_[0, np.cumsum(d["lens"])]
    return [X[a:b] for a, b in zip(st[:-1], st[1:])], [d[f"y{i}"] for i in range(int(d["n"]))]


class BB(nn.Module):
    def __init__(s, ch, hid, layers):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(NMEL, ch, 5, 2, 2), nn.ReLU(), nn.Conv1d(ch, ch, 3, 1, 1), nn.ReLU())
        s.rnn = nn.GRU(ch, hid, layers, batch_first=True, bidirectional=True, dropout=0.1 if layers > 1 else 0)
        s.out = nn.Linear(2 * hid, NPH + 1)  # index NPH = blank

    def forward(s, x, lens=None):  # x (B, T, 80) -> (B, T/2, 40) log-probs; lens -> packed GRU (padding never enters the recurrence)
        h = s.conv(x.transpose(1, 2)).transpose(1, 2)
        if lens is None: return s.out(s.rnn(h)[0]).log_softmax(-1)
        pk = nn.utils.rnn.pack_padded_sequence(h, (lens + 1) // 2, batch_first=True, enforce_sorted=False)
        return s.out(nn.utils.rnn.pad_packed_sequence(s.rnn(pk)[0], batch_first=True, total_length=h.shape[1])[0]).log_softmax(-1)


def starts(name):
    """Per-utterance TRUE segment start frames (relative to the utterance), same order as labelled()."""
    d = np.load(f"{C}/ph_{name}.npz") if name in ("fit", "cal") else np.load(f"{C}/realcut_{name}.npz")
    off = np.r_[0, np.cumsum(d["lens"])]; u0 = np.searchsorted(off[1:], d["segs"][:, 0], side="right")
    return [np.unique(d["segs"][u0 == u, 0] - off[u]).astype(np.int64) for u in range(len(d["lens"]))]


def plan(T, st, k):
    """Windows (w0, w1) per segment and, per output frame j < ceil(T/2), (window, position in that window's output)."""
    T2 = (T + 1) // 2; st = st[st < T]; st = st if len(st) and st[0] == 0 else np.r_[0, st]; j2 = 2 * np.arange(T2)
    own = np.clip(np.searchsorted(st, j2, side="right") - 1, 0, len(st) - 1); e = np.r_[st[1:], T]
    w0 = np.maximum(0, st - k) // 2 * 2; w1 = np.minimum(T, e + k); return np.c_[w0, w1], own, (j2 - w0[own]) // 2


def fwd_ctx(m, xs, pl):  # xs: list of (T, 80); pl: their plan()s -> list of (ceil(T/2), 40) log-probs
    W, own, pos, base = [], [], [], 0
    for x, (win, o, p) in zip(xs, pl):
        W += [x[a:b] for a, b in win]; own.append(torch.from_numpy(o + base)); pos.append(torch.from_numpy(p)); base += len(win)
    out = m(nn.utils.rnn.pad_sequence(W, batch_first=True), torch.tensor([len(w) for w in W])); return [out[o, p] for o, p in zip(own, pos)]


def build(shape, P):
    layers = dict(wide2=2, narrow3=3)[shape]; best = None
    for hid in range(16, 1024):
        ch = hid; m = BB(ch, hid, layers); n = sum(p.numel() for p in m.parameters())
        if best is None or abs(n - P) < abs(best[1] - P): best = (hid, n)
        if n > P: break
    m = BB(best[0], best[0], layers); n = sum(p.numel() for p in m.parameters()); assert abs(n - P) / P < 0.02, (n, P); return m, n


def decode(m, X, PL=None):
    m.eval(); H = []
    with torch.no_grad():
        for i, x in enumerate(X):
            a = (fwd_ctx(m, [x], [PL[i]])[0] if PL else m(x[None])[0]).argmax(-1).tolist(); H.append([p for i, p in enumerate(a) if p != NPH and (i == 0 or p != a[i - 1])])
    return H


def score(H, Y):
    n = sum(len(y) for y in Y); e = [int(p46.p44.edit(h, list(y))) for h, y in zip(H, Y)]
    return dict(PER=round(sum(e) / n, 4), hyp_over_ref=round(sum(map(len, H)) / n, 3)), [[a, len(y)] for a, y in zip(e, Y)]


def specaug(x, lens, rng):  # x (B, T, 80) padded; masks set to 0 = the per-utterance mean
    x = x.clone()
    for b, L in enumerate(lens):
        for _ in range(2): f = int(rng.integers(0, 16)); f0 = int(rng.integers(0, NMEL - f + 1)); x[b, :, f0:f0 + f] = 0
        for _ in range(2): t = int(rng.integers(0, min(40, L // 10) + 1)); t0 = int(rng.integers(0, max(1, L - t + 1))); x[b, t0:t0 + t] = 0
    return x


def train(shape, lr, seed, evals, epochs=60):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    P = int(os.environ.get("PARAMS") or json.load(open(f"{O}/budget.json"))["params"]); m, n = build(shape, P)
    X, Y = [], []
    for s_ in ("fit", "cal"): x, y = load(s_); X += x; Y += y
    if os.environ.get("SHUF"): Y = [Y[i] for i in rng.permutation(len(Y))]
    E = {s_: load(s_) for s_ in evals}; tag = f"{shape}_lr{lr}_s{seed}" + ("_shuf" if os.environ.get("SHUF") else ""); B0 = os.environ.get("RECIPE") == "b0"; tag += "_b0" if B0 else ""; snaps = []
    CTX = os.environ.get("CTX"); assert not (CTX and B0), "exp144 uses the minimal recipe"; PL, EP = None, {}
    if CTX:
        k = int(CTX); tag += f"_ctx{k}"; S = starts("fit") + starts("cal"); assert len(S) == len(X); PL = [plan(len(x), st, k) for x, st in zip(X, S)]
        EP = {s_: [plan(len(x), st, k) for x, st in zip(E[s_][0], starts(s_))] for s_ in evals}
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-2) if B0 else torch.optim.Adam(m.parameters(), lr); ctc = nn.CTCLoss(blank=NPH, zero_infinity=True)
    order = np.argsort([len(x) for x in X]); batches = []; cur = []; fr = 0
    for i in order:  # length-sorted batches of <= 30k input frames
        if cur and fr + len(X[i]) > 30000: batches.append(cur); cur, fr = [], 0
        cur.append(i); fr += len(X[i])
    batches.append(cur); steps = epochs * len(batches); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    best, bad, log, t0 = None, 0, [], time.time(); stop = evals[0]
    for ep in range(epochs):
        m.train(); tot = 0.0
        for b in rng.permutation(len(batches)):
            ix = batches[b]; xs = nn.utils.rnn.pad_sequence([X[i] for i in ix], batch_first=True); xs = specaug(xs, [len(X[i]) for i in ix], rng) if B0 else xs
            lp = nn.utils.rnn.pad_sequence(fwd_ctx(m, [X[i] for i in ix], [PL[i] for i in ix])) if CTX else m(xs).transpose(0, 1); il = torch.tensor([(len(X[i]) + 1) // 2 for i in ix]); tl = torch.tensor([len(Y[i]) for i in ix])
            loss = ctc(lp, torch.from_numpy(np.concatenate([Y[i] for i in ix])), il, tl)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step(); tot += loss.item() * len(ix)
        r = dict(epoch=ep + 1, loss=round(tot / len(X), 4), secs=round(time.time() - t0))
        if (ep + 1) % int(os.environ.get("EVALEVERY", 2)) == 0 or ep + 1 == epochs:  # EVALEVERY: exp157 dose reruns (default 2 = unchanged)
            r[stop] = score(decode(m, E[stop][0], EP.get(stop)), E[stop][1])[0]
            if best is None or r[stop]["PER"] < best[0]: best, bad = (r[stop]["PER"], ep + 1, {k: v.clone() for k, v in m.state_dict().items()}), 0
            elif not (os.environ.get("BLANKWAIT") and r[stop]["PER"] >= 0.95): bad += 1  # exp157 fix: no patience spent on the blank-only plateau
            if B0: snaps = sorted(snaps + [(r[stop]["PER"], ep + 1, {k: v.clone() for k, v in m.state_dict().items()})], key=lambda t: t[0])[:3]
        print(json.dumps(r), flush=True); log.append(r)
        if bad >= 5: break
    m.load_state_dict(best[2])
    if B0 and len(snaps) > 1:
        avg = {k: sum(t[2][k] for t in snaps) / len(snaps) for k in best[2]}; m.load_state_dict(avg); ra = score(decode(m, E[stop][0], EP.get(stop)), E[stop][1])[0]
        print(json.dumps(dict(avg_of=[t[1] for t in snaps], **{stop: ra}, best_single=best[0])), flush=True)
        if ra["PER"] <= best[0]: best = (ra["PER"], "avg" + "+".join(str(t[1]) for t in snaps), avg)
        else: m.load_state_dict(best[2])
    torch.save(best[2], f"{O}/bb_{tag}.pt")
    q = torch.ao.quantization.quantize_dynamic(m, {nn.GRU, nn.Linear}, dtype=torch.qint8)  # int8 weights; the 2 convs stay fp32
    R = dict(tag=tag, params=n, budget=P, best_epoch=best[1], log=log, sets={})
    for s_ in evals:
        f32, pu = score(decode(m, E[s_][0], EP.get(s_)), E[s_][1]); i8, pu8 = (None, pu) if CTX else score(decode(q, E[s_][0]), E[s_][1]); R["sets"][s_] = dict(fp32=f32, int8=i8, per_utt_int8=pu8)
        print(s_, "fp32", f32, "int8", i8, flush=True)
    json.dump(R, open(f"{O}/bb_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(O, exist_ok=True); torch.set_num_threads(int(os.environ.get("THREADS", 6)))
    if sys.argv[1] == "feats": feats()
    else: train(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split(","), int(sys.argv[6]) if len(sys.argv) > 6 else 60)
