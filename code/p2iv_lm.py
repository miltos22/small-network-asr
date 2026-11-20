"""docs/PLAN.md Phase 2 row 2-IV, step 1: text knowledge = phone n-gram from the TRAINING transcripts (fit + cal, the 5.4 h set base6
was trained on) + CTC prefix beam with shallow fusion, on cached base6 log-probs. CPU, inference only (python3, nice 19, 2 processes).
Rule (plan): LM gain over the same beam without LM < 0.5 points on hard (mean libriA, testother) -> park THIS n-gram.
Selection (order n in {2,3,4}, weight lam, insertion bonus beta) on 150 libriB utterances ONLY (libriB is never decisive); reported: libriA,
testother, cleanS, author2, him with greedy / beam W / beam W + LM. Models: base6 e180_r12_causal_s0 and s1 (the product schedule; two seeds).
Known answers: KN distribution sums to 1 over the vocabulary for random contexts; lam = beta = 0 gives exactly the no-LM beam.
LM bytes: number of stored n-grams x (1 byte prob + 1 byte backoff) as a lower bound (dense trigram 40^3 = 64,000 B for reference).
17 Nov ~23:0x (reviewer). Usage: RIGHT=12 NORM=causal python3 p2iv_lm.py"""
import os
import json, os, sys, time, itertools, collections
os.environ["CUDA_VISIBLE_DEVICES"] = ""; assert os.environ.get("RIGHT") == "12" and os.environ.get("NORM", "causal") == "causal"
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, torch
import base5_next as B, base6 as B6
from multiprocessing import Pool
torch.set_num_threads(2); os.nice(19); I = os.environ.get("SCRATCH_DIR", "scratch") + "/p2iv"; os.makedirs(I, exist_ok=True)
OUTJ = D + "/logs_base6/p2iv_lm_" + os.environ.get("RUN", "2026-11-17") + ".json"; NPH = B.NPH; V = NPH + 1  # 39 phones + </s>; <s> only as context
# 18 Nov 00:1x (reviewer): grid from env for the extended run (defaults = the 17 Nov grid)
GN = [int(v) for v in os.environ.get("GN", "2,3,4").split(",")]; GL = [float(v) for v in os.environ.get("GL", "0.1,0.2,0.4,0.7").split(",")]
GB = [float(v) for v in os.environ.get("GB", "0.0,0.5,1.0,2.0").split(",")]
W = 16; SEL_N = 150; MODELS = os.environ.get("MODELS", "e180_r12_causal_s0,e180_r12_causal_s1").split(","); SETS = ["libriB", "libriA", "testother", "cleanS", "author2", "author"]


def lps(tag, name):
    f = f"{I}/lp_{tag}_{name}.npz"
    if os.path.exists(f): d = np.load(f, allow_pickle=True); return list(d["lp"]), list(d["y"])
    m = B6.Base6(B.build(655924)); m.load_state_dict(torch.load(f"{B6.OUT}/{tag}.pt", map_location="cpu")); m.eval(); X, Y = B.load(name); out = []
    with torch.no_grad():
        for x in X: out.append(m(B6.norm(x.float())[None])[0].numpy().astype(np.float32))
    np.savez(f, lp=np.array(out, dtype=object), y=np.array([np.asarray(y) for y in Y], dtype=object)); return out, list(Y)


class KN:  # interpolated Kneser-Ney, fixed discount 0.75, order n; tokens 0..NPH-1 phones, NPH = </s>, -1 = <s>
    def __init__(s, sents, n):
        s.n = n; s.c = [collections.Counter() for _ in range(n + 1)]
        for y in sents:
            t = [-1] * (n - 1) + list(map(int, y)) + [NPH]
            for i in range(n - 1, len(t)):
                for k in range(1, n + 1): s.c[k][tuple(t[i - k + 1:i + 1])] += 1
        s.cont = [collections.Counter() for _ in range(n + 1)]  # continuation counts for lower orders
        for k in range(2, n + 1):
            for g in s.c[k]: s.cont[k - 1][g[1:]] += 1
        s.ctx = [collections.defaultdict(lambda: [0, 0]) for _ in range(n + 1)]  # history -> [total, distinct followers]
        for k in range(1, n + 1):
            src = s.c[k] if k == n else s.cont[k]
            for g, v in src.items(): h = s.ctx[k][g[:-1]]; h[0] += v; h[1] += 1
        s.cache = {}
    def p(s, h, w, k=None):  # P(w | last k-1 tokens of h)
        k = s.n if k is None else k
        if k == 0: return 1.0 / V
        hh = tuple(h[len(h) - (k - 1):]) if k > 1 else (); key = (hh, w, k)
        if key in s.cache: return s.cache[key]
        src = s.c[k] if k == s.n else s.cont[k]; tot, dist = s.ctx[k].get(hh, (0, 0)) if hh in s.ctx[k] else (0, 0)
        if tot == 0: r = s.p(h, w, k - 1)
        else: r = max(src.get(hh + (w,), 0) - 0.75, 0) / tot + 0.75 * dist / tot * s.p(h, w, k - 1)
        s.cache[key] = r; return r
    def lp(s, h, w): return np.log(s.p(([-1] * (s.n - 1) + list(h))[-(s.n - 1):] if s.n > 1 else [], w))
    def nbytes(s): return 2 * sum(len(c) for c in s.c[1:])


def lse(a, b): return a if b == -np.inf else b if a == -np.inf else max(a, b) + np.log1p(np.exp(-abs(a - b)))


def beam(lp, lm, lam, beta, Wd=W):  # CTC prefix beam; prefix score = acoustic lse(pb, pnb) + lam * LM(prefix) + beta * len
    beams = {(): (0.0, -np.inf, 0.0)}  # prefix -> (pb, pnb, lm log prob of the prefix)
    for t in range(len(lp)):
        row = lp[t]; cand = np.nonzero(row > -12.0)[0]; nb = {}
        for pre, (pb, pnb, lmp) in beams.items():
            tot = lse(pb, pnb)
            for k in cand:
                p = float(row[k])
                if k == NPH:
                    b_, n_, l_ = nb.get(pre, (-np.inf, -np.inf, lmp)); nb[pre] = (lse(b_, tot + p), n_, lmp); continue
                last = pre[-1] if pre else None; new = pre + (int(k),)
                if new in nb: b_, n_, l_ = nb[new]
                else: b_, n_, l_ = -np.inf, -np.inf, lmp + (lm.lp(pre, int(k)) if lam else 0.0)
                nb[new] = (b_, lse(n_, (pb + p) if k == last else (tot + p)), l_)
                if k == last:
                    b2, n2, l2 = nb.get(pre, (-np.inf, -np.inf, lmp)); nb[pre] = (b2, lse(n2, pnb + p), l2)
        sc = lambda kv: lse(kv[1][0], kv[1][1]) + lam * kv[1][2] + beta * len(kv[0])
        beams = dict(sorted(nb.items(), key=lambda kv: -sc(kv))[:Wd])
    fin = lambda kv: lse(kv[1][0], kv[1][1]) + lam * (kv[1][2] + (lm.lp(kv[0], NPH) if lam else 0.0)) + beta * len(kv[0])
    return list(max(beams.items(), key=fin)[0])


def old_beam():
    # the original no-LM beam of phase0_beam.py, taken from its source (importing phase0_beam pulls struct2c, which refuses RIGHT in the env)
    import ast
    src = open(D + "/phase0_beam.py").read(); tree = ast.parse(src); ns = dict(np=np)
    for nd in tree.body:
        if isinstance(nd, ast.FunctionDef) and nd.name in ("lse", "beam"): exec(ast.get_source_segment(src, nd), ns)
    return ns["beam"]


G = {}
def _init(sents): G["sents"] = sents; G["lm"] = {}
def _job(a):
    n, lam, beta, lpl, yl = a
    if n not in G["lm"]: G["lm"][n] = KN(G["sents"], n)
    e = t = h = 0
    for lp_, y in zip(lpl, yl):
        hyp = beam(lp_, G["lm"][n], lam, beta); y = list(map(int, y)); e += B.edit(hyp, y); t += len(y); h += len(hyp)
    return e, t, h


def per(pool, sents, n, lam, beta, LP, Y):
    ch = [(n, lam, beta, LP[i::8], Y[i::8]) for i in range(8)]; r = pool.map(_job, ch); e, t, h = map(sum, zip(*r))
    return dict(PER=round(100 * e / t, 2), hyp_ref=round(h / t, 3))


if __name__ == "__main__":
    t0 = time.time(); sents = []
    for s_ in ("fit", "cal"): sents += [np.asarray(y) for y in B.load(s_)[1]]
    R = dict(text_sents=len(sents), text_phones=int(sum(len(y) for y in sents)), W=W, sel_n=SEL_N, models={}, lm_bytes={})
    for n in sorted(set(GN) | {2}):  # known answer 1: distributions sum to 1
        lm = KN(sents, n); R["lm_bytes"][n] = lm.nbytes(); rng = np.random.default_rng(n)
        for _ in range(20):
            h = list(rng.integers(0, NPH, size=rng.integers(0, 5))); s = sum(lm.p(([-1] * (n - 1) + h)[-(n - 1):] if n > 1 else [], w) for w in range(V))
            assert abs(s - 1) < 1e-9, (n, h, s)
    print("KN sums ok", R["lm_bytes"], flush=True)
    with Pool(2, initializer=_init, initargs=(sents,)) as pool:
        for tag in MODELS:
            LPS = {s_: lps(tag, s_) for s_ in SETS}; print(tag, "log-probs cached", round(time.time() - t0), flush=True)
            # known answer 2: lam = beta = 0 equals the no-LM beam (phase0_beam.beam) on 20 utterances
            PB = old_beam()
            for lp_ in LPS["libriA"][0][:20]: assert beam(lp_, None, 0.0, 0.0) == PB(lp_, NPH, W)
            rng = np.random.default_rng(3); sel = rng.choice(len(LPS["libriB"][0]), SEL_N, replace=False)
            SL, SY = [LPS["libriB"][0][i] for i in sel], [LPS["libriB"][1][i] for i in sel]; grid = []
            base = per(pool, sents, 2, 0.0, 0.0, SL, SY); grid.append(dict(n=0, lam=0, beta=0, **base))
            for n, lam, beta in itertools.product(GN, GL, GB):
                r = per(pool, sents, n, lam, beta, SL, SY); grid.append(dict(n=n, lam=lam, beta=beta, **r)); print(tag, grid[-1], flush=True)
            best = min(grid[1:], key=lambda g: g["PER"]); edge = best["lam"] in (GL[0], GL[-1]) or best["beta"] in (GB[0], GB[-1]) or best["n"] == GN[-1]
            out = dict(grid=grid, best=best, best_at_grid_edge=edge, sets={})
            for s_ in SETS[1:]:
                L_, Y_ = LPS[s_]; g = sum(B.edit(B.collapse(lp_.argmax(-1).tolist()), list(map(int, y))) for lp_, y in zip(L_, Y_)); tt = sum(len(y) for y in Y_)
                out["sets"][s_] = dict(greedy=round(100 * g / tt, 2), beam=per(pool, sents, 2, 0.0, 0.0, L_, Y_)["PER"],
                                       beam_lm=per(pool, sents, best["n"], best["lam"], best["beta"], L_, Y_)["PER"])
                print(tag, s_, out["sets"][s_], round(time.time() - t0), flush=True)
            h = {k: np.mean([out["sets"][s_][k] for s_ in ("libriA", "testother")]) for k in ("greedy", "beam", "beam_lm")}
            out["hard"] = {k: round(float(v), 2) for k, v in h.items()}; out["lm_gain_hard"] = round(float(h["beam"] - h["beam_lm"]), 2)
            out["rule"] = "PARK this n-gram" if out["lm_gain_hard"] < 0.5 else "LM gain >= 0.5: n-gram KEPT as the text-knowledge baseline"
            R["models"][tag] = out; json.dump(R, open(OUTJ, "w"), indent=1); print(tag, "HARD", out["hard"], out["rule"], flush=True)
    print("P2IV_DONE", round(time.time() - t0), flush=True)
