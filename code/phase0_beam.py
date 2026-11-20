"""Phase 0e + 0f (docs/PLAN.md section 2) on the cached log-probabilities of phase0_score.py. CPU. EXPLORATION.
0e: greedy vs CTC prefix beam search (beam 32, 64; no language model; classes with log-prob < -12 at a frame are skipped) -> PER, S / D / I.
    Known answer: beam 1 == greedy (asserted on the first 50 utterances of every cache).
0f: deletions by position from phase0_score.jsonl del_pos (output frame of a deleted reference phone, proxy = midpoint of its aligned neighbours):
    share of deletions in the first / last 25 output frames (0.5 s) next to the share of frames there, per model x set.
Usage: python3 phase0_beam.py"""
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D + "/logs_2c/run2")
import numpy as np
import base5_next as B, phase0_score as P

os.nice(19); LP = B.CACHE + "/phase0_lp"; SETS = ["cleanS", "libriA", "libriB", "testother", "author"]
MODELS = [("base232_s0", "base"), ("full232SKIP_s0", "full"), ("base100_s0", "base")]


def lse(a, b): return a if b == -np.inf else b if a == -np.inf else max(a, b) + np.log1p(np.exp(-abs(a - b)))


def beam(lp, blank, W):  # standard CTC prefix beam search; returns the best label prefix (class ids)
    beams = {(): (0.0, -np.inf)}  # prefix -> (log p ending in blank, log p ending in non-blank)
    for t in range(len(lp)):
        row = lp[t]; cand = np.nonzero(row > -12.0)[0]; nb = {}
        for pre, (pb, pnb) in beams.items():
            tot = lse(pb, pnb)
            for k in cand:
                p = float(row[k])
                if k == blank:
                    b_, n_ = nb.get(pre, (-np.inf, -np.inf)); nb[pre] = (lse(b_, tot + p), n_)
                    continue
                last = pre[-1] if pre else None; new = pre + (int(k),)
                b_, n_ = nb.get(new, (-np.inf, -np.inf))
                nb[new] = (b_, lse(n_, (pb + p) if k == last else (tot + p)))
                if k == last:  # repeated symbol without blank stays on the same prefix
                    b2, n2 = nb.get(pre, (-np.inf, -np.inf)); nb[pre] = (b2, lse(n2, pnb + p))
        beams = dict(sorted(nb.items(), key=lambda kv: -lse(*kv[1]))[:W])
    return list(max(beams.items(), key=lambda kv: lse(*kv[1]))[0])


def to39(arm, ids):
    if arm == "base": return [i for i in ids if i != B.NPH]
    out = []
    for t in ids:
        if t not in (P.S.BLANKV, P.S.L2.SIL): out += P.S.L2.TO39[t]
    return out


def counts(H, Y):
    s = d = i = n = 0
    for h, y in zip(H, Y):
        a, b_, c, _, _ = P.ops(h, list(map(int, y))); s += a; d += b_; i += c; n += len(y)
    return dict(PER=round(100 * (s + d + i) / n, 2), sub=round(100 * s / n, 2), dele=round(100 * d / n, 2), ins=round(100 * i / n, 2))


def selftest():  # known answer: on tiny random inputs a beam wide enough to keep every prefix finds the brute-force best label sequence
    import itertools
    rng = np.random.default_rng(0)
    for _ in range(200):
        T, V = int(rng.integers(2, 6)), 3; lp = np.log(rng.dirichlet(np.ones(V), T)); tot = {}
        for path in itertools.product(range(V), repeat=T):
            lab = tuple(k for i, k in enumerate(path) if k != 0 and (i == 0 or k != path[i - 1]))
            tot[lab] = np.logaddexp(tot.get(lab, -np.inf), sum(lp[t, k] for t, k in enumerate(path)))
        best = max(tot.items(), key=lambda kv: kv[1])
        got = tuple(beam(lp, 0, 10 ** 6))
        assert np.isclose(tot[got], best[1]), (got, best)
    print("selftest ok: beam = brute force on 200 tiny cases", flush=True)


if __name__ == "__main__":
    selftest(); out = {"beam": [], "positions": []}
    for tag, arm in MODELS:
        blank = B.NPH if arm == "base" else P.S.BLANKV
        for sname in SETS:
            f = f"{LP}/{tag}_{sname}.npz"
            if not os.path.exists(f): print("missing", f); continue
            z = np.load(f); st = np.r_[0, np.cumsum(z["lens"])]; L = [z["lp"][a:c].astype(np.float32) for a, c in zip(st[:-1], st[1:])]
            Y = B.load(sname)[1]; assert len(Y) == len(L)
            ix = sorted(np.random.default_rng(2).choice(len(L), min(100, len(L)), replace=False).tolist())  # cost: ~2.4 s / utterance at beam 32
            L = [L[i] for i in ix]; Y = [Y[i] for i in ix]
            G = [P.greedy(arm, l)[0] for l in L]
            r = dict(model=tag, set=sname, utts=len(L), greedy=counts(G, Y))
            for W in ((32, 64) if tag == "base232_s0" and sname in ("cleanS", "libriB") else (32,)): r[f"beam{W}"] = counts([to39(arm, beam(l, blank, W)) for l in L], Y)
            out["beam"].append(r); print(json.dumps(r), flush=True)
    for line in open(D + "/logs_base5/phase0_score.jsonl"):
        j = json.loads(line)
        if not j["model"].endswith("_s0"): continue
        dp = np.array(j["del_pos"], float)
        if len(dp) == 0: continue
        T = dp[:, 1]; a = dp[:, 0]; edge = (a < 25) | (a >= T - 25)
        per_utt_T = None  # frame share at the edges: from the same utterances' lengths is not stored per utt; use the deletion rows' T
        fr_share = float(np.mean(np.minimum(50, T) / T))
        out["positions"].append(dict(model=j["model"], set=j["set"], deletions=len(dp), share_first_last_0p5s=round(float(edge.mean()), 3),
                                     frame_share_of_edges_approx=round(fr_share, 3)))
    for r in out["positions"]: print(json.dumps(r))
    json.dump(out, open(D + "/logs_base5/phase0_beam_2026-11-17.json", "w"), indent=1)
