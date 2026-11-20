"""5a: exact chain decoding over adjacent segments: states (prev group, phone), prior-free context likelihoods x phone bigram.
Full forward-backward and fixed lag 2. NIGHT_REPORT 5a."""
import glob, json
import numpy as np
from scipy.special import logsumexp
import phones, phones12, phones15, phones16, phones19 as c19, phones24 as h, phones25 as b, phones28 as m

K = phones.NP; G = c19.G_OF; EDGE = 4; NINF = -1e30

def emis(M, pc, UD):
    """log l_j(gp, k, gn) prior-free (shared constant dropped): (n, 5, K, 5)."""
    Z = (UD - M["m"]) / M["s"]; out = np.empty((len(UD), 5, K, 5), np.float32)
    for c in range(25): out[:, c // 5, :, c % 5] = Z @ M["W"][c].T + M["b"][c] - np.log(pc[c])
    return out

def runs(S):
    prev, nxt = c19.adjacency(S); st = np.where(prev < 0)[0]; en = np.where(nxt < 0)[0]; return st, en  # runs are consecutive indices

def step_fwd(la, l, logB, kappa):
    """la (B,5,K) at j, l (B,5,K,5) -> la at j+1 (B,5,K)."""
    lat = logsumexp(la[..., None] + kappa * l, axis=1)  # (B,K,5): (y_j, g_{j+1})
    T = lat[:, :, G] + logB[None]  # (B, y_j, y_{j+1}) using g_{j+1} = G(y_{j+1})
    out = np.full((len(la), 5, K), NINF)
    for g in range(4): out[:, g] = logsumexp(T[:, G == g, :], axis=1)
    return out

def step_bwd(lb_next, l, logB, kappa):
    """lb_next (B,5,K) at j+1 (index gp = G(y_j)) and l at j -> lb at j (B,5,K)."""
    lbt = np.full((len(l), K, 5), NINF)  # (y_j, g_{j+1})
    for g in range(4):
        sel = G == g; lbt[:, :, g] = logsumexp(logB[None][:, :, sel] + lb_next[:, G, :][:, :, sel], axis=2)  # lb_next[b, G(y_j), y']
    return logsumexp(kappa * l + lbt[:, None, :, :], axis=3)

def decode(L, st, en, logB, lstart, kappa, lag=None):
    n = len(L); la = np.full((n, 5, K), NINF); run_end = np.empty(n, int)
    for s, e in zip(st, en): run_end[s:e + 1] = e
    la[st, EDGE] = lstart; maxlen = int((en - st).max()) + 1
    for t in range(maxlen - 1):
        j = st[(en - st) > t] + t; la[j + 1] = step_fwd(la[j], L[j], logB, kappa)
    endl = lambda j: logsumexp(kappa * L[j][:, :, :, EDGE], axis=1)  # emission with g_next = edge (run end)
    if lag is None:
        lb = np.full((n, 5, K), NINF); lb[en] = kappa * L[en][:, :, :, EDGE]
        for t in range(maxlen - 1, 0, -1):
            j = st[(en - st) >= t] + t; lb[j - 1] = step_bwd(lb[j], L[j - 1], logB, kappa)
    else:
        lb = np.empty((n, 5, K))
        for j0 in range(0, n, 20000):  # each segment: backward over at most `lag` future segments, beta = 1 beyond
            J = np.arange(j0, min(n, j0 + 20000)); last = np.minimum(J + lag, run_end[J])
            cur = np.where((last == run_end[J])[:, None, None], kappa * L[last][:, :, :, EDGE], logsumexp(kappa * L[last] + 0.0, axis=3))
            for d in range(lag, 0, -1):
                jj = J + d - 1; act = jj < last; 
                if act.any(): cur[act] = step_bwd(cur[act], L[jj[act]], logB, kappa)
            lb[J] = cur
    g = logsumexp(la + lb, axis=1); g -= logsumexp(g, axis=1, keepdims=True); return np.exp(g)

def brute(L, logB, lstart, kappa):
    import itertools
    n = len(L); lp = []; paths = list(itertools.product(range(K), repeat=n))
    for p in paths:
        s = lstart[p[0]] + sum(logB[p[i], p[i + 1]] for i in range(n - 1))
        for i in range(n): s += kappa * L[i, EDGE if i == 0 else G[p[i - 1]], p[i], EDGE if i == n - 1 else G[p[i + 1]]]
        lp.append(s)
    lp = np.array(lp); w = np.exp(lp - logsumexp(lp)); P = np.zeros((n, K))
    for p, wi in zip(paths, w):
        for i in range(n): P[i, p[i]] += wi
    return P

def bigram(yc, S, alpha=10.0):
    prev, nxt = c19.adjacency(S); pk = np.bincount(yc, minlength=K) / len(yc); cnt = np.zeros((K, K)); np.add.at(cnt, (yc[prev[prev >= 0]], yc[prev >= 0]), 1)
    B = (cnt + alpha * pk[None]) / (cnt.sum(1, keepdims=True) + alpha); s0 = np.bincount(yc[prev < 0], minlength=K) + alpha * pk
    return np.log(B), np.log(s0 / s0.sum())

def per_speaker(P0, P1, y, spk_of, sel):
    sp = np.unique(spk_of[sel]); d = np.array([np.mean(P1[sel & (spk_of == s)].argmax(1) == y[sel & (spk_of == s)]) - np.mean(P0[sel & (spk_of == s)].argmax(1) == y[sel & (spk_of == s)]) for s in sp])
    bs = np.random.default_rng(0).choice(d, (2000, len(d))).mean(1); return dict(improved=int((d > 0).sum()), speakers=len(d), mean=round(float(d.mean()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC, RD = phones12.build("cal", fc, C["lens"]), phones12.build("dev_other", fd, D["lens"]); yc, y = C["segs"][:, 2], D["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    XC = np.c_[phones15.feats(phones15.causal_norm(RC, fc, C, 200, pm_, pv_), C), phones16.rate(C, fc, 5, pl)]
    XD = np.c_[phones15.feats(phones15.causal_norm(RD, fd, D, 200, pm_, pv_), D), phones16.rate(D, fd, 5, pl)]; del RC, RD
    spk = np.array([f.split("/")[-3] for f in fd]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt]); spk_of = spk[utt]
    cc = c19.ctx_true(C); Mfull = c19.fit(XC, yc, cc, 200); V = b.projection(XC, yc, cc, Mfull, 76)
    UC, UD = ((XC - Mfull["m"]) / Mfull["s"]) @ V, ((XD - Mfull["m"]) / Mfull["s"]) @ V; del XC, XD
    B0 = c19.fit(UC, yc, np.zeros(len(yc), int), 1.0); M = c19.fit(UC, yc, cc, 200); P = c19.post(B0, UD, np.zeros(len(y), int))
    for _ in (1, 2, 3): P = c19.post_mix(M, UD, c19.ctx_soft(D, P))
    Pmf = P; Phmf = h.hybrid(UC, yc, cc, UD, c19.ctx_soft(D, P), P, M, 1.0)
    R = {"meanfield_flat": dict(A=c19.acc(Pmf, y, A), B=c19.acc(Pmf, y, ~A)), "meanfield_hybrid": dict(A=c19.acc(Phmf, y, A), B=c19.acc(Phmf, y, ~A))}; print(R, flush=True)
    pc = m.ctx_priors(yc, cc); L = emis(M, pc, UD); logB, lst = bigram(yc, C); st, en = runs(D)
    print(f"runs: {len(st)}, mean length {np.mean(en - st + 1):.1f}, max {int((en - st).max()) + 1}", flush=True)
    for kappa in (0.3, 0.5, 1.0):
        for lag in (None, 2):
            Pc = decode(L, st, en, logB, lst, kappa, lag); k = f"k{kappa}_{'full' if lag is None else 'lag2'}"
            R[k] = dict(A=c19.acc(Pc, y, A), B=c19.acc(Pc, y, ~A)); print(k, R[k], flush=True)
            if lag == 2: R[k]["P"] = Pc
    best = max((k for k in R if k.endswith("lag2")), key=lambda k: R[k]["A"][1]); Pb = R[best].pop("P")
    for k in list(R):
        R[k].pop("P", None)
    Phc = h.hybrid(UC, yc, cc, UD, c19.ctx_soft(D, Pb), Pb, M, 1.0); R["chain_hybrid"] = dict(A=c19.acc(Phc, y, A), B=c19.acc(Phc, y, ~A), kappa_lag2=best)
    R["per_speaker_B_chain_vs_meanfield_flat"] = per_speaker(Pmf, Pb, y, spk_of, ~A); R["per_speaker_B_chain_hybrid_vs_meanfield_hybrid"] = per_speaker(Phmf, Phc, y, spk_of, ~A)
    print("chosen on half A:", best, R[best], "chain_hybrid", R["chain_hybrid"], R["per_speaker_B_chain_vs_meanfield_flat"], R["per_speaker_B_chain_hybrid_vs_meanfield_hybrid"], flush=True)
    b0, bb = R["meanfield_hybrid"]["B"], R["chain_hybrid"]["B"]; ps = R["per_speaker_B_chain_hybrid_vs_meanfield_hybrid"]
    keep = ((bb[0] - b0[0] >= 0.002 and bb[1] >= b0[1]) or (bb[1] - b0[1] >= 0.002 and bb[0] >= b0[0])) and ps["improved"] * 3 >= 2 * ps["speakers"]
    print("KEEP" if keep else "DROP", flush=True); json.dump(dict(R=R, keep=keep), open(f"{phones.OUT}/phones33.json", "w"), indent=1, default=str)
