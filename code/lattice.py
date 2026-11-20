"""Segmental lattice Viterbi: nodes 0..M (candidate boundaries), arcs (i, k); our segment model picks segmentation and phones.
State at node = (group of previous phone, phone starting here); the next group of a segment is delayed as in segdec. NIGHT_REPORT 8h."""
import itertools
import numpy as np

NEG = -1e30

def viterbi(M, arcs, EA, BN, logB, lst, G, psi, init=None):
    """arcs (A, 2) node pairs; EA (A, 5, K, 5) arc scores [g_prev, phone, g_next]; BN (M+1, K, 5) end-node score (0 at M). Returns phones, score."""
    K = len(lst); d = np.full((M + 1, 5, K), NEG); d[0, 4] = lst + psi
    if init is not None: d[0] = init  # (5, K) entry scores at node 0 (restart after a commit)
    bd = np.zeros((M + 1, 5, K), np.int64)
    v = np.full((M + 1, K, 5), NEG); ba = np.zeros((M + 1, K, 5), np.int64); bg = np.zeros((M + 1, K, 5), np.int64)
    by_end = {}
    for a, (i, k) in enumerate(arcs): by_end.setdefault(int(k), []).append(a)
    for k in range(1, M + 1):
        if k not in by_end: continue
        idx = np.array(by_end[k]); c = d[arcs[idx, 0]][:, :, :, None] + EA[idx]  # (n, gp, X, gn)
        gp = c.argmax(1); cm = np.take_along_axis(c, gp[:, None], 1)[:, 0]; best = cm.argmax(0)  # (X, gn)
        v[k] = np.take_along_axis(cm, best[None], 0)[0] + BN[k]; ba[k] = idx[best]; bg[k] = np.take_along_axis(gp, best[None], 0)[0]
        if k < M:
            t = v[k][:, G] + logB  # (X, Y): v[k, X, G(Y)] + logB[X, Y]
            for g in range(4):
                sel = np.where(G == g)[0]
                if len(sel): j = t[sel].argmax(0); d[k, g] = t[sel][j, np.arange(K)] + psi; bd[k, g] = sel[j]
    X = int(v[M, :, 4].argmax()); score = float(v[M, X, 4]); gn, k, out, path = 4, M, [], []
    while True:
        a = ba[k, X, gn]; g = bg[k, X, gn]; out.append(X); i = int(arcs[a, 0])
        path.append(k)
        if i == 0: path.append(0); break
        W = int(bd[i, g, X]); gn, k, X = G[X], i, W
    viterbi.nodes = path[::-1]  # node path of the best segmentation (last call)
    viterbi.tables = (v, ba, bg, bd)
    return out[::-1], score

def trace(arcs, G, ba, bg, bd, k, X, gn):
    """Arcs [(i, k, phone)] of the best partial path ending at node k in state (X, gn)."""
    out = []
    while True:
        a = ba[k, X, gn]; g = bg[k, X, gn]; i = int(arcs[a, 0]); out.append((i, k, X))
        if i == 0: return out[::-1]
        W = int(bd[i, g, X]); gn, k, X = G[X], i, W

def viterbi_lag(M, arcs, EA, BN, logB, lst, G, psi, tn, L):
    """Streaming viterbi with a fixed lag L (frames, tn = node times): the first arc of the best partial path is committed once the
    frontier node is >= L frames past that arc's end; the search then restarts from the committed node in the committed state (phone,
    next group), so committed output never changes. No commit possible before the region end = flush the best full path. L None = viterbi."""
    ph, _ = viterbi(M, arcs, EA, BN, logB, lst, G, psi)
    if L is None: return ph, [(tn[i], tn[k]) for i, k in zip(viterbi.nodes[:-1], viterbi.nodes[1:])]
    K = len(lst); c, init, out, spans = 0, None, [], []
    while True:
        m = arcs[:, 0] >= c; sub = arcs[m] - c; Ms = M - c; php, _ = viterbi(Ms, sub, EA[m], BN[c:], logB, lst, G, psi, init)
        nf = [x + c for x in viterbi.nodes]; v, ba, bg, bd = viterbi.tables; ok = v.reshape(Ms + 1, -1).max(1) > NEG / 2; hit = None
        for j in range(int(np.searchsorted(tn[c:], tn[c] + L + 1)), Ms):
            if not ok[j]: continue
            X, gn = np.unravel_index(int(v[j].argmax()), v[j].shape); P = trace(sub, G, ba, bg, bd, j, int(X), int(gn))
            if tn[c + P[0][1]] <= tn[c + j] - L: hit = P; break
        if hit is None: return out + php, spans + [(tn[i], tn[k]) for i, k in zip(nf[:-1], nf[1:])]
        _, k0, X0 = hit[0]; gn0 = G[hit[1][2]]; out.append(int(X0)); spans.append((tn[c], tn[c + k0]))
        init = np.full((5, K), NEG); Y = np.where(G == gn0)[0]; init[G[X0], Y] = logB[X0, Y] + psi; c += k0

def brute(M, arcs, EA, BN, logB, lst, G, psi):
    aid = {(int(i), int(k)): a for a, (i, k) in enumerate(arcs)}; K = len(lst); best = (NEG, None)
    for r in range(M):
        for mid in itertools.combinations(range(1, M), r):
            nodes = (0,) + mid + (M,)
            if any((nodes[j], nodes[j + 1]) not in aid for j in range(len(nodes) - 1)): continue
            for ph in itertools.product(range(K), repeat=len(nodes) - 1):
                s = lst[ph[0]] + psi * len(ph)
                for j, X in enumerate(ph):
                    gp = 4 if j == 0 else G[ph[j - 1]]; gn = 4 if j == len(ph) - 1 else G[ph[j + 1]]; s += EA[aid[(nodes[j], nodes[j + 1])], gp, X, gn] + BN[nodes[j + 1], X, gn]
                    if j: s += logB[ph[j - 1], X]
                if s > best[0]: best = (s, list(ph))
    return best[1], best[0]

def _check():
    rng = np.random.default_rng(0)
    for trial in range(30):
        M, K = int(rng.integers(1, 6)), 3; G = np.array([0, 1, 1]) if trial % 2 else np.array([0, 1, 2])
        arcs = np.array([(i, k) for i in range(M) for k in range(i + 1, min(M, i + 3) + 1)]); EA = rng.normal(size=(len(arcs), 5, K, 5))
        BN = rng.normal(size=(M + 1, K, 5)); BN[M] = 0; logB = rng.normal(size=(K, K)); lst = rng.normal(size=K); psi = float(rng.normal())
        a, b = viterbi(M, arcs, EA, BN, logB, lst, G, psi), brute(M, arcs, EA, BN, logB, lst, G, psi)
        assert a[0] == b[0] and abs(a[1] - b[1]) < 1e-9, (trial, a, b)
        tn = np.r_[0, np.cumsum(rng.integers(1, 5, M))]; n_ = viterbi.nodes
        assert viterbi_lag(M, arcs, EA, BN, logB, lst, G, psi, tn, 10 ** 6) == (a[0], [(tn[i], tn[k]) for i, k in zip(n_[:-1], n_[1:])]), trial  # huge lag = exact
        ph2, sp2 = viterbi_lag(M, arcs, EA, BN, logB, lst, G, psi, tn, 2); ts = {(tn[i], tn[k]) for i, k in arcs}
        assert len(ph2) == len(sp2) and sp2[0][0] == 0 and sp2[-1][1] == tn[M] and all(x[1] == y[0] for x, y in zip(sp2, sp2[1:])) and all(x in ts for x in sp2), trial
_check()
