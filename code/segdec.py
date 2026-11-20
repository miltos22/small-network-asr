"""Segmental chain decoding: each boundary may move by OFF frames; state at a boundary = (offset, prev group, phone). Emission tensor
E[j, d_start, d_end, g_prev, phone, g_next] (prior-free, already a background ratio), phone bigram, acoustic scale kappa. Full forward-backward."""
import numpy as np
from scipy.special import logsumexp
import phones19 as c19

OFF = np.array([-2, -1, 0, 1, 2]); NEG = -1e30

def forward_backward(E, st, en, logB, lstart, G, kappa, ldur=None):
    """E (n, O, O, 5, K, 5) with -inf for invalid (d_start, d_end); returns phone marginals (n, K) and end-offset marginals (n, O)."""
    n, O, _, _, K, _ = E.shape; S = kappa * E if ldur is None else kappa * E + ldur
    la = np.full((n, O, 5, K), NEG); la[st, :, 4, :] = lstart[None, :]  # start boundary offset free (uniform), prev group = edge
    maxlen = int((en - st).max()) + 1
    def fwd_tilde(a, s):  # a (B, O, 5, K), s (B, O, O, 5, K, 5) -> (B, O_end, K, 5)
        return logsumexp(a[:, :, None, :, :, None] + s, axis=(1, 3))
    def advance(t_, logB):  # t_ (B, O, K, 5) -> (B, O, 5, K): next boundary state (offset, group of this phone, next phone)
        T = t_[:, :, :, G] + logB[None, None]; out = np.full((len(t_), O, 5, K), NEG)
        for g in range(4): out[:, :, g, :] = logsumexp(T[:, :, G == g, :], axis=2)
        return out
    for t in range(maxlen - 1):
        j = st[(en - st) > t] + t; la[j + 1] = advance(fwd_tilde(la[j], S[j]), logB)
    lb = np.full((n, O, 5, K), NEG)
    lb[en] = logsumexp(S[en][:, :, :, :, :, 4], axis=2)  # run end: next group = edge, end offset free
    for t in range(maxlen - 1, 0, -1):
        j = st[(en - st) >= t] + t; nb = lb[j]  # (B, O, 5, K) at boundary before segment j: (offset, g(phone_{j-1}), phone_j)
        tb = np.full((len(j), O, K, 5), NEG)  # (offset_end of j-1, phone_{j-1}, g_next)
        for g in range(4):
            sel = G == g; tb[:, :, :, g] = logsumexp(logB[None, None][:, :, :, sel] + nb[:, :, G, :][:, :, :, sel], axis=3)
        lb[j - 1] = logsumexp(S[j - 1] + tb[:, None, :, None, :, :], axis=(2, 5))
    post = logsumexp(la + lb, axis=(1, 2)); post -= logsumexp(post, 1, keepdims=True)
    # end-offset marginal of segment j: sum over states of la (start) x S x right message
    return np.exp(post)

def brute(E, logB, lstart, G, kappa):
    import itertools
    n, O, _, _, K, _ = E.shape; P = np.zeros((n, K)); tot = []
    for ph in itertools.product(range(K), repeat=n):
        for offs in itertools.product(range(O), repeat=n + 1):
            s = lstart[ph[0]] + sum(logB[ph[i], ph[i + 1]] for i in range(n - 1))
            for i in range(n):
                gp = 4 if i == 0 else G[ph[i - 1]]; gn = 4 if i == n - 1 else G[ph[i + 1]]; s += kappa * E[i, offs[i], offs[i + 1], gp, ph[i], gn]
            tot.append((s, ph))
    lp = np.array([s for s, _ in tot]); w = np.exp(lp - logsumexp(lp))
    for (s, ph), wi in zip(tot, w):
        for i in range(n): P[i, ph[i]] += wi
    return P
