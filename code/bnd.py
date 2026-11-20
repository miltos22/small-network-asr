"""Signed group-pair boundary score (reviewer r7): per (group, next group) LDA log-likelihood ratio of 'true boundary frame' vs 'off by 1-4',
on the frame Legendre slopes/curvature. NIGHT_REPORT 8e."""
import glob, json
import numpy as np
import frontend as fe, phones, phones12, phones15, phones19 as c19

W0, W1 = fe._legendre(2, 3), fe._legendre(6, 4); DEL = np.arange(-4, 5)

def bvec(Tr, lens, frames):
    """93-dim boundary vector at global frame indices (each within its utterance)."""
    ends = np.cumsum(lens); st = ends - lens; out = np.zeros((len(frames), 93)); u_of = np.searchsorted(ends, frames, side="right")
    for u in np.unique(u_of):
        i = np.where(u_of == u)[0]; x = Tr[st[u]:ends[u]].astype(np.float64); f = frames[i] - st[u]
        a = np.stack([fe.legendre(x[:, k], W0) for k in range(x.shape[1])], 1); b = np.stack([fe.legendre(x[:, k], W1)[:, 1] for k in range(x.shape[1])], 1)
        out[i] = np.c_[a[f, :, 1], a[f, :, 2], b[f]]
    return out

def pairs(S):
    prev, nxt = c19.adjacency(S); j = np.where(nxt >= 0)[0]; g = c19.G_OF[S["segs"][:, 2]]
    return S["segs"][j, 1], g[j], g[nxt[j]]  # boundary frame, left group, right group

def fit(Tr, S, tau=200.0):
    b, gl, gr = pairs(S); ends = np.cumsum(S["lens"]); u = np.searchsorted(ends, b, side="right"); st = ends - S["lens"]
    rows, pos, pr = [], [], []
    for d in DEL:
        ok = (b + d >= st[u]) & (b + d < ends[u]); rows.append(b[ok] + d); pos.append(np.full(ok.sum(), d == 0)); pr.append(4 * gl[ok] + gr[ok])
    fr, pos, pr = np.concatenate(rows), np.concatenate(pos), np.concatenate(pr); X = bvec(Tr, S["lens"], fr)
    m, s = X.mean(0), X.std(0) + 1e-9; Z = (X - m) / s
    mp0, mn0 = Z[pos].mean(0), Z[~pos].mean(0); Sw = np.cov(np.r_[Z[pos] - mp0, Z[~pos] - mn0].T) + 1e-3 * np.eye(93); Si = np.linalg.inv(Sw)
    Wv, bv = np.zeros((16, 93)), np.zeros(16)
    for k in range(16):
        P, N = Z[pos & (pr == k)], Z[~pos & (pr == k)]
        mp = (P.sum(0) + tau * mp0) / (len(P) + tau); mn = (N.sum(0) + tau * mn0) / (len(N) + tau); w = Si @ (mp - mn); Wv[k], bv[k] = w, -0.5 * w @ (mp + mn)
    w0 = Si @ (mp0 - mn0); return dict(m=m, s=s, W=Wv, b=bv, w0=w0, b0=-0.5 * w0 @ (mp0 + mn0))

def score(B, X, pair=None):
    Z = (X - B["m"]) / B["s"]; return Z @ B["w0"] + B["b0"] if pair is None else np.einsum("nd,nd->n", Z, B["W"][pair]) + B["b"][pair]

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC = phones12.build("cal", fc, C["lens"]); spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0)
    TC = phones15.causal_norm(RC, fc, C, 200, pm_, pv_).astype(np.float32); del RC; Bd = fit(TC, C); del TC
    TD = phones15.causal_norm(phones12.build("dev_other", fd, D["lens"]), fd, D, 200, pm_, pv_).astype(np.float32)
    b, gl, gr = pairs(D); ends = np.cumsum(D["lens"]); u = np.searchsorted(ends, b, side="right"); st = ends - D["lens"]
    jit = b + np.random.default_rng(0).integers(-2, 3, len(b)); C5 = jit[:, None] + np.arange(-2, 3)[None]
    ok = (C5.min(1) >= st[u]) & (C5.max(1) < ends[u]); b, gl, gr, jit, C5 = b[ok], gl[ok], gr[ok], jit[ok], C5[ok]
    X = bvec(TD, D["lens"], C5.ravel()); pr = np.repeat(4 * gl + gr, 5); R = {"boundaries": int(len(b)), "no_snap": round(float(np.abs(jit - b).mean()), 3)}
    for k, sc in (("pooled", score(Bd, X)), ("per_pair", score(Bd, X, pr))):
        t = C5[np.arange(len(b)), sc.reshape(-1, 5).argmax(1)]; R[k] = round(float(np.abs(t - b).mean()), 3); R[k + "_exact"] = round(float((t == b).mean()), 3)
    R["no_snap_exact"] = round(float((jit == b).mean()), 3); print(json.dumps(R), flush=True)
    np.save(f"{phones.OUT}/bnd_model.npy", Bd); json.dump(R, open(f"{phones.OUT}/bnd.json", "w"), indent=1)
