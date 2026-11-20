"""D1 L0 views (D1_DESIGN.md v5 section 3.1): fixed maths, no weights. Written 12 Nov 2026 (the agent).
BANDS = 12 log-energy bands from the 80 power-mel bins of bb142 (same STFT: NFFT 400, HOP 160, Hann), edges 0 250 500 750 1000 1400 1800
2300 2800 3500 4500 6000 8000 Hz; a mel bin belongs to the band holding its centre frequency (every bin in exactly one band, asserted).
Per-utterance z-score per channel (as B0). TIME COURSE: each channel -> 10 numbers per frame: current value + Legendre coefficients
P0, P1, P2 over +-2 / +-6 / +-20 frames (+-20 / +-60 / +-200 ms), least-squares projections on the discrete window, edge-padded.
P1 is the odd (before-vs-after) term. selftest() checks known answers both ways (constant, ramp, parabola, band coverage)."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb142 as b

EDGES = [0, 250, 500, 750, 1000, 1400, 1800, 2300, 2800, 3500, 4500, 6000, 8000]
HALVES = (2, 6, 20)


def mel_centres(sr=16000, n=b.NMEL):
    hz = lambda m: 700 * (10 ** (m / 2595) - 1); mel = lambda f: 2595 * np.log10(1 + f / 700)
    return hz(np.linspace(mel(20), mel(sr / 2), n + 2))[1:-1]


def band_matrix():
    c = mel_centres(); M = np.zeros((len(EDGES) - 1, len(c)), np.float32)
    for j, f in enumerate(c):
        k = int(np.searchsorted(EDGES, f, side="right") - 1); M[min(k, len(EDGES) - 2), j] = 1
    assert (M.sum(0) == 1).all() and (M.sum(1) > 0).all(), "every mel bin in exactly one band, every band non-empty"
    return M


BM = torch.from_numpy(band_matrix())


def bands(path, raw=False):
    """(frames, 12) z-scored log band energies (raw=True: natural-log energies, not normalised)."""
    x = torch.from_numpy(b.fe.load(path)); S = torch.stft(x, b.NFFT, b.HOP, window=torch.hann_window(b.NFFT), return_complex=True).abs() ** 2
    e = torch.log(BM @ (b.FB @ S) + 1e-6).T.numpy()
    return e if raw else (e - e.mean(0)) / (e.std(0) + 1e-5)


def legendre_basis(h):
    """(3, 2h+1) least-squares projection rows for P0, P1, P2 on u = j/h, j = -h..h (discrete orthogonal fit)."""
    u = np.arange(-h, h + 1) / h; V = np.stack([np.ones_like(u), u, 1.5 * u ** 2 - 0.5], 1)  # P0, P1, P2
    return np.linalg.pinv(V)  # coefficients = pinv(V) @ window


def time_course(X):
    """X (T, C) -> (T, C, 10): current value + (P0, P1, P2) for each half-width in HALVES."""
    T, C = X.shape; out = [X[:, :, None]]
    for h in HALVES:
        Pm = legendre_basis(h); Xp = np.pad(X, ((h, h), (0, 0)), mode="edge")
        W = np.lib.stride_tricks.sliding_window_view(Xp, 2 * h + 1, axis=0)  # (T, C, 2h+1)
        out.append(np.einsum("tcw,kw->tck", W, Pm))
    return np.concatenate(out, 2).astype(np.float32)


def selftest():
    M = band_matrix(); print("bins per band", M.sum(1).astype(int).tolist())
    T = 100; t = np.arange(T, dtype=np.float64)
    const = time_course(np.full((T, 1), 3.0))[:, 0]; assert np.allclose(const[:, [1, 4, 7]], 3) and np.allclose(const[:, [2, 3, 5, 6, 8, 9]], 0, atol=1e-5), "constant"
    ramp = time_course((0.1 * t)[:, None])[30:70, 0]
    for i, h in enumerate(HALVES):
        assert np.allclose(ramp[:, 1 + 3 * i], 0.1 * t[30:70], atol=1e-4), ("ramp P0", h)
        assert np.allclose(ramp[:, 2 + 3 * i], 0.1 * h, atol=1e-4), ("ramp P1 = slope * h", h)
        assert np.allclose(ramp[:, 3 + 3 * i], 0, atol=1e-4), ("ramp P2", h)
    par = time_course(((t - 50) ** 2)[:, None])[50, 0]; h = 20
    assert abs(par[3 + 6] - h * h / 1.5) / (h * h / 1.5) < 0.05, ("parabola P2 ~ h^2 / 1.5", par[9])  # u^2 = (2 P2 + 1) / 3
    assert abs(par[2 + 6]) < 1e-3, "parabola centred: P1 = 0"
    rev = time_course((-(0.1 * t))[:, None])[30:70, 0]; assert np.allclose(rev[:, 2], -0.2, atol=1e-4), "falling ramp: P1 negative (both ways)"
    print("selftest ok: constant, ramp, falling ramp, parabola, band coverage")


if __name__ == "__main__":
    selftest()
    if len(sys.argv) > 1:
        Bd = bands(sys.argv[1]); V = time_course(Bd); print("bands", Bd.shape, "views", V.shape, "finite", np.isfinite(V).all())
