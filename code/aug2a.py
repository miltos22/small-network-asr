"""Speaker dials for arm 2a AND its comparator (ARM_2a_DESIGN_2026-11-14.md section 5), on raw log-mel (bb142 melbank, 80 bands, 16 kHz).
vtlp(x, alpha): frequency warp f -> alpha*f below F_HI = 4800*min(alpha,1)/alpha (Jaitly & Hinton 2013 piecewise form), linear above so Nyquist
maps to Nyquist; done as interpolation of mel-band POWER at warped centre frequencies (a band-level approximation, stated).
add_noise(x, n, snr_db): power-domain mix of a noise log-mel n at the given utterance-level SNR (mel-power sums, not waveform SNR; stated).
selftest(): known answers both ways."""
import numpy as np, torch
SR, NMEL = 16000, 80
_hz = lambda m: 700 * (10 ** (m / 2595) - 1); _mel = lambda f: 2595 * np.log10(1 + f / 700)
CEN = torch.tensor(_hz(np.linspace(_mel(20), _mel(SR / 2), NMEL + 2))[1:-1], dtype=torch.float32)


def warp_freq(f, alpha, fhi=4800.0, nyq=SR / 2):
    b = fhi * min(alpha, 1.0) / alpha
    return torch.where(f <= b, f * alpha, nyq - (nyq - b * alpha) / (nyq - b) * (nyq - f))


def vtlp(x, alpha):
    # x (T, 80) raw log-mel. Output band k takes the input power at the frequency that maps ONTO CEN[k], i.e. warp^-1(CEN[k]).
    if alpha == 1.0: return x
    c = CEN.to(x.device); src = torch.from_numpy(np.interp(c.cpu().numpy(), warp_freq(c, alpha).cpu().numpy(), c.cpu().numpy())).to(x.device).float()
    i = torch.searchsorted(c, src).clamp(1, NMEL - 1); lo, hi = c[i - 1], c[i]; w = ((src - lo) / (hi - lo)).clamp(0, 1)
    P = x.exp(); return torch.log((P[:, i - 1] * (1 - w) + P[:, i] * w).clamp_min(1e-6))


def add_noise(x, n, snr_db, rng):
    # n (Tn, 80) raw noise log-mel; tiled / cut to len(x) from a random offset.
    T = len(x); reps = int(np.ceil(T / len(n))) + 1; nn_ = n.repeat(reps, 1); o = int(rng.integers(0, len(nn_) - T + 1)); nn_ = nn_[o:o + T]
    Ps, Pn = x.exp(), nn_.exp(); g = Ps.sum() / (Pn.sum() * 10 ** (snr_db / 10))
    return torch.log(Ps + g * Pn)


def add_noise_events(x, NOISE, rng, kmax=4, snr=(0.0, 20.0), fade=10):
    # the author 15 Nov: "random background audio ... thats not stable per clip". 1..kmax noise EVENTS per utterance, each a random clip, a random
    # onset and length (0.5 s .. whole utterance), its own SNR that ramps linearly from one random value to another, 100 ms fades; events overlap
    # freely. SNR per frame = mean speech frame power / noise frame power (mel-power).
    T = len(x); Ps = x.exp(); ref = Ps.sum(1).mean(); out = Ps.clone()
    for _ in range(int(rng.integers(1, kmax + 1))):
        n = NOISE[int(rng.integers(len(NOISE)))].to(x.device).float(); L = int(rng.integers(min(50, T), T + 1)); o = int(rng.integers(0, T - L + 1))
        reps = int(np.ceil(L / len(n))) + 1; seg = n.repeat(reps, 1); q = int(rng.integers(0, len(seg) - L + 1)); Pn = seg[q:q + L].exp()
        sdb = torch.linspace(float(rng.uniform(*snr)), float(rng.uniform(*snr)), L, device=x.device)
        g = ref / (Pn.sum(1).mean() * 10 ** (sdb / 10)); f = torch.ones(L, device=x.device); k = min(fade, L // 2)
        if k > 0: f[:k] = torch.linspace(0, 1, k, device=x.device); f[L - k:] = torch.linspace(1, 0, k, device=x.device)
        out[o:o + L] += (g * f)[:, None] * Pn
    return torch.log(out)


def selftest():
    rng = np.random.default_rng(0); x = torch.randn(200, NMEL) - 5
    assert torch.equal(vtlp(x, 1.0), x)
    # a flat spectrum stays flat under any warp (interpolation of equal powers); 1+1e-6 on random spectra moved fp32 weights by ~1e-2 in log
    flat = torch.full((5, NMEL), -3.0)
    for al in (0.9, 1.1): assert (vtlp(flat, al) - flat).abs().max() < 1e-5  # no second floor: raw_logmel already floors power at 1e-6; a second +1e-6 shifted silent bands by +0.69
    # a pure tone band moves the right way: energy at band 30 with alpha 1.1 (longer tract scaled up) lands at a HIGHER band
    t = torch.full((1, NMEL), -10.0); t[0, 30] = 0.0
    up = vtlp(t, 1.1).argmax().item(); dn = vtlp(t, 0.9).argmax().item(); assert up > 30 > dn, (up, dn)
    n = torch.randn(80, NMEL) - 6
    for snr in (5.0, 20.0):
        z = add_noise(x, n, snr, rng); mix_noise = (z.exp() - x.exp()).sum(); got = 10 * np.log10(float(x.exp().sum() / mix_noise))
        assert abs(got - snr) < 0.05, (snr, got)
    assert (add_noise(x, n, 80.0, rng) - x).abs().max() < 1e-3
    # events: a single constant-SNR whole-utterance event reproduces the requested SNR (outside the fades); several events vary over time
    class R:  # rng stub forcing one event, full length, offset 0, SNR 10 -> 10
        def __init__(s): s.v = iter([1, 0, len(x), 0, 0]); s.u = iter([10.0, 10.0])  # count, clip, length, onset, noise offset
        def integers(s, a, b=None): return next(s.v)
        def uniform(s, a, b): return next(s.u)
    z = add_noise_events(x, [n], R(), kmax=1); mid = slice(20, len(x) - 20)
    got = 10 * np.log10(float(x.exp().sum(1).mean() / ((z.exp() - x.exp())[mid].sum(1).mean()))); assert abs(got - 10.0) < 0.3, got
    zs = [add_noise_events(x, [n, n * 0 - 4], rng) for _ in range(20)]; var = torch.stack([((q.exp() - x.exp()).sum(1)) for q in zs]).std(1).mean()
    assert float(var) > 0, var
    print("aug2a selftest ok: vtlp identity, tone moves up at 1.1 (band", up, ") down at 0.9 (band", dn, "), SNR exact at 5/20 dB")


if __name__ == "__main__":
    selftest()
