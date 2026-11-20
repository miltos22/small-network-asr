"""Layer 0 front end (fixed maths, no learned weights). DESIGN.md v2 section "Layer 0".
16 kHz audio -> 100 frames/s of named physical tracks. Every track's job is in the comment beside it."""
import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import butter, sosfilt, resample_poly, lfilter

SR, WIN, HOP, NFFT, NMEL = 16000, 400, 160, 512, 40


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def dither(x, rng, db=-70.0):
    # equal noise floor for every source: ESC-50 has digital-zero padding, LibriSpeech never does (domain artefact otherwise)
    return x + rng.standard_normal(x.size).astype(np.float32) * 10 ** (db / 20)


def _melbank():
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    pts = 700 * (10 ** (np.linspace(hz2mel(60), hz2mel(7600), NMEL + 2) / 2595) - 1)
    bins = np.fft.rfftfreq(NFFT, 1 / SR)
    fb = np.zeros((NMEL, bins.size))
    for m in range(NMEL):
        lo, c, hi = pts[m:m + 3]
        fb[m] = np.clip(np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)), 0, None)
    return fb


MEL = _melbank()
HANN = np.hanning(WIN).astype(np.float32)
# periodicity: autocorrelation of the window itself, to undo the Hann taper's bias on long lags
_rw = np.fft.irfft(np.abs(np.fft.rfft(HANN, 2 * NFFT)) ** 2)[:NFFT]
RW = _rw / _rw[0]
LAG = slice(40, 267)  # 400 Hz .. 60 Hz pitch range
MOD_SOS = butter(2, [2.5, 8.0], btype="band", fs=100, output="sos")  # syllable-rate band of the energy envelope


def frames(x):
    n = 1 + max(0, (x.size - WIN) // HOP)
    idx = np.arange(WIN)[None] + HOP * np.arange(n)[:, None]
    return x[idx] * HANN


def tracks(x):
    """Return dict of per-frame tracks (T,) and logmel (T,40)."""
    f = frames(x)
    X = np.fft.rfft(f, NFFT)
    P = (X.real ** 2 + X.imag ** 2).astype(np.float64)
    mel = P @ MEL.T + 1e-12
    logmel = 10 * np.log10(mel)
    E = 10 * np.log10(P.sum(1) + 1e-12)  # total energy, dB

    # causal noise floor: follows dips fast, rises 1 dB/s, so steady noise sinks to ~0 relative energy
    nf = np.empty_like(E); cur = E[0]
    for t, e in enumerate(E):
        cur = 0.7 * cur + 0.3 * e if e < cur else cur + 0.01
        nf[t] = cur
    Erel = E - nf  # loudness above the background, level/gain independent

    p = mel / mel.sum(1, keepdims=True)
    H = -(p * np.log(p)).sum(1) / np.log(NMEL)  # spectral entropy: tone ~0, hiss ~1, vowel in between

    r = np.fft.irfft(np.abs(np.fft.rfft(f, 2 * NFFT)) ** 2)[:, :NFFT]
    r = r / (r[:, :1] + 1e-12) / RW
    R = np.clip(r[:, LAG].max(1), 0, 1.5)  # periodicity: glottal pulses in human pitch range

    c = dct(logmel - logmel.mean(1, keepdims=True), type=2, norm="ortho", axis=1)
    C1, C2 = c[:, 1], c[:, 2]  # spectral tilt and curvature (gain independent)

    # harmonic/formant peaks: local maxima standing >=6 dB over the 5-band neighbourhood minimum
    lm = logmel
    pad = np.pad(lm, ((0, 0), (2, 2)), mode="edge")
    nb_min = np.minimum.reduce([pad[:, k:k + NMEL] for k in range(5)])
    is_pk = (lm > pad[:, 1:NMEL + 1]) & (lm >= pad[:, 3:NMEL + 3]) & (lm - nb_min > 6)
    peaks = is_pk.sum(1).astype(np.float64)

    d = np.zeros_like(lm); d[2:] = lm[2:] - lm[:-2]
    flux = np.clip(d, 0, None).mean(1)  # onset strength (bursts, syllable onsets, clicks)

    m = sosfilt(MOD_SOS, Erel)
    mp = lfilter([0.05], [1, -0.95], m * m)
    mod = np.log10(mp + 1e-3)  # syllable-rhythm power (speech 3-8 Hz; tones, hum, fans have little)

    return dict(Erel=Erel, H=H, R=R, C1=C1, C2=C2, peaks=peaks, flux=flux, mod=mod, E=E, logmel=logmel)


def _legendre(half=8, order=3):
    tau = np.arange(-half, half + 1, dtype=np.float64)
    V = np.stack([tau ** k for k in range(order)], 1)
    Q, _ = np.linalg.qr(V)  # discrete orthonormal polynomials: level, slope, curvature
    return Q


LEG = _legendre()


def legendre(u, basis=LEG):
    """Project track u (T,) onto the basis over a centred 17-frame (+-80 ms) window -> (T, order)."""
    h = (basis.shape[0] - 1) // 2
    up = np.pad(u, h, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(up, basis.shape[0])
    return win @ basis


# ---------------- Module 2 channels (more information per frame; still fixed maths) ----------------
from scipy.ndimage import correlate

FREQ = np.fft.rfftfreq(NFFT, 1 / SR)
_band = lambda lo, hi: (FREQ >= lo) & (FREQ < hi)
BANDS = dict(B1=_band(200, 900), B2=_band(800, 2800), B3=_band(2200, 3800), BH=_band(2000, 8000))


def _gabors():
    """4 orientations on a 7-frame x 9-band log-mel patch: vertical edge (onset/offset across all bands: bursts, closures),
    horizontal bar (steady formant), rising and falling diagonals (formant transitions: place of stops/nasals)."""
    t = np.arange(-3, 4)[:, None]; b = np.arange(-4, 5)[None, :]
    env = np.exp(-(t / 2.0) ** 2 - (b / 3.0) ** 2)
    ks = {"g_vert": np.sign(t) * env, "g_horz": np.cos(np.pi * b / 4) * env,
          "g_rise": np.cos(np.pi * (b - t) / 4) * env, "g_fall": np.cos(np.pi * (b + t) / 4) * env}
    return {k: (v - v.mean()) / np.abs(v).sum() for k, v in ks.items()}


GAB = _gabors()
LEG4 = _legendre(order=4)  # level, slope, curvature, S-shape over +-80 ms


def tracks2(x):
    """Module 2 tracks: everything from tracks() plus band energies, pole-safe formant proxies, pitch, pitch-free shape, Gabors."""
    tr = tracks(x)
    f = frames(x)
    P = np.abs(np.fft.rfft(f, NFFT)) ** 2 + 1e-12
    E = P.sum(1)
    tr["Elow"] = 10 * np.log10(P[:, _band(60, 400)].sum(1) / E)   # voicing bar share (b/d/g closures, nasals, vowels)
    tr["Ehigh"] = 10 * np.log10(P[:, _band(3800, 7600)].sum(1) / E)  # frication / aspiration share
    tr["Emid"] = 10 * np.log10(P[:, _band(1000, 3000)].sum(1) / E)  # F2/F3 region share (vowels vs nasals)
    for k, m in BANDS.items():  # energy-weighted centroid + spread inside fixed bands: formant proxies that cannot pole-swap
        w = P[:, m]; fr = FREQ[m]; s = w.sum(1)
        c = (w * fr).sum(1) / s
        tr[k] = c / 1000.0
        tr["W" + k[1:]] = np.sqrt((w * (fr - c[:, None]) ** 2).sum(1) / s) / 1000.0
    r = np.fft.irfft(np.abs(np.fft.rfft(f, 2 * NFFT)) ** 2)[:, :NFFT]
    r = r / (r[:, :1] + 1e-12) / RW
    lag = np.argmax(r[:, LAG], 1) + LAG.start
    tr["F0"] = np.where(tr["R"] > 0.45, 12 * np.log2(SR / lag / 100.0), 0.0)  # semitones re 100 Hz, 0 when unvoiced
    c = dct(tr["logmel"] - tr["logmel"].mean(1, keepdims=True), type=2, norm="ortho", axis=1)
    for i in range(3, 7):
        tr[f"C{i}"] = c[:, i]  # finer spectral shape (vowel quality, r/l)
    lm = tr["logmel"]
    for k, K in GAB.items():
        resp = correlate(lm, K, mode="nearest")
        tr[k + "_lo"] = resp[:, 4:22].max(1)   # ~100-1700 Hz region
        tr[k + "_hi"] = resp[:, 18:36].max(1)  # ~1300-5500 Hz region
    return tr


# ---------------- noise-subtracted tracks (gate v4; reviewer's per-band noise floor, which v1-v3 lacked) ----------------
def noise_floor(L, up=0.01, down=0.3):
    """Causal per-bin noise floor on a dB matrix (T, B): follows dips quickly, rises 1 dB/s (stationary noise sinks into it)."""
    nf = np.empty_like(L); cur = L[0].copy()
    for t in range(L.shape[0]):
        x = L[t]; low = x < cur
        cur = np.where(low, (1 - down) * cur + down * x, cur + up); nf[t] = cur
    return nf


def tracks_ns(x):
    """Speech evidence measured AFTER removing the tracked noise floor in every FFT bin."""
    tr = tracks(x)
    f = frames(x)
    X = np.fft.rfft(f, NFFT); P = X.real ** 2 + X.imag ** 2 + 1e-12
    LdB = 10 * np.log10(P); N = 10 ** (noise_floor(LdB) / 10)
    Pc = np.maximum(P - 1.5 * N, 1e-3 * P)  # spectral subtraction with a small floor
    mel = Pc @ MEL.T + 1e-12; lm = 10 * np.log10(mel)
    snr_b = 10 * np.log10((P @ MEL.T + 1e-12) / (N @ MEL.T + 1e-12))  # per-band SNR above the floor
    p = mel / mel.sum(1, keepdims=True)
    o = dict(Erel=tr["Erel"], mod=tr["mod"])
    o["H"] = -(p * np.log(p)).sum(1) / np.log(NMEL)
    c = dct(lm - lm.mean(1, keepdims=True), type=2, norm="ortho", axis=1); o["C1"], o["C2"] = c[:, 1], c[:, 2]
    pad = np.pad(lm, ((0, 0), (2, 2)), mode="edge"); nb_min = np.minimum.reduce([pad[:, k:k + NMEL] for k in range(5)])
    o["peaks"] = ((lm > pad[:, 1:NMEL + 1]) & (lm >= pad[:, 3:NMEL + 3]) & (lm - nb_min > 6)).sum(1).astype(np.float64)
    d = np.zeros_like(lm); d[2:] = lm[2:] - lm[:-2]; o["flux"] = np.clip(d, 0, None).mean(1)
    r = np.fft.irfft(np.concatenate([Pc, np.zeros((Pc.shape[0], NFFT // 2))], 1))[:, :NFFT]  # autocorr of cleaned spectrum
    r = r / (r[:, :1] + 1e-12)
    o["R"] = np.clip(r[:, LAG].max(1), 0, 1.5)
    o["snr6"] = (snr_b > 6).mean(1)       # share of bands clearly above the noise
    o["snrtop"] = np.sort(snr_b, 1)[:, -8:].mean(1)  # strength of the best 8 bands (formants / harmonics poking out)
    o["snr_lo_hi"] = snr_b[:, 4:20].mean(1) - snr_b[:, 24:].mean(1)  # voiced speech lifts low-mid bands most
    o["E"] = tr["E"]
    return o
