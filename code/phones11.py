"""Vowel fix step 2: real formants. B1-B3 (fixed-band energy centroids) compress vowel space (F1 AA/IY 1.43x vs ~2.5x real).
LPC formant tracker (fixed maths, no data): pre-emphasis 0.97, Hann 25 ms, LPC order 18 (Levinson), roots -> resonances with
bandwidth < 500 Hz in 150-5000 Hz, F1-F3 = three lowest, 5-frame median. Tracks: log F1, log F2, log F3, log(F2/F1), each as
4 Legendre coefficients over +-80 ms (16 numbers/frame). Test: SEG108 vs SEG108+formants, both per-speaker normalised,
LDA fit on dev-clean cal, scored on dev-other (all / vowels / consonants)."""
import glob, os
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import lfilter
import frontend as fe
import phones, phones3, phones8

ORDER = 18

def lpc(fr):
    R = np.fft.irfft(np.abs(np.fft.rfft(fr, 1024)) ** 2)[:, :ORDER + 1]; R[:, 0] *= 1 + 1e-6; R[:, 0] += 1e-10
    a = np.zeros((len(fr), ORDER + 1)); a[:, 0] = 1; e = R[:, 0].copy()
    for i in range(1, ORDER + 1):
        k = -(a[:, :i] * R[:, i:0:-1]).sum(1) / e
        a[:, 1:i + 1] = a[:, 1:i + 1] + k[:, None] * a[:, i - 1::-1][:, :i]; e *= 1 - k ** 2
    return a

def formants(x, T):
    y = lfilter([1, -0.97], [1], x); a = lpc(fe.frames(y))
    Cm = np.zeros((len(a), ORDER, ORDER)); Cm[:, 0, :] = -a[:, 1:]; Cm[:, np.arange(1, ORDER), np.arange(ORDER - 1)] = 1
    z = np.linalg.eigvals(Cm); f = np.angle(z) * fe.SR / (2 * np.pi); bw = -np.log(np.abs(z) + 1e-12) * fe.SR / np.pi
    f = np.where((f > 150) & (f < 5000) & (bw < 500), f, np.inf); f = np.sort(f, 1)[:, :3]
    prior = np.array([500.0, 1500.0, 2500.0])
    for j in range(3):  # missing resonance -> hold the last value (start: neutral-vowel prior)
        col = f[:, j]; last = prior[j]
        for t in range(len(col)):
            if np.isfinite(col[t]): last = col[t]
            else: col[t] = last
    f = median_filter(f, size=(5, 1), mode="nearest")
    F = np.zeros((T, 3)); n = min(T, len(f)); F[:n] = f[:n]; F[n:] = f[n - 1] if n else prior
    L = np.log(F); tr = [L[:, 0], L[:, 1], L[:, 2], L[:, 1] - L[:, 0]]
    return np.concatenate([fe.legendre(u, fe.LEG4) for u in tr], 1).astype(np.float32)

def job(a):
    path, T = a; return formants(fe.load(path).astype(np.float64), T)

def build(name, files, lens):
    p = f"{phones.CACHE}/fmt_{name}.npy"
    if not os.path.exists(p):
        with Pool(4) as pool: np.save(p, np.concatenate(pool.map(job, list(zip(files, lens)), chunksize=8)))
    return np.load(p)

def spk_norm(X, files, D):
    spk = np.array([f.split("/")[-3] for f in files]); fs = np.repeat(spk, D["lens"]); X = X.astype(np.float64)
    sp = (D["ph"] >= 0) & (D["ph"] != phones.SILID)
    for s in np.unique(spk):
        m = fs == s; ref = X[m & sp]; X[m] = (X[m] - ref.mean(0)) / (ref.std(0) + 1e-9)
    return X

if __name__ == "__main__":
    VOW = np.array(sorted(phones8.VOWID))
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    FC, FD = build("cal", fc, C["lens"]), build("dev_other", fd, D["lens"])
    s = C["segs"]; mid = (s[:, 0] + s[:, 1]) // 2
    for p in "IY IH EH AE AA AO UH UW".split():  # sanity: Legendre coef 0 of log F1/log F2 (sign of the basis may be negative)
        m = s[:, 2] == phones.PID[p]; print(f"{p:3s} logF1c0={FC[mid[m], 0].mean():7.3f} logF2c0={FC[mid[m], 4].mean():7.3f}", flush=True)
    for name, (Xc, Xd) in {"SEG108 norm": (C["X"], D["X"]), "SEG108+formants norm": (np.c_[C["X"], FC], np.c_[D["X"], FD])}.items():
        Xc, Xd = spk_norm(Xc, fc, C), spk_norm(Xd, fd, D)
        c = phones3.lda_fit(phones3.segvec(Xc, C["segs"], len(Xc)), C["segs"][:, 2], phones.NP); F = phones3.segvec(Xd, D["segs"], len(Xd)); y = D["segs"][:, 2]
        pr = (((F - c["m"]) / c["s"]) @ c["W"].T + c["b"]).argmax(1); v = np.isin(y, VOW)
        print(f"{name:24s} all={np.mean(pr == y):.3f} vowels={np.mean(pr[v] == y[v]):.3f} consonants={np.mean(pr[~v] == y[~v]):.3f}", flush=True)
