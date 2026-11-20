"""Window fix (root cause confirmed twice: stops read half-voiced; vowel formants compressed ~1.4x instead of ~2.5x): every track
was averaged over +-80 ms (170 ms) while short vowels/stops last 30-80 ms. Same 31 tracks (27 Module-2 tracks + log F1-F3 +
log F2/F1), per-speaker normalised, three windows:
  W80  : 4 Legendre coefs over +-80 ms (current)      W20 : 3 coefs over +-20 ms
  DUAL : 3 coefs over +-20 ms + 4 coefs over +-60 ms   (manner/laryngeal events + transitions)
Segment vector (phones3.segvec), LDA fit on dev-clean cal, scored on dev-other: all / vowels / consonants."""
import glob, os
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import lfilter
import frontend as fe
import af, phones, phones3, phones8, phones11

NAMES = af.ALL_TRACKS + ["lF1", "lF2", "lF3", "lF21"]

def raw_formants(x, T):
    y = lfilter([1, -0.97], [1], x); a = phones11.lpc(fe.frames(y)); O = phones11.ORDER
    Cm = np.zeros((len(a), O, O)); Cm[:, 0, :] = -a[:, 1:]; Cm[:, np.arange(1, O), np.arange(O - 1)] = 1
    z = np.linalg.eigvals(Cm); f = np.angle(z) * fe.SR / (2 * np.pi); bw = -np.log(np.abs(z) + 1e-12) * fe.SR / np.pi
    f = np.sort(np.where((f > 150) & (f < 5000) & (bw < 500), f, np.inf), 1)[:, :3]
    for j, last in enumerate((500.0, 1500.0, 2500.0)):
        col = f[:, j]
        for t in range(len(col)):
            if np.isfinite(col[t]): last = col[t]
            else: col[t] = last
    f = median_filter(f, size=(5, 1), mode="nearest"); F = np.empty((T, 3)); n = min(T, len(f)); F[:n] = f[:n]; F[n:] = f[n - 1]
    L = np.log(F); return np.c_[L, L[:, 1] - L[:, 0]]

def job(a):
    path, T = a; x = fe.load(path).astype(np.float64); tr = fe.tracks2(x.astype(np.float32))
    cols = []
    for t in af.ALL_TRACKS:
        u = np.asarray(tr[t], np.float64); v = np.empty(T); n = min(T, len(u)); v[:n] = u[:n]; v[n:] = u[n - 1]; cols.append(v)
    return np.c_[np.stack(cols, 1), raw_formants(x, T)].astype(np.float32)

def build(name, files, lens):
    p = f"{phones.CACHE}/trk31_{name}.npy"
    if not os.path.exists(p):
        with Pool(4) as pool: np.save(p, np.concatenate(pool.map(job, list(zip(files, lens)), chunksize=8)))
    return np.load(p)

def project(Tr, lens, windows):
    out, o = [], 0
    for n in lens:
        u = Tr[o:o + n].astype(np.float64); o += n
        out.append(np.concatenate([fe.legendre(u[:, i], B) for B in windows for i in range(u.shape[1])], 1))
    return np.concatenate(out)

if __name__ == "__main__":
    VOW = np.array(sorted(phones8.VOWID))
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    TC, TD = build("cal", fc, C["lens"]), build("dev_other", fd, D["lens"])
    s = C["segs"]; mid = (s[:, 0] + s[:, 1]) // 2
    for p in "IY IH EH AE AA AO UH UW".split():  # raw mid-frame formants in Hz (no window): is the tracker itself right?
        m = s[:, 2] == phones.PID[p]; print(f"{p:3s} F1={np.exp(np.median(TC[mid[m], 27])):6.0f} F2={np.exp(np.median(TC[mid[m], 28])):6.0f}", flush=True)
    variants = {"W80": [fe._legendre(8, 4)], "W20": [fe._legendre(2, 3)], "DUAL": [fe._legendre(2, 3), fe._legendre(6, 4)]}
    y = D["segs"][:, 2]; v = np.isin(y, VOW)
    for name, W in variants.items():
        Xc = phones11.spk_norm(project(TC, C["lens"], W), fc, C); Xd = phones11.spk_norm(project(TD, D["lens"], W), fd, D)
        c = phones3.lda_fit(phones3.segvec(Xc, C["segs"], len(Xc)), C["segs"][:, 2], phones.NP); F = phones3.segvec(Xd, D["segs"], len(Xd))
        pr = (((F - c["m"]) / c["s"]) @ c["W"].T + c["b"]).argmax(1)
        print(f"{name:5s} dims/frame={Xc.shape[1]:4d} all={np.mean(pr == y):.3f} vowels={np.mean(pr[v] == y[v]):.3f} consonants={np.mean(pr[~v] == y[~v]):.3f}", flush=True)
