"""4c: boundary-robust calibration: cal cuts exact + jittered +-1 + +-2, test exact / +-1 / +-2. NIGHT_REPORT 4c."""
import glob, json
import numpy as np
import phones, phones12, phones15, phones16, phones19 as c19, phones27 as cur

def jitter(S, j, rng):
    sg = S["segs"].copy(); ends = np.cumsum(S["lens"]); starts = ends - S["lens"]; utt = np.searchsorted(ends, sg[:, 0], side="right")
    sg[:, 0] = np.clip(sg[:, 0] + rng.integers(-j, j + 1, len(sg)), starts[utt], None); sg[:, 1] = np.clip(sg[:, 1] + rng.integers(-j, j + 1, len(sg)), None, ends[utt])
    sg[:, 1] = np.maximum(sg[:, 1], sg[:, 0] + 1); sg[:, 0] = np.minimum(sg[:, 0], sg[:, 1] - 1); return dict(S, segs=sg)

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC, RD = phones12.build("cal", fc, C["lens"]), phones12.build("dev_other", fd, D["lens"]); yc, y = C["segs"][:, 2], D["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    TC, TD = phones15.causal_norm(RC, fc, C, 200, pm_, pv_), phones15.causal_norm(RD, fd, D, 200, pm_, pv_); del RC, RD
    f = lambda T, S, fl: np.c_[phones15.feats(T, S), phones16.rate(S, fl, 5, pl)]
    rc = np.random.default_rng(1); XC = np.r_[f(TC, C, fc), f(TC, jitter(C, 1, rc), fc), f(TC, jitter(C, 2, rc), fc)]; del TC
    cc1 = c19.ctx_true(C); ycs, ccs = np.r_[yc, yc, yc], np.r_[cc1, cc1, cc1]
    spk = np.array([f_.split("/")[-3] for f_ in fd]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt])
    rd = np.random.default_rng(0); R = {}; Mfull = V = None
    for j in (0, 1, 2):
        Dj = D if j == 0 else jitter(D, j, rd); XD = f(TD, Dj, fd)
        _, Ph, Mfull, V = cur.current(XC, ycs, ccs, XD, D, Mfull, V); R[f"test_jitter{j}"] = dict(A=c19.acc(Ph, y, A), B=c19.acc(Ph, y, ~A)); print(f"test_jitter{j}", R[f"test_jitter{j}"], flush=True)
    keep = R["test_jitter2"]["B"][1] - 0.4594 >= 0.01 and 0.5567 - R["test_jitter0"]["B"][1] <= 0.005
    print("reference half B 39-way: exact 0.5567, +-1 0.5199, +-2 0.4594 ->", "KEEP" if keep else "DROP", flush=True)
    json.dump(dict(R=R, keep=keep), open(f"{phones.OUT}/phones32.json", "w"), indent=1)
