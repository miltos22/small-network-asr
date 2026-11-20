"""3a: failure points of the current version (r76 hybrid). Confusion pairs on half A, what separates each pair (stress from the
TextGrid digits, duration, position, raw formants), and brittleness to boundary jitter (+-1, +-2 frames). NIGHT_REPORT 3a."""
import glob, json, re
import numpy as np
import af, phones, phones12, phones15, phones16, phones19 as c19, phones24 as h, phones25 as b

def current(XC, yc, cc, XD, D, Mfull=None, V=None):
    """The r76 hybrid (phones25) as one function; returns flat and hybrid posteriors and the projection."""
    if Mfull is None: Mfull = c19.fit(XC, yc, cc, 200); V = b.projection(XC, yc, cc, Mfull, 76)
    UC, UD = ((XC - Mfull["m"]) / Mfull["s"]) @ V, ((XD - Mfull["m"]) / Mfull["s"]) @ V
    B0 = c19.fit(UC, yc, np.zeros(len(yc), int), 1.0); P = c19.post(B0, UD, np.zeros(len(UD), int)); M = c19.fit(UC, yc, cc, 200)
    for _ in (1, 2, 3): w = c19.ctx_soft(D, P); P = c19.post_mix(M, UD, w)
    return P, h.hybrid(UC, yc, cc, UD, c19.ctx_soft(D, P), P, M, 1.0), Mfull, V

def stress(files, D):
    out = []
    for f in files:
        tg = f.replace(f"{phones.DATA}/LibriSpeech", f"{phones.DATA}/align").replace(".flac", ".TextGrid"); s = open(tg).read(); tier = s[s.index('name = "phones"'):]
        for a, bb, txt in re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', tier):
            p = re.sub(r"\d", "", txt.upper()); i0, i1 = int(round(float(a) * 100)), int(round(float(bb) * 100))
            if i1 > i0 and p in phones.PID: out.append(int(re.sub(r"\D", "", txt) or -1))
    return np.array(out)

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC, RD = phones12.build("cal", fc, C["lens"]), phones12.build("dev_other", fd, D["lens"]); yc, y = C["segs"][:, 2], D["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    TD = phones15.causal_norm(RD, fd, D, 200, pm_, pv_)
    XC = np.c_[phones15.feats(phones15.causal_norm(RC, fc, C, 200, pm_, pv_), C), phones16.rate(C, fc, 5, pl)]; XD = np.c_[phones15.feats(TD, D), phones16.rate(D, fd, 5, pl)]
    spk = np.array([f.split("/")[-3] for f in fd]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt]); cc = c19.ctx_true(C)
    P, Ph, Mfull, V = current(XC, yc, cc, XD, D); pr = Ph.argmax(1); print("current check A/B", c19.acc(Ph, y, A), c19.acc(Ph, y, ~A), flush=True)
    st = stress(fd, D); assert len(st) == len(y), (len(st), len(y))
    vow = np.isin(y, [phones.PID[p] for p in af.VOW]); print("vowel stress share 0/1/2:", np.round(np.bincount(st[vow] + 0, minlength=3)[:3] / vow.sum(), 3).tolist(), flush=True)
    dur = D["segs"][:, 1] - D["segs"][:, 0]; prev, nxt = c19.adjacency(D); N = phones12.NAMES; i1, i2 = N.index("lF1"), N.index("lF2")
    mid = (D["segs"][:, 0] + D["segs"][:, 1]) // 2; F1, F2 = RD[mid, i1], RD[mid, i2]
    E = A & (pr != y); nerr = E.sum(); pairs = {}
    for t_, p_ in zip(y[E], pr[E]): pairs[(t_, p_)] = pairs.get((t_, p_), 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:20]; rows = []
    print(f"half A: {A.sum()} segments, {nerr} errors; top 20 pairs = {sum(v for _, v in top) / nerr:.1%} of errors", flush=True)
    print("true->pred  n  %err | err vs correct-of-true: unstressed%  dur(frames)  before-pause%  after-vowel%  F1  F2 (raw mid)", flush=True)
    for (t_, p_), n in top:
        e = A & (y == t_) & (pr == p_); ok = A & (y == t_) & (pr == t_)
        f = lambda m: (np.mean(st[m] == 0) * 100 if vow[m].any() else float("nan"), np.median(dur[m]), np.mean(nxt[m] < 0) * 100,
                       np.mean(np.isin(y[np.maximum(prev[m], 0)], [phones.PID[q] for q in af.VOW]) & (prev[m] >= 0)) * 100, np.median(F1[m]), np.median(F2[m]))
        fe_, fo = f(e), f(ok); rows.append(dict(true=af.PHONES[t_], pred=af.PHONES[p_], n=int(n), err=fe_, ok=fo))
        print(f"{af.PHONES[t_]:>3}->{af.PHONES[p_]:<3} {n:5d} {n / nerr:5.1%} | " + "  ".join(f"{x:6.1f}/{z:6.1f}" for x, z in zip(fe_, fo)), flush=True)
    rng = np.random.default_rng(0); R = {}
    for j in (1, 2):
        sg = D["segs"].copy(); ends = np.cumsum(D["lens"]); starts = ends - D["lens"]
        sg[:, 0] = np.clip(sg[:, 0] + rng.integers(-j, j + 1, len(sg)), starts[utt], None); sg[:, 1] = np.clip(sg[:, 1] + rng.integers(-j, j + 1, len(sg)), None, ends[utt])
        sg[:, 1] = np.maximum(sg[:, 1], sg[:, 0] + 1); Dj = dict(D, segs=sg)
        XJ = np.c_[phones15.feats(TD, Dj), phones16.rate(Dj, fd, 5, pl)]; _, PJ, _, _ = current(XC, yc, cc, XJ, D, Mfull, V)
        R[f"jitter{j}"] = dict(A=c19.acc(PJ, y, A), B=c19.acc(PJ, y, ~A)); print(f"boundary jitter +-{j} frames:", R[f"jitter{j}"], flush=True)
    json.dump(dict(pairs=rows, jitter=R), open(f"{phones.OUT}/phones27.json", "w"), indent=1, default=float)
