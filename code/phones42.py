"""8b: decoder-chosen cuts on real (our aligner's) cuts: segmental chain over boundary offsets -2..+2. NIGHT_REPORT 8b."""
import glob, json, sys
import numpy as np
sys.path.insert(0, "code")
import asr_compare as ac
import frontend as fe, phones, phones8, phones12, phones15, phones16, phones19 as c19, phones32 as j32, phones33 as ch, phones38 as s38, phones39 as m39, segdec as sd

def tracks(files, lens):
    return np.concatenate([phones12.job((f, int(T))) for f, T in zip(files, lens)])

def emissions(Mod, T, S, files, pl):
    n = len(S["segs"]); O = len(sd.OFF); E = np.full((n, O, O, 5, phones.NP, 5), sd.NEG, np.float32); ends = np.cumsum(S["lens"]); st0 = ends - S["lens"]
    utt = np.searchsorted(ends, S["segs"][:, 0], side="right")
    for a, da in enumerate(sd.OFF):
        for b, db in enumerate(sd.OFF):
            sg = S["segs"].copy(); sg[:, 0] = np.clip(sg[:, 0] + da, st0[utt], ends[utt] - 1); sg[:, 1] = np.clip(sg[:, 1] + db, st0[utt] + 1, ends[utt])
            ok = sg[:, 1] > sg[:, 0]; sg[~ok, 1] = sg[~ok, 0] + 1; Sx = dict(S, segs=sg)
            X = np.c_[phones15.feats(T, Sx), phones16.rate(Sx, files, 5, pl)]; L = s38.emis_sub(Mod["Ms"], Mod["pcs"], m39.proj(Mod, X), Mod["parent"])
            L[~ok] = sd.NEG; E[:, a, b] = L
    return E

def dur_table(C):
    d = np.clip(C["segs"][:, 1] - C["segs"][:, 0], 1, 60); t = np.ones((phones.NP, 61)); np.add.at(t, (C["segs"][:, 2], d), 1)
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2); t = np.stack([np.convolve(r, k / k.sum(), "same") for r in t]); return np.log(t / t.sum(1, keepdims=True))

if __name__ == "__main__":
    C = phones.load("cal"); fc = phones.split()["cal"]; RC = phones12.build("cal", fc, C["lens"]); yc = C["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    TC = phones15.causal_norm(RC, fc, C, 200, pm_, pv_); del RC; f_ = lambda T, S, fl: np.c_[phones15.feats(T, S), phones16.rate(S, fl, 5, pl)]; rc = np.random.default_rng(1)
    XC = np.r_[f_(TC, C, fc), f_(TC, j32.jitter(C, 1, rc), fc), f_(TC, j32.jitter(C, 2, rc), fc)]; del TC
    cc1 = c19.ctx_true(C); sub = phones8.sublabels(C["segs"], C["lens"], phones8.stress(fc, C["lens"]))
    Mod = m39.build(XC, np.r_[yc, yc, yc], np.r_[cc1, cc1, cc1], C, np.r_[sub, sub, sub]); del XC; LD = dur_table(C); R = {}; Pkeep = {}
    for name in ("libriA", "libriB", "author"):
        d = np.load(f"{phones.CACHE}/realcut_{name}.npz"); S = dict(segs=d["segs"], lens=d["lens"], ph=d["ph"])
        files = list(d["files"]) if name != "author" else list(ac.TRUTH)[:10]; spk_files = files if name != "author" else [f"x/author/c/{i}.flac" for i in range(10)]
        T = phones15.causal_norm(tracks(files, S["lens"]), spk_files, S, 200, pm_, pv_); E = emissions(Mod, T, S, spk_files, pl); del T
        y = S["segs"][:, 2]; rs, re_ = ch.runs(S); m = np.ones(len(y), bool); dur = S["segs"][:, 1] - S["segs"][:, 0]
        dd = np.clip(dur[:, None, None] + sd.OFF[None, None, :] - sd.OFF[None, :, None], 1, 60)  # (n, O_start, O_end)
        ldur = np.transpose(LD[:, dd], (1, 2, 3, 0))[:, :, :, None, :, None].astype(np.float32)  # (n, O, O, 1, K, 1)
        res = {}
        Pf = sd.forward_backward(E[:, 2:3, 2:3], rs, re_, Mod["logB"], Mod["lst"], c19.G_OF, 0.5); res["fixed"] = c19.acc(Pf, y, m)
        Ps = sd.forward_backward(E, rs, re_, Mod["logB"], Mod["lst"], c19.G_OF, 0.5); res["segmental"] = c19.acc(Ps, y, m)
        Pd = sd.forward_backward(E, rs, re_, Mod["logB"], Mod["lst"], c19.G_OF, 0.5, ldur); res["segmental_dur"] = c19.acc(Pd, y, m)
        if name == "libriB":
            spk_of = np.array([files[u].split("/")[-3] for u in np.searchsorted(np.cumsum(S["lens"]), S["segs"][:, 0], side="right")])
            for k, P in (("segmental", Ps), ("segmental_dur", Pd)): res[f"per_speaker_{k}"] = ch.per_speaker(Pf, P, y, spk_of, m)
        R[name] = res; print(name, len(y), "segments", res, flush=True); del E
    best = max(("segmental", "segmental_dur"), key=lambda k: R["libriA"][k][1]); b0, bb = R["libriB"]["fixed"], R["libriB"][best]; ps = R["libriB"][f"per_speaker_{best}"]
    keep = bb[1] - b0[1] >= 0.002 and bb[0] >= b0[0] and ps["improved"] * 3 >= 2 * ps["speakers"]
    print("chosen on libriA:", best, "libriB:", bb, "vs fixed", b0, ps, "KEEP" if keep else "DROP", flush=True)
    json.dump(dict(R=R, chosen=best, keep=keep), open(f"{phones.OUT}/phones42.json", "w"), indent=1)
