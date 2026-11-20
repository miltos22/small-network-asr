"""7: consolidated model + unsupervised speaker profile (bias / diagonal affine in r76 from the model's own posteriors). NIGHT_REPORT 7."""
import glob, json, sys, zlib
import numpy as np
import align, phones, phones8, phones12, phones15, phones16, phones19 as c19, phones24 as h, phones25 as b, phones32 as j32, phones33 as ch, phones34 as hv, phones38 as s38

def build(XC, yc, cc, C, sub):
    Mfull = c19.fit(XC, yc, cc, 200); V = b.projection(XC, yc, cc, Mfull, 76); UC = ((XC - Mfull["m"]) / Mfull["s"]) @ V
    M = c19.fit(UC, yc, cc, 200); lab = 2 * yc + sub; used, z = np.unique(lab, return_inverse=True)
    Ms = c19.fit(UC, z, cc, 200, K=len(used)); pcs = np.zeros((25, len(used))); np.add.at(pcs, (cc, z), 1); pk = np.bincount(z, minlength=len(used)) / len(z)
    pcs = (pcs + 200 * pk) / (pcs.sum(1, keepdims=True) + 200); logB, lst = ch.bigram(yc[: len(C["segs"])], C)
    mu = np.stack([UC[yc == k].mean(0) for k in range(phones.NP)])
    return dict(Mfull=Mfull, V=V, UC=UC, yc=yc, cc=cc, M=M, Ms=Ms, pcs=pcs, parent=used // 2, logB=logB, lst=lst, mu=mu)

def proj(Mod, X): return ((X - Mod["Mfull"]["m"]) / Mod["Mfull"]["s"]) @ Mod["V"]

def infer(Mod, U, S):
    rs, re_ = ch.runs(S); P = ch.decode(s38.emis_sub(Mod["Ms"], Mod["pcs"], U, Mod["parent"]), rs, re_, Mod["logB"], Mod["lst"], 0.5, 2)
    return h.hybrid(Mod["UC"], Mod["yc"], Mod["cc"], U, c19.ctx_soft(S, P), P, Mod["M"], 1.0)

def adapt(Mod, U, S, spk_of, mode, tau=50.0, iters=2):
    P = infer(Mod, U, S); Ua = U.copy()
    for _ in range(iters):
        m = P @ Mod["mu"]
        for s in np.unique(spk_of):
            i = spk_of == s; z, mm, n = U[i], m[i], i.sum()
            if mode == "bias": a, bb = np.ones(U.shape[1]), (z - mm).sum(0) / (n + tau)
            else:
                ta = tau * (mm ** 2).mean(0); s_mm, s_m, s_zm, s_z = (mm ** 2).sum(0), mm.sum(0), (z * mm).sum(0), z.sum(0)
                det = (s_mm + ta) * (n + tau) - s_m ** 2; a = ((s_zm + ta) * (n + tau) - s_m * s_z) / det; bb = ((s_mm + ta) * s_z - s_m * (s_zm + ta)) / det
            Ua[i] = (z - bb) / a
        P = infer(Mod, Ua, S)
    return P

def his_clips(pl, pm_, pv_):
    model = np.load(f"{phones.OUT}/model.npy", allow_pickle=True).item(); lex = align.lexicon(); files = list(hv.ac.TRUTH)[:10]
    out = [hv.clip(f, model, lex) for f in files]; off = np.cumsum([0] + [len(o[1]) for o in out])[:-1]
    H = dict(lens=np.array([len(o[1]) for o in out]), ph=np.concatenate([o[1] for o in out]), segs=np.concatenate([o[2] + np.array([a, a, 0]) for o, a in zip(out, off)]))
    fake = [f"x/author/c/{i}.flac" for i in range(len(files))]; RH = np.concatenate([o[0] for o in out])
    return H, np.c_[phones15.feats(phones15.causal_norm(RH, fake, H, 200, pm_, pv_), H), phones16.rate(H, fake, 5, pl)]

if __name__ == "__main__":
    C = phones.load("cal"); D = phones.load("dev_other"); fc = phones.split()["cal"]; fd = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    RC, RD = phones12.build("cal", fc, C["lens"]), phones12.build("dev_other", fd, D["lens"]); yc, y = C["segs"][:, 2], D["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    TC, TD = phones15.causal_norm(RC, fc, C, 200, pm_, pv_), phones15.causal_norm(RD, fd, D, 200, pm_, pv_); del RC, RD
    f_ = lambda T, S, fl: np.c_[phones15.feats(T, S), phones16.rate(S, fl, 5, pl)]; rc = np.random.default_rng(1)
    XC = np.r_[f_(TC, C, fc), f_(TC, j32.jitter(C, 1, rc), fc), f_(TC, j32.jitter(C, 2, rc), fc)]; del TC
    cc1 = c19.ctx_true(C); st = phones8.stress(fc, C["lens"]); sub = phones8.sublabels(C["segs"], C["lens"], st)
    Mod = build(XC, np.r_[yc, yc, yc], np.r_[cc1, cc1, cc1], C, np.r_[sub, sub, sub]); del XC
    spk = np.array([f.split("/")[-3] for f in fd]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt]); spk_of = spk[utt]
    UD = proj(Mod, f_(TD, D, fd)); P0 = infer(Mod, UD, D); R = {"consolidated_exact": dict(A=c19.acc(P0, y, A), B=c19.acc(P0, y, ~A))}; print(R, flush=True)
    Dj = j32.jitter(D, 2, np.random.default_rng(0)); Pj = infer(Mod, proj(Mod, f_(TD, Dj, fd)), D); R["consolidated_jitter2"] = dict(A=c19.acc(Pj, y, A), B=c19.acc(Pj, y, ~A)); print(R["consolidated_jitter2"], flush=True); del TD
    for mode in ("bias", "diag"):
        Pa = adapt(Mod, UD, D, spk_of, mode); R[f"profile_{mode}"] = dict(A=c19.acc(Pa, y, A), B=c19.acc(Pa, y, ~A), per_speaker_B=ch.per_speaker(P0, Pa, y, spk_of, ~A)); print(mode, R[f"profile_{mode}"], flush=True)
    best = max(("profile_bias", "profile_diag"), key=lambda k: R[k]["A"][1]); b0, bb = R["consolidated_exact"]["B"], R[best]["B"]; ps = R[best]["per_speaker_B"]
    keep = ((bb[0] - b0[0] >= 0.002 and bb[1] >= b0[1]) or (bb[1] - b0[1] >= 0.002 and bb[0] >= b0[0])) and ps["improved"] * 3 >= 2 * ps["speakers"]
    print("profile chosen on half A:", best, "KEEP" if keep else "DROP", flush=True)
    H, XH = his_clips(pl, pm_, pv_); UH = proj(Mod, XH); yh = H["segs"][:, 2]; mh = np.ones(len(yh), bool); spk_h = np.zeros(len(yh), int)
    PH0 = infer(Mod, UH, H); R["him_consolidated"] = c19.acc(PH0, yh, mh)
    for mode in ("bias", "diag"): R[f"him_profile_{mode}"] = c19.acc(adapt(Mod, UH, H, spk_h, mode), yh, mh)
    print("author:", {k: R[k] for k in R if k.startswith("author")}, flush=True)
    json.dump(dict(R=R, chosen=best, keep=keep), open(f"{phones.OUT}/phones39.json", "w"), indent=1)
