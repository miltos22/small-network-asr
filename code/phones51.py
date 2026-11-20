"""8n: new frame cutter = closed-form LDA over 118 phone substates on tonight's per-frame Legendre features, in the same phone loop. NIGHT_REPORT 8n."""
import glob, json, sys
import numpy as np
sys.path.insert(0, "code")
import asr_compare as ac
import frontend as fe, phones, phones12, phones15, phones42 as p42, phones44 as p44, phones46 as p46

WIN = [fe._legendre(2, 3), fe._legendre(6, 4)]; CTX = 6

def frame_feats(T, lens, stack):
    Xw = phones12.project(T, lens, WIN).astype(np.float32)
    if not stack: return Xw
    ends = np.cumsum(lens); st = ends - lens; u = np.repeat(np.arange(len(lens)), lens); t = np.arange(len(Xw))
    lo, hi = np.maximum(t - CTX, st[u]), np.minimum(t + CTX, ends[u] - 1); return np.c_[Xw[lo], Xw, Xw[hi]]

def fit(X, y, K):
    m, s = X.mean(0), X.std(0) + 1e-6; Z = (X - m) / s; mu = np.stack([Z[y == k].mean(0) if (y == k).any() else np.zeros(Z.shape[1]) for k in range(K)])
    R = Z - mu[y]; Sw = R.T @ R / len(R) + 1e-3 * np.eye(Z.shape[1]); W = np.linalg.solve(Sw, mu.T).T; cnt = np.bincount(y, minlength=K)
    return dict(m=m, s=s, W=W.astype(np.float32), b=(-0.5 * np.einsum("kj,kj->k", W, mu) + np.log((cnt + 1) / (cnt.sum() + K))).astype(np.float32))

def scores(Mc, X):
    L = ((X - Mc["m"]) / Mc["s"]).astype(np.float32) @ Mc["W"].T + Mc["b"]; return L - np.logaddexp.reduce(L, 1, keepdims=True)

if __name__ == "__main__":
    Mod, pl, pm_, pv_ = p46.kept_model(); model = np.load(f"{phones.OUT}/model.npy", allow_pickle=True).item(); K = 3 * phones.NP + 1
    F = phones.load("fit"); ff = phones.split()["fit"]; TF = phones15.causal_norm(phones12.build("fit", ff, F["lens"]), ff, F, 200, pm_, pv_).astype(np.float32)
    y = np.where(F["ph"] == phones.SILID, 3 * phones.NP, F["ph"].astype(np.int64) * 3 + F["sub"]); ok = F["ph"] >= 0; sel = np.where(ok)[0][::3]
    MC = {}
    for v, stack in (("A", False), ("B", True)):
        X = frame_feats(TF, F["lens"], stack); MC[v] = fit(X[sel], y[sel], K); acc = float((scores(MC[v], X[sel[::10]]).argmax(1) == y[sel[::10]]).mean())
        print(v, "fit-frame state accuracy (training frames)", round(acc, 4), flush=True); del X
    del TF; np.save(f"{phones.OUT}/cutter51.npy", MC); R = {}
    for name in ("libriA", "libriB", "author"):
        d = np.load(f"{phones.CACHE}/realcut_{name}.npz"); lens = d["lens"]; files = list(d["files"]) if name != "author" else list(ac.TRUTH)[:10]
        sf = files if name != "author" else [f"x/author/c/{i}.flac" for i in range(10)]; u0 = np.searchsorted(np.cumsum(lens), d["segs"][:, 0], side="right"); refs = [list(d["segs"][u0 == u, 2]) for u in range(len(lens))]
        S0 = p44.free_set([p44.loop_viterbi(p44.scores(f, model), -4) for f in files], lens); T = phones15.causal_norm(p42.tracks(files, lens), sf, S0, 200, pm_, pv_)
        st = np.cumsum(np.r_[0, lens])[:-1]; R[name] = {}; grid = [(v, ip) for v in ("A", "B") for ip in (0, -4, -8)] if name == "libriA" else [best]
        mfa = None
        if name == "libriA":
            mfa = []
            for f, Tn in zip(files, lens):
                tg = f.replace(f"{phones.DATA}/LibriSpeech", f"{phones.DATA}/align").replace(".flac", ".TextGrid"); lab = np.full(Tn, -1)
                for a, b, p in phones.segments(tg):
                    if p in phones.PID: lab[a:min(b, Tn)] = phones.PID[p]
                mfa.append(lab)
        for v in sorted({g[0] for g in grid}):
            SC = scores(MC[v], frame_feats(T, lens, v == "B"))
            for (_, ip) in [g for g in grid if g[0] == v]:
                segs = [p44.loop_viterbi(SC[st[u]:st[u] + lens[u]], ip) for u in range(len(lens))]; S = p44.free_set(segs, lens)
                utt, _, Ps = p44.decode(Mod, T, S, sf, pl); hyp = Ps.argmax(1); r = p44.per(utt, hyp, refs); r["first_pass"] = p44.per(utt, S["segs"][:, 2], refs)
                if mfa is not None:
                    o = [[int(np.bincount(l[l >= 0]).argmax()) if (l := mfa[u][a:b])[l >= 0].size else -1 for a, b, p in segs[u] if p >= 0] for u in range(len(lens))]
                    r["pure_segmentation_PER"] = round(sum(p44.edit(h, rr) for h, rr in zip(o, refs)) / sum(map(len, refs)), 4)
                R[name][f"{v}_ip{ip}"] = r; print(name, v, ip, r, flush=True)
                if name == "libriB":
                    spk = [f.split("/")[-3] for f in files]; H1 = [list(hyp[utt == u]) for u in range(len(refs))]
                    utt0, _, P0 = p44.decode(Mod, T, S0, sf, pl); h0 = P0.argmax(1); H0 = [list(h0[utt0 == u]) for u in range(len(refs))]
                    spb = lambda HH: {k: sum(p44.edit(HH[u], refs[u]) for u in range(len(refs)) if spk[u] == k) / sum(len(refs[u]) for u in range(len(refs)) if spk[u] == k) for k in sorted(set(spk))}
                    a0, a1 = spb(H0), spb(H1); R["libriB_baseline_PER_recomputed"] = round(sum(p44.edit(h, rr) for h, rr in zip(H0, refs)) / sum(map(len, refs)), 4); R["libriB_speakers_improved"] = [sum(a1[k] < a0[k] for k in a0), len(a0)]
            del SC
        if name == "libriA": best = min(grid, key=lambda g: R["libriA"][f"{g[0]}_ip{g[1]}"]["PER"]); print("chosen on libriA:", best, flush=True)
    kb = f"{best[0]}_ip{best[1]}"; drop = R["libriB_baseline_PER_recomputed"] - R["libriB"][kb]["PER"]; iv, ns = R["libriB_speakers_improved"]
    R["chosen"] = kb; R["keep"] = bool(drop >= 0.005 and iv * 3 >= 2 * ns); print("libriB", R["libriB"][kb], "baseline", R["libriB_baseline_PER_recomputed"], "speakers", [iv, ns], "KEEP" if R["keep"] else "DROP", flush=True)
    json.dump(R, open(f"{phones.OUT}/phones51.json", "w"), indent=1)
