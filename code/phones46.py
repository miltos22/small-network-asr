"""8h: transcript-free segmental lattice: our model chooses the segmentation (lattice.py), boundary score + per-segment psi. NIGHT_REPORT 8h."""
import json, sys
import numpy as np
from scipy.signal import argrelextrema
sys.path.insert(0, "code")
import asr_compare as ac
import bnd, frontend as fe, lattice, phones, phones3, phones8, phones12, phones13, phones15, phones16, phones19 as c19, phones32 as j32, phones38 as s38, phones39 as m39, phones42 as p42, phones44 as p44

PSI, WS, LMAX, DMIN, DMAX = (-2.0, -1.0, 0.0, 1.0, 2.0), (0.0, 0.5, 1.0, 2.0), 10, 2, 40

def ck(c): return "_".join(f"{k}{v}" for k, v in sorted(c.items())) if isinstance(c, dict) else c

def anti_score(an, X):
    Z = (X - an["m"]) / an["s"]
    if "W" in an:  # mixture of per-type anti nodes (8k)
        from scipy.special import logsumexp
        return ((logsumexp(Z @ an["W"].T + an["B"], 1) - np.log(len(an["B"]))) / an["sd"]).astype(np.float32)
    return ((Z @ an["w"] + an["b"]) / an["sd"]).astype(np.float32)

def pair_scores(Bd, T, lens, frames):
    X = bnd.bvec(T, lens, frames); return np.stack([bnd.score(Bd, X, np.full(len(X), k)) for k in range(16)], 1)  # (n, 16) = 4*gl + gr

def run_set(Mod, Bd, model, name, pl, pm_, pv_, limit=None, anti=None, cells=None, cutter=None, emis=None, lag=None, node_score=None, arc_ok=None, arc_emis=None):
    """lag: commit lag in frames (None = whole region); node_score(T, lens, frames) -> extra end-node score; arc_ok(T, lens, t0, t1) -> arc mask (adjacent-node arcs always kept)."""
    cells = cells or [(p, w, 0.0) for p in PSI for w in WS]
    d = np.load(f"{phones.CACHE}/realcut_{name}.npz"); lens = d["lens"][:limit]; files = (list(d["files"]) if name != "author" else list(ac.TRUTH)[:10])[:limit]
    sf = files if name != "author" else [f"x/author/c/{i}.flac" for i in range(len(files))]; u0 = np.searchsorted(np.cumsum(lens), d["segs"][:, 0], side="right")
    refs = [list(d["segs"][u0 == u, 2]) for u in range(len(lens))]; Sc = [p44.scores(f, model) for f in files]; st = np.cumsum(np.r_[0, lens])[:-1]
    S = p44.free_set([p44.loop_viterbi(s, -4) for s in Sc], lens); T = phones15.causal_norm(p42.tracks(files, lens), sf, S, 200, pm_, pv_)
    if cutter:  # 8o: new frame cutter's state scores replace the old detector's for candidates + frame evidence (regions/normalisation stay)
        import phones51; SCn = phones51.scores(cutter[0], phones51.frame_feats(T, lens, cutter[1])); Sc = [SCn[st[u]:st[u] + lens[u]] for u in range(len(lens))]; del SCn
    rate = phones16.rate(S, sf, 5, pl); Xw = phones12.project(T, lens, [fe._legendre(2, 3), fe._legendre(6, 4)])
    prev, nxt = c19.adjacency(S); rs, re_ = np.where(prev < 0)[0], np.where(nxt < 0)[0]
    regions = [[] for _ in lens]  # per utterance: list of node arrays (global frames)
    for u in range(len(lens)):
        a = {x for x, _, _ in p44.loop_viterbi(Sc[u], 4)}; fr = st[u] + np.arange(lens[u]); m = pair_scores(Bd, T, lens, fr).max(1)
        e = argrelextrema(m, np.greater_equal, order=1)[0]; a |= set(e[m[e] > 0].tolist()); cand = np.array(sorted(a)) + st[u]
        for r0, r1 in zip(rs, re_):
            if st[u] <= S["segs"][r0, 0] < st[u] + lens[u]:
                lo, hi = S["segs"][r0, 0], S["segs"][r1, 1]; regions[u].append(np.r_[lo, cand[(cand > lo) & (cand < hi)], hi])
    run_set.regions = regions  # candidate nodes per utterance (diag67)
    hyp = {ck(c): [[] for _ in lens] for c in cells}; run_set.segs = {ck(c): [[] for _ in lens] for c in cells}  # chosen spans (global frames)
    DC = [c for c in cells if isinstance(c, dict)]
    if DC:  # 8m: per-context log phone prior (for the posterior vs mixture), 3-substate frame prefix sums, duration table
        lp = np.log(np.stack([Mod["pcs"][:, Mod["parent"] == k].sum(1) for k in range(phones.NP)], 1) + 1e-12).reshape(5, 5, phones.NP).transpose(0, 2, 1)  # [gp, k, gn]
        L3 = np.concatenate([s[:, :3 * phones.NP] - np.logaddexp.reduce(s[:, :3 * phones.NP], 1, keepdims=True) for s in Sc]); CS3 = np.vstack([np.zeros((1, L3.shape[1])), np.cumsum(L3, 0)]).astype(np.float32); del L3
        LD = p42.dur_table(phones.load("cal")).astype(np.float32)
    if any(not isinstance(c, dict) and len(c) == 4 for c in cells):  # 8l: frame detector log phone posteriors, cumulative over global frames
        LP = np.concatenate([np.log(np.stack([np.exp(s[:, 3 * k:3 * k + 3]).sum(1) for k in range(phones.NP)], 1) + 1e-12) for s in Sc])
        LP -= np.logaddexp.reduce(LP, 1, keepdims=True); CS = np.vstack([np.zeros((1, phones.NP)), np.cumsum(LP, 0)]).astype(np.float32); del LP
    for c0 in range(0, len(lens), 20):
        items = [(u, nodes) for u in range(c0, min(c0 + 20, len(lens))) for nodes in regions[u]]; arcs_l, segs = [], []
        for u, nodes in items:
            A = np.array([(i, k) for i in range(len(nodes) - 1) for k in range(i + 1, min(len(nodes), i + LMAX + 1)) if DMIN <= nodes[k] - nodes[i] <= DMAX] or [(0, len(nodes) - 1)])
            if arc_ok: A = A[arc_ok(T, lens, nodes[A[:, 0]], nodes[A[:, 1]]) | (A[:, 1] == A[:, 0] + 1)]
            arcs_l.append(A); segs.append(np.c_[nodes[A[:, 0]], nodes[A[:, 1]], np.zeros(len(A), int)])
        sg = np.concatenate(segs).astype(np.int64); fi = np.clip(np.searchsorted(S["segs"][:, 0], sg[:, 0], side="right") - 1, 0, len(rate) - 1)
        X = np.c_[phones3.segvec(Xw, sg, len(Xw)), phones13.seg_features(T, sg, lens), rate[fi]]; UD = m39.proj(Mod, X); EA = 0.5 * (emis(Mod, UD, np.concatenate([np.full(len(A_), u_) for (u_, _), A_ in zip(items, arcs_l)])) if emis else s38.emis_sub(Mod["Ms"], Mod["pcs"], UD, Mod["parent"])); AS = anti_score(anti, X) if anti else np.zeros(len(X), np.float32); del X
        if arc_emis: EA = arc_emis(EA, T, sg, lens, rate[fi])  # step 9: per-arc tree term (None reproduces)
        FS = (CS[sg[:, 1]] - CS[sg[:, 0]]) if any(not isinstance(c, dict) and len(c) == 4 for c in cells) else None
        if DC:
            E0 = EA / 0.5; from scipy.special import logsumexp as _lse; TL = (E0 - _lse(E0 + lp[None], axis=2, keepdims=True)).astype(np.float32); del E0
            dd = (sg[:, 1] - sg[:, 0]).astype(np.float32); q1 = sg[:, 0] + (sg[:, 1] - sg[:, 0]) // 3; q2 = sg[:, 0] + 2 * (sg[:, 1] - sg[:, 0]) // 3
            F3 = sum((CS3[b_][:, q::3] - CS3[a_][:, q::3]) for q, (a_, b_) in enumerate(((sg[:, 0], q1), (q1, q2), (q2, sg[:, 1]))))
            LDa = LD[:, np.clip(sg[:, 1] - sg[:, 0], 1, 60)].T
        allnodes = np.concatenate([n for _, n in items]); PS = pair_scores(Bd, T, lens, allnodes); o = 0; oa = 0; NX = node_score(T, lens, allnodes) if node_score else np.zeros(len(allnodes))
        for (u, nodes), A in zip(items, arcs_l):
            ps = PS[o:o + len(nodes)]; o += len(nodes); BN = np.zeros((len(nodes), phones.NP, 5), np.float32)
            for gl in range(4):
                for gr in range(4): BN[:, c19.G_OF == gl, gr] = ps[:, 4 * gl + gr][:, None]
            BN[-1] = 0; nx = NX[o - len(nodes):o].copy(); nx[-1] = 0; nx = nx[:, None, None].astype(np.float32); ea = EA[oa:oa + len(A)]; as_ = AS[oa:oa + len(A)][:, None, None, None]; fs = FS[oa:oa + len(A)][:, None, :, None] if FS is not None else None
            if DC: sl = slice(oa, oa + len(A)); tl, d_, f3, ld = TL[sl], dd[sl][:, None, None, None], F3[sl][:, None, :, None], LDa[sl][:, None, :, None]
            oa += len(A)
            for c in cells:
                if isinstance(c, dict):  # 8m arc: kappa d^gamma [log P(c|z) - log pi_c] + dur x log P(d|c) + b3 x 3-substate frame sum (+ anti)
                    p, w, a = c["psi"], c["w"], c.get("a", 0.0); e_ = 0.5 * d_ ** c["g"] * tl + c.get("dur", 0.0) * ld + c.get("b3", 0.0) * f3
                    if a: e_ = e_ - 0.5 * a * as_
                    ph_, sp_ = lattice.viterbi_lag(len(nodes) - 1, A, e_, w * BN + nx, Mod["logB"], Mod["lst"], c19.G_OF, p, nodes, lag); hyp[ck(c)][u] += ph_; run_set.segs[ck(c)][u] += sp_; continue
                p, w, a = c[:3]; e_ = ea - 0.5 * a * as_ if a else ea
                if len(c) == 4: e_ = e_ + c[3] * fs
                ph_, sp_ = lattice.viterbi_lag(len(nodes) - 1, A, e_, w * BN + nx, Mod["logB"], Mod["lst"], c19.G_OF, p, nodes, lag); hyp[ck(c)][u] += ph_; run_set.segs[ck(c)][u] += sp_
        del EA
    res = {}
    for key, H in hyp.items():
        if isinstance(key, str): n = sum(len(r) for r in refs); res[key] = dict(PER=round(sum(p44.edit(h, r) for h, r in zip(H, refs)) / n, 4), hyp_over_ref=round(sum(map(len, H)) / n, 3)); continue
        errs = sum(p44.edit(h, r) for h, r in zip(H, refs)); n = sum(len(r) for r in refs); res[f"psi{key[0]}_w{key[1]}" + (f"_a{key[2]}" if anti else "") + (f"_b{key[3]}" if len(key) == 4 else "")] = dict(PER=round(errs / n, 4), hyp_over_ref=round(sum(map(len, H)) / n, 3))
    spk = [f.split("/")[-3] for f in files] if name != "author" else ["author"] * len(files); base = None
    if name == "libriB":  # phones44 baseline hypotheses on the same free cuts, for the paired per-speaker test
        utt, _, Ps = p44.decode(Mod, T, S, sf, pl); base = [list(Ps.argmax(1)[utt == u]) for u in range(len(lens))]
    return res, hyp, refs, spk, base

def kept_model():
    """The kept consolidated model (m39.build, gamma 0, jitter-calibrated), cached: deterministic seeds, ~10 min to build."""
    import os, pickle
    p = f"{phones.OUT}/mod_m39_g0.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    C = phones.load("cal"); fc = phones.split()["cal"]; RC = phones12.build("cal", fc, C["lens"]); yc = C["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    TC = phones15.causal_norm(RC, fc, C, 200, pm_, pv_); del RC; f_ = lambda T, S, fl: np.c_[phones15.feats(T, S), phones16.rate(S, fl, 5, pl)]; rc = np.random.default_rng(1)
    XC = np.r_[f_(TC, C, fc), f_(TC, j32.jitter(C, 1, rc), fc), f_(TC, j32.jitter(C, 2, rc), fc)]; del TC
    cc1 = c19.ctx_true(C); sub = phones8.sublabels(C["segs"], C["lens"], phones8.stress(fc, C["lens"]))
    Mod = m39.build(XC, np.r_[yc, yc, yc], np.r_[cc1, cc1, cc1], C, np.r_[sub, sub, sub]); del XC
    Mod.pop("UC", None); out = (Mod, pl, pm_, pv_); pickle.dump(out, open(p, "wb")); return out

if __name__ == "__main__":
    Bd = np.load(f"{phones.OUT}/bnd_model.npy", allow_pickle=True).item()
    Mod, pl, pm_, pv_ = kept_model()
    model = np.load(f"{phones.OUT}/model.npy", allow_pickle=True).item(); R = {}; b44 = json.load(open(f"{phones.OUT}/phones44.json"))
    for name in ("libriA", "libriB", "author"):
        res, hyp, refs, spk, base = run_set(Mod, Bd, model, name, pl, pm_, pv_); R[name] = res; print(name, res, flush=True)
        if name == "libriA": best = min(res, key=lambda k: res[k]["PER"]); bk = tuple(float(x[3:] if i == 0 else x[1:]) for i, x in enumerate(best.split("_"))) + (0.0,); print("chosen on libriA:", best, flush=True)
        if name == "libriB":
            H = hyp[bk]; spb = lambda HH: {s: sum(p44.edit(HH[u], refs[u]) for u in range(len(refs)) if spk[u] == s) / sum(len(refs[u]) for u in range(len(refs)) if spk[u] == s) for s in sorted(set(spk))}
            a, b = spb(base), spb(H); R["libriB_baseline_PER_recomputed"] = round(sum(p44.edit(h, r) for h, r in zip(base, refs)) / sum(map(len, refs)), 4)
            R["libriB_speakers_improved"] = [sum(b[s] < a[s] for s in a), len(a)]; R["libriB_spk"] = {s: round(sum(p44.edit(H[u], refs[u]) for u in range(len(refs)) if spk[u] == s) / sum(len(refs[u]) for u in range(len(refs)) if spk[u] == s), 4) for s in sorted(set(spk))}
    R["chosen"] = best; R["baseline_libriB_PER"] = b44["libriB"][f"ip{b44['chosen_ip']}"]["segmental"]["PER"]
    drop = R["baseline_libriB_PER"] - R["libriB"][best]["PER"]; iv, ns = R["libriB_speakers_improved"]; R["keep"] = bool(drop >= 0.005 and iv * 3 >= 2 * ns)
    print("libriB", R["libriB"][best], "vs baseline", R["baseline_libriB_PER"], "drop", round(drop, 4), "speakers", R["libriB_speakers_improved"], "KEEP" if R["keep"] else "DROP", flush=True)
    json.dump(R, open(f"{phones.OUT}/phones46.json", "w"), indent=1)
