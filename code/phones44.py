"""8d: transcript-FREE decoding (phone-loop first pass + kept model, chain / segmental +-2), scored by phone error rate. NIGHT_REPORT 8d."""
import json, sys, zlib
import numpy as np
sys.path.insert(0, "code")
import asr_compare as ac
import af, frontend as fe, phones, phones8, phones12, phones15, phones16, phones19 as c19, phones32 as j32, phones33 as ch, phones39 as m39, phones42 as p42, segdec as sd

NS = 3 * phones.NP + 1; SIL = 3 * phones.NP; S0 = 3 * np.arange(phones.NP); EXITS = np.r_[S0 + 2, SIL]

def scores(path, model):  # the detector state log-scores align.py uses (phones34.clip, same stable dither seed)
    x = fe.load(path); T = len(fe.frames(x.astype(np.float32)))
    tr = fe.tracks2(fe.dither(x, np.random.default_rng(zlib.crc32(path.encode())))); X = np.concatenate([fe.legendre(tr[t], fe.LEG4) for t in af.ALL_TRACKS], 1).astype(np.float32)[:T]
    L = phones.detector_logodds({"X": X, "ph": np.zeros(T)}, model); S = L @ model["comp"]["W"].T + model["comp"]["b"]; return S - np.logaddexp.reduce(S, 1, keepdims=True)

def loop_viterbi(E, ip):
    """Phone loop: substates 0->1->2 per phone, exit (substate 2 or silence) -> substate 0 of any phone or silence at cost ip. Returns [(t0, t1, phone|-1)]."""
    T = len(E); D = np.full(NS, -np.inf); D[S0] = E[0, S0] + ip; D[SIL] = E[0, SIL] + ip; B = np.zeros((T, NS), np.int16); B[0] = -1
    for t in range(1, T):
        p = D; ei = EXITS[np.argmax(p[EXITS])]; ev = p[ei] + ip; new = p.copy(); bp = np.arange(NS, dtype=np.int16)
        tk = ev > p[S0]; new[S0] = np.where(tk, ev, p[S0]); bp[S0] = np.where(tk, ei, S0)
        for s in (1, 2):
            c = p[S0 + s - 1] > p[S0 + s]; new[S0 + s] = np.where(c, p[S0 + s - 1], p[S0 + s]); bp[S0 + s] = np.where(c, S0 + s - 1, S0 + s)
        if ev > p[SIL] and ei != SIL: new[SIL] = ev; bp[SIL] = ei
        D = new + E[t]; B[t] = bp
    j = EXITS[np.argmax(D[EXITS])]; path = np.zeros(T, np.int64)
    for t in range(T - 1, -1, -1): path[t] = j; j = B[t, j] if t else j
    starts = [0] + [t for t in range(1, T) if (path[t] == SIL or path[t] % 3 == 0) and B[t, path[t]] != path[t]]  # entry events
    starts = sorted(set(starts)); seg = []
    for a, b in zip(starts, starts[1:] + [T]):
        p = -1 if path[a] == SIL else int(path[a] // 3)
        if seg and p == -1 and seg[-1][2] == -1: seg[-1] = (seg[-1][0], b, -1)
        else: seg.append((a, b, p))
    assert seg[0][0] == 0 and seg[-1][1] == T and all(x[1] == y[0] for x, y in zip(seg, seg[1:]))
    return seg

def edit(a, b):
    d = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        prev, d[0] = d.copy(), i
        for j, y in enumerate(b, 1): d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (x != y))
    return int(d[-1])
assert edit([1, 2, 3], [1, 3]) == 1 and edit([], [1, 2]) == 2 and edit([1, 2], [2, 1]) == 2

def free_set(segs_by_utt, lens):
    off = np.cumsum(np.r_[0, lens])[:-1]; ph = np.full(int(lens.sum()), phones.SILID, np.int16); out = []
    for u, sg in enumerate(segs_by_utt):
        for a, b, p in sg:
            if p >= 0: ph[off[u] + a: off[u] + b] = p; out.append((off[u] + a, off[u] + b, p))
    return dict(lens=lens, ph=ph, segs=np.array(out, np.int32))

def decode(Mod, T, S, sf, pl, chunk=60):
    """Per-segment marginals for fixed cuts (offset 0) and segmental +-2, utterance chunks."""
    utt = np.searchsorted(np.cumsum(S["lens"]), S["segs"][:, 0], side="right"); Pf = np.zeros((len(utt), phones.NP)); Ps = Pf.copy()
    for c0 in range(0, len(S["lens"]), chunk):
        sel = np.where((utt >= c0) & (utt < c0 + chunk))[0]
        if not len(sel): continue
        Sc = dict(S, segs=S["segs"][sel]); E = p42.emissions(Mod, T, Sc, sf, pl); rs, re_ = ch.runs(Sc)
        Pf[sel] = sd.forward_backward(E[:, 2:3, 2:3], rs, re_, Mod["logB"], Mod["lst"], c19.G_OF, 0.5); Ps[sel] = sd.forward_backward(E, rs, re_, Mod["logB"], Mod["lst"], c19.G_OF, 0.5); del E
    return utt, Pf, Ps

def per(hyp_utt, hyp, refs):
    errs = sum(edit(list(hyp[hyp_utt == u]), r) for u, r in enumerate(refs)); n = sum(len(r) for r in refs)
    nh = len(hyp); return dict(PER=round(errs / n, 4), hyp_over_ref=round(nh / n, 3))

if __name__ == "__main__":
    C = phones.load("cal"); fc = phones.split()["cal"]; RC = phones12.build("cal", fc, C["lens"]); yc = C["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    TC = phones15.causal_norm(RC, fc, C, 200, pm_, pv_); del RC; f_ = lambda T, S, fl: np.c_[phones15.feats(T, S), phones16.rate(S, fl, 5, pl)]; rc = np.random.default_rng(1)
    XC = np.r_[f_(TC, C, fc), f_(TC, j32.jitter(C, 1, rc), fc), f_(TC, j32.jitter(C, 2, rc), fc)]; del TC
    cc1 = c19.ctx_true(C); sub = phones8.sublabels(C["segs"], C["lens"], phones8.stress(fc, C["lens"]))
    Mod = m39.build(XC, np.r_[yc, yc, yc], np.r_[cc1, cc1, cc1], C, np.r_[sub, sub, sub]); del XC
    model = np.load(f"{phones.OUT}/model.npy", allow_pickle=True).item(); R = {}
    def run(name, ips):
        d = np.load(f"{phones.CACHE}/realcut_{name}.npz"); lens = d["lens"]; files = list(d["files"]) if name != "author" else list(ac.TRUTH)[:10]
        sf = files if name != "author" else [f"x/author/c/{i}.flac" for i in range(10)]; u0 = np.searchsorted(np.cumsum(lens), d["segs"][:, 0], side="right")
        refs = [list(d["segs"][u0 == u, 2]) for u in range(len(lens))]; Tr = p42.tracks(files, lens); Sc = [scores(f, model) for f in files]
        assert all(len(s) == l for s, l in zip(Sc, lens)); res = {}
        if name != "libriA":  # like-for-like reference: forced cuts through the same chunked path
            Sf = dict(segs=d["segs"], lens=lens, ph=d["ph"]); T = phones15.causal_norm(Tr, sf, Sf, 200, pm_, pv_); utt, Pf, Ps = decode(Mod, T, Sf, sf, pl)
            res["forced"] = dict(fixed=per(utt, Pf.argmax(1), refs), segmental=per(utt, Ps.argmax(1), refs), acc_segmental=c19.acc(Ps, d["segs"][:, 2], np.ones(len(utt), bool)))
            print(name, "forced", res["forced"], flush=True)
        for ip in ips:
            S = free_set([loop_viterbi(s, ip) for s in Sc], lens); T = phones15.causal_norm(Tr, sf, S, 200, pm_, pv_); utt, Pf, Ps = decode(Mod, T, S, sf, pl)
            res[f"ip{ip}"] = dict(first_pass=per(utt, S["segs"][:, 2], refs), fixed=per(utt, Pf.argmax(1), refs), segmental=per(utt, Ps.argmax(1), refs))
            print(name, f"ip{ip}", res[f"ip{ip}"], flush=True)
        R[name] = res
    run("libriA", (0, -4, -8, -16)); best = min((k for k in R["libriA"]), key=lambda k: R["libriA"][k]["segmental"]["PER"]); ipb = int(best[2:])
    print("chosen on libriA:", best, flush=True); run("libriB", (ipb,)); run("author", (ipb,)); R["chosen_ip"] = ipb
    json.dump(R, open(f"{phones.OUT}/phones44.json", "w"), indent=1)

def loop_viterbi_bg(E, ip, logB, lst, lam):
    """loop_viterbi + phone bigram on phone-to-phone entries (lam x logB[p, q]; from silence lam x lst[q]). lam 0 == loop_viterbi."""
    T = len(E); K = len(lst); D = np.full(NS, -np.inf); D[S0] = E[0, S0] + ip + lam * lst; D[SIL] = E[0, SIL] + ip; B = np.zeros((T, NS), np.int16); B[0] = -1
    for t in range(1, T):
        p = D; Mq = p[S0 + 2][:, None] + lam * logB; pa = Mq.argmax(0); ev = Mq[pa, np.arange(K)]; es = p[SIL] + lam * lst
        fs = es > ev; ev = np.where(fs, es, ev) + ip; ei = np.where(fs, SIL, S0[pa] + 2)
        new = p.copy(); bp = np.arange(NS, dtype=np.int16)
        tk = ev > p[S0]; new[S0] = np.where(tk, ev, p[S0]); bp[S0] = np.where(tk, ei, S0)
        for s in (1, 2):
            c = p[S0 + s - 1] > p[S0 + s]; new[S0 + s] = np.where(c, p[S0 + s - 1], p[S0 + s]); bp[S0 + s] = np.where(c, S0 + s - 1, S0 + s)
        xi = EXITS[:-1][np.argmax(p[EXITS[:-1]])]; xv = p[xi] + ip
        if xv > p[SIL]: new[SIL] = xv; bp[SIL] = xi
        D = new + E[t]; B[t] = bp
    j = EXITS[np.argmax(D[EXITS])]; path = np.zeros(T, np.int64)
    for t in range(T - 1, -1, -1): path[t] = j; j = B[t, j] if t else j
    starts = [0] + [t for t in range(1, T) if (path[t] == SIL or path[t] % 3 == 0) and B[t, path[t]] != path[t]]
    seg = []
    for a, b in zip(starts, starts[1:] + [T]):
        q = -1 if path[a] == SIL else int(path[a] // 3)
        if seg and q == -1 and seg[-1][2] == -1: seg[-1] = (seg[-1][0], b, -1)
        else: seg.append((a, b, q))
    return seg
