"""Step 5a: his 10 clips, dictionary labels timed by our aligner, current hybrid r76 model (LibriSpeech cal only). NIGHT_REPORT step 5."""
import glob, json, sys, zlib
import numpy as np
sys.path.insert(0, "code")
import asr_compare as ac
import af, align, frontend as fe, phones, phones12, phones14, phones15, phones16, phones19 as c19, phones27 as cur

def clip(path, model, lex):
    # stable dither seed: Python's hash() changes every process, which moved the cuts between runs
    x = fe.load(path); T = len(fe.frames(x.astype(np.float32))); R = phones12.job((path, T))
    tr = fe.tracks2(fe.dither(x, np.random.default_rng(zlib.crc32(path.encode())))); X = np.concatenate([fe.legendre(tr[t], fe.LEG4) for t in af.ALL_TRACKS], 1).astype(np.float32)[:T]
    L = phones.detector_logodds({"X": X, "ph": np.zeros(T)}, model); S = L @ model["comp"]["W"].T + model["comp"]["b"]; S = S - np.logaddexp.reduce(S, 1, keepdims=True)
    words = [w.upper() for w in ac.norm(ac.TRUTH[path])]; st = align.chain(words, lex); path_, _ = align.viterbi(S, st); seg = align.segments(path_, st)
    ph = np.full(T, phones.SILID, np.int16); segs = []
    for t0, t1, p in seg:
        if p: ph[t0:t1] = phones.PID[p]; segs.append((t0, t1, phones.PID[p]))
    want = sum(len(lex[w]) for w in words); assert len(segs) == want, (path, len(segs), want)  # every dictionary phone gets a segment
    return R, ph, np.array(segs, np.int32)

if __name__ == "__main__":
    model = np.load(f"{phones.OUT}/model.npy", allow_pickle=True).item(); lex = align.lexicon(); files = list(ac.TRUTH)
    out = [clip(f, model, lex) for f in files]; off = np.cumsum([0] + [len(o[1]) for o in out])[:-1]
    H = dict(lens=np.array([len(o[1]) for o in out]), ph=np.concatenate([o[1] for o in out]), segs=np.concatenate([o[2] + np.array([a, a, 0]) for o, a in zip(out, off)]))
    RH = np.concatenate([o[0] for o in out]); fake = [f"x/author/c/{i}.flac" for i in range(len(files))]  # one speaker stream
    print(f"his clips: {len(files)}, frames {len(RH)}, phone segments {len(H['segs'])}, median dur {np.median(H['segs'][:, 1] - H['segs'][:, 0])} frames", flush=True)
    C = phones.load("cal"); fc = phones.split()["cal"]; RC = phones12.build("cal", fc, C["lens"]); yc = C["segs"][:, 2]
    spc = (C["ph"] >= 0) & (C["ph"] != phones.SILID); pm_, pv_ = RC[spc].mean(0), RC[spc].var(0); pl = float(np.log(C["segs"][:, 1] - C["segs"][:, 0]).mean())
    XC = np.c_[phones15.feats(phones15.causal_norm(RC, fc, C, 200, pm_, pv_), C), phones16.rate(C, fc, 5, pl)]; del RC
    XH = np.c_[phones15.feats(phones15.causal_norm(RH, fake, H, 200, pm_, pv_), H), phones16.rate(H, fake, 5, pl)]
    P, Ph, _, _ = cur.current(XC, yc, c19.ctx_true(C), XH, H); y = H["segs"][:, 2]; allm = np.ones(len(y), bool)
    top3 = lambda Q: round(float(np.mean([y[i] in np.argsort(-Q[i])[:3] for i in range(len(y))])), 4)
    R = dict(flat=dict(acc=c19.acc(P, y, allm), top3=top3(P)), hybrid=dict(acc=c19.acc(Ph, y, allm), top3=top3(Ph)))
    utt = np.searchsorted(np.cumsum(H["lens"]), H["segs"][:, 0], side="right")
    R["per_clip_hybrid_39"] = [round(float(np.mean(Ph[utt == u].argmax(1) == y[utt == u])), 3) for u in range(len(files))]
    pr = Ph.argmax(1); pairs = {}
    for a, b_ in zip(y[pr != y], pr[pr != y]): pairs[(af.PHONES[a], af.PHONES[b_])] = pairs.get((af.PHONES[a], af.PHONES[b_]), 0) + 1
    R["top_confusions"] = sorted(([a, b_, n] for (a, b_), n in pairs.items()), key=lambda t: -t[2])[:15]
    R["per_phone"] = {af.PHONES[k]: [int((y == k).sum()), round(float(np.mean(pr[y == k] == k)), 2)] for k in range(phones.NP) if (y == k).any()}
    print(json.dumps(R, indent=0), flush=True); json.dump(R, open(f"{phones.OUT}/phones34_him.json", "w"), indent=1)
