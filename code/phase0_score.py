"""Phase 0b / 0e / 0f input (docs/PLAN.md section 2): rescore saved checkpoints on clean / hard / his voice, CPU, inference only.
Per (model, set): greedy PER with sub / del / ins, hyp/ref, per-utterance (edits, ref phones, sub, del, ins) for the speaker bootstrap (0g).
Log-probabilities (float16) are cached for the first seed of each family (0e beam, 0f deletion positions) under CACHE/phase0_lp/.
Resumable: one JSON line per (model, set) in logs_base5/phase0_score.jsonl; done pairs are skipped.
Order: cleanS + him for every model first (new information), then the hard sets.
EXPLORATION. cleanS is TRAINING-speaker data for base100 (the 100 h model).
Usage: python3 phase0_score.py   (nice 19, 2 threads)"""
import json, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "2"
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D + "/logs_2c/run2")
import numpy as np, torch
import struct2c_AS_RUN as S, base5_next as B

torch.set_num_threads(2); os.nice(19); O = D + "/phones_out"; LP = B.CACHE + "/phase0_lp"; os.makedirs(LP, exist_ok=True)
RES = D + "/logs_base5/phase0_score.jsonl"
# (family, arm, P, folder, file kind, skip, seeds)
FAM = [("base232", "base", 655924, "struct2c", "base", False, range(8)),
       ("base232_s1a", "base", 655924, "base5", "base", False, range(5)),
       ("base284", "base", 959385, "struct2c_diag", "base", False, range(3)),
       ("full232SKIP", "full", 972200, "struct2c_diag2", "full", True, range(3)),
       ("full232", "full", 959385, "struct2c_diag", "full", False, range(3)),
       ("full066", "full", 655924, "struct2c", "full", False, range(8)),
       ("lite", "lite", 655924, "struct2c", "lite", False, range(8)),
       ("tok", "tok", 655924, "struct2c", "tok", False, range(8)),
       ("base100", "base", 1700000, "base5", "base", False, range(1))]
# author2 added 17 Nov 13:1x for the resume run (additive; no other change)
SETS_FIRST = ["cleanS", "author", "author2"]; SETS_HARD = ["libriA", "libriB", "testother"]


def ops(h, r):  # (sub, del, ins, list of (ref index, op) for deleted / substituted ref phones, alignment pairs ref->hyp index)
    n, m = len(r), len(h); d = np.zeros((n + 1, m + 1), np.int32); d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = r[i - 1]; row, prev = d[i], d[i - 1]
        for j in range(1, m + 1): row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ri != h[j - 1]))
    i, j, s_, de, ins = n, m, 0, 0, 0; dels, pair = [], {}
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            if r[i - 1] != h[j - 1]: s_ += 1
            pair[i - 1] = j - 1; i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1: de += 1; dels.append(i - 1); i -= 1
        else: ins += 1; j -= 1
    return s_, de, ins, dels, pair


def greedy(arm, lp):  # -> 39-phone hyp and the output frame of each hyp phone
    ids = lp.argmax(-1).tolist(); out, fr, prev = [], [], None
    for t, k in enumerate(ids):
        if k != prev:
            ph = [k] if arm in ("base", "lite") and k != B.NPH else (S.L2.TO39[k] if arm not in ("base", "lite") and k not in (S.BLANKV, S.L2.SIL) else [])
            out += ph; fr += [t] * len(ph)
        prev = k
    return out, fr


def model(fam, arm, P, folder, kind, skip, seed):
    S.SKIP = skip; S.MIDS = True
    m = S.make(arm, S.build(arm, P)) if arm != "base" else B.Base(S.build("base", P) if folder.startswith("struct") else B.build(P))
    m.load_state_dict(torch.load(f"{O}/{folder}/{kind}_P{P}_s{seed}.pt", map_location="cpu")); return m.eval()


def run_one(m, arm, X, Y, keep):
    tot = dict(sub=0, dele=0, ins=0, N=0, H=0); per = []; dpos = []; LPs = []
    with torch.no_grad():
        for x, y in zip(X, Y):
            x = x.to("cpu") if hasattr(x, "to") and not isinstance(x, np.ndarray) else torch.from_numpy(np.asarray(x, np.float32))
            lp = m(B.cnorm(x.float())[None])[0].float().numpy(); h, fr = greedy(arm, lp); y = list(map(int, y))
            s_, de, ins, dels, pair = ops(h, y); T = len(lp)
            for i in dels:  # frame of a deleted ref phone ~ midpoint of the nearest aligned neighbours (proxy for its time)
                a = max([pair[k] for k in pair if k < i], default=None); b_ = min([pair[k] for k in pair if k > i], default=None)
                fa = fr[a] if a is not None else 0; fb = fr[b_] if b_ is not None else T - 1; dpos.append(((fa + fb) / 2, T))
            tot["sub"] += s_; tot["dele"] += de; tot["ins"] += ins; tot["N"] += len(y); tot["H"] += len(h); per.append([s_ + de + ins, len(y), s_, de, ins])
            if keep: LPs.append(lp.astype(np.float16))
    N = tot["N"]; r = dict(PER=round(100 * (tot["sub"] + tot["dele"] + tot["ins"]) / N, 2), sub=round(100 * tot["sub"] / N, 2),
                         dele=round(100 * tot["dele"] / N, 2), ins=round(100 * tot["ins"] / N, 2), hyp_ref=round(tot["H"] / N, 3), N=N)
    return r, per, dpos, LPs


if __name__ == "__main__":
    done = set()
    if os.path.exists(RES):
        for line in open(RES): j = json.loads(line); done.add((j["model"], j["set"]))
    data = {}
    for stage in (SETS_FIRST, SETS_HARD):
        for fam, arm, P, folder, kind, skip, seeds in FAM:
            for seed in seeds:
                tag = f"{fam}_s{seed}"; todo = [s for s in stage if (tag, s) not in done]
                if not todo: continue
                m = model(fam, arm, P, folder, kind, skip, seed)
                for sname in todo:
                    if sname not in data: data[sname] = B.load(sname)
                    t0 = time.time(); keep = seed == min(seeds)
                    r, per, dpos, LPs = run_one(m, arm, *data[sname], keep)
                    if keep: np.savez(f"{LP}/{tag}_{sname}.npz", lp=np.concatenate(LPs), lens=np.array([len(z) for z in LPs]))
                    with open(RES, "a") as fh:
                        fh.write(json.dumps(dict(model=tag, set=sname, arm=arm, P=P, **r, sec=round(time.time() - t0), per_utt=per,
                                                 del_pos=[[round(a, 1), b_] for a, b_ in dpos])) + "\n")
                    print(f"{tag:16s} {sname:9s} PER {r['PER']:6.2f} S {r['sub']:5.2f} D {r['dele']:5.2f} I {r['ins']:5.2f} h/r {r['hyp_ref']} "
                          f"[{time.time() - t0:.0f}s]", flush=True)
    print("ALL DONE", flush=True)
