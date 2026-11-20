"""Warp alphas for train-clean-100 (19 Nov, reviewer; plan step a: warp + noise recipe at 2 MB / 100 h). CALIBRATION.
Same rule and scorer as vtln_grid.py / WARP=W (stage-1a base5 0.66 MB s0; mean negative entropy per alpha on GRID 0.80-1.20 step 0.04;
speaker alpha = argmax of the summed scores), run on the GPU batched over the grid (vtln_self.py path, known answer below).
One deviation, for cost: only PER_SPK utterances per speaker (seeded) are scored at each speed; a scored utterance gets the leave-one-out
alpha of its speaker's other scored utterances (as vtln_grid), an unscored one the argmax of all scored ones. Writes cache/vtln_tc100{,_sp0.9,_sp1.1}.npz
(alpha, scores of the scored rows, scored, spk, grid). Known answer: this path reproduces vtln_libriA.npz scores on 20 utts (atol 1e-3).
Usage: RIGHT=12 python3 -I vtln_tc100.py [sets...]   env PER_SPK (default 20), LIMIT (score only the first N speakers: smoke)"""
import os, sys, time, json
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, torch
import base5_next as B, aug2a
_r = os.environ.get("RIGHT"); os.environ["RIGHT"] = "2"
import base5 as b5
os.environ["RIGHT"] = _r; assert b5.RIGHT == 2
C = B.CACHE; PER_SPK = int(os.environ.get("PER_SPK", 20)); LIMIT = int(os.environ.get("LIMIT", 0))
GRID = np.round(np.arange(0.80, 1.2001, 0.04), 2)
dev = torch.device("cuda:0"); torch.cuda.set_per_process_memory_fraction(900 / 8192, 0)


def scores(fwd, X, G):
    S = np.zeros((len(X), len(G)))
    with torch.no_grad():
        for i, x in enumerate(X):
            x = torch.as_tensor(np.asarray(x, np.float32)) if not torch.is_tensor(x) else x
            xs = torch.stack([aug2a.vtlp(x.float(), float(a)) for a in G]).to(dev); lp = fwd(xs).float().log_softmax(-1)
            S[i] = (lp.exp() * lp).sum(-1).mean(-1).cpu().numpy()
    return S


if __name__ == "__main__":
    old = b5.Base(b5.build(655924)); old.load_state_dict(torch.load(D + "/phones_out/base5/base_P655924_s0.pt", map_location="cpu")); old.eval().to(dev)
    fwd = lambda xs: old(torch.stack([b5.cnorm(x) for x in xs]))
    v = np.load(f"{C}/vtln_libriA.npz"); assert np.array_equal(v["grid"], GRID)
    assert not B.WARP, "run with WARP unset: the scorer reads unwarped audio"
    X, _ = B.load("libriA")
    err = float(np.abs(scores(fwd, X[:20], GRID) - v["scores"][:20]).max())
    print("known answer: max |diff| vs vtln_libriA.npz", err, flush=True); assert err < 1e-3, err
    for nm in sys.argv[1:] or ["tc100", "tc100_sp0.9", "tc100_sp1.1"]:
        t0 = time.time(); meta = np.load(f"{C}/base5_{nm}_meta.npz"); lens = meta["lens"]; st = np.r_[0, np.cumsum(lens)]
        Xm = np.memmap(f"{C}/base5_{nm}_X.f16", np.float16, "r", shape=(int(st[-1]), B.NMEL))
        spk = [os.path.basename(str(f)).split("-")[0] for f in meta["files"]]; by = {}
        for i, s in enumerate(spk): by.setdefault(s, []).append(i)
        r = np.random.default_rng(77); keys = sorted(by)[:LIMIT] if LIMIT else sorted(by)
        pick = {s: sorted(r.choice(by[s], min(PER_SPK, len(by[s])), replace=False).tolist()) for s in keys}
        scored = sorted(i for ix in pick.values() for i in ix)
        S = scores(fwd, [Xm[st[i]:st[i + 1]] for i in scored], GRID); row = {i: k for k, i in enumerate(scored)}
        al = np.ones(len(lens))
        for s, ix in pick.items():
            T = S[[row[i] for i in ix]].sum(0)
            for i in by[s]: al[i] = GRID[int(np.argmax(T - S[row[i]] if i in row else T))]
        out = f"{C}/vtln_{nm}{'_smoke' if LIMIT else ''}.npz"
        np.savez(out, alpha=al, scores=S, scored=np.array(scored), spk=np.array(spk), grid=GRID, per_spk=PER_SPK)
        u, c = np.unique(al[[i for s in keys for i in by[s]]], return_counts=True)
        print(json.dumps(dict(set=nm, speakers=len(keys), scored=len(scored), secs=round(time.time() - t0), frames_scored=int(sum(lens[i] for i in scored)),
                              alpha_counts={str(a): int(b) for a, b in zip(u, c)}, out=out)), flush=True)
    print("VTLN_TC100_OK", flush=True)
