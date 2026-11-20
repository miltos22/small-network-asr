"""Self-scored warp (18 Nov, reviewer; EXPLORATION after 2-I / 2aW_): the same unsupervised VTLN rule as vtln_grid.py (per utterance and alpha in
GRID: mean negative entropy of the model's output on the warped input; alpha = argmax of the summed scores of the speaker's OTHER utterances),
but the SCORER is base6 itself (default e60_r12_causal_s0, plain), not the stage-1a base5 model: on the phone no second model would be needed.
Writes cache/vtln_S_{set}.npz (alpha, scores, grid, spk; spk copied from vtln_{set}.npz). Known answer: the scores of the stage-1a scorer
re-computed here for 20 libriA utts with this code path equal vtln_libriA.npz scores (atol 1e-3), so the only change is the scorer.
GPU (python3, python -I, under the GPU lock). Usage: RIGHT=12 NORM=causal python -I vtln_self.py"""
import os, sys
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, torch
import base5_next as B, base6 as B6, aug2a
# base5.py reads RIGHT at import: the old stage-1a scorer was trained with RIGHT 2, so import it with RIGHT 2, then restore the env
_r = os.environ.get("RIGHT"); os.environ["RIGHT"] = "2"
import base5 as b5
os.environ["RIGHT"] = _r; assert b5.RIGHT == 2 and B.RIGHT == 12
C = B.CACHE; SCORER = os.environ.get("SCORER", "e60_r12_causal_s0"); OUTPRE = os.environ.get("OUTPRE", "S_"); DRAW = os.environ.get("DRAW", "")
# 18 Nov (plan 2-V): DRAW=d scores only the draw's training utterances (base6.draw_indices); every other training utterance gets alpha 1 (never read)
SETS = ["fit", "cal", "fit_sp0.9", "cal_sp0.9", "fit_sp1.1", "cal_sp1.1", "libriA", "libriB", "testother", "author", "cleanS", "author2"]
dev = torch.device("cuda:0"); torch.cuda.set_per_process_memory_fraction(900 / 8192, 0)


def scores(fwd, X, G):
    S = np.zeros((len(X), len(G)))
    with torch.no_grad():
        for i, x in enumerate(X):
            xs = torch.stack([aug2a.vtlp(x.float(), float(a)) for a in G]).to(dev); lp = fwd(xs).float().log_softmax(-1)
            S[i] = (lp.exp() * lp).sum(-1).mean(-1).cpu().numpy()
    return S


if __name__ == "__main__":
    old = b5.Base(b5.build(655924)); old.load_state_dict(torch.load(D + "/phones_out/base5/base_P655924_s0.pt", map_location="cpu")); old.eval().to(dev)
    v = np.load(f"{C}/vtln_libriA.npz"); X, _ = B.load("libriA")
    S0 = scores(lambda xs: old(torch.stack([b5.cnorm(x) for x in xs])), X[:20], v["grid"]); err = float(np.abs(S0 - v["scores"][:20]).max())
    print("known answer: old scorer via this path, max |diff| vs vtln_libriA.npz", err, flush=True); assert err < 1e-3, err
    m = B6.Base6(B.build(655924)); m.load_state_dict(torch.load(f"{B6.OUT}/{SCORER}.pt", map_location="cpu")); m.eval().to(dev)
    fwd = lambda xs: m(torch.stack([B6.norm(x) for x in xs]))
    for nm in SETS:
        v = np.load(f"{C}/vtln_{nm}.npz"); G, spk = v["grid"], list(v["spk"]); X, _ = B.load(nm); assert len(X) == len(spk), nm
        keep = list(range(len(X)))
        if DRAW != "" and nm.split("_sp")[0] in ("fit", "cal"):
            ix, _, _ = B6.draw_indices(DRAW); nf = len(np.load(f"{C}/base5_fit.npz")["lens"])
            keep = [i for i in ix if i < nf] if nm.startswith("fit") else [i - nf for i in ix if i >= nf]
        S = np.zeros((len(X), len(G))); S[keep] = scores(fwd, [X[i] for i in keep], G); al = np.ones(len(X)); by = {}
        for i in keep: by.setdefault(spk[i], []).append(i)
        for s, ix in by.items():
            T = S[ix].sum(0)
            for i in ix: al[i] = G[int(np.argmax(T - S[i]))] if len(ix) > 1 else 1.0
        np.savez(f"{C}/vtln_{OUTPRE}{nm}.npz", alpha=al, scores=S, grid=G, spk=np.array(spk), scorer=SCORER, scored=np.array(keep))
        print(nm, len(keep), "of", len(X), "alpha agree with old scorer", round(float((al == v["alpha"]).mean()), 3), "corr", round(float(np.corrcoef(al, v["alpha"])[0, 1]), 3),
              "alpha counts", dict(zip(*np.unique(al, return_counts=True))), flush=True)
    print("VTLN_SELF_OK", flush=True)
