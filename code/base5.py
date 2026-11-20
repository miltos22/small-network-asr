"""Stage 1 BASE (docs/PLAN.md sections 4 + 11; design v3 = intended_model_v3.html): a fair streaming black box.
Input: RAW log-mel 80 (base5_prep.py). Causal normalisation: per channel, mean / std over the last NORMW frames (default 300 = 3 s) of
PAST audio only (boxcar; exact at utterance start). Trunk: conv subsampling x2 (k5, 20 ms look-ahead) + 4 blocks of
[LayerNorm -> depthwise conv k15 (RIGHT frames ahead, rest behind) -> GELU -> pointwise Linear -> dropout] with residual, then a ONE-DIRECTION
GRU (seconds of past) -> Linear to 39 phones + blank. Look-ahead = 20 ms + 4 x RIGHT x 20 ms (RIGHT=2: 180 ms <= 200 ms limit).
Width d = GRU hidden, searched so the parameter count hits PARAMS (B0 = 655924; 2 MB target ~1.7 M).
Recipe = B0's (bb142 RECIPE=b0): AdamW lr 3e-3 wd 1e-2, OneCycle, 60 epochs, SpecAugment (2 freq <= 15, 2 time <= min(40, 10 %)) applied
after the causal norm, select on libriA every 2 epochs, average of the 3 best, stop after 5 bad evals. PLUS speed perturbation:
each epoch every utterance uses speed 0.9 / 1.0 / 1.1 at random (precomputed by base5_prep.py).
GPU: RTX 3070 shared with the agent; HARD LIMIT 1 GB total for this process (nvidia-smi; checked after the first epoch and every eval;
the run aborts above LIMIT_MIB). torch allocator capped at ALLOC_MIB. CTC + log_softmax in fp32; trunk under bf16 autocast.
Arms: base | broken (labels deranged across utterances; must fail).
Usage: base5.py ARM SEED [EPOCHS]   env: PARAMS, RIGHT, NORMW, MAXFR (frames per batch, default 30000 as B0), OUTDIR, SPEEDS ("0.9,1.1" or "")."""
import os
import json, os, subprocess, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 14 Nov smoke: OOM at 406 MiB used + 182 MiB fragmented under the cap
import numpy as np, torch, torch.nn as nn

CACHE = os.environ.get("DATA_DIR", "data") + "/cache"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/phones_out/base5"
NPH, NMEL = 39, 80
RIGHT = int(os.environ.get("RIGHT", 2)); NORMW = int(os.environ.get("NORMW", 300)); K = 15
LIMIT_MIB = 1000; ALLOC_MIB = int(os.environ.get("ALLOC_MIB", 620))
SPEEDS = [s for s in os.environ.get("SPEEDS", "0.9,1.1").split(",") if s]


def load(name):
    d = np.load(f"{CACHE}/base5_{name}.npz"); X = torch.from_numpy(d["X"].astype(np.float32)); st = np.r_[0, np.cumsum(d["lens"])]
    return [X[a:c] for a, c in zip(st[:-1], st[1:])], [d[f"y{i}"].astype(np.int64) for i in range(int(d["n"]))]


def cnorm(x, w=NORMW):  # x (T, 80) raw log-mel -> causal mean/std over the last w frames (incl. the current one)
    cs = torch.cumsum(torch.cat([x.new_zeros(1, x.shape[1]), x]), 0); cs2 = torch.cumsum(torch.cat([x.new_zeros(1, x.shape[1]), x * x]), 0)
    t = torch.arange(1, len(x) + 1, device=x.device); lo = (t - w).clamp(min=0); n = (t - lo).unsqueeze(1).float()
    m = (cs[t] - cs[lo]) / n; v = ((cs2[t] - cs2[lo]) / n - m * m).clamp(min=0)
    return (x - m) / (v.sqrt() + 1.0)  # +1.0 dB-ish floor keeps the first frames finite (std 0 at t=1)


class Block(nn.Module):
    def __init__(s, d):
        super().__init__(); s.ln = nn.LayerNorm(d); s.dw = nn.Conv1d(d, d, K, groups=d); s.pw = nn.Linear(d, d); s.drop = nn.Dropout(0.1)

    def forward(s, h):  # h (B, T, d)
        z = s.ln(h).transpose(1, 2); z = nn.functional.pad(z, (K - 1 - RIGHT, RIGHT))
        return h + s.drop(s.pw(nn.functional.gelu(s.dw(z).transpose(1, 2))))


class Base(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.sub = nn.Conv1d(NMEL, d, 5, 2)  # padded (2, 2): output j sees input frames 2j-2 .. 2j+2 (20 ms ahead)
        s.blocks = nn.ModuleList([Block(d) for _ in range(4)]); s.rnn = nn.GRU(d, d, 1, batch_first=True); s.out = nn.Linear(d, NPH + 1)

    def trunk(s, x):  # x (B, T, 80) normalised -> list of 4 block outputs (B, ceil(T/2), d) + GRU output
        h = torch.relu(s.sub(nn.functional.pad(x.transpose(1, 2), (2, 2)))).transpose(1, 2); L = []
        for b in s.blocks: h = b(h); L.append(h)
        return L, s.rnn(h)[0]

    def forward(s, x):
        return s.out(s.trunk(x)[1]).float().log_softmax(-1)


def nparams(m): return sum(p.numel() for p in m.parameters())


def build(P):
    best = None
    for d in range(32, 1024, 4):
        n = nparams(Base(d))
        if best is None or abs(n - P) < abs(best[1] - P): best = (d, n)
        if n > P: break
    assert abs(best[1] - P) / P < 0.02, (best, P); return best[0]


def edit(a, b):
    d = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        prev, d[0] = d.copy(), i
        for j, y in enumerate(b, 1): d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (x != y))
    return int(d[-1])


def collapse(a): return [p for i, p in enumerate(a) if p != NPH and (i == 0 or p != a[i - 1])]


def evaluate(m, X, Y, dev):
    m.eval(); H = []
    with torch.no_grad():
        for x in X:
            H.append(collapse(m(cnorm(x.to(dev))[None])[0].argmax(-1).tolist()))
    n = sum(len(y) for y in Y); e = sum(edit(h, list(y)) for h, y in zip(H, Y))
    return dict(PER=round(e / n, 4), hyp_over_ref=round(sum(map(len, H)) / n, 3))


def specaug(x, lens, rng):  # bb142.specaug on causally normalised input (0 = local mean)
    x = x.clone()
    for bi, L in enumerate(lens):
        for _ in range(2): f = int(rng.integers(0, 16)); f0 = int(rng.integers(0, NMEL - f + 1)); x[bi, :, f0:f0 + f] = 0
        for _ in range(2): t = int(rng.integers(0, min(40, L // 10) + 1)); t0 = int(rng.integers(0, max(1, L - t + 1))); x[bi, t0:t0 + t] = 0
    return x


def gpu_mib():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return sum(int(l.split(",")[1]) for l in out.splitlines() if l.split(",")[0].strip() == str(os.getpid()))


def stream_check(m, x, dev):  # output at frame j must not change when audio after the look-ahead is cut off
    m.eval(); la = 2 + 4 * RIGHT * 2  # look-ahead in 10 ms input frames (sub: 2; each block RIGHT output frames = 2*RIGHT input frames)
    with torch.no_grad():
        full = m(cnorm(x.to(dev))[None])[0]
        for j in (5, len(full) // 2, len(full) - 12):
            cut = 2 * j + la + 1; part = m(cnorm(x[:cut].to(dev))[None])[0]
            assert torch.allclose(full[j], part[j], atol=1e-4), ("stream mismatch", j, (full[j] - part[j]).abs().max().item())


def train(arm, seed, epochs=60):
    dev = torch.device("cuda:0"); torch.cuda.set_per_process_memory_fraction(ALLOC_MIB / 8192, 0)
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    P = int(os.environ.get("PARAMS", 655924)); d = build(P); m = Base(d).to(dev); n = nparams(m)
    tag = f"{arm}_P{P}_s{seed}"; O = os.environ.get("OUTDIR", OUT); os.makedirs(O, exist_ok=True)
    assert not os.path.exists(f"{O}/{tag}.json"), "output exists: never mix runs"
    V = {}  # speed versions of the training set: V[speed] = (X, Y)
    for sp in ["1.0"] + SPEEDS:
        X, Y = [], []
        for s_ in ("fit", "cal"): x, y = load(s_ if sp == "1.0" else f"{s_}_sp{sp}"); X += x; Y += y
        V[sp] = (X, Y)
    Y = V["1.0"][1]; N = len(Y)
    if arm == "broken":
        pr = np.random.default_rng(1_000 + seed); perm = pr.permutation(N)
        while (perm == np.arange(N)).any(): perm = pr.permutation(N)
        Y = [Y[i] for i in perm]
    evals = ["libriA", "libriB", "testother", "author"]; E = {s_: load(s_) for s_ in evals}
    cfg = dict(maxfr=int(os.environ.get("MAXFR", 24000)), tag=tag, arm=arm, seed=seed, d=d, params=n, P=P, RIGHT=RIGHT, lookahead_ms=20 + 4 * RIGHT * 20, NORMW=NORMW, speeds=["1.0"] + SPEEDS,
               train_utts=N, train_h=round(sum(len(x) for x in V["1.0"][0]) / 100 / 3600, 3), epochs=epochs, alloc_mib=ALLOC_MIB)
    print(json.dumps(dict(config=cfg)), flush=True)
    stream_check(m, E["libriA"][0][0], dev); print(json.dumps(dict(stream_check="ok")), flush=True)
    maxfr = int(os.environ.get("MAXFR", 24000)); nb = int(np.ceil(sum(len(x) for x in V["1.0"][0]) / maxfr * 1.05))
    # MAXFR: B0 used 30000; 24000 keeps the 1 GB limit (14 Nov smoke)
    opt = torch.optim.AdamW(m.parameters(), 0.003, weight_decay=1e-2); ctc = nn.CTCLoss(blank=NPH, zero_infinity=True)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 0.003, total_steps=epochs * nb + 50, pct_start=0.1)
    best, bad, snaps, log, t0 = None, 0, [], [], time.time()
    for ep in range(epochs):
        m.train(); sp = rng.choice(list(V.keys()), size=N); X = [V[sp[i]][0][i] for i in range(N)]
        order = np.argsort([len(x) for x in X]); batches, cur, fr = [], [], 0
        for i in order:
            if cur and fr + len(X[i]) > maxfr: batches.append(cur); cur, fr = [], 0
            cur.append(i); fr += len(X[i])
        batches.append(cur); tot = 0.0
        for bi in rng.permutation(len(batches)):
            ix = batches[bi]; lens = [len(X[i]) for i in ix]
            xs = nn.utils.rnn.pad_sequence([cnorm(X[i].to(dev)) for i in ix], batch_first=True); xs = specaug(xs, lens, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L, g = m.trunk(xs)
            lp = m.out(g.float()).log_softmax(-1)
            loss = ctc(lp.transpose(0, 1), torch.from_numpy(np.concatenate([Y[i] for i in ix])), torch.tensor([(l + 1) // 2 for l in lens]),
                       torch.tensor([len(Y[i]) for i in ix]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            if sch.last_epoch < sch.total_steps - 1: sch.step()
            tot += loss.item() * len(ix)
        r = dict(epoch=ep + 1, loss=round(tot / N, 4), secs=round(time.time() - t0), gpu_mib=gpu_mib(), batches=len(batches))
        assert r["gpu_mib"] <= LIMIT_MIB, f"GPU memory {r['gpu_mib']} MiB > {LIMIT_MIB}: abort (the agent's 1 GB limit)"
        if (ep + 1) % 2 == 0 or ep + 1 == epochs:
            r["libriA"] = evaluate(m, *E["libriA"], dev); per = r["libriA"]["PER"]
            st = {k: v.detach().clone() for k, v in m.state_dict().items()}
            if best is None or per < best[0]: best, bad = (per, ep + 1, st), 0
            else: bad += 1
            snaps = sorted(snaps + [(per, ep + 1, st)], key=lambda t: t[0])[:3]
        print(json.dumps(r), flush=True); log.append(r)
        if bad >= 5: break
    m.load_state_dict(best[2])
    if len(snaps) > 1:
        avg = {k: sum(t[2][k] for t in snaps) / len(snaps) for k in best[2]}; m.load_state_dict(avg); ra = evaluate(m, *E["libriA"], dev)
        print(json.dumps(dict(avg_of=[t[1] for t in snaps], libriA=ra, best_single=best[0])), flush=True)
        if ra["PER"] <= best[0]: best = (ra["PER"], "avg" + "+".join(str(t[1]) for t in snaps), avg)
        else: m.load_state_dict(best[2])
    torch.save(best[2], f"{O}/{tag}.pt")
    mc = Base(d); mc.load_state_dict({k: v.cpu() for k, v in best[2].items()}); mc.eval()
    q = torch.ao.quantization.quantize_dynamic(mc, {nn.GRU, nn.Linear}, dtype=torch.qint8)  # depthwise + subsampling convs stay fp32 (noted)
    R = dict(config=cfg, best_epoch=best[1], log=log, sets={})
    for s_ in evals:
        f32 = evaluate(m, *E[s_], dev); i8 = evaluate(q, *E[s_], torch.device("cpu"))
        R["sets"][s_] = dict(fp32=f32, int8=i8); print(s_, "fp32", f32, "int8", i8, flush=True)
    json.dump(R, open(f"{O}/{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    train(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 60)
