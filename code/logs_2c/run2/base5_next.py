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


# FEAT: prefix of offline-transformed feature files (removal2a.py: "2aN_" arm 2a, "2aR_" control b, "2aI_" identity); "" = originals.
FEAT = os.environ.get("FEAT", ""); assert FEAT in ("", "2aI_", "2aN_", "2aR_"), FEAT


def load(name):
    d = np.load(f"{CACHE}/base5_{FEAT}{name}.npz"); X = torch.from_numpy(d["X"].astype(np.float32)); st = np.r_[0, np.cumsum(d["lens"])]
    return [X[a:c] for a, c in zip(st[:-1], st[1:])], [d[f"y{i}"].astype(np.int64) for i in range(int(d["n"]))]


class Row:
    # one utterance in a memmapped f16 feature file; read only when batched (100 h = 17 GB, too big for RAM)
    __slots__ = ("a",)
    def __init__(s, a): s.a = a
    def __len__(s): return len(s.a)
    def to(s, dev): return torch.from_numpy(np.asarray(s.a, np.float32)).to(dev)


def load_mm(name):
    # base5_prep100.py format: base5_{name}_X.f16 (rows = lens.sum(), 80 cols) + base5_{name}_meta.npz (lens, yoff, ycat)
    d = np.load(f"{CACHE}/base5_{name}_meta.npz"); lens = d["lens"]; st = np.r_[0, np.cumsum(lens)]
    X = np.memmap(f"{CACHE}/base5_{name}_X.f16", np.float16, "r", shape=(int(st[-1]), NMEL)); yo, yc = d["yoff"], d["ycat"].astype(np.int64)
    assert len(yo) == len(lens) + 1 and yo[-1] == len(yc)
    return [Row(X[a:c]) for a, c in zip(st[:-1], st[1:])], [yc[a:c] for a, c in zip(yo[:-1], yo[1:])]


DATA = os.environ.get("DATA", "small")
# DATA: small = fit+cal (5.4 h dev-clean pool); tc100 = train-clean-100
# AUG: comma list of speaker dials (ARM_2a_DESIGN section 5), applied to raw log-mel BEFORE the causal norm; default none = stage-1 base.
AUG = [a for a in os.environ.get("AUG", "").split(",") if a]; assert set(AUG) <= {"vtlp", "noise", "events"}, AUG
NOISE = None
if AUG: sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; aug2a.py lives next to this file
if "noise" in AUG or "events" in AUG:
    import aug2a
    _nd = np.load(f"{CACHE}/noise_esc50_f1234.npz")  # folds 1-4 only; fold 5 = future noisy TEST set
    _ns = np.r_[0, np.cumsum(_nd["lens"])]; _nx = torch.from_numpy(_nd["X"])
    NOISE = [_nx[a:c] for a, c in zip(_ns[:-1], _ns[1:])]
elif "vtlp" in AUG:
    import aug2a


# SPKNORM=cmvn: per-speaker mean/var normalisation from the speaker's OTHER utterances (spk_stats.py), before the causal norm (comparator
# 'base+std', ARM_2a_DESIGN section 3). Training: applied AFTER the dials with the clean pool's stats (mismatch on warped / noisy copies = extra
# variation, chosen and stated); test: no dials, stats match.
SPKNORM = os.environ.get("SPKNORM", ""); assert SPKNORM in ("", "cmvn"), SPKNORM
# Scale back by the training pool's median speaker sd (3.549, spk_stats.py on fit) so the causal norm's +1.0 floor sees the raw-log-mel scale it
# was set for: plain (x - mu) / sd made inputs ~3.5x smaller and the 4-epoch check stalled (hyp/ref .079 vs .673 plain). A global constant, so
# the per-speaker normalisation itself is unchanged.
GSD = 3.549
LR = float(os.environ.get("LR", 0.003)); TAIL = int(os.environ.get("TAIL", 0)); TAIL_LR = float(os.environ.get("TAIL_LR", 0.003))
PATIENCE = int(os.environ.get("PATIENCE", 5))
# REWARD (the author 15 Nov ~06:3x: reward the right answer more than the wrong ones are punished, ratio "5:1"): CTC's gradient on the logits per
# frame is p - gamma: -gamma pushes the phones of the right path UP (reward), +p pushes every phone DOWN (punishment); both sum to 1 per
# frame. REWARD = k scales the push-up only: grad = k * (-gamma) + p. 1 = standard CTC. Grad clipping (5.0) still applies after.
REWARD = float(os.environ.get("REWARD", 1.0))
# REWARD_BLANK (15 Nov ~07:1x, after REWARD=5 collapsed to all-blank): the push-up scale for the blank class only; default = REWARD (old
# behaviour). 1 = phones get k:1, blank stays 1:1 (blank is the right answer on ~75 % of frames, so a uniform k mostly rewards "nothing").
REWARD_BLANK = float(os.environ.get("REWARD_BLANK", REWARD))
# REWARD_PROJ=1 (15 Nov ~07:2x, after k = 1/3 ALSO collapsed although the custom path equals plain CTC at k = 1 on a real batch): remove the
# per-frame mean over classes from the modified logit gradient. Softmax ignores a uniform logit shift, so that part changes no output, but it
# is the same direction every step and Adam turns it into full-size steps in the trunk (hypothesis for the collapse; this flag is its test).
REWARD_PROJ = os.environ.get("REWARD_PROJ", "0") == "1"
# BLANKPEN = lambda (the author 15 Nov ~07:2x: "punish nodes for choosing silence when its just anothers turn, making low values acceptable but
# punish high"): extra loss lambda * sum over frames of w * p_blank * (1 - gamma_blank), gamma = CTC's alignment share (detached). A frame
# where another phone is due costs in proportion to how HIGH blank is there; a low blank costs little; blank's own frames cost nothing.
# Same per-sample weight w as the CTC 'mean' loss. Its logit gradient sums to 0 over classes (no uniform part).
BLANKPEN = float(os.environ.get("BLANKPEN", 0.0))


def reward_backward(z, loss, flen, tlen, k, kb=None):  # kb = blank push-up scale (None = k)
    # z (B, T, C) logits, loss = CTC 'mean' (per-sample nll / target length, mean over batch); flen / tlen = frame / target lengths
    gz, = torch.autograd.grad(loss, z, retain_graph=True); B, T, _ = z.shape; p = z.detach().softmax(-1)
    w = (1.0 / (tlen.to(z.device).float() * B)); mask = (torch.arange(T, device=z.device)[None] < flen.to(z.device)[:, None]).float()
    valid = (gz.abs().flatten(1).sum(1) > 0).float()  # zero_infinity zeroes impossible samples: no reward for them either
    gam = p * (w * valid)[:, None, None] * mask[..., None] - gz  # the push-up part (>= 0), same normalisation as the loss
    kc = torch.full((z.shape[-1],), k - 1.0, device=z.device); kc[NPH] = (k if kb is None else kb) - 1.0  # per-class extra push-up (blank = index NPH)
    G = gz - kc * gam
    if BLANKPEN > 0:
        gb = (gam[..., NPH] / (w[:, None] + 1e-30)).clamp(0, 1); pb = p[..., NPH]; on = torch.zeros_like(p); on[..., NPH] = 1.0
        G = G + BLANKPEN * ((w * valid)[:, None] * mask * (1 - gb) * pb)[..., None] * (on - p)
    if REWARD_PROJ: G = G - G.mean(-1, keepdim=True) * mask[..., None]
    z.backward(G); return gam


def selftest_reward():  # CPU known answers: k = 1 equals loss.backward; gamma >= 0; gamma sums to the per-sample weight on every valid frame
    torch.manual_seed(0); B, T, C = 3, 20, NPH + 1; ctc = nn.CTCLoss(blank=NPH, zero_infinity=True)
    flen = torch.tensor([20, 15, 9]); tlen = torch.tensor([5, 4, 3]); y = torch.randint(0, NPH, (int(tlen.sum()),))
    for k in (1.0, 5.0):
        z0 = torch.randn(B, T, C); z = z0.clone().requires_grad_(); loss = ctc(z.log_softmax(-1).transpose(0, 1), y, flen, tlen)
        gam = reward_backward(z, loss, flen, tlen, k); zr = z0.clone().requires_grad_(); ctc(zr.log_softmax(-1).transpose(0, 1), y, flen, tlen).backward()
        assert gam.min() > -1e-6, gam.min()
        for b in range(B): assert torch.allclose(gam[b, :flen[b]].sum(-1), torch.full((int(flen[b]),), 1.0 / (float(tlen[b]) * B)), atol=1e-6)
        assert torch.allclose(z.grad, zr.grad - (k - 1) * gam, atol=1e-7)
        if k == 1.0: assert torch.allclose(z.grad, zr.grad, atol=1e-7)
        up, down = -(k * gam).sum(), (zr.grad + gam).sum(); print(f"reward selftest k={k}: total push-up {float(up):.4f} push-down {float(down):.4f} ratio {float(-up / down):.2f}")
    print("reward selftest ok")


def spkstats(name):
    d = np.load(f"{CACHE}/spk_{FEAT}{name}.npz"); return [torch.from_numpy(v) for v in d["mu"]], [torch.from_numpy(v) for v in d["sd"]]


def dials(x, rng):
    # x raw log-mel on the device; one VTLP alpha ~ U(0.9, 1.1) per utterance; noise on half the utterances at SNR ~ U(5, 20) dB
    if "vtlp" in AUG: x = aug2a.vtlp(x, float(rng.uniform(0.9, 1.1)))
    if NOISE is not None and rng.random() < 0.5:
        if "events" in AUG: x = aug2a.add_noise_events(x, NOISE, rng)  # time-varying background (his 15 Nov request), replaces the steady mix
        else: n = NOISE[int(rng.integers(len(NOISE)))].to(x.device).float(); x = aug2a.add_noise(x, n, float(rng.uniform(5, 20)), rng)
    return x


# CHAN (the author 15 Nov ~07:0x "more channels more good doesnt just apply to us"): extra input channels for the black box, appended AFTER the
# causal norm so they keep what the norm removes. 'level' = 3 causal channels from the raw total log power E (log sum of the 80 band powers):
# (E - fit median 2.856) / 3.617 (absolute loudness), (E - min of E over the last 3 s) / 10 (loudness above a running floor, minimum
# statistics), its 20 ms rise (E_t - E_t-2) / 5 (onsets). 'noise3' = 3 channels of N(0, 1) (same-width noise control, realization 84).
CHAN = os.environ.get("CHAN", ""); assert CHAN in ("", "level", "noise3"), CHAN; NCH = 3 if CHAN else 0


def channels(x, w=NORMW):
    if CHAN == "noise3": return torch.randn(len(x), 3, device=x.device)
    E = torch.logsumexp(x, 1); lo = -nn.functional.max_pool1d(nn.functional.pad(-E[None, None], (w - 1, 0), value=-float("inf")), w, 1)[0, 0]
    d = E - torch.cat([E[:1], E[:1], E])[: len(E)]
    return torch.stack([(E - 2.856) / 3.617, (E - lo) / 10.0, d / 5.0], 1)


def cnorm(x, w=NORMW):
    z = _cnorm(x, w)
    return torch.cat([z, channels(x, w)], 1) if CHAN else z


def _cnorm(x, w=NORMW):  # x (T, 80) raw log-mel -> causal mean/std over the last w frames (incl. the current one)
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
        s.sub = nn.Conv1d(NMEL + NCH, d, 5, 2)  # padded (2, 2): output j sees input frames 2j-2 .. 2j+2 (20 ms ahead)
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
        if DATA == "tc100":
            X, Y = load_mm("tc100" if sp == "1.0" else f"tc100_sp{sp}")
        else:
            for s_ in ("fit", "cal"): x, y = load(s_ if sp == "1.0" else f"{s_}_sp{sp}"); X += x; Y += y
        V[sp] = (X, Y)
    assert not FEAT or (DATA == "small" and SPKNORM), "FEAT files exist for the 5.4 h sets and are used with their own cmvn stats"
    MS = {}
    if SPKNORM:
        assert DATA == "small", "speaker stats exist for the 5.4 h sets only"
        for sp in V:
            mu, sd = [], []
            for s_ in ("fit", "cal"): a, b_ = spkstats(s_ if sp == "1.0" else f"{s_}_sp{sp}"); mu += a; sd += b_
            assert len(mu) == len(V[sp][0]); MS[sp] = (mu, sd)
    Y = V["1.0"][1]; N = len(Y)
    if arm == "broken":
        pr = np.random.default_rng(1_000 + seed); perm = pr.permutation(N)
        while (perm == np.arange(N)).any(): perm = pr.permutation(N)
        Y = [Y[i] for i in perm]
    evals = ["libriA", "libriB", "testother", "author"]; E = {s_: load(s_) for s_ in evals}
    # dev-other (5.13 h, 33 speakers): reported at the end only, not used for selection (selection stays libriA, as for the 5.4 h base)
    if SPKNORM:  # devother has no speaker stats yet: not reported in SPKNORM runs
        for s_ in evals:
            mu, sd = spkstats(s_); E[s_] = ([(x - a) / b_ * GSD for x, a, b_ in zip(E[s_][0], mu, sd)], E[s_][1])
    else:
        evals.append("devother"); E["devother"] = load_mm("devother")
    cfg = dict(maxfr=int(os.environ.get("MAXFR", 24000)), tag=tag, arm=arm, seed=seed, d=d, params=n, P=P, RIGHT=RIGHT, lookahead_ms=20 + 4 * RIGHT * 20, NORMW=NORMW, speeds=["1.0"] + SPEEDS,
               lr=LR, tail=TAIL, tail_lr=TAIL_LR, patience=PATIENCE, data=DATA, aug=AUG, spknorm=SPKNORM, feat=FEAT, reward=REWARD, reward_blank=REWARD_BLANK, reward_proj=REWARD_PROJ, blankpen=BLANKPEN, chan=CHAN, train_utts=N, train_h=round(sum(len(x) for x in V["1.0"][0]) / 100 / 3600, 3), epochs=epochs, alloc_mib=ALLOC_MIB)
    print(json.dumps(dict(config=cfg)), flush=True)
    stream_check(m, E["libriA"][0][0], dev); print(json.dumps(dict(stream_check="ok")), flush=True)
    maxfr = int(os.environ.get("MAXFR", 24000)); nb = int(np.ceil(sum(len(x) for x in V["1.0"][0]) / maxfr * 1.05))
    # MAXFR: B0 used 30000; 24000 keeps the 1 GB limit (14 Nov smoke)
    # LR = OneCycle peak (B0 recipe 0.003). TAIL extra epochs after the main phase at TAIL_LR falling linearly to 0 (the author 15 Nov: "set it to
    # 0.05 for 20 passes and then potentially have a few extra to fix the rough edges"). PATIENCE = evals without libriA gain before stopping.
    opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-2); ctc = nn.CTCLoss(blank=NPH, zero_infinity=True)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=epochs * nb + 50, pct_start=0.1)
    best, bad, snaps, log, t0 = None, 0, [], [], time.time(); ck = f"{O}/{tag}.ckpt"; ep0 = 0
    # RESUME (researcher.md 5.7): state written after every epoch; a crashed run restarts from its last finished epoch with the same config
    if os.path.exists(ck):
        S = torch.load(ck, map_location=dev, weights_only=False); assert S["cfg"] == cfg, "resume with a different config"
        m.load_state_dict(S["model"]); opt.load_state_dict(S["opt"]); sch.load_state_dict(S["sch"]); rng.bit_generator.state = S["rng"]
        torch.set_rng_state(S["trng"].cpu()); torch.cuda.set_rng_state(S["crng"].cpu()); best, bad, snaps, log, ep0 = S["best"], S["bad"], S["snaps"], S["log"], S["ep"]
        t0 -= log[-1]["secs"] if log else 0; print(json.dumps(dict(resumed_from_epoch=ep0)), flush=True)
    for ep in range(ep0, epochs + TAIL):
        m.train(); sp = rng.choice(list(V.keys()), size=N); X = [V[sp[i]][0][i] for i in range(N)]
        if SPKNORM: MU = [MS[sp[i]][0][i] for i in range(N)]; SD = [MS[sp[i]][1][i] for i in range(N)]
        order = np.argsort([len(x) for x in X]); batches, cur, fr = [], [], 0
        for i in order:
            if cur and fr + len(X[i]) > maxfr: batches.append(cur); cur, fr = [], 0
            cur.append(i); fr += len(X[i])
        batches.append(cur); tot = 0.0; nbat = 0
        for bi in rng.permutation(len(batches)):
            ix = batches[bi]; lens = [len(X[i]) for i in ix]
            def inp(i):
                x = X[i].to(dev)
                if AUG: x = dials(x, rng)
                if SPKNORM: x = (x - MU[i].to(dev)) / SD[i].to(dev) * GSD
                return cnorm(x)
            xs = nn.utils.rnn.pad_sequence([inp(i) for i in ix], batch_first=True); xs = specaug(xs, lens, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L, g = m.trunk(xs)
            z = m.out(g.float()); lp = z.log_softmax(-1); flen = torch.tensor([(l + 1) // 2 for l in lens]); tlen = torch.tensor([len(Y[i]) for i in ix])
            loss = ctc(lp.transpose(0, 1), torch.from_numpy(np.concatenate([Y[i] for i in ix])), flen, tlen)
            if ep >= epochs:
                for pg in opt.param_groups: pg["lr"] = TAIL_LR * max(0.0, 1 - ((ep - epochs) * len(batches) + nbat) / (TAIL * len(batches)))
            opt.zero_grad()
            if REWARD == 1.0 and REWARD_BLANK == 1.0 and BLANKPEN == 0: loss.backward()
            else: reward_backward(z, loss, flen, tlen, REWARD, REWARD_BLANK)
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); nbat += 1
            if ep < epochs and sch.last_epoch < sch.total_steps - 1: sch.step()
            tot += loss.item() * len(ix)
        r = dict(epoch=ep + 1, lr_end=float(f"{opt.param_groups[0]['lr']:.3g}"), loss=round(tot / N, 4), secs=round(time.time() - t0), gpu_mib=gpu_mib(), batches=len(batches))
        assert r["gpu_mib"] <= LIMIT_MIB, f"GPU memory {r['gpu_mib']} MiB > {LIMIT_MIB}: abort (the agent's 1 GB limit)"
        if (ep + 1) % 2 == 0 or ep + 1 in (epochs, epochs + TAIL):
            r["libriA"] = evaluate(m, *E["libriA"], dev); per = r["libriA"]["PER"]
            st = {k: v.detach().clone() for k, v in m.state_dict().items()}
            if best is None or per < best[0]: best, bad = (per, ep + 1, st), 0
            else: bad += 1
            snaps = sorted(snaps + [(per, ep + 1, st)], key=lambda t: t[0])[:3]
        print(json.dumps(r), flush=True); log.append(r)
        torch.save(dict(cfg=cfg, model=m.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), rng=rng.bit_generator.state, trng=torch.get_rng_state(),
                        crng=torch.cuda.get_rng_state(), best=best, bad=bad, snaps=snaps, log=log, ep=ep + 1), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        if bad >= PATIENCE: break
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
    if sys.argv[1:] == ["selftest_reward"]: selftest_reward(); sys.exit()
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    train(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 60)
