"""base6 = the repaired conventional comparator (docs/PLAN.md section 3, Phase 1). CALIBRATION. GPU <= 1 GB (GPU protocol).
Changes vs base5_next.py (everything else imported unchanged: features, causal norm, SpecAugment, speeds, GRU size search, edit / collapse):
  1. PADDING FIX in every cell: invalid positions are zeroed after the subsampler, after each LayerNorm and after each block, so a clip padded in a
     batch gives the same outputs as the clip alone (asserted at start on 8 libriA clips, atol 1e-4).
  2. SCHEDULE: every epoch's speed draw and batch list are planned up front with their own rng, so OneCycle total_steps = the real number of
     updates; no early stop; the END POINT is the model (plus a curve of libriA + cleanS every EVAL_EVERY epochs).
  3. Options (env): RIGHT (look-ahead blocks, base5 default 2), NORM = causal (3 s causal, base5) | fixed (per-band mean / sd of the original-speed
     training frames, (x - mu) / sd.clamp_min(0.1), nothing after it), EPOCHS (argv 3).
Report at the end (fp32, greedy, CPU-equal decoding): own training speakers WITHOUT augmentation (300 random fit utts), cleanS, libriA, libriB,
testother, author2, him. cleanT is never scored here.
Usage: python3 -I base6.py <tag> <seed> <epochs>     (outputs phones_out/base6/<tag>_s<seed>.{pt,json,ckpt})"""
import os
import json, os, sys, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, torch, torch.nn as nn
import base5_next as B

NORM = os.environ.get("NORM", "causal"); assert NORM in ("causal", "fixed"), NORM
RIGHT = B.RIGHT; K = B.K; OUT = D + "/phones_out/base6"; EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 6))
# Resume checkpoints (8 MB) go to scratch storage, every CKPT_EVERY epochs: the author 17 Nov, constant writers must not hit the project disk (btrfs backs up every write).
# Final .pt / .json (small, written once) stay in OUT.
CKDIR = os.environ.get("CKDIR", os.environ.get("SCRATCH_DIR", "scratch") + "/base6"); CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 10))


class Block6(B.Block):
    def forward(s, h, mask=None):
        z = s.ln(h)
        if mask is not None: z = z * mask
        z = nn.functional.pad(z.transpose(1, 2), (K - 1 - RIGHT, RIGHT))
        out = h + s.drop(s.pw(nn.functional.gelu(s.dw(z).transpose(1, 2))))
        return out * mask if mask is not None else out


class Base6(B.Base):
    def __init__(s, d):
        super().__init__(d); s.blocks = nn.ModuleList([Block6(d) for _ in range(4)])

    def trunk(s, x, lens=None):
        h = torch.relu(s.sub(nn.functional.pad(x.transpose(1, 2), (2, 2)))).transpose(1, 2); mask = None
        if lens is not None:
            ol = (torch.as_tensor(lens, device=x.device) + 1) // 2
            mask = (torch.arange(h.shape[1], device=x.device)[None] < ol[:, None]).to(h.dtype)[..., None]; h = h * mask
        L = []
        for b in s.blocks: h = b(h, mask); L.append(h)
        return L, s.rnn(h)[0]

    def forward(s, x, lens=None): return s.out(s.trunk(x, lens)[1]).float().log_softmax(-1)


MU = SD = None
MIX = os.environ.get("MIX", "A"); assert MIX in ("A", "B"), MIX


# Split lock (MASTER_PLAN Part C v6.3 C2.0, 20 Nov): training scripts score VALIDATION sets only. Exposed sets need LEGACY_EVAL=1 (old-run
# reproduction only); sealed test sets (cleanT, hard*) are never loaded here: they are scored only by step0/score_locked.py.
VALID_SETS = ("fit", "cleanS", "libriA"); EXPOSED_SETS = ("libriB", "testother", "devother", "author2", "author")   # v6.4: devother is exposed too
EVALSETS = tuple(s for s in os.environ.get("EVALSETS", ",".join(VALID_SETS)).split(",") if s)
assert all(s in VALID_SETS or (s in EXPOSED_SETS and os.environ.get("LEGACY_EVAL") == "1") for s in EVALSETS), ("eval allowlist", EVALSETS)
assert "libriA" in EVALSETS and "cleanS" in EVALSETS, "curve + selection sets are required"
DRAW = os.environ.get("DRAW", ""); SUB_H = float(os.environ.get("SUB_H", 1.0))


def draw_indices(d, hours=SUB_H):
    # Plan 2-V (18 Nov, reviewer): speakers of fit + cal in a seeded random order (rng 500 + d), taken whole until their speed-1.0 audio reaches
    # `hours`; returns the utterance indices (fit then cal order, the order of V) and the speakers. Same indices at every speed.
    SPK = json.load(open(f"{B.CACHE}/base5_speakers.json")); cf = SPK["fit"] + SPK["cal"]
    lens = np.concatenate([np.load(f"{B.CACHE}/base5_{s_}.npz")["lens"] for s_ in ("fit", "cal")]); assert len(lens) == len(cf)
    order = np.random.default_rng(500 + int(d)).permutation(sorted(set(cf))); out, spk, fr = [], [], 0
    for s in order:
        if fr >= hours * 360000: break
        ix = [i for i, c in enumerate(cf) if c == s]; out += ix; spk.append(str(s)); fr += int(lens[ix].sum())
    return sorted(out), spk, round(fr / 360000, 3)


def mix_b(V):
    # Plan 1c arm B: 2.7 h clean (23 of the 40 fit+cal speakers, seeded) + 2.7 h hard (the 23 dev-other speakers not in libriA / libriB),
    # equal frames (each = half of arm A's original-speed frames); the same utterances at every speed.
    # speaker per utterance from cache/base5_speakers.json (bb142.labelled order; the GPU venv cannot import bb142: no scipy)
    spk = lambda f: os.path.basename(str(f)).split("-")[0]; SPK = json.load(open(f"{B.CACHE}/base5_speakers.json"))
    cf = SPK["fit"] + SPK["cal"]; assert len(cf) == len(V["1.0"][0]), (len(cf), len(V["1.0"][0]))
    test = set(SPK["libriA"]) | set(SPK["libriB"])
    dm = np.load(f"{B.CACHE}/base5_devother_meta.npz"); df = [spk(f) for f in dm["files"]]
    r = np.random.default_rng(1234); half = sum(len(x) for x in V["1.0"][0]) // 2
    hard_spk = sorted(set(df) - test); assert len(hard_spk) == 23, len(hard_spk)
    clean_spk = sorted(r.choice(sorted(set(cf)), 23, replace=False).tolist())

    def pick(ids, lens):
        out, fr = [], 0
        for i in r.permutation(ids):
            if fr >= half: break
            out.append(int(i)); fr += int(lens[i])
        assert fr >= half, ("not enough speech", fr, half)
        return sorted(out)
    ci = pick([i for i, s_ in enumerate(cf) if s_ in clean_spk], [len(x) for x in V["1.0"][0]])
    hi = pick([i for i, s_ in enumerate(df) if s_ in hard_spk], dm["lens"])
    W, X1, Y1 = {}, None, None
    for sp in V:
        Xh, Yh = B.load_mm("devother" if sp == "1.0" else f"devother_sp{sp}"); assert len(Xh) == len(df) == len(Yh)
        if sp == "1.0": X1, Y1 = Xh, Yh
        else:
            for i in hi[:50]:
                assert np.array_equal(Yh[i], Y1[i]); assert abs(len(Xh[i]) * float(sp) / len(X1[i]) - 1) < 0.05, (sp, i)
        W[sp] = ([V[sp][0][i] for i in ci] + [torch.from_numpy(np.asarray(Xh[i].a, np.float32)) for i in hi],
                 [V[sp][1][i] for i in ci] + [np.asarray(Yh[i], np.int64) for i in hi])
    info = dict(clean_spk=clean_spk, hard_spk=hard_spk, n_clean=len(ci), n_hard=len(hi),
                clean_h=round(sum(len(V["1.0"][0][i]) for i in ci) / 360000, 3), hard_h=round(sum(int(dm["lens"][i]) for i in hi) / 360000, 3))
    assert not set(hard_spk) & test and not set(clean_spk) & test
    return W, info


def norm(x):
    if NORM == "causal": return B.cnorm(x)
    return (x - MU.to(x.device)) / SD.to(x.device).clamp_min(0.1)


def evaluate(m, X, Y, dev, n=None, items=False):
    m.eval(); e = t = h = 0; it = []; ix = range(len(X)) if n is None else np.random.default_rng(7).choice(len(X), n, replace=False)
    with torch.no_grad():
        for i in ix:
            hyp = B.collapse(m(norm(X[i].float().to(dev))[None])[0].argmax(-1).tolist()); y = list(map(int, Y[i]))
            ei = B.edit(hyp, y); e += ei; t += len(y); h += len(hyp); it.append([int(i), int(ei), len(y), len(hyp)])
    r = dict(PER=round(100 * e / t, 2), hyp_ref=round(h / t, 3))
    if items:  # per-item counts (C9 5); totals must equal the item sums
        assert sum(x[1] for x in it) == e and sum(x[2] for x in it) == t and sum(x[3] for x in it) == h and len(it) == len(ix)
        r.update(e=int(e), t=int(t), h=int(h), n_items=len(it), items=it)
    return r


def pad_check(m, X, dev):  # alone == padded, the property the fix must give
    # Run on a CPU float64 copy: GPU fp32 kernels differ between batched and single calls by ~1e-4 (fixed-norm smoke 17 Nov),
    # while on CPU fp64 the fixed model gives 0 and the mask-removed model 0.055-0.067 (padcheck_diag.py), so 1e-9 separates a leak from noise.
    import copy
    m = copy.deepcopy(m).cpu().double().eval()
    with torch.no_grad():
        xs = [norm(x.float()).double() for x in X]; lens = [len(x) for x in xs]
        bat = m(nn.utils.rnn.pad_sequence(xs, batch_first=True), lens)
        for i, x in enumerate(xs):
            a = m(x[None], [len(x)])[0]; b = bat[i, :len(a)]; d = float((a - b).abs().max())
            assert d < 1e-9, f"padding fix broken: clip {i} max diff {d}"
    return True


def train(tag, seed, epochs):
    global MU, SD
    dev = torch.device("cuda:0"); torch.cuda.set_per_process_memory_fraction(B.ALLOC_MIB / 8192, 0)
    torch.manual_seed(seed); rng = np.random.default_rng(seed); prng = np.random.default_rng(10_000 + seed)
    P = int(os.environ.get("PARAMS", 655924)); d = B.build(P); m = Base6(d).to(dev)
    name = f"{tag}_s{seed}"; os.makedirs(OUT, exist_ok=True); os.makedirs(CKDIR, exist_ok=True); assert not os.path.exists(f"{OUT}/{name}.json"), "output exists: never mix runs"
    V = {}
    for sp in ["1.0"] + B.SPEEDS:
        X, Y = [], []
        for s_ in ("fit", "cal"): x, y = B.load(s_ if sp == "1.0" else f"{s_}_sp{sp}"); X += x; Y += y
        V[sp] = (X, Y)
    MIXINFO = None
    if DRAW != "":
        assert MIX == "A"; ix, spk, hh = draw_indices(DRAW); V = {k: ([v[0][i] for i in ix], [v[1][i] for i in ix]) for k, v in V.items()}
        MIXINFO = dict(draw=int(DRAW), sub_h=SUB_H, speakers=spk, hours=hh, utts=len(ix)); print(json.dumps(dict(draw=MIXINFO)), flush=True)
    if MIX == "B":
        V, MIXINFO = mix_b(V); print(json.dumps(dict(mix=MIXINFO)), flush=True)
    if NORM == "fixed":
        allx = torch.cat(V["1.0"][0]).float(); MU, SD = allx.mean(0), allx.std(0)
    Y = V["1.0"][1]; N = len(Y); E = {s_: B.load(s_) for s_ in EVALSETS if s_ != "fit"}; E["fit"] = V["1.0"]
    maxfr = int(os.environ.get("MAXFR", 24000)); plans = []
    for ep in range(epochs):  # the whole schedule, planned up front
        sp = prng.choice(list(V.keys()), size=N); order = np.argsort([len(V[sp[i]][0][i]) for i in range(N)]); bat, cur, fr = [], [], 0
        for i in order:
            l = len(V[sp[i]][0][i])
            if cur and fr + l > maxfr: bat.append(cur); cur, fr = [], 0
            cur.append(i); fr += l
        bat.append(cur); plans.append((sp, [bat[j] for j in prng.permutation(len(bat))]))
    total = sum(len(p[1]) for p in plans)
    cfg = dict(tag=name, seed=seed, d=d, params=B.nparams(m), RIGHT=RIGHT, lookahead_ms=20 + 4 * RIGHT * 20 + 12.5, NORM=NORM, MIX=MIX, FEAT=B.FEAT, WARP=B.WARP, DRAW=DRAW, AUG=B.AUG, epochs=epochs,
               total_steps=total, maxfr=maxfr, lr=B.LR, speeds=["1.0"] + B.SPEEDS, train_h=round(sum(len(x) for x in V["1.0"][0]) / 360000, 3))
    print(json.dumps(dict(config=cfg)), flush=True)
    pad_check(m, E["libriA"][0][:8], dev); B.stream_check(m, E["libriA"][0][0], dev) if NORM == "causal" else None
    print(json.dumps(dict(pad_check="ok")), flush=True)
    opt = torch.optim.AdamW(m.parameters(), B.LR, weight_decay=1e-2); ctc = nn.CTCLoss(blank=B.NPH, zero_infinity=True)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, B.LR, total_steps=total, pct_start=0.1)
    log, t0, ep0, ck = [], time.time(), 0, f"{CKDIR}/{name}.ckpt"
    if os.path.exists(ck):
        S = torch.load(ck, map_location=dev, weights_only=False); assert S["cfg"] == cfg, "resume with a different config"
        m.load_state_dict(S["model"]); opt.load_state_dict(S["opt"]); sch.load_state_dict(S["sch"]); rng.bit_generator.state = S["rng"]
        torch.set_rng_state(S["trng"].cpu()); torch.cuda.set_rng_state(S["crng"].cpu()); log, ep0 = S["log"], S["ep"]; t0 -= log[-1]["secs"] if log else 0
    steps = sum(len(p[1]) for p in plans[:ep0])
    for ep in range(ep0, epochs):
        m.train(); sp, bats = plans[ep]; tot = 0.0
        for ix in bats:
            xs_raw = [V[sp[i]][0][i] for i in ix]; lens = [len(x) for x in xs_raw]
            xs = nn.utils.rnn.pad_sequence([norm(B.dials(x.float().to(dev), rng) if B.AUG else x.float().to(dev)) for x in xs_raw], batch_first=True); xs = B.specaug(xs, lens, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L, g = m.trunk(xs, lens)
            lp = m.out(g.float()).log_softmax(-1); flen = torch.tensor([(l + 1) // 2 for l in lens]); tlen = torch.tensor([len(Y[i]) for i in ix])
            loss = ctc(lp.transpose(0, 1), torch.from_numpy(np.concatenate([Y[i] for i in ix])), flen, tlen)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step(); steps += 1; tot += loss.item() * len(ix)
        r = dict(epoch=ep + 1, steps=steps, lr=float(f"{opt.param_groups[0]['lr']:.3g}"), loss=round(tot / N, 4), secs=round(time.time() - t0), gpu_mib=B.gpu_mib())
        assert r["gpu_mib"] <= B.LIMIT_MIB, f"GPU memory {r['gpu_mib']} MiB > {B.LIMIT_MIB}"
        # E1 (20 Nov, MASTER_PLAN C 4): EVAL_N = number of curve points per run, spread evenly over the epochs; absent = old rule.
        # Evaluation draws no training randomness, so the training path is unchanged; only fewer curve points are taken.
        _evn = os.environ.get("EVAL_N")
        if (((ep + 1) * int(_evn)) // epochs > (ep * int(_evn)) // epochs if _evn else (ep + 1) % EVAL_EVERY == 0) or ep + 1 == epochs:
            r["libriA"] = evaluate(m, *E["libriA"], dev, 100); r["cleanS"] = evaluate(m, *E["cleanS"], dev, 100)
        log.append(r); print(json.dumps(r), flush=True)
        if (ep + 1) % CKPT_EVERY and ep + 1 != epochs: continue
        torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), rng=rng.bit_generator.state, trng=torch.get_rng_state(),
                        crng=torch.cuda.get_rng_state(), log=log, ep=ep + 1, cfg=cfg), ck + ".tmp"); os.replace(ck + ".tmp", ck)
    assert steps == total == sch.last_epoch, (steps, total, sch.last_epoch)
    torch.save(m.state_dict(), f"{OUT}/{name}.pt"); R = dict(config=cfg, mix=MIXINFO, log=log, sets={})
    for s_ in ("fit",) + tuple(s for s in EVALSETS if s != "fit"):
        R["sets"][s_] = evaluate(m, *E[s_], dev, 300 if s_ == "fit" else None, items=True)
        print(s_, {k: v for k, v in R["sets"][s_].items() if k != "items"}, flush=True)
    json.dump(R, open(f"{OUT}/{name}.json", "w"), indent=1)


if __name__ == "__main__":
    torch.set_num_threads(2); train(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
