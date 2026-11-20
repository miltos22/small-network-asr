"""2-II on the base6 trunk (docs/PLAN.md Phase 2 row 2-II). Written 18 Nov 2026 by the agent (reviewer 5.5).
Arms (same audio, same base6 recipe: padding fix, planned OneCycle schedule, no early stop, causal 3 s norm, RIGHT from env (12 = frozen base6)):
  S  his structured readout = struct2c "full" + SKIP on a 232-wide base6 trunk: 18 split heads (12-unit middles) on trunk layers 1-4 hand their
     middles down to 54 specialists (53 tokens + SIL) on their own tree paths (16-unit middle, one logit each) + blank logit + direct linear readout
     trunk-top -> 55 logits; one joint softmax. Teaching loss on the heads (AUXW 0.3, MFA frame labels).
  G  the generic control: same 232 trunk, same 55 outputs, same teaching heads (training only, not exported), readout
     direct(g) + Linear(GELU(Linear(concat(g, L1..L4)))), hidden width H chosen so the exported bytes match S.
  P  plain base6 widened to the same exported bytes (39 phones + blank, no heads).
Bytes = the T8 export of phase1_export.py (torch dynamic int8 of Linear + GRU, state_dict file), training-only modules removed; for S the masked
entries (structural zeros, fixed by the tree, no index needed) are subtracted at 1 byte each. Known answer: plain base6 d 232 -> 1,010,565.
The tree code is executed from the frozen logs_2c/run2/struct2c_AS_RUN.py (ast extract), not copied, so the wiring is the one that ran on 15-16 Nov.
Usage: python3 -I base6s.py <arm S|G|P> <tag> <seed> <epochs>   env: DEV (cuda:0), SMOKE=N (N utts, CPU check), DRAW/SUB_H (2-V),
       WARP (base5_next), AUXW, ALLOC_MIB.     base6s.py bytes  -> prints the budget table.     base6s.py selftest"""
import os
import ast, io, json, os, sys, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, torch, torch.nn as nn
import base5_next as B
import base6 as B6
import labels2c as L2

AUXW = float(os.environ.get("AUXW", 0.3)); W_S = 232
FROZEN = D + "/logs_2c/run2/struct2c_AS_RUN.py"
_WANT = {"C", "NPH", "SILF", "NT", "NS", "BLANKV", "SPL", "LAYER", "NH", "HM", "SM", "LEFT", "RIGHTS", "PATHS", "HT",
         "leaves", "path", "tok_family", "Heads", "decode39"}
_src = ast.parse(open(FROZEN).read()); _keep = []
for node in _src.body:
    names = set()
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)): names = {node.name}
    elif isinstance(node, ast.Assign):
        for t in node.targets: names |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "g": names = {"HT"}  # the HT fill loop
    elif isinstance(node, ast.Assert): names = {"_assert"}
    if names & (_WANT | {"_assert"}): _keep.append(node)
T = dict(B=B, L2=L2, np=np, nn=nn, torch=torch); exec(compile(ast.Module(body=_keep, type_ignores=[]), FROZEN, "exec"), T)
assert all(k in T for k in _WANT), sorted(_WANT - set(T))
NT, NS, NH, HM, SM, PATHS, HT, BLANKV = (T[k] for k in ("NT", "NS", "NH", "HM", "SM", "PATHS", "HT", "BLANKV"))
NOUT = NT + 1  # 55


class Trunk(B6.Base6):  # base6 trunk (padding fix) without its output layer
    def __init__(s, w): super().__init__(w); del s.out


class ArmS(nn.Module):
    def __init__(s, w):
        super().__init__(); s.t = Trunk(w); s.heads = T["Heads"](w); s.spec1 = nn.Linear(w, SM * NS); s.specm = nn.Linear(HM * NH, SM * NS, bias=False)
        mask = torch.zeros(SM * NS, HM * NH)
        for k, pth in enumerate(PATHS):
            for g in pth: mask[SM * k:SM * (k + 1), HM * g:HM * (g + 1)] = 1
        s.register_buffer("mask", mask); s.specm.weight.data *= mask; s.specm.weight.register_hook(lambda gr: gr * s.mask)
        s.spec2 = nn.Linear(SM * NS, NS); m2 = torch.zeros(NS, SM * NS)
        for k in range(NS): m2[k, SM * k:SM * (k + 1)] = 1
        s.register_buffer("mask2", m2); s.spec2.weight.data *= m2; s.spec2.weight.register_hook(lambda gr: gr * s.mask2)
        s.blank = nn.Linear(w, 1); s.skip = nn.Linear(w, NOUT)

    def top(s, L, g):
        hl, mids = s.heads(L)
        sp = s.spec2(nn.functional.gelu(s.spec1(g) + s.specm(torch.cat(mids, -1))))
        return torch.cat([sp, s.blank(g)], -1) + s.skip(g), hl

    def logits(s, x, lens=None): return s.top(*s.t.trunk(x, lens))

    def forward(s, x, lens=None): return s.logits(x, lens)[0].float().log_softmax(-1)


class ArmG(nn.Module):
    def __init__(s, w, H):
        super().__init__(); s.t = Trunk(w); s.heads = T["Heads"](w)  # heads: teaching only
        s.direct = nn.Linear(w, NOUT); s.h1 = nn.Linear(5 * w, H); s.h2 = nn.Linear(H, NOUT)

    def top(s, L, g):
        hl = s.heads(L)[0]
        return s.direct(g) + s.h2(nn.functional.gelu(s.h1(torch.cat([g] + L, -1)))), hl

    def logits(s, x, lens=None): return s.top(*s.t.trunk(x, lens))

    def forward(s, x, lens=None): return s.logits(x, lens)[0].float().log_softmax(-1)


def export_bytes(m):  # T8 as phase1_export.py, training-only parts removed, S's structural zeros subtracted
    import copy
    c = copy.deepcopy(m).cpu().eval(); zeros = 0
    if isinstance(c, ArmG): c.heads = None
    if isinstance(c, ArmS): zeros = int((c.mask == 0).sum() + (c.mask2 == 0).sum())
    q = torch.ao.quantization.quantize_dynamic(c, {nn.GRU, nn.Linear}, dtype=torch.qint8)
    b = io.BytesIO(); torch.save(q.state_dict(), b)
    # masks are buffers of S: not needed in a compact file (the tree fixes them); their fp32 bytes are removed too
    if isinstance(c, ArmS): zeros += 4 * (c.mask.numel() + c.mask2.numel())
    # shape ablations (base6sx.py) declare their own structural zeros + mask buffers
    if hasattr(c, "struct_zero_bytes"): zeros += c.struct_zero_bytes()
    return len(b.getvalue()) - zeros


def budget():
    MS = export_bytes(ArmS(W_S)); out = dict(S=dict(w=W_S, bytes=MS))
    best = None
    for H in range(8, 2048, 8):
        n = export_bytes(ArmG(W_S, H))
        if best is None or abs(n - MS) < abs(best[1] - MS): best = (H, n)
        if n > MS: break
    out["G"] = dict(w=W_S, H=best[0], bytes=best[1]); best = None
    for d in range(200, 400, 2):
        n = export_bytes(B6.Base6(d))
        if best is None or abs(n - MS) < abs(best[1] - MS): best = (d, n)
        if n > MS: break
    out["P"] = dict(d=best[0], bytes=best[1]); out["base6_232"] = export_bytes(B6.Base6(232))
    return out


MAKE_HOOK = None  # base6sx.py (shape ablations) sets this; None = the original arms


def make(arm, bud):
    if MAKE_HOOK is not None: return MAKE_HOOK(arm, bud)
    if arm == "S": return ArmS(bud["S"]["w"])
    if arm == "G": return ArmG(bud["G"]["w"], bud["G"]["H"])
    return B6.Base6(bud["P"]["d"])


def evaluate(arm, m, X, Y, dev, n=None, items=False):
    m.eval(); e = t = h = 0; it = []; ix = range(len(X)) if n is None else np.random.default_rng(7).choice(len(X), min(n, len(X)), replace=False)
    with torch.no_grad():
        for i in ix:
            ids = m(B6.norm(X[i].float().to(dev))[None])[0].argmax(-1).tolist()
            hyp = B.collapse(ids) if arm == "P" else T["decode39"]("full", ids); y = list(map(int, Y[i]))
            ei = B.edit(hyp, y); e += ei; t += len(y); h += len(hyp); it.append([int(i), int(ei), len(y), len(hyp)])
    r = dict(PER=round(100 * e / t, 2), hyp_ref=round(h / t, 3))
    if items:  # per-item counts (C9 5); totals must equal the item sums
        assert sum(x[1] for x in it) == e and sum(x[2] for x in it) == t and sum(x[3] for x in it) == h and len(it) == len(ix)
        r.update(e=int(e), t=int(t), h=int(h), n_items=len(it), items=it)
    return r


def selftest():
    bud = budget(); print(json.dumps(bud), flush=True)
    assert bud["base6_232"] == 1010565 or abs(bud["base6_232"] - 1010565) < 2000, ("T8 known answer", bud["base6_232"])
    for a in ("S", "G", "P"): assert abs(bud[a]["bytes"] - bud["S"]["bytes"]) / bud["S"]["bytes"] < 0.01, (a, bud)
    m = ArmS(64); x = torch.randn(2, 50, B.NMEL)
    with torch.no_grad():
        L, g = m.t.trunk(x); hl, mids = m.heads(L); k = L2.TID["IY"]
        f = lambda md: m.spec2(nn.functional.gelu(m.spec1(g) + m.specm(torch.cat(md, -1))))[..., k]; b0 = f(mids)
        off = [q for q in range(NH) if q not in PATHS[k]][0]; m2 = list(mids); m2[off] = m2[off] + 5.0; assert torch.allclose(b0, f(m2)), "off-path leak"
        on = PATHS[k][-1]; m3 = list(mids); m3[on] = m3[on] + 5.0; assert not torch.allclose(b0, f(m3)), "on-path head ignored"
    assert PATHS[L2.TID["IY"]] == [0, 1, 2, 3, 4, 5, 6] and PATHS[L2.SIL] == [0] and PATHS[L2.TID["TS"]] == [0, 1, 10, 11]
    assert NOUT == 55 and BLANKV == 54
    print("base6s selftest ok", flush=True); return bud


def train(arm, tag, seed, epochs):
    assert arm in ("S", "G", "P")
    dev = torch.device(os.environ.get("DEV", "cuda:0")); SMOKE = int(os.environ.get("SMOKE", 0)); STOP_AFTER = int(os.environ.get("STOP_AFTER", 0))
    if dev.type == "cuda": torch.cuda.set_per_process_memory_fraction(B.ALLOC_MIB / 8192, 0)
    torch.manual_seed(seed); rng = np.random.default_rng(seed); prng = np.random.default_rng(10_000 + seed)
    bud = json.load(open(os.environ.get("BUDF", D + "/logs_base6/base6s_budget.json"))); m = make(arm, bud).to(dev); nbytes = export_bytes(m)
    assert abs(nbytes - bud[arm]["bytes"]) < 1, (nbytes, bud[arm])
    name = f"{tag}_s{seed}"; os.makedirs(B6.OUT, exist_ok=True); os.makedirs(B6.CKDIR, exist_ok=True)
    assert not os.path.exists(f"{B6.OUT}/{name}.json"), "output exists: never mix runs"
    V = {}
    for sp in ["1.0"] + B.SPEEDS:
        X, Y = [], []
        for s_ in ("fit", "cal"): x, y = B.load(s_ if sp == "1.0" else f"{s_}_sp{sp}"); X += x; Y += y
        V[sp] = (X, Y)
    LV, F0 = [], []
    for s_ in ("fit", "cal"):
        d_ = np.load(f"{B.CACHE}/v2c_{s_}.npz"); LV += [d_[f"y{i}"].astype(np.int64) for i in range(int(d_["n"]))]
        st = np.r_[0, np.cumsum(d_["flens"])]; F0 += [d_["f"][a:b] for a, b in zip(st[:-1], st[1:])]
    assert len(LV) == len(V["1.0"][1]) and all(len(f) == len(x) for f, x in zip(F0, V["1.0"][0])), "frame labels do not match the features"
    assert all(L2.to39(a.tolist()) == b.tolist() for a, b in zip(LV, V["1.0"][1])), "variant labels do not map back to the base labels"
    ix = list(range(len(LV))); info = None
    if B6.DRAW != "":
        ix, spk, hh = B6.draw_indices(B6.DRAW); info = dict(draw=int(B6.DRAW), sub_h=B6.SUB_H, speakers=spk, hours=hh, utts=len(ix))
    if SMOKE: ix = ix[:SMOKE]
    V = {k: ([v[0][i] for i in ix], [v[1][i] for i in ix]) for k, v in V.items()}; LV = [LV[i] for i in ix]; F0 = [F0[i] for i in ix]
    Y = V["1.0"][1] if arm == "P" else LV; N = len(Y); aux = arm in ("S", "G") and AUXW > 0
    evals = tuple(s for s in B6.EVALSETS if s != "fit"); E = {s_: B.load(s_) for s_ in evals}; E["fit"] = V["1.0"]
    if SMOKE: E = {k: (v[0][:6], v[1][:6]) for k, v in E.items()}
    maxfr = int(os.environ.get("MAXFR", 24000)); plans = []
    for ep in range(epochs):
        sp = prng.choice(list(V.keys()), size=N); order = np.argsort([len(V[sp[i]][0][i]) for i in range(N)]); bat, cur, fr = [], [], 0
        for i in order:
            l = len(V[sp[i]][0][i])
            if cur and fr + l > maxfr: bat.append(cur); cur, fr = [], 0
            cur.append(i); fr += l
        bat.append(cur); plans.append((sp, [bat[j] for j in prng.permutation(len(bat))]))
    total = sum(len(p[1]) for p in plans)
    import hashlib as _hl  # C9 4: pairing evidence recorded in every run (Q2 (c) pair check, step0/q2c/runner_q2c.py)
    _ph = _hl.sha256()
    for _sp, _bat in plans:
        _ph.update(",".join(map(str, _sp)).encode())
        for _b in _bat: _ph.update(np.asarray(_b, np.int64).tobytes())
    _th = _hl.sha256(); [_th.update(np.asarray(y_, np.int64).tobytes()) for y_ in Y]
    _mk = _hl.sha256(m.mask.detach().cpu().numpy().tobytes()).hexdigest() if hasattr(m, "mask") else None
    _rw = os.environ.get("RWIRE_SEED")
    if os.environ.get("Q2C_PAIR") == "1":
        assert arm == "S" and os.environ.get("Q2C_ARM") in ("S", "R"), ("Q2C arm", arm, os.environ.get("Q2C_ARM"))
        assert ("RWIRE sigma=" in FROZEN) == (os.environ["Q2C_ARM"] == "R"), "Q2C_ARM does not match the wiring in force"
        assert os.environ["Q2C_ARM"] == "S" or (_rw is not None and int(_rw) == seed), ("RWIRE_SEED must equal the seed", _rw, seed)
    cfg = dict(script="base6s.py", arm=arm, tag=name, seed=seed, budget=bud[arm], bytes=nbytes, params=B.nparams(m), RIGHT=B6.RIGHT,
               lookahead_ms=20 + 4 * B6.RIGHT * 20 + 12.5, NORM=B6.NORM, FEAT=B.FEAT, WARP=B.WARP, DRAW=B6.DRAW, AUG=B.AUG, SMOKE=SMOKE, auxw=AUXW if aux else 0.0,
               outputs=NOUT if arm != "P" else B.NPH + 1, epochs=epochs, total_steps=total, maxfr=maxfr, lr=B.LR, speeds=["1.0"] + B.SPEEDS,
               train_h=round(sum(len(x) for x in V["1.0"][0]) / 360000, 3), frozen_tree=FROZEN,
               plan_sha256=_ph.hexdigest(), targets_sha256=_th.hexdigest(), train_ix_sha256=_hl.sha256(np.asarray(ix, np.int64).tobytes()).hexdigest(),
               mask_sha256=_mk, RWIRE_SEED=_rw, Q2C_ARM=os.environ.get("Q2C_ARM"))
    print(json.dumps(dict(config=cfg, draw=info)), flush=True)
    B6.pad_check(m, E["libriA"][0][:8], dev); B.stream_check(m, E["libriA"][0][0], dev); print(json.dumps(dict(pad_stream_check="ok")), flush=True)
    opt = torch.optim.AdamW(m.parameters(), B.LR, weight_decay=1e-2); blank = BLANKV if arm != "P" else B.NPH
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True); bce = nn.BCEWithLogitsLoss(reduction="none"); HTd = torch.from_numpy(HT).to(dev)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, B.LR, total_steps=total, pct_start=0.1)

    def ftarget(i, sp, T_out):  # struct2c: label at input frame 2j of the speed-changed utterance -> original frame round(2j * speed)
        f = F0[i]; idx = np.minimum(np.round(np.arange(T_out) * 2 * float(sp)).astype(np.int64), len(f) - 1)
        return torch.from_numpy(f[idx].astype(np.int64))
    log, t0, ep0, ck = [], time.time(), 0, f"{B6.CKDIR}/{name}.ckpt"
    # E1 harness (v6.3, review r3 C5.4): STOP_AFTER runs are isolated: fresh start, no resume, no checkpoint writes; TRACE = per-update log
    # (loss / lr as repr) + per-pass sha256 of the state_dict and of both RNG states; BROKEN_E1=1 = the deliberate failure (eval draws RNG).
    assert not STOP_AFTER or not os.path.exists(ck), 'STOP_AFTER run must start fresh (checkpoint exists)'
    assert not STOP_AFTER or not os.listdir(B6.CKDIR), ('STOP_AFTER run needs an EMPTY checkpoint dir', B6.CKDIR)  # v6.4 r4 C5.4
    assert not STOP_AFTER or not os.path.exists(f'{B6.OUT}/{name}.json'), 'STOP_AFTER run must not overwrite an earlier result'
    TRACE = os.environ.get('TRACE'); BROKEN = os.environ.get('BROKEN_E1') == '1'; trf = open(TRACE, 'x') if TRACE else None
    import hashlib
    def _hopt(o):   # v6.4: the WHOLE optimizer state (every parameter's step / exp_avg / exp_avg_sq + param_groups)
        sd = o.state_dict(); hh = hashlib.sha256(repr(sd['param_groups']).encode())
        for i in sorted(sd['state']):
            for k in sorted(sd['state'][i]): v = sd['state'][i][k]; hh.update(f'{i}{k}'.encode() + (v.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(v) else repr(v).encode()))
        return hh.hexdigest()[:16]
    def _h(sd): hh = hashlib.sha256(); [hh.update(k.encode() + v.detach().cpu().contiguous().numpy().tobytes()) for k, v in sd.items()]; return hh.hexdigest()[:16]
    if os.path.exists(ck) and not SMOKE:
        S_ = torch.load(ck, map_location=dev, weights_only=False); assert S_["cfg"] == cfg, "resume with a different config"
        m.load_state_dict(S_["model"]); opt.load_state_dict(S_["opt"]); sch.load_state_dict(S_["sch"]); rng.bit_generator.state = S_["rng"]
        torch.set_rng_state(S_["trng"].cpu()); torch.cuda.set_rng_state(S_["crng"].cpu()); log, ep0 = S_["log"], S_["ep"]; t0 -= log[-1]["secs"] if log else 0
    steps = sum(len(p[1]) for p in plans[:ep0])
    for ep in range(ep0, epochs):
        m.train(); sp, bats = plans[ep]; tot = tota = 0.0
        for bx in bats:
            xs_raw = [V[sp[i]][0][i] for i in bx]; lens = [len(x) for x in xs_raw]
            xs = nn.utils.rnn.pad_sequence([B6.norm(B.dials(x.float().to(dev), rng) if B.AUG else x.float().to(dev)) for x in xs_raw], batch_first=True); xs = B.specaug(xs, lens, rng)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                if arm == "P": L, g = m.trunk(xs, lens); hl = None
                else: z, hl = m.logits(xs, lens)
            z = m.out(g.float()) if arm == "P" else z.float(); lp = z.log_softmax(-1)
            flen = torch.tensor([(l + 1) // 2 for l in lens]); tlen = torch.tensor([len(Y[i]) for i in bx])
            loss = ctc(lp.transpose(0, 1), torch.from_numpy(np.concatenate([Y[i] for i in bx])), flen, tlen); la = torch.zeros((), device=dev)
            if aux:
                hl = hl.float(); Tm = hl.shape[1]; lab = torch.full((len(bx), Tm), -1, dtype=torch.long)
                for r_, i in enumerate(bx): T_ = int(flen[r_]); lab[r_, :T_] = ftarget(i, sp[i], T_)
                lab = lab.to(dev); valid = lab >= 0; tg = HTd[:, lab.clamp(min=0)].permute(1, 2, 0)
                mk = (tg >= 0) & valid.unsqueeze(-1); l_ = bce(hl, tg.clamp(min=0)) * mk
                la = (l_.sum((0, 1)) / mk.sum((0, 1)).clamp(min=1)).mean()
            opt.zero_grad(); (loss + AUXW * la if aux else loss).backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
            steps += 1; tot += loss.item() * len(bx); tota += la.item() * len(bx)
            if trf: trf.write(f"u {steps} {loss.item()!r} {la.item()!r} {opt.param_groups[0]['lr']!r} {all(x.training for x in m.modules())}\n")
        r = dict(epoch=ep + 1, steps=steps, lr=float(f"{opt.param_groups[0]['lr']:.3g}"), loss=round(tot / N, 4), aux=round(tota / N, 4),
                 secs=round(time.time() - t0), gpu_mib=B.gpu_mib() if dev.type == "cuda" else 0)
        assert r["gpu_mib"] <= B.LIMIT_MIB, f"GPU memory {r['gpu_mib']} MiB > {B.LIMIT_MIB}"
        # E1 (20 Nov, MASTER_PLAN C 4): EVAL_N = number of curve points per run, spread evenly over the epochs; absent = old rule.
        # Evaluation draws no training randomness, so the training path is unchanged; only fewer curve points are taken.
        _evn = os.environ.get("EVAL_N")
        if (((ep + 1) * int(_evn)) // epochs > (ep * int(_evn)) // epochs if _evn else (ep + 1) % B6.EVAL_EVERY == 0) or ep + 1 == epochs:
            r["libriA"] = evaluate(arm, m, *E["libriA"], dev, 100); r["cleanS"] = evaluate(arm, m, *E["cleanS"], dev, 100)
            if BROKEN: torch.rand(1); rng.random()
            if trf: trf.write(f"e {ep + 1} {all(x.training for x in m.modules())}\n")   # v6.4: the evaluation event + mode after it
        log.append(r); print(json.dumps(r), flush=True)
        if trf: trf.write(f"p {ep + 1} {_h(m.state_dict())} {_hopt(opt)} {hashlib.sha256(repr(sch.state_dict()).encode()).hexdigest()[:16]} "
                          f"{hashlib.sha256(repr(rng.bit_generator.state).encode()).hexdigest()[:16]} {hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16]}\n"); trf.flush()
        # E1 test (MASTER_PLAN C5 4): STOP_AFTER = stop after this many passes, keeping the full planned schedule (calibration only).
        if STOP_AFTER and ep + 1 >= STOP_AFTER: break
        if SMOKE or STOP_AFTER or ((ep + 1) % B6.CKPT_EVERY and ep + 1 != epochs): continue
        torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), rng=rng.bit_generator.state, trng=torch.get_rng_state(),
                        crng=torch.cuda.get_rng_state(), log=log, ep=ep + 1, cfg=cfg), ck + ".tmp"); os.replace(ck + ".tmp", ck)
    assert STOP_AFTER or steps == total == sch.last_epoch, (steps, total, sch.last_epoch)
    assert not STOP_AFTER or steps == sum(len(p[1]) for p in plans[:STOP_AFTER]) == sch.last_epoch, ("stop_after", steps, sch.last_epoch)
    if arm == "S": assert (m.specm.weight[m.mask == 0] == 0).all() and (m.spec2.weight[m.mask2 == 0] == 0).all(), "mask broken"
    if not SMOKE and not STOP_AFTER: torch.save(m.state_dict(), f"{B6.OUT}/{name}.pt")
    R = dict(config=cfg, draw=info, log=log, sets={}, stop_after=STOP_AFTER, eval_n=os.environ.get("EVAL_N"))
    for s_ in ("fit",) + evals:
        R["sets"][s_] = evaluate(arm, m, *E[s_], dev, 300 if s_ == "fit" else None, items=True)
        print(s_, {k: v for k, v in R["sets"][s_].items() if k != "items"}, flush=True)
    if not SMOKE: json.dump(R, open(f"{B6.OUT}/{name}.json", "w"), indent=1)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    if sys.argv[1:] == ["selftest"]: selftest(); sys.exit()
    if sys.argv[1:] == ["bytes"]:
        b = selftest(); json.dump(b, open(D + "/logs_base6/base6s_budget.json", "w"), indent=1); sys.exit()
    train(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
