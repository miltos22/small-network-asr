"""ARM 2c (ARM_2c_DESIGN_2026-11-15.md; spec draft 4 sections 4, 4a, 4b, 4d, 6): the tree structure against the fair black box, same size
(PARAMS 655924 counted numbers), same input (raw log-mel 80 + causal 3 s norm), same data (fit+cal 5.394 h, speeds 0.9/1.0/1.1), same recipe
(base5_next.py default path: AdamW 3e-3 OneCycle 60 ep, SpecAugment, libriA selection every 2 ep, average of 3 best, patience 5, MAXFR 24000).
Arms:
  base   base5 network, 39 phones + blank (the comparator, rerun here so both sides come from one script)
  tok    base5 network, 53 tokens (39 phonemes with K, G split + 16 variants, labels2c.py) + SIL + blank: "more output tokens" alone
  lite   base5 network + 18 split heads on the trunk blocks, TRAINING ONLY (aux loss from MFA frame labels); 0 numbers at inference
  full   trunk (conv + 4 blocks + GRU, narrower) + 18 split heads (12-unit middle layers) on trunk layers 1-4 + 54 specialists (53 tokens + SIL),
         each reading the GRU output + the middle layers of the splits on its own path (16-unit middle layer, one logit) + "nothing new"
         (blank) logit; one joint softmax over 55 (the joint decision, logits not 0-1 scores: review DECISIONS 3)
  broken full with labels deranged across utterances and AUXW 0: must fail
Scoring: every hypothesis is mapped back to the 39 phonemes (SIL dropped, TS -> T S, DZ -> D Z, each other variant -> its phoneme) and scored
against the same 39-phone references as the base. Int8 = dynamic quantisation of every Linear + GRU (convs fp32, as base5).
Usage: struct2c.py ARM SEED [EPOCHS]   env: AUXW (0.3), PARAMS, OUTDIR, MAXFR, THREADS; GPU 3070 under the 1 GB limit (base5_next guard)."""
import os
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
for k in ("FEAT", "AUG", "SPKNORM", "CHAN", "REWARD", "REWARD_BLANK", "REWARD_PROJ", "BLANKPEN", "DATA", "TAIL", "LR", "RIGHT", "NORMW", "SPEEDS"):
    assert k not in os.environ, f"{k} set: arm 2c runs the default base5 path only"
import numpy as np, torch, torch.nn as nn
import base5_next as B
import labels2c as L2

C = B.CACHE; NPH = B.NPH; SILF = NPH  # frame label 39 = silence
NT = len(L2.TOK) + 1; NS = NT; BLANKV = NT  # 53 tokens + SIL = 54 classes with a specialist; blank = 54 -> 55 outputs
AUXW = float(os.environ.get("AUXW", 0.3))
# diagnostics (15 Nov 19:1x, EXPLORATION; defaults = the tested arm): SKIP=1 adds a direct linear readout trunk-top -> 55 logits next to the
# specialists (does restoring the plain path remove the specialist layer's cost?); MIDS=0 cuts the head middles out of the specialists.
SKIP = os.environ.get("SKIP", "0") == "1"; MIDS = os.environ.get("MIDS", "1") == "1"
# tree (spec 4a): split id -> (left, right); int = split, str = family (phones), "SIL" = silence frames
SPL = {0: ("SIL", 1), 1: (2, 10), 2: (3, 9), 3: (4, 8), 4: (5, "ER"), 5: (6, 7), 6: ("IY IH", "EH AE"), 7: ("UW UH", "AA AO AH"),
       8: ("AY AW", "EY OW OY"), 9: ("M N NG", "W Y L R"), 10: (11, 14), 11: (12, "CH JH"), 12: ("P B", 13), 13: ("T D", "K G"),
       14: ("HH", 15), 15: (16, 17), 16: ("S Z", "SH ZH"), 17: ("F V", "TH DH")}
LAYER = {0: 0, 1: 1, 2: 1, 10: 1, 3: 2, 4: 2, 5: 2, 9: 2, 11: 2, 14: 2, 15: 2, 6: 3, 7: 3, 8: 3, 12: 3, 13: 3, 16: 3, 17: 3}  # spec 4 rule 2
NH, HM, SM = 18, 12, 16


def leaves(n):
    if isinstance(n, int): return leaves(SPL[n][0]) | leaves(SPL[n][1])
    return {SILF} if n == "SIL" else {L2.PID[p] for p in n.split()}


assert leaves(1) == set(range(NPH)) and all(not (leaves(SPL[g][0]) & leaves(SPL[g][1])) for g in SPL)
LEFT = [leaves(SPL[g][0]) for g in range(NH)]; RIGHTS = [leaves(SPL[g][1]) for g in range(NH)]


def path(ph):  # splits from G0 down to the family of 39-phone id ph (or silence)
    out, g = [], 0
    while True:
        out.append(g); nxt = SPL[g][0] if ph in leaves(SPL[g][0]) else SPL[g][1]
        if not isinstance(nxt, int): return out
        g = nxt


def tok_family(t):  # specialist t (token id, or L2.SIL) -> the 39-phone id whose family it sits in (spec 4b, 4d)
    if t == L2.SIL: return SILF
    name = L2.TOK[t]
    return L2.PID["CH"] if name in ("TS", "DZ") else L2.TO39[t][0]


PATHS = [path(tok_family(t)) for t in range(NS)]
# head target table: HT[g, label] = 0 (left) / 1 (right) / -1 (not under this split); label 0..38 phones, 39 silence
HT = -np.ones((NH, NPH + 1), np.float32)
for g in range(NH):
    for p in LEFT[g]: HT[g, p] = 0
    for p in RIGHTS[g]: HT[g, p] = 1


class Heads(nn.Module):  # 18 split heads: trunk layer -> 12-unit middle layer (GELU) -> one logit (spec 6A, without special channels: 2a's job)
    def __init__(s, w):
        super().__init__(); s.a = nn.ModuleList([nn.Linear(w, HM) for _ in range(NH)]); s.b = nn.ModuleList([nn.Linear(HM, 1) for _ in range(NH)])

    def forward(s, Ls):
        mids = [nn.functional.gelu(s.a[g](Ls[LAYER[g]])) for g in range(NH)]
        return torch.cat([s.b[g](mids[g]) for g in range(NH)], -1), mids


class Full(nn.Module):
    def __init__(s, w):
        super().__init__()
        s.sub = nn.Conv1d(B.NMEL, w, 5, 2); s.blocks = nn.ModuleList([B.Block(w) for _ in range(4)]); s.rnn = nn.GRU(w, w, 1, batch_first=True)
        s.heads = Heads(w); s.spec1 = nn.Linear(w, SM * NS); s.specm = nn.Linear(HM * NH, SM * NS, bias=False)
        mask = torch.zeros(SM * NS, HM * NH)
        for k, pth in enumerate(PATHS):
            for g in pth: mask[SM * k:SM * (k + 1), HM * g:HM * (g + 1)] = 1 if MIDS else 0
        s.register_buffer("mask", mask); s.specm.weight.data *= mask; s.specm.weight.register_hook(lambda gr: gr * s.mask)
        s.spec2 = nn.Linear(SM * NS, NS)  # used block-diagonally: only the 16 middle units of specialist k feed logit k
        m2 = torch.zeros(NS, SM * NS)
        for k in range(NS): m2[k, SM * k:SM * (k + 1)] = 1
        s.register_buffer("mask2", m2); s.spec2.weight.data *= m2; s.spec2.weight.register_hook(lambda gr: gr * s.mask2)
        s.blank = nn.Linear(w, 1); s.skip = nn.Linear(w, NT + 1) if SKIP else None

    def trunk(s, x):
        h = torch.relu(s.sub(nn.functional.pad(x.transpose(1, 2), (2, 2)))).transpose(1, 2); Ls = []
        for b in s.blocks: h = b(h); Ls.append(h)
        return Ls, s.rnn(h)[0]

    def top(s, Ls, g):  # heads + specialists + joint-decision logits from the trunk outputs
        hl, mids = s.heads(Ls)
        sp = s.spec2(nn.functional.gelu(s.spec1(g) + s.specm(torch.cat(mids, -1))))
        z = torch.cat([sp, s.blank(g)], -1)
        return (z + s.skip(g) if s.skip is not None else z), hl

    def logits(s, x): return s.top(*s.trunk(x))

    def forward(s, x): return s.logits(x)[0].float().log_softmax(-1)


class Tok(B.Base):
    def __init__(s, d): super().__init__(d); s.out = nn.Linear(d, NT + 1)


class Lite(nn.Module):  # base5 network + training-only heads; .net is what runs on the phone
    def __init__(s, d): super().__init__(); s.net = B.Base(d); s.heads = Heads(d)

    def forward(s, x): return s.net(x)


def make(arm, w): return {"base": B.Base, "tok": Tok, "lite": Lite, "full": Full, "broken": Full}[arm](w)


def counted(m):  # numbers that run at inference: masked-out entries and training-only heads are not counted
    if isinstance(m, Lite): return B.nparams(m.net)
    n = B.nparams(m)
    if isinstance(m, Full): n -= int((m.mask == 0).sum() + (m.mask2 == 0).sum())
    return n


def build(arm, P):
    best = None
    for w in range(32, 1024, 2):
        n = counted(make(arm, w))
        if best is None or abs(n - P) < abs(best[1] - P): best = (w, n)
        if n > P: break
    assert abs(best[1] - P) / P < 0.01, (arm, best, P); return best[0]


def decode39(arm, ids):
    if arm in ("base", "lite"): return B.collapse(ids)
    out, prev = [], None
    for t in ids:
        if t != prev and t != BLANKV and t != L2.SIL: out += L2.TO39[t]
        prev = t
    return out


def evaluate(arm, m, X, Y, dev):
    m.eval(); n = e = h = 0
    with torch.no_grad():
        for x, y in zip(X, Y):
            hyp = decode39(arm, m(B.cnorm(x.to(dev))[None])[0].argmax(-1).tolist()); e += B.edit(hyp, list(y)); n += len(y); h += len(hyp)
    return dict(PER=round(e / n, 4), hyp_over_ref=round(h / n, 3))


def selftest():
    for a in ("base", "tok", "lite", "full"): print(a, "width", build(a, 655924), flush=True)
    m = Full(64); x = torch.randn(2, 50, B.NMEL); z, hl = m.logits(x); assert z.shape == (2, 25, NT + 1) and hl.shape == (2, 25, NH)
    # path wiring: specialist k's logit must not move when a head off its path changes
    k = L2.TID["IY"]; mids0 = None
    with torch.no_grad():
        Ls, g = m.trunk(x); hl, mids = m.heads(Ls); base_ = m.spec2(nn.functional.gelu(m.spec1(g) + m.specm(torch.cat(mids, -1))))[..., k]
        off = [gg for gg in range(NH) if gg not in PATHS[k]][0]; mids2 = list(mids); mids2[off] = mids2[off] + 5.0
        assert torch.allclose(base_, m.spec2(nn.functional.gelu(m.spec1(g) + m.specm(torch.cat(mids2, -1))))[..., k]), "off-path head leaks"
        on = PATHS[k][-1]; mids3 = list(mids); mids3[on] = mids3[on] + 5.0
        assert not torch.allclose(base_, m.spec2(nn.functional.gelu(m.spec1(g) + m.specm(torch.cat(mids3, -1))))[..., k]), "on-path head ignored"
    assert PATHS[L2.TID["IY"]] == [0, 1, 2, 3, 4, 5, 6] and PATHS[L2.SIL] == [0] and PATHS[L2.TID["TS"]] == [0, 1, 10, 11]
    assert decode39("full", [L2.TID["TS"], L2.TID["TS"], BLANKV, L2.SIL, L2.TID["SCHWA"]]) == [L2.PID["T"], L2.PID["S"], L2.PID["AH"]]
    print("struct2c selftest ok; paths:", {(L2.TOK + ["SIL"])[k]: PATHS[k] for k in (0, L2.SIL, L2.TID["TS"], L2.TID["DARK_L"])}, flush=True)


def train(arm, seed, epochs=60):
    dev = torch.device("cuda:0"); torch.cuda.set_per_process_memory_fraction(B.ALLOC_MIB / 8192, 0)
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    P = int(os.environ.get("PARAMS", 655924)); w = build(arm, P); m = make(arm, w).to(dev); n = counted(m)
    tag = f"{arm}_P{P}_s{seed}"; O = os.environ.get("OUTDIR", os.path.dirname(os.path.abspath(__file__)) + "/phones_out/struct2c")
    os.makedirs(O, exist_ok=True); assert not os.path.exists(f"{O}/{tag}.json"), "output exists: never mix runs"
    var = arm in ("tok", "full", "broken"); aux = arm in ("lite", "full") and AUXW > 0
    V, FR = {}, {}
    for sp in ["1.0"] + B.SPEEDS:
        X, Y = [], []
        for s_ in ("fit", "cal"): x, y = B.load(s_ if sp == "1.0" else f"{s_}_sp{sp}"); X += x; Y += y
        V[sp] = (X, Y)
    LV, F0 = [], []
    for s_ in ("fit", "cal"):
        d = np.load(f"{C}/v2c_{s_}.npz"); LV += [d[f"y{i}"].astype(np.int64) for i in range(int(d["n"]))]
        st = np.r_[0, np.cumsum(d["flens"])]; F0 += [d["f"][a:b] for a, b in zip(st[:-1], st[1:])]
    N = len(LV); assert N == len(V["1.0"][1]) and all(len(f) == len(x) for f, x in zip(F0, V["1.0"][0]))
    assert all(L2.to39(a.tolist()) == b.tolist() for a, b in zip(LV, V["1.0"][1])), "variant labels do not map back to the base labels"
    Y = LV if var else V["1.0"][1]
    if arm == "broken":
        pr = np.random.default_rng(1_000 + seed); perm = pr.permutation(N)
        while (perm == np.arange(N)).any(): perm = pr.permutation(N)
        Y = [Y[i] for i in perm]; aux = False
    HTd = torch.from_numpy(HT).to(dev)
    def ftarget(i, sp, T_out):  # frame label at input frame 2j of the (speed-changed) utterance -> original frame round(2j * speed)
        f = F0[i]; s = float(sp); idx = np.minimum(np.round(np.arange(T_out) * 2 * s).astype(np.int64), len(f) - 1)
        return torch.from_numpy(f[idx].astype(np.int64))
    evals = ["libriA", "libriB", "testother", "author", "devother"]; E = {s_: B.load(s_) for s_ in evals[:4]}; E["devother"] = B.load_mm("devother")
    cfg = dict(script="struct2c.py", maxfr=int(os.environ.get("MAXFR", 24000)), tag=tag, arm=arm, seed=seed, width=w, params_counted=n,
               params_all=B.nparams(m), P=P, RIGHT=B.RIGHT, lookahead_ms=20 + 4 * B.RIGHT * 20, NORMW=B.NORMW, speeds=["1.0"] + B.SPEEDS, lr=B.LR,
               patience=B.PATIENCE, auxw=AUXW if aux else 0.0, skip=SKIP, mids=MIDS, outputs=(NT + 1) if var else NPH + 1, train_utts=N,
               train_h=round(sum(len(x) for x in V["1.0"][0]) / 100 / 3600, 3), epochs=epochs, alloc_mib=B.ALLOC_MIB)
    print(json.dumps(dict(config=cfg)), flush=True)
    B.stream_check(m, E["libriA"][0][0], dev); print(json.dumps(dict(stream_check="ok")), flush=True)
    maxfr = cfg["maxfr"]; nb = int(np.ceil(sum(len(x) for x in V["1.0"][0]) / maxfr * 1.05))
    opt = torch.optim.AdamW(m.parameters(), B.LR, weight_decay=1e-2); blank = BLANKV if var else NPH
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True); bce = nn.BCEWithLogitsLoss(reduction="none")
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, B.LR, total_steps=epochs * nb + 50, pct_start=0.1)
    best, bad, snaps, log, t0 = None, 0, [], [], time.time(); ck = f"{O}/{tag}.ckpt"; ep0 = 0
    if os.path.exists(ck):
        S = torch.load(ck, map_location=dev, weights_only=False); assert S["cfg"] == cfg, "resume with a different config"
        m.load_state_dict(S["model"]); opt.load_state_dict(S["opt"]); sch.load_state_dict(S["sch"]); rng.bit_generator.state = S["rng"]
        torch.set_rng_state(S["trng"].cpu()); torch.cuda.set_rng_state(S["crng"].cpu()); best, bad, snaps, log, ep0 = S["best"], S["bad"], S["snaps"], S["log"], S["ep"]
        t0 -= log[-1]["secs"] if log else 0; print(json.dumps(dict(resumed_from_epoch=ep0)), flush=True)
    for ep in range(ep0, epochs):
        m.train(); spv = rng.choice(list(V.keys()), size=N); X = [V[spv[i]][0][i] for i in range(N)]
        order = np.argsort([len(x) for x in X]); batches, cur, fr = [], [], 0
        for i in order:
            if cur and fr + len(X[i]) > maxfr: batches.append(cur); cur, fr = [], 0
            cur.append(i); fr += len(X[i])
        batches.append(cur); tot = tota = 0.0
        for bi in rng.permutation(len(batches)):
            ix = batches[bi]; lens = [len(X[i]) for i in ix]
            xs = nn.utils.rnn.pad_sequence([B.cnorm(X[i].to(dev)) for i in ix], batch_first=True); xs = B.specaug(xs, lens, rng)
            flen = torch.tensor([(l + 1) // 2 for l in lens]); tlen = torch.tensor([len(Y[i]) for i in ix]); hl = None
            net = m.net if isinstance(m, Lite) else m
            # trunk, heads and specialists under bf16 autocast (an fp32 copy of the 4 trunk layers ran out of the 593 MiB cap in the 15 Nov
            # smokes); the base / tok / lite output layer stays fp32 as in base5_next; CTC and BCE always in fp32
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Ls, g = net.trunk(xs)
                if isinstance(m, Full): z, hl = m.top(Ls, g)
                elif aux: hl = m.heads(Ls)[0]
            if not isinstance(m, Full): z = net.out(g.float())
            z = z.float(); hl = hl.float() if hl is not None else None
            lp = z.log_softmax(-1)
            loss = ctc(lp.transpose(0, 1), torch.from_numpy(np.concatenate([Y[i] for i in ix])), flen, tlen); la = torch.zeros((), device=dev)
            if aux:
                Tm = hl.shape[1]; lab = torch.full((len(ix), Tm), -1, dtype=torch.long)
                for r, i in enumerate(ix): T_ = int(flen[r]); lab[r, :T_] = ftarget(i, spv[i], T_)
                lab = lab.to(dev); valid = lab >= 0; tg = HTd[:, lab.clamp(min=0)].permute(1, 2, 0)  # (B, T, 18)
                mk = (tg >= 0) & valid.unsqueeze(-1); l_ = bce(hl.float(), tg.clamp(min=0)) * mk
                la = (l_.sum((0, 1)) / mk.sum((0, 1)).clamp(min=1)).mean()
            opt.zero_grad(); (loss + AUXW * la if aux else loss).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            if sch.last_epoch < sch.total_steps - 1: sch.step()
            tot += loss.item() * len(ix); tota += la.item() * len(ix)
        r = dict(epoch=ep + 1, lr_end=float(f"{opt.param_groups[0]['lr']:.3g}"), loss=round(tot / N, 4), aux=round(tota / N, 4),
                 secs=round(time.time() - t0), gpu_mib=B.gpu_mib(), batches=len(batches))
        assert r["gpu_mib"] <= B.LIMIT_MIB, f"GPU memory {r['gpu_mib']} MiB > {B.LIMIT_MIB}: abort (the agent's 1 GB limit)"
        if (ep + 1) % 2 == 0 or ep + 1 == epochs:
            r["libriA"] = evaluate(arm, m, *E["libriA"], dev); per = r["libriA"]["PER"]
            st = {k: v.detach().clone() for k, v in m.state_dict().items()}
            if best is None or per < best[0]: best, bad = (per, ep + 1, st), 0
            else: bad += 1
            snaps = sorted(snaps + [(per, ep + 1, st)], key=lambda t: t[0])[:3]
        print(json.dumps(r), flush=True); log.append(r)
        torch.save(dict(cfg=cfg, model=m.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), rng=rng.bit_generator.state, trng=torch.get_rng_state(),
                        crng=torch.cuda.get_rng_state(), best=best, bad=bad, snaps=snaps, log=log, ep=ep + 1), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        if bad >= B.PATIENCE: break
    m.load_state_dict(best[2])
    if len(snaps) > 1:
        avg = {k: sum(t[2][k] for t in snaps) / len(snaps) for k in best[2]}; m.load_state_dict(avg); ra = evaluate(arm, m, *E["libriA"], dev)
        print(json.dumps(dict(avg_of=[t[1] for t in snaps], libriA=ra, best_single=best[0])), flush=True)
        if ra["PER"] <= best[0]: best = (ra["PER"], "avg" + "+".join(str(t[1]) for t in snaps), avg)
        else: m.load_state_dict(best[2])
    torch.save(best[2], f"{O}/{tag}.pt")
    mc = make(arm, w); mc.load_state_dict({k: v.cpu() for k, v in best[2].items()}); mc.eval()
    if isinstance(mc, Full):  # masked entries must still be exactly zero after training
        assert (mc.specm.weight[mc.mask == 0] == 0).all() and (mc.spec2.weight[mc.mask2 == 0] == 0).all(), "mask broken"
    q = torch.ao.quantization.quantize_dynamic(mc, {nn.GRU, nn.Linear}, dtype=torch.qint8)
    R = dict(config=cfg, best_epoch=best[1], log=log, sets={})
    for s_ in evals:
        f32 = evaluate(arm, m, *E[s_], dev); i8 = evaluate(arm, q, *E[s_], torch.device("cpu"))
        R["sets"][s_] = dict(fp32=f32, int8=i8); print(s_, "fp32", f32, "int8", i8, flush=True)
    json.dump(R, open(f"{O}/{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    if sys.argv[1:] == ["selftest"]: selftest(); sys.exit()
    train(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 60)
