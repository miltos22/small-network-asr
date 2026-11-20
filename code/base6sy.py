"""Top-only readouts at equal bytes: shared vs private nonlinear correction (EXPLORATION). Written 18 Nov 2026 ~20:1x by the agent (reviewer 5.5).
Source: review council 18 Nov (all 3: "decisive next control is a byte-matched top-only shared head") + reviewer Q8 test 2 (add private narrowed to the
shared width = containment at equal width). Checks in council_2026-11-18/COUNCIL_CHECK.md, Q7_Q8_CHECK.md.
All arms: base6 trunk w 232, linear skip Linear(w, 55) + a separate linear blank Linear(w, 1) (S's blank parameterisation, so blank paths match),
one nonlinear bank h1 = Linear(w, U) reading ONLY the trunk top g, GELU, h2 = Linear(U, 54) writing the 54 nonblank outputs (blank stays linear, as S).
Heads exist for the auxiliary teaching loss only (same AUXW as S and G) and are dropped at export (subclass of ArmG, as G).
  TS  shared:  h2 dense. U fitted to S's exported bytes.
  TP  private: h2 block-diagonal, output k reads only its own group (np.array_split of U over 54). U fitted to S's bytes.
  TN  private narrowed to TS's U (fewer bytes; equal width and inputs: the containment comparison).
Init: nn.Linear default, except the surviving h2 weights of TP/TN are redrawn for their TRUE fan-in (uniform +-1/sqrt(group size)), so the private
branch is not ~sqrt(54) quieter at the start (reviewer Q2 / council: spec2 started 7.35x small).
Known answers (selftest): TS has zero structural zeros; TP/TN masks block-diagonal with every output >= 1 unit; TS / TP bytes within 0.5 % of S;
a TP forward with an off-block h1 unit perturbed leaves every other output's logit unchanged.
Usage: python3 -I base6sy.py <TS|TP|TN> <tag> <seed> <epochs>; base6sy.py bytes (writes logs_base6/base6sy_budget_<A>.json)."""
import os
import json, os, sys, math
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, torch, torch.nn as nn
import base6s as S6
NS, NOUT, BLANKV = S6.NS, S6.NOUT, S6.BLANKV


def groups(U, k): return [list(a) for a in np.array_split(np.arange(U), k)]


class ArmT(S6.ArmG):
    def __init__(s, w, U, private):
        nn.Module.__init__(s); s.t = S6.Trunk(w); s.heads = S6.T["Heads"](w); s.private = private; s.U = U
        s.skip = nn.Linear(w, NOUT); s.blank = nn.Linear(w, 1); s.h1 = nn.Linear(w, U); s.h2 = nn.Linear(U, NS)
        if private:
            m2 = torch.zeros(NS, U)
            with torch.no_grad():
                for k, g in enumerate(groups(U, NS)):
                    m2[k, g] = 1; b = 1 / math.sqrt(len(g)); s.h2.weight[k, g] = torch.empty(len(g)).uniform_(-b, b)
                s.h2.weight.mul_(m2)
            s.register_buffer("mask2", m2); s.h2.weight.register_hook(lambda gr: gr * s.mask2)

    def top(s, L, g):
        hl = s.heads(L)[0]
        return torch.cat([s.h2(nn.functional.gelu(s.h1(g))), s.blank(g)], -1) + s.skip(g), hl

    def struct_zero_bytes(s): return int((s.mask2 == 0).sum()) + 4 * s.mask2.numel() if s.private else 0


class ArmGT(S6.ArmG):
    # G-top (reviewer audit 19 Nov 06:2x): G unchanged (direct g->55, nonlinear h2 to all 55 incl. blank, aux heads for training only) except
    # that h1 reads ONLY the trunk top g; width fitted to S's bytes.
    def __init__(s, w, H):
        nn.Module.__init__(s); s.t = S6.Trunk(w); s.heads = S6.T["Heads"](w)
        s.direct = nn.Linear(w, S6.NOUT); s.h1 = nn.Linear(w, H); s.h2 = nn.Linear(H, S6.NOUT)

    def top(s, L, g):
        hl = s.heads(L)[0]
        return s.direct(g) + s.h2(nn.functional.gelu(s.h1(g))), hl

    def struct_zero_bytes(s): return 0


def build(A, U): return ArmGT(232, U) if A == "GT" else ArmT(232, U, A in ("TP", "TN"))


def fit(A, target):
    lo, hi = NS, 6000; cache = {}
    nb = lambda p: cache.setdefault(p, S6.export_bytes(build(A, p)))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if nb(mid) <= target: lo = mid
        else: hi = mid
    return min(((p, nb(p)) for p in (lo, hi)), key=lambda t: abs(t[1] - target))


def selftest():
    bud = json.load(open(D + "/logs_base6/base6s_budget.json")); T = bud["S"]["bytes"]; out = {}
    m = build("TS", 300); assert m.struct_zero_bytes() == 0 and not hasattr(m, "mask2")
    m = build("TP", 300); assert (m.mask2.sum(1) >= 1).all() and m.mask2.sum() == 300 and (m.mask2.sum(0) == 1).all(), "block-diagonal"
    with torch.no_grad():  # off-block leak check on the nonlinear branch
        h = torch.randn(3, 300); z0 = m.h2(nn.functional.gelu(h)); g0 = groups(300, NS)[0]; h2 = h.clone(); h2[:, g0] += 5.0
        z1 = m.h2(nn.functional.gelu(h2)); assert torch.allclose(z0[:, 1:], z1[:, 1:]) and not torch.allclose(z0[:, 0], z1[:, 0]), "private leak"
        r = m.h2.weight[0, g0].abs().max().item(); assert r <= 1 / math.sqrt(len(g0)) + 1e-6 and r > 0.5 / math.sqrt(len(g0)) * 0.5, "true fan-in init"
    for A in ("TS", "TP"):
        p, n = fit(A, T); assert abs(n - T) / T < 0.005, (A, p, n, T); out[A] = dict(A=A, U=p, bytes=n); print(A, p, n, round(100 * (n - T) / T, 3), "%", flush=True)
    U = out["TS"]["U"]; out["TN"] = dict(A="TN", U=U, bytes=S6.export_bytes(build("TN", U))); print("TN", U, out["TN"]["bytes"], flush=True)
    print("base6sy selftest ok", flush=True); return {A: {"G": dict(variant=A, w=232, U=v["U"], bytes=v["bytes"]), "S_bytes": T} for A, v in out.items()}


def hook(arm, bud):
    b = bud[arm]; A = b["variant"]; assert A == os.environ["SY"], (A, os.environ.get("SY")); print(f"SY variant {A} U {b['U']}", flush=True)
    return build(A, b["U"])


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    if sys.argv[1:] == ["bytesTSN"]:  # dense TS narrowed to TN's bytes (reviewer audit 19 Nov 08:5x: size vs block structure)
        T = json.load(open(D + "/logs_base6/base6sy_budget_TN.json"))["G"]["bytes"]
        p, n = fit("TSN", T); m = build("TSN", p); assert m.struct_zero_bytes() == 0 and not hasattr(m, "mask2")
        assert abs(n - T) / T < 0.005, (p, n, T); print("TSN", p, n, round(100 * (n - T) / T, 3), "%", flush=True)
        json.dump({"G": dict(variant="TSN", w=232, U=p, bytes=n), "TN_bytes": T}, open(f"{D}/logs_base6/base6sy_budget_TSN.json", "w"), indent=1)
        sys.exit()
    if sys.argv[1:] == ["bytesGT"]:
        T = json.load(open(D + "/logs_base6/base6s_budget.json"))["S"]["bytes"]; m = build("GT", 300)
        assert m.h1.in_features == 232 and m.h2.out_features == S6.NOUT and m.struct_zero_bytes() == 0
        p, n = fit("GT", T); assert abs(n - T) / T < 0.005, (p, n, T); print("GT", p, n, round(100 * (n - T) / T, 3), "%", flush=True)
        json.dump({"G": dict(variant="GT", w=232, U=p, bytes=n), "S_bytes": T}, open(f"{D}/logs_base6/base6sy_budget_GT.json", "w"), indent=1)
        sys.exit()
    if sys.argv[1:] == ["bytes"]:
        for A, v in selftest().items(): json.dump(v, open(f"{D}/logs_base6/base6sy_budget_{A}.json", "w"), indent=1)
        sys.exit()
    A = sys.argv[1]; assert A in ("TS", "TP", "TN", "GT", "TSN"); os.environ["SY"] = A
    os.environ["BUDF"] = f"{D}/logs_base6/base6sy_budget_{A}.json"; S6.MAKE_HOOK = hook
    S6.train("G", sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
