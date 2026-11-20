"""Step 0 (Part C v6.3, review r3 C2.2): dry-build of base6 / base6s batch plans from utterance lengths only (CPU, no features, no GPU).
Replicates base6.py:135,152-160 (prng = default_rng(10_000 + seed) used first by the plan loop; speed per utterance; length sort; MAXFR pack;
batch permutation) and base6.draw_indices (base6.py:56-66). Known answer: draw 1 seed 3 at 1 h x 972 passes must give 16,524 updates
(phones_out/base6/b6t1h_TS_d1_s3.json config.total_steps). Usage: python -I batchplan_dry.py"""
import os
import json, numpy as np
C = os.environ.get("DATA_DIR", "data") + "/cache"; SPEEDS = ["1.0", "0.9", "1.1"]; MAXFR = 24000
L = {sp: np.concatenate([np.load(f"{C}/base5_{s}{'' if sp == '1.0' else '_sp' + sp}.npz")["lens"] for s in ("fit", "cal")]) for sp in SPEEDS}
SPK = json.load(open(f"{C}/base5_speakers.json")); cf = SPK["fit"] + SPK["cal"]; assert all(len(v) == len(cf) for v in L.values())

def draw(d, hours):
    order = np.random.default_rng(500 + int(d)).permutation(sorted(set(cf))); out, fr = [], 0
    for s in order:
        if fr >= hours * 360000: break
        ix = [i for i, c in enumerate(cf) if c == s]; out += ix; fr += int(L["1.0"][ix].sum())
    return sorted(out), round(fr / 360000, 3)

def plan(ix, seed, epochs):
    prng = np.random.default_rng(10_000 + seed); N = len(ix); per = []
    for ep in range(epochs):
        sp = prng.choice(SPEEDS, size=N); ln = [int(L[sp[k]][ix[k]]) for k in range(N)]; order = np.argsort(ln); bat, cur, fr = [], [], 0
        for k in order:
            if cur and fr + ln[k] > MAXFR: bat.append(cur); cur, fr = [], 0
            cur.append(k); fr += ln[k]
        bat.append(cur); prng.permutation(len(bat)); per.append(len(bat))
    return per

ix, h = draw(1, 1.0); per = plan(ix, 3, 972); assert sum(per) == 16524, ("known answer failed", sum(per)); print("known answer OK: d1 s3 1 h 972 ->", sum(per), "h", h)
out = {}
for d in (0, 1, 2, 3, 4):
    for hours, ep in ((0.5, 1944), (1.0, 972)):
        ix, h = draw(d, hours)
        for seed in range(5):
            per = plan(ix, seed, ep); out[f"{hours}h_d{d}_s{seed}"] = dict(hours=h, utts=len(ix), passes=ep, updates=int(sum(per)), min=min(per), max=max(per))
json.dump(out, open("code/step0/batchplan_dry.json", "w"), indent=1)
for k, v in out.items():
    if k.startswith("0.5h") or k.endswith("s0"): print(k, v)
