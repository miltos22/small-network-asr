"""Q2 (c) runner pieces (Part C C9 4): job specs for the five paired seeds and the POST-RUN PAIR CHECK that every (S, R) pair must pass before
BIND (reader_v65.bind). Launching goes through gpu_gate2 / the queue (C9 6), never directly.
jobs(recipe, seeds) -> list of {arm, seed, env, argv}: two FRESH processes per seed; arm S = base6s.py S, arm R = base6s_R.py with
  RWIRE_SEED = seed; both with Q2C_PAIR=1 and Q2C_ARM, so the trainer itself asserts the wiring and RWIRE_SEED == seed (patch_q2c.py).
pair_check(s_json, r_json, seed) refuses unless: both runs complete (len(log) == epochs, no stop_after); seed fields == seed; S has no
  'RWIRE sigma=' and R's sigma == the derangement base6s_R derives from RWIRE_SEED == seed (recomputed here); RWIRE_SEED recorded == seed;
  plan_sha256, targets_sha256, train_ix_sha256, total_steps, epochs, bytes, budget and every recipe field equal; mask_sha256 present and
  DIFFERENT (the wiring really changed). Usage: python -I runner_q2c.py selftest | check <S.json> <R.json> <seed>"""
import json, sys, os
import numpy as np
NH = 18
RECIPE_FIELDS = ("RIGHT", "NORM", "FEAT", "WARP", "DRAW", "AUG", "maxfr", "lr", "speeds", "outputs", "auxw", "params", "train_h", "SMOKE")
H = "code"


def sigma(seed):  # identical to base6s_R.py lines 12-13 (checked against a real R record in the selftest when one exists)
    rng = np.random.default_rng(int(seed)); s = rng.permutation(NH)
    while (s == np.arange(NH)).any(): s = rng.permutation(NH)
    return ",".join(map(str, s.tolist()))


def jobs(recipe, seeds=(0, 1, 2, 3, 4), tag="q2c", epochs=1836):
    assert list(seeds) == [0, 1, 2, 3, 4], "Q2 (c) has exactly seeds 0-4"
    out = []
    for s in seeds:
        base = dict(recipe, Q2C_PAIR="1")
        out.append(dict(arm="S", seed=s, env=dict(base, Q2C_ARM="S"), argv=["base6s.py", "S", f"{tag}_S", str(s), str(epochs)]))
        out.append(dict(arm="R", seed=s, env=dict(base, Q2C_ARM="R", RWIRE_SEED=str(s)), argv=["base6s_R.py", f"{tag}_R", str(s), str(epochs)]))
    for j in out: assert "RWIRE_SEED" not in j["env"] if j["arm"] == "S" else j["env"]["RWIRE_SEED"] == str(j["seed"])
    return out


def pair_check(sj, rj, seed):
    S, R = json.load(open(sj)), json.load(open(rj)); cs, cr = S["config"], R["config"]
    for X, c in ((S, cs), (R, cr)):
        assert len(X.get("log", [])) == int(c["epochs"]) and not X.get("stop_after"), ("run incomplete", c["tag"])
        assert int(c["seed"]) == int(seed), ("seed", c["tag"], c["seed"], seed)
        for k in ("plan_sha256", "targets_sha256", "train_ix_sha256", "mask_sha256"): assert c.get(k), ("missing pairing evidence", k, c["tag"])
    assert "RWIRE sigma=" not in cs["frozen_tree"] and cs.get("Q2C_ARM") == "S", "S record carries a shuffle"
    assert cr.get("Q2C_ARM") == "R" and str(cr.get("RWIRE_SEED")) == str(seed), ("R RWIRE_SEED", cr.get("RWIRE_SEED"), seed)
    assert cr["frozen_tree"] == cs["frozen_tree"] + " + RWIRE sigma=" + sigma(seed), "R sigma is not the derangement of its seed"
    for k in ("plan_sha256", "targets_sha256", "train_ix_sha256", "total_steps", "epochs", "bytes", "budget") + RECIPE_FIELDS:
        assert cs.get(k) == cr.get(k), ("pair differs", k, cs.get(k), cr.get(k))
    assert cs["mask_sha256"] != cr["mask_sha256"], "wiring masks equal: the shuffle did not take effect"
    return True


def selftest():
    import tempfile, copy
    tmp = tempfile.mkdtemp(); res = {}
    base = dict(tag="x", seed=2, epochs=3, frozen_tree="/t.py", plan_sha256="p", targets_sha256="t", train_ix_sha256="i", total_steps=27,
                bytes=1376393, budget={"w": 232}, mask_sha256="mS", Q2C_ARM="S", RWIRE_SEED=None, **{k: 1 for k in RECIPE_FIELDS})
    S = dict(config=dict(base), log=[1, 2, 3], sets={}); R = copy.deepcopy(S)
    R["config"].update(Q2C_ARM="R", RWIRE_SEED="2", mask_sha256="mR", frozen_tree="/t.py + RWIRE sigma=" + sigma(2))
    def w(n, o): p = f"{tmp}/{n}.json"; json.dump(o, open(p, "w")); return p
    def refused(Sx, Rx, seed=2):
        try: pair_check(w("s", Sx), w("r", Rx), seed); return False
        except (AssertionError, KeyError): return True
    res["good_pair_passes"] = pair_check(w("s", S), w("r", R), 2)
    for name, f in [("wrong_seed_arg", lambda s, r: None), ("plan_differs", lambda s, r: r["config"].update(plan_sha256="q")),
                    ("targets_differ", lambda s, r: r["config"].update(targets_sha256="q")),
                    ("rwire_seed_wrong", lambda s, r: r["config"].update(RWIRE_SEED="3", frozen_tree="/t.py + RWIRE sigma=" + sigma(3))),
                    ("sigma_not_of_seed", lambda s, r: r["config"].update(frozen_tree="/t.py + RWIRE sigma=" + sigma(3))),
                    ("same_mask", lambda s, r: r["config"].update(mask_sha256="mS")), ("recipe_differs", lambda s, r: r["config"].update(WARP="S")),
                    ("incomplete_run", lambda s, r: r.update(log=[1])), ("stop_after_run", lambda s, r: s.update(stop_after=6)),
                    ("S_carries_shuffle", lambda s, r: s["config"].update(frozen_tree="/t.py + RWIRE sigma=" + sigma(2))),
                    ("missing_evidence", lambda s, r: s["config"].pop("plan_sha256"))]:
        Sx, Rx = copy.deepcopy(S), copy.deepcopy(R); f(Sx, Rx); res[name + "_refused"] = refused(Sx, Rx, 5 if name == "wrong_seed_arg" else 2)
    J = jobs(dict(DEV="cuda:0", RIGHT="12")); res["jobs_ten_fresh_processes"] = len(J) == 10 and all(j["env"]["Q2C_PAIR"] == "1" for j in J)
    try: jobs({}, seeds=(0, 1, 2, 3)); res["jobs_refuses_four_seeds"] = False
    except AssertionError: res["jobs_refuses_four_seeds"] = True
    rr = json.load(open(H + "/phones_out/base6/b6ii_R_s0.json"))["config"]["frozen_tree"]  # the 18 Nov R run = RWIRE_SEED 0
    res["sigma_matches_real_R_record_seed0"] = rr.endswith("RWIRE sigma=" + sigma(0))
    res["ALL_OK"] = all(res.values()); json.dump(res, open(os.path.dirname(os.path.abspath(__file__)) + "/runner_q2c_selftest_2026-11-20.json", "w"), indent=1)
    print(json.dumps(res, indent=1)); return res["ALL_OK"]


if __name__ == "__main__":
    if sys.argv[1] == "selftest": sys.exit(0 if selftest() else 1)
    if sys.argv[1] == "check": print(pair_check(sys.argv[2], sys.argv[3], int(sys.argv[4])))
