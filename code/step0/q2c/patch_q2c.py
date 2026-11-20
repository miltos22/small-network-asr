"""C9 4 instrumentation (apply ONLY after the v6.5 E1 runs end: they import these files). base6s.train records in its run config:
plan_sha256 (speed choices + batch order of every pass), targets_sha256 (training label sequences), train_ix_sha256 (training utterance
indices), mask_sha256 (arm S wiring mask), RWIRE_SEED (env, recorded); with Q2C_PAIR=1 it asserts Q2C_ARM matches the wiring (R <=> 'RWIRE
sigma=' in FROZEN) and RWIRE_SEED == seed for R. base6s_R.py: the docstring's old 'shape, not the phonetics' reading replaced by the v6.3
limit. Usage: python -I patch_q2c.py <dir>. Asserts each replacement count."""
import os
import sys, os
d = sys.argv[1]
def patch(fn, reps):
    p = os.path.join(d, fn); s = open(p).read()
    for a, b in reps:
        assert s.count(a) == 1, (fn, a[:80], s.count(a)); s = s.replace(a, b)
    open(p, "w").write(s); print("patched", p)
patch("base6s.py", [
 ("    total = sum(len(p[1]) for p in plans)\n    cfg = dict(script=\"base6s.py\",",
  "    total = sum(len(p[1]) for p in plans)\n"
  "    import hashlib as _hl  # C9 4: pairing evidence recorded in every run (Q2 (c) pair check, step0/q2c/runner_q2c.py)\n"
  "    _ph = _hl.sha256()\n"
  "    for _sp, _bat in plans:\n"
  "        _ph.update(\",\".join(map(str, _sp)).encode())\n"
  "        for _b in _bat: _ph.update(np.asarray(_b, np.int64).tobytes())\n"
  "    _th = _hl.sha256(); [_th.update(np.asarray(y_, np.int64).tobytes()) for y_ in Y]\n"
  "    _mk = _hl.sha256(m.mask.detach().cpu().numpy().tobytes()).hexdigest() if hasattr(m, \"mask\") else None\n"
  "    _rw = os.environ.get(\"RWIRE_SEED\")\n"
  "    if os.environ.get(\"Q2C_PAIR\") == \"1\":\n"
  "        assert arm == \"S\" and os.environ.get(\"Q2C_ARM\") in (\"S\", \"R\"), (\"Q2C arm\", arm, os.environ.get(\"Q2C_ARM\"))\n"
  "        assert (\"RWIRE sigma=\" in FROZEN) == (os.environ[\"Q2C_ARM\"] == \"R\"), \"Q2C_ARM does not match the wiring in force\"\n"
  "        assert os.environ[\"Q2C_ARM\"] == \"S\" or (_rw is not None and int(_rw) == seed), (\"RWIRE_SEED must equal the seed\", _rw, seed)\n"
  "    cfg = dict(script=\"base6s.py\","),
 ("train_h=round(sum(len(x) for x in V[\"1.0\"][0]) / 360000, 3), frozen_tree=FROZEN)",
  "train_h=round(sum(len(x) for x in V[\"1.0\"][0]) / 360000, 3), frozen_tree=FROZEN,\n"
  "               plan_sha256=_ph.hexdigest(), targets_sha256=_th.hexdigest(), train_ix_sha256=_hl.sha256(np.asarray(ix, np.int64).tobytes()).hexdigest(),\n"
  "               mask_sha256=_mk, RWIRE_SEED=_rw, Q2C_ARM=os.environ.get(\"Q2C_ARM\"))"),
])
patch("base6s_R.py", [
 ("but a specialist now reads heads trained on questions unrelated to its phoneme. Same bytes as S. If R ~ S, the gain is the shape, not the phonetics.",
  "but a specialist now reads heads trained on other questions. Same bytes as S. READING LIMIT (Part C v6.3): a derangement of indices does not\n"
  "guarantee that a specialist loses its whole original head SET, nor that the substituted questions are unrelated; so a WIN supports 'the semantic\n"
  "wiring beats a shuffled one', and R ~ S is NOT read as 'shape, not phonetics'. Q2 (c): RWIRE_SEED = seed (asserted with Q2C_PAIR=1)."),
])
