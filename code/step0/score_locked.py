"""Locked scorer for SEALED test sets (MASTER_PLAN Part C v6.3 C2.0; review round 3 C2.0 / C5.4). The ONLY code path that reads a sealed set.
Training scripts (base6.py / base6s.py) score validation sets only (EVALSETS allowlist) and cannot load sealed files: the sealed caches live in
cache/sealed/, outside the path base5_next.load() reads.

Protocol (all asserted, fail closed):
1. A LOCK file step0/locks/<question>.lock.json is written BEFORE any sealed access (make_lock): question id, the comparison (arm -> list of
   model .pt paths with sha256 and their run json), the sealed set name + sha256 of its cache file, the statistic, and the lock's own sha256
   recorded in step0/locks/LOCKS.txt (append-only). A lock cannot be rewritten (exists -> refuse).
2. score <lock>: re-hash every model and the set; any mismatch -> refuse. Every model of the lock is scored in one call (no partial opening);
   a (lock, model sha) pair already in step0/locks/SCORED.jsonl -> refuse (one locked pass per model and set).
3. Output: per-utterance edits e, reference length t, hypothesis length h, utterance id; totals; written next to the lock as
   <question>.scores.json (exists -> refuse). Scoring runs on the CPU in float32 with the same decode as the training script's evaluate().
Models: base6 plain (config without 'arm', NORM must be causal: fixed-norm statistics are not saved in the .pt) and base6s arms (make(arm, bud)).
Usage: python -I score_locked.py selftest | lock <spec.json> | score <question>      (CPU venv with torch; no GPU)."""
import os
import hashlib, json, os, sys, time
import numpy as np
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, D)
LOCKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locks")
SEALED = os.environ.get("DATA_DIR", "data") + "/cache/sealed"
SEALED_SETS = {"cleanT": "base5_cleanT.npz"}  # hardT (train-other-500 panel) is added when built and verified


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def make_lock(spec_path, locks=LOCKS, sealed=SEALED):
    S = json.load(open(spec_path)); q = S["question"]; os.makedirs(locks, exist_ok=True); p = f"{locks}/{q}.lock.json"
    assert not os.path.exists(p), "lock exists: a lock is never rewritten"
    assert S["set"] in SEALED_SETS, ("not a sealed set", S["set"]); setf = f"{sealed}/{SEALED_SETS[S['set']]}"
    L = dict(question=q, set=S["set"], set_sha256=sha(setf), statistic=S["statistic"], created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             models={a: [dict(pt=pt, pt_sha256=sha(pt), run_json=pt[:-3] + ".json") for pt in pts] for a, pts in S["arms"].items()})
    for a in L["models"].values():
        for r in a: assert os.path.exists(r["run_json"]), ("run json missing", r["run_json"])
    json.dump(L, open(p, "w"), indent=1); h = sha(p)
    with open(f"{locks}/LOCKS.txt", "a") as f: f.write(f"{q} {h} {L['created']}\n")
    return p, h


def load_model(r):
    import torch
    cfg = json.load(open(r["run_json"]))["config"]; os.environ["RIGHT"] = str(cfg["RIGHT"]); os.environ["NORM"] = cfg["NORM"]
    assert cfg["NORM"] == "causal", "fixed-norm runs cannot be rescored (statistics not saved)"
    import base5_next as B, base6 as B6
    if "arm" in cfg and cfg.get("outputs"):
        import base6s as BS
        bud = json.load(open(os.environ.get("BUDF", D + "/logs_base6/base6s_budget.json"))); m = BS.make(cfg["arm"], bud)
        dec = (lambda ids: B.collapse(ids)) if cfg["arm"] == "P" else (lambda ids: BS.T["decode39"]("full", ids))
    else:
        m = B6.Base6(B.build(int(cfg["params"]) if "P" not in cfg else int(cfg["P"]))); dec = B.collapse
    m.load_state_dict(torch.load(r["pt"], map_location="cpu")); m.eval()
    return m, dec, B, B6


def score_items(m, dec, norm, X, Y, edit):
    import torch
    out = []
    with torch.no_grad():
        for i in range(len(X)):
            hyp = dec(m(norm(torch.as_tensor(X[i]).float())[None])[0].argmax(-1).tolist()); y = list(map(int, Y[i]))
            out.append((int(edit(hyp, y)), len(y), len(hyp)))
    return out


def read_set(setf):
    d = np.load(setf, allow_pickle=True); st = np.r_[0, np.cumsum(d["lens"])]
    X = [d["X"][a:c].astype(np.float32) for a, c in zip(st[:-1], st[1:])]; n = int(d["n"])
    Y = [d[f"y{i}"] for i in range(n)] if "y0" in d else list(d["Y"]); return X, Y, [str(f) for f in d["files"]]


def score(q, locks=LOCKS, sealed=SEALED, loader=load_model, reader=read_set):
    p = f"{locks}/{q}.lock.json"; L = json.load(open(p)); h = sha(p)
    reg = [l.split() for l in open(f"{locks}/LOCKS.txt")]; assert [q, h] in [r[:2] for r in reg], "lock not registered or changed"
    out_p = f"{locks}/{q}.scores.json"; assert not os.path.exists(out_p), "already scored"
    setf = f"{sealed}/{SEALED_SETS[L['set']]}"; assert sha(setf) == L["set_sha256"], "sealed set changed"
    done = [json.loads(l) for l in open(f"{locks}/SCORED.jsonl")] if os.path.exists(f"{locks}/SCORED.jsonl") else []
    for a in L["models"].values():
        for r in a:
            assert sha(r["pt"]) == r["pt_sha256"], ("model changed", r["pt"])
            assert not any(d["set"] == L["set"] and d["pt_sha256"] == r["pt_sha256"] for d in done), ("model already scored on this set", r["pt"])
    X, Y, files = reader(setf); R = dict(lock=q, lock_sha256=h, set=L["set"], scored=time.strftime("%Y-%m-%dT%H:%M:%S%z"), arms={})
    for arm, rs in L["models"].items():
        R["arms"][arm] = []
        for r in rs:
            m, dec, B, B6 = loader(r); it = score_items(m, dec, B6.norm, X, Y, B.edit)
            e, t, hh = (sum(z[k] for z in it) for k in range(3))
            R["arms"][arm].append(dict(pt=r["pt"], pt_sha256=r["pt_sha256"], PER=100 * e / t, hyp_ref=hh / t, items=[dict(utt=f, e=z[0], t=z[1], h=z[2]) for f, z in zip(files, it)]))
    json.dump(R, open(out_p, "w"), indent=1)
    with open(f"{locks}/SCORED.jsonl", "a") as f:
        for rs in L["models"].values():
            for r in rs: f.write(json.dumps(dict(lock=q, set=L["set"], pt_sha256=r["pt_sha256"])) + "\n")
    return out_p


def selftest():
    # synthetic sealed set + fake models with known per-item answers; every refusal path must fire
    import tempfile, types
    tmp = tempfile.mkdtemp(); locks, sealed = f"{tmp}/locks", f"{tmp}/sealed"; os.makedirs(sealed)
    X = [np.zeros((5, 3), np.float32), np.zeros((4, 3), np.float32)]; Y = [np.array([1, 2, 3]), np.array([4, 4])]
    np.savez(f"{sealed}/base5_cleanT.npz", X=np.concatenate(X), lens=np.array([5, 4]), n=2, files=np.array(["a-1", "b-2"]), y0=Y[0], y1=Y[1])
    pts = {}
    for k in ("A", "B"):
        open(f"{tmp}/{k}.pt", "wb").write(k.encode()); json.dump({"config": {}}, open(f"{tmp}/{k}.json", "w")); pts[k] = [f"{tmp}/{k}.pt"]
    hyps = {"A": ([1, 2, 3], [4]), "B": ([1, 9], [4, 4])}  # A: e 0 + 1; B: e 2 + 0

    def fake_loader(r):
        k = os.path.basename(r["pt"])[0]; calls = iter(hyps[k])
        def m(x): import torch; return torch.zeros(1, 1, 1)
        B = types.SimpleNamespace(edit=lambda a, b: __import__("base5_next").edit(a, b)); B6 = types.SimpleNamespace(norm=lambda x: x)
        return m, (lambda ids: next(calls)), B, B6
    spec = f"{tmp}/spec.json"; json.dump(dict(question="T1", set="cleanT", statistic="PER", arms=pts), open(spec, "w"))
    try: score("T1", locks, sealed, fake_loader); raise SystemExit("FAIL: scored without a lock")
    except (FileNotFoundError, AssertionError): pass
    make_lock(spec, locks, sealed)
    try: make_lock(spec, locks, sealed); raise SystemExit("FAIL: lock rewritten")
    except AssertionError: pass
    out = json.load(open(score("T1", locks, sealed, fake_loader)))
    got = {a: [i["e"] for i in v[0]["items"]] for a, v in out["arms"].items()}; assert got == {"A": [0, 1], "B": [2, 0]}, got
    assert abs(out["arms"]["A"][0]["PER"] - 20.0) < 1e-9 and abs(out["arms"]["B"][0]["PER"] - 40.0) < 1e-9
    try: score("T1", locks, sealed, fake_loader); raise SystemExit("FAIL: second scoring")
    except AssertionError: pass
    json.dump(dict(question="T2", set="cleanT", statistic="PER", arms={"A": pts["A"]}), open(spec, "w")); make_lock(spec, locks, sealed)
    try: score("T2", locks, sealed, fake_loader); raise SystemExit("FAIL: same model scored twice on the set")
    except AssertionError: pass
    json.dump(dict(question="T3", set="cleanT", statistic="PER", arms={"B": [f"{tmp}/C.pt"]}), open(spec, "w"))
    open(f"{tmp}/C.pt", "wb").write(b"C"); json.dump({"config": {}}, open(f"{tmp}/C.json", "w")); make_lock(spec, locks, sealed)
    open(f"{tmp}/C.pt", "wb").write(b"C2")
    try: score("T3", locks, sealed, fake_loader); raise SystemExit("FAIL: changed model scored")
    except AssertionError: pass
    print("selftest OK: per-item answers exact; refusals: no lock, lock rewrite, second scoring, model reuse, changed model")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "selftest": selftest()
    elif c == "lock": print(make_lock(sys.argv[2]))
    elif c == "score": print(score(sys.argv[2]))
