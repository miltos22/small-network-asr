"""Controlled reader for SEALED panels: REGISTER / BIND / RELEASE (MASTER_PLAN Part C v6.5 C9 2). Fail closed everywhere.

REGISTER (before a comparison's first run): register(reg_name, protocol_path) writes locks/<reg_name>.register.json with the protocol sha256
  and appends it to locks/REGISTRY.jsonl. Refused if any panel the protocol names already has an access record (every comparison that will
  use a panel must be registered before that panel's first release). Protocol fields used here: arms, seeds, panels[{name, set_file,
  set_sha256, items[{utt, speaker, t}], release_path}]. (sec2_q2c_entry.py reads the same protocol and the release records.)
BIND (after training): bind(reg_name, slots) with slots = {arm: {seed: pt_path}}: exactly the protocol's arms x seeds; each checkpoint and
  its run json are hashed; the run must be complete (len(log) == epochs, no stop_after) and its GPU job eligible (gpu_gate2.eligible) unless
  the slot is listed in the protocol's 'legacy_slots' (old runs from before the gate; recorded as such). Written once (O_EXCL).
RELEASE (once per panel): release(panel) finds every registration naming the panel; each must be bound. It creates the ACCESS record
  locks/<panel>.access.json with O_EXCL BEFORE any model reads the set (a second or concurrent release fails there); then scores every
  bound slot, each in a FRESH python process whose recipe environment (RIGHT, NORM, FEAT, WARP, RWIRE_SEED) is set from the run's own
  config before import, using the training script's own evaluate(..., items=True) (parity by construction); the child re-hashes the
  checkpoint and asserts RIGHT, NORM, budget and frozen-tree string equal the run config. Rows {panel, arm, seed, utt, speaker, e, t, h}
  are checked against the protocol items (same utts, speakers, reference counts) and written once per registration; then the release
  record (panel, items_sha256 = sec2_q2c_entry.items_digest, rows sha256, bind sha256) is written once. A crash after the access record
  and before the release record leaves the panel COMPROMISED: release refuses forever; the owner decides (no silent retry).
Recipes the child cannot reproduce exactly are REFUSED: FEAT != '' (feature copies), WARP != '' (needs the panel's own warp estimates,
  realization 134; adapter TO DO for Q4), NORM fixed (statistics not in the checkpoint).
Usage: python -I reader_v65.py selftest | parity <tag> <set> | register <name> <protocol> | bind <name> <slots.json> | release <panel>"""
import os
import hashlib, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.dirname(HERE)
LOCKS = os.environ.get("READER_LOCKS", HERE + "/locks")
PY = "python3"
sys.path.insert(0, "code/small_net")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def items_digest(items):  # identical to sec2_q2c_entry.items_digest (asserted in selftest)
    return hashlib.sha256(json.dumps(sorted((i["utt"], i["speaker"], int(i["t"])) for i in items)).encode()).hexdigest()


def write_once(p, obj):  # atomic create: fails if the file exists (O_EXCL), content complete before the name appears
    import threading; tmp = f"{p}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as f: json.dump(obj, f, indent=1); f.flush(); os.fsync(f.fileno())
    try: os.link(tmp, p)
    finally: os.unlink(tmp)
    return sha(p)


def now(): return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def registry(locks):
    p = f"{locks}/REGISTRY.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def register(name, protocol_path, locks=LOCKS):
    os.makedirs(locks, exist_ok=True); P = json.load(open(protocol_path))
    for pn in P["panels"]:
        assert not os.path.exists(f"{locks}/{pn['name']}.access.json"), ("panel already accessed; registration closed", pn["name"])
        assert len({i["utt"] for i in pn["items"]}) == len(pn["items"]), ("duplicate utt", pn["name"])
    assert all(r["name"] != name for r in registry(locks)), ("registration exists", name)
    rec = dict(name=name, protocol_path=os.path.abspath(protocol_path), protocol_sha256=sha(protocol_path),
               panels=[p["name"] for p in P["panels"]], created=now())
    h = write_once(f"{locks}/{name}.register.json", rec)
    with open(f"{locks}/REGISTRY.jsonl", "a") as f: f.write(json.dumps(dict(name=name, sha256=h, panels=rec["panels"])) + "\n")
    return h


def _reg(name, locks):
    p = f"{locks}/{name}.register.json"; R = json.load(open(p))
    assert any(r["name"] == name and r["sha256"] == sha(p) for r in registry(locks)), ("registration changed or unlisted", name)
    assert sha(R["protocol_path"]) == R["protocol_sha256"], ("protocol changed after registration", name)
    return R, json.load(open(R["protocol_path"]))


def run_complete(rj):
    R = json.load(open(rj)); c = R["config"]
    return len(R.get("log", [])) == int(c["epochs"]) and not R.get("stop_after") and bool(R.get("sets")), c


def bind(name, slots, locks=LOCKS, eligible=None):
    R, P = _reg(name, locks)
    want = {(a, int(s)) for a in P["arms"] for s in P["seeds"]}; got = {(a, int(s)) for a, ss in slots.items() for s in ss}
    assert want == got, ("slots must be exactly arms x seeds", sorted(want ^ got))
    legacy = {(a, int(s)) for a, s in P.get("legacy_slots", [])}
    if eligible is None:
        import gpu_gate2; eligible = gpu_gate2.eligible
    out = []
    for a, ss in slots.items():
        for s, v in ss.items():
            pt, job = (v["pt"], v.get("gate_job")) if isinstance(v, dict) else (v, None)
            rj = pt[:-3] + ".json"; ok, c = run_complete(rj); assert ok, ("run incomplete", pt)
            if (a, int(s)) in legacy: el = "legacy (before the gate; listed in the protocol)"
            else: assert job and eligible(job), ("GPU job not eligible", pt, job); el = f"gate job {job}"
            out.append(dict(arm=a, seed=int(s), pt=os.path.abspath(pt), pt_sha256=sha(pt), run_json=os.path.abspath(rj), run_json_sha256=sha(rj),
                            eligibility=el))
    return write_once(f"{locks}/{name}.bind.json", dict(registration=name, registration_sha256=sha(f"{locks}/{name}.register.json"),
                                                         slots=out, created=now()))


def child_env(cfg, extra=None):
    assert cfg.get("FEAT", "") in ("", None), ("FEAT copies not reproducible by the reader", cfg.get("FEAT"))
    assert cfg.get("WARP", "") in ("", None), ("WARP needs a panel warp adapter (TO DO)", cfg.get("WARP"))
    assert cfg["NORM"] == "causal", "fixed-norm statistics are not saved in the checkpoint"
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "OMP_NUM_THREADS")}
    env.update(RIGHT=str(cfg["RIGHT"]), NORM=cfg["NORM"], FEAT="", WARP="", CUDA_VISIBLE_DEVICES="")
    ft = cfg.get("frozen_tree", "")
    if "RWIRE sigma=" in ft: env["RWIRE_SEED"] = str(cfg.get("RWIRE_SEED", (extra or {}).get("RWIRE_SEED", 0)))
    return env


def score_slot(pt, pt_sha, rj, set_file, set_sha, out_path, fake=None):
    """Run the child in a fresh process. fake: path of a JSON {utt: [e, t, h]} used by the selftest instead of a model."""
    c = json.load(open(rj))["config"]
    env = child_env(c) if fake is None else {"PATH": os.environ["PATH"], "READER_FAKE": fake}
    r = subprocess.run([PY, "-I", os.path.abspath(__file__), "_child", pt, pt_sha, rj, set_file, set_sha, out_path], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, ("child failed", pt, r.stderr[-800:])
    return [json.loads(l) for l in open(out_path)]


def _child(pt, pt_sha, rj, set_file, set_sha, out_path):
    import numpy as np
    assert sha(pt) == pt_sha, "checkpoint changed since bind"; assert sha(set_file) == set_sha, "panel set file changed"
    d = np.load(set_file, allow_pickle=True); files = [str(f).split(".")[0] for f in d["files"]]
    if os.environ.get("READER_FAKE"):
        F = json.load(open(os.environ["READER_FAKE"])); items = [[i, *F[u]] for i, u in enumerate(files) if u in F]
        if F.get("__crash__"): sys.exit(3)
    else:
        sys.path.insert(0, D); os.chdir(D)
        import torch
        cfg = json.load(open(rj))["config"]
        import base5_next as B, base6 as B6
        assert B.RIGHT == cfg["RIGHT"] and B6.NORM == cfg["NORM"] and not B.FEAT and not B.WARP, ("recipe env", B.RIGHT, B6.NORM)
        st = np.r_[0, np.cumsum(d["lens"])]; X = torch.from_numpy(d["X"].astype(np.float32))
        xs = [X[a:b] for a, b in zip(st[:-1], st[1:])]; ys = [d[f"y{i}"].astype(np.int64) for i in range(int(d["n"]))]
        if cfg.get("script") == "base6s.py":
            if "RWIRE sigma=" in cfg.get("frozen_tree", ""): import base6s_R; import base6s as S
            else: import base6s as S
            assert S.FROZEN == cfg["frozen_tree"], ("frozen tree / shuffle differs from the run", S.FROZEN[-80:], cfg["frozen_tree"][-80:])
            bud = S.budget(); key = "P" if cfg["arm"] == "P" else cfg["arm"]
            assert {k: bud[key][k] for k in cfg["budget"] if k in bud[key]} == {k: v for k, v in cfg["budget"].items() if k in bud[key]}, ("budget", bud[key], cfg["budget"])
            m = S.make(cfg["arm"], bud); m.load_state_dict(torch.load(pt, map_location="cpu"))
            r = S.evaluate(cfg["arm"], m, xs, ys, "cpu", items=True)
        else:
            raise SystemExit("plain base6.py checkpoints: model geometry not reconstructable here yet (TO DO with the Q1/Q4 recipes)")
        items = r["items"]
    with open(out_path, "w") as f:
        for i, e, t, h in items: f.write(json.dumps(dict(utt=files[i], e=int(e), t=int(t), h=int(h))) + "\n")


def release(panel, locks=LOCKS, eligible_fake=None, fake_map=None):
    acc = f"{locks}/{panel}.access.json"
    regs = [r for r in registry(locks) if panel in r["panels"]]; assert regs, ("no registration names this panel", panel)
    assert not os.path.exists(acc) or os.path.exists(f"{locks}/{panel}.released.json"), ("COMPROMISED: access begun without release", panel)
    work = []
    for r in regs:
        R, P = _reg(r["name"], locks); bp = f"{locks}/{r['name']}.bind.json"; assert os.path.exists(bp), ("registration not bound", r["name"])
        Bd = json.load(open(bp)); assert Bd["registration_sha256"] == sha(f"{locks}/{r['name']}.register.json"), "bind does not match registration"
        pn = [p for p in P["panels"] if p["name"] == panel][0]; work.append((r["name"], P, pn, Bd, sha(bp)))
    sets = {(w[2]["set_file"], w[2]["set_sha256"]) for w in work}; assert len(sets) == 1, ("registrations disagree on the panel file", sets)
    set_file, set_sha = sets.pop(); assert sha(set_file) == set_sha, "panel set file changed since registration"
    write_once(acc, dict(panel=panel, set_sha256=set_sha, registrations=[w[0] for w in work], pid=os.getpid(), begun=now()))  # O_EXCL gate
    out = {}
    for name, P, pn, Bd, bsha in work:
        ref = {i["utt"]: i for i in pn["items"]}; rows = []
        for s in Bd["slots"]:
            o = f"{locks}/{panel}.{name}.{s['arm']}.{s['seed']}.child.jsonl"
            got = score_slot(s["pt"], s["pt_sha256"], s["run_json"], set_file, set_sha, o, fake=(fake_map or {}).get((s["arm"], s["seed"])))
            gu = [g["utt"] for g in got]; assert sorted(gu) == sorted(ref) and len(gu) == len(set(gu)), ("coverage differs from protocol", s["pt"])
            for g in got:
                i = ref[g["utt"]]; assert g["t"] == int(i["t"]), ("reference count differs", g["utt"])
                rows.append(dict(panel=panel, arm=s["arm"], seed=s["seed"], utt=g["utt"], speaker=i["speaker"], e=g["e"], t=g["t"], h=g["h"]))
        rp = f"{locks}/{panel}.{name}.rows.jsonl"; tmp = rp + f".tmp.{os.getpid()}.{id(rows)}"
        with open(tmp, "w") as f: f.write("".join(json.dumps(x) + "\n" for x in rows)); f.flush(); os.fsync(f.fileno())
        os.link(tmp, rp); os.unlink(tmp)
        write_once(pn["release_path"], dict(panel=panel, items_sha256=items_digest(pn["items"]), rows_path=rp, rows_sha256=sha(rp),
                                            bind_sha256=bsha, registration=name, released=now()))
        out[name] = rp
    write_once(f"{locks}/{panel}.released.json", dict(panel=panel, rows=out, done=now()))
    return out


def parity(tag, set_name):
    """Reader counts vs the trainer's saved validation counts for the same checkpoint (C9 2 real-checkpoint parity)."""
    import tempfile
    O = D + "/phones_out/base6"; pt, rj = f"{O}/{tag}.pt", f"{O}/{tag}.json"; R = json.load(open(rj))
    sf = f"{os.environ.get('DATA_DIR', 'data')}/cache/base5_{set_name}.npz"; o = tempfile.mktemp(suffix=".jsonl")
    got = score_slot(pt, sha(pt), rj, sf, sha(sf), o); e = sum(g["e"] for g in got); t = sum(g["t"] for g in got); h = sum(g["h"] for g in got)
    saved = R["sets"][set_name]; per = round(100 * e / t, 2)
    return dict(tag=tag, set=set_name, n=len(got), e=e, t=t, PER=per, hyp_ref=round(h / t, 3), saved=saved, diff=round(per - saved["PER"], 2))


def selftest():
    import tempfile, numpy as np, threading
    from sec2_q2c_entry import items_digest as q2d
    tmp = tempfile.mkdtemp(); L = f"{tmp}/locks"; os.makedirs(L)
    utts = ["11-1-0001", "11-1-0002", "22-2-0001"]; T = [3, 2, 4]
    np.savez(f"{tmp}/panel.npz", X=np.zeros((9, 2), np.float32), lens=np.array([3, 3, 3]), n=3, files=np.array([u + ".flac" for u in utts]),
             y0=np.zeros(3), y1=np.zeros(2), y2=np.zeros(4))
    items = [dict(utt=u, speaker=u.split("-")[0], t=t) for u, t in zip(utts, T)]
    assert items_digest(items) == q2d(items), "digest differs from sec2_q2c_entry"
    P = dict(arms=["S", "R"], seeds=[0, 1], legacy_slots=[["R", 1]], panels=[dict(name="pX", set_file=f"{tmp}/panel.npz", set_sha256=sha(f"{tmp}/panel.npz"),
             items=items, release_path=f"{L}/pX.release.json", panel_weight=1.0)])
    json.dump(P, open(f"{tmp}/proto.json", "w"))
    slots = {}
    for a in ("S", "R"):
        for s in (0, 1):
            pt = f"{tmp}/{a}{s}.pt"; open(pt, "w").write(a + str(s))
            json.dump(dict(config=dict(epochs=2, RIGHT=12, NORM="causal"), log=[1, 2], sets={"x": 1}), open(pt[:-3] + ".json", "w"))
            slots.setdefault(a, {})[s] = dict(pt=pt, gate_job=f"J{a}{s}")
    fm = {}
    for a in ("S", "R"):
        for s in (0, 1):
            f = f"{tmp}/fake_{a}{s}.json"; json.dump({u: [s + (a == "R"), t, t] for u, t in zip(utts, T)}, open(f, "w")); fm[(a, s)] = f
    res = {}
    def refuses(fn):
        try: fn(); return False
        except (AssertionError, FileExistsError, FileNotFoundError): return True
    res["release_without_registration"] = refuses(lambda: release("pX", L, fake_map=fm))
    register("q", f"{tmp}/proto.json", L)
    res["release_without_bind"] = refuses(lambda: release("pX", L, fake_map=fm))
    bad = {a: dict(v) for a, v in slots.items()}; del bad["R"][0]
    res["bind_missing_slot"] = refuses(lambda: bind("q", bad, L, eligible=lambda j: True))
    res["bind_ineligible_job"] = refuses(lambda: bind("q", slots, L, eligible=lambda j: j != "JS1"))
    inc = f"{tmp}/S0.json"; good = open(inc).read(); json.dump(dict(config=dict(epochs=3, RIGHT=12, NORM="causal"), log=[1, 2], sets={"x": 1}), open(inc, "w"))
    res["bind_incomplete_run"] = refuses(lambda: bind("q", slots, L, eligible=lambda j: True)); open(inc, "w").write(good)
    elig_calls = []
    bind("q", slots, L, eligible=lambda j: elig_calls.append(j) or True)
    res["legacy_slot_skips_gate_only_for_listed"] = sorted(elig_calls) == ["JR0", "JS0", "JS1"]
    res["bind_twice"] = refuses(lambda: bind("q", slots, L, eligible=lambda j: True))
    open(f"{tmp}/S1.pt", "w").write("changed")
    res["changed_checkpoint_refused_in_child"] = refuses(lambda: release("pX", L, fake_map=fm))
    res["refused_release_leaves_panel_compromised"] = os.path.exists(f"{L}/pX.access.json") and refuses(lambda: release("pX", L, fake_map=fm))
    # fresh store for the success / concurrency / crash paths
    L2 = f"{tmp}/locks2"; os.makedirs(L2); P["panels"][0]["release_path"] = f"{L2}/pX.release.json"; json.dump(P, open(f"{tmp}/proto2.json", "w"))
    open(f"{tmp}/S1.pt", "w").write("S1"); register("q", f"{tmp}/proto2.json", L2); bind("q", slots, L2, eligible=lambda j: True)
    errs = []
    def go():
        try: errs.append(("ok", release("pX", L2, fake_map=fm)))
        except (AssertionError, FileExistsError) as e: errs.append(("refused", repr(e)[:60]))
    th = [threading.Thread(target=go) for _ in range(2)]; [t.start() for t in th]; [t.join() for t in th]
    res["concurrent_release_one_wins"] = sorted(k for k, _ in errs) == ["ok", "refused"]
    rows = [json.loads(l) for l in open(f"{L2}/pX.q.rows.jsonl")]
    res["rows_known_answer"] = len(rows) == 12 and sum(r["e"] for r in rows) == 3 * (0 + 1 + 1 + 2) and all(r["t"] == dict(zip(utts, T))[r["utt"]] for r in rows)
    rel = json.load(open(f"{L2}/pX.release.json")); res["release_digest_matches_entry"] = rel["items_sha256"] == q2d(items)
    res["second_release_refused"] = refuses(lambda: release("pX", L2, fake_map=fm))
    res["register_after_access_refused"] = refuses(lambda: register("q2", f"{tmp}/proto2.json", L2))
    # crash mid-release
    L3 = f"{tmp}/locks3"; os.makedirs(L3); P["panels"][0]["release_path"] = f"{L3}/pX.release.json"; json.dump(P, open(f"{tmp}/proto3.json", "w"))
    register("q", f"{tmp}/proto3.json", L3); bind("q", slots, L3, eligible=lambda j: True)
    cf = f"{tmp}/crash.json"; json.dump({"__crash__": 1}, open(cf, "w")); fm2 = dict(fm); fm2[("R", 0)] = cf
    res["crash_child_refused"] = refuses(lambda: release("pX", L3, fake_map=fm2))
    res["after_crash_compromised_no_retry"] = not os.path.exists(f"{L3}/pX.release.json") and refuses(lambda: release("pX", L3, fake_map=fm))
    # coverage: a fake child that drops one utt
    L4 = f"{tmp}/locks4"; os.makedirs(L4); P["panels"][0]["release_path"] = f"{L4}/pX.release.json"; json.dump(P, open(f"{tmp}/proto4.json", "w"))
    register("q", f"{tmp}/proto4.json", L4); bind("q", slots, L4, eligible=lambda j: True)
    df = f"{tmp}/drop.json"; json.dump({u: [0, t, t] for u, t in list(zip(utts, T))[:2]}, open(df, "w")); fm3 = dict(fm); fm3[("S", 1)] = df
    res["missing_item_refused"] = refuses(lambda: release("pX", L4, fake_map=fm3))
    # protocol edited after registration
    L5 = f"{tmp}/locks5"; os.makedirs(L5); json.dump(P, open(f"{tmp}/proto5.json", "w")); register("q", f"{tmp}/proto5.json", L5)
    P5 = dict(P); P5["seeds"] = [0]; json.dump(P5, open(f"{tmp}/proto5.json", "w"))
    res["protocol_edit_refused"] = refuses(lambda: bind("q", slots, L5, eligible=lambda j: True))
    res["child_env_refuses_warp"] = refuses(lambda: child_env(dict(RIGHT=12, NORM="causal", WARP="W")))
    res["child_env_refuses_feat"] = refuses(lambda: child_env(dict(RIGHT=12, NORM="causal", FEAT="2aN_")))
    res["ALL_OK"] = all(res.values())
    json.dump(res, open(HERE + "/reader_v65_selftest_2026-11-20.json", "w"), indent=1); print(json.dumps(res, indent=1))
    return res["ALL_OK"]


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "_child": _child(*sys.argv[2:8])
    elif c == "selftest": sys.exit(0 if selftest() else 1)
    elif c == "parity": print(json.dumps(parity(sys.argv[2], sys.argv[3])))
    elif c == "register": print(register(sys.argv[2], sys.argv[3]))
    elif c == "bind": print(bind(sys.argv[2], json.load(open(sys.argv[3]))))
    elif c == "release": print(release(sys.argv[2]))
