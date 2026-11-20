"""C9 1 winner binding for tuned rows (v6.5, 20 Nov 2026, the agent).

Two records per tuned row, both written once (O_EXCL), both hashed:
  1. FREEZE (before the grid's first run): candmap_<row>.json = ordered candidates (each a dict of config keys, its sha256),
     the seeds, the rule file path + sha256 (a python file defining choose(rows) -> candidate index), the validation sets the rule
     may read. Written before any run of the grid exists; a second freeze is refused.
  2. BIND (after the grid): winner_<row>.json. Inputs: the candmap and one run record per (candidate, seed). The binder
     a. refuses if the rule file's sha256 changed since the freeze,
     b. refuses a run record whose config does not equal its candidate on every candidate key, a seed outside the frozen seeds,
        a missing or duplicate (candidate, seed), a SMOKE / STOP_AFTER record,
     c. refuses a record that scored any set outside the validation list (sealed or exposed panels must never reach a winner choice),
     d. calls choose() on validation numbers read from the records themselves (never typed numbers),
     e. refuses a returned index that is not a frozen candidate.
  check_launch(row, cfg): at launch of the row's own job, cfg must equal the bound winner (by sha256), else refused.
Run: python -I winner_bind.py selftest
"""
import hashlib, json, os, sys, tempfile, importlib.util

VALID = ("cleanS", "libriA")


def _sha(b): return hashlib.sha256(b).hexdigest()
def _cfg_sha(c): return _sha(json.dumps(c, sort_keys=True).encode())
def _fsha(p): return _sha(open(p, "rb").read())


def _write_once(p, obj):
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)  # FileExistsError = refused
    with os.fdopen(fd, "w") as f: json.dump(obj, f, indent=1, sort_keys=True)


def freeze(row, candidates, seeds, rule_path, d):
    assert candidates and len({_cfg_sha(c) for c in candidates}) == len(candidates), "empty or duplicate candidates"
    assert seeds and len(set(seeds)) == len(seeds), "seeds"
    p = f"{d}/candmap_{row}.json"
    _write_once(p, dict(row=row, candidates=[dict(cfg=c, sha256=_cfg_sha(c)) for c in candidates], seeds=list(seeds),
                        rule_path=os.path.abspath(rule_path), rule_sha256=_fsha(rule_path), valid_sets=list(VALID)))
    return p


def bind(row, records, d):
    """records: list of run-record paths (one per candidate x seed)."""
    cp = f"{d}/candmap_{row}.json"; M = json.load(open(cp))
    assert _fsha(M["rule_path"]) == M["rule_sha256"], "rule file changed after the freeze"
    keys = sorted({k for c in M["candidates"] for k in c["cfg"]})
    got = {}
    for rp in records:
        R = json.load(open(rp)); cfg = R["config"]
        assert not cfg.get("SMOKE") and not R.get("stop_after"), ("smoke / prefix record", rp)
        assert len(R.get("log", [])) == cfg.get("epochs"), ("record does not cover its whole schedule", rp)
        hit = [i for i, c in enumerate(M["candidates"]) if all(cfg.get(k) == c["cfg"].get(k) for k in keys)]
        assert len(hit) == 1, ("record matches no frozen candidate (or several)", rp)
        s = cfg.get("seed"); assert s in M["seeds"], ("seed not frozen", rp, s)
        assert (hit[0], s) not in got, ("duplicate candidate/seed", rp)
        extra = set(R["sets"]) - set(M["valid_sets"]) - {"fit"}
        assert not extra, ("record scored sets outside validation", rp, sorted(extra))
        got[(hit[0], s)] = dict(per={k: R["sets"][k]["PER"] for k in M["valid_sets"]}, path=os.path.abspath(rp), sha256=_fsha(rp))
    need = {(i, s) for i in range(len(M["candidates"])) for s in M["seeds"]}
    assert set(got) == need, ("missing runs", sorted(need - set(got)))
    spec = importlib.util.spec_from_file_location("rule_" + row, M["rule_path"]); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [dict(candidate=i, seed=s, **got[(i, s)]["per"]) for (i, s) in sorted(got)]
    w = mod.choose(rows)
    assert isinstance(w, int) and 0 <= w < len(M["candidates"]), ("rule returned a non-candidate", w)
    rec = dict(row=row, candmap_sha256=_fsha(cp), rule_sha256=M["rule_sha256"], winner=w, winner_cfg=M["candidates"][w]["cfg"],
               winner_sha256=M["candidates"][w]["sha256"], runs=[dict(candidate=i, seed=s, **got[(i, s)]) for (i, s) in sorted(got)])
    p = f"{d}/winner_{row}.json"; _write_once(p, rec); return p, w


def check_launch(row, cfg, d):
    W = json.load(open(f"{d}/winner_{row}.json"))
    assert _fsha(f"{d}/candmap_{row}.json") == W["candmap_sha256"], "candmap changed after binding"
    assert _cfg_sha(cfg) == W["winner_sha256"], "launch config is not the bound winner"
    return True


def selftest():
    out = {}; d = tempfile.mkdtemp(prefix="wbind.")
    rule = f"{d}/rule.py"
    open(rule, "w").write("def choose(rows):\n    import collections\n    m = collections.defaultdict(list)\n"
                          "    for r in rows: m[r['candidate']].append(r['cleanS'])\n"
                          "    return min(sorted(m), key=lambda i: sum(m[i]) / len(m[i]))\n")
    C = [dict(lr=1e-3, wd=0.0), dict(lr=3e-3, wd=0.0), dict(lr=1e-2, wd=0.0)]
    def rec(name, cfg, seed, cs, la, extra=None, **kw):
        sets = dict(fit=dict(PER=10.0), cleanS=dict(PER=cs), libriA=dict(PER=la)); sets.update(extra or {})
        c = dict(cfg, seed=seed, arm="S", SMOKE=0, epochs=3); p = f"{d}/{name}.json"
        json.dump(dict(config=c, sets=sets, log=[{}] * kw.get("nlog", 3), stop_after=kw.get("stop_after", 0)), open(p, "w")); return p
    def T(k, f):
        try: f(); out[k] = "accepted"
        except (AssertionError, FileExistsError, KeyError) as e: out[k] = "refused: " + str(e)[:90]
    freeze("q", C, [0, 1], rule, d)
    T("second_freeze", lambda: freeze("q", C, [0, 1], rule, d))
    good = [rec(f"r{i}s{s}", C[i], s, [30, 28, 29][i] + s * 0.1, 40.0) for i in range(3) for s in (0, 1)]
    T("missing_run", lambda: bind("q", good[:-1], d))
    T("non_candidate_record", lambda: bind("q", good[:-1] + [rec("bad", dict(lr=5e-3, wd=0.0), 1, 20, 40)], d))
    T("sealed_set_scored", lambda: bind("q", good[:-1] + [rec("seal", C[2], 1, 29.1, 40, dict(cleanT=dict(PER=1.0)))], d))
    T("exposed_set_scored", lambda: bind("q", good[:-1] + [rec("expo", C[2], 1, 29.1, 40, dict(testother=dict(PER=1.0)))], d))
    T("prefix_record", lambda: bind("q", good[:-1] + [rec("pre", C[2], 1, 29.1, 40, stop_after=6)], d))
    T("short_log", lambda: bind("q", good[:-1] + [rec("short", C[2], 1, 29.1, 40, nlog=2)], d))
    T("wrong_seed", lambda: bind("q", good[:-1] + [rec("s7", C[2], 7, 29.1, 40)], d))
    T("duplicate", lambda: bind("q", good + [good[0]], d))
    # rule changed after freeze
    src = open(rule).read(); open(rule, "w").write(src.replace("min(", "max(")); T("changed_rule", lambda: bind("q", good, d))
    open(rule, "w").write(src)
    # rule returning a non-candidate
    rule2 = f"{d}/rule2.py"; open(rule2, "w").write("def choose(rows):\n    return 7\n"); freeze("q2", C, [0, 1], rule2, d)
    T("rule_returns_non_candidate", lambda: bind("q2", good, d))
    # known answer: candidate 1 (cleanS 28.0 / 28.1) wins
    p, w = bind("q", good, d); out["known_answer_winner"] = w
    T("second_bind", lambda: bind("q", good, d))
    T("launch_winner", lambda: check_launch("q", C[1], d))
    T("launch_other", lambda: check_launch("q", C[0], d))
    exp_acc = {"launch_winner"}
    ok = out["known_answer_winner"] == 1 and all((v == "accepted") == (k in exp_acc) for k, v in out.items() if k != "known_answer_winner")
    out["ALL_OK"] = ok; return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        r = selftest(); print(json.dumps(r, indent=1))
        json.dump(r, open(os.path.dirname(os.path.abspath(__file__)) + "/winner_bind_selftest_2026-11-20.json", "w"), indent=1)
        sys.exit(0 if r["ALL_OK"] else 1)
