"""GPU cap gate v2 (Part C v6.5; review round 5 C5.7). Replaces gpu_gate.py for every GPU job. The cap comes from the APPROVAL record, the
commitments from the APPROVED LEDGER, never from the caller.

Files (default in this folder):
  approved.json   {cap_h, option, ledger_path, ledger_sha256, reserve_frac}  written when the author approves; ABSENT = every job refused.
  gpu_spend.jsonl events {ev: "hist"|"begin"|"charge", job, gpu_h?, t?, interrupted?}; ABSENT = refused. Seeded with the historical spend.
Rule for job J (a row key of the approved ledger, scope inside the approved option):
  need = spent(all charges) + allowance(J) + (1 + reserve_frac) * sum(expected_gpu_h of every approved row not yet charged, J excluded)
  need <= cap_h, else refused. allowance(J) = the row's worst_gpu_h (the job runs under that hard timeout).
Refused also: ledger file digest != approved; J not an approved row / out of scope; J already charged; any job begun and not charged
(one GPU job at a time; run `recover` first); a caller cannot pass cap, allowance or commitments (no such arguments exist).
`run J -- cmd ...` = check, begin, run cmd under `timeout allowance`, charge exactly once (interrupted if rc != 0 or timeout).
`recover` charges every begun-but-uncharged job at min(allowance, now - begin) as INTERRUPTED; interrupted jobs are never scoring-eligible
(`eligible J` returns False). Runners must launch GPU work only through `run` (audit: grep the runner for 'gpu_gate2.py run')."""
import os
import hashlib, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = "scratch/heartbeat/GPU.lockdir"  # the shared GPU lock (C9 6)
KILL_GRACE_S = 60


def _load(d):
    ap = f"{d}/approved.json"; sp = f"{d}/gpu_spend.jsonl"
    assert os.path.exists(ap), "no approval record (approved.json): every GPU job refused"
    assert os.path.exists(sp), "no spend record (gpu_spend.jsonl): every GPU job refused"
    A = json.load(open(ap))
    assert hashlib.sha256(open(A["ledger_path"], "rb").read()).hexdigest() == A["ledger_sha256"], "ledger changed after approval"
    scopes = {"R": ["R"], "R+H1": ["R", "H1"], "R+H": ["R", "H1", "H2"]}[A["option"]]
    rows = {r["key"]: r for r in json.load(open(A["ledger_path"]))["rows"] if r["stage"] != "spent" and r["scope"] in scopes}
    ev = [json.loads(l) for l in open(sp) if l.strip()]
    return A, rows, ev, sp


def _state(ev):
    charged = {e["job"] for e in ev if e["ev"] == "charge"}
    open_ = [e for e in ev if e["ev"] == "begin" and e["job"] not in charged]
    spent = sum(e["gpu_h"] for e in ev if e["ev"] in ("hist", "charge"))
    return charged, open_, spent


def check(job, d=HERE):
    A, rows, ev, _ = _load(d); charged, open_, spent = _state(ev)
    assert job in rows, f"GATE REFUSED {job}: not an approved row in option {A['option']}"
    assert job not in charged, f"GATE REFUSED {job}: already charged"
    assert not open_, f"GATE REFUSED {job}: job(s) begun and not charged {[e['job'] for e in open_]} (run recover)"
    allow = float(rows[job]["worst_gpu_h"])
    commit = (1 + float(A["reserve_frac"])) * sum(float(r["expected_gpu_h"]) for k, r in rows.items() if k != job and k not in charged)
    need = spent + allow + commit
    assert need <= float(A["cap_h"]) + 1e-9, f"GATE REFUSED {job}: spent {spent:.4f} + allowance {allow:.4f} + commitments {commit:.4f} = {need:.4f} > cap {A['cap_h']}"
    return dict(job=job, spent=spent, allowance=allow, commitments=commit, need=need, cap=A["cap_h"], headroom=float(A["cap_h"]) - need)


def _append(sp, e):
    with open(sp, "a") as f: f.write(json.dumps(e) + "\n"); f.flush(); os.fsync(f.fileno())


def begin(job, d=HERE, t=None):
    c = check(job, d); _append(f"{d}/gpu_spend.jsonl", dict(ev="begin", job=job, t=time.time() if t is None else t, allowance=c["allowance"])); return c


def charge(job, gpu_h, interrupted, d=HERE):
    _, _, ev, sp = _load(d); charged, open_, _ = _state(ev)
    assert job not in charged, f"double charge refused: {job}"
    assert any(e["job"] == job for e in open_), f"charge without begin refused: {job}"
    _append(sp, dict(ev="charge", job=job, gpu_h=float(gpu_h), interrupted=bool(interrupted)))


def recover(d=HERE, now=None):
    _, _, ev, _ = _load(d); _, open_, _ = _state(ev); now = time.time() if now is None else now
    for e in open_: charge(e["job"], min(e["allowance"], (now - e["t"]) / 3600), True, d)
    return [e["job"] for e in open_]


def eligible(job, d=HERE):
    _, _, ev, _ = _load(d)
    c = [e for e in ev if e["ev"] == "charge" and e["job"] == job]
    return len(c) == 1 and not c[0]["interrupted"]


def _lock_owned(lock, owner):
    """C9 6: a GPU job starts only while the caller holds the shared GPU.lockdir (owner file line starts with GATE_OWNER)."""
    assert owner, "GATE REFUSED: no GATE_OWNER (the runner's take() must export it)"
    assert os.path.isdir(lock), f"GATE REFUSED: GPU lock {lock} not held"
    first = open(f"{lock}/owner").read().split()
    assert first and first[0] == owner, f"GATE REFUSED: GPU lock owned by {first[:1]}, not {owner}"


def run(job, cmd, d=HERE, lock=LOCK, owner=None):
    """check + begin + run cmd in its OWN process group under the allowance; on timeout the whole group gets TERM, then KILL after
    KILL_GRACE_S (data-loader workers and grandchildren included); charge exactly once (interrupted if rc != 0 or timeout)."""
    _lock_owned(lock, owner if owner is not None else os.environ.get("GATE_OWNER", ""))
    c = begin(job, d); t0 = time.time(); rc = None
    try:
        p = subprocess.Popen(cmd, start_new_session=True)
        try: rc = p.wait(timeout=c["allowance"] * 3600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 15)
            try: p.wait(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired: pass
            try: os.killpg(p.pid, 9)
            except ProcessLookupError: pass
            p.wait(); rc = 124
        if rc == 0:  # a group member still alive after the leader exited is a fault too
            try: os.killpg(p.pid, 0); os.killpg(p.pid, 9); rc = 125
            except ProcessLookupError: pass
    finally:
        try: os.killpg(p.pid, 9)
        except (ProcessLookupError, NameError, UnboundLocalError): pass
        charge(job, (time.time() - t0) / 3600, rc != 0, d)
    return rc


def selftest():
    import tempfile
    d = tempfile.mkdtemp(); L = f"{d}/ledger.json"
    rows = [dict(key=f"J{i}", stage="primary", scope="R", expected_gpu_h=1.0, worst_gpu_h=1.5) for i in range(4)] + \
           [dict(key="H", stage="primary", scope="H1", expected_gpu_h=5.0, worst_gpu_h=5.0), dict(key="old", stage="spent", scope="X", expected_gpu_h=0.85, worst_gpu_h=0.85)]
    json.dump(dict(rows=rows), open(L, "w")); sha = hashlib.sha256(open(L, "rb").read()).hexdigest()
    R = {}
    def refused(f):
        try: f(); return False
        except (AssertionError, KeyError): return True
    R["no_approval_refused"] = refused(lambda: check("J0", d))
    # cap exactly at the edge: hist 0.85 + allow 1.5 + 1.1 * 3 = 5.65
    json.dump(dict(cap_h=5.65, option="R", ledger_path=L, ledger_sha256=sha, reserve_frac=0.1), open(f"{d}/approved.json", "w"))
    R["no_spend_record_refused"] = refused(lambda: check("J0", d))
    open(f"{d}/gpu_spend.jsonl", "w").write(json.dumps(dict(ev="hist", job="old", gpu_h=0.85)) + "\n")
    R["at_cap_allowed"] = abs(check("J0", d)["headroom"]) < 1e-9
    json.dump(dict(cap_h=5.64, option="R", ledger_path=L, ledger_sha256=sha, reserve_frac=0.1), open(f"{d}/approved.json", "w"))
    R["just_beyond_cap_refused_with_history_counted"] = refused(lambda: check("J0", d))
    json.dump(dict(cap_h=5.65, option="R", ledger_path=L, ledger_sha256=sha, reserve_frac=0.1), open(f"{d}/approved.json", "w"))
    R["out_of_scope_row_refused"] = refused(lambda: check("H", d))
    R["unknown_job_refused"] = refused(lambda: check("ZZ", d))
    R["caller_override_impossible"] = check.__code__.co_varnames[:check.__code__.co_argcount] == ("job", "d")  # no cap / allowance args
    begin("J0", d, t=0.0)
    R["second_job_while_one_open_refused"] = refused(lambda: check("J1", d))
    R["recover_charges_interrupted"] = recover(d, now=3600 * 9.0) == ["J0"] and not eligible("J0", d)
    ev = [json.loads(l) for l in open(f"{d}/gpu_spend.jsonl")]
    R["recover_charge_capped_at_allowance"] = [e for e in ev if e["ev"] == "charge"][0]["gpu_h"] == 1.5
    R["double_charge_refused"] = refused(lambda: charge("J0", 1.0, False, d))
    R["rerun_of_charged_job_refused"] = refused(lambda: check("J0", d))
    # after J0 (1.5 charged): 0.85 + 1.5 + allow 1.5 + 1.1*2 = 6.05 > 5.65 -> overrun of J0 is felt by the next job
    R["overrun_propagates_refusal"] = refused(lambda: check("J1", d))
    json.dump(dict(cap_h=7.0, option="R", ledger_path=L, ledger_sha256=sha, reserve_frac=0.1), open(f"{d}/approved.json", "w"))
    lk = f"{d}/GPU.lockdir"
    R["run_without_lock_refused"] = refused(lambda: run("J1", ["true"], d, lock=lk, owner="the agent-t"))
    os.mkdir(lk); open(f"{lk}/owner", "w").write("the agent-x 2026-11-20T08:00\n")
    R["run_with_other_owner_refused"] = refused(lambda: run("J1", ["true"], d, lock=lk, owner="the agent-t"))
    R["run_without_owner_refused"] = refused(lambda: run("J1", ["true"], d, lock=lk, owner=""))
    R["lock_refusals_charge_nothing"] = not any(e["job"] == "J1" for e in map(json.loads, open(f"{d}/gpu_spend.jsonl")))
    open(f"{lk}/owner", "w").write("the agent-t 2026-11-20T08:00\n")
    rc = run("J1", ["true"], d, lock=lk, owner="the agent-t"); R["run_ok_charged_once_eligible"] = rc == 0 and eligible("J1", d)
    rc = run("J2", ["false"], d, lock=lk, owner="the agent-t"); R["run_fail_interrupted_not_eligible"] = rc != 0 and not eligible("J2", d)
    # timeout + process-group kill: separate ledger with a 1.8 s allowance; the command leaves a grandchild that ignores nothing
    d2 = tempfile.mkdtemp(); L2 = f"{d2}/ledger.json"
    json.dump(dict(rows=[dict(key="T", stage="primary", scope="R", expected_gpu_h=0.0005, worst_gpu_h=0.0005)]), open(L2, "w"))
    json.dump(dict(cap_h=1.0, option="R", ledger_path=L2, ledger_sha256=hashlib.sha256(open(L2, "rb").read()).hexdigest(), reserve_frac=0.1),
              open(f"{d2}/approved.json", "w")); open(f"{d2}/gpu_spend.jsonl", "w").write("")
    pidf = f"{d2}/gc.pid"; t0 = time.time()
    rc = run("T", ["bash", "-c", f"sleep 300 & echo $! > {pidf}; wait"], d2, lock=lk, owner="the agent-t")
    gc = int(open(pidf).read()); time.sleep(0.5)
    try: os.kill(gc, 0); alive = True
    except ProcessLookupError: alive = False
    R["timeout_kills_group_charged_interrupted"] = rc == 124 and not alive and not eligible("T", d2) and time.time() - t0 < 30
    open(L, "a").write(" "); R["ledger_edit_after_approval_refused"] = refused(lambda: check("J3", d))
    R["ALL_OK"] = all(R.values())
    return R


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "selftest":
        R = selftest(); print(json.dumps(R, indent=1)); sys.exit(0 if R["ALL_OK"] else 1)
    elif a[0] == "check": print(check(a[1]))
    elif a[0] == "recover": print(recover())
    elif a[0] == "eligible": print(eligible(a[1]))
    elif a[0] == "run": assert a[2] == "--"; sys.exit(run(a[1], a[3:]))
