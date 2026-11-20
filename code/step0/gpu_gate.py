"""All-job GPU cap gate (MASTER_PLAN Part C v6.3 C5.7; review round 3 C5.7). Every Tranche 1 GPU job is launched through this gate; a runner
calls `gpu_gate.py check <job> <allowance_h> <protected_h>` before the GPU lock and `gpu_gate.py charge <job> <start_epoch> <end_epoch>`
after the job exits (charged = measured wall time, whatever the outcome; a killed or failed job is charged too).
Rule (asserted, fail closed):   charged spend + this job's allowance + protected remaining commitments  <=  approved cap.
  allowance  = the job's ENFORCED wall-time limit: the runner wraps the job in `timeout <allowance>`, so in-flight spend cannot exceed it;
               allowance = worst ledger price of the job (ledger_v63.json worst_gpu_h) unless a smaller value is given.
  protected  = expected price of the unstarted rows the plan never drops (C5 6 never-drop list: the primaries' remaining runs).
  cap        = the approved Tranche 1 amount in step0/gpu_cap.json (written only from the author's review note answer; absent = refuse every job).
A job that hits its timeout is recorded as INTERRUPTED; its comparison stays open (never read as complete) until a new review note.
Files: step0/gpu_cap.json {"cap_h": float, "review note": "<review file>", "set": "<iso time>"}, step0/gpu_spend.jsonl (append-only).
Usage: python -I gpu_gate.py check JOB ALLOW_H PROTECTED_H | charge JOB T0 T1 [interrupted] | status | selftest"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))


def spent(f):
    # round 5 C5.7: a missing spend record is a refusal, never 0 h (the 0.85 h already spent is seeded from ledger_v64 'spent' rows)
    assert os.path.exists(f), f"no spend record {f}: every GPU job refused"
    return sum(json.loads(l)["gpu_h"] for l in open(f))


def check(job, allow_h, protected_h, cap_f=f"{HERE}/gpu_cap.json", spend_f=f"{HERE}/gpu_spend.jsonl"):
    assert os.path.exists(cap_f), "no approved cap (gpu_cap.json): every GPU job refused"
    cap = json.load(open(cap_f))["cap_h"]; s = spent(spend_f); need = s + float(allow_h) + float(protected_h)
    assert allow_h > 0 and protected_h >= 0
    assert need <= cap + 1e-9, f"GATE REFUSED {job}: spent {s:.3f} + allowance {allow_h} + protected {protected_h} = {need:.3f} > cap {cap}"
    return dict(job=job, spent=round(s, 3), allowance=allow_h, protected=protected_h, cap=cap, headroom=round(cap - need, 3))


def charge(job, t0, t1, interrupted=False, spend_f=f"{HERE}/gpu_spend.jsonl"):
    h = (float(t1) - float(t0)) / 3600; assert h >= 0
    with open(spend_f, "a") as f: f.write(json.dumps(dict(job=job, t0=float(t0), t1=float(t1), gpu_h=round(h, 4), interrupted=bool(interrupted))) + "\n")
    return h


def selftest():
    import tempfile
    d = tempfile.mkdtemp(); cap, sp = f"{d}/cap.json", f"{d}/spend.jsonl"
    try: check("j", 1, 0, cap, sp); raise SystemExit("FAIL: ran without a cap")
    except AssertionError: pass
    json.dump(dict(cap_h=10.0), open(cap, "w"))
    try: check("j0", 1, 0, cap, sp); raise SystemExit("FAIL: ran without a spend record")
    except AssertionError: pass
    open(sp, "w").write(json.dumps(dict(job="history", gpu_h=0.0)) + "\n")
    assert check("j1", 2.0, 5.0, cap, sp)["headroom"] == 3.0
    charge("j1", 0, 3 * 3600, False, sp)                        # charged 3 h (over its forecast: measured time counts)
    assert check("j2", 2.0, 5.0, cap, sp)["headroom"] == 0.0     # 3 + 2 + 5 = 10: allowed at the edge
    try: check("j3", 2.5, 5.0, cap, sp); raise SystemExit("FAIL: over-cap job allowed")
    except AssertionError: pass
    print("selftest OK: no cap -> refuse; no spend record -> refuse; edge allowed; over cap refused; spend is measured wall time")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "selftest": selftest()
    elif a[0] == "check": print(json.dumps(check(a[1], float(a[2]), float(a[3]))))
    elif a[0] == "charge": print(charge(a[1], a[2], a[3], len(a) > 4))
    elif a[0] == "status": print(dict(spent=spent(f"{HERE}/gpu_spend.jsonl"), cap=json.load(open(f"{HERE}/gpu_cap.json")) if os.path.exists(f"{HERE}/gpu_cap.json") else None))
