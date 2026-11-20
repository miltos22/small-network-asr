"""C9 6 launch-path audit (v6.5, 20 Nov 2026, the agent). python -I audit_launch.py <runner.sh ...> | selftest
A runner PASSES only if: it sources step0/gpu_runner_lib.sh; it defines no take/rel/gjob of its own; every line that starts the GPU
venv's python (python3, or a variable assigned from it) is a `gjob <key> ...` line; it has no other launch (nohup/setsid/&/eval
/timeout wrappers around a GPU python, or `gpu_gate2.py run` bypassing gjob is allowed only via gjob); CUDA_VISIBLE_DEVICES is set
only to 0. Exit 1 on any failure (printed with line numbers)."""
import re, sys, os, tempfile, json

LIB = "gpu_runner_lib.sh"; GPU_PY = "python3"


def audit(path):
    src = open(path).read().splitlines(); errs = []
    if not any(re.match(rf"\s*(source|\.)\s+\S*{LIB}\b", l) for l in src): errs.append((0, "does not source " + LIB))
    gvars = set()
    while True:  # transitive: a variable assigned from the venv path or from another GPU variable is a GPU variable
        new = {m.group(1) for l in src for m in [re.match(r"\s*(?:export\s+)?([A-Za-z_]\w*)=(.*)", l)] if m and
               (GPU_PY in m.group(2) or any(re.search(r"\$\{?" + v + r"\b", m.group(2)) for v in gvars))}
        if new <= gvars: break
        gvars |= new
    for i, l in enumerate(src, 1):
        s = l.split("#", 1)[0] if not l.lstrip().startswith("#!") else ""
        if not s.strip(): continue
        if re.search(r"(^|[;&|]\s*)(take|rel|gjob)\s*\(\)", s): errs.append((i, "redefines take/rel/gjob"))
        m = re.search(r"CUDA_VISIBLE_DEVICES=(\S*)", s)
        if m and m.group(1).strip("\"'") != "0": errs.append((i, "CUDA_VISIBLE_DEVICES not 0"))
        if re.match(r"\s*(export\s+)?[A-Za-z_]\w*=\S*\s*$", s) or re.match(r"\s*(export\s+)?[A-Za-z_]\w*=\"[^\"]*\"\s*$", s): continue  # assignment only
        uses = GPU_PY in s or any(re.search(r"\$\{?" + v + r"\b", s) for v in gvars)
        if uses and not re.match(r"\s*gjob\s+\S+\s", s): errs.append((i, "GPU python launched outside gjob: " + s.strip()[:80]))
        if uses and re.search(r"\b(nohup|setsid|eval)\b|&\s*$", s): errs.append((i, "detached / eval GPU launch"))
        if "gpu_gate2.py" in s and "run" in s: errs.append((i, "direct gpu_gate2 run (use gjob)"))
    return errs


def selftest():
    d = tempfile.mkdtemp(); R = {}
    hdr = "#!/bin/bash\nsource /x/step0/gpu_runner_lib.sh\nPY=\"/opt/python3/bin/python -I\"\nexport CUDA_VISIBLE_DEVICES=0\n"
    cases = dict(
        good=hdr + "take q2c 120\ngjob Q2c_S_s0 $PY base6s.py S q2c_S_s0 0 1836\nrel\n",
        no_lib="#!/bin/bash\nPY=\"/opt/python3/bin/python -I\"\ngjob K $PY a.py\n",
        direct=hdr + "take q 1\n$PY base6s.py S x 0 60\nrel\n",
        direct_literal=hdr + "/opt/python3/bin/python -I base6s.py S x 0 60\n",
        nohup=hdr + "gjob K nohup $PY a.py &\n",
        own_take=hdr + "take(){ mkdir /tmp/l; }\ngjob K $PY a.py\n",
        gate_direct=hdr + "python gpu_gate2.py run K -- $PY a.py\n",
        other_gpu=hdr + "export CUDA_VISIBLE_DEVICES=1\ngjob K $PY a.py\n",
        alias_var=hdr + "P2=$PY\n$P2 a.py\n")
    for k, t in cases.items():
        p = f"{d}/{k}.sh"; open(p, "w").write(t); R[k] = audit(p)
    # alias_var: P2 is assigned from $PY, not from the venv path -> must still be caught (second pass below)
    ok = {k: (not v) == (k == "good") for k, v in R.items()}
    return dict(results={k: [list(e) for e in v] for k, v in R.items()}, ok=ok, ALL_OK=all(ok.values()))


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        r = selftest(); print(json.dumps(r, indent=1))
        json.dump(r, open(os.path.dirname(os.path.abspath(__file__)) + "/audit_launch_selftest_2026-11-20.json", "w"), indent=1)
        sys.exit(0 if r["ALL_OK"] else 1)
    bad = 0
    for p in sys.argv[1:]:
        e = audit(p); print(p, "PASS" if not e else "FAIL"); bad += bool(e)
        for i, m in e: print(f"  line {i}: {m}")
    sys.exit(1 if bad else 0)
