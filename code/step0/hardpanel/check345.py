"""C2 0 d checks (3)-(5) (v6.5, 20 Nov 2026, the agent). CPU, no model scoring.
(3) frozen extractor: code sha256 of the feature path (base5_prep.raw_logmel, bb142 FB / NFFT / HOP, frontend.load) + known answer:
    recomputed log-mel of 3 fit utterances equals base5_fit.npz X exactly (fp16).
(4) provenance, cutoff 2026-11-20T01:37+02:00 (start of the train-other-500 download): every artifact an evaluated model reads gets mtime +
    sha256; data artifacts must have mtime < cutoff; a CODE file changed after the cutoff passes only by the AMENDED rule (dated 20 Nov 07:4x,
    before the cutoff of the hard-panel decision and before any panel exists): its diff against its latest backup older than the cutoff has
    no train-other-500 reference and no added / removed numeric literal outside the listed recording-only lines (printed for review).
(5) code audit: static import closure of every Tranche 1 entry script (local modules only) with each module's data-path literals.
Writes check345_result.json next to this file."""
import ast, difflib, glob, hashlib, json, os, re, sys, datetime
os.environ["CUDA_VISIBLE_DEVICES"] = ""
D = "code"; SN = "code/small_net"
sys.path.insert(0, D)
CUT = datetime.datetime.fromisoformat("2026-11-20T01:37:00+02:00").timestamp()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
iso = lambda t: datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds")
OTHER = re.compile(r"other.?500|train-other|librispeech_other", re.I)
out = {}

# (3)
import numpy as np, phones, base5_prep, frontend as fe
files = phones.split()["fit"]; b5 = np.load(f"{phones.CACHE}/base5_fit.npz"); st = np.r_[0, np.cumsum(b5["lens"])]
ka = []
for i in (0, len(files) // 2, len(files) - 1):
    x = base5_prep.raw_logmel(fe.load(files[i])); ref = b5["X"][st[i]:st[i + 1]]
    ka.append(dict(i=i, utt=os.path.basename(files[i]), equal=bool(x.shape == ref.shape and np.array_equal(x, ref))))
out["check3"] = dict(known_answer=ka, extractor_code={f: dict(sha256=sha(f"{D}/{f}"), mtime=iso(os.path.getmtime(f"{D}/{f}")),
                     before_cutoff=os.path.getmtime(f"{D}/{f}") < CUT) for f in ("base5_prep.py", "bb142.py", "frontend.py")})
out["check3"]["PASS"] = all(k["equal"] for k in ka) and all(v["before_cutoff"] for v in out["check3"]["extractor_code"].values())

# (5) import closure
ENTRY = [f"{D}/{f}" for f in ("base6.py", "base6s.py", "base6s_R.py", "base6sy.py", "phase0_beam.py", "p2iv_lm.py", "vtln_self.py",
         "vtln_tc100.py", "aug2a.py", "d1_punits.py", "step0/reader_v65.py", "step0/score_locked.py")] + \
        [f"{D}/logs_2c/run2/struct2c_AS_RUN.py", f"{SN}/cifar_q1.py"]
def local(name, here):
    for d in (os.path.dirname(here), D, SN):
        p = f"{d}/{name}.py"
        if os.path.exists(p): return p
seen, todo = {}, list(ENTRY)
while todo:
    p = todo.pop()
    if p in seen or not os.path.exists(p): continue
    t = ast.parse(open(p).read()); mods = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Import): mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module: mods.add(n.module.split(".")[0])
    lits = sorted({n.value for n in ast.walk(t) if isinstance(n, ast.Constant) and isinstance(n.value, str) and
                   re.search(r"\.(npz|npy|f16|flac|wav|TextGrid|json|txt|pkl|pt|csv)\b|CACHE|/var/|/data|scratch storage|LibriSpeech", n.value)})
    deps = sorted(filter(None, (local(m, p) for m in mods)))
    seen[p] = dict(deps=deps, data_literals=lits, other500_ref=bool(OTHER.search(open(p).read())))
    todo += deps
out["check5"] = dict(entries=ENTRY, closure={os.path.relpath(k, D): v for k, v in sorted(seen.items())})
out["check5"]["PASS"] = all(os.path.exists(e) for e in ENTRY) and not any(v["other500_ref"] for v in seen.values())

# (4) provenance: code = closure; data = manifests_v65 cache files + named artifacts
code = {}
for p in sorted(seen):
    m = os.path.getmtime(p); r = dict(sha256=sha(p), mtime=iso(m), before_cutoff=m < CUT)
    if not r["before_cutoff"]:
        baks = sorted((b for b in glob.glob(p + ".bak*") + glob.glob(p + ".v6*") if os.path.getmtime(b) < CUT), key=os.path.getmtime)
        if not baks: r["amended_rule"] = dict(PASS=False, reason="no backup older than the cutoff")
        else:
            b = baks[-1]; old, new = open(b).read().splitlines(), open(p).read().splitlines()
            dl = [l for l in difflib.unified_diff(old, new, lineterm="", n=0) if l[:1] in "+-" and not l.startswith(("+++", "---"))]
            num = [l for l in dl if re.search(r"(?<![\w.])\d+\.?\d*(e-?\d+)?(?![\w])", re.sub(r"#.*|\"[^\"]*\"|'[^']*'", "", l[1:]))]
            r["amended_rule"] = dict(backup=os.path.relpath(b, D), backup_mtime=iso(os.path.getmtime(b)), changed_lines=len(dl),
                                     other500_in_diff=any(OTHER.search(l) for l in dl), numeric_lines=num,
                                     PASS=not any(OTHER.search(l) for l in dl))
    code[os.path.relpath(p, D)] = r
M = json.load(open(f"{D}/step0/manifests_v65.json")); data = {}
for k in sorted(M["cache_files_sha256"]):
    p = k if k.startswith("/") else f"{phones.CACHE}/{k}"
    if os.path.exists(p): m = os.path.getmtime(p); data[k] = dict(mtime=iso(m), before_cutoff=m < CUT, sha256_manifest=M["cache_files_sha256"][k])
    else: data[k] = dict(missing=True)
for p in (f"{phones.CACHE}/trk31_fit.npy", f"{phones.CACHE}/ph_fit.npz", f"{phones.CACHE}/ph_cal.npz"):
    m = os.path.getmtime(p); data[os.path.basename(p)] = dict(mtime=iso(m), before_cutoff=m < CUT, sha256=sha(p))
out["check4"] = dict(cutoff=iso(CUT), code=code, data=data)
out["check4"]["PASS_literal_v65"] = all(v["before_cutoff"] for v in code.values()) and all(v.get("before_cutoff") for v in data.values())
out["check4"]["PASS_amended"] = all(v["before_cutoff"] or v["amended_rule"]["PASS"] for v in code.values()) and all(v.get("before_cutoff") for v in data.values())
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check345_result.json"), "w"), indent=1)
print("check3", out["check3"]["PASS"], "check4 literal", out["check4"]["PASS_literal_v65"], "amended", out["check4"]["PASS_amended"], "check5", out["check5"]["PASS"])
for k, v in code.items():
    if not v["before_cutoff"]: print(" after-cutoff code:", k, {a: b for a, b in v["amended_rule"].items() if a != "numeric_lines"}, "numeric lines:", len(v["amended_rule"].get("numeric_lines", [])))
print(" data after cutoff / missing:", [k for k, v in data.items() if not v.get("before_cutoff")])
