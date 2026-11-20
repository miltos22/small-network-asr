"""Step 0.1 speaker-exposure audit (MASTER_PLAN 4b.4/4b.5, 20 Nov). CPU, read-only on data. Lists LibriSpeech pool speakers, which of our sets
use them, and how often each old eval set name appears in results/logs (a LOWER-BOUND count of scorings). Writes step0/exposure_audit.json."""
import os
import json, os, glob, re, collections, numpy as np
C = os.environ.get("DATA_DIR", "data") + "/cache"; LS = os.environ.get("DATA_DIR", "data") + "/LibriSpeech"
W = "code"
spk = lambda f: os.path.basename(str(f)).split("-")[0]
pools = {p: sorted(os.listdir(f"{LS}/{p}")) for p in os.listdir(LS) if os.path.isdir(f"{LS}/{p}")}
S = json.load(open(f"{C}/base5_speakers.json"))
used = {k: sorted(set(S[k])) for k in ("fit", "cal", "libriA", "libriB")}
used["testother"] = sorted({spk(f) for f in np.load(f"{C}/realcut_testother.npz", allow_pickle=True)["files"]})
used["cleanS"] = sorted({spk(f) for f in np.load(f"{C}/base5_cleanS.npz", allow_pickle=True)["files"]})
if os.path.exists(f"{C}/base5_cleanT.npz"): used["cleanT"] = sorted({spk(f) for f in np.load(f"{C}/base5_cleanT.npz", allow_pickle=True)["files"]})
used["tc100_train"] = sorted({spk(f) for f in np.load(f"{C}/base5_tc100_meta.npz", allow_pickle=True)["files"]})
used["devother_mixB_pool"] = sorted({spk(f) for f in np.load(f"{C}/base5_devother_meta.npz", allow_pickle=True)["files"]})
where = {k: {p: len(set(v) & set(pools[p])) for p in pools if set(v) & set(pools[p])} for k, v in used.items()}
eval_used = set(used["libriA"]) | set(used["libriB"]) | set(used["testother"])
out = dict(pools={p: len(v) for p, v in pools.items()}, sets={k: dict(n=len(v), pools=where[k]) for k, v in used.items()})
out["dev_other_unused_by_eval"] = sorted(set(pools.get("dev-other", [])) - eval_used)
out["test_other_unused_by_eval"] = sorted(set(pools.get("test-other", [])) - eval_used)
out["test_clean_note"] = "cleanT built 17 Nov from all test-clean speakers; scoring count below"
# exposure: count result rows / log lines naming each set (lower bound; one line may hold many models)
cnt = collections.Counter(); files_seen = 0
for f in glob.glob(f"{W}/**/*.json*", recursive=True) + glob.glob(f"{W}/logs_base*/**/*.log", recursive=True) + [f"{W}/logs_base5/STAGE1_RESULTS.md"]:
    try: t = open(f, errors="ignore").read()
    except Exception: continue
    files_seen += 1
    for k in ("libriA", "libriB", "testother", "cleanS", "cleanT", "author2"): cnt[k] += len(re.findall(rf"\b{k}\b", t))
out["mentions_lower_bound"] = dict(cnt); out["files_scanned"] = files_seen
json.dump(out, open(f"{W}/step0/exposure_audit.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items()}, indent=0)[:3000])
