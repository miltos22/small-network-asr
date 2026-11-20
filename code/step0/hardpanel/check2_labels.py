"""C2 0 d check (2) (v6.5, 20 Nov 2026, the agent): the UNCHANGED label pipeline (phones.job: TextGrid -> phone segments) reproduces the
cached training labels of the 20 fit utterances with the smallest sha256(utterance id) EXACTLY, against BOTH caches the trainers read
lineage-wise: ph_fit.npz segs (phones.build) and base5_fit.npz y_i (bb142.labelled -> base5_prep). Deliberately broken control: the
same comparison with one phone id changed must fail. Writes check2_labels_result.json next to this file. CPU only."""
import os, sys, json, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "code")
import numpy as np, phones
C = phones.CACHE; files = phones.split()["fit"]
d = np.load(f"{C}/ph_fit.npz"); lens = d["lens"]; assert len(lens) == len(files)
u0 = np.searchsorted(np.cumsum(lens), d["segs"][:, 0], side="right")
b5 = np.load(f"{C}/base5_fit.npz"); assert int(b5["n"]) == len(files)
uid = lambda p: os.path.basename(p).rsplit(".", 1)[0]
pick = sorted(range(len(files)), key=lambda i: hashlib.sha256(uid(files[i]).encode()).hexdigest())[:20]
rows = []; allok = True
for i in pick:
    r = phones.job(files[i]); seq = r[5][:, 2].astype(np.int64)
    a = d["segs"][u0 == i, 2].astype(np.int64); b = b5[f"y{i}"].astype(np.int64)
    ok = bool(np.array_equal(seq, a) and np.array_equal(seq, b) and len(r[1]) == lens[i])
    broken = seq.copy(); broken[len(broken) // 2] = (broken[len(broken) // 2] + 1) % 54
    rows.append(dict(utt=uid(files[i]), n_phones=int(len(seq)), frames=int(len(r[1])), cache_frames=int(lens[i]), equal=ok,
                     broken_control_detected=not np.array_equal(broken, a)))
    allok &= ok and rows[-1]["broken_control_detected"]
src = {f: hashlib.sha256(open(f"code/{f}", "rb").read()).hexdigest() for f in ("phones.py", "frontend.py", "bb142.py", "base5_prep.py")}
out = dict(rule="C2 0 d (2)", n=len(rows), all_equal=allok, rows=rows, code_sha256=src,
           caches={k: dict(mtime=os.path.getmtime(f"{C}/{k}"), size=os.path.getsize(f"{C}/{k}")) for k in ("ph_fit.npz", "base5_fit.npz")})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check2_labels_result.json"), "w"), indent=1)
print("n", len(rows), "ALL_EQUAL" if allok else "MISMATCH", [r["utt"] for r in rows if not r["equal"]])
