"""Independent item-level known answer (C9 5): per-utterance (e, t, h) of the trainer's evaluate(items=True), reached through the reader's
fresh child (reader_v65.score_slot), vs score_locked.score_items, a separately written per-item loop (own set reader, own decode wiring),
on the same checkpoint. argv: tag set. Run in a FRESH process with the run's RIGHT in the env (asserted)."""
import os
import sys, os, json, tempfile
H = "code"; sys.path[:0] = [H + "/step0", H]
import reader_v65 as RV, score_locked as SL
tag, sn = sys.argv[1], sys.argv[2]; O = H + "/phones_out/base6"; pt, rj = f"{O}/{tag}.pt", f"{O}/{tag}.json"
cfg = json.load(open(rj))["config"]; assert os.environ.get("RIGHT") == str(cfg["RIGHT"]), "set RIGHT before start"
sf = f"{os.environ.get('DATA_DIR', 'data')}/cache/base5_{sn}.npz"; o = tempfile.mktemp(suffix=".jsonl")
A = {r["utt"]: (r["e"], r["t"], r["h"]) for r in RV.score_slot(pt, RV.sha(pt), rj, sf, RV.sha(sf), o)}
m, dec, B, B6 = SL.load_model(dict(pt=pt, run_json=rj)); assert B.RIGHT == cfg["RIGHT"]
X, Y, files = SL.read_set(sf); it = SL.score_items(m, dec, B6.norm, X, Y, B.edit)
Bm = {f.split(".")[0]: z for f, z in zip(files, it)}
diff = [u for u in Bm if A.get(u) != tuple(Bm[u])]
R = dict(tag=tag, set=sn, n_reader=len(A), n_indep=len(Bm), same_keys=sorted(A) == sorted(Bm), items_differing=len(diff), first=diff[:5],
         e=(sum(v[0] for v in A.values()), sum(v[0] for v in Bm.values())))
R["PASS"] = R["same_keys"] and R["items_differing"] == 0; print(json.dumps(R))
json.dump(R, open(f"{os.path.dirname(os.path.abspath(__file__))}/indep_{tag}_{sn}.json", "w"), indent=1)
