"""End-to-end refusal + known-answer tests for sec2_q2c_entry.decide_q2c (review round 5 C4). Writes results/sec2_q2c_entry_test.json."""
import copy, json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec2_abs as SA, sec2_q2c_entry as Q

D = tempfile.mkdtemp(); rng = np.random.default_rng(0)


def build(n_spk=(40, 10), utt_per=4, effect=-1.0):
    panels = []
    for k, n in enumerate(n_spk):
        items = [dict(utt=f"p{k}_s{c}_u{j}", speaker=f"p{k}_s{c}", t=int(rng.integers(40, 90))) for c in range(n) for j in range(utt_per)]
        rel = f"{D}/rel_{k}.json"; json.dump(dict(panel=f"P{k}", items_sha256=Q.items_digest(items)), open(rel, "w"))
        panels.append(dict(name=f"P{k}", panel_weight=0.5, items=items, release_path=rel))
    prot = f"{D}/protocol.json"; json.dump(dict(arms=["S", "R"], seeds=[0, 1, 2, 3, 4], panels=panels), open(prot, "w"))
    reg = f"{D}/register.json"; json.dump(dict(protocol_path=prot, protocol_sha256=Q.sha(prot)), open(reg, "w"))
    rows = []
    for p in panels:
        for s in range(5):
            for a in ("S", "R"):
                for i in p["items"]:
                    rate = 0.20 + (effect / 100 if a == "S" else 0) + rng.normal(0, 0.02)
                    rows.append(dict(panel=p["name"], arm=a, seed=s, utt=i["utt"], speaker=i["speaker"], e=int(max(0, round(rate * i["t"]))), t=i["t"]))
    return reg, prot, rows


def write(rows, name="rows.jsonl"):
    p = f"{D}/{name}"; open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows)); return p


def refused(reg, rows):
    try: Q.decide_q2c(reg, write(rows)); return False
    except (AssertionError, KeyError): return True


res = {}
reg, prot, rows = build()
ok = Q.decide_q2c(reg, write(rows))
direct = SA.decide(SA.from_counts(rows, ["P0", "P1"]), [0.5, 0.5])
res["known_answer_equals_direct"] = abs(ok["Delta"] - direct["Delta"]) < 1e-12 and ok["verdict"] == direct["verdict"]
res["known_answer_verdict"] = ok["verdict"]; res["Delta"] = ok["Delta"]
res["dup_row_refused"] = refused(reg, rows + [rows[0]])
u0 = rows[0]["utt"]; res["same_item_deleted_everywhere_refused"] = refused(reg, [r for r in rows if r["utt"] != u0])
res["one_cell_item_missing_refused"] = refused(reg, rows[1:])
x = copy.deepcopy(rows[0]); x["panel"] = "P9"; res["extra_panel_refused"] = refused(reg, rows + [x])
y = copy.deepcopy(rows); y[0]["speaker"] = "other"; res["speaker_swap_refused"] = refused(reg, y)
y = copy.deepcopy(rows); y[0]["t"] += 1; res["reference_count_changed_refused"] = refused(reg, y)
y = copy.deepcopy(rows); y[0]["seed"] = 5; res["extra_seed_refused"] = refused(reg, y)
y = copy.deepcopy(rows); y[0]["e"] = -1; res["negative_count_refused"] = refused(reg, y)
P = json.load(open(prot)); P["seeds"] = [0, 1, 2, 3]; json.dump(P, open(prot, "w"))
res["protocol_edited_after_registration_refused"] = refused(reg, rows)
reg2 = f"{D}/register2.json"; json.dump(dict(protocol_path=prot, protocol_sha256=Q.sha(prot)), open(reg2, "w"))
res["four_seed_protocol_refused"] = refused(reg2, rows)
reg, prot, rows = build(); json.dump(dict(panel="P1", items_sha256="0" * 64), open(f"{D}/rel_1.json", "w"))
res["unreleased_panel_refused"] = refused(reg, rows)
res["ALL_OK"] = all(v for k, v in res.items() if k.endswith("refused") or k == "known_answer_equals_direct")
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sec2_q2c_entry_test.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float)); sys.exit(0 if res["ALL_OK"] else 1)
