"""Known-answer test for patch_peritem.py on a real checkpoint (b6ii_S_s0, base6s arm S), CPU.
argv: patched_dir model_dir. Checks: (1) item sums == totals (asserted inside evaluate); (2) items=True totals == items=False totals;
(3) PER vs the run's own saved json (GPU-scored; small CPU/GPU argmax flips allowed, difference printed); (4) a broken item list
(one item's e changed) must fail the sum check."""
import sys, json, os
pd, M = sys.argv[1], sys.argv[2]; sys.path[:0] = [pd, M]; os.chdir(M)
import torch, base6s as S, base6 as B6, base5_next as B
assert S.__file__.startswith(pd) and B6.__file__.startswith(pd), (S.__file__, B6.__file__)
O = M + "/phones_out/base6"; R = json.load(open(O + "/b6ii_S_s0.json"))
assert B.RIGHT == R["config"]["RIGHT"], ("set env RIGHT to the run config", B.RIGHT, R["config"]["RIGHT"])
m = S.make("S", S.budget()); m.load_state_dict(torch.load(O + "/b6ii_S_s0.pt", map_location="cpu")); dev = "cpu"
out = {}
for s_ in ("cleanS", "libriA"):
    X, Y = B.load(s_)[:2]
    a = S.evaluate("S", m, X, Y, dev); b = S.evaluate("S", m, X, Y, dev, items=True)
    assert a["PER"] == b["PER"] and a["hyp_ref"] == b["hyp_ref"], (a, b)
    it = b["items"]; assert len(it) == len(X) and sorted(x[0] for x in it) == list(range(len(X)))
    bad = [list(x) for x in it]; bad[0][1] += 1
    broken_fails = sum(x[1] for x in bad) != b["e"]
    out[s_] = dict(cpu=a, saved=R["sets"][s_], diff=round(a["PER"] - R["sets"][s_]["PER"], 2), n=b["n_items"], e=b["e"], t=b["t"],
                   broken_control_fails=broken_fails)
    print(s_, out[s_], flush=True)
ok = all(v["broken_control_fails"] and abs(v["diff"]) <= 0.5 for v in out.values())
out["verdict"] = "PASS" if ok else "FAIL"; print(out["verdict"])
json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/test_peritem_result.json", "w"), indent=1)
