"""Same known-answer test for base6.evaluate (plain trainer) using b6ii_P_s0 (base6s arm P = B6.Base6 + greedy collapse, the same
model class and decode base6.evaluate uses; base6.py's own runs need deleted feature copies). argv: patched_dir; env RIGHT from config."""
import sys, json, os
pd = sys.argv[1]; sys.path[:0] = [pd]; os.chdir(pd)
import torch, base6s as S, base6 as B6, base5_next as B
assert B6.__file__.startswith(pd)
O = pd + "/phones_out/base6"; R = json.load(open(O + "/b6ii_P_s0.json")); c = R["config"]
assert B.RIGHT == c["RIGHT"] and c["arm"] == "P", (B.RIGHT, c)
m = S.make("P", S.budget()); m.load_state_dict(torch.load(O + "/b6ii_P_s0.pt", map_location="cpu"))
assert type(m) is B6.Base6
out = {}
for s_ in ("cleanS", "libriA"):
    X, Y = B.load(s_)[:2]; a = B6.evaluate(m, X, Y, "cpu"); b = B6.evaluate(m, X, Y, "cpu", items=True)
    assert a["PER"] == b["PER"] and len(b["items"]) == len(X)
    out[s_] = dict(cpu=a, saved=R["sets"][s_], diff=round(a["PER"] - R["sets"][s_]["PER"], 2), e=b["e"], t=b["t"]); print(s_, out[s_], flush=True)
out["verdict"] = "PASS" if all(abs(v["diff"]) <= 0.5 for v in out.values()) else "FAIL"; print(out["verdict"])
json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/test_peritem_b6_result.json", "w"), indent=1)
