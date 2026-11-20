"""Q2(c) PRODUCTION decision entry (Part C v6.5; review round 5 C4). The ONLY function that may produce the Q2(c) verdict.

decide_q2c(register_path, rows_path) -> verdict dict. Nothing the caller passes can change slots, seeds, weights or the pair count:
  * register_path: the REGISTRATION record written before the first Q2(c) run (C9.2 REGISTER). It names the protocol file and its sha256.
  * the protocol (JSON) fixes: arms ["S","R"], seeds [0..4] (exactly 5, K_REQUIRED of sec2_abs), panels [{name, panel_weight,
    items: [{utt, speaker, t}]}] (t = reference phone count), and the release record path for each panel.
  * each panel's RELEASE record (C9.2 BIND/RELEASE) must exist and carry the sha256 of that panel's item list; a panel without a
    release cannot be scored.
  * rows_path: JSONL rows {panel, arm, seed, utt, speaker, e, t} from the locked scorer.
Refusals (AssertionError): protocol digest != registered; release missing or item digest mismatch; any row for an unknown panel / arm /
seed / item; a DUPLICATE row for the same (panel, arm, seed, utt); any (panel, arm, seed, item) missing; speaker or t different from the
protocol's reference; non-finite or negative e; panel weights not summing to 1; seeds != [0, 1, 2, 3, 4].
Q2(c) is a fixed-recipe comparison: it has NO validation gate (an explicit exception to any inherited B 2 same-sign screen rule)."""
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec2_abs as SA


def sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()
def items_digest(items): return hashlib.sha256(json.dumps(sorted((i["utt"], i["speaker"], int(i["t"])) for i in items)).encode()).hexdigest()


def decide_q2c(register_path, rows_path):
    reg = json.load(open(register_path))
    assert sha(reg["protocol_path"]) == reg["protocol_sha256"], "protocol changed after registration"
    P = json.load(open(reg["protocol_path"]))
    arms, seeds = tuple(P["arms"]), tuple(P["seeds"])
    assert arms == ("S", "R") and seeds == (0, 1, 2, 3, 4) and len(seeds) == SA.K_REQUIRED, ("protocol slots", arms, seeds)
    pw = [float(p["panel_weight"]) for p in P["panels"]]; assert abs(sum(pw) - 1) < 1e-12 and all(w > 0 for w in pw), ("panel weights", pw)
    ref = {}
    for p in P["panels"]:
        rel = json.load(open(p["release_path"]))
        assert rel.get("panel") == p["name"] and rel.get("items_sha256") == items_digest(p["items"]), ("panel not released / items differ", p["name"])
        assert len({i["utt"] for i in p["items"]}) == len(p["items"]), ("duplicate utt in protocol", p["name"])
        ref[p["name"]] = {i["utt"]: (i["speaker"], int(i["t"])) for i in p["items"]}
    seen = {}
    for line in open(rows_path):
        if not line.strip(): continue
        r = json.loads(line); key = (r["panel"], r["arm"], r["seed"], r["utt"])
        assert r["panel"] in ref, ("unknown panel", r["panel"])
        assert r["arm"] in arms and r["seed"] in seeds, ("unknown slot", r["arm"], r["seed"])
        assert r["utt"] in ref[r["panel"]], ("unknown item", key)
        assert key not in seen, ("duplicate row", key)
        assert (r["speaker"], int(r["t"])) == ref[r["panel"]][r["utt"]], ("speaker / reference count differs from protocol", key)
        assert np.isfinite(r["e"]) and r["e"] >= 0 and float(r["e"]) == int(r["e"]), ("bad edit count", key)
        seen[key] = r
    for pn, items in ref.items():
        for a in arms:
            for s in seeds:
                miss = [u for u in items if (pn, a, s, u) not in seen]
                assert not miss, ("missing items", pn, a, s, len(miss), miss[:3])
    d = SA.from_counts(list(seen.values()), [p["name"] for p in P["panels"]], arms=arms, seeds=seeds)
    out = SA.decide(d, pw)
    out.update(protocol_sha256=reg["protocol_sha256"], panels=[p["name"] for p in P["panels"]], n_rows=len(seen))
    return out


if __name__ == "__main__":
    print(json.dumps(decide_q2c(sys.argv[1], sys.argv[2]), indent=1, default=float))
