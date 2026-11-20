"""Step 0.1 DRAFT speech manifests (MASTER_PLAN 4b.4/4b.5). DRAFT: not adopted, nothing scored. Seed 20261120.
Training pool = train-clean-100 minus the 20 cleanS speakers (cleanS_EXCLUDE). Nested whole-speaker draws: 1 h subset of 5.4 h subset of all.
Validation = the 23 dev-other speakers no old eval set used (exposure audit). Test candidates listed with their exposure. Writes step0/draft_manifests.json."""
import os
import json, os, hashlib, numpy as np
C = os.environ.get("DATA_DIR", "data") + "/cache"; W = "code"
spk = lambda f: os.path.basename(str(f)).split("-")[0]
A = json.load(open(f"{W}/step0/exposure_audit.json"))
m = np.load(f"{C}/base5_tc100_meta.npz", allow_pickle=True); files = [str(f) for f in m["files"]]; lens = m["lens"]
excl = {spk(f) for f in np.load(f"{C}/base5_cleanS.npz", allow_pickle=True)["files"]}  # the 20 cleanS speakers (same as cleanS_EXCLUDE list)
assert len(excl) == 20, len(excl)
keep = [i for i, f in enumerate(files) if spk(f) not in excl]
order = np.random.default_rng(20261120).permutation(sorted({spk(files[i]) for i in keep}))
fr = {}; 
for i in keep: fr[spk(files[i])] = fr.get(spk(files[i]), 0) + int(lens[i])
def draw(h):
    out, t = [], 0
    for s in order:
        if t >= h * 360000: break
        out.append(str(s)); t += fr[s]
    return out, round(t / 360000, 3)
fp = lambda L: hashlib.sha256("\n".join(sorted(L)).encode()).hexdigest()[:16]
s1, h1 = draw(1.0); s5, h5 = draw(5.4); sall = [str(s) for s in order]; hall = round(sum(fr.values()) / 360000, 3)
assert set(s1) <= set(s5) <= set(sall)
D = dict(status="DRAFT 20 Nov: not adopted, not scored", seed=20261120,
  train=dict(pool="train-clean-100 minus 20 cleanS speakers", n_spk=len(sall), hours=hall, utts=len(keep), fp=fp(sall),
             h1=dict(n_spk=len(s1), hours=h1, fp=fp(s1), speakers=s1), h5_4=dict(n_spk=len(s5), hours=h5, fp=fp(s5), speakers=s5)),
  validation=dict(pool="dev-other", speakers=A["dev_other_unused_by_eval"], n_spk=len(A["dev_other_unused_by_eval"]), fp=fp(A["dev_other_unused_by_eval"]),
             exposure="never in an eval set; TRAINING data of the parked 1c arm B (MIX=B) runs only"),
  test_candidates=dict(test_clean=dict(n_spk=A["pools"]["test-clean"], exposure="cleanT built 17 Nov, never scored (grep of results/logs)"),
             test_other=dict(n_spk=A["pools"]["test-other"], exposure="ALL 33 speakers in old 'testother' (330 utts), scored many times: NOT sealed"),
             hard_untouched="none on disk; candidate: download LibriSpeech train-other-500 and draw held-out speakers (selected by the same difficulty criterion as dev/test-other)"))
json.dump(D, open(f"{W}/step0/draft_manifests.json", "w"), indent=1)
print({k: (v if k != "train" else {kk: (vv if not isinstance(vv, dict) else {a: b for a, b in vv.items() if a != "speakers"}) for kk, vv in v.items()}) for k, v in D.items() if k != "validation"}, D["validation"]["n_spk"])
# ---- Option B (20 Nov): whole-speaker draws give only 3 speakers at 1 h (train-clean-100 has ~25 min per speaker), so size curves would also
# change speaker count 3 -> 14 -> 231. B: the same 40 seeded speakers at 1 h and 5.4 h, utterances nested (seeded order per speaker, equal time share).
r = np.random.default_rng(20261121); spk40 = [str(s) for s in order[:40]]
per = {s: [i for i in r.permutation(keep) if spk(files[i]) == s] for s in spk40}
def slice_(h):
    q = h * 360000 / 40; ix = []
    for s in spk40:
        t = 0
        for i in per[s]:
            if t >= q: break
            ix.append(int(i)); t += int(lens[i])
    return sorted(ix)
b1, b5 = slice_(1.0), slice_(5.4); assert set(b1) <= set(b5)
D["optionB_fixed40"] = dict(speakers=spk40, h1=dict(utts=len(b1), hours=round(sum(int(lens[i]) for i in b1) / 360000, 3), fp=fp([files[i] for i in b1])),
                            h5_4=dict(utts=len(b5), hours=round(sum(int(lens[i]) for i in b5) / 360000, 3), fp=fp([files[i] for i in b5])),
                            note="speaker count fixed at 40 for 1 h and 5.4 h; 100 h = all 231")
json.dump(D, open(f"{W}/step0/draft_manifests.json", "w"), indent=1); print("B", {k: v for k, v in D["optionB_fixed40"].items() if k != "speakers"})
