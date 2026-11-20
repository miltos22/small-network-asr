"""Split manifests for MASTER_PLAN Part C v6.3 C2.0 (review round 3 C2.0). CPU, read-only on data. Writes step0/manifests_v63.json:
per role the speaker list, utterance ids (cache order), hours, utterance count, and sha256 of the sorted id list; asserts disjoint roles and
the nesting 0.5 h (draw 0) <= 1 h (draw 0) <= 5.4 h. Roles (v6.3):
  train_5.4h = fit + cal (40 DEV-CLEAN speakers, base5 cache; review r3: not train-clean-100 as v6.2 said); train_1h / train_0.5h = whole-speaker
  draws with base6.draw_indices rule, draw id 0 (frozen; d0 gives exactly 16,524 updates at 1 h / 972 and 0.5 h / 1,836 passes);
  validation = libriA (dev-other, 4 spk) + cleanS (train-clean-100, 20 spk); clean_test = cleanT (test-clean, 40 spk, sealed, never scored);
  hard_test = NOT ON DISK (all 33 dev-other and all 33 test-other speakers exposed, exposure_audit_v2.json); exposed = libriB, testother,
  devother (report-only, never confirmation); reserved = train-other-500 (download running; untouched).
Usage: python -I manifests_v63.py"""
import os
import hashlib, json, os, sys
import numpy as np
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, W)
C = os.environ.get("DATA_DIR", "data") + "/cache"
import phones
spk = lambda f: os.path.basename(str(f)).split("-")[0]
S = json.load(open(f"{C}/base5_speakers.json"))
files = {n: [os.path.basename(str(f)).split(".")[0] for f in phones.split()[n]] for n in ("fit", "cal")}
for n in ("fit", "cal"):
    assert len(files[n]) == len(S[n]) and all(spk(f) == s for f, s in zip(files[n], S[n])), ("cache order mismatch", n)
lens = {n: np.load(f"{C}/base5_{n}.npz")["lens"] for n in ("fit", "cal")}
cf, lf = files["fit"] + files["cal"], np.concatenate([lens["fit"], lens["cal"]]); cs = [spk(f) for f in cf]


def draw(d, hours):  # = base6.draw_indices (base6.py:56-66)
    order = np.random.default_rng(500 + int(d)).permutation(sorted(set(cs))); out, fr = [], 0
    for s in order:
        if fr >= hours * 360000: break
        ix = [i for i, c in enumerate(cs) if c == s]; out += ix; fr += int(lf[ix].sum())
    return sorted(out)


def role(ids, ln):
    ids = list(ids); return dict(speakers=sorted({spk(f) for f in ids}), n_spk=len({spk(f) for f in ids}), utts=len(ids),
                                 hours=round(float(np.sum(ln)) / 360000, 3), sha256=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(), ids=ids)


M = {"train_5.4h": role(cf, lf)}
for h, k in ((1.0, "train_1h"), (0.5, "train_0.5h")):
    ix = draw(0, h); M[k] = role([cf[i] for i in ix], lf[ix])
# validation / test / exposed
lA = S["libriA"]; la = np.load(f"{C}/base5_libriA.npz")["lens"]
fA = [os.path.basename(str(f)).split(".")[0] for f in phones.split().get("libriA", [])] if "libriA" in phones.split() else None
M["validation_libriA"] = dict(speakers=sorted(set(lA)), n_spk=len(set(lA)), utts=len(lA), hours=round(float(la.sum()) / 360000, 3),
                              ids=fA, sha256=hashlib.sha256("\n".join(sorted(fA)).encode()).hexdigest() if fA else "speaker-level only (no ids in cache)")
for k, p in (("validation_cleanS", f"{C}/base5_cleanS.npz"), ("clean_test_cleanT", f"{C}/sealed/base5_cleanT.npz")):
    d = np.load(p, allow_pickle=True); M[k] = role([os.path.basename(str(f)).split(".")[0] for f in d["files"]], d["lens"])
dm = np.load(f"{C}/base5_devother_meta.npz", allow_pickle=True)
M["exposed_devother"] = role([os.path.basename(str(f)).split(".")[0] for f in dm["files"]], dm["lens"])
to = np.load(f"{C}/realcut_testother.npz", allow_pickle=True); M["exposed_testother"] = dict(speakers=sorted({spk(f) for f in to["files"]}), n_spk=len({spk(f) for f in to["files"]}), utts=len(to["files"]))
M["exposed_libriB"] = dict(speakers=sorted(set(S["libriB"])), n_spk=len(set(S["libriB"])), utts=len(S["libriB"]))
M["hard_test"] = "NOT ON DISK: candidate = a sealed train-other-500 speaker panel (labels: Lugosch MFA zip md5 2bab567d0ace651a4ba254e813629f46 verified; audio md5 pending)"
# asserts: disjoint roles, nesting
T = set(M["train_5.4h"]["speakers"]); V = set(M["validation_libriA"]["speakers"]) | set(M["validation_cleanS"]["speakers"]); CT = set(M["clean_test_cleanT"]["speakers"])
assert not (T & V) and not (T & CT) and not (V & CT), "roles overlap"
assert set(M["train_0.5h"]["ids"]) <= set(M["train_1h"]["ids"]) <= set(M["train_5.4h"]["ids"]), "nesting broken"
assert not (CT & (set(M["exposed_devother"]["speakers"]) | set(M["exposed_testother"]["speakers"]))), "clean test overlaps exposed"
out = {k: ({kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in M.items()}
json.dump(out, open(f"{W}/step0/manifests_v63.json", "w"), indent=1)
for k, v in M.items(): print(k, {kk: vv for kk, vv in v.items() if kk not in ("ids", "speakers")} if isinstance(v, dict) else v)
print("asserts OK: roles disjoint (train / validation / clean test), nesting 0.5 h <= 1 h <= 5.4 h, clean test disjoint from exposed")
