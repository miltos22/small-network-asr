"""Split manifests for MASTER_PLAN Part C v6.3 C2.0 (review round 3 C2.0). CPU, read-only on data. Writes step0/manifests_v63.json:
per role the speaker list, utterance ids (cache order), hours, utterance count, and sha256 of the sorted id list; asserts disjoint roles and
the nesting 0.5 h (draw 0) <= 1 h (draw 0) <= 5.4 h. Roles (v6.3):
  train_5.4h = fit + cal (40 DEV-CLEAN speakers, base5 cache; review r3: not train-clean-100 as v6.2 said); train_1h / train_0.5h = whole-speaker
  draws with base6.draw_indices rule, draw id 0 (frozen; d0 gives exactly 16,524 updates at 1 h / 972 and 0.5 h / 1,836 passes);
  validation = libriA (dev-other, 4 spk) + cleanS (train-clean-100, 20 spk); clean_test = cleanT (test-clean, 40 spk, sealed, never scored);
  hard_test = NOT ON DISK (all 33 dev-other and all 33 test-other speakers exposed, exposure_audit_v2.json); exposed = libriB, testother,
  devother (report-only, never confirmation); reserved = train-other-500 (download running; untouched).
v6.4: speaker nesting, training vs exposed, ordered-id hashes, cache file sha256.
v6.5 (review round 5 C2.0): EXPLICIT required-input list (missing file = assert, no glob filter), cleanS vs exposed, clean test vs libriB,
libriA exactly 4 speakers, libriB inside exposed dev-other (exposed sub-roles MAY overlap: that is the invariant, not pairwise disjointness),
validation vs exposed test-other; ordered (within-speaker) id hashes are what the trainer must reproduce at load (C9 1 parity check).
Usage: python -I manifests_v64.py"""
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
# v6.4 (round 4 C2.0): speaker nesting, every role, ordered-id hashes, cache-file hashes
assert set(M["train_0.5h"]["speakers"]) <= set(M["train_1h"]["speakers"]) <= set(M["train_5.4h"]["speakers"]), "speaker nesting broken"
EX = set(M["exposed_devother"]["speakers"]) | set(M["exposed_testother"]["speakers"]) | set(M["exposed_libriB"]["speakers"])
assert not (T & EX), "training overlaps an exposed set"
assert set(M["validation_libriA"]["speakers"]) <= set(M["exposed_devother"]["speakers"]), "libriA is expected to be 4 dev-other speakers (exposed, validation only)"
# v6.5 (round 5 C2.0)
assert M["validation_libriA"]["n_spk"] == 4, ("libriA must be exactly 4 speakers", M["validation_libriA"]["n_spk"])
assert not (set(M["validation_cleanS"]["speakers"]) & EX), "cleanS overlaps an exposed set"
assert not (CT & set(M["exposed_libriB"]["speakers"])), "clean test overlaps libriB"
assert set(M["exposed_libriB"]["speakers"]) <= set(M["exposed_devother"]["speakers"]), "libriB expected inside exposed dev-other"
for n in ("fit", "cal"):
    assert len(set(files[n])) == len(files[n]), ("duplicate utterance ids in cache order", n)
for k, v in M.items():
    if isinstance(v, dict) and v.get("ids"): v["sha256_ordered"] = hashlib.sha256("\n".join(v["ids"]).encode()).hexdigest()
def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""): h.update(blk)
    return h.hexdigest()
# v6.5: explicit list of what base6 / base6s read for the R paths (base5_next.load / spkstats / noise; vtln_* for warp arms, S_ = label-free)
REQ = ["base5_speakers.json", "noise_esc50_f1234.npz"]
for s_ in ("fit", "cal"):
    for sp in ("", "_sp0.9", "_sp1.1"):
        REQ += [f"base5_{s_}{sp}.npz", f"spk_{s_}{sp}.npz", f"vtln_{s_}{sp}.npz", f"vtln_S_{s_}{sp}.npz"]
for s_ in ("cleanS", "libriA"): REQ += [f"base5_{s_}.npz", f"spk_{s_}.npz", f"vtln_{s_}.npz", f"vtln_S_{s_}.npz"]
REQ += ["sealed/base5_cleanT.npz"]
miss = [f for f in REQ if not os.path.exists(f"{C}/{f}")]
M["required_inputs_missing"] = miss
M["cache_files_sha256"] = {f: fsha(f"{C}/{f}") for f in REQ if f not in miss}
M["cache_files_sha256_custodian_note"] = "the sealed cleanT file is HASHED here (custodian read of bytes), never loaded as features by this script"
M["hard_test"] = "decided at the freeze by Part C C2 0 d (train-other-500 archive md5 = openslr list d1a0fd59409feb2c614ce4d30c387708; labels Lugosch zip md5 2bab567d0ace651a4ba254e813629f46)"
out = {k: ({kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in M.items()}
assert not miss, ("required inputs missing", miss)
json.dump(out, open(f"{W}/step0/manifests_v65.json", "w"), indent=1)
for k, v in M.items(): print(k, {kk: vv for kk, vv in v.items() if kk not in ("ids", "speakers")} if isinstance(v, dict) else v)
print("v6.5 asserts OK + explicit inputs; asserts OK: roles disjoint (train / validation / clean test / exposed), utterance AND speaker nesting 0.5 h <= 1 h <= 5.4 h, clean test disjoint from exposed")
