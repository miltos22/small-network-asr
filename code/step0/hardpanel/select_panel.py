"""C2 0 d hard-panel SELECTION (v6.5 rule, verbatim in Part C C2 0 d; 20 Nov 2026, the agent). No model, no audio, no score: TextGrids only.
Speakers ordered by sha256(speaker id); a speaker qualifies with >= 8 utterances of 2-20 s that have a TextGrid; take the first 40
qualifying (fewer -> all, panel exists only with >= 10); per speaker the 8 utterances with the smallest sha256(utterance id); an utterance
with a phone the map cannot place is replaced by the next of that speaker in hash order; a speaker without 8 mappable utterances is
replaced by the next qualifying speaker. Duration = the TextGrid's top-level xmax (MFA aligns the whole file). 'Map cannot place' =
a phones-tier label that is neither in phones.PID nor one of the labels the TRAINING pipeline treats as non-phone ("", SIL, SP; SPN
is dropped by the training pipeline, phones.job / base5_prep100 docstring) -> an SPN utterance is mappable (choice recorded, count reported).
Writes hard_panel_record.json (ids, per-speaker lists, input digests). Disjointness from every training / validation / exposed speaker set
of manifests_v65 is asserted."""
import os
import glob, hashlib, json, os, re, sys
sys.path.insert(0, "code")
import phones
A = os.environ.get("SCRATCH_DIR", "scratch") + "/align_other500/train-other-500"
H = os.path.dirname(os.path.abspath(__file__)); h = lambda x: hashlib.sha256(x.encode()).hexdigest()
NONPHONE = {"", "SIL", "SP", "SPN"}
tg = sorted(glob.glob(f"{A}/*/*/*.TextGrid")); assert len(tg) == 148645, len(tg)
by = {}
for p in tg:
    u = os.path.basename(p)[:-9]; s = u.split("-")[0]; by.setdefault(s, []).append(p)
def info(p):
    t = open(p).read(); xmax = float(re.search(r"xmax = ([\d.]+)", t).group(1))
    labs = {ph for _, _, ph in phones.segments(p)}
    return xmax, sorted(labs - set(phones.PID) - NONPHONE), "SPN" in labs
order = sorted(by, key=h); panel, log = [], []
for s in order:
    if len(panel) == 40: break
    us = sorted(by[s], key=lambda p: h(os.path.basename(p)[:-9])); inr = []
    for p in us:
        d, bad, spn = info(p)
        if 2.0 <= d <= 20.0: inr.append((p, d, bad, spn))
    if len(inr) < 8: log.append((s, "not qualifying", len(inr))); continue
    good = [x for x in inr if not x[2]][:8]
    if len(good) < 8: log.append((s, "qualifying but < 8 mappable", len(good))); continue
    panel.append(dict(speaker=s, utts=[os.path.basename(p)[:-9] for p, *_ in good], dur=[round(d, 2) for _, d, _, _ in good],
                      spn=sum(x[3] for x in good), replaced_unmappable=sum(1 for x in inr[:inr.index(good[-1]) + 1] if x[2])))
M = json.load(open("code/step0/manifests_v65.json"))
others = set()
for k, v in M.items():
    if isinstance(v, dict) and isinstance(v.get("speakers"), list): others |= set(map(str, v["speakers"]))
assert others, "no speaker sets found in manifests_v65"
ps = {x["speaker"] for x in panel}; assert not (ps & others), ("panel overlaps a manifest speaker set", sorted(ps & others))
exists = len(panel) >= 10
rec = dict(rule="Part C v6.5 C2 0 d selection", created_by="select_panel.py", exists=exists, n_speakers=len(panel),
           n_utts=sum(len(x["utts"]) for x in panel), hours=round(sum(sum(x["dur"]) for x in panel) / 3600, 3), panel=panel,
           skipped_before_last=log, speakers_scanned=order.index(panel[-1]["speaker"]) + 1 if panel else len(order),
           n_speakers_pool=len(order), textgrids=len(tg), nonphone_labels=sorted(NONPHONE), spn_utts=sum(x["spn"] for x in panel),
           manifest_speaker_sets_checked=len(others), select_code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           phones_py_sha256=hashlib.sha256(open(phones.__file__, "rb").read()).hexdigest(),
           ids_sha256=h("\n".join(u for x in panel for u in x["utts"])))
p = f"{H}/hard_panel_record.json"; fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w") as f: json.dump(rec, f, indent=1)
print({k: v for k, v in rec.items() if k not in ("panel", "skipped_before_last")}, "skipped", log[:5])
