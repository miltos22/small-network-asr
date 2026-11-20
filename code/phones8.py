"""Experiment C (the author: "split 1 into 2-3"): layer 1 gives each phone ONE mean shape, but several phones have 2 forms
(allophones). Split rules are programmed from phonetics, fixed before results; only each sub-version's mean is measured.
  vowels    : stressed (1/2) | unstressed (0)                       -> reduced vowels drift to schwa (AH/IH)
  T, D      : tap (after vowel/R/ER, before unstressed vowel, no gaps) | other
  fricatives: before a pause / utterance end | other              -> final Z devoices (S/Z)
  M, N, NG  : before a vowel | other
Recognition needs no labels: phone score = logsumexp over its sub-versions. Same data as phones6: layer 1 fit on dev-clean cal,
layer 2 (L2plus) fit on dev-other half A, chosen on half B; test-clean/test-other once if the split wins on half B."""
import glob, json, re
import numpy as np
import af, phones, phones3, phones5, phones6

FRIC = {phones.PID[p] for p in "F V TH DH S Z SH ZH".split()}; NAS = {phones.PID[p] for p in "M N NG".split()}
TD = {phones.PID["T"], phones.PID["D"]}; VOWID = {phones.PID[p] for p in af.VOW}; RLIKE = VOWID | {phones.PID["R"]}

def stress(files, lens):
    """Stress digit per cached segment (-1 = consonant), rebuilt with exactly phones.job's filter."""
    out = []
    for f, T in zip(files, lens):
        tg = f.replace(f"{phones.DATA}/LibriSpeech", f"{phones.DATA}/align").replace(".flac", ".TextGrid")
        try: s = open(tg).read()
        except FileNotFoundError: continue
        tier = s[s.index('name = "phones"'):]
        for a, b, txt in re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', tier):
            i0, i1 = int(round(float(a) * 100)), min(int(round(float(b) * 100)), T); p = re.sub(r"\d", "", txt.upper())
            if i1 > i0 and p in phones.PID: out.append(int(txt[-1]) if txt[-1].isdigit() else -1)
    return np.array(out)

def sublabels(segs, lens, st):
    y = segs[:, 2]; utt = np.searchsorted(np.cumsum(lens), segs[:, 0], side="right")
    sp = np.r_[False, utt[1:] == utt[:-1]]; sn = np.r_[utt[1:] == utt[:-1], False]
    prev = np.where(sp, np.r_[-1, y[:-1]], -1); nxt = np.where(sn, np.r_[y[1:], -1], -1); nst = np.where(sn, np.r_[st[1:], -1], -1)
    gp = np.where(sp, segs[:, 0] - np.r_[0, segs[:-1, 1]], 99); gn = np.where(sn, np.r_[segs[1:, 0], 0] - segs[:, 1], 99)
    sub = np.zeros(len(y), int)
    for j in range(len(y)):
        k = y[j]
        if k in VOWID: sub[j] = st[j] == 0
        elif k in TD: sub[j] = prev[j] in RLIKE and nxt[j] in VOWID and nst[j] == 0 and gp[j] == 0 and gn[j] == 0
        elif k in FRIC: sub[j] = gn[j] > 0
        elif k in NAS: sub[j] = nxt[j] in VOWID
    return sub

def split_fit(X, y, sub):
    lab = 2 * y + sub; used, z = np.unique(lab, return_inverse=True)
    c = phones3.lda_fit(X, z, len(used)); c["parent"] = used // 2; return c

def split_logp(c, X):
    S = ((X - c["m"]) / c["s"]) @ c["W"].T + c["b"]; out = np.full((len(X), phones.NP), -np.inf)
    for k in range(phones.NP):
        cols = np.where(c["parent"] == k)[0]
        if len(cols): out[:, k] = np.logaddexp.reduce(S[:, cols], 1)
    return out - np.logaddexp.reduce(out, 1, keepdims=True)

def top(P, y):
    o = np.argsort(-P, 1); return round(float((o[:, 0] == y).mean()), 4), round(float((o[:, :3] == y[:, None]).any(1).mean()), 4)

if __name__ == "__main__":
    model = np.load(f"{phones.OUT}/model.npy", allow_pickle=True).item()
    C = phones.load("cal"); MC = np.load(f"{phones.CACHE}/segmel_cal.npy"); XC = phones6.l1_inputs(C, MC, model); yc = C["segs"][:, 2]
    stc = stress(phones.split()["cal"], C["lens"]); assert len(stc) == len(yc), (len(stc), len(yc))
    subc = sublabels(C["segs"], C["lens"], stc)
    R = {"split_counts": {af.PHONES[k]: [int(((yc == k) & (subc == s)).sum()) for s in (0, 1)] for k in range(phones.NP)}}
    l1 = phones3.lda_fit(XC, yc, phones.NP); l1s = split_fit(XC, yc, subc); del C, XC
    files = sorted(glob.glob(f"{phones.DATA}/LibriSpeech/dev-other/*/*/*.flac"))
    D, MD = phones5.build("dev_other", files); y = D["segs"][:, 2]; X1 = phones6.l1_inputs(D, MD, model)
    spk = np.array([f.split("/")[-3] for f in files]); order = sorted(set(spk)); A_spk = set(order[: len(order) // 2])
    utt = np.searchsorted(np.cumsum(D["lens"]), D["segs"][:, 0], side="right"); A = np.array([spk[u] in A_spk for u in utt])
    l2 = {}
    for name, P in (("unsplit", phones6.logp(l1, X1)), ("split", split_logp(l1s, X1))):
        R[f"L1_{name}_val"] = top(P[~A], y[~A])
        V = np.concatenate([phones6.context(P, D["segs"], D["lens"]), X1], 1)
        l2[name] = phones3.lda_fit(V[A], y[A], phones.NP); R[f"L2plus_{name}_val"] = list(phones3.acc(l2[name], V[~A], y[~A]))
        print(name, R[f"L1_{name}_val"], R[f"L2plus_{name}_val"], flush=True)
    R["stored_L1"] = dict(unsplit=int(l1["n"]), split=int(l1s["n"]))
    if R["L2plus_split_val"][0] > R["L2plus_unsplit_val"][0]:
        P = split_logp(l1s, X1); V = np.concatenate([phones6.context(P, D["segs"], D["lens"]), X1], 1); c2 = phones3.lda_fit(V, y, phones.NP); del D, X1, MD
        for name in ("test", "test_other"):
            T = phones.load(name); MT = np.load(f"{phones.CACHE}/segmel_{name}.npy"); XT = phones6.l1_inputs(T, MT, model); yt = T["segs"][:, 2]
            PT = split_logp(l1s, XT); VT = np.concatenate([phones6.context(PT, T["segs"], T["lens"]), XT], 1)
            R[f"TEST_{name}"] = dict(L1_split=top(PT, yt), L2plus_split=list(phones3.acc(c2, VT, yt))); print("TEST", name, R[f"TEST_{name}"], flush=True)
            del T, XT, MT
    json.dump(R, open(f"{phones.OUT}/eval_split.json", "w"), indent=1); print(json.dumps(R))
