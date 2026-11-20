"""Arm 2c labels (ARM_2c_DESIGN_2026-11-15.md section 3): variant token sequences (spec 4d: 39 phonemes -> 53 tokens + SIL) and
per-frame phone labels for the split heads, for the 5.4 h training pool (fit + cal, the base5 utterance order).
Variants are set by RULES FROM CONTEXT on the MFA alignment (phones with stress digits + word tier); the audio is not inspected.
Hard check: every variant sequence, with SIL removed and each token mapped back to its 39-phone spelling, must equal the base5 label
sequence of the same utterance exactly (base5_{fit,cal}.npz y_i). Any mismatch aborts.
Frame labels: ph_{fit,cal}.npz 'ph' (39 = silence, -1 = unknown) cut to base5 frame counts (diff <= 3 frames; missing tail = -1).
Writes cache/v2c_{fit,cal}.npz: y0..y{n-1} (variant ids, int16), n, f (frame labels, int8, concatenated), flens.
Usage: python3 labels2c.py [selftest]"""
import os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import af  # phones (scipy) is imported only where the alignments are read, so the GPU venv can import this file

C = os.environ.get("DATA_DIR", "data") + "/cache"
P39 = list(af.PHONES); PID = {p: i for i, p in enumerate(P39)}
VAR = ["K_FRONT", "K_BACK", "G_FRONT", "G_BACK", "FLAP_T", "FLAP_D", "GLOTTAL_T", "T_NOREL", "P_NOPUFF", "T_NOPUFF", "K_NOPUFF",
       "DARK_L", "R_POSTV", "TS", "DZ", "SCHWA"]
TOK = [p for p in P39 if p not in ("K", "G")] + VAR  # 53 tokens
assert len(TOK) == 53, len(TOK)
TID = {t: i for i, t in enumerate(TOK)}; SIL = len(TOK)  # 53 = silence token; blank = 54 in the models
BACK = {"K_FRONT": ["K"], "K_BACK": ["K"], "G_FRONT": ["G"], "G_BACK": ["G"], "FLAP_T": ["T"], "FLAP_D": ["D"], "GLOTTAL_T": ["T"],
        "T_NOREL": ["T"], "P_NOPUFF": ["P"], "T_NOPUFF": ["T"], "K_NOPUFF": ["K"], "DARK_L": ["L"], "R_POSTV": ["R"], "TS": ["T", "S"],
        "DZ": ["D", "Z"], "SCHWA": ["AH"]}
TO39 = [[PID[q] for q in BACK.get(t, [t])] for t in TOK]  # token id -> list of 39-phone ids
VOW = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
FRONT = {"IY", "IH", "EY", "EH", "AE", "Y"}
MINSIL = 10  # silence intervals >= 100 ms become a SIL token (FILL, mine)


def to39(seq):
    out = []
    for t in seq:
        if t != SIL: out += TO39[t]
    return out


def parse(tg):
    s = open(tg).read(); w0 = s.index('name = "words"'); p0 = s.index('name = "phones"')
    rx = r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"'
    words = [(float(a), float(b), t) for a, b, t in re.findall(rx, s[w0:p0] if w0 < p0 else s[w0:])]
    ph = [(int(round(float(a) * 100)), int(round(float(b) * 100)), t.upper()) for a, b, t in re.findall(rx, s[p0:] if p0 > w0 else s[p0:w0])]
    return words, ph


def variants(items):
    """items: list of runs; each item = (base, stress, word_id) or None for a pause/unknown. Returns token list (SIL handled by caller)."""
    out, i, n = [], 0, len(items)
    nb = lambda k: items[k] if 0 <= k < n else None
    while i < n:
        it = items[i]
        if it is None or it == "SIL":
            if it == "SIL": out.append(SIL)
            i += 1; continue
        p, st, w = it; pr, nx = nb(i - 1), nb(i + 1)
        prv = pr if isinstance(pr, tuple) else None; nxt = nx if isinstance(nx, tuple) else None
        pause_next = nxt is None
        same_w_next = nxt is not None and nxt[2] == w; same_w_prev = prv is not None and prv[2] == w
        prev_vow = prv is not None and prv[0] in VOW
        def next_vowel_of_k():  # K/G colour: next vowel or Y, looking through one L/R/W in the same run
            if nxt is None: return None
            if nxt[0] in VOW or nxt[0] == "Y": return nxt[0]
            if nxt[0] in ("L", "R", "W"):
                n2 = nb(i + 2)
                if isinstance(n2, tuple) and (n2[0] in VOW): return n2[0]
            return None
        flap = (prv is not None and (prv[0] in VOW or (prv[0] == "R" and isinstance(nb(i - 2), tuple) and nb(i - 2)[0] in VOW)) and nxt is not None
                and nxt[0] in VOW and (nxt[1] == "0" or (nxt[2] != w)))
        t = p
        if p == "T":
            n2 = nb(i + 2); n3 = nb(i + 3)
            if same_w_next and nxt[0] == "S": out.append(TID["TS"]); i += 2; continue
            if same_w_prev and prv[0] == "S": t = "T_NOPUFF"
            elif (prv is not None and (prv[0] in VOW or prv[0] == "N") and same_w_next and nxt[0] == "AH" and nxt[1] == "0"
                  and isinstance(n2, tuple) and n2[0] == "N" and n2[2] == w and not (isinstance(n3, tuple) and n3[2] == w and n3[0] in VOW)):
                t = "GLOTTAL_T"
            elif flap: t = "FLAP_T"
            elif pause_next: t = "T_NOREL"
        elif p == "D":
            if same_w_next and nxt[0] == "Z": out.append(TID["DZ"]); i += 2; continue
            if flap: t = "FLAP_D"
        elif p == "P":
            if same_w_prev and prv[0] == "S": t = "P_NOPUFF"
        elif p in ("K", "G"):
            if p == "K" and same_w_prev and prv[0] == "S": t = "K_NOPUFF"
            else: t = p + ("_FRONT" if next_vowel_of_k() in FRONT else "_BACK")
        elif p in ("L", "R"):
            if prev_vow and not (nxt is not None and nxt[0] in VOW): t = "DARK_L" if p == "L" else "R_POSTV"
        elif p == "AH" and st == "0": t = "SCHWA"
        out.append(TID[t]); i += 1
    return out


def utt(path, T):
    import phones
    tg = path.replace(f"{phones.DATA}/LibriSpeech", f"{phones.DATA}/align").replace(".flac", ".TextGrid")
    words, ph = parse(tg); items = []
    for i0, i1, raw in ph:
        i1 = min(i1, T)
        if i1 <= i0: continue  # same filter as phones.job
        base = re.sub(r"\d", "", raw); st = re.sub(r"\D", "", raw)
        if base in ("", "SIL", "SP"):
            items.append("SIL" if i1 - i0 >= MINSIL else None)
        elif base in PID:
            mid = (i0 + i1) / 200.0; w = next((k for k, (a, b, _) in enumerate(words) if a <= mid < b), -1)
            items.append((base, st, w))
        else:
            items.append(None)  # spn etc.: a break, no label (phones.job drops it)
    seq = variants(items)
    while seq and seq[0] == SIL and len(seq) > 1 and seq[1] == SIL: seq = seq[1:]
    return [t for k, t in enumerate(seq) if not (t == SIL and k > 0 and seq[k - 1] == SIL)]


def build(name):
    import phones
    files = phones.split()[name]; d = np.load(f"{C}/ph_{name}.npz"); lens = d["lens"]; st = np.r_[0, np.cumsum(lens)]
    b5 = np.load(f"{C}/base5_{name}.npz"); flens = b5["lens"]; assert len(files) == len(lens) == len(flens) == int(b5["n"])
    Y, F, cnt = [], [], np.zeros(SIL + 1, np.int64)
    for u, f in enumerate(files):
        seq = utt(f, int(lens[u])); ref = b5[f"y{u}"].astype(np.int64).tolist()
        assert to39(seq) == ref, (name, u, f, to39(seq)[:20], ref[:20])
        Y.append(np.array(seq, np.int16)); np.add.at(cnt, seq, 1)
        fr = d["ph"][st[u]:st[u + 1]].astype(np.int8); n = int(flens[u]); assert abs(n - len(fr)) <= 3
        F.append(np.r_[fr[:n], -np.ones(max(0, n - len(fr)), np.int8)])
    np.savez(f"{C}/v2c_{name}.npz", f=np.concatenate(F), flens=flens, n=len(Y), **{f"y{i}": y for i, y in enumerate(Y)})
    print(name, len(Y), "utts; token counts:", {(TOK + ["SIL"])[i]: int(c) for i, c in enumerate(cnt)}, flush=True)
    return cnt


def selftest():
    W = lambda *ps: [(p[:-1], p[-1], 0) if p[-1].isdigit() else (p, "", 0) for p in ps]
    names = lambda s: [(TOK + ["SIL"])[t] for t in variants(s)]
    assert names(W("W", "AO1", "T", "ER0")) == ["W", "AO", "FLAP_T", "ER"]                     # water
    assert names(W("B", "AH1", "T", "AH0", "N")) == ["B", "AH", "GLOTTAL_T", "SCHWA", "N"]      # button
    assert names(W("S", "P", "IH1", "N")) == ["S", "P_NOPUFF", "IH", "N"]                       # spin
    assert names(W("K", "IY1")) == ["K_FRONT", "IY"] and names(W("K", "UW1", "L")) == ["K_BACK", "UW", "DARK_L"]
    assert names(W("K", "AE1", "T", "S")) == ["K_FRONT", "AE", "TS"]                            # cats
    assert names(W("K", "AA1", "R")) == ["K_BACK", "AA", "R_POSTV"] and names(W("L", "IY1", "F")) == ["L", "IY", "F"]
    assert names(W("K", "AE1", "T") + [None]) == ["K_FRONT", "AE", "T_NOREL"]
    assert names(W("AH0", "B", "AW1", "T")) == ["SCHWA", "B", "AW", "T_NOREL"]  # utterance end counts as a pause
    for t in range(SIL): assert t == TID[TOK[t]]
    print("labels2c selftest ok", flush=True)


if __name__ == "__main__":
    selftest()
    if sys.argv[1:] != ["selftest"]:
        tot = sum(build(n) for n in ("fit", "cal")); print("total", {(TOK + ["SIL"])[i]: int(c) for i, c in enumerate(tot)}, flush=True)
