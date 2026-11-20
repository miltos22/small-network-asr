"""Our own forced aligner (no external tools): transcript -> CMUdict phones -> left-to-right chain of phone substates with
optional silence between words -> Viterbi over Module 2's state scores. Output = phone segments like the LibriSpeech alignments.
Used to label his non-test recordings; his 10 test clips are never aligned or used for calibration."""
import os
import json, re, sys
import numpy as np
import af, phones

CMU = os.environ.get("DATA_DIR", "data") + "/cmudict/cmudict.dict"


def lexicon():
    lex = {}
    for line in open(CMU, encoding="utf-8"):
        w, *ph = line.split("#")[0].split()
        w = re.sub(r"\(\d+\)$", "", w)
        lex.setdefault(w.upper(), [re.sub(r"\d", "", p) for p in ph])  # first pronunciation (shortcut: variants unused; W1 layer will model variation)
    return lex


def chain(words, lex):
    """states: (phone_index or SIL, substate, optional)"""
    st = [(phones.SILID, 0, True)]
    for w in words:
        for p in lex[w]:
            for s in range(3):
                st.append((phones.PID[p], s, False))
        st.append((phones.SILID, 0, True))
    return st


def viterbi(S, st, skip_pen=-1.0):
    """S: (T, 118) state log-scores. Left-to-right; optional silence states can be skipped."""
    T, N = len(S), len(st)
    col = np.array([3 * phones.NP if p == phones.SILID else 3 * p + s for p, s, _ in st])
    E = S[:, col]
    opt = np.array([o for _, _, o in st])
    D = np.full((T, N), -np.inf); B = np.zeros((T, N), np.int32)
    D[0, 0] = E[0, 0]
    if N > 1 and opt[0]:
        D[0, 1] = E[0, 1] + skip_pen
    for t in range(1, T):
        prev = D[t - 1]
        stay = prev
        nxt = np.concatenate([[-np.inf], prev[:-1]])
        skp = np.full(N, -np.inf); skp[2:] = np.where(opt[1:-1], prev[:-2] + skip_pen, -np.inf)
        cand = np.stack([stay, nxt, skp]); arg = cand.argmax(0)
        D[t] = cand[arg, np.arange(N)] + E[t]; B[t] = arg
    end = N - 1 if not opt[-1] or D[-1, N - 1] >= D[-1, N - 2] else N - 2
    path = np.zeros(T, np.int32); j = end
    for t in range(T - 1, -1, -1):
        path[t] = j; j -= B[t, j]
    return path, float(D[-1, end])


def segments(path, st):
    out = []; t0 = 0
    for t in range(1, len(path) + 1):
        if t == len(path) or st[path[t]][0] != st[path[t0]][0] or (st[path[t]][1] < st[path[t - 1]][1]):
            p = st[path[t0]][0]
            out.append((t0, t, "" if p == phones.SILID else af.PHONES[p])); t0 = t
    return out
