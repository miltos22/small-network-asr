"""2-II arm R (CRITERIA_2II.md rule 3: run only if S LEADS): S with the head-to-specialist wiring SHUFFLED before training, fan-in kept.
Every head index g in every specialist path is replaced by sigma(g), sigma a fixed permutation of the 18 heads with no fixed point (seed 0): each
specialist still reads the same number of heads, specialists that shared heads still share them, the heads are still trained on their own questions,
but a specialist now reads heads trained on other questions. Same bytes as S. READING LIMIT (Part C v6.3): a derangement of indices does not
guarantee that a specialist loses its whole original head SET, nor that the substituted questions are unrelated; so a WIN supports 'the semantic
wiring beats a shuffled one', and R ~ S is NOT read as 'shape, not phonetics'. Q2 (c): RWIRE_SEED = seed (asserted with Q2C_PAIR=1).
Usage: python3 -I base6s_R.py <tag> <seed> <epochs>  |  base6s_R.py selftest"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import base6s as X

# v6.3 Q2(c): RWIRE_SEED = the shuffle draw (one fresh derangement per paired seed); absent = 0 = the 18 Nov sigma
rng = np.random.default_rng(int(os.environ.get("RWIRE_SEED", 0))); sig = rng.permutation(X.NH)
while (sig == np.arange(X.NH)).any(): sig = rng.permutation(X.NH)
TRUE = [list(p) for p in X.PATHS]
X.PATHS[:] = [[int(sig[g]) for g in p] for p in TRUE]
X.FROZEN = X.FROZEN + " + RWIRE sigma=" + ",".join(map(str, sig.tolist()))


def selftest():
    assert [len(p) for p in X.PATHS] == [len(p) for p in TRUE], "fan-in changed"
    assert sum(a != b for p, q in zip(X.PATHS, TRUE) for a, b in zip(p, q)) == sum(len(p) for p in TRUE), "some head kept its place"
    m = X.ArmS(64); ms = X.ArmS(64)
    assert int(m.mask.sum()) == sum(len(p) for p in TRUE) * X.SM * X.HM, "mask size changed"
    assert X.export_bytes(X.ArmS(X.W_S)) == 1376393, "bytes differ from S"
    print("R selftest ok; sigma", sig.tolist(), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]: selftest(); sys.exit()
    X.torch.set_num_threads(2); X.train("S", sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
