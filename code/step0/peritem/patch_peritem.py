"""C9 5 per-item logging: final scoring of base6.py / base6s.py also records [index, e, t, h] per utterance (index = position in the
set's cache arrays, bound to the cache file sha256 in manifests). Curve evaluations (during training) are unchanged, so E1 traces and
training are unaffected. Usage: python -I patch_peritem.py <dir containing base6.py and base6s.py>. Asserts every replacement count."""
import sys, os
d = sys.argv[1]
def patch(fn, reps):
    p = os.path.join(d, fn); s = open(p).read()
    for a, b in reps:
        assert s.count(a) == 1, (fn, a[:80], s.count(a)); s = s.replace(a, b)
    open(p, "w").write(s); print("patched", p)
patch("base6.py", [
 ("def evaluate(m, X, Y, dev, n=None):\n    m.eval(); e = t = h = 0;",
  "def evaluate(m, X, Y, dev, n=None, items=False):\n    m.eval(); e = t = h = 0; it = [];"),
 ("            e += B.edit(hyp, y); t += len(y); h += len(hyp)\n    return dict(PER=round(100 * e / t, 2), hyp_ref=round(h / t, 3))\n\n\ndef pad_check",
  "            ei = B.edit(hyp, y); e += ei; t += len(y); h += len(hyp); it.append([int(i), int(ei), len(y), len(hyp)])\n"
  "    r = dict(PER=round(100 * e / t, 2), hyp_ref=round(h / t, 3))\n"
  "    if items:  # per-item counts (C9 5); totals must equal the item sums\n"
  "        assert sum(x[1] for x in it) == e and sum(x[2] for x in it) == t and sum(x[3] for x in it) == h and len(it) == len(ix)\n"
  "        r.update(e=int(e), t=int(t), h=int(h), n_items=len(it), items=it)\n"
  "    return r\n\n\ndef pad_check"),
 ("        R[\"sets\"][s_] = evaluate(m, *E[s_], dev, 300 if s_ == \"fit\" else None); print(s_, R[\"sets\"][s_], flush=True)",
  "        R[\"sets\"][s_] = evaluate(m, *E[s_], dev, 300 if s_ == \"fit\" else None, items=True)\n"
  "        print(s_, {k: v for k, v in R[\"sets\"][s_].items() if k != \"items\"}, flush=True)"),
])
patch("base6s.py", [
 ("def evaluate(arm, m, X, Y, dev, n=None):\n    m.eval(); e = t = h = 0;",
  "def evaluate(arm, m, X, Y, dev, n=None, items=False):\n    m.eval(); e = t = h = 0; it = [];"),
 ("            e += B.edit(hyp, y); t += len(y); h += len(hyp)\n    return dict(PER=round(100 * e / t, 2), hyp_ref=round(h / t, 3))\n\n\ndef selftest",
  "            ei = B.edit(hyp, y); e += ei; t += len(y); h += len(hyp); it.append([int(i), int(ei), len(y), len(hyp)])\n"
  "    r = dict(PER=round(100 * e / t, 2), hyp_ref=round(h / t, 3))\n"
  "    if items:  # per-item counts (C9 5); totals must equal the item sums\n"
  "        assert sum(x[1] for x in it) == e and sum(x[2] for x in it) == t and sum(x[3] for x in it) == h and len(it) == len(ix)\n"
  "        r.update(e=int(e), t=int(t), h=int(h), n_items=len(it), items=it)\n"
  "    return r\n\n\ndef selftest"),
 ("        R[\"sets\"][s_] = evaluate(arm, m, *E[s_], dev, 300 if s_ == \"fit\" else None); print(s_, R[\"sets\"][s_], flush=True)",
  "        R[\"sets\"][s_] = evaluate(arm, m, *E[s_], dev, 300 if s_ == \"fit\" else None, items=True)\n"
  "        print(s_, {k: v for k, v in R[\"sets\"][s_].items() if k != \"items\"}, flush=True)"),
])
