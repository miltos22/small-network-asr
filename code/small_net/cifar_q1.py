"""CIFAR-100 arms for v5.4 Q1 (Step 0 calibration baselines; PROVISIONAL geometry until the 20 Nov freeze).

Usage: python -I cifar_q1.py ARM SEED [key=value ...]
  ARM: S (shared dense head), P (private per-class heads), G (G_mix = P + fixed dense orthogonal C x C mix), R (compact ResNet-32)
  keys: frac=1.0|0.3|0.1  ep=60  wd=5e-4  drop=0  lr=0.1  bs=128  out=<json path>  eval_every=5
Rules (v5.4 section 1 + 3): learned fp32 weights <= 2,000,000 bytes (500,000 params; asserted); the official TEST images are never
loaded by this script; validation = the fixed 5,000-image stratified split; nested 10 % / 30 % subsets by per-class prefix.
Per-item validation correctness is saved for every seed (section 2). Same trunk for S / P / G; dropout only on the D-dim pre-head
tensor; weight decay only on conv / linear weight matrices (biases and norm parameters excluded identically in every arm).
"""
import os
import hashlib, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DATA = os.environ.get("SCRATCH_DIR", "scratch") + "/cifar100"
HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(HERE, "cifar100_splits.npz")
C, D, H_P, H_S = 100, 256, 6, 433   # S head 154,681 vs P head 154,900 learned (closest S <= P); width ratio 600/433 = 1.386
MAX_PARAMS = 500_000
SPLIT_SEED, MIX_SEED = 20261119, 20261120


def load_train():
    import torchvision
    ds = torchvision.datasets.CIFAR100(DATA, train=True, download=False)  # integrity (md5) checked by torchvision
    return ds.data, np.asarray(ds.targets, dtype=np.int64)            # uint8 (50000,32,32,3), labels


def make_splits(y):
    rng = np.random.default_rng(SPLIT_SEED)
    tr, va, n10, n30 = [], [], [], []
    for c in range(C):
        ix = rng.permutation(np.flatnonzero(y == c))
        assert len(ix) == 500
        va.append(ix[:50]); t = ix[50:]; tr.append(t); n10.append(t[:45]); n30.append(t[:135])
    cat = lambda L: np.sort(np.concatenate(L))
    S = dict(train=cat(tr), val=cat(va), train10=cat(n10), train30=cat(n30))
    assert len(S["train"]) == 45000 and len(S["val"]) == 5000 and not set(S["train"]) & set(S["val"])
    assert set(S["train10"]) <= set(S["train30"]) <= set(S["train"])
    return S


def fingerprint(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()


def get_splits(y):
    if os.path.exists(SPLIT):
        z = np.load(SPLIT); S = {k: z[k] for k in z.files}
        T = make_splits(y)
        assert all(np.array_equal(S[k], T[k]) for k in T), "saved splits differ from the generator"
    else:
        S = make_splits(y); np.savez(SPLIT, **S)
    return S


class Trunk(nn.Module):
    """6 conv layers + BN + ReLU, 3 max-pools, global average pool -> D=256."""
    def __init__(self):
        super().__init__()
        ch = [(3, 48, 3), (48, 48, 3), "M", (48, 96, 3), (96, 96, 3), "M", (96, 176, 3), "M", (176, D, 1)]
        L = []
        for c in ch:
            if c == "M": L.append(nn.MaxPool2d(2)); continue
            i, o, k = c
            L += [nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
        self.f = nn.Sequential(*L)
    def forward(self, x):
        return self.f(x).mean((2, 3))


class PrivHeads(nn.Module):
    """C separate heads, each D -> H_P (ReLU) -> 1 scalar. Params C*(H_P*(D+2)+1)."""
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(C, H_P, D) * math.sqrt(2.0 / D)); self.b1 = nn.Parameter(torch.zeros(C, H_P))
        self.w2 = nn.Parameter(torch.randn(C, H_P) * math.sqrt(1.0 / H_P)); self.b2 = nn.Parameter(torch.zeros(C))
    def forward(self, x):
        h = F.relu(torch.einsum("bd,chd->bch", x, self.w1) + self.b1)
        return torch.einsum("bch,ch->bc", h, self.w2) + self.b2


def mix_matrix():
    g = torch.Generator().manual_seed(MIX_SEED)
    A = torch.randn(C, C, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)               # frozen sign convention: diag(R) > 0
    I = Q.T @ Q
    assert torch.allclose(I, torch.eye(C, dtype=torch.float64), atol=1e-10)
    assert abs(float(torch.linalg.det(Q))) > 0.5
    pr = (Q.pow(2).sum(1) ** 2 / Q.pow(4).sum(1))                     # participation ratio per row (C = fully spread)
    assert float(Q.abs().max()) < 0.5 and float(pr.min()) > C / 10, "M not delocalised"
    return Q.float(), dict(max_abs=float(Q.abs().max()), pr_min=float(pr.min()), pr_mean=float(pr.mean()))


class Net(nn.Module):
    def __init__(self, arm, drop):
        super().__init__()
        self.arm = arm; self.trunk = Trunk(); self.drop = nn.Dropout(drop)
        if arm == "S":
            self.head = nn.Sequential(nn.Linear(D, H_S), nn.ReLU(inplace=True), nn.Linear(H_S, C))
        else:
            self.head = PrivHeads()
        if arm == "G":
            M, self.mix_info = mix_matrix(); self.register_buffer("M", M)   # fixed, not trained; no post-M bias
    def forward(self, x):
        z = self.head(self.drop(self.trunk(x)))
        return z @ self.M.T if self.arm == "G" else z


class Block(nn.Module):
    def __init__(self, i, o, s):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, s, 1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(o)
        self.pad = (o - i) if (s != 1 or i != o) else 0; self.s = s
    def forward(self, x):
        y = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        sc = x[:, :, ::self.s, ::self.s] if self.s != 1 else x
        if self.pad: sc = F.pad(sc, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))   # option A shortcut
        return F.relu(y + sc)


class ResNet32(nn.Module):
    """He et al. 2016 CIFAR ResNet-32 (16/32/64, 5 blocks per stage, parameter-free shortcuts)."""
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 16, 3, 1, 1, bias=False); self.b = nn.BatchNorm2d(16)
        L, i = [], 16
        for o, s in [(16, 1), (32, 2), (64, 2)]:
            for k in range(5): L.append(Block(i, o, s if k == 0 else 1)); i = o
        self.L = nn.Sequential(*L); self.fc = nn.Linear(64, C)
    def forward(self, x):
        return self.fc(self.L(F.relu(self.b(self.c(x)))).mean((2, 3)))


def counts(m):
    learned = sum(p.numel() for p in m.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in m.buffers())
    per = {n: p.numel() for n, p in m.named_parameters()}
    return learned, buffers, per


def main():
    arm, seed = sys.argv[1], int(sys.argv[2])
    kv = dict(a.split("=", 1) for a in sys.argv[3:])
    frac = float(kv.get("frac", 1.0)); ep = int(kv.get("ep", 60)); wd = float(kv.get("wd", 5e-4))
    drop = float(kv.get("drop", 0.0)); lr = float(kv.get("lr", 0.1)); bs = int(kv.get("bs", 128))
    ev = int(kv.get("eval_every", 5)); out = kv.get("out")
    assert arm in "SPGR" and len(arm) == 1 and frac in (1.0, 0.3, 0.1)
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda")
    X, y = load_train(); S = get_splits(y)
    tk = {1.0: "train", 0.3: "train30", 0.1: "train10"}[frac]
    tri, vai = S[tk], S["val"]
    mean = X[S["train"]].reshape(-1, 3).mean(0) / 255.0; std = X[S["train"]].reshape(-1, 3).std(0) / 255.0  # from the 45k split only
    to = lambda a: torch.tensor(a).permute(0, 3, 1, 2).contiguous().to(dev)
    Xtr, ytr = to(X[tri]), torch.tensor(y[tri]).to(dev); Xva, yva = to(X[vai]), torch.tensor(y[vai]).to(dev)
    mu = torch.tensor(mean, device=dev, dtype=torch.float32).view(1, 3, 1, 1); sd = torch.tensor(std, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    norm = lambda u8: (u8.float() / 255.0 - mu) / sd
    model = (ResNet32() if arm == "R" else Net(arm, drop)).to(dev).to(memory_format=torch.channels_last)
    learned, buffers, per = counts(model)
    assert learned <= MAX_PARAMS, f"{learned} learned params > {MAX_PARAMS}"
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    if arm in "PG":   # private-head weights are 3-D / 2-D tensors: decay w1, w2; b1 (C,H_P) is a bias -> move to no-decay
        decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and not n.endswith("head.b1")]
        nodecay = [p for n, p in model.named_parameters() if p.ndim < 2 or n.endswith("head.b1")]
    opt = torch.optim.SGD([dict(params=decay, weight_decay=wd), dict(params=nodecay, weight_decay=0.0)], lr=lr, momentum=0.9, nesterov=True)
    n = len(tri); spe = n // bs; total = ep * spe; warm = max(1, round(total * 2 / 60))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm))))
    g = torch.Generator(device=dev).manual_seed(seed)
    trace, t0 = [], time.time()

    def evaluate():
        model.eval(); corr = []; loss = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(vai), 1000):
                o = model(norm(Xva[i:i + 1000]).contiguous(memory_format=torch.channels_last)).float()
                loss += F.cross_entropy(o, yva[i:i + 1000], reduction="sum").item(); corr.append((o.argmax(1) == yva[i:i + 1000]).cpu())
        model.train(); c = torch.cat(corr).numpy().astype(np.uint8)
        return float(c.mean()), loss / len(vai), c

    for e in range(ep):
        perm = torch.randperm(n, device=dev, generator=g); tl, tc = 0.0, 0
        for b in range(spe):
            ix = perm[b * bs:(b + 1) * bs]; xb = Xtr[ix]
            xb = F.pad(xb, (4, 4, 4, 4))                                          # random crop (zero pad 4) + horizontal flip
            ox = torch.randint(0, 9, (len(ix),), device=dev, generator=g); oy = torch.randint(0, 9, (len(ix),), device=dev, generator=g)
            ar = torch.arange(32, device=dev)
            rows = (oy[:, None] + ar[None, :])[:, None, :, None]; cols = (ox[:, None] + ar[None, :])[:, None, None, :]
            xb = xb[torch.arange(len(ix), device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None], rows, cols]
            fl = torch.rand(len(ix), device=dev, generator=g) < 0.5; xb = torch.where(fl[:, None, None, None], xb.flip(3), xb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(norm(xb).contiguous(memory_format=torch.channels_last)); loss = F.cross_entropy(o.float(), ytr[ix])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(ix); tc += (o.argmax(1) == ytr[ix]).sum().item()
        row = dict(epoch=e + 1, train_loss=tl / (spe * bs), train_acc=tc / (spe * bs), lr=sched.get_last_lr()[0], sec=round(time.time() - t0, 1))
        if (e + 1) % ev == 0 or e + 1 == ep:
            va, vl, _ = evaluate(); row.update(val_acc=va, val_loss=vl)
        trace.append(row); print(json.dumps(row), flush=True)
        assert math.isfinite(row["train_loss"]), "non-finite loss"
    va, vl, corr = evaluate()
    res = dict(arm=arm, seed=seed, frac=frac, ep=ep, wd=wd, drop=drop, lr=lr, bs=bs, learned_params=learned, fp32_bytes=4 * learned,
               buffers=buffers, per_param=per, val_acc=va, val_loss=vl, val_correct_bits=np.packbits(corr).tobytes().hex(),
               trace=trace, gpu_sec=round(time.time() - t0, 1), torch=torch.__version__, device=torch.cuda.get_device_name(0),
               code_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
               split_fp={k: fingerprint(v) for k, v in S.items()}, test_images_loaded=False,
               mix=getattr(model, "mix_info", None))
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, "w"))
    print("FINAL", json.dumps({k: res[k] for k in ("arm", "seed", "frac", "ep", "val_acc", "learned_params", "gpu_sec")}), flush=True)


if __name__ == "__main__":
    main()
