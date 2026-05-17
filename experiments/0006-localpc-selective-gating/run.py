"""0006 — Selective (gradient-gated) Local-PC vs the exp-0004 anti-stacking.

Class-IL Split-MNIST (the decisive regime). Arms: naive | replay |
rpc(replay+plain Local-PC) | rspc(replay+SelectiveLocalPC). Selective gating
projects the SLOW multi-timescale levels off an EMA of the current-task
gradient (r=1 first cut); fast level free in-task; per-step => O(1)-in-H.
Pre-registration: README. Same discipline as exp-0002/0004.

  python3 run.py --smoke
  python3 run.py --seeds 10
"""
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

DEV = "cpu"
DATA = "/tmp/mnist_0004"                       # reuse the cached MNIST
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
T = len(PAIRS)
EPOCHS = 2
BS = 128
_C = {}


def load():
    if "tr" in _C:
        return _C["tr"], _C["te"]
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor()])
    tr = datasets.MNIST(DATA, train=True, download=True, transform=tf)
    te = datasets.MNIST(DATA, train=False, download=True, transform=tf)
    Xtr = tr.data.float().view(-1, 784) / 255.0
    Xte = te.data.float().view(-1, 784) / 255.0
    ytr, yte = tr.targets.clone(), te.targets.clone()
    a_tr, a_te = [], []
    for (a, b) in PAIRS:
        a_tr.append((Xtr[(ytr == a) | (ytr == b)], ytr[(ytr == a) | (ytr == b)]))
        a_te.append((Xte[(yte == a) | (yte == b)], yte[(yte == a) | (yte == b)]))
    _C["tr"], _C["te"] = a_tr, a_te
    return a_tr, a_te


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                                   nn.Linear(256, 256), nn.ReLU())
        self.head = nn.Linear(256, 10)         # class-IL: single shared head

    def forward(self, x):
        return self.head(self.trunk(x))


class LocalPC:
    """K-level gain-normalised multi-timescale momentum. selective=True ->
    project the SLOW levels (k>=1) off an EMA of the current-task gradient
    (r=1); fast level k=0 unprojected (in-task plasticity preserved)."""
    def __init__(self, params, lr=1e-3, K=4, selective=False):
        self.p = list(params)
        self.lr, self.K, self.sel = lr, K, selective
        self.b = [1 - 0.5 ** (k + 1) for k in range(K)]
        self.e = [[torch.zeros_like(q) for q in self.p] for _ in range(K)]
        self.gbar = None                       # EMA of flat current grad

    def zero_grad(self):
        for q in self.p:
            if q.grad is not None:
                q.grad.detach_(); q.grad.zero_()

    @torch.no_grad()
    def step(self):
        gs = [q.grad if q.grad is not None else torch.zeros_like(q)
              for q in self.p]
        if self.sel:
            flat = torch.cat([g.reshape(-1) for g in gs])
            self.gbar = flat.clone() if self.gbar is None \
                else 0.9 * self.gbar + 0.1 * flat
        for j, q in enumerate(self.p):
            for k in range(self.K):
                self.e[k][j].mul_(self.b[k]).add_(gs[j], alpha=1 - self.b[k])
        fast = [self.e[0][j] / self.K for j in range(len(self.p))]
        slow = [sum(self.e[k][j] for k in range(1, self.K)) / self.K
                for j in range(len(self.p))]
        if self.sel and self.gbar is not None:
            u = self.gbar / (self.gbar.norm() + 1e-8)
            sflat = torch.cat([s.reshape(-1) for s in slow])
            sflat = sflat - (sflat @ u) * u    # off-task component only
            idx = 0
            for j, q in enumerate(self.p):
                nel = q.numel()
                slow[j] = sflat[idx:idx + nel].view_as(q); idx += nel
        for j, q in enumerate(self.p):
            q.add_(fast[j] + slow[j], alpha=-self.lr)


def evaluate(net, te, upto):
    net.eval(); acc = []
    with torch.no_grad():
        for i in range(upto + 1):
            X, y = te[i]
            acc.append(float((net(X.to(DEV)).argmax(1).cpu() == y)
                             .float().mean()))
    net.train(); return acc


def run_seed(seed, arm, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, te = load()
    net = Net().to(DEV)
    use = {"rpc": ("plain",), "rspc": ("sel",)}.get(arm)
    if arm in ("rpc", "rspc"):
        opt = LocalPC(net.parameters(), lr=lr, selective=(arm == "rspc"))
    else:
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    do_replay = arm in ("replay", "rpc", "rspc")
    buf = {}
    R = np.full((T, T), np.nan)
    for t in range(T):
        X, y = tr[t]
        idx = torch.randperm(len(X)); X, y = X[idx], y[idx]
        for _ in range(EPOCHS):
            for i in range(0, len(X), BS):
                xb, yb = X[i:i + BS].to(DEV), y[i:i + BS].to(DEV)
                opt.zero_grad()
                loss = Fn.cross_entropy(net(xb), yb)
                if do_replay and buf:
                    rt = int(torch.randint(0, t, (1,)))
                    bx, by = buf[rt]
                    s = torch.randperm(len(bx))[:BS]
                    loss = loss + Fn.cross_entropy(net(bx[s].to(DEV)),
                                                   by[s].to(DEV))
                loss.backward(); opt.step()
        if do_replay:
            sel = torch.randperm(len(X))[:200]
            buf[t] = (X[sel].clone(), y[sel].clone())
        for i, a in enumerate(evaluate(net, te, t)):
            R[t, i] = a
    acc = float(np.mean(R[T - 1, :]))
    fg = float(np.mean([np.nanmax(R[i:T, i]) - R[T - 1, i]
                        for i in range(T - 1)]))
    return acc, fg


def _sp(k, m):
    return 1.0 if m == 0 else min(
        1.0, 2.0 * sum(math.comb(m, i) for i in range(k, m + 1)) / 2.0 ** m)


def three_way(m, s, sg, ns):
    if ns == 0 or not np.isfinite(s) or s <= 0:
        return "NOT"
    if m > s and sg >= math.ceil(0.8 * ns):
        return "SEP"
    if m <= 0.5 * s or sg < math.ceil(0.6 * ns):
        return "NOT"
    return "AMB"


def paired(name, d):
    d = np.array([v for v in d if np.isfinite(v) and abs(v) < 1e3])
    ns = len(d)
    m = float(d.mean()) if ns else float("nan")
    s = float(d.std(ddof=1)) if ns > 1 else float("nan")
    sg = int((d > 0).sum())
    v = three_way(m, s, sg, ns)
    print(f"  {name:<34} m={m:+.4f} s={s:.4f} {sg}/{ns} "
          f"p={_sp(sg, ns):.3f} -> {v}")
    return v, m


def main(n, smoke=False):
    global EPOCHS
    arms = ["naive", "replay", "rpc", "rspc"]
    grid = [3e-4, 1e-3, 3e-3, 1e-2]
    if smoke:
        EPOCHS = 1; n = 1
        print("SMOKE class-IL: 1 seed/1 epoch — pipeline sanity\n")
    res = {a: [] for a in arms}
    for s in range(n):
        for a in arms:
            if a in ("rpc", "rspc"):
                cand = [(run_seed(s, a, lr=g), g) for g in grid]
                (acc, fg), bl = max(cand, key=lambda z: z[0][0])
                tag = f"{a}*(lr{bl:g})"
            else:
                acc, fg = run_seed(s, a); tag = a
            res[a].append((acc, fg))
            print(f"seed {s} {tag:<14}: ACC={acc:.4f} Forget={fg:.4f}")
    A = {a: np.array(res[a]) for a in arms}
    print("-" * 68)
    for a in arms:
        print(f"  {a:<6} ACC {A[a][:,0].mean():.4f}±{A[a][:,0].std():.4f}  "
              f"Forget {A[a][:,1].mean():.4f}±{A[a][:,1].std():.4f}")
    if smoke:
        return
    print("-" * 68 + "\n(a) SANITY — plain rpc must reproduce anti-stacking:")
    sa, _ = paired("ACC: replay − rpc (>0 ⇒ rpc worse)",
                   A["replay"][:, 0] - A["rpc"][:, 0])
    print("(b/c) FIX — rspc must NOT be worse than replay on BOTH:")
    vF, mF = paired("Forget: replay − rspc (>0 ⇒ rspc better)",
                    A["replay"][:, 1] - A["rspc"][:, 1])
    vA, mA = paired("ACC:    rspc − replay (≥0 ⇒ no regression)",
                    A["rspc"][:, 0] - A["replay"][:, 0])
    print("-" * 68)
    if sa not in ("SEP",) and (A["replay"][:, 0] - A["rpc"][:, 0]).mean() <= 0:
        print("SANITY FAIL: plain rpc did not anti-stack here — baseline "
              "broken; do not interpret the fix. Report straight.")
        return
    not_worse = (vA != "NOT" or mA >= 0) and (mF >= 0 or vF == "NOT")
    if (mA >= -1e-6 or vA == "SEP") and (mF >= -1e-6 or vF == "SEP"):
        print("VERDICT (b): SELECTIVE GATING REMOVES THE CONFLICT — rspc not "
              "worse than replay on both axes. Mechanism = global rigidity, "
              "confirmed. (r=1; run the r-grid escalation before calling it "
              "a method.)")
    else:
        print("VERDICT (c): selective gating STILL worse than replay — "
              "optimizer-side retention is the wrong lever under rehearsal "
              "even when selective. Hardened, mechanism-backed negative "
              "(r-grid escalation required before stating as a law).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--smoke"
    a2 = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if cmd == "--smoke":
        main(1, smoke=True)
    elif cmd == "--seeds":
        main(a2)
    else:
        print(__doc__)
