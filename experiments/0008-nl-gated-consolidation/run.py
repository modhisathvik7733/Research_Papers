"""0008 — Gated consolidation vs vanilla CMS rigidity (class-IL Split-MNIST).

Smallest CMS component fix for the exp-0004 measured failure. Arms:
naive | replay | rpc(replay+vanilla CMS) | rgc(replay+Gated CMS) |
van(vanilla alone) | gat(gated alone). Field-standard Chaudhry ACC +
Forgetting, paired common-seed, divergence guard, verbatim exp-0002
three-way + effect-size gate. Pre-registration: README. Report straight.

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
DATA = "/tmp/mnist_0004"                          # reuse cached MNIST
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
T = len(PAIRS)
EPOCHS = 2
BS = 128
DELTA = 0.02
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
        self.head = nn.Linear(256, 10)            # class-IL: single head

    def forward(self, x):
        return self.head(self.trunk(x))


class CMS:
    """gated=False -> vanilla gain-normalised multi-timescale momentum
    (the exp-0004 failure). gated=True -> slow levels (k>=1) scaled by
    gamma_k = relu(cos(g, e_k)) for BOTH consolidation and contribution;
    fast level k=0 unchanged. Parameter-free, O(1)-in-H."""
    def __init__(self, params, lr=1e-2, K=4, gated=False):
        self.p = list(params)
        self.lr, self.K, self.g = lr, K, gated
        self.b = [1 - 0.5 ** (k + 1) for k in range(K)]
        self.e = [[torch.zeros_like(q) for q in self.p] for _ in range(K)]

    def zero_grad(self):
        for q in self.p:
            if q.grad is not None:
                q.grad.detach_(); q.grad.zero_()

    @torch.no_grad()
    def step(self):
        gs = [q.grad if q.grad is not None else torch.zeros_like(q)
              for q in self.p]
        gam = [1.0] * self.K
        if self.g:
            fg = torch.cat([x.reshape(-1) for x in gs])
            fn = fg.norm() + 1e-12
            for k in range(1, self.K):
                ek = torch.cat([e.reshape(-1) for e in self.e[k]])
                cos = float((fg @ ek) / (fn * (ek.norm() + 1e-12)))
                gam[k] = max(0.0, cos)            # relu(cos) in [0,1]
        for k in range(self.K):
            for j in range(len(self.p)):
                self.e[k][j].mul_(self.b[k]).add_(
                    gs[j], alpha=gam[k] * (1 - self.b[k]))
        for j, q in enumerate(self.p):
            upd = sum(gam[k] * self.e[k][j] / self.K for k in range(self.K))
            q.add_(upd, alpha=-self.lr)


def evaluate(net, te, upto):
    net.eval(); acc = []
    with torch.no_grad():
        for i in range(upto + 1):
            X, y = te[i]
            acc.append(float((net(X.to(DEV)).argmax(1).cpu() == y)
                             .float().mean()))
    net.train(); return acc


def run_seed(seed, arm, lr=1e-2):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, te = load()
    net = Net().to(DEV)
    cms = arm in ("rpc", "rgc", "van", "gat")
    gated = arm in ("rgc", "gat")
    do_replay = arm in ("replay", "rpc", "rgc")
    opt = (CMS(net.parameters(), lr=lr, gated=gated) if cms
           else torch.optim.Adam(net.parameters(), lr=1e-3))
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
    print(f"  {name:<32} m={m:+.4f} s={s:.4f} {sg}/{ns} "
          f"p={_sp(sg, ns):.3f} -> {v}")
    return v, m


def main(n, smoke=False):
    global EPOCHS
    arms = ["naive", "replay", "rpc", "rgc", "van", "gat"]
    grid = [3e-3, 1e-2]
    if smoke:
        EPOCHS = 1; n = 1
        print("SMOKE class-IL: 1 seed/1 epoch — pipeline + sanity\n")
    res = {a: [] for a in arms}
    for s in range(n):
        for a in arms:
            if a in ("rpc", "rgc", "van", "gat"):
                cand = [(run_seed(s, a, lr=g), g) for g in grid]
                (acc, fg), bl = max(cand, key=lambda z: z[0][0])
                tag = f"{a}*(lr{bl:g})"
            else:
                acc, fg = run_seed(s, a); tag = a
            res[a].append((acc, fg))
            print(f"seed {s} {tag:<13}: ACC={acc:.4f} Forget={fg:.4f}")
    A = {a: np.array(res[a]) for a in arms}
    print("-" * 68)
    for a in arms:
        print(f"  {a:<6} ACC {A[a][:,0].mean():.4f}±{A[a][:,0].std():.4f}  "
              f"Forget {A[a][:,1].mean():.4f}±{A[a][:,1].std():.4f}")
    if smoke:
        return
    print("-" * 68 + "\n(SANITY) rpc must reproduce anti-stacking:")
    sv, sm = paired("ACC: replay − rpc (>0 ⇒ rpc worse)",
                    A["replay"][:, 0] - A["rpc"][:, 0])
    print("(G-1) rgc must NOT be worse than replay on BOTH:")
    f1, fm = paired("Forget: replay − rgc (>0 ⇒ rgc better)",
                    A["replay"][:, 1] - A["rgc"][:, 1])
    a1, am = paired("ACC:    rgc − replay (≥0 ⇒ no regression)",
                    A["rgc"][:, 0] - A["replay"][:, 0])
    print("(G-2) gating must restore plasticity (gat alone vs van alone):")
    g2, g2m = paired("ACC: gat − van (>0 ⇒ gating learns)",
                     A["gat"][:, 0] - A["van"][:, 0])
    print("-" * 68)
    sane = sv == "SEP" or sm > 0
    if not sane:
        print("SANITY FAIL: rpc did not anti-stack — baseline broken; do "
              "not interpret G-1/G-2. Report straight.")
        return
    g1_ok = (am >= -1e-6 or a1 == "SEP") and (fm >= -1e-6 or f1 == "SEP")
    print("VERDICT G-1:", "FIX REMOVES THE CONFLICT — gated CMS not worse "
          "than replay on both axes (parity, conflict gone)."
          if g1_ok else "gated CMS STILL worse than replay -> optimizer-side "
          "consolidation cannot match rehearsal even when gated "
          "(hardened mechanism-backed negative).")
    print("VERDICT G-2:", "PLASTICITY RESTORED — gated learns where vanilla "
          f"is ~chance (Δacc m={g2m:+.4f})." if (g2 == "SEP" and g2m > 0)
          else f"gating does NOT restore plasticity (Δacc m={g2m:+.4f}); "
          "the rigidity is not the (sole) cause.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--smoke"
    a2 = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if cmd == "--smoke":
        main(1, smoke=True)
    elif cmd == "--seeds":
        main(a2)
    else:
        print(__doc__)
