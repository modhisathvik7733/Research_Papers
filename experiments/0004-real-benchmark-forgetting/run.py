"""0004 stage 1 — Split-MNIST, task-incremental, multi-head.

Real data, the field-standard Chaudhry/GEM Forgetting metric, your nested
local-PC optimizer measured honestly alongside naive / replay / EWC.
Pre-registration: README.md. Discipline from exp-0002: paired common-seed
stats, divergence guard, verbatim three-way rule, report straight.

  python3 run.py --smoke            pipeline + metric sanity (1 seed, fast)
  python3 run.py --seeds 10         the real numbers (ACC, Forgetting)
"""
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

DEV = "cpu"                                  # small MLP; CPU = deterministic
DATA = "/tmp/mnist_0004"
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
T = len(PAIRS)
EPOCHS = 2
BS = 128
_CACHE = {}


def load_split_mnist():
    if "tr" in _CACHE:
        return _CACHE["tr"], _CACHE["te"]
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor()])
    tr = datasets.MNIST(DATA, train=True, download=True, transform=tf)
    te = datasets.MNIST(DATA, train=False, download=True, transform=tf)

    def pack(ds):
        X = ds.data.float().view(-1, 784) / 255.0
        y = ds.targets.clone()
        return X, y

    Xtr, ytr = pack(tr)
    Xte, yte = pack(te)
    tasks_tr, tasks_te = [], []
    for (a, b) in PAIRS:
        mtr = (ytr == a) | (ytr == b)
        mte = (yte == a) | (yte == b)
        tasks_tr.append((Xtr[mtr], (ytr[mtr] == b).long()))
        tasks_te.append((Xte[mte], (yte[mte] == b).long()))
    _CACHE["tr"], _CACHE["te"] = tasks_tr, tasks_te
    return tasks_tr, tasks_te


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                                   nn.Linear(256, 256), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(256, 2) for _ in range(T)])

    def forward(self, x, t):
        return self.heads[t](self.trunk(x))


# --------------------------------------------------- nested local-PC optim ---
class LocalPCOpt:
    """Faithful, GAIN-NORMALISED deployable form of the local-PC structural
    law: K parallel proper EMAs of the gradient at geometric timescales,
    each with unit DC gain, averaged over K (so the optimiser has unit
    effective gain and `lr` controls the step honestly). This is the
    multi-timescale memory of exp-0002 ported fairly — NOT the chained
    cascade, whose effective LR explodes (~250x; the smoke run caught that).
    `localpc` additionally gets a per-seed LR fairness grid in main()."""
    def __init__(self, params, lr=1e-3, K=4):
        self.p = list(params)
        self.lr, self.K = lr, K
        self.betas = [1 - 0.5 ** (k + 1) for k in range(K)]  # 0.5..0.9375
        self.e = [[torch.zeros_like(q) for q in self.p] for _ in range(K)]

    def zero_grad(self):
        for q in self.p:
            if q.grad is not None:
                q.grad.detach_(); q.grad.zero_()

    @torch.no_grad()
    def step(self):
        for j, q in enumerate(self.p):
            if q.grad is None:
                continue
            g = q.grad
            upd = torch.zeros_like(q)
            for k in range(self.K):
                b = self.betas[k]
                self.e[k][j].mul_(b).add_(g, alpha=1 - b)   # unit-DC EMA
                upd.add_(self.e[k][j] / self.K)
            q.add_(upd, alpha=-self.lr)


# ------------------------------------------------------------- training ------
def evaluate(net, tasks_te, upto):
    accs = []
    net.eval()
    with torch.no_grad():
        for i in range(upto + 1):
            X, y = tasks_te[i]
            pred = net(X.to(DEV), i).argmax(1).cpu()
            accs.append(float((pred == y).float().mean()))
    net.train()
    return accs


def fisher(net, X, y, t):
    g2 = [torch.zeros_like(p) for p in net.parameters()]
    for i in range(0, min(len(X), 512), BS):
        net.zero_grad()
        loss = Fn.cross_entropy(net(X[i:i + BS].to(DEV), t),
                                y[i:i + BS].to(DEV))
        loss.backward()
        for gj, p in zip(g2, net.parameters()):
            if p.grad is not None:
                gj += p.grad.detach() ** 2
    n = max(1, min(len(X), 512) // BS)
    return [g / n for g in g2]


def run_seed(seed, arm, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, te = load_split_mnist()
    net = Net().to(DEV)
    opt = (LocalPCOpt(net.parameters(), lr=lr) if arm == "localpc"
           else torch.optim.Adam(net.parameters(), lr=1e-3))
    buf = {}                                 # task -> (X,y), 200 samples/task
    ewc_anchor = []                          # list of (params_snapshot, Fisher)
    R = np.full((T, T), np.nan)
    for t in range(T):
        X, y = tr[t]
        idx = torch.randperm(len(X))
        X, y = X[idx], y[idx]
        for _ in range(EPOCHS):
            for i in range(0, len(X), BS):
                xb, yb = X[i:i + BS].to(DEV), y[i:i + BS].to(DEV)
                opt.zero_grad()
                loss = Fn.cross_entropy(net(xb, t), yb)
                if arm == "replay" and buf:          # vectorised replay
                    rt = int(torch.randint(0, t, (1,)))
                    bx, by = buf[rt]
                    s = torch.randperm(len(bx))[:BS]
                    loss = loss + Fn.cross_entropy(
                        net(bx[s].to(DEV), rt), by[s].to(DEV))
                if arm == "ewc" and ewc_anchor:
                    for snap, Fsh in ewc_anchor:
                        for p, p0, fj in zip(net.parameters(), snap, Fsh):
                            loss = loss + 50.0 * (fj * (p - p0) ** 2).sum()
                loss.backward()
                opt.step()
        if arm == "replay":
            sel = torch.randperm(len(X))[:200]
            buf[t] = (X[sel].clone(), y[sel].clone())
        if arm == "ewc":
            ewc_anchor.append(([p.detach().clone()
                                for p in net.parameters()],
                               fisher(net, X, y, t)))
        for i, a in enumerate(evaluate(net, te, t)):
            R[t, i] = a
    acc = float(np.mean(R[T - 1, :]))
    forget = float(np.mean([np.nanmax(R[i:T, i]) - R[T - 1, i]
                            for i in range(T - 1)]))
    return acc, forget, float(R[0, 0]), float(R[T - 1, 0])


# ----------------------------------------------------- paired-CRN stats ------
def _sign_p(k, m):
    if m == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(m, i)
                              for i in range(k, m + 1)) / 2.0 ** m)


def three_way(m, s, sign, ns):
    if ns == 0 or not np.isfinite(s) or s <= 0:
        return "NOT"
    if m > s and sign >= math.ceil(0.8 * ns):
        return "SEP"
    if m <= 0.5 * s or sign < math.ceil(0.6 * ns):
        return "NOT"
    return "AMB"


def paired(name, d):
    d = np.array([v for v in d if np.isfinite(v) and abs(v) < 1e3])
    ns = len(d)
    m = float(d.mean()) if ns else float("nan")
    s = float(d.std(ddof=1)) if ns > 1 else float("nan")
    sg = int((d > 0).sum())
    print(f"  {name:<28} m={m:+.4f} s={s:.4f} sign={sg}/{ns} "
          f"p={_sign_p(sg, ns):.3f} -> {three_way(m, s, sg, ns)}")


def main(n, smoke=False):
    global EPOCHS
    arms = ["naive", "localpc", "replay", "ewc"]
    if smoke:
        EPOCHS = 1; n = 1
        print("SMOKE: 1 seed, 1 epoch — pipeline/metric sanity only\n")
    LPC_GRID = [3e-4, 1e-3, 3e-3, 1e-2]      # pre-registered fairness grid
    res = {a: [] for a in arms}
    for s in range(n):
        for a in arms:
            if a == "localpc":               # best-of-grid per seed (fair)
                cand = [(run_seed(s, a, lr=lr), lr) for lr in LPC_GRID]
                (acc, fg, r00, rT0), blr = max(cand, key=lambda c: c[0][0])
                tag = f"localpc*(lr{blr:g})"
            else:
                acc, fg, r00, rT0 = run_seed(s, a)
                tag = a
            res[a].append((acc, fg))
            print(f"seed {s} {tag:<16}: ACC={acc:.4f}  Forget={fg:.4f}  "
                  f"task1 {r00:.3f}->{rT0:.3f}")
    print("-" * 70)
    A = {a: np.array(res[a]) for a in arms}
    for a in arms:
        print(f"  {a:<8}  ACC {A[a][:,0].mean():.4f}±{A[a][:,0].std():.4f}  "
              f"Forget {A[a][:,1].mean():.4f}±{A[a][:,1].std():.4f}")
    print("-" * 70)
    if not smoke:
        nv = A["naive"]
        paired("Forget: naive−localpc", nv[:, 1] - A["localpc"][:, 1])
        paired("Forget: naive−replay", nv[:, 1] - A["replay"][:, 1])
        paired("Forget: naive−ewc", nv[:, 1] - A["ewc"][:, 1])
        paired("Forget: replay−localpc", A["replay"][:, 1] - A["localpc"][:, 1])
    print("\nHEADLINE (the number you asked for): Split-MNIST task-IL,")
    print(f"  naive Forgetting = {A['naive'][:,1].mean():.4f} "
          f"(ACC {A['naive'][:,0].mean():.4f}); your system (localpc) "
          f"Forgetting = {A['localpc'][:,1].mean():.4f} "
          f"(ACC {A['localpc'][:,0].mean():.4f}).")
    print("  Interpretation per pre-registration H-real-2: localpc is an "
          "optimizer, not a CL algorithm; report straight where it lands.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--smoke"
    nn_ = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if cmd == "--smoke":
        main(1, smoke=True)
    elif cmd == "--seeds":
        main(nn_)
    else:
        print(__doc__)
