"""0004 — Split-MNIST real-benchmark forgetting + the 2x2 stacking question.

Does Local-PC help ON TOP OF replay? 2x2 = {Adam,Local-PC}x{no-replay,replay}
arms: naive / localpc / replay / rpc(=replay+localpc). Task-IL (easy, floor)
and class-IL (decisive, headroom). Field-standard Chaudhry ACC + Forgetting,
paired common-seed stats, divergence guard, verbatim exp-0002 three-way rule.
Pre-registration: README.md (stage 1 + stage 1c). Report straight.

  python3 run.py --smoke           task-IL pipeline sanity (1 seed)
  python3 run.py --seeds 10        task-IL 2x2 (floor-limited, pre-declared)
  python3 run.py --classil 10      class-IL 2x2 (THE decisive stacking test)
"""
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

DEV = "cpu"
DATA = "/tmp/mnist_0004"
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
T = len(PAIRS)
EPOCHS = 2
BS = 128
CLASSIL = False
_C = {}


def load():
    if "tr" in _C:
        return _C["tr"], _C["te"]
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor()])
    tr = datasets.MNIST(DATA, train=True, download=True, transform=tf)
    te = datasets.MNIST(DATA, train=False, download=True, transform=tf)
    Xtr = tr.data.float().view(-1, 784) / 255.0
    ytr = tr.targets.clone()
    Xte = te.data.float().view(-1, 784) / 255.0
    yte = te.targets.clone()
    a_tr, a_te = [], []
    for (a, b) in PAIRS:                       # store GLOBAL digit labels
        mtr = (ytr == a) | (ytr == b)
        mte = (yte == a) | (yte == b)
        a_tr.append((Xtr[mtr], ytr[mtr].clone()))
        a_te.append((Xte[mte], yte[mte].clone()))
    _C["tr"], _C["te"] = a_tr, a_te
    return a_tr, a_te


def lbl(yg, t):
    """task-IL: local {0,1} (is it the 2nd digit of the pair). class-IL: the
    global digit 0..9."""
    if CLASSIL:
        return yg
    return (yg == PAIRS[t][1]).long()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                                   nn.Linear(256, 256), nn.ReLU())
        if CLASSIL:
            self.head = nn.Linear(256, 10)
        else:
            self.heads = nn.ModuleList([nn.Linear(256, 2) for _ in range(T)])

    def forward(self, x, t):
        h = self.trunk(x)
        return self.head(h) if CLASSIL else self.heads[t](h)


class LocalPCOpt:
    """Gain-normalised parallel multi-timescale momentum (the faithful
    deployable form of the local-PC structural law; unit DC gain, lr honest).
    Used by `localpc` and `rpc`; both get the pre-registered LR grid."""
    def __init__(self, params, lr=1e-3, K=4):
        self.p = list(params)
        self.lr, self.K = lr, K
        self.b = [1 - 0.5 ** (k + 1) for k in range(K)]
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
            g, upd = q.grad, torch.zeros_like(q)
            for k in range(self.K):
                self.e[k][j].mul_(self.b[k]).add_(g, alpha=1 - self.b[k])
                upd.add_(self.e[k][j] / self.K)
            q.add_(upd, alpha=-self.lr)


def evaluate(net, te, upto):
    net.eval(); accs = []
    with torch.no_grad():
        for i in range(upto + 1):
            X, yg = te[i]
            pred = net(X.to(DEV), i).argmax(1).cpu()
            accs.append(float((pred == lbl(yg, i)).float().mean()))
    net.train(); return accs


def run_seed(seed, arm, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, te = load()
    net = Net().to(DEV)
    use_lpc = arm in ("localpc", "rpc")
    do_replay = arm in ("replay", "rpc")
    opt = (LocalPCOpt(net.parameters(), lr=lr) if use_lpc
           else torch.optim.Adam(net.parameters(), lr=1e-3))
    buf = {}
    R = np.full((T, T), np.nan)
    for t in range(T):
        X, yg = tr[t]
        idx = torch.randperm(len(X)); X, yg = X[idx], yg[idx]
        yl = lbl(yg, t)
        for _ in range(EPOCHS):
            for i in range(0, len(X), BS):
                xb = X[i:i + BS].to(DEV)
                yb = yl[i:i + BS].to(DEV)
                opt.zero_grad()
                loss = Fn.cross_entropy(net(xb, t), yb)
                if do_replay and buf:
                    rt = int(torch.randint(0, t, (1,)))
                    bx, bl = buf[rt]
                    s = torch.randperm(len(bx))[:BS]
                    loss = loss + Fn.cross_entropy(
                        net(bx[s].to(DEV), rt), bl[s].to(DEV))
                loss.backward(); opt.step()
        if do_replay:
            sel = torch.randperm(len(X))[:200]
            buf[t] = (X[sel].clone(), yl[sel].clone())
        for i, a in enumerate(evaluate(net, te, t)):
            R[t, i] = a
    acc = float(np.mean(R[T - 1, :]))
    forget = float(np.mean([np.nanmax(R[i:T, i]) - R[T - 1, i]
                            for i in range(T - 1)]))
    return acc, forget


# ---------------------------------------------------- paired-CRN stats -------
def _sp(k, m):
    if m == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(m, i)
                              for i in range(k, m + 1)) / 2.0 ** m)


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
    print(f"  {name:<30} m={m:+.4f} s={s:.4f} sign={sg}/{ns} "
          f"p={_sp(sg, ns):.3f} -> {v}")
    return v


def main(n, smoke=False):
    global EPOCHS
    arms = ["naive", "localpc", "replay", "rpc"]
    grid = [3e-4, 1e-3, 3e-3, 1e-2]
    if smoke:
        EPOCHS = 1; n = 1
        print("SMOKE: 1 seed, 1 epoch — pipeline sanity\n")
    proto = "class-IL" if CLASSIL else "task-IL"
    print(f"2x2 stacking test — protocol={proto}, seeds={n}\n"
          f"arms: naive | localpc | replay | rpc(replay+localpc)\n")
    res = {a: [] for a in arms}
    for s in range(n):
        for a in arms:
            if a in ("localpc", "rpc"):
                c = [(run_seed(s, a, lr=g), g) for g in grid]
                (acc, fg), bl = max(c, key=lambda z: z[0][0])
                tag = f"{a}*(lr{bl:g})"
            else:
                acc, fg = run_seed(s, a); tag = a
            res[a].append((acc, fg))
            print(f"seed {s} {tag:<15}: ACC={acc:.4f} Forget={fg:.4f}")
    A = {a: np.array(res[a]) for a in arms}
    print("-" * 70)
    for a in arms:
        print(f"  {a:<8} ACC {A[a][:,0].mean():.4f}±{A[a][:,0].std():.4f}  "
              f"Forget {A[a][:,1].mean():.4f}±{A[a][:,1].std():.4f}")
    if smoke:
        return
    print("-" * 70 + f"\nTHE BIG QUESTION (protocol={proto}):")
    big = paired("Forget: replay − rpc  (>0 ⇒ Local-PC HELPS on top)",
                 A["replay"][:, 1] - A["rpc"][:, 1])
    paired("Forget: naive − localpc", A["naive"][:, 1] - A["localpc"][:, 1])
    paired("Forget: naive − replay", A["naive"][:, 1] - A["replay"][:, 1])
    paired("ACC:    rpc − replay  (≥0 ⇒ no regression)",
           A["rpc"][:, 0] - A["replay"][:, 0])
    print("-" * 70)
    rp, rc = A["replay"][:, 1].mean(), A["rpc"][:, 1].mean()
    if big == "SEP":
        print(f"VERDICT: Local-PC STACKS on replay — Forget {rp:.4f} -> "
              f"{rc:.4f}, paired-significant. Orthogonal mechanisms.")
    elif big == "NOT":
        floor = (not CLASSIL)
        print(f"VERDICT: Local-PC is REDUNDANT with replay (Forget "
              f"{rp:.4f} vs {rc:.4f}, not separated)."
              + (" [task-IL is floor-limited — pre-declared NON-decisive; "
                 "the class-IL run is the real test.]" if floor else
                 " [class-IL has headroom — this IS the decisive verdict.]"))
    else:
        print(f"VERDICT: AMBIGUOUS (Forget {rp:.4f} vs {rc:.4f}); n=60 "
              f"confirmation pre-committed.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--smoke"
    a2 = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if cmd == "--smoke":
        main(1, smoke=True)
    elif cmd == "--seeds":
        main(a2)
    elif cmd == "--classil":
        CLASSIL = True
        main(a2)
    else:
        print(__doc__)
