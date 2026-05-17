"""0007 — a continual benchmark that certifies its own non-degeneracy.

OPTIMIZER-AXIS certificate (the axis P1/NL live on). An instance is
ADMISSIBLE iff the best non-trivial optimizer (global hypergradient — the
establishment standard HOPE uses, NOT our Local-PC) beats the best trivial
optimizer {sgd, best-β momentum, fixed multi-β EMA} by the verbatim
exp-0002 paired three-way rule with the effect-size gate AND a pre-registered
practical effect δ. Else REJECTED (degenerate | underpowered). Pre-reg:
README. exp-0002 construction lineage so §4.3/Eq.45 reproduces exactly.

  python3 run.py --smoke
  python3 run.py --certify 10
  python3 run.py --sweep 10
"""
import math
import sys
import numpy as np
import torch

torch.set_num_threads(4)
D_IN, D_FEAT = 16, 24
N_TASKS, STREAM = 8, 120
ETA = 0.05
ORTHO, HET, GAP = 1.0, 0.0, None          # knobs; (1,0,None)=§4.3/Eq.45
DELTA = 0.02                              # pre-registered practical effect
_F = _R = None


def reseed(s):
    global _F, _R
    torch.manual_seed(s); np.random.seed(s)
    g = torch.Generator().manual_seed(7 + s)
    _F = torch.randn(D_FEAT, D_IN, generator=g) / math.sqrt(D_IN)
    Q = torch.linalg.qr(torch.randn(D_FEAT, D_FEAT, generator=g))[0][:N_TASKS]
    Rr = torch.randn(N_TASKS, D_FEAT, generator=g)
    Rr = Rr / Rr.norm(dim=1, keepdim=True)
    mix = ORTHO * Q + (1 - ORTHO) * Rr     # ortho 1->orthonormal, 0->random
    _R = mix / mix.norm(dim=1, keepdim=True)


def task(t):
    gc = torch.Generator().manual_seed(5000 + t)
    if HET > 0:
        A = torch.randn(D_IN, D_IN, generator=gc) / math.sqrt(D_IN)
        an = (0.3 + 1.7 * torch.rand(D_IN, generator=gc)) ** HET
        L = A * an
    else:
        L = None

    def smp(n=64):
        x = torch.randn(n, D_IN)
        if L is not None:
            x = x @ L
        f = torch.tanh(x @ _F.t())
        return f, f @ _R[t % N_TASKS]
    return smp


def grad(theta, smp):
    th = theta.detach().requires_grad_(True)
    f, y = smp()
    g = torch.autograd.grad(((f @ th - y) ** 2).mean(), th)[0]
    return g.detach()


def schedule():
    B = STREAM if GAP is None else GAP
    nc = STREAM // B
    return [t for _ in range(nc) for t in range(N_TASKS) for _ in range(B)]


def stream(kind, beta=0.0, phi=None, K=4):
    tks = [task(t) for t in range(N_TASKS)]
    theta = torch.zeros(D_FEAT)
    m = [torch.zeros(D_FEAT) for _ in range(K)]
    emab = [0.5, 0.75, 0.875, 0.9375]
    ema = [torch.zeros(D_FEAT) for _ in range(4)]
    mom = torch.zeros(D_FEAT)
    sch = schedule()
    for t in sch:
        g = grad(theta, tks[t])
        if kind == "sgd":
            upd = g
        elif kind == "mom":
            mom = beta * mom + g; upd = mom
        elif kind == "emaK":
            upd = torch.zeros(D_FEAT)
            for k in range(4):
                ema[k] = emab[k] * ema[k] + (1 - emab[k]) * g
                upd = upd + ema[k] / 4
        elif kind == "global":
            with torch.no_grad():
                a = torch.sigmoid(phi["a"]); gg = phi["g"]
                new = [a[0] * m[0] + g]
                for k in range(1, K):
                    new.append(a[k] * m[k] + gg[k - 1] * new[k - 1])
                m = new
                upd = sum(phi["c"][k] * m[k] for k in range(K))
        theta = (theta - ETA * upd).detach()
    return float(np.mean([float(((lambda fy: ((fy[0] @ theta - fy[1]) ** 2)
                                  .mean())(tk(512)))) for tk in tks]))


def meta_global(seed, K=4):
    reseed(seed)
    phi = {"a": torch.zeros(K, requires_grad=True),
           "g": torch.zeros(K - 1, requires_grad=True),
           "c": (torch.ones(K) / K).clone().requires_grad_(True)}
    opt = torch.optim.Adam(list(phi.values()), lr=0.05)
    for it in range(40):
        cur = task(100 + it % 7)
        theta = torch.zeros(D_FEAT)
        for _ in range(15):
            theta = theta - ETA * grad(theta, cur)
        theta = theta.detach().requires_grad_(True)
        m = [torch.zeros(D_FEAT) for _ in range(K)]
        for _ in range(8):
            f, y = cur()
            g = torch.autograd.grad(((f @ theta - y) ** 2).mean(), theta,
                                    create_graph=True)[0]
            a = torch.sigmoid(phi["a"]); gg = phi["g"]
            new = [a[0] * m[0] + g]
            for k in range(1, K):
                new.append(a[k] * m[k] + gg[k - 1] * new[k - 1])
            m = new
            theta = theta - ETA * sum(phi["c"][k] * m[k] for k in range(K))
        f, y = cur()
        loss = ((f @ theta - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in phi.values():
        p.requires_grad_(False)
    return phi


def _sp(k, m):
    return 1.0 if m == 0 else min(
        1.0, 2.0 * sum(math.comb(m, i) for i in range(k, m + 1)) / 2.0 ** m)


def certify(name, n):
    grid = [0.0, 0.5, 0.7, 0.9, 0.95]
    DIV = lambda v: (not np.isfinite(v)) or v > 1e3
    bt, rf = [], []
    for s in range(n):
        reseed(s)
        triv = [stream("sgd")]
        triv += [stream("mom", beta=b) for b in grid]
        triv.append(stream("emaK"))
        phi = meta_global(s)
        reseed(s)
        g = stream("global", phi=phi)
        bt.append(min(v for v in triv if not DIV(v)))
        rf.append(g)
        print(f"  seed {s}: best_trivial={bt[-1]:.4f} global={g:.4f}")
    bt, rf = np.array(bt), np.array(rf)
    ok = np.array([not (DIV(a) or DIV(b)) for a, b in zip(bt, rf)])
    bt, rf = bt[ok], rf[ok]
    d = bt - rf                                  # >0 => global beats trivial
    ns = len(d)
    m = float(d.mean()); s_ = float(d.std(ddof=1)) if ns > 1 else float("nan")
    sg = int((d > 0).sum())
    sep = (m > s_) and sg >= math.ceil(0.8 * ns)
    print(f"[{name}] ortho={ORTHO} het={HET} gap={GAP}  n={ns}")
    print(f"  best_trivial {bt.mean():.4f} | global {rf.mean():.4f} | "
          f"Δ m={m:+.4f} s_pair={s_:.4f} sign={sg}/{ns} p={_sp(sg,ns):.3f}")
    if sep and m >= DELTA:
        verd = "ADMISSIBLE (optimizer axis discriminable)"
    elif (not sep) and abs(m) < max(s_, 1e-9):
        verd = ("REJECTED/degenerate (a tuned trivial optimizer matches the "
                "best non-trivial one — P1-type)")
    else:
        verd = (f"REJECTED/underpowered (gap m={m:+.4f} < δ={DELTA} or "
                f"≤ s_pair — exp-0005 lesson)")
    print(f"  CERTIFICATE: {verd}\n")
    return verd.split()[0].split("/")[0]         # ADMISSIBLE | REJECTED


def main(mode, n):
    global ORTHO, HET, GAP
    if mode == "smoke":
        n = 1
        ORTHO, HET, GAP = 1.0, 0.0, None
        print("SMOKE: §4.3/Eq.45 must trend REJECTED/degenerate (C-1)\n")
        certify("§4.3/Eq.45", 1)
        return
    if mode == "certify":
        ORTHO, HET, GAP = 1.0, 0.0, None
        c1 = certify("§4.3/Eq.45 (C-1 sanity: expect REJECT)", n)
        ORTHO, HET, GAP = 0.3, 2.0, 30
        c2 = certify("P3-lineage (C-2: expect ADMIT)", n)
        print("=" * 64)
        print(f"C-1 §4.3 -> {c1}  (must be REJECTED)")
        print(f"C-2 P3-lineage -> {c2}  (predicted ADMISSIBLE)")
        if c1 != "REJECTED":
            print("C-1 SANITY FAIL: harness broken — do not trust C-2.")
        return
    if mode == "sweep":
        print("SWEEP: locate the REJECT->ADMIT boundary (C-3)\n")
        for o, h, gp in [(1.0, 0.0, None), (1.0, 0.0, 30), (0.6, 1.0, 30),
                         (0.3, 2.0, 30), (0.3, 2.0, 15)]:
            ORTHO, HET, GAP = o, h, gp
            certify(f"ortho{o} het{h} gap{gp}", n)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--smoke"
    nn = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main({"--smoke": "smoke", "--certify": "certify",
          "--sweep": "sweep"}.get(cmd, "smoke"), nn)
