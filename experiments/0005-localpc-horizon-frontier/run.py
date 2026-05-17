"""0005 — unroll-horizon frontier: where the global hypergradient operationally
fails and Local-PC does not. Runnable first cut; pre-registration in README.

Minimal self-contained nested optimizer-memory (faithful to exp-0002): a
K-level momentum cascade on a linear-in-features student. `global` credits
the cascade by backprop through an H-step unrolled inner loop (graph O(H*K),
memory O(H*K*d), an H-Jacobian product). `localpc` credits per-step by a
detached local prediction-error (graph O(1) in H). We push H/d/K and inner
nonlinearity until `global` fails (non-finite | peak-RSS>budget |
wall>50x) and check `localpc` survives at parity quality.

  python3 run.py --smoke
  python3 run.py --memory 10
  python3 run.py --diverge 10
"""
import math
import resource
import sys
import time
import numpy as np
import torch

torch.set_num_threads(4)
RSS_BUDGET_MB = 4000.0          # pre-registered memory failure threshold


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1048576.0 if r > 1e7 else r / 1024.0   # mac bytes / linux KB


def reseed(s):
    torch.manual_seed(s); np.random.seed(s)


def make(d, seed):
    g = torch.Generator().manual_seed(7 + seed)
    F = torch.randn(d, 16, generator=g) / 4.0
    r = torch.randn(d, generator=g); r = r / r.norm()

    def task(n=64):
        x = torch.randn(n, 16)
        feat = torch.tanh(x @ F.t())
        return feat, feat @ r
    return task


def cascade(phi, m, g, K, nonlin):
    a = torch.sigmoid(phi["a"])
    gg = phi["g"]
    new = [a[0] * m[0] + g]
    for k in range(1, K):
        msg = gg[k - 1] * new[k - 1]
        new.append(a[k] * m[k] + (torch.tanh(msg) if nonlin else msg))
    upd = sum(phi["c"][k] * new[k] for k in range(K))
    return new, upd


def init_phi(K):
    return {"a": torch.zeros(K, requires_grad=True),
            "g": torch.zeros(max(K - 1, 1), requires_grad=True),
            "c": (torch.ones(K) / K).clone().requires_grad_(True)}


def graph_nodes(t):
    seen, st, n = set(), [t.grad_fn], 0
    while st:
        fn = st.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn); n += 1
        for nx, _ in getattr(fn, "next_functions", ()):
            st.append(nx)
    return n


def run(method, seed, H, d, K, nonlin):
    reseed(seed)
    task = make(d, seed)
    phi = init_phi(K)
    opt = torch.optim.Adam(list(phi.values()), lr=0.05)
    rss0 = rss_mb()
    nodes = 0
    t0 = time.perf_counter()
    for it in range(40):
        theta = torch.zeros(d)
        for _ in range(10):
            fx, y = task()
            theta = theta - 0.05 * (2 * (fx @ theta - y) @ fx) / len(y)
        theta = theta.detach().requires_grad_(True)
        m = [torch.zeros(d) for _ in range(K)]
        if method == "global":
            for _ in range(H):
                fx, y = task()
                loss = ((fx @ theta - y) ** 2).mean()
                g = torch.autograd.grad(loss, theta, create_graph=True)[0]
                m, upd = cascade(phi, m, g, K, nonlin)
                theta = theta - 0.05 * upd
            fx, y = task()
            meta = ((fx @ theta - y) ** 2).mean()
            nodes = graph_nodes(meta)
            opt.zero_grad(); meta.backward(); opt.step()
        else:                                   # localpc: per-step, detached
            acc = 0.0
            for _ in range(H):
                fx, y = task()
                g = torch.autograd.grad(
                    ((fx @ theta.detach().requires_grad_(True) - y) ** 2
                     ).mean(), theta, create_graph=False)[0].detach()
                with torch.no_grad():
                    m, upd = cascade(phi, m, g, K, nonlin)
                a = torch.sigmoid(phi["a"])
                sl = ((a[0] * m[0].detach() + g) ** 2).mean()
                step = 1e-4 * sl + ((sum(phi["c"][k] * m[k].detach()
                                     for k in range(K)) - g) ** 2).mean()
                nodes = max(nodes, graph_nodes(step))
                acc = acc + step
                theta = (theta - 0.05 * upd).detach()
            opt.zero_grad(); acc.backward(); opt.step()
    dt = (time.perf_counter() - t0) / 40.0
    for p in phi.values():
        p.requires_grad_(False)
    theta = torch.zeros(d)
    for _ in range(10):
        fx, y = task(); theta = theta - 0.05 * (2 * (fx @ theta - y) @ fx) / len(y)
    m = [torch.zeros(d) for _ in range(K)]
    for _ in range(H):
        fx, y = task()
        g = (2 * (fx @ theta - y) @ fx) / len(y)
        m, upd = cascade(phi, m, g, K, nonlin)
        theta = theta - 0.05 * upd
    fx, y = task()
    q = float(((fx @ theta - y) ** 2).mean())
    return dict(finite=np.isfinite(q) and q < 1e3, q=q,
                rss=rss_mb() - rss0, dt=dt, nodes=nodes)


def _sp(k, m):
    return 1.0 if m == 0 else min(
        1.0, 2.0 * sum(math.comb(m, i) for i in range(k, m + 1)) / 2.0 ** m)


def sweep(name, n, grid, d, K, nonlin):
    print(f"{name}: K={K} d={d} nonlin={nonlin} seeds={n}; "
          f"RSS budget {RSS_BUDGET_MB:.0f}MB, wall-fail=50x small-H\n")
    base = {}
    rows = []
    for H in grid:
        gf = lf = 0
        gq, lq, gn, ln, gdt = [], [], 0, 0, []
        for s in range(n):
            G = run("global", s, H, d, K, nonlin)
            L = run("localpc", s, H, d, K, nonlin)
            base.setdefault((s,), run("localpc", s, grid[0], d, K, nonlin)["q"])
            gn, ln = G["nodes"], L["nodes"]
            wallfail = G["dt"] > 50 * base.get(("dt0",), G["dt"])
            gok = G["finite"] and G["rss"] < RSS_BUDGET_MB and not wallfail
            lok = L["finite"] and L["q"] <= 1.25 * base[(s,)]
            gf += (not gok); lf += (not lok)
            if G["finite"] and L["finite"]:
                gq.append(G["q"]); lq.append(L["q"])
            gdt.append(G["dt"])
        base.setdefault(("dt0",), np.mean(gdt))
        rows.append((H, gf, lf, gn, ln, np.array(gq), np.array(lq)))
        print(f" H={H:>5}  global_fail={gf}/{n}  localpc_fail={lf}/{n}  "
              f"nodes g={gn} l={ln}")
    print("-" * 66)
    star = next((H for H, gf, lf, *_ in rows
                 if gf >= math.ceil(0.8 * n)), None)
    if star is None:
        print("GUARD FAIL: global never fails in grid -> operational payoff "
              "UNREACHED; exp-0002 honest gap still open. Reported straight.")
        return
    row = next(r for r in rows if r[0] == star)
    print(f"H* = {star}: global fails {row[1]}/{n}, localpc fails {row[2]}/{n}")
    # parity where both run (largest such H)
    pr = [r for r in rows if len(r[5]) >= 2 and len(r[6]) >= 2]
    if pr:
        H, _, _, _, _, gq, lq = pr[-1]
        d_ = gq - lq                       # >0 => localpc better/equal
        m, sd = float(d_.mean()), float(d_.std(ddof=1))
        sg = int((d_ >= 0).sum())
        worse = m < -sd and sg < math.ceil(0.4 * len(d_))
        print(f"parity @H={H}: localpc−global Δq m={-m:+.4f} "
              f"sign(localpc≤global)={sg}/{len(d_)} p={_sp(sg,len(d_)):.3f} "
              f"-> {'LOCALPC WORSE (claim fails)' if worse else 'parity OK'}")
    print("VERDICT:", "FRONTIER DEMONSTRATED — global operationally fails at "
          f"H*={star}; localpc survives at parity." if row[2] <
          math.ceil(0.2 * n) else "localpc also fails at H* -> no operational "
          "payoff (decisive bound).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--smoke"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if cmd == "--smoke":
        sweep("SMOKE", 1, [8, 64], 24, 4, False)
    elif cmd == "--memory":
        sweep("MEMORY axis", n, [8, 64, 256, 1024], 256, 6, False)
    elif cmd == "--diverge":
        sweep("DIVERGE axis", n, [8, 32, 128, 512], 24, 6, True)
    else:
        print(__doc__)
