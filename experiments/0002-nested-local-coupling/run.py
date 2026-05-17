"""
Experiment 0002 — Does LOCAL adjacent-level prediction-error coupling reach the
same nested-learning solution as GLOBAL backprop-through-nesting, at flat memory?

This is the single falsifiable core claim of the "post-HOPE" direction:

  A K-level nested optimizer-memory, credited by LOCAL predictive-coding errors
  between adjacent levels (detached across levels/steps -> backward graph O(1) in
  K and in unroll length H), should:
    (a) match a GLOBAL backprop-through-the-unrolled-inner-loop nested optimizer
        on continual-learning quality, and
    (b) keep per-meta-step cost ~flat as #levels K grows, while GLOBAL grows.
  A FLAT SGD baseline is included to show forgetting exists at all.

Toy faithful to Nested Learning paper sec 4.3 (orthogonal/diverse tasks, shared
params -> catastrophic forgetting) and sec 4.4-4.5 (momentum-as-memory cascade,
the §4.3 remedy = update projected off the old-task gradient subspace, GPM-style).
The nested memory's job is to *reproduce that retentive update* from a multi-level
state; the experimental variable is ONLY how its meta-params are credited.

CPU, float32, deterministic. Designed to finish in well under a minute on an
M1 Max. No external data.
"""

import time
import math
import resource
import numpy as np
import torch

torch.set_num_threads(4)
DEV = "cpu"

# ---------------------------------------------------------------- problem ----
D_IN, D_FEAT, N_TASKS = 16, 24, 10  # §4.3/Eq.45 construction
BATCH = 64
STREAM_STEPS = 120          # short budget/task -> later tasks must overwrite
FLAT_MOMENTUM = 0.9         # §4.3 forgetting mechanism: momentum mixes directions
ETA = 0.05                  # inner learning rate (applied to the produced update)

# Fixed (non-learned) random feature map shared by ALL tasks; each task t is a
# readout along an ORTHONORMAL direction r_t -> task gradients live in mutually
# orthogonal directions (paper Eq. 45). Momentum then drags the shared readout
# across these orthogonal directions => provable catastrophic forgetting.
_F = None
_R = None
HETERO = False      # set by --p2: break BOTH degeneracy sources of §4.3/Eq.45
                    #   (1) per-task input covariance C_t -> heterogeneous
                    #       Hessians Σ_t (not one shared Σ);
                    #   (2) non-orthogonal correlated task directions r_t.
                    # Under H-deg this is the construction predicted to make
                    # credit assignment matter (single scalar should separate).
EVAL_BLOCK = None   # P3: round-robin block size for the eval stream. None =
                    # original single-pass block-sequential. Set small (<<
                    # STREAM_STEPS) -> cyclic task reactivation (tasks recur).
HET_STRENGTH = 1.0  # P2-stronger knob: per-task anisotropy = (0.3+1.7u)^h.
                    # h=0 homogeneous (degenerate anchor); h=1 reproduces the
                    # original C-hetero point exactly; h>1 raises per-task
                    # condition-number spread monotonically.


def reseed(seed):
    """Re-seed EVERYTHING (torch, numpy, the shared feature map _F, the
    orthonormal task directions _R) so multi-seed runs are independent."""
    global _F, _R
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(7 + seed)
    _F = torch.randn(D_FEAT, D_IN, generator=g) / math.sqrt(D_IN)
    if HETERO:                       # P2: NON-orthogonal correlated directions
        Rr = torch.randn(N_TASKS, D_FEAT, generator=g)
        _R = Rr / Rr.norm(dim=1, keepdim=True)
    else:                            # §4.3/Eq.45: orthonormal (degenerate)
        _R = torch.linalg.qr(torch.randn(D_FEAT, D_FEAT, generator=g))[0][:N_TASKS]


reseed(0)  # preserve original single-seed behaviour by default
KS = [1, 2, 4, 8]           # nesting depths to sweep
META_ITERS = 60             # meta-training iterations for the nested optimizers
UNROLL_H = 8                # inner-loop horizon used during meta-training
LAM_RETAIN = 1.0            # weight of the retention term in the global meta-loss


def theta_dim():
    return D_FEAT  # student = a linear readout over the shared fixed features


def features(x):
    return torch.tanh(x @ _F.t())  # (B, D_FEAT), shared & non-learned


def forward(theta, x):
    return features(x) @ theta  # (B,)


def make_task(seed):
    """Task = readout along an orthonormal direction r (Eq. 45). Stream tasks
    (seed < N_TASKS) use the mutually-orthogonal rows of _R; meta-train tasks
    use a seeded random unit direction."""
    if seed < N_TASKS:
        r = _R[seed]
    else:
        g = torch.Generator().manual_seed(1000 + seed)
        r = torch.randn(D_FEAT, generator=g)
        r = r / r.norm()

    if HETERO:
        # P2: per-task anisotropic input transform L_t -> per-task Hessian
        # Σ_t = E_{x~N(0, L_t L_tᵀ)}[φ(x)φ(x)ᵀ] genuinely differs per task,
        # while the optimum stays θ*=r_t (y=φ(x)·r_t for any input law), so
        # the continual problem is well-posed but no single scalar
        # momentum/decay is simultaneously optimal across curvatures.
        gc = torch.Generator().manual_seed(5000 + seed)
        A = torch.randn(D_IN, D_IN, generator=gc) / math.sqrt(D_IN)
        # per-task anisotropy; ^HET_STRENGTH scales condition-number spread.
        # h=1 -> exactly the original C-hetero formula (backward-compatible).
        aniso = (0.3 + 1.7 * torch.rand(D_IN, generator=gc)) ** HET_STRENGTH
        L = A * aniso

        def sample(n):
            x = torch.randn(n, D_IN) @ L
            return x, features(x) @ r
    else:
        def sample(n):
            x = torch.randn(n, D_IN)
            return x, features(x) @ r

    return sample


def task_loss(theta, sample, n=BATCH):
    x, y = sample(n)
    return ((forward(theta, x) - y) ** 2).mean()


def grad_of(theta, sample, n=BATCH):
    """First-order gradient w.r.t. a fresh grad-leaf copy of theta (no graph)."""
    th = theta.detach().requires_grad_(True)
    return torch.autograd.grad(task_loss(th, sample, n), th)[0].detach()


def graph_nodes(t):
    """Exact size of the autograd graph behind a scalar loss = the structural
    quantity the claim is about (global ~ H*K, local ~ O(1))."""
    seen, stack, n = set(), [t.grad_fn], 0
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        n += 1
        for nxt, _ in getattr(fn, "next_functions", ()):
            stack.append(nxt)
    return n


# ----------------------------------------------- nested optimizer-memory ----
# meta-params Phi (per depth K): per-level decay logit a_k, per-level gain g_k
# (k>=2), combiner weights c_k. Update given surprise grad gT:
#   m_1 = sig(a_1)*m_1 + gT
#   m_k = sig(a_k)*m_k + g_k * m_{k-1}        (slower level compresses faster one)
#   delta = sum_k c_k * m_k ;  theta <- theta - ETA*delta
NONLINEAR = False   # set by the --nonlinear entrypoint: HOPE-style MLP memory
DEEP = False        # set by --deep: nonlinear PER-LEVEL recurrence too
MLP_HID = 16


def init_phi(K):
    phi = {
        "a": torch.zeros(K, requires_grad=True),          # sigmoid(0)=0.5 decay
        "g": torch.zeros(max(K - 1, 1), requires_grad=True),  # gains for k>=2
        "c": (torch.ones(K) / K).clone().requires_grad_(True),
    }
    if NONLINEAR:                                          # M(·)=(·)+W1 σ(W2 ·)
        d = theta_dim()
        phi["W2"] = (torch.randn(MLP_HID, K * d) * (1.0 / math.sqrt(K * d))
                     ).requires_grad_(True)
        phi["W1"] = (torch.randn(d, MLP_HID) * (1.0 / math.sqrt(MLP_HID))
                     ).requires_grad_(True)
    return phi


def combine(phi, levels, K):
    """Readout memory. Linear: Σ c_k m_k. Nonlinear (HOPE form): a residual
    2-layer MLP over the stacked level states (no global parallel dual form)."""
    c = phi["c"]
    lin = sum(c[k] * levels[k] for k in range(K))
    if not NONLINEAR:
        return lin
    x = torch.cat([levels[k] for k in range(K)])
    return lin + phi["W1"] @ torch.tanh(phi["W2"] @ x)


def step_memory(phi, m, gT, K):
    a, g = torch.sigmoid(phi["a"]), phi["g"]
    new = [a[0] * m[0] + gT]
    for k in range(1, K):
        msg = g[k - 1] * new[k - 1]
        # DEEP: nonlinear per-level recurrence (no global dual form at all)
        new.append(a[k] * m[k] + (torch.tanh(msg) if DEEP else msg))
    return new, combine(phi, new, K)


# ------------------------------------------- §4.3 retentive target update ----
# GPM / orthogonal-gradient remedy: descend current task but project the update
# off the stored earlier-task gradient direction. Computed WITHOUT any nesting
# graph -> a valid *local* target the memory should learn to reproduce.
def retentive_target(theta, cur_sample, anchor_dir):
    g = grad_of(theta, cur_sample)
    if anchor_dir is not None:
        g = g - (g @ anchor_dir) * anchor_dir  # remove old-task component
    return g.detach()


def anchor_direction(theta, sample):
    g = grad_of(theta, sample, 256)
    n = g.norm()
    return g / n if n > 1e-8 else g


# -------------------------------------------------- meta-train: GLOBAL -------
# Phi credited by backprop through the H-step unrolled inner loop. The retention
# meta-loss is the §4.3 objective (do well on current AND a past anchor task).
# Backward graph length ~ H*K  -> cost grows with K.
def meta_train_global(K, lr=0.05):
    phi = init_phi(K)
    opt = torch.optim.Adam(list(phi.values()), lr=lr)
    step_times, nodes = [], 0
    for it in range(META_ITERS):
        past = make_task(100 + (it % 7))      # an earlier task to NOT forget
        cur = make_task(200 + (it % 7))       # the new task
        theta = torch.zeros(theta_dim())
        # warm theta on the past task so there is something to forget
        for _ in range(20):
            theta = theta - ETA * grad_of(theta, past)
        theta = theta.detach().requires_grad_(True)
        anchor_x, anchor_y = past(256)

        t0 = time.perf_counter()
        m = [torch.zeros(theta_dim()) for _ in range(K)]
        for _ in range(UNROLL_H):
            gT = torch.autograd.grad(task_loss(theta, cur), theta,
                                     create_graph=True)[0]
            m, delta = step_memory(phi, m, gT, K)
            theta = theta - ETA * delta       # graph kept through theta & m
        retain = ((forward(theta, anchor_x) - anchor_y) ** 2).mean()
        meta_loss = task_loss(theta, cur) + LAM_RETAIN * retain
        nodes = graph_nodes(meta_loss)        # backward-graph size ~ H*K
        opt.zero_grad()
        meta_loss.backward()
        opt.step()
        step_times.append(time.perf_counter() - t0)
    for p in phi.values():
        p.requires_grad_(False)
    return phi, float(np.mean(step_times[5:])), nodes


# -------------------------------------------------- meta-train: LOCAL PC -----
# Phi credited ONLY by local adjacent-level prediction-error, everything
# detached across levels and steps -> backward graph O(1) in K and H.
#   - combiner c: local loss ||delta - p||^2 with p = §4.3 retentive target
#   - level k:    local loss ||m_k - stopgrad(slowEMA(m_{k-1}))||^2
def meta_train_localpc(K, ablate=None):
    """ablate: None=hand-designed targets (original);
       'combiner'=strip §4.3 anchor-projection (plain-grad combiner target);
       'levels'=replace slow-EMA level targets with fixed random ones;
       'both'=A1+A2 (fully de-tuned local targets)."""
    phi = init_phi(K)
    opt = torch.optim.Adam(list(phi.values()), lr=0.05)
    step_times, nodes = [], 0
    rand_tgt = [torch.randn(theta_dim()) * 0.1 for _ in range(K)]  # A2 target
    for it in range(META_ITERS):
        past = make_task(100 + (it % 7))
        cur = make_task(200 + (it % 7))
        theta = torch.zeros(theta_dim())
        for _ in range(20):
            theta = theta - ETA * grad_of(theta, past)
        a_dir = anchor_direction(theta, past)

        t0 = time.perf_counter()
        m = [torch.zeros(theta_dim()) for _ in range(K)]
        slow = [torch.zeros(theta_dim()) for _ in range(K)]
        acc = 0.0                              # one Adam step / meta-iter (fair)
        for _ in range(UNROLL_H):
            gT = grad_of(theta, cur)
            if ablate in ("combiner", "both"):
                p = gT                          # A1: plain grad, no §4.3 design
            else:
                p = retentive_target(theta, cur, a_dir)      # local target
            with torch.no_grad():
                m_next, delta = step_memory(phi, m, gT, K)
            for k in range(K):
                slow[k] = 0.9 * slow[k] + 0.1 * m_next[k].detach()

            # ---- local credit: each term's graph is O(1) in K and in H ----
            a, g, c = torch.sigmoid(phi["a"]), phi["g"], phi["c"]
            step_loss = 1e-4 * ((a[0] * m[0].detach() + gT) ** 2).mean()
            for k in range(1, K):
                prev = (a[k - 1] * m[k - 1].detach() + (gT if k == 1
                        else g[k - 2] * m[k - 2].detach()))
                pred_k = a[k] * m[k].detach() + g[k - 1] * prev.detach()
                tgt_k = (rand_tgt[k] if ablate in ("levels", "both")
                         else slow[k].detach())   # A2: uninformative target
                step_loss = step_loss + ((pred_k - tgt_k) ** 2).mean()
            delta_c = combine(phi, [mn.detach() for mn in m_next], K)
            step_loss = step_loss + ((delta_c - p) ** 2).mean()  # combiner
            nodes = max(nodes, graph_nodes(step_loss))  # per-step graph: O(1)
            acc = acc + step_loss

            m = [t.detach() for t in m_next]
            theta = (theta - ETA * delta).detach()
        opt.zero_grad()
        acc.backward()                         # single step, like global
        opt.step()
        step_times.append(time.perf_counter() - t0)
    for p in phi.values():
        p.requires_grad_(False)
    return phi, float(np.mean(step_times[5:])), nodes


# -------------------------------------------- evaluate on continual stream ---
def run_stream(phi, K, block=None, flat_beta=None):
    """Continual stream with the frozen optimizer. `block` = #consecutive steps
    per task before switching (round-robin). block>=STREAM_STEPS or None ->
    fully block-sequential (default, original behaviour); block=1 -> fully
    interleaved (~i.i.d.). Total steps/task held = STREAM_STEPS (compute-matched).
    Returns (final avg loss, forgetting = final - loss at end of task's last
    block)."""
    tasks = [make_task(seed) for seed in range(N_TASKS)]
    theta = torch.zeros(theta_dim())
    m = [torch.zeros(theta_dim()) for _ in range(K)] if phi else None
    buf = torch.zeros(theta_dim())                # flat-baseline momentum buffer
    eff = block if block is not None else EVAL_BLOCK   # P3 uses EVAL_BLOCK
    B = STREAM_STEPS if (eff is None or eff >= STREAM_STEPS) else eff
    ncyc = STREAM_STEPS // B
    schedule = [t for _ in range(ncyc) for t in range(N_TASKS) for _ in range(B)]
    last_block_end = {}                            # step idx ending task t's run
    for i, t in enumerate(schedule):
        if i + 1 == len(schedule) or schedule[i + 1] != t:
            last_block_end[t] = i
    loss_after_train = [None] * N_TASKS
    for i, t in enumerate(schedule):
        tk = tasks[t]
        gT = grad_of(theta, tk)
        if phi is None:                   # single-scalar momentum-SGD
            beta = FLAT_MOMENTUM if flat_beta is None else flat_beta
            buf = beta * buf + gT
            theta = (theta - ETA * buf).detach()
        else:
            with torch.no_grad():
                m, delta = step_memory(phi, m, gT, K)
            theta = (theta - ETA * delta).detach()
        if last_block_end.get(t) == i:
            with torch.no_grad():
                loss_after_train[t] = float(task_loss(theta, tk, 512))
    with torch.no_grad():
        final = [float(task_loss(tk, 0) if False else task_loss(theta, tk, 512))
                 for tk in tasks]
    forgetting = float(np.mean([final[i] - loss_after_train[i]
                                for i in range(N_TASKS - 1)]))
    return float(np.mean(final)), forgetting


# ------------------------------------------------------------------- main ----
def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if r > 1e7 else r / 1024  # mac=bytes, linux=KB


def main(plot=True):
    print(f"theta_dim={theta_dim()}  tasks={N_TASKS}  stream/step={STREAM_STEPS}")
    print("-" * 78)

    favg_flat, forg_flat = run_stream(None, 1)
    print(f"FLAT momentum-SGD baseline  final_avg_loss={favg_flat:.4f}  "
          f"forgetting={forg_flat:+.4f}   (momentum={FLAT_MOMENTUM})")
    print("-" * 78)
    hdr = (f"{'method':9} {'K':>2} {'final_avg':>10} {'forget':>9} "
           f"{'graph_nodes':>12} {'meta_step_s':>12}")
    print(hdr)

    res = {"global": {}, "localpc": {}}
    for K in KS:
        phi_g, tg, ng = meta_train_global(K)
        fg, frg = run_stream(phi_g, K)
        res["global"][K] = (fg, frg, tg, ng)
        print(f"{'global':9} {K:>2} {fg:>10.4f} {frg:>+9.4f} "
              f"{ng:>12d} {tg:>12.5f}")

        phi_l, tl, nl = meta_train_localpc(K)
        fl, frl = run_stream(phi_l, K)
        res["localpc"][K] = (fl, frl, tl, nl)
        print(f"{'localpc':9} {K:>2} {fl:>10.4f} {frl:>+9.4f} "
              f"{nl:>12d} {tl:>12.5f}")

    # ---- structural claim: backward-graph size vs K (exact, unconfounded) ----
    g1, g8 = res["global"][KS[0]][3], res["global"][KS[-1]][3]
    l1, l8 = res["localpc"][KS[0]][3], res["localpc"][KS[-1]][3]
    print("-" * 78)
    print(f"backward-graph nodes  K{KS[0]}->K{KS[-1]} :  "
          f"global {g1}->{g8} ({g8 / g1:.1f}x)   "
          f"localpc {l1}->{l8} ({l8 / max(l1,1):.1f}x)")
    print(f"global/localpc node ratio at K{KS[-1]}: {g8 / max(l8,1):.1f}x "
          f"(global carries the x{UNROLL_H} unroll factor; localpc does not)")

    # ---- pre-registered verdict ----
    K = KS[-1]
    fg, frg, _, _ = res["global"][K]
    fl, frl, _, _ = res["localpc"][K]
    c_quality = fl <= 1.25 * fg and fl <= favg_flat
    c_helps = forg_flat > 1e-3 and frl <= forg_flat + 1e-6
    c_flat = (g8 / g1) > 1.8 and g8 > 4 * l8
    print("-" * 78)
    print("PRE-REGISTERED VERDICT (toy scale):")
    print(f"  [{'PASS' if c_quality else 'FAIL'}] localpc final loss within 25% "
          f"of global and not worse than flat   ({fl:.4f} vs g={fg:.4f}, "
          f"flat={favg_flat:.4f})")
    print(f"  [{'PASS' if c_helps else 'FAIL'}] real forgetting exists AND "
          f"localpc <= flat                ({frl:+.4f} vs {forg_flat:+.4f})")
    print(f"  [{'PASS' if c_flat else 'FAIL'}] localpc backward-graph flat vs "
          f"global's H*K growth        (g {g8//max(g1,1)}x vs ratio "
          f"{g8//max(l8,1)}x)")
    ok = c_quality and c_helps and c_flat
    print("-" * 78)
    print("RESULT:", "post-HOPE local-coupling SUPPORTED at toy scale"
          if ok else "core claim NOT supported at toy scale (see failed checks)")

    summary = dict(
        flat=favg_flat, forg_flat=forg_flat,
        g8=res["global"][KS[-1]][0], l8=res["localpc"][KS[-1]][0],
        frl8=res["localpc"][KS[-1]][1],
        gnodes1=g1, gnodes8=g8, lnodes1=l1, lnodes8=l8,
        c_quality=c_quality, c_helps=c_helps, c_flat=c_flat, ok=ok)
    if not plot:
        return summary

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].axhline(favg_flat, ls="--", c="gray", label="flat SGD")
        ax[0].plot(KS, [res["global"][k][0] for k in KS], "o-", label="global")
        ax[0].plot(KS, [res["localpc"][k][0] for k in KS], "s-", label="localpc")
        ax[0].set_xlabel("nesting depth K"); ax[0].set_ylabel("final avg loss")
        ax[0].set_title("quality"); ax[0].legend()
        ax[1].plot(KS, [res["global"][k][2] for k in KS], "o-", label="global")
        ax[1].plot(KS, [res["localpc"][k][2] for k in KS], "s-", label="localpc")
        ax[1].set_xlabel("nesting depth K")
        ax[1].set_ylabel("meta-step wall time (s)")
        ax[1].set_title("memory/compute scaling"); ax[1].legend()
        fig.tight_layout()
        fig.savefig("experiments/0002-nested-local-coupling/result.png", dpi=110)
        print("saved plot -> experiments/0002-nested-local-coupling/result.png")
    except Exception as e:
        print("plot skipped:", e)
    return summary


def multiseed(n):
    import io
    import contextlib
    rows = []
    t0 = time.perf_counter()
    for s in range(n):
        reseed(s)
        with contextlib.redirect_stdout(io.StringIO()):
            rows.append(main(plot=False))
        r = rows[-1]
        print(f"seed {s}: ok={r['ok']}  q={int(r['c_quality'])} "
              f"h={int(r['c_helps'])} f={int(r['c_flat'])}  "
              f"l8={r['l8']:.4f} g8={r['g8']:.4f} flat={r['flat']:.4f}  "
              f"forget l/flat={r['frl8']:+.3f}/{r['forg_flat']:+.3f}  "
              f"nodes g/l={r['gnodes8']}/{r['lnodes8']}")

    def ms(key):
        v = np.array([x[key] for x in rows], float)
        return v.mean(), v.std()
    print("-" * 78)
    print(f"seeds={n}   gate pass-rates: "
          f"quality={np.mean([x['c_quality'] for x in rows]):.0%}  "
          f"helps={np.mean([x['c_helps'] for x in rows]):.0%}  "
          f"flat-graph={np.mean([x['c_flat'] for x in rows]):.0%}  "
          f"ALL={np.mean([x['ok'] for x in rows]):.0%}")
    for k, lbl in [("l8", "localpc K8 final loss"),
                   ("g8", "global  K8 final loss"),
                   ("flat", "flat        final loss"),
                   ("frl8", "localpc K8 forgetting"),
                   ("forg_flat", "flat        forgetting")]:
        m, sd = ms(k)
        print(f"  {lbl:24}: {m:+.4f} ± {sd:.4f}")
    gm, _ = ms("gnodes8")
    lm, _ = ms("lnodes8")
    print(f"  graph-node ratio g8/l8  : {gm / max(lm,1):.1f}x  "
          f"(global {gm:.0f} vs localpc {lm:.0f})")
    print(f"total wall time: {time.perf_counter() - t0:.1f}s")


def ablation(n):
    """Target-design ablation at K=8: does local-PC stay competitive with the
    global hypergradient when its hand-designed local targets are de-tuned?
    If yes -> the result is about COUPLING; if it collapses toward flat ->
    the gain was the hand-designed (§4.3 / slow-EMA) targets, not nesting."""
    K = KS[-1]
    arms = [None, "combiner", "levels", "both"]
    agg = {a: [] for a in arms}
    aggg, aggf = [], []
    t0 = time.perf_counter()
    for s in range(n):
        reseed(s)
        favg_flat, _ = run_stream(None, K)
        phi_g, _, _ = meta_train_global(K)
        fg, _ = run_stream(phi_g, K)
        aggf.append(favg_flat)
        aggg.append(fg)
        line = f"seed {s}: flat={favg_flat:.4f} global={fg:.4f} |"
        for a in arms:
            reseed(s)  # identical task draw per arm
            phi_l, _, _ = meta_train_localpc(K, ablate=a)
            fl, _ = run_stream(phi_l, K)
            agg[a].append(fl)
            line += f" lp[{a or 'orig'}]={fl:.4f}"
        print(line)

    g = np.array(aggg)
    f = np.array(aggf)
    print("-" * 78)
    print(f"seeds={n}  global={g.mean():.4f}±{g.std():.4f}  "
          f"flat={f.mean():.4f}±{f.std():.4f}")
    print(f"{'arm':10} {'final_mean':>11} {'<=1.25*global':>14} "
          f"{'<=flat':>8}  interpretation")
    for a in arms:
        v = np.array(agg[a])
        q = np.mean(v <= 1.25 * g)
        bf = np.mean(v <= f)
        tag = ("reference" if a is None else
               "survives -> COUPLING" if (v.mean() <= 1.25 * g.mean()
               and v.mean() <= f.mean()) else
               "collapses -> target-design")
        print(f"{(a or 'orig'):10} {v.mean():>11.4f} {q:>13.0%} "
              f"{bf:>7.0%}  {tag}")
    print("-" * 78)
    de = np.array(agg["both"])
    verdict = (de.mean() <= 1.25 * g.mean() and de.mean() <= f.mean())
    print("ABLATION VERDICT:",
          "fully de-tuned local-PC STILL competitive -> the contribution is "
          "the COUPLING (thesis hardened)" if verdict else
          "de-tuned local-PC COLLAPSES toward flat -> quality came from the "
          "hand-designed targets, not nesting (thesis weakened)")
    print(f"total wall time: {time.perf_counter() - t0:.1f}s")


def nonlinear(n):
    """HOPE-style nonlinear (2-layer-MLP) memory at K=8: the regime where the
    global hypergradient has NO parallel dual form and the unroll-depth wall
    actually bites. Does local-PC stay competitive AND keep its flat graph?"""
    global NONLINEAR
    NONLINEAR = True
    K = KS[-1]
    fl_, fg_, ff_, frl_, frf_, ng_, nl_, q_, h_, s_ = ([] for _ in range(10))
    t0 = time.perf_counter()
    for s in range(n):
        reseed(s)
        ff, frf = run_stream(None, K)
        phi_g, _, ng = meta_train_global(K)
        fg, _ = run_stream(phi_g, K)
        reseed(s)
        phi_l, _, nl = meta_train_localpc(K, ablate=None)
        fl, frl = run_stream(phi_l, K)
        ff_.append(ff); fg_.append(fg); fl_.append(fl)
        frf_.append(frf); frl_.append(frl); ng_.append(ng); nl_.append(nl)
        q_.append(fl <= 1.25 * fg)
        h_.append(frf > 1e-3 and frl <= frf + 1e-6)
        s_.append(ng > 4 * nl)
        print(f"seed {s}: flat={ff:.4f} global={fg:.4f} localpc={fl:.4f}  "
              f"forget l/flat={frl:+.3f}/{frf:+.3f}  nodes g/l={ng}/{nl}")
    A = np.array
    print("-" * 78)
    print(f"NONLINEAR MLP memory, K={K}, seeds={n}")
    print(f"  flat    final {A(ff_).mean():.4f}±{A(ff_).std():.4f}")
    print(f"  global  final {A(fg_).mean():.4f}±{A(fg_).std():.4f}  "
          f"graph_nodes {A(ng_).mean():.0f}")
    print(f"  localpc final {A(fl_).mean():.4f}±{A(fl_).std():.4f}  "
          f"graph_nodes {A(nl_).mean():.0f}")
    print(f"  gate pass-rate: quality={A(q_).mean():.0%}  "
          f"helps={A(h_).mean():.0%}  structural={A(s_).mean():.0%}")
    print(f"  node ratio g/l = {A(ng_).mean()/max(A(nl_).mean(),1):.1f}x "
          f"(structural claim under nonlinear memory)")
    ok = A(q_).mean() >= 0.7 and A(s_).mean() >= 0.9
    print("NONLINEAR VERDICT:",
          "local-PC stays competitive with the no-dual-form hypergradient AND "
          "keeps a flat graph -> thesis holds in the regime that matters"
          if ok else "thesis does NOT carry to nonlinear memory (see gates)")
    print(f"total wall time: {time.perf_counter() - t0:.1f}s")


def _global_meta_iter(K, H):
    """One faithful GLOBAL meta-iter (second-order, create_graph) at horizon H.
    Returns (wall_s, graph_nodes(meta_loss))."""
    phi = init_phi(K)
    opt = torch.optim.Adam(list(phi.values()), lr=0.05)
    past, cur = make_task(101), make_task(201)
    theta = torch.zeros(theta_dim())
    for _ in range(20):
        theta = theta - ETA * grad_of(theta, past)
    theta = theta.detach().requires_grad_(True)
    ax, ay = past(256)
    t0 = time.perf_counter()
    m = [torch.zeros(theta_dim()) for _ in range(K)]
    for _ in range(H):
        gT = torch.autograd.grad(task_loss(theta, cur), theta,
                                 create_graph=True)[0]
        m, delta = step_memory(phi, m, gT, K)
        theta = theta - ETA * delta
    meta_loss = task_loss(theta, cur) + LAM_RETAIN * (
        (forward(theta, ax) - ay) ** 2).mean()
    n = graph_nodes(meta_loss)
    opt.zero_grad()
    meta_loss.backward()
    opt.step()
    return time.perf_counter() - t0, n


def _localpc_meta_iter_online(K, H):
    """One ONLINE local-PC meta-iter at horizon H: per-step detached local
    update, graph discarded each step => O(1) in H by construction.
    Returns (wall_s, per_step_graph_nodes)."""
    phi = init_phi(K)
    opt = torch.optim.Adam(list(phi.values()), lr=0.05)
    past, cur = make_task(101), make_task(201)
    theta = torch.zeros(theta_dim())
    for _ in range(20):
        theta = theta - ETA * grad_of(theta, past)
    a_dir = anchor_direction(theta, past)
    rand_tgt = [torch.randn(theta_dim()) * 0.1 for _ in range(K)]
    t0, pn = time.perf_counter(), 0
    m = [torch.zeros(theta_dim()) for _ in range(K)]
    slow = [torch.zeros(theta_dim()) for _ in range(K)]
    for _ in range(H):
        gT = grad_of(theta, cur)
        p = retentive_target(theta, cur, a_dir)
        with torch.no_grad():
            m_next, delta = step_memory(phi, m, gT, K)
        for k in range(K):
            slow[k] = 0.9 * slow[k] + 0.1 * m_next[k].detach()
        a, g = torch.sigmoid(phi["a"]), phi["g"]
        sl = 1e-4 * ((a[0] * m[0].detach() + gT) ** 2).mean()
        for k in range(1, K):
            prev = (a[k - 1] * m[k - 1].detach() + (gT if k == 1
                    else g[k - 2] * m[k - 2].detach()))
            pred_k = a[k] * m[k].detach() + g[k - 1] * prev.detach()
            sl = sl + ((pred_k - slow[k].detach()) ** 2).mean()
        sl = sl + ((combine(phi, [mn.detach() for mn in m_next], K) - p)
                   ** 2).mean()
        pn = max(pn, graph_nodes(sl))
        opt.zero_grad()
        sl.backward()                      # graph freed every step -> O(1) in H
        opt.step()
        m = [t.detach() for t in m_next]
        theta = (theta - ETA * delta).detach()
    return time.perf_counter() - t0, pn


def scale_to_failure(budget_s=25.0):
    """Push the unroll horizon H (NL's O(H*K) wall driver) with nonlinear
    memory (no global dual form). global is aborted once one meta-iter exceeds
    the wall-time budget; online local-PC continues, flat in H."""
    global NONLINEAR
    NONLINEAR = True
    reseed(0)
    K = KS[-1]
    Hs = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    print(f"scale-to-failure: nonlinear MLP memory, K={K}, "
          f"global wall-time budget={budget_s}s/meta-iter")
    print(f"{'H':>5} | {'global_s':>9} {'g_nodes':>9} | "
          f"{'localpc_s':>10} {'l_nodes':>8} | {'speedup':>8} {'node_x':>7}")
    g_dead = False
    rows = []
    for H in Hs:
        if not g_dead:
            gt, gn = _global_meta_iter(K, H)
        else:
            gt, gn = float("nan"), -1
        lt, ln = _localpc_meta_iter_online(K, H)
        sp = (gt / lt) if gt == gt else float("nan")
        nx = (gn / max(ln, 1)) if gn > 0 else float("nan")
        rows.append((H, gt, gn, lt, ln, sp, nx))
        gs = f"{gt:9.3f} {gn:9d}" if gt == gt else f"{'ABORTED':>9} {'-':>9}"
        print(f"{H:>5} | {gs} | {lt:>10.3f} {ln:>8d} | "
              f"{sp:>8.1f} {nx:>7.1f}" if gt == gt else
              f"{H:>5} | {gs} | {lt:>10.3f} {ln:>8d} | {'-':>8} {'-':>7}")
        if gt == gt and gt > budget_s:
            print(f"      -> GLOBAL HIT THE WALL at H={H} "
                  f"({gt:.1f}s/meta-iter > {budget_s}s). local-PC unaffected.")
            g_dead = True
    last_ok = [r for r in rows if r[1] == r[1]][-1]
    print("-" * 78)
    print(f"At the largest H both ran (H={last_ok[0]}): "
          f"global {last_ok[1]:.2f}s / {last_ok[2]} nodes  vs  "
          f"local-PC {last_ok[3]:.2f}s / {last_ok[4]} nodes  "
          f"=> {last_ok[5]:.0f}x faster, {last_ok[6]:.0f}x smaller graph.")
    print("local-PC wall-time & graph are ~flat in H (online, graph freed "
          "per step); global scales ~linearly in H and OOM/time-bounds.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        H = [r[0] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        gw = [r[1] for r in rows]
        ax[0].plot(H, gw, "o-", label="global (backprop-thru-nesting)")
        ax[0].plot(H, [r[3] for r in rows], "s-", label="local-PC (online)")
        ax[0].axhline(budget_s, ls="--", c="r", label="wall-time budget")
        ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
        ax[0].set_xlabel("unroll horizon H"); ax[0].set_ylabel("meta-iter s")
        ax[0].set_title("scale-to-failure: wall time"); ax[0].legend()
        ax[1].plot(H, [r[2] for r in rows], "o-", label="global")
        ax[1].plot(H, [r[4] for r in rows], "s-", label="local-PC")
        ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
        ax[1].set_xlabel("unroll horizon H")
        ax[1].set_ylabel("backward-graph nodes")
        ax[1].set_title("backward-graph size"); ax[1].legend()
        fig.tight_layout()
        fig.savefig("experiments/0002-nested-local-coupling/scale.png", dpi=110)
        print("saved -> experiments/0002-nested-local-coupling/scale.png")
    except Exception as e:
        print("plot skipped:", e)


def deep(n):
    """Deep per-level NONLINEAR recursive memory (global has no dual form at
    ANY level). Fairness: global gets the BEST of a meta-LR grid per seed;
    local-PC keeps a single default LR (no tuning advantage). If local-PC is
    still competitive vs best-tuned global, the result is not a global-undertune
    artifact."""
    global NONLINEAR, DEEP
    NONLINEAR = True
    DEEP = True
    K = KS[-1]
    LRS = [0.01, 0.02, 0.05, 0.1, 0.2]
    fl_, fg_, ff_, frl_, frf_, q_, h_, s_, lrpick = ([] for _ in range(9))
    t0 = time.perf_counter()
    for s in range(n):
        reseed(s)
        ff, frf = run_stream(None, K)
        # global: best of the LR grid (give it its best shot)
        best = (1e9, None, None)
        for lr in LRS:
            reseed(s)
            phi_g, _, ng = meta_train_global(K, lr=lr)
            fg_lr, _ = run_stream(phi_g, K)
            if fg_lr < best[0]:
                best = (fg_lr, lr, ng)
        fg, lr_star, ng = best
        reseed(s)
        phi_l, _, nl = meta_train_localpc(K, ablate=None)
        fl, frl = run_stream(phi_l, K)
        ff_.append(ff); fg_.append(fg); fl_.append(fl)
        frf_.append(frf); frl_.append(frl); lrpick.append(lr_star)
        q_.append(fl <= 1.25 * fg)
        h_.append(frf > 1e-3 and frl <= frf + 1e-6)
        s_.append(ng > 4 * nl)
        print(f"seed {s}: flat={ff:.4f}  global*={fg:.4f}(lr*={lr_star})  "
              f"localpc={fl:.4f}  forget l/flat={frl:+.3f}/{frf:+.3f}  "
              f"nodes g/l={ng}/{nl}")
    A = np.array
    print("-" * 78)
    print(f"DEEP nonlinear recursive memory, K={K}, seeds={n} "
          f"(global = best-of-LR-grid per seed)")
    print(f"  flat     final {A(ff_).mean():.4f}±{A(ff_).std():.4f}")
    print(f"  global*  final {A(fg_).mean():.4f}±{A(fg_).std():.4f}  "
          f"(picked LRs: {sorted(set(lrpick))})")
    print(f"  localpc  final {A(fl_).mean():.4f}±{A(fl_).std():.4f}  "
          f"(single default LR, untuned)")
    print(f"  gate pass-rate: quality(<=1.25*global*)={A(q_).mean():.0%}  "
          f"helps={A(h_).mean():.0%}  structural={A(s_).mean():.0%}")
    ok = A(q_).mean() >= 0.7 and A(s_).mean() >= 0.9
    print("DEEP+FAIRNESS VERDICT:",
          "local-PC (untuned) stays competitive with BEST-tuned global under "
          "fully nonlinear recursive memory -> not a global-undertune artifact"
          if ok else "thesis weakens once global is fairly tuned / memory is "
          "deeply nonlinear (see gates)")
    print(f"total wall time: {time.perf_counter() - t0:.1f}s")


def p1(n):
    """ADVERSARIAL CONTROL (§7.6). The single scalar = momentum β (this *is*
    the K=1 degenerate case of the cascade). Per seed: grid-search β, take its
    best continual loss; compare to K=8 local-PC and K=8 global on the SAME
    task draw. Pre-registered rule:
      - if best-β ties BOTH (≤1.25× and within ~1 std) -> benchmark DEGENERATE,
        Contribution B deflated (credit assignment moot; indicts hypergradient
        too);
      - if both nested clearly beat best-β -> NESTING MATTERS, B stands.
    Linear regime, K=8 (matches the §5.1 headline parity claim)."""
    K = KS[-1]
    betas = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    sc_, bbeta_, ll_, lg_ = [], [], [], []
    t0 = time.perf_counter()
    for s in range(n):
        reseed(s)
        scores = [(run_stream(None, 1, flat_beta=b)[0], b) for b in betas]
        best_sc, best_b = min(scores, key=lambda x: x[0])
        reseed(s)
        phi_l, _, _ = meta_train_localpc(K, ablate=None)
        Ll, _ = run_stream(phi_l, K)
        reseed(s)
        phi_g, _, _ = meta_train_global(K)
        Lg, _ = run_stream(phi_g, K)
        sc_.append(best_sc); bbeta_.append(best_b); ll_.append(Ll); lg_.append(Lg)
        print(f"seed {s}: best-scalar={best_sc:.4f}(β={best_b}) "
              f"localPC={Ll:.4f} global={Lg:.4f}")
    A = np.array
    sc, ll, lg = A(sc_), A(ll_), A(lg_)

    # --- divergence guard (added 2026-05-17, backward-compatible) ----------
    # The pre-registered tie test silently averaged losses; on heterogeneous
    # geometry the global hypergradient can DIVERGE (classic L2O pathology),
    # and a single 1e23 corrupts mean & pooled std so the tie test passes
    # vacuously. We treat a run as diverged if non-finite or > 1e3 (on the
    # §4.3/Eq.45 degenerate benchmark losses are ~0.6 and NOTHING diverges,
    # so --p1's original output/verdict is unchanged). Divergence rate is a
    # first-class result about the method, not something to average through.
    DIV = lambda v: (not np.isfinite(v)) or v > 1e3
    d_sc = int(sum(DIV(v) for v in sc_))
    d_ll = int(sum(DIV(v) for v in ll_))
    d_lg = int(sum(DIV(v) for v in lg_))
    stable = A([not (DIV(ll_[i]) or DIV(lg_[i]) or DIV(sc_[i]))
                for i in range(n)])
    ns = int(stable.sum())
    scS, llS, lgS = sc[stable], ll[stable], lg[stable]
    nested_best = np.minimum(llS, lgS) if ns else A([np.nan])
    print("-" * 74)
    print(f"P1 (linear, K={K}, seeds={n})")
    print(f"  divergences: scalar={d_sc}/{n}  local-PC={d_ll}/{n}  "
          f"global={d_lg}/{n}   (stable seeds used for verdict: {ns}/{n})")
    if ns:
        print(f"  best single-scalar : {scS.mean():.4f} ± {scS.std():.4f}  "
              f"med {np.median(scS):.4f}  (β picks: {sorted(set(bbeta_))})")
        print(f"  local-PC  (K=8)    : {llS.mean():.4f} ± {llS.std():.4f}  "
              f"med {np.median(llS):.4f}")
        print(f"  global    (K=8)    : {lgS.mean():.4f} ± {lgS.std():.4f}  "
              f"med {np.median(lgS):.4f}")
    margin = (scS.mean() - nested_best.mean()) if ns else float("nan")
    pooled = float(np.std(np.concatenate([llS, lgS]))) if ns else float("nan")
    # verdict on the STABLE subset (pre-registered rule, unchanged); global
    # divergence is reported separately and never silently averaged.
    ties = ns > 0 and scS.mean() <= 1.25 * nested_best.mean() \
        and margin <= pooled
    nest_wins = ns > 0 and scS.mean() > nested_best.mean() + 0.5 * pooled
    print(f"  best-scalar − best-nested = {margin:+.4f}  "
          f"(pooled std≈{pooled:.3f}; stable subset)")
    if d_lg or d_ll:
        print(f"  ** INSTABILITY: the {'global hypergradient' if d_lg else ''}"
              f"{' & ' if d_lg and d_ll else ''}"
              f"{'local-PC' if d_ll else ''} diverged on "
              f"{max(d_lg, d_ll)}/{n} seeds — a first-class finding: on this "
              f"geometry nested credit assignment is NOT free/robust.")
    print("-" * 74)
    if ties:
        print("P1 VERDICT: DEGENERATE — a single tuned scalar matches the "
              "K=8 nested optimizers. Credit-assignment is moot on this "
              "benchmark; Contribution B is deflated (and the expensive "
              "hypergradient is indicted on the same grounds).")
    elif nest_wins:
        print("P1 VERDICT: NESTING MATTERS — both nested optimizers clearly "
              "beat the best single scalar. The benchmark is NOT degenerate; "
              "the credit-assignment parity (Contribution B) stands.")
    else:
        print("P1 VERDICT: AMBIGUOUS — single scalar neither clearly ties nor "
              "is clearly beaten. Report as inconclusive; do not over-claim B.")
    if d_lg or d_ll:
        print("  (NOTE: verdict is on the STABLE subset only. The divergence "
              "reported above is a SEPARATE, first-class result and must be "
              "reported alongside the verdict — a 'DEGENERATE'/'MATTERS' label "
              "on the stable subset does NOT erase that the hypergradient is "
              "unstable on this geometry.)")
    print(f"total wall time: {time.perf_counter() - t0:.1f}s")
    verdict = "DEGENERATE" if ties else ("NESTING" if nest_wins else "AMBIG")
    return {"verdict": verdict, "ns": ns, "n": n,
            "scalar": float(scS.mean()) if ns else float("nan"),
            "nested_best": float(nested_best.mean()) if ns else float("nan"),
            "margin": float(margin), "pooled": float(pooled),
            "div_gl": d_lg, "div_ll": d_ll}


def p2_strength(n):
    """P2-stronger (§7.6.3). Sweep heterogeneity strength h; per-task
    anisotropy = (0.3+1.7u)^h. h=0 homogeneous (degenerate anchor), h=1 = the
    reproduced C-hetero point, h>1 = more heterogeneous Hessians. Same p1()
    decision rule reused verbatim per h. Pre-registered: margin grows
    monotone in h and the rule flips DEGENERATE->NESTING at some h*, OR it
    never flips (=> the §4.3/Eq.45 family is robustly degenerate, which
    BOUNDS — does not rescue — the structural law)."""
    global HETERO, HET_STRENGTH
    HETERO = True
    grid = [0.0, 1.0, 2.0, 3.0, 4.0]
    print("P2-STRONGER heterogeneity sweep (toy size, K=8)")
    print("PRE-REGISTERED: margin grows monotone in h; verbatim rule flips "
          "DEGENERATE->NESTING at some h*, else family robustly degenerate "
          "(bounds Contribution A). h=0 must read DEGENERATE (sanity).\n")
    rows = []
    for h in grid:
        HET_STRENGTH = h
        reseed(0)
        print(f"\n========== heterogeneity strength h = {h} ==========")
        r = p1(n)
        rows.append((h, r))
    print("\n" + "=" * 74)
    print("P2-STRONGER SUMMARY")
    print(f"{'h':>4} {'scalar':>8} {'nest_best':>10} {'margin':>8} "
          f"{'pooled':>8} {'div_gl':>7} {'verdict':>11}")
    for h, r in rows:
        print(f"{h:>4} {r['scalar']:>8.4f} {r['nested_best']:>10.4f} "
              f"{r['margin']:>+8.4f} {r['pooled']:>8.4f} "
              f"{r['div_gl']:>4}/{r['n']:<2} {r['verdict']:>11}")
    margins = [r["margin"] for _, r in rows]
    mono = all(margins[i] <= margins[i + 1] + 1e-6
               for i in range(len(margins) - 1))
    flip = next((h for h, r in rows if r["verdict"] == "NESTING"), None)
    h0 = next(r for h, r in rows if h == 0.0)
    print("-" * 74)
    print(f"margin monotone non-decreasing in h: {mono}")
    print(f"h=0 sanity (must be DEGENERATE): {h0['verdict']}")
    if flip is not None:
        print(f"P2-STRONGER VERDICT: rule FLIPS to NESTING MATTERS at h*={flip}"
              " — heterogeneity lifts the degeneracy; the structural law has a "
              "regime where credit assignment genuinely matters.")
    else:
        print("P2-STRONGER VERDICT: rule NEVER flips up to h=4 — the "
              "§4.3/Eq.45 family is ROBUSTLY optimizer-degenerate even under "
              "severe per-task curvature heterogeneity. This BOUNDS (does not "
              "rescue) Contribution A: the next benchmark must leave the "
              "family entirely (P3).")


def p3(n):
    """P3 (§7.6.5) — the decisive non-degenerate benchmark. Leaves the
    §4.3/Eq.45 family on the TEMPORAL axis: cyclic task reactivation. Small
    task set, short blocks, many recurrences -> the recurrence gap exceeds any
    single momentum/decay's useful horizon, so a tuned scalar provably cannot
    both adapt-in-block and retain-across-gap; a fast+slow nested memory can.

    Two pre-registered verdicts:
      P3-A non-degeneracy: best scalar CLEARLY worse than best nested
        (scalar_mean - nested_best_mean > pooled std, stable subset) — the
        inverse of the P1 tie rule. If it fails, P3 is itself degenerate.
      P3-B which-is-better (only if P3-A holds): qgap = global - localPC.
        |qgap|<=pooled -> quality tie -> decided by stability (divergences)
        then cost (graph nodes) -> dominance verdict; qgap>pooled -> local-PC
        better on quality; qgap<-pooled -> global better on quality."""
    global N_TASKS, STREAM_STEPS, EVAL_BLOCK
    N_TASKS, STREAM_STEPS, EVAL_BLOCK = 3, 600, 30   # 20 recurrences/task,
    K = KS[-1]                                       # gap 60 >> short-β horizon
    betas = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    DIV = lambda v: (not np.isfinite(v)) or v > 1e3
    sc_, bb_, ll_, lg_ = [], [], [], []
    nl = ng = 0
    t0 = time.perf_counter()
    print(f"P3 (cyclic reactivation): tasks={N_TASKS} block={EVAL_BLOCK} "
          f"stream={STREAM_STEPS} (recur every {N_TASKS*EVAL_BLOCK} steps, "
          f"gap {(N_TASKS-1)*EVAL_BLOCK}); K={K}, seeds={n}")
    print("PRE-REGISTERED: P3-A scalar CLEARLY worse than best nested "
          "(margin > pooled); P3-B then decides local-PC vs global.\n")
    for s in range(n):
        reseed(s)
        best_sc, best_b = min(((run_stream(None, 1, flat_beta=b)[0], b)
                               for b in betas), key=lambda x: x[0])
        reseed(s)
        phi_l, _, nlk = meta_train_localpc(K, ablate=None)
        Ll, _ = run_stream(phi_l, K)
        reseed(s)
        phi_g, _, ngk = meta_train_global(K)
        Lg, _ = run_stream(phi_g, K)
        nl, ng = nlk, ngk
        sc_.append(best_sc); bb_.append(best_b); ll_.append(Ll); lg_.append(Lg)
        print(f"seed {s}: scalar={best_sc:.4f}(β={best_b}) "
              f"localPC={Ll:.4f} global={Lg:.4f}")
    A = np.array
    d_ll = int(sum(DIV(v) for v in ll_)); d_lg = int(sum(DIV(v) for v in lg_))
    d_sc = int(sum(DIV(v) for v in sc_))
    st = A([not (DIV(ll_[i]) or DIV(lg_[i]) or DIV(sc_[i])) for i in range(n)])
    ns = int(st.sum())
    sc, ll, lg = A(sc_)[st], A(ll_)[st], A(lg_)[st]
    nb = np.minimum(ll, lg)
    pooled = float(np.std(np.concatenate([ll, lg]))) if ns else float("nan")
    print("-" * 74)
    print(f"P3  stable={ns}/{n}  divergences sc={d_sc} lpc={d_ll} gl={d_lg}")
    print(f"  best scalar : {sc.mean():.4f} ± {sc.std():.4f}")
    print(f"  local-PC    : {ll.mean():.4f} ± {ll.std():.4f}  "
          f"(graph {nl} nodes, O(1) in H)")
    print(f"  global      : {lg.mean():.4f} ± {lg.std():.4f}  "
          f"(graph {ng} nodes, O(H·K))")
    sc_margin = float(sc.mean() - nb.mean())
    nondegen = ns > 0 and sc_margin > pooled
    print(f"  scalar − best-nested = {sc_margin:+.4f}  (pooled std≈"
          f"{pooled:.3f})")
    print("-" * 74)
    if not nondegen:
        print("P3-A VERDICT: DEGENERATE — the tuned scalar is NOT clearly "
              "beaten (margin ≤ pooled). Even cyclic reactivation does not "
              "make credit assignment matter at this toy scale. The local "
              "idea remains UNDETERMINED; report straight, do not spin.")
        print(f"total wall time: {time.perf_counter() - t0:.1f}s")
        return
    print("P3-A VERDICT: NON-DEGENERATE — the tuned scalar is clearly beaten "
          "by the nested optimizer. Credit assignment MATTERS here. P3-B now "
          "decides which nested rule is better:")
    qgap = float(lg.mean() - ll.mean())   # >0 => local-PC better quality
    print(f"  P3-B  qgap (global − local-PC) = {qgap:+.4f}  "
          f"(pooled≈{pooled:.3f}); div lpc={d_ll} gl={d_lg}; "
          f"graph lpc={nl} gl={ng}")
    if qgap > pooled:
        print("P3-B VERDICT: ** LOCAL-PC BETTER ** — strictly better quality "
              "AND O(1)-in-H vs O(H·K). Unambiguous.")
    elif qgap < -pooled:
        print("P3-B VERDICT: ** GLOBAL BETTER on quality ** — the "
              "hypergradient wins quality beyond pooled std. Weigh against "
              "its O(H·K) backward cost and any divergence; the local idea "
              "is cheaper/stabler but quality-inferior here. Reported "
              "straight (this refutes the pre-registered prediction).")
    else:
        cheaper = nl < ng
        stable_ok = d_ll <= d_lg
        if stable_ok and cheaper:
            print(f"P3-B VERDICT: ** LOCAL-PC BETTER (by dominance) ** — "
                  f"quality statistically EQUIVALENT (|qgap|≤pooled), local-PC "
                  f"strictly cheaper (graph {nl} vs {ng}, O(1) vs O(H·K)) and "
                  f"{'strictly more stable' if d_ll < d_lg else '≥ as stable'} "
                  f"(div {d_ll} vs {d_lg}). Equal quality at lower cost ⇒ "
                  f"local-PC is the better method on a non-degenerate "
                  f"benchmark.")
        elif not stable_ok:
            print(f"P3-B VERDICT: MIXED — quality tie (|qgap|≤pooled) and "
                  f"local-PC cheaper (graph {nl} vs {ng}) BUT local-PC "
                  f"diverged MORE (div {d_ll} vs {d_lg}). Not a clean win; "
                  f"reported straight as mixed.")
        else:
            print(f"P3-B VERDICT: quality tie; local-PC NOT cheaper "
                  f"(graph {nl} vs {ng}). Report as equivalent, no dominance.")
    print(f"total wall time: {time.perf_counter() - t0:.1f}s")


# ---------------------------------------------------------------- paired ----
# §7.6.7 — pre-registered NEW analysis (separately coded, not a re-slice).
# p1/p3 already reseed identically per seed -> common random numbers ->
# paired design. The verbatim rule used the BETWEEN-seed std (conservative /
# mis-specified for a paired design); the correct variance is the std of the
# per-seed DIFFERENCES. Original between-seed verdicts are RE-PRINTED here
# unchanged, with the paired verdict alongside. Honesty guardrail enforced
# in the printout.
def _sign_p(k, m):
    """Exact two-sided sign-test p for k successes out of m fair trials."""
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(k, m + 1)) / (2.0 ** m)
    return min(1.0, 2.0 * tail)


def _three_way(m, s, sign, ns, lo=0.5, hi=1.0, fhi=0.8, flo=0.6):
    """Mirror of the verbatim rule, paired variance. Returns
    'SEP'|'NOT'|'AMB'. SEP iff m>hi*s and sign>=ceil(fhi*ns); NOT iff
    m<=lo*s or sign<ceil(flo*ns); AMB otherwise."""
    import math as _m
    if ns == 0:
        return "NOT"
    need_hi, need_lo = _m.ceil(fhi * ns), _m.ceil(flo * ns)
    if m > hi * s and sign >= need_hi:
        return "SEP"
    if m <= lo * s or sign < need_lo:
        return "NOT"
    return "AMB"


def paired_core(n):
    """One per-seed loop, common random numbers (as the existing code already
    does). Returns stable per-seed arrays + graph nodes + divergence counts."""
    K = KS[-1]
    betas = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    DIV = lambda v: (not np.isfinite(v)) or v > 1e3
    sc_, ll_, lg_ = [], [], []
    nl = ng = 0
    for s in range(n):
        reseed(s)
        best_sc = min(run_stream(None, 1, flat_beta=b)[0] for b in betas)
        reseed(s)
        phi_l, _, nl = meta_train_localpc(K, ablate=None)
        Ll, _ = run_stream(phi_l, K)
        reseed(s)
        phi_g, _, ng = meta_train_global(K)
        Lg, _ = run_stream(phi_g, K)
        sc_.append(best_sc); ll_.append(Ll); lg_.append(Lg)
        print(f"  seed {s}: scalar={best_sc:.4f} localPC={Ll:.4f} "
              f"global={Lg:.4f}")
    A = np.array
    d_sc = int(sum(DIV(v) for v in sc_)); d_ll = int(sum(DIV(v) for v in ll_))
    d_lg = int(sum(DIV(v) for v in lg_))
    st = A([not (DIV(sc_[i]) or DIV(ll_[i]) or DIV(lg_[i]))
            for i in range(n)])
    return (A(sc_)[st], A(ll_)[st], A(lg_)[st], nl, ng, d_sc, d_ll, d_lg,
            int(st.sum()))


def paired_report(label, sc, ll, lg, nl, ng, d_sc, d_ll, d_lg, ns):
    nb = np.minimum(ll, lg)
    print(f"\n----- {label}  (stable {ns}; div sc={d_sc} lpc={d_ll} "
          f"gl={d_lg}) -----")
    print(f"  scalar {sc.mean():.4f}±{sc.std():.4f} | local-PC "
          f"{ll.mean():.4f}±{ll.std():.4f} (g{nl}) | global "
          f"{lg.mean():.4f}±{lg.std():.4f} (g{ng})")
    # ORIGINAL between-seed verdict — recomputed, REPRINTED UNCHANGED
    pooled = float(np.std(np.concatenate([ll, lg]))) if ns else float("nan")
    margin = float(sc.mean() - nb.mean())
    o_ties = ns > 0 and sc.mean() <= 1.25 * nb.mean() and margin <= pooled
    o_nest = ns > 0 and sc.mean() > nb.mean() + 0.5 * pooled
    o_v = "DEGENERATE" if o_ties else ("NESTING" if o_nest else "AMBIGUOUS")
    print(f"  ORIGINAL (between-seed, AS PREVIOUSLY REPORTED — UNCHANGED): "
          f"margin {margin:+.4f} vs pooled {pooled:.4f} -> {o_v}")
    # PAIRED (new pre-registered analysis, correct variance)
    d = sc - nb                                  # >0 => nested beats scalar
    m, s = float(d.mean()), float(d.std(ddof=1)) if ns > 1 else float("nan")
    sgn = int((d > 0).sum())
    t = m / (s / math.sqrt(ns)) if (ns > 1 and s > 0) else float("nan")
    nd = _three_way(m, s, sgn, ns)
    nd_txt = {"SEP": "NON-DEGENERATE (scalar clearly beaten)",
              "NOT": "DEGENERATE (scalar not beaten)",
              "AMB": "AMBIGUOUS"}[nd]
    print(f"  PAIRED non-degeneracy: Δ(scalar−nested) m={m:+.4f} "
          f"s_pair={s:.4f} t={t:+.2f} sign={sgn}/{ns} "
          f"(p={_sign_p(sgn, ns):.3f}) -> {nd_txt}")
    if nd != "SEP":
        print("  PAIRED head-to-head: GATED OFF (pre-registered: only if "
              "non-degenerate).")
        return label, o_v, nd, None
    dd = lg - ll                                 # >0 => local-PC better
    m2 = float(dd.mean())
    s2 = float(dd.std(ddof=1)) if ns > 1 else float("nan")
    sg2 = int((dd > 0).sum())
    hh = _three_way(abs(m2), s2, max(sg2, ns - sg2), ns)
    if hh == "SEP" and m2 > 0:
        v = "LOCAL-PC BETTER (quality, paired)"
    elif hh == "SEP" and m2 < 0:
        v = "GLOBAL BETTER (quality, paired)"
    else:
        dom = (nl < ng) and (d_ll <= d_lg)
        v = ("LOCAL-PC BETTER by dominance (paired quality tie; cheaper "
             f"g{nl}<g{ng} & {'≥' if d_ll <= d_lg else '<'}-stable)" if dom
             else "QUALITY TIE, no dominance")
    print(f"  PAIRED head-to-head: Δ(global−localPC) m={m2:+.4f} "
          f"s_pair={s2:.4f} sign(localPC win)={sg2}/{ns} -> {v}")
    return label, o_v, nd, v


def paired(n):
    """Run the pre-registered paired analysis on the three decisive configs:
    canonical P1, C-hetero, P3. Prints original (unchanged) + paired verdicts
    side by side. Honesty guardrail printed explicitly."""
    global D_IN, D_FEAT, N_TASKS, STREAM_STEPS, EVAL_BLOCK, HETERO, HET_STRENGTH
    snap = (D_IN, D_FEAT, N_TASKS, STREAM_STEPS, EVAL_BLOCK, HETERO,
            HET_STRENGTH)
    print("PAIRED ANALYSIS (§7.6.7) — pre-registered NEW analysis. The "
          "ORIGINAL between-seed verdicts are reprinted UNCHANGED; the paired "
          "verdict (correct variance for this common-random-number design) is "
          "shown alongside. We do NOT delete or overwrite the originals.")
    print("PRE-REGISTERED: P1-canon & C-scale stay NOT-separated even paired; "
          "C-hetero & P3 become SEPARATED paired; head-to-head stays a paired "
          "tie (local-PC better only by dominance). Reported either way.\n")
    rows = []
    t0 = time.perf_counter()

    print("== canonical P1 (orthonormal/shared-Σ, toy) ==")
    D_IN, D_FEAT, N_TASKS, STREAM_STEPS, EVAL_BLOCK, HETERO, HET_STRENGTH = \
        16, 24, 10, 120, None, False, 1.0
    reseed(0)
    rows.append(paired_report("canonical-P1", *paired_core(n)))

    print("\n== C-hetero (per-task Σ_t + non-orthogonal r_t, toy) ==")
    D_IN, D_FEAT, N_TASKS, STREAM_STEPS, EVAL_BLOCK, HETERO, HET_STRENGTH = \
        16, 24, 10, 120, None, True, 1.0
    reseed(0)
    rows.append(paired_report("C-hetero", *paired_core(n)))

    print("\n== P3 (cyclic reactivation, out-of-family) ==")
    D_IN, D_FEAT, N_TASKS, STREAM_STEPS, EVAL_BLOCK, HETERO, HET_STRENGTH = \
        16, 24, 3, 600, 30, False, 1.0
    reseed(0)
    rows.append(paired_report("P3-cyclic", *paired_core(n)))

    (D_IN, D_FEAT, N_TASKS, STREAM_STEPS, EVAL_BLOCK, HETERO,
     HET_STRENGTH) = snap
    print("\n" + "=" * 74)
    print("PAIRED SUMMARY (original UNCHANGED | paired)")
    for lab, ov, nd, hh in rows:
        print(f"  {lab:>14}: original={ov:<10}  paired={nd:<3}"
              f"{('  head2head=' + hh) if hh else ''}")
    print("-" * 74)
    canon_nd = rows[0][2]
    if canon_nd != "NOT":
        print("NOTE: canonical-P1 flipped to non-NOT under pairing — this "
              "WEAKENS the headline (the canonical benchmark would not be "
              "cleanly degenerate). Reported as pre-committed.")
    else:
        print("Canonical-P1 stays NOT-separated even paired -> the canonical "
              "degeneracy is genuine, not a between-seed variance artifact.")
    print(f"total wall time: {time.perf_counter() - t0:.1f}s")


def p1_scale(n):
    """C-scale (§7.6.1). SAME §4.3/Eq.45 construction, ~10x/5x/5x/5x larger.
    Pre-registered prediction: STAYS DEGENERATE (H-deg says degeneracy is
    task-geometry, not size). Same p1() decision rule, reused verbatim."""
    global D_IN, D_FEAT, N_TASKS, STREAM_STEPS
    D_IN, D_FEAT, N_TASKS, STREAM_STEPS = 64, 256, 50, 600
    reseed(0)
    print(f"P1-AT-SCALE: D_in={D_IN} D_feat={D_FEAT} tasks={N_TASKS} "
          f"stream={STREAM_STEPS} (same orthonormal/shared-Σ geometry)")
    print("PRE-REGISTERED PREDICTION: stays DEGENERATE\n")
    p1(n)


def p2(n):
    """C-hetero (§7.6.1) — the decisive control. Break BOTH degeneracy
    sources at toy size (isolates geometry from scale): per-task input
    covariance -> heterogeneous Σ_t, AND non-orthogonal correlated r_t.
    Pre-registered prediction: NESTING MATTERS (single scalar separates).
    Same p1() decision rule, reused verbatim."""
    global HETERO
    HETERO = True
    reseed(0)
    print(f"P2 (C-hetero): theta_dim={theta_dim()} tasks={N_TASKS} "
          f"stream={STREAM_STEPS}; per-task Σ_t + non-orthogonal r_t")
    print("PRE-REGISTERED PREDICTION: NESTING MATTERS (scalar separates)\n")
    p1(n)


def p2_scale(n):
    """Both controls combined: heterogeneous geometry AND large scale."""
    global D_IN, D_FEAT, N_TASKS, STREAM_STEPS, HETERO
    D_IN, D_FEAT, N_TASKS, STREAM_STEPS = 64, 256, 50, 600
    HETERO = True
    reseed(0)
    print(f"P2-AT-SCALE: D_in={D_IN} D_feat={D_FEAT} tasks={N_TASKS} "
          f"stream={STREAM_STEPS}; heterogeneous Σ_t + non-orthogonal r_t")
    print("PRE-REGISTERED PREDICTION: NESTING MATTERS (scalar separates)\n")
    p1(n)


def lambda_sweep(n):
    """Theory test 1 (§7, F3/C2): local-PC's combiner target is the HARD
    projection (the λ→∞ limit), so it is λ-independent; global's retention
    weight is λ. Prediction: |global − local-PC| decreases in λ, consistent
    with Θ(1/λ). Linear regime (proposition is exact there). K=8."""
    global LAM_RETAIN
    K = KS[-1]
    grid = [0.25, 1.0, 4.0, 16.0, 64.0]
    print(f"lambda-sweep (linear, K={K}, seeds={n}); local-PC is "
          f"λ-independent (hard-projection target = λ→∞ limit)")
    gap_by_l = {L: [] for L in grid}
    for s in range(n):
        reseed(s)
        phi_l, _, _ = meta_train_localpc(K, ablate=None)
        Ll, _ = run_stream(phi_l, K)
        for L in grid:
            LAM_RETAIN = L
            reseed(s)
            phi_g, _, _ = meta_train_global(K)
            Lg, _ = run_stream(phi_g, K)
            gap_by_l[L].append(abs(Lg - Ll))
    LAM_RETAIN = 1.0
    A = np.array
    print(f"{'lambda':>8} {'|global-localPC|':>18} {'gap*lambda':>12}")
    gm = {}
    for L in grid:
        g = A(gap_by_l[L]).mean()
        gm[L] = g
        print(f"{L:>8.2f} {g:>18.4f} {g * L:>12.4f}")
    ls = A([math.log(L) for L in grid])
    lg = A([math.log(max(gm[L], 1e-6)) for L in grid])
    slope = np.polyfit(ls, lg, 1)[0]
    mono = all(gm[grid[i + 1]] <= gm[grid[i]] + 1e-4
               for i in range(len(grid) - 1))
    print("-" * 70)
    print(f"log-log slope d(log gap)/d(log λ) = {slope:+.2f}  "
          f"(Θ(1/λ) => slope ≈ -1; monotone-decreasing={mono})")
    ok = mono and slope < -0.5
    print("LAMBDA VERDICT:",
          "gap shrinks with λ, consistent with the Θ(1/λ) prediction (F3/C2 "
          "supported)" if ok else "gap does NOT follow Θ(1/λ) — theory's "
          "finite-λ bias claim is NOT supported (falsified at toy scale)")


def lambda_sweep_fair(n):
    """Confound-controlled re-test of F3: at fixed meta-LR global DIVERGES for
    large λ (never reaches the hard-projection asymptote). Here global gets the
    best of an LR grid PER λ (consistent with the fairness control we already
    adopted). One shot; report whatever it shows."""
    global LAM_RETAIN
    K = KS[-1]
    grid = [0.25, 1.0, 4.0, 16.0, 64.0]
    LRS = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    print(f"lambda-sweep FAIR (global = best-of-LR per λ; linear K={K}, "
          f"seeds={n})")
    gap_by_l = {L: [] for L in grid}
    for s in range(n):
        reseed(s)
        phi_l, _, _ = meta_train_localpc(K, ablate=None)
        Ll, _ = run_stream(phi_l, K)
        for L in grid:
            LAM_RETAIN = L
            best = 1e18
            for lr in LRS:
                reseed(s)
                phi_g, _, _ = meta_train_global(K, lr=lr)
                Lg, _ = run_stream(phi_g, K)
                if np.isfinite(Lg):
                    best = min(best, abs(Lg - Ll))
            gap_by_l[L].append(best)
    LAM_RETAIN = 1.0
    A = np.array
    print(f"{'lambda':>8} {'min|global-localPC|':>20} {'gap*lambda':>12}")
    gm = {}
    for L in grid:
        g = A(gap_by_l[L]).mean()
        gm[L] = g
        print(f"{L:>8.2f} {g:>20.4f} {g * L:>12.4f}")
    ls = A([math.log(L) for L in grid])
    lg = A([math.log(max(gm[L], 1e-6)) for L in grid])
    slope = np.polyfit(ls, lg, 1)[0]
    mono = all(gm[grid[i + 1]] <= gm[grid[i]] + 1e-4
               for i in range(len(grid) - 1))
    print("-" * 70)
    print(f"log-log slope = {slope:+.2f}  (Θ(1/λ)=> ≈ -1; monotone={mono})")
    print("LAMBDA-FAIR VERDICT:",
          "Θ(1/λ) supported once global's meta-LR is controlled"
          if (mono and slope < -0.5) else
          "Θ(1/λ) still NOT supported even with fair global tuning — F3 "
          "falsified robustly")


def interleave_sweep(n):
    """Theory test 2 (§7, F4): as task presentation goes block-sequential ->
    i.i.d. the timescale separation vanishes; prediction = the nested memory's
    advantage over flat (and local-PC↔global parity) DEGRADES. λ=1, linear,
    K=8."""
    K = KS[-1]
    blocks = [120, 40, 12, 4, 1]   # ncyc = 1,3,10,30,120 ; i.i.d. at block=1
    print(f"interleave-sweep (linear, λ=1, K={K}, seeds={n}); "
          f"block=120 sequential -> block=1 i.i.d.")
    adv, par = {b: [] for b in blocks}, {b: [] for b in blocks}
    for s in range(n):
        reseed(s)
        phi_l, _, _ = meta_train_localpc(K, ablate=None)
        reseed(s)
        phi_g, _, _ = meta_train_global(K)
        for b in blocks:
            ff, _ = run_stream(None, K, block=b)
            Ll, _ = run_stream(phi_l, K, block=b)
            Lg, _ = run_stream(phi_g, K, block=b)
            adv[b].append(ff - Ll)            # local-PC gain over flat
            par[b].append(abs(Lg - Ll))       # local-PC↔global parity gap
    A = np.array
    print(f"{'block':>6} {'advantage(flat-lpc)':>20} {'parity|g-l|':>13}")
    for b in blocks:
        print(f"{b:>6} {A(adv[b]).mean():>20.4f} {A(par[b]).mean():>13.4f}")
    a_seq, a_iid = A(adv[120]).mean(), A(adv[1]).mean()
    degr = a_iid < 0.5 * a_seq and a_seq > 1e-3
    print("-" * 70)
    print(f"advantage: sequential={a_seq:+.4f} -> i.i.d.={a_iid:+.4f}")
    print("F4 VERDICT:",
          "nested-memory advantage collapses toward i.i.d. as predicted "
          "(timescale-separation mechanism supported)" if degr else
          "advantage does NOT collapse toward i.i.d. — F4 (and the "
          "timescale-separation mechanism) NOT supported")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--p1":
        p1(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif len(sys.argv) > 1 and sys.argv[1] == "--p1-scale":
        p1_scale(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif len(sys.argv) > 1 and sys.argv[1] == "--p2":
        p2(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif len(sys.argv) > 1 and sys.argv[1] == "--p2-scale":
        p2_scale(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif len(sys.argv) > 1 and sys.argv[1] == "--p2-strength":
        p2_strength(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif len(sys.argv) > 1 and sys.argv[1] == "--p3":
        p3(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif len(sys.argv) > 1 and sys.argv[1] == "--paired":
        paired(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif len(sys.argv) > 1 and sys.argv[1] == "--lambda":
        lambda_sweep(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif len(sys.argv) > 1 and sys.argv[1] == "--lambda-fair":
        lambda_sweep_fair(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif len(sys.argv) > 1 and sys.argv[1] == "--interleave":
        interleave_sweep(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif len(sys.argv) > 1 and sys.argv[1] == "--deep":
        deep(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif len(sys.argv) > 1 and sys.argv[1] == "--scale":
        scale_to_failure(float(sys.argv[2]) if len(sys.argv) > 2 else 25.0)
    elif len(sys.argv) > 1 and sys.argv[1] == "--seeds":
        multiseed(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif len(sys.argv) > 1 and sys.argv[1] == "--ablation":
        ablation(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif len(sys.argv) > 1 and sys.argv[1] == "--nonlinear":
        nonlinear(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    else:
        t0 = time.perf_counter()
        main()
        print(f"total wall time: {time.perf_counter() - t0:.1f}s")
