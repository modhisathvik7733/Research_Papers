"""0003 — Plasticity loss vs catastrophic forgetting (cost-asymmetry).

Pre-registered design: see README.md. Self-contained, CPU, deterministic.
Discipline carried from exp-0002: paired common-random-number stats, the
verbatim three-way decision rule, a divergence guard, and a non-degeneracy
sanity that MUST pass before any H-diss interpretation.

Entry points:
  python3 run.py --sanity [n]      non-degeneracy guard (plasticity slope)
  python3 run.py --seeds  [n]      paired double dissociation (H-diss)
  python3 run.py --capacity [n]    the P->F capacity flip (H-cap)
This is a runnable FIRST CUT; iterate the arms/metrics, not the rule.
"""
import math
import sys
import numpy as np
import torch

torch.set_num_threads(4)

# ----------------------------------------------------------- construction ----
D_IN, D_FEAT = 16, 32
H_WID = 16                 # student hidden width = the capacity knob (H-cap)
TEACH_HID = 12
T_TASKS = 8
BLOCK = 80                 # steps per task block
BATCH = 64
LR = 3e-3
REPLAY_FRAC = 0.25         # O(buffer) arm: replay batch fraction
SP_ALPHA, SP_SIGMA = 0.8, 0.01   # O(1) arm: shrink-and-perturb at boundaries
_F = None


def reseed(s):
    global _F
    torch.manual_seed(s)
    np.random.seed(s)
    g = torch.Generator().manual_seed(7 + s)
    _F = torch.randn(D_FEAT, D_IN, generator=g) / math.sqrt(D_IN)


def feat(x):
    return torch.tanh(x @ _F.t())


def make_teacher(seed):
    g = torch.Generator().manual_seed(3000 + seed)
    V = torch.randn(TEACH_HID, D_FEAT, generator=g) / math.sqrt(D_FEAT)
    u = torch.randn(TEACH_HID, generator=g) / math.sqrt(TEACH_HID)

    def sample(n):
        x = torch.randn(n, D_IN)
        with torch.no_grad():
            y = torch.relu(feat(x) @ V.t()) @ u
        return x, y

    return sample


def new_student(width=None):
    w = width or H_WID
    W1 = (torch.randn(w, D_FEAT) / math.sqrt(D_FEAT)).requires_grad_(True)
    w2 = (torch.randn(w) / math.sqrt(w)).requires_grad_(True)
    return [W1, w2]


def fwd(stu, x):
    W1, w2 = stu
    return torch.relu(feat(x) @ W1.t()) @ w2


def loss_on(stu, task, n=512):
    with torch.no_grad():
        x, y = task(n)
        return float(((fwd(stu, x) - y) ** 2).mean())


def train_block(stu, task, B, buf=None, width=None):
    opt = torch.optim.Adam(stu, lr=LR)
    for _ in range(B):
        x, y = task(BATCH)
        if buf:                                 # O(buffer) replay arm
            k = max(1, int(REPLAY_FRAC * BATCH))
            xb, yb = buf.sample(k)
            if xb is not None:
                x, y = torch.cat([x, xb]), torch.cat([y, yb])
        loss = ((fwd(stu, x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return loss_on(stu, task)


class Buf:
    def __init__(self, cap=512):
        self.cap, self.x, self.y = cap, None, None

    def add(self, task):
        x, y = task(128)
        self.x = x if self.x is None else torch.cat([self.x, x])[-self.cap:]
        self.y = y if self.y is None else torch.cat([self.y, y])[-self.cap:]

    def sample(self, k):
        if self.x is None:
            return None, None
        i = torch.randint(0, len(self.x), (k,))
        return self.x[i], self.y[i]


def shrink_perturb(stu):
    with torch.no_grad():
        for p in stu:
            p.mul_(SP_ALPHA).add_(SP_SIGMA * torch.randn_like(p))


# ------------------------------------------------- continual run + metrics ---
def run_arm(seed, arm, width=None, want_Pt=False):
    """Returns (F, P) for one arm. P uses a per-task FRESH-net reference
    (arm-independent, computed once per seed via the 'fresh' pseudo-arm)."""
    reseed(seed)
    tasks = [make_teacher(seed * 100 + t) for t in range(T_TASKS)]
    stu = new_student(width)
    buf = Buf() if arm == "replay" else None
    after = [None] * T_TASKS
    cont_tr = [None] * T_TASKS
    for t, tk in enumerate(tasks):
        if arm == "plast" and t > 0:
            shrink_perturb(stu)
        cont_tr[t] = train_block(stu, tk, BLOCK, buf, width)
        after[t] = loss_on(stu, tk)
        if buf is not None:
            buf.add(tk)
    final = [loss_on(stu, tk) for tk in tasks]
    F = float(np.mean([final[i] - after[i] for i in range(T_TASKS - 1)]))
    # fresh-net reference (arm-independent): freshly init per task, same budget
    fresh_tr = []
    for t, tk in enumerate(tasks):
        fs = new_student(width)
        fresh_tr.append(train_block(fs, tk, BLOCK, None, width))
    Pt = [cont_tr[t] - fresh_tr[t] for t in range(T_TASKS)]
    P = float(np.mean(Pt))
    return (F, P, Pt) if want_Pt else (F, P)


# ----------------------------------------------------- paired-CRN stats ------
def _sign_p(k, m):
    if m == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(m, i)
                              for i in range(k, m + 1)) / (2.0 ** m))


def three_way(m, s, sign, ns):
    if ns == 0 or s <= 0:
        return "NOT"
    if m > s and sign >= math.ceil(0.8 * ns):
        return "SEP"
    if m <= 0.5 * s or sign < math.ceil(0.6 * ns):
        return "NOT"
    return "AMB"


def paired_verdict(name, d):
    """d>0 means the intervention REDUCED the metric (plain − arm)."""
    d = np.array([v for v in d if np.isfinite(v) and abs(v) < 1e3])
    ns = len(d)
    m = float(d.mean()) if ns else float("nan")
    s = float(d.std(ddof=1)) if ns > 1 else float("nan")
    sgn = int((d > 0).sum())
    t = m / (s / math.sqrt(ns)) if (ns > 1 and s > 0) else float("nan")
    v = three_way(m, s, sgn, ns)
    print(f"  {name:<22} m={m:+.4f} s_pair={s:.4f} t={t:+.2f} "
          f"sign={sgn}/{ns} p={_sign_p(sgn, ns):.3f} -> {v}")
    return v


# ------------------------------------------------------------- entrypoints ---
def sanity(n):
    """Non-degeneracy guard: plain-model per-task plasticity slope > 0."""
    print(f"SANITY (non-degeneracy): plain P_t must rise in t. seeds={n}")
    slopes = []
    for s in range(n):
        _, _, Pt = run_arm(s, "plain", want_Pt=True)
        ts = np.arange(T_TASKS)
        slopes.append(np.polyfit(ts, Pt, 1)[0])
    sl = np.array(slopes)
    m, sd = sl.mean(), sl.std(ddof=1)
    t = m / (sd / math.sqrt(n))
    print(f"  P_t slope: m={m:+.5f} ± {sd:.5f}  t={t:+.2f}  "
          f"sign={(sl > 0).sum()}/{n}")
    ok = m > 0 and t > 2.0 and (sl > 0).sum() >= math.ceil(0.8 * n)
    print("SANITY VERDICT:", "PASS — plasticity loss exists; H-diss "
          "interpretable." if ok else "FAIL — construction degenerate for "
          "this question (no plasticity loss). Report straight; do NOT "
          "interpret H-diss (the P1 lesson).")


def seeds(n):
    """H-diss double dissociation, paired CRN."""
    print(f"H-diss double dissociation (paired CRN, seeds={n}, "
          f"width={H_WID})\nPRE-REGISTERED: plast↓P & not-↓F ; replay↓F & "
          f"not-↓P ; claim holds iff ALL FOUR.\n")
    dPp, dFp, dPr, dFr = [], [], [], []
    for s in range(n):
        Fpl, Ppl = run_arm(s, "plain")
        Fpa, Ppa = run_arm(s, "plast")
        Frp, Prp = run_arm(s, "replay")
        dPp.append(Ppl - Ppa); dFp.append(Fpl - Fpa)
        dPr.append(Ppl - Prp); dFr.append(Fpl - Frp)
        print(f"seed {s}: plain(F={Fpl:.3f},P={Ppl:.3f}) "
              f"plast(F={Fpa:.3f},P={Ppa:.3f}) "
              f"replay(F={Frp:.3f},P={Prp:.3f})")
    print("-" * 70)
    a = paired_verdict("plast: ΔP (want SEP)", dPp)
    b = paired_verdict("plast: ΔF (want NOT)", dFp)
    c = paired_verdict("replay: ΔF (want SEP)", dFr)
    e = paired_verdict("replay: ΔP (want NOT)", dPr)
    print("-" * 70)
    clean = (a == "SEP" and b == "NOT" and c == "SEP" and e == "NOT")
    print("H-DISS VERDICT:", "CLEAN DOUBLE DISSOCIATION — the O(1) fix and "
          "the O(buffer) fix attack different failures." if clean else
          "NOT a clean cross (need plast:SEP/NOT and replay:SEP/NOT). "
          "Report straight as partial/negative; do not spin.")


def capacity(n):
    """H-cap: does P-dominance (small width) flip to F-dominance (large)?"""
    print(f"H-cap capacity flip (plain arm, seeds={n})")
    print(f"{'width':>6} {'F':>8} {'P':>8} {'dominant':>10}")
    global H_WID
    snap = H_WID
    for w in [4, 8, 16, 32, 64]:
        H_WID = w
        FP = np.array([run_arm(s, "plain", width=w)[:2] for s in range(n)])
        F, P = FP[:, 0].mean(), FP[:, 1].mean()
        print(f"{w:>6} {F:>8.4f} {P:>8.4f} {'P' if P > F else 'F':>10}")
    H_WID = snap
    print("PRE-REGISTERED: P-dominant at small width -> F-dominant at large; "
          "report the flip width or 'no flip' straight.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--sanity"
    nn = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if cmd == "--sanity":
        sanity(nn)
    elif cmd == "--seeds":
        seeds(nn)
    elif cmd == "--capacity":
        capacity(nn)
    else:
        print(__doc__)
