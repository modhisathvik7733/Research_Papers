# 0007 — A continual benchmark that certifies its own non-degeneracy

**Status:** PRE-REGISTERED + runnable first cut. Registration fixed here
before results. Idea:
[ideas/nl-nondegenerate-benchmark.md](../../ideas/nl-nondegenerate-benchmark.md).

## What this is

A synthetic continual task family with three knobs that control degeneracy
(the levers exp-0002→0005 identified), shipping with a **mandatory
degeneracy certificate** — a pre-registered control that declares an instance
ADMISSIBLE or REJECTED *before* any method comparison on it is allowed.

## Task family (exp-0002 lineage, so §4.3/Eq.45 reproduces exactly)

Shared/per-task random features, linear readout θ. Three knobs:
- **ortho ∈ [0,1]**: task directions from orthonormal (1.0) → random
  correlated (0.0).
- **het ≥ 0**: per-task input-anisotropy exponent (0 = shared Hessian;
  >0 = heterogeneous Hessians; the C-stronger knob).
- **gap**: cyclic-reactivation block size (None = single-pass sequential;
  small = P3-style recurrence with an inter-block gap).
§4.3/Eq.45 = (ortho 1.0, het 0, gap None).

## The certificate (pre-registered, NL-neutral)

**Scoping (design-time correction, logged before any run):** this is the
*optimizer-axis* certificate (the axis P1/NL live on). Degeneracy is
axis-specific — replay exploits the *data* axis and would wrongly "admit"
§4.3 (rehearsal fixes orthonormal forgetting), so replay is **not** the
reference. NL-neutral reference = the **establishment strong optimizer: the
global hypergradient (what HOPE uses), NOT our Local-PC** (so the certificate
cannot favour our method). Reproduces P1 exactly.

Trivial baseline set (must be defeated): `sgd` (β=0), `mom*` (best-grid
single-β momentum — the P1 scalar), `emaK` (fixed multi-β EMA — structure-
free multi-timescale, so multi-timescale-ness *alone* cannot pass).
Non-trivial reference: **`global`** (global-hypergradient nested optimizer).

Paired CRN over n seeds, divergence guard. Instance **ADMISSIBLE** iff:
1. **Discrimination:** `best_trivial − global` (final loss, lower better)
   passes the verbatim exp-0002 three-way **SEP** with the effect-size gate
   (`m > s_pair` AND sign ≥ ⌈0.8n⌉), AND
2. **Practical effect:** mean gap ≥ pre-registered δ = 0.02 (loss units) —
   no certifying on a negligible-but-consistent margin (exp-0005 lesson),
   AND
3. **Power:** the gap clears seed variance at the certified n
   (m > s_pair already enforces this; logged explicitly).
Else **REJECTED**, reason = `degenerate` (gap≈0) or `underpowered`
(gap>0 but m≤s_pair). Exact sign-test p reported as corroboration only.

## Pre-registered predictions

- **C-1 (sanity, must hold):** §4.3/Eq.45 instance → **REJECTED/degenerate**
  (reproduces P1). If it does not, the harness is broken — stop.
- **C-2:** P3-lineage instance (gap small, het>0, ortho<1) → **ADMITTED**.
- **C-3:** knob sweep flips REJECT→ADMIT at a locatable boundary (the
  operating manual).
- **Honest risk:** no instance passes all clauses at feasible n ⇒ report
  "toy-scale CL evaluation underpowered," straight.

n=10 certify; n=60 pre-committed for any instance whose certificate is
AMBIGUOUS at n=10.

## Escalation

1. `--smoke` — pipeline + C-1 sanity (§4.3 must REJECT).
2. `--certify` — the two named instances (§4.3 reject, P3-lineage admit).
3. `--sweep` — the REJECT→ADMIT boundary over the three knobs.
4. n=60 confirmation on any AMBIGUOUS certificate.

## Honest caveats baked in

Synthetic (the certificate *logic* is the deliverable). δ and certified-n
are pre-registered, reported with the boundary, never tuned post hoc.
"Reference = replay" certifies *discriminability*, not universal
informativeness — stated, not hidden. A method result on a REJECTED instance
is inadmissible by construction; that is the entire point.
