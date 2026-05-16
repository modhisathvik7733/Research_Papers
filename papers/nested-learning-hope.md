# Nested Learning (NL) + HOPE

- **Authors / venue / year:** Behrouz et al. (Google), 2025
- **Link:** (NeurIPS 2025 "It's All Connected" / Nested Learning line of work — Titans / Atlas / NL)
- **Read on:** 2026-05-16
- **Pass reached:** 1 (intuition + Sections 2–3 read closely; later sections pending)
- **Status:** building-on

> Living notes. Section A = intuition layer (plain English). Section B = formal layer
> (the math from Preliminaries §2 and Nested Learning §3). More info to be added.

---

## In one sentence

A neural network is not "architecture + an external optimizer" — it is a single
**nested system of associative memories**, where every component (attention, MLP,
the optimizer itself, even backprop) is a memory that compresses some *context flow*
by solving its own optimization problem, and these problems are stacked by **update
frequency** the way the brain stacks learning across brainwave timescales.

---

# A. Intuition Layer

## A1. The core problem
Current LLMs: **train once → freeze forever**.
- Can adapt *temporarily* in-context, cannot *permanently* learn after deployment.
- Failure modes: catastrophic forgetting, static weights, weak continual learning,
  no lifelong memory.

## A2. Paradigm shift
| Old view | NL view |
|---|---|
| Architecture + external optimizer (separate) | Architecture **and** optimizer are *both* learning systems |
| One big optimization | Many nested optimizations, each at its own speed |
| Optimizer = dumb update rule | Optimizer = an associative memory (it stores gradient history) |

Russian-doll picture: a learning system contains smaller learning systems, each with
its own memory, gradients, objective, and **update frequency**.

## A3. Key reframings
- **Optimizers are memory systems.** Momentum/Adam secretly store past gradients,
  variance, directions → they are *associative memory*.
- **Associative memory** = mapping keys → values by relationship (cat → whiskers).
  Transformers, optimizers, and explicit memory modules are *all* this same object.
- **Training = compression.** Pretraining compresses the internet into weights;
  in-context learning compresses the prompt into attention state. Same operation,
  different *duration / scale / update speed*. (Hence the "pretraining IS in-context
  learning" claim.)

## A4. Brain inspiration → frequencies
Brain learns at **many speeds simultaneously**: Gamma (perception, fastest) → Beta
(active thinking) → Theta (learning) → Delta (memory consolidation, slowest).

Transformers only have **two** frequencies:
- **Attention ≈ infinite frequency** (updates every token, temporary).
- **MLP weights ≈ zero frequency** (frozen at inference, no online learning).

That two-speed limitation is the thing NL/HOPE attacks.

## A5. The proposed direction
- **Continuum Memory System (CMS):** a spectrum of memories with different update
  rates (attention/per-token → episodic/minutes → session/hours → semantic/months →
  identity/years) instead of just short- vs long-term.
- **HOPE:** a self-modifying sequence model — the model learns, the optimizer learns,
  the memory-update rule learns; learning loops are nested.
- **Delta Gradient Descent (DGD):** an update that depends not only on the current
  gradient but on current weights + memory state + prior dependencies (relaxes the
  i.i.d. assumption that real markets/conversations/games violate).
- Borrows from how brains beat LLMs: online consolidation, offline replay, multi-
  timescale stabilization.

## A6. One-line takeaway
**Intelligence may come less from stacking layers and more from stacking adaptive
learning processes.** Everything reduces to *memory compression + nested optimization*.

---

# B. Formal Layer (Preliminaries §2 + Nested Learning §3)

## B1. Notation
- Input $x \in \mathbb{R}^{N\times d_{in}}$; $\mathcal{M}_t$ = state of memory/model at time $t$.
- $K$ keys, $V$ values, $Q$ queries; per-token vectors $k_t, v_t, q_t$.
- $p(\mathcal{T})$ = distribution of random variable $\mathcal{T}$.
- Memory module $\mathcal{M}(\cdot)$ = MLP with $\mathcal{L_M}\ge 1$ layers + residual,
  params $\theta_\mathcal{M}\supseteq\{W_1,\dots,W_{\mathcal{L_M}}\}$.
- **Level superscript** $W^{(\ell)}$ / frequency $W^{(f_\ell)}$ marks which *level*
  (update frequency) a parameter lives at.

## B2. Gradient descent, three equivalent forms
1. **SGD:** $W_{t+1}=W_t-\eta_t\nabla_{W_t}\mathcal{L}(W_t;x_t)$ — (1)
2. **Steepest descent / proximal:**
   $W_{t+1}=\arg\min_W \{\langle\nabla_W\mathcal{L},W\rangle+\frac{1}{2\eta_t}\|W-W_t\|_2^2\}$ — (2)
   → GD has an *implicit bias toward small $L_2$ moves*.
3. **FTRL (accumulated):** $\arg\min_W\{\langle\sum_{s=1}^t\nabla\mathcal{L},W\rangle+\frac{1}{2\eta}\|W-W_1\|_2^2\}$ — (3)

Used interchangeably; the proximal view is what lets *training itself* be written as
an associative-memory problem.

## B3. Background objects
- **Meta-learning:** two-level optimization; outer param $\Phi$ set over a task
  distribution: $\Phi^*=\arg\min_\Phi \mathbb{E}_{\mathcal{T}_i\sim p(\mathcal{T})}[\ell(\theta,\mathcal{T}_i;\Phi)]$ — (4)
- **Fast Weight Programmers / linear Transformers:** matrix-valued state
  $\mathcal{M}_t\in\mathbb{R}^{d_{out}\times d_{key}}$, written by rank-one update, read
  by matrix–vector product:
  $\mathcal{M}_t=\alpha_t\mathcal{M}_{t-1}+v_t\phi(k_t)^\top$, $\;y_t=\mathcal{M}_t\phi(q_t)$ — (5)
- **In-context learning (general def):** any model adapting to / learning from its
  context — not Transformer-specific. NL ties ICL to associative memory.

## B4. Associative memory (Definition 1)
Given keys $\mathcal{K}\subseteq\mathbb{R}^{d_k}$, values $\mathcal{V}\subseteq\mathbb{R}^{d_v}$,
AM is an operator $\mathcal{M}(\cdot)$ mapping $\mathcal{K}\to\mathcal{V}$, learned by:
$$\mathcal{M}^*=\arg\min_\mathcal{M}\ \tilde{\mathcal{L}}(\mathcal{M}(\mathcal{K});\mathcal{V}) \quad (6)$$
- Neuropsych distinction the paper adopts (Okano 2000): **"Memory is a neural update
  caused by an input; learning is the process of acquiring effective, useful memory."**
- Keys/values need not be tokens — can be gradients, sub-sequences, etc.
- Equivalent to data compression: $\mathcal{M}$ compresses the mapping into a
  lower-dimensional parameter space.

## B5. Worked decompositions (the heart of §3)
**1-layer MLP trained by SGD** is a 1-level AM. The update
$W_{t+1}=W_t-\eta_{t+1}\nabla_W\mathcal{L}=W_t-\eta_{t+1}\,u_{t+1}\otimes x_{t+1}$ — (8)
with **"surprise"** $\nabla_W\mathcal{L}$ and **local surprise signal (LSS)**
$u_{t+1}=\nabla_{y_{t+1}}\mathcal{L}$. Rewriting as a proximal problem (9) shows:

> A linear layer trained with backprop = an associative memory mapping each data
> sample to the **error of its prediction (its LSS)**. Backprop *is* a surprise-based
> memory. Single gradient flow, frozen at test time.

**GD with momentum** (10)-(13): the momentum term $m$ is itself an AM that compresses
past gradients. ⇒ momentum-SGD is a **2-level** optimization (inner: GD on the
momentum memory; outer: weight update).

**Linear attention** (14)-(18): recurrence $\mathcal{M}_t=\mathcal{M}_{t-1}+v_tk_t^\top$
is GD on a matrix-valued AM with $\tilde{\mathcal{L}}=-\langle\mathcal{M}_{t-1}k_t,v_t\rangle$.
⇒ Training linear attention is 2-level: outer optimizes projections $W_k,W_v,W_q$;
inner optimizes the memory $\mathcal{M}_t$. (Matches the FWP slow/fast-weight view.)

## B6. Update frequency, levels, nested systems
- **Definition 2 (Update Frequency):** $f_A$ = #updates of component $A$ per unit
  time (one update on one datapoint = unit of time). Parametric *or* non-parametric
  components (attention block counts too).
- Order: $A\succ B$ ("$A$ faster") if $f_A>f_B$, or equal freq but $B$'s state
  computation needs $A$'s. Equal & independent ⇒ $A\stackrel{f}{=}B$ (e.g. AdamW).
- Sort components into ordered **levels**: same level = same frequency; higher
  level = lower frequency.

**Definition 3 (Nested System):** $K$ ordered levels, each level $k$ a set of
optimization problems $\{(\mathcal{L}_i^{(k)},C_i^{(k)},\Theta_i^{(k)})\}$ (objective,
context = data optimized on, feasible params), each solved by the proximal-GD form (19).
Each problem = its own "box of gradient flow."

**Definition 4 (NSAM):** a nested system where every box is an *associative memory*
over key–value pairs $C_i=\{(k_j,v_j)\}$ — Eq. (20). Modern architectures + common
optimizers are instances of NSAM. Stacking levels adds a **new depth dimension**
(higher-order ICL, latent computation à la Loop Transformers, multi-memory systems,
more expressive optimizers).

**MLP layer vs. linear attention (AdaTransformer):** identical block math; only
difference is the *level* of the last weight. $W_{MLP}$ = level 1 (persistent,
context-frozen). $W_{LinAttn}$ = level 2, updated in-context by
$\mathcal{M}_t=\mathcal{M}_{t-1}+v_tk_t^\top$; its init $\mathcal{M}_0$ is
meta-learned at level 1. ⇒ "expressive softmax attention + efficient recurrent model"
hybrids are misleading framing — it's just a Transformer with **ICL-capable MLP blocks**.

> **Stacking levels in NL:** models with multiple levels learn & process data at
> different *abstraction levels and update frequencies* — the new axis vs. just depth.

**Neural Learning Module:** architecture + optimization process jointly = one model;
its outputs depend on the whole interconnected system, not isolated parts. Matters
most in the *no-train/test-phase* continual setting NL advocates.

## B7. Knowledge transfer between levels (§3.3)
Two blocks $\mathcal{B}^{(0)}$ (high-freq) and $\mathcal{B}^{(1)}$ (low-freq):
- **Direct, parametric:** low-freq forward pass conditioned on high-freq params,
  $\mathcal{M}^{(0)}(\cdot):=\mathcal{M}^{(0)}(\cdot;\Theta^{(1)})$ — (24); or on its
  output (25). Linear-Transformer read $y_t=\mathcal{M}_t[x_tW_q]$ is exactly this — (26).
- **Direct, non-parametric:** condition on context/output of a non-parametric solve
  (27) — e.g. softmax attention. *No backprop between the two levels; each is a
  hyperparameter for the other.*
- **Via backpropagation:** both states in one gradient flow but updated at different
  frequencies (the CMS design, Section 7).
- **Via initialization:** MAML — higher level learns the best init for the inner
  loop, $\Theta_0^{(1)}=\arg\min_\Phi\mathbb{E}_{C}[\ell(\mathcal{M}^{(1)}(\cdot;\Phi),C)]$ — (28).
- **Via generation:** one block generates the other's weights (Hypernetworks) or
  context (optimizer: architecture generates the gradients that feed the optimizer).

> **Designing a Neural Learning Module = two choices:** (1) the optimization
> problems and their frequencies (the NSAM design); (2) the knowledge-transfer
> scheme between levels. Meta-learning / MAML / hypernetworks all fall out as
> special cases of choice (2).

---

## Where it still fails / open questions (to expand)
- Cost honesty: nested/multi-timescale updates at inference vs. compute budget —
  is the continual gain worth it? (ties to docs/prior_art.md gap analysis)
- Stability of self-modifying loops (HOPE) — does the optimizer-of-the-optimizer
  converge, or drift?
- Does CMS actually beat plain replay/EWC on the forgetting benchmark (cf.
  experiments/0001-reproduce-forgetting)?

## Questions this raises for me
1. Can the momentum-as-AM view be turned into a measurable forgetting metric on
   experiment 0001?
2. Minimal HOPE: smallest model where a learned inner update rule beats Adam on a
   non-i.i.d. stream?
3. Is "pretraining = ICL" testable as: does scaling context-compression match
   weight-compression on a held-out continual task?

<!-- Section C reserved for additional info the user will provide. -->
