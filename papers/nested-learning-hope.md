# Nested Learning (NL) + HOPE

- **Authors / venue / year:** Behrouz et al. (Google), 2025
- **Link:** (NeurIPS 2025 "It's All Connected" / Nested Learning line of work — Titans / Atlas / NL)
- **Read on:** 2026-05-16
- **Pass reached:** 2 (full paper §1–§10 read closely)
- **Status:** building-on

> Notes map: **A** intuition · **B** Preliminaries+NL (§2–3) · **C** optimizers (§4) ·
> **D** architectures as NSAM (§5) · **E** revisited terms (§6) · **F** CMS (§7) ·
> **G** HOPE module (§8) · **H** experiments (§9) · **I** conclusion/limits (§10).

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

---

# C. Optimizers as Learning Modules (§4)

Thesis of the section: backprop, momentum, Adam, AdaGrad, Shampoo/Soap, Muon —
**all** are associative memories that compress gradients, and each decomposes into
nested GD problems. NL then prescribes how to make them *more expressive*.

## C1. Backpropagation as an associative memory (§4.1)
For an $L$-layer MLP, the per-layer gradient is
$\frac{\partial\mathcal{L}}{\partial W_\ell}=\delta_\ell\,\hat{x}_{\ell-1}^\top$,
with **local output surprise** $\delta_\ell=J_{\phi_\ell}(z_\ell)^\top(W_{\ell+1}^\top\delta_{\ell+1})$ — (29),
$z_\ell=W_\ell\hat{x}_{\ell-1}+b_\ell$. The GD step (30) rewrites as the proximal /
AM problem (31):
$$W_{\ell_{t+1}}=\arg\min_W\ \langle W\hat{x}_{\ell-1},\delta_\ell\rangle+\tfrac{1}{2\eta}\|W-W_{\ell_t}\|_F^2$$
⇒ **each layer is an AM mapping its input $\hat{x}_{\ell-1}$ → its local error $\delta_\ell$**;
training = compression of (input, local-error) pairs.

> Box: a deep net trained with backprop *learns by memorizing how surprising its
> outputs are*; backprop maps each datapoint → its prediction error.

**Backprop ≠ linear attention.** Common misread: treat $\delta_\ell$ as
pre-computed ⇒ backprop = Hebbian/linear attention on gradients. False — $\delta_\ell$
is generated by the memory itself ⇒ backprop is **self-referential** (Schmidhuber
1993), a richer AM than linear attention. (Made precise in §4.5.)

## C2. Momentum-based optimizers as AMs (§4.2)
Plain GD (32) ignores the traversed landscape. Momentum adds an **EMA of past
gradients**:
$$W_{\ell_{t+1}}=W_{\ell_t}+m_{\ell_{t+1}},\quad m_{\ell_{t+1}}=\alpha_{t+1}m_{\ell_t}-\eta_{t+1}\delta_\ell\hat{x}_{\ell-1}^\top \;(33)$$
With $\alpha_{t+1}=1$ the momentum solves $\min_m\langle m\hat{x}_{\ell-1},\delta_\ell\rangle$ — (34);
$\alpha\neq1$ adds $\ell_2$ reg on $m$. ⇒ momentum is an AM compressing past
gradients ⇒ **momentum-SGD is 2-level**: inner learns $m$, outer uses $m$ to update $W$.

Generalized momentum: $m_\ell$ = solution of *any* AM $\min_m\tilde{\mathcal{L}}(m;\hat{x}_{\ell-1},-\delta_\ell)$ — (37).
Appendix B: **Adam = optimal AM for the $L_2$-regression objective predicting
gradient variance**; RMSProp, SignSGD(+mom), NAdam, AMSGrad, RAdam, Lion, AdaGrad
all fall out as AMs. AdaGrad ↔ Shampoo/Soap (preconditioner approximation) ⇒ the
whole optimizer zoo is re-expressible as AM.

**Preconditioning = learning a coordinate system.** $W_{t+1}=W_t-\eta P_{t+1}^{-1}g_{t+1}$ — (38/39);
$P$ is an AM mapping $\hat{g}\to g$ via $\min_P\tilde{\mathcal{L}}(P(\hat g);g)$ — (40),
optionally itself learned by GD (41) → another nested level. Identity map recovers
Adam/AdaGrad. **Muon** (42): $W_{t+1}=W_t+\text{NewtonSchulz}_k(m_{t+1})$ — NewtonSchulz
is the preconditioner that **orthogonalizes** the momentum; objective
$\tilde{\mathcal{L}}=\|P(g)^\top P(g)-I\|_F^2$ — (43), one GD step gives the
degree-3 polynomial update (44). ⇒ higher-freq level learns the orthogonal map,
lower-freq level uses it.

## C3. Long context in optimizers → continual learning failure (§4.3)
Momentum is just a **low-pass filter** with tiny effective memory: with $\beta=0.9$,
the last ~6 (resp. ~43) gradients carry 50% (resp. 99%) of $S_t=\sum\beta^i(1-\beta)$;
anything beyond ~43 steps contributes <1%.

**Orthogonal-tasks continual learning:** tasks $\mathcal{L}_i(W)=\mathbb{E}[(W^\top x-y)^2]$
with gradients in orthogonal directions $u_i$. After many steps on task $t$, the
momentum rotates toward $u_t$ and *forgets the old gradient subspace it should
avoid* ⇒ weights move in directions that damage past tasks ⇒ catastrophic
forgetting. **Key point: this failure is the optimizer's memory management, not
model capacity.**

> Box: continual learning needs the *optimizer itself* to keep a long-term
> compression of the gradient subspace to find effective solutions over time.

## C4. More expressive momentum designs (§4.4)
Vanilla momentum = value-less, Hebbian (low capacity, state-independent updates).
NL upgrades along 5 axes:
1. **More expressive association** — give it values $v_i=P_i$, minimize
   $\langle m\nabla\mathcal{L}^\top,P\rangle$ (46) ⇒ preconditioned momentum GD (47).
2. **More expressive objective** — swap dot-product for $L_2$ regression
   $\|m\nabla\mathcal{L}^\top-P_i\|_2^2$ ⇒ **delta-rule** update (48/49); state-dependent
   decay, can *learn to forget* past gradients. Call these **Delta Momentum**.
3. **More expressive memory** — replace matrix $m$ with an **MLP**:
   $m_{i+1}=\alpha m_i-\eta\nabla\mathcal{L}^{(2)}(m_i;u_i,\mathbb{1})$ — (50) ⇒
   **Deep Momentum GD (DMGD)**.
4. **Higher-order feature maps** on the gradient keys, $\phi(\nabla\mathcal{L})$ — (51).
5. **Nonlinear outputs** — $W_{i+1}=W_i+\sigma(m_{i+1}(u_i))$ — (52); with
   $\sigma=$ NewtonSchulz and linear $m$ this **recovers Muon**.

**Toy demo (Fig. 4):** time-varying-curvature surface $\psi(r,\theta)=r^2+k(r-\theta+\alpha\sin\omega r)^2$.
Standard momentum (low-pass) is dragged by stale gradients when the landscape
changes fast; **Delta Momentum converges faster** via gradient-dependent weight
decay (momentum decays/stops when needed).

## C5. Beyond GD/momentum: DGD and GGD (§4.5)
Backprop-GD is the AM (54)/(55) learning the negative gradient direction. Drawback:
dot-product inner objective treats each gradient **independent of state** — fine for
i.i.d., bad for dependent streams (sequence tokens, continual tasks). Fixes:

- **Delta Gradient Descent (DGD):** use $L_2$ regression inner objective (56); with
  normalized inputs ($\|x_t\|_2=\lambda$) Sherman–Morrison gives
  $W_{t+1}=W_t(I-\eta'_t x_t x_t^\top)-\eta'_t\nabla_{y_t}\mathcal{L}\otimes x_t$ — (57).
  ⇒ update now carries an **adaptive, data-dependent decay** of the previous weight
  state (not just the current gradient).
- **Self-referential view:** $W_{t+1}=W_t+\eta v_t\otimes x_t$ with
  $v_t=f_{W_t}(x_t)=-\nabla_{y_t}\mathcal{L}$ — (58): values are *generated by the
  memory itself*, so it's not parallelizable linear recurrence — it controls its own
  learning.
- **Definition 5 — Generalized Gradient Descent (GGD):**
  $W_{t+1}=\arg\min_W\tilde{\mathcal{L}}(x_t,u_t)+\text{Ret}(W,\{W_i\}_{i=t-c+1}^t)$ — (59),
  $u_t=f_{W_t}(x_t)$ self-generated (60); $\text{Ret}(\cdot)$ keeps the new solution
  near the current state. Same form for momentum ⇒ **Generalized Momentum (GM)**.

> **Optimizers in continual learning:** momentum stores knowledge about the loss
> landscape (not conventionally "trainable"). Ending pretraining and discarding
> momentum states erases that knowledge → hurts further learning. In a true
> continual setup: data knowledge lives in backprop'd weights; *knowledge of how
> the model optimizes itself* lives in the lower-frequency momentum levels.

### Why §4 matters for our experiments
- C3 gives a **clean, mechanistic story for catastrophic forgetting as an optimizer
  memory failure** — directly testable on experiments/0001 with the orthogonal-tasks
  construction (Eq. 45).
- Delta Momentum / DGD (C4–C5) are concrete, small interventions to try as the
  "fix" arm vs. plain SGD-momentum on that benchmark.

---

# D. Architectures as NSAM (§5)

Frame (Miras, Behrouz 2025b): sequence models = AMs solving an internal objective
("attentional bias") with some optimizer over a function class (memory architecture).
Keys/values/queries are projections $k_t=x_tW_k$, $v_t=x_tW_v$, $q_t=x_tW_q$ — (61).
Projections live at a **lower-frequency level**; the sequence-model component is
**higher-frequency** — so the AM's internal learning is the high-freq level discussed
here.

## D1. Softmax attention = non-parametric AM
Softmax attention = closed-form (non-parametric) solution to an $\ell_2$ regression
via Nadaraya–Watson:
$$\mathcal{M}^*=\arg\min_\mathcal{M}\sum_{i=1}^L s(k_i,q)\|v_i-\mathcal{M}\|_2^2=\sum_i\frac{s(k_i,q)}{\sum_j s(k_j,q)}v_i \;(62)$$
Restrict the sum to past $c$ tokens ⇒ **sliding-window attention** (63). So attention
is Definition 1 with a *non-parametric* solver instead of GD.

## D2. The recurrent-model ladder (same AM, richer learning rule)
- **Hebbian RNNs** (Linear attention, RetNet, RWKV, lightning attn): inner objective
  = dot-product $\tilde{\mathcal{L}}=-2\langle\mathcal{M}k_t,v_t\rangle$; GD+decay gives
  $\mathcal{M}_t=\alpha_t\mathcal{M}_{t-1}+\eta_t v_t\phi(k_t)^\top$ — (64). Choices of
  $\alpha_t$ (1 / learnable / channel / input-dep) and $\phi$ (identity/poly kernel)
  recover all linear-attention variants.
- **Delta-rule RNNs** (DeltaNet, Longhorn, RWKV7): inner objective = MSE
  $\|\mathcal{M}_t k_t-v_t\|_2^2$ with retention $\text{Ret}=\|\mathcal{M}_t-\mathcal{M}_{t-1}\|_F^2$;
  SGD ⇒ $\mathcal{M}_t=(I-\eta_t k_t k_t^\top)\mathcal{M}_{t-1}+\eta_t v_t k_t^\top$ — (65).
  Different retention gates / weight-decay norms / multi-step GD / learnable $\eta_t,\alpha_t$
  span the delta-rule family.
- **Beyond Hebbian/Delta:** **Oja's rule / OjaNet** adds a unit-norm constraint
  $\mathcal{M}_t=\alpha_t\mathcal{M}_{t-1}+\eta_t v_t(\phi(k_t)^\top-\mathcal{M}_{t-1}^\top v_t)$ — (66),
  = one GD step on $\tilde{\mathcal{L}}=-2\langle Mk_t,v_t\rangle+\|M^\top v_t\|_2^2$ (67)
  (stabilizes Hebbian but empirically < Delta). Behrouz 2025b: use **non-Euclidean
  $L_p=\|\cdot\|_p^p$** internal objective → better long-context. **Omega rule**
  (Behrouz 2025a): update over a *window of past inputs* not just current —
  $\mathcal{M}_t=\alpha_t\mathcal{M}_{t-1}-\sum_{i=t-c+1}^t\gamma_{t,i}\tilde{\mathcal{L}}(\mathcal{M}_t;\phi(k_i),v_i)$ — (68);
  $\gamma{=}1$, $c{=}$full context ⇒ collapses to online case.

## D3. Gating note & brain perspective (§5.1)
- **Gating of a linear layer** by the sequence-model output: when the memory's init
  is *not* meta-learned (early linear transformers), the gate **acts as the persistent
  memory and the memory-module initialization** — i.e. it supplies the level-1
  knowledge the meta-learned init would otherwise provide.
- **Uniform/reusable structure:** every component (attention, MLP, Linear Attn++,
  projections) is a matrix-valued or deep feedforward net; AdaTransformer block
  Eq. (69) makes this explicit. Attention = non-parametric solution of a regression;
  Linear Attn++ = dot-product AM over linear functions. **The only difference between
  architecture components is their level, objective, and learning rule.**

> Box: the heterogeneity we *see* in architectures is an illusion from viewing only
> the *solution* of each optimization problem, missing the NL (level) axis. Modern
> models are uniform reusable feedforward nets at different timescales.

---

# E. Takeaways & Revisiting Common Terms (§6)

NL re-defines the standard vocabulary:

- **Memory & learning:** memory is *not* an isolated block. **Any input-caused update
  (at any level, by any optimizer) is a stored memory; learning = the process of
  acquiring useful memory** (Okano 2000). Aligns with §4.1 (backprop = self-referential
  AM). CMS = memories stored at many timescales ⇒ robust to forgetting.
- **Parameters:** models have **more parameters than the "learnable" ones**. Momentum
  states and RNN hidden states store knowledge (loss landscape / current context)
  even though they aren't optimized at the lowest-frequency level. Discarding them
  (e.g. context change, end of pretraining) erases that knowledge — they all
  contribute to expressivity.
- **More computation per neuron:** stacking levels isn't only CMS — it adds compute
  *depth per parameter* (e.g. Muon's $k$ NewtonSchulz steps = an internal optimization
  per momentum update). NL ≠ just multi-memory.
- **In-context learning:** ICL = adapting to a context, defined per level/block.
  Softmax-attention conditioning on full context = *non-parametric ICL*; recurrent
  memory conditioning on *compressed* context = *parametric ICL*. **ICL is not
  emergent — it's a direct consequence of having multiple NL levels.** But good ICL
  still needs a well-trained low-frequency level so the high-freq level can adapt fast.
- **Test-time training / memorization** = instances of **parametric ICL**; knowledge
  vanishes when context is removed. Misleading label once you move to continual
  learning (no test/train boundary).
- **Pretraining = ICL with ultra-large context** (context = the whole pretraining
  set; objective = NTP; optimizer = AdamW). The train/test distinction is just an
  artifact of disconnecting knowledge transfer from the highest-freq level to the
  low-freq (pretraining) level.
- **Continual learning:** every model (pretraining or test) is *already* doing
  continual learning per sample; the failure is that knowledge doesn't **transfer**
  to persistent levels. **No real train/test boundary** — a neural learning module
  only has two states: receiving input, or being an isolated learning system.
- **Architectures & hybrids:** recurrent (deep/linear memory) models = **MLP blocks
  with an added level of internal computation**. Hybrid architectures = vanilla
  Transformers where some MLP blocks got an extra level. *Key gap:* deep memory
  modules (Titans, Atlas, Miras, TTT) **do** knowledge transfer high→low via
  meta-learning the memory's init; **most linear-memory recurrent models do not**
  transfer knowledge between levels.
- **Inter-connected system:** architecture *generates the context (gradients) for
  the optimizer*. So the optimal optimizer is architecture-specific → future:
  design architecture-specific optimizers.
- **Optimizers vs. learned optimizers:** same concept, different frequency/context —
  learned-optimizer params sit at the lowest-freq level (trained with everything);
  vanilla-optimizer params sit at *their own* level with their own gradient flow.

---

# F. Continuum Multi-Timescale Memory System (§7)

Motivation: classic models have only working memory (attention) + persistent memory
(MLP, frozen after pretraining). Replace the binary with a **knowledge-storage
feedforward per frequency level**.

## F1. CMS definition
Chain of MLP blocks $\text{MLP}^{(f_1)},\dots,\text{MLP}^{(f_k)}$, block $\ell$ has
chunk size $C^{(\ell)}:=\max_i C^{(i)}/f_\ell$:
$$y_t=\text{MLP}^{(f_k)}(\text{MLP}^{(f_{k-1})}(\cdots\text{MLP}^{(f_1)}(x_t))) \;(70)$$
Block $\ell$ params update **only every $C^{(\ell)}$ steps** (chunked accumulated
gradient), else no-op — Eq. (71). Conventional Transformer = special case $k{=}1$,
update frequency 0. Each $\theta^{(f_\ell)}$ compresses *its own* context → represents
abstract knowledge at its timescale.

## F2. Three CMS connection variants
- **Nested CMS:** init of level $s{+}1$ is meta-learned in level $s$ —
  $\theta_0^{(f_{s+1})}=\arg\min_\Phi\mathbb{E}_{\mathcal{T}\sim C^{(s)}}[\ell(\Theta,\mathcal{T};\Phi)]$ — (72);
  each level re-initializes after its context ends ⇒ higher-order ICL.
- **Sequential CMS:** blocks chained (output of $s$ → input of $s{+}1$); **all** inits
  connected via backprop at the lowest-frequency level — Eq. (73). Most persistent
  knowledge = compression of the *same* context flow.
- **Independent (head-wise) CMS:** same knowledge transfer as (73) but parallel
  independent blocks aggregated: $y_t=\text{Agg}(\text{MLP}^{(f_k)}(x_t),\dots,\text{MLP}^{(f_1)}(x_t))$ — (74);
  e.g. learnable weighted sum.

## F3. Why CMS resists forgetting + efficiency
- **Anti-forgetting:** if updating $\text{MLP}^{(f_s)}$ overwrites knowledge, that
  knowledge still lives in slower blocks $\text{MLP}^{(f_{s'})}$, $s'<s$; backprop
  through their meta-learned inits can **circle the knowledge back** → a loop through
  the time dimension → hard to forget important knowledge.
- **Efficient:** updates only hit blocks at their scheduled time. For a Transformer
  with MLP→CMS (4 levels, top freq $\hat f$, $L_{layer}$ layers, dim $d_{in}$):
  avg update cost $O(\tfrac{1}{\hat f}\times\tfrac{L_{layer}}{5}\times d_{in}^2)$ —
  small subset per step. Eq. (71) also **unlocks sequence parallelism**: within a
  chunk ($i\not\equiv0\bmod C^{(\ell)}$) there's no sequential dependency (training à
  la Sun 2024 / Behrouz 2025c).

## F4. CMS in optimizers — M3 (Multi-scale Momentum Muon) (§7.2)
Proof-of-concept: replace the AM behind Adam's $H$ term with CMS (independent
variant) → two-level momentum $M^{(1)}$ (fast, per-step) and $M^{(2)}$ (slow, every
$\hat C$ steps), aggregated by weighted sum with coeff $\alpha$, both passed through
$\text{NewtonSchulz}_T$ before aggregation — Eq. (75).

**Algorithm 1 (M3) skeleton:** for each slow iter $k$: slow momentum
$M^{(2)}\!+\!=\beta_3\sum g_i$, $O^{(2)}\!=\!\text{NewtonSchulz}_T(M^{(2)})$; inner
loop over $f$ steps: $g_t=\nabla\mathcal{L}$, $M^{(1)}\!+\!=\beta_1 g_t$,
$V_t\!+\!=\beta_2 g_t^2$, $O^{(1)}=\text{NewtonSchulz}_T(M^{(1)})$, update
$\Theta_t=\Theta_{t-1}-\eta\frac{O^{(1)}+\alpha O^{(2)}}{\sqrt{V_t}+\epsilon}$.
⇒ **M3 = Adam + Muon + CMS.** Caveat: compute overhead; M3 is a POC, may not scale
(Fig. 12). Core idea: *delay momentum update to gain longer context.*

## F5. Ad-hoc level stacking: init CMS from a pretrained model (§7.3)
Each level has its own context/optimizer, so init each level independently. Use a
pretrained model's MLP blocks as CMS block inits:
$\text{MLP}_0^{(f_i)}(\cdot)=\text{MLP}_{\text{pre-trained}_i}(\cdot)$.
Works because high-freq levels exploit low-freq stored knowledge to adapt fast; and
the high-freq internal LR controls adaptability — $\eta_t^{(\ell)}\to0$ keeps blocks
at their pretrained state (no adaptation, = directly using pretrained weights). Used
in §9 to port pretrained Transformers into HOPE's setup.

## F6. HOPE backbone (Fig. 5)
HOPE = **Self-Modifying Titans + CMS chain**. Sketch from Fig. 5: q/k/v linears →
Self-Modifying Titans block → stack of FFNs at increasing frequency (High→Mid→Low
freq FFN), with chunk lengths spanning 16 → 1M → 16M and per-level BPTT lengths.
Contrast: Transformer = q/k/v linear → Attention (chunk ∞, freq ∞ / 0 in pretraining)
→ single FFN (chunk ∞, freq 0). HOPE replaces the two-frequency Transformer with a
**continuum of frequencies + a self-modifying memory**.

---

# G. HOPE: Self-Referential Module + Continuum Memory (§8)

**Why Transformers are limited (setup):** softmax attention = a *perfect*
non-parametric AM (caches all tokens, freq ∞, $L_2$/Nadaraya–Watson, Eq. 62) →
parametric RNNs with the same (matrix-valued) search space *can't* beat it at scale.
Transformers are **2-level**: level 1 = projections+MLP (static after pretraining);
level 2 = non-parametric ICL. Limits: (1) shallow computational depth (state-tracking
results, Merrill 2024 etc.); (2) projections $W_k,W_v,W_q$ frozen post-pretraining ⇒
token contextualization bounded by pretrained knowledge (a 1-layer Transformer's
projection depends only on the token + position, misses context-dependent word
meanings). Short convs / canon layers help local mixing but can't capture global
context. Fix = make *all* components able to self-modify in-context.

## G1. Deep Self-Referential Titans (§8.1)
General parametric AM: project to **k, v, q, η, α** ($\eta$=LR, $\alpha$=retention/forget
gate) — Eq. (76); optimize memory (77); read $y_t=\mathcal{M}_t q_t$ (78).
- **Fully adaptive (shared-value) version:** add a level so each projection is itself
  a memory updated in-context, sharing values for efficiency — Eq. (79–82); all init
  states $\mathcal{M}_{\square,0}$ **meta-learned across contexts** (essential for
  fast adaptation, training stability, noise robustness).
- **Self-modifying version:** each memory **generates its own values**
  $\hat v_{\square,t}=\mathcal{M}_{\square,t-1}(v_t)$ — Eq. (83–85); only $q_t=x_tW_q$
  stays non-adaptive. Objective = $L_2$ regression $\|\mathcal{M}(k)-v\|_2^2$, but
  since tokens are correlated, optimize with **DGD + weight decay**:
  $\mathcal{M}_{\square,t}=\mathcal{M}_{\square,t-1}(\alpha_t I-\eta_t k_t k_t^\top)-\eta_t\nabla\mathcal{L}(\mathcal{M}_{\square,t-1};k_t,\hat v_{\square,t})$ — Eq. (88).
- Memory architecture = **2-layer residual MLP** for all components:
  $\mathcal{M}_\square(\cdot)=(\cdot)+W_{\square,1}\sigma(W_{\square,2}(\cdot))$ — Eq. (89).

## G2. Fast & parallelizable training (§8.2)
Chunk-wise training (à la Sun 2024 / Behrouz 2025c), update freq $f_\square=L/C$,
**two chunk sizes** (one for $\mathcal{M}_{memory}$, one for the rest). Split into
$\lceil L/C\rceil$ chunks; generate all keys/values/LRs/decays at chunk end for the
*next* chunk, take gradient w.r.t. the previous chunk's last state ⇒ whole chunk
computed in parallel (dual form, Eq. 90–93). Recurrent form derived for dot-product
($\nabla=vk^\top$, Eq. 92) and $L_2$ ($\nabla=(\mathcal{M}k-v)k^\top$, Eq. 93).

## G3. HOPE = Self-Modifying Titans → CMS (§8.3)
CMS = high capacity / simple rule; Self-Modifying Titans = small capacity / expressive
rule → **complementary**. HOPE stacks them: self-modifying Titans block produces
$o_t$, then a CMS chain of MLPs at increasing frequency:
$y_t=\text{MLP}^{(f_k)}(\cdots\text{MLP}^{(f_1)}(o_t))$ — Eq. (94–97). Practical:
$L_2$-normalize q,k; local conv window = 4.
**HOPE-Attention** = HOPE variant with self-modifying Titans replaced by softmax
global attention (isolates the CMS contribution).

---

# H. Experiments (§9)

Backbones: continual-pretrain Llama-3B / Llama3-8B (§7.3 init, +15B tokens) for
adaptation tasks; train 760M/1.3B from scratch (30B/100B FineWeb-Edu + long-context,
vocab 32K, AdamW) for LM. Baselines use *best reported* results.

## H1. Continual learning & long context (§9.1)
- **Class-incremental (Fig. 6, CLINC/Banking/DBpedia):** HOPE > InCA > EWC > ICL on
  *both* Llama-3B and Llama3-8B. E.g. CLINC/Llama3-8B ≈ 93% (HOPE) vs ~88 (InCA) /
  ~80 (EWC) / ~78 (ICL). HOPE > ICL ⇒ multi-level ICL (CMS) helps; HOPE > InCA/EWC
  ⇒ inter-level knowledge transfer matters.
- **Levels effect on ICL (Fig. 7, MK-NIAH/LongHealth/QASPER):** HOPE (any #levels,
  any lowest-freq) beats ICL baseline *and* DuoAttention. (1) **more memory levels →
  better long-context**; (2) **higher lowest-frequency → worse** (less persistent
  long-term memory) but more efficient. **"Lowest Freq = 2K" = sweet spot** (≈512's
  accuracy, much cheaper forward pass).
- **Continual Translation of a Novel Language (CTNL, Fig. 8):** MTOB(Kalamang)+Manchu,
  translate→English. Single-language (red) vs sequential continual (blue). ICL
  collapses under continual (catastrophic forgetting, falls back to pretraining);
  more HOPE levels recover it — **HOPE-3 ≈ recovers single-language ICL** without the
  forgetting.

## H2. HOPE long-context from scratch (§9.2)
- **NIAH / RULER (Table 1):** among attention-free, **HOPE best across all tasks &
  lengths**; deep memory > linear at long seq (more capacity); HOPE > Titans at long
  context ⇒ self-referential + CMS both help. **HOPE-Attention > Transformer** on
  S-NIAH-1 (100/100/100 vs 88.6/76.4/79.8 @4/8/16K) and roughly matches/exceeds on
  the rest ⇒ CMS helps even with softmax attention.
- **BABILong (Fig. 9):** large models (GPT4/4o-mini) fail ~128–256K; RAG-Llama
  plateaus lower; Titans/ARMT competitive to ~1M then drop; **HOPE holds performance
  to 10M context** (CMS). Without fine-tuning all small models (incl. HOPE) drop
  (need capacity + tuned low-freq levels).

## H3. LM & common-sense reasoning (§9.3, Table 2)
**HOPE best average** at both scales. 760M/30B: avg **52.28** (Titans 51.68, Samba
51.46, Comba 50.89); Wiki ppl 18.68 / LMB ppl 20.07. 1.3B/100B: avg **58.04**
(Titans 56.82, Samba 54.46); Wiki ppl **14.39** / LMB ppl **10.08**, best on PIQA
73.9 / Hella 57.5 / Wino 61.2 / ARC-e 73.8 / ARC-c 42.7. Gains *grow with scale*.

## H4. Recall, MAD, formal languages (§9.4–9.5)
- **Short in-context recall (Table 3):** Transformers best; **HOPE best attention-free**,
  closes the gap (e.g. SWDE 65.9 vs Transformer 71.4 vs Titans 60.8).
- **MAD synthetic (Table 4):** **HOPE best overall, beating Transformers** (Compress
  51.2, Fuzzy-ICR 52.1, Selective-Copying 99.7, Memory 85.2; ICR 100 for all).
- **Formal language recognition (Table 5):** HOPE = **100% on all** (parity, (aa)*,
  (abab)*, counters, shuffle-2), matching LSTM/SRWM where Transformer/Linear/DeltaNet
  fail Parity & star-free — *but HOPE is parallelizable* (its key edge over LSTM/SRWM).

## H5. Ablations & scaling (§9.6–9.7)
- **Ablation (Table 6):** HOPE 12.24 ppl / 58.1 acc. Removing each component hurts:
  w/o DGD 13.41/56.5, w/o momentum 13.58/56.9, w/o weight-decay 13.71/57.2, w/o CMS
  13.04/57.3, w/o inner-proj k 13.77/56.9, **w/o inner-proj v 13.90/55.1 (worst)**,
  w/o inner-proj q 12.19/57.4 (ppl ~flat, acc down). Every piece is net-positive.
- **Context usage (Fig. 10):** HOPE perplexity keeps dropping with more context;
  HOPE w/o CMS and w/o DGD degrade — confirms memory-management claim.
- **M3 optimizer (§9.7):** **ViT/ImageNet-21K (Fig. 11): M3 best train & test loss
  vs AdamW, Muon.** **Efficiency (Fig. 12, 140M & 1.3B LM):** M3 slower than Muon
  (multiple momentum memories), ≈ on par with AdaMuon.

---

# I. Conclusion & honest limitations (§10)

NL recasts ML as inter-connected multi-level optimization; architectures *and*
optimizers = nested associative memories compressing their own context; pretraining
/ ICL / continual learning = one mechanism (compress+reuse context at different
timescales). Backprop, momentum, preconditioning = AM mechanisms; existing methods =
points in a previously-hidden design space. New artifacts: **DGD, Delta Momentum,
M3, CMS, HOPE**.

**Is catastrophic forgetting solved? No.** The paper explicitly says forgetting is
*not* solved in general — from the NL view it's a **natural consequence of
compression under limited capacity** (the model must forget to free capacity). NL is
positioned as a **roadmap, not a destination**: progress should come from exploiting
the *levels* axis, not ever-deeper static nets.

My added limitations / cautions:
- Baselines = "best reported," not all re-run under matched compute → effect sizes
  (e.g. +0.6–1.2 avg in Table 2) are real but not huge; cost-honesty caveat from
  docs/prior_art.md stands.
- M3 is a POC with compute overhead (Fig. 12) and explicitly may not scale; HOPE
  excludes Cartridges comparison citing higher memory/compute — apples-to-apples
  efficiency unresolved.
- HOPE-without-finetuning collapses on 10M BABILong → the long-context win is partly
  a fine-tuning artifact, not pure architecture.
- Many "X is an associative memory" results assume specific objectives/normalization
  (e.g. Sherman–Morrison needs $\|x\|_2=\lambda$); generality of the unification is
  somewhat narrower than the framing suggests.

---

## Reproduction notes (now that the paper is fully read)
- **Cheapest entry point for our repo:** the §4.3 orthogonal-tasks construction
  (Eq. 45) + Delta Momentum (Eq. 49) vs SGD-momentum — a few-MLP toy, fits experiment
  0001's budget, directly tests "forgetting = optimizer memory failure."
- **DGD (Eq. 57)** is a ~5-line optimizer change (normalized input + Sherman–Morrison
  decay term) — cheap ablation arm.
- **Full HOPE / M3 / CMS** require from-scratch 760M+/30B-token training and
  chunk-wise parallel kernels — out of scope on local hardware; treat as
  reference/diagnostic, not reproduction.

## Status
Full paper read (§1–§10). Pass 2 reached. Building-on: feed C3 + DGD/Delta-Momentum
into experiments/0001-reproduce-forgetting as the mechanism + fix arms.
