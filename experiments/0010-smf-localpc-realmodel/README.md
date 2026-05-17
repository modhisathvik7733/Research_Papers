# 0010 — The REAL test: SMF (official, Qwen-0.5B + PKM) + Local-PC

**Status:** PRE-REGISTERED + integration scaffolded. **NOT run** — requires a
real GPU (Vast.ai-class) and explicit budget go. This README is the
registration, fixed before any run. Idea lineage:
[ideas/nl-sparse-memory-localpc.md](../../ideas/nl-sparse-memory-localpc.md);
fixes the exp-0009 testbed-inadequacy by using the **official SMF code**
([github.com/prakharg55/SMF-ICML](https://github.com/prakharg55/SMF-ICML),
arXiv 2510.15103) on a real model, not a toy reimplementation.

## What is actually tested

Official SMF pipeline: Qwen-2.5-0.5B-Instruct retrofitted with Product-Key
Memory, two-stage (dense retrofit on OASST1 → sparse memory-row finetune on
MedMCQA), evaluated by the repo's own evaluator: MedMCQA accuracy (new
knowledge), WikiText sliding-window perplexity + TriviaQA alias accuracy
(retention / forgetting). Single experimental variable: the **optimizer**
in the sparse phase.

## Arms (their baselines + our one new arm)

`smf-adam` (official SMF, AdamW — their result) · `smf-localpc` (official
SMF, optimizer swapped to deployable Local-PC = gain-normalised K-level
multi-timescale momentum, O(1)-in-H; `local_pc.py` here) · `lora` ·
`full-ft` (both already in the repo, as reference). All other code,
data, slot-selection (TF-IDF) identical — only `create_optimizer` differs
for `smf-localpc`.

## Pre-registered hypothesis & honest prior

The synthesis claim: sparse memory handles retention; Local-PC handles
optimisation; they do **not** conflict (different axes), unlike exp-0008
where the NL-style optimiser anti-stacked with replay.

> - **R-1 (sanity):** `smf-adam` reproduces the paper's qualitative result
>   — substantially less forgetting than `full-ft`/`lora` at matched new
>   knowledge. If not, the environment is wrong; stop.
> - **R-2 (THE test — non-conflict):** `smf-localpc` is **not worse than
>   `smf-adam`** on *both* retention (WikiText ppl / TriviaQA) and new
>   knowledge (MedMCQA), by the verbatim exp-0002 paired three-way rule
>   over seeds. Coexistence = the synthesis is coherent on a real model.
> - **HONEST PRIOR (stated before running, from 9 prior experiments):**
>   On a non-nested model, deployable Local-PC is just a multi-timescale
>   momentum optimiser; its distinctive O(1)-credit value lives in the
>   nested deep-unroll regime (exp-0005), NOT here. Most likely outcome:
>   **`smf-localpc` ≈ `smf-adam` (Local-PC inert — no conflict, no gain)**.
>   A clean *worse* result = optimiser conflicts with retention even when
>   retention is architectural (hardened negative, closes the line). A
>   *better* result would be genuinely surprising and would itself need
>   replication before any claim. We are testing coexistence, **not**
>   promoting Local-PC as superior — no quality-win is pre-claimed.
> - **R-3 (scope):** this still does not test Local-PC's distinctive
>   deep-unroll value (exp-0005 owns that); 0010 tests only whether the
>   two survivors coexist on a real SMF model.

Decision rule: verbatim exp-0002 paired-CRN three-way + effect-size gate +
δ; ≥3 seeds (LLM runs are costly — n pre-registered at the budget granted,
n≥3 minimum, n=5 target); divergence guard. Report straight; no unqualified
repo-default metric without the paired contrast.

## Cost / feasibility (honest, decision-gating)

Qwen-0.5B two-stage + eval, ×(arms)×(seeds). Real GPU hours, real $ on
Vast.ai-class hardware; competes with the DiffuZamba budget. **Not runnable
meaningfully on the M1 Max.** Free, already-done de-risking: the Local-PC
optimiser + the trainer patch + this pre-registration + (optionally) a
CPU/MPS *integration smoke* that proves the code path runs (NOT a result).
The actual GPU runs require an explicit budget go.

## Escalation

1. (free) integration smoke — tiny steps/sample on CPU/MPS, proves the
   optimizer swap + pipeline execute; explicitly NOT a result.
2. (GPU $) `smf-adam` R-1 sanity at the granted scale.
3. (GPU $) `smf-localpc` R-2, paired vs `smf-adam`, n seeds.
4. n-up confirmation only if R-2 is AMBIGUOUS.

## Honest caveats baked in

Official code but a "rebuild" (anonymous-submission repo) — R-1 sanity is
the guard that it reproduces the paper. 0.5B scale (paper-faithful, not
frontier). Local-PC = deployable multi-timescale form; deep-unroll value
not tested here. No quality-win pre-claimed; the realistic, pre-registered
expectation is *coexistence with Local-PC inert*, which would still be the
first time the NL-side mechanism did not actively harm a real retention
system — a modest but real, honestly-bounded result.
