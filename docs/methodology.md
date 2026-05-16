# How to do this properly

You've been brute-forcing: "implement X" without knowing why X, what it's
competing against, or how you'd know if it worked. This document is the
antidote. Follow it even when it feels slow. It is slow on purpose — the slow
part is the thinking, and the thinking is the research.

## 1. Reading a paper

Don't read linearly start-to-finish on the first pass. Three passes:

**Pass 1 (5–10 min) — should I even read this?**
Title, abstract, intro, figures, conclusion. Answer: what problem, what's the
claimed contribution, do I believe the setup is relevant to me. If no, stop and
record one line in `papers/README.md` saying why you skipped it.

**Pass 2 (~1 hr) — what did they actually do?**
Read the method and experiments. Ignore proofs. For each figure/table ask "what
variable changed and what happened." Fill the paper note template as you go.

**Pass 3 (only for papers you'll build on) — could I reproduce this?**
Reconstruct the method from memory. Where you can't, that's the part you didn't
understand — go back. This is the pass that separates "I read it" from "I get it".

The note you write must answer, in your own words and without jargon copied from
the abstract: *what was broken before, what they changed, what evidence shows it
helped, and where it still fails.* If you can't write the last one, you didn't
read critically.

## 2. From paper to question

A good research question is specific and answerable with an experiment you can
actually run on your hardware. Bad: "can diffusion LLMs be better?" Good: "does
replacing 2 of N attention layers with bidirectional Mamba-2 preserve HumanEval
Pass@1 within 1 point while cutting decode latency?"

The question almost always comes from a paper's stated limitation, an
unexplained result, or a "we leave this to future work."

## 3. From question to hypothesis

Rewrite the question as a falsifiable prediction with a number and a decision
rule, **before running anything**:

> H: Grafting 2 Mamba-2 layers into Open-dCoder-0.5B, then 200-step continued
> pretrain, yields HumanEval Pass@1 ≥ baseline − 1.0 pt.
> Decision: if Pass@1 < baseline − 1.0, the graft hurts quality → reject.

If there is no number and no decision rule, it is not a hypothesis yet.

## 4. Designing the experiment

- **Baseline first.** Reproduce the published/known number on your setup. If you
  can't reproduce the baseline, every later comparison is meaningless. This is
  the single most common place research goes wrong.
- **Change exactly one thing** between baseline and variant.
- **Pre-register** the expected result in the experiment README before the run.
  Predicting and being wrong teaches you more than not predicting.
- **Smallest experiment that can answer the question.** Subsample data, fewer
  steps, smaller model — just enough signal to decide. Scale only after a
  positive small result.

## 5. Running and recording

Every experiment folder must end up with: the exact config, the git commit it
ran at, the environment, the seed, the raw numbers, and a 3-line conclusion that
explicitly says "hypothesis supported / rejected / inconclusive" and why.

## 6. Concluding honestly

Three valid outcomes:
- **Supported** — note the effect size, not just "it worked". Then ask what's
  the next variable.
- **Rejected** — write *why* you think it failed. A well-understood negative
  result is a contribution and prevents you repeating it.
- **Inconclusive** — usually means the experiment was underpowered or had a
  confound. Fix the design, don't just rerun and hope.

Never delete a failed experiment. The failure is the data.

## Anti-patterns (what brute-forcing looked like)

- Implementing an architecture before reading the paper that motivates it.
- No baseline → "it gets 18.2" with nothing to compare to.
- Changing five things, one number moves, attributing it to your favorite one.
- Tweaking until the number looks good, no hypothesis (this is p-hacking).
- Throwing away runs that "didn't work" so the repo only shows wins.
