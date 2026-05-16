# Prior art & the gap (read with the charter)

The honest premise: **almost everything in this thesis is ~80–90% already
implemented by someone.** That is the normal condition of good research, not a
kill signal. A direction nobody has touched is usually one people found not
worth touching. Contribution = the load-bearing residual, executed rigorously
at a defensible scale.

## Coverage map — who already did what

| Thesis component | Closest prior work | How covered |
|---|---|---|
| Continual learning without forgetting | EWC/SI, GEM/A-GEM, replay, parameter isolation | Heavily. Mature method families. |
| Few-shot acquisition | MAML, ProtoNets, Reptile | Heavily. |
| Continual **+** few-shot together | OML (Javed & White 2019), ANML (Beaulieu 2020), MER (Riemer 2018) | Done at small scale, classification. |
| Continual + few-shot + **generative LM** | **Nested Learning / Hope** (Google, [2512.24695](https://arxiv.org/abs/2512.24695), NeurIPS 2025) | Closest single work. Abstract claims LM + knowledge incorporation + few-shot + continual + long-context. |
| Continual fine-tuning "without full retraining" as the goal | [Future of CL in Foundation Models](https://arxiv.org/abs/2506.03320) | Names it as a key direction; agenda-setting, not a method. |

## The gap (verified, not assumed)

Checked Hope's abstract directly on 2026-05-16:

- ❌ **No training-cost / compute / wall-clock measurement** vs from-scratch
  retraining anywhere in the abstract.
- ❌ **No scale disclosed**; big-lab paper, almost certainly not reproducible on
  an M1 Max.
- ❌ Abstract presents "promising results", **no stated limitations** — i.e. the
  cost/scale honesty is exactly what's missing.

The "Future of CL" paper independently frames the prize as adaptation
*"without full retraining, avoiding computationally expensive"* paths — i.e. the
cost framing is acknowledged as important but treated as a direction, not
measured.

## Therefore — this project's defensible residual

The load-bearing 20% nobody is cleanly doing:

> **Does continual/few-shot learning actually reduce training cost — measured
> (wall-clock / FLOPs / $), reproducibly, at a scale a researcher with no GPU
> farm can verify — while holding generative quality against a fair same-scale
> baseline?**

Every experiment in this repo exists to answer that one question. Overlap with
Hope/OML/MER is expected and fine; we are not racing them on capability, we are
supplying the cost-honest, reproducible evidence they omit.

## Rules this implies

1. Reading Hope at Pass 3 is **mandatory before experiment 0002+** — you must
   know exactly what it does so you don't accidentally reinvent it without the
   cost angle.
2. "Someone already did X" is never a reason to stop. The only valid reasons to
   stop: the residual is *not* load-bearing, or it's been measured already.
3. Novelty is not a goal. Cost-honest reproducible evidence is the goal.
