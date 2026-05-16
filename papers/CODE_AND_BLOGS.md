# Code, libraries, blogs, living lists

Not papers — the engineering and explainer layer. Use these to (a) not
reimplement EWC/GEM/replay from scratch, (b) get fair baselines for free, and
(c) track new work without manual arXiv trawling.

## Continual-learning libraries (use these for baselines — don't hand-roll)

- **[ContinualAI / Avalanche](https://github.com/ContinualAI/avalanche)** —
  PyTorch, MIT-licensed, end-to-end CL: benchmarks (Split-MNIST etc.),
  strategies, metrics, logging. `pip install avalanche-lib`. **This is your
  fastest path to a correct experiment 0001** — Split-MNIST + naive baseline +
  forgetting metrics already implemented. Use it; don't reinvent the harness.
  - [continual-learning-baselines](https://github.com/ContinualAI/continual-learning-baselines)
    — reference EWC/SI/GEM/A-GEM/LwF/iCaRL/GDumb implementations + expected
    numbers. Your "reproduce the baseline" step, pre-built.
- **[aimagelab / Mammoth](https://github.com/aimagelab/mammoth)** — 70+ models,
  23 datasets, official home of Dark Experience Replay. Heavier than Avalanche;
  good once you're comparing many methods.
- **[GMvandeVen / continual-learning](https://github.com/GMvandeVen/continual-learning)**
  — clean, readable reference impl tied to the "three scenarios" paper. Best for
  *understanding* the code, not for scale. Read this one to learn.

## Living lists (track new work without trawling arXiv yourself)

- **[zzz47zzz / awesome-lifelong-learning-methods-for-llm](https://github.com/zzz47zzz/awesome-lifelong-learning-methods-for-llm)**
  — companion to the CSUR 2025 LLM-CL survey; updated regularly. Your primary
  LLM-CL tracker.
- [xialeiliu / Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
  — consolidates NeurIPS/CVPR/ICML continual-learning papers per year.
- [Ghy0501 / Awesome-Continual-Learning-in-Generative-Models](https://github.com/Ghy0501/Awesome-Continual-Learning-in-Generative-Models)
  — directly on-thesis: CL *in generative models* (your quality bar).
- [Wang-ML-Lab / llm-continual-learning-survey](https://github.com/Wang-ML-Lab/llm-continual-learning-survey)
  — repo behind the comprehensive LLM-CL survey.

## Blogs / explainers (for intuition — verify claims against papers)

- [Together AI — Continued Fine-tuning of LLMs: A Technical Deep Dive](https://www.together.ai/blog/continued-fine-tuning)
  — practical: data mixing/replay ratios, how dataset similarity drives
  forgetting. Concrete and on-thesis.
- [Cameron R. Wolfe — Continual Learning with RL for LLMs](https://cameronrwolfe.substack.com/p/rl-continual-learning)
  — why on-policy RL incidentally mitigates forgetting.
- [Baicen Xiao — Avoiding Amnesia: Practical Guides to Mitigate Catastrophic Forgetting](https://medium.com/@baicenxiao/avoiding-amnesia-some-practical-guides-to-mitigate-catastrophic-forgetting-in-llms-post-training-6a23e4f064cb)
- [Google Research — Introducing Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
  — the accessible version of the Hope paper.

## How to use this without procrastinating

1. For experiment 0001: install **Avalanche**, run its Split-MNIST + naive
   baseline, confirm you reproduce a known forgetting number. That's the whole
   baseline step — hours, not days.
2. Subscribe (star) the two living lists. Check them weekly, not daily.
3. A blog claim never enters an experiment README without the paper behind it.
