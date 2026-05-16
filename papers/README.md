# Reading list

Two parts:

1. **The core path** — read these now, in order, top-down. The spine of the
   thesis. Don't skip ahead.
2. **The map** — everything directly/indirectly related, grouped by theme. This
   is a *library you pull from when an experiment or idea sends you there*. Do
   **not** read it all before doing experiment 0001 — that's procrastination
   disguised as rigor. When you do read one, promote it to its own note
   (`<short-name>.md` from `TEMPLATE.md`).

Links are arXiv `abs` pages unless noted. A few classics predate arXiv — venue
given instead.

---

## Part 1 — The core path (read now, in order)

| # | Paper | Why (re: charter) | Status |
|---|-------|-------------------|--------|
| 1 | [Kirkpatrick et al. 2017 — Overcoming Catastrophic Forgetting (EWC)](https://arxiv.org/abs/1612.00796) | Canonical statement of the problem + the regularization fix. Foundation. | queued |
| 2 | [Lopez-Paz & Ranzato 2017 — Gradient Episodic Memory (GEM)](https://arxiv.org/abs/1706.08840) | Source of the metrics you'll use forever: Avg Accuracy, **Backward/Forward Transfer**. | queued |
| 3 | [Prabhu et al. 2020 — GDumb](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470120.pdf) | A dumb buffer+retrain baseline that beats fancy methods. Keeps you skeptical of complexity. *(ECCV 2020 PDF; no arXiv.)* | queued |
| 4 | [Finn et al. 2017 — Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400) | The *acquisition* face: learning a task from few examples. The other half of the thesis. | queued |
| 5 | [Wang et al. 2023 — A Comprehensive Survey of Continual Learning](https://arxiv.org/abs/2302.00487) | The map of method families. Pass 2 only — it's a map, not a method. | queued |

**Reading rule:** papers 1–4 to **Pass 3** (could-I-reproduce-it). #5 Pass 2.

---

## Part 2 — The map (pull from as needed)

### A. The phenomenon — why forgetting happens
- [McCloskey & Cohen 1989 — Catastrophic Interference in Connectionist Networks](https://doi.org/10.1016/S0079-7421(08)60536-8) — origin of the problem. *(Psych. of Learning & Motivation; no arXiv.)*
- [French 1999 — Catastrophic forgetting in connectionist networks](https://doi.org/10.1016/S1364-6613(99)01294-2) — the classic review. *(Trends Cogn. Sci.; no arXiv.)*
- [Goodfellow et al. 2013 — An Empirical Investigation of Catastrophic Forgetting](https://arxiv.org/abs/1312.6211) — first modern deep-net study of it.

### B. Continual learning — regularization family
- [Zenke et al. 2017 — Synaptic Intelligence (SI)](https://arxiv.org/abs/1703.04200)
- [Aljundi et al. 2018 — Memory Aware Synapses (MAS)](https://arxiv.org/abs/1711.09601)
- [Li & Hoiem 2016 — Learning without Forgetting (LwF)](https://arxiv.org/abs/1606.09282)

### C. Continual learning — replay / rehearsal
- [Chaudhry et al. 2019 — A-GEM (efficient GEM)](https://arxiv.org/abs/1812.00420)
- [Rebuffi et al. 2017 — iCaRL](https://arxiv.org/abs/1611.07725)
- [Shin et al. 2017 — Deep Generative Replay](https://arxiv.org/abs/1705.08690) — generate old data instead of storing it; ties to your "generative" bar.
- [Rolnick et al. 2019 — Experience Replay (CLEAR)](https://arxiv.org/abs/1811.11682)
- [Buzzega et al. 2020 — Dark Experience Replay (DER)](https://arxiv.org/abs/2004.07211)

### D. Continual learning — parameter isolation / dynamic architecture
- [Rusu et al. 2016 — Progressive Neural Networks](https://arxiv.org/abs/1606.04671)
- [Mallya & Lazebnik 2018 — PackNet](https://arxiv.org/abs/1711.05769)
- [Mallya et al. 2018 — Piggyback](https://arxiv.org/abs/1801.06519)
- [Serra et al. 2018 — Hard Attention to the Task (HAT)](https://arxiv.org/abs/1801.01423)
- [Yoon et al. 2018 — Dynamically Expandable Networks](https://arxiv.org/abs/1708.01547)

### E. Evaluation, scenarios, and honest baselines (read early — anti-self-deception)
- [van de Ven & Tolias 2019 — Three Scenarios for Continual Learning](https://arxiv.org/abs/1904.07734) — task/domain/class-incremental. Read before designing experiment 0001's setting.
- [Farquhar & Gal 2018 — Towards Robust Evaluations of Continual Learning](https://arxiv.org/abs/1805.09733)
- [De Lange et al. 2021 — A Continual Learning Survey (classification)](https://arxiv.org/abs/1909.08383)
- [Hadsell et al. 2020 — Embracing Change: Continual Learning in Deep NNs](https://doi.org/10.1016/j.tics.2020.09.004) — *(Trends Cogn. Sci.; no arXiv.)*

### F. Meta-learning / few-shot — the acquisition face
- [Nichol et al. 2018 — Reptile (first-order meta-learning)](https://arxiv.org/abs/1803.02999)
- [Snell et al. 2017 — Prototypical Networks](https://arxiv.org/abs/1703.05175)
- [Vinyals et al. 2016 — Matching Networks](https://arxiv.org/abs/1606.04080)
- [Santoro et al. 2016 — One-shot Learning with Memory-Augmented NNs](https://arxiv.org/abs/1605.06065)
- [Hospedales et al. 2020 — Meta-Learning in Neural Networks: A Survey](https://arxiv.org/abs/2004.05439)

### G. The bridge — meta + continual together (THIS is your "both at once")
- [Javed & White 2019 — Meta-Learning Representations for Continual Learning (OML)](https://arxiv.org/abs/1905.12588) — read this once core path is done; closest to your target regime.
- [Beaulieu et al. 2020 — Learning to Continually Learn (ANML)](https://arxiv.org/abs/2002.09571)
- [Finn et al. 2019 — Online Meta-Learning](https://arxiv.org/abs/1902.08438)
- [Riemer et al. 2018 — Learning to Learn without Forgetting (MER)](https://arxiv.org/abs/1810.11910)
- [Nagabandi et al. 2018 — Deep Online Learning via Meta-Learning](https://arxiv.org/abs/1812.07671)

### H. Why nets are sample-inefficient — generalization, scale, theory
- [Kaplan et al. 2020 — Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — quantifies the data/compute brute force you're fighting.
- [Hoffmann et al. 2022 — Chinchilla: Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556)
- [Lake et al. 2016 — Building Machines That Learn and Think Like People](https://arxiv.org/abs/1604.00289) — the human-vs-net sample-efficiency gap, stated sharply.
- [Power et al. 2022 — Grokking](https://arxiv.org/abs/2201.02177) — generalization long after memorization; structure vs. memorization.
- [Zhang et al. 2016 — Rethinking Generalization](https://arxiv.org/abs/1611.03530)

### I. Cost-reduction levers (the "bill" side of the thesis)
- [Hu et al. 2021 — LoRA](https://arxiv.org/abs/2106.09685) — cheap adaptation; also a parameter-isolation CL tool.
- [Houlsby et al. 2019 — Adapters](https://arxiv.org/abs/1902.00751)
- [Frankle & Carbin 2018 — Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
- [Hinton et al. 2015 — Distilling Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Sorscher et al. 2022 — Beyond Neural Scaling Laws (data pruning)](https://arxiv.org/abs/2206.14486) — beating power-law scaling with better data, not more.

### J. In-context learning — the AR contrast you explicitly called out
- [Brown et al. 2020 — GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — few-shot *without weight updates*; understand what you're NOT doing and why.
- [Wei et al. 2022 — Emergent Abilities of LLMs](https://arxiv.org/abs/2206.07682)
- [Olsson et al. 2022 — In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) — mechanistic account of where few-shot ability comes from.

### K. Architectures relevant to efficient sequence learning (indirect; ties to your prior Mamba/diffusion work)
- [Gu & Dao 2023 — Mamba](https://arxiv.org/abs/2312.00752)
- [Sun et al. 2024 — Learning to (Learn at Test Time): TTT layers](https://arxiv.org/abs/2407.04620) — learning *during* inference; conceptually adjacent to continual learning.
- [Munkhdalai et al. 2024 — Infini-attention](https://arxiv.org/abs/2404.07143)
- [Graves et al. 2014 — Neural Turing Machines](https://arxiv.org/abs/1410.5401) — external memory; the ancestor of memory-based CL.

---

## Skipped (with reason)
- _(none yet)_
