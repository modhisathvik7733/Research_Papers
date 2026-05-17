# Integration — apply Local-PC to the official SMF repo (no fork)

Goal: add **one arm** (`smf-localpc`) to the official SMF pipeline
([github.com/prakharg55/SMF-ICML](https://github.com/prakharg55/SMF-ICML))
by swapping only the optimizer in the sparse phase. Everything else
(model, PKM, data, TF-IDF slot selection, eval) stays identical to their
`smf-adam` so the comparison is clean. We do **not** fork their code into
this repo; we patch at runtime.

## Steps (run only when GPU budget is greenlit)

1. Env (real GPU box / Vast.ai):
   ```
   git clone --depth 1 https://github.com/prakharg55/SMF-ICML
   cd SMF-ICML && pip install -r requirements.txt
   cp /path/to/Research_Papers/experiments/0010-smf-localpc-realmodel/local_pc.py .
   ```
2. Subclass their trainer — minimal override, drop in `smf_rebuild/`:
   ```python
   # smf_rebuild/localpc_trainer.py
   from trainer_utils import SparseMemoryTrainer        # their HF Trainer subclass
   from local_pc import LocalPC

   class LocalPCSparseTrainer(SparseMemoryTrainer):
       def create_optimizer(self):
           if self.optimizer is None:
               decay, no_decay = [], []
               for n, p in self.model.named_parameters():
                   if not p.requires_grad:
                       continue
                   (no_decay if p.ndim < 2 else decay).append(p)
               self.optimizer = LocalPC(
                   [{"params": decay,
                     "weight_decay": self.args.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0}],
                   lr=self.args.learning_rate)
           return self.optimizer
   ```
3. In `train_sparse.py`, switch the trainer class by a flag
   (add `--optimizer {adamw,localpc}`; default `adamw` = their exact
   baseline). Only the sparse-phase trainer changes; the dense retrofit
   stage is shared/reused across arms.
4. Run, per pre-registered design (README §Arms/§Escalation):
   - stage 1 dense retrofit once (shared checkpoint, all arms).
   - stage 2 sparse: `--optimizer adamw` (R-1 sanity) and
     `--optimizer localpc` (R-2), ≥3 seeds each; also their `lora` /
     `full-ft` baselines for reference.
   - eval with their `eval_tasks.py` (MedMCQA / WikiText ppl / TriviaQA).
5. Apply the verbatim exp-0002 paired-CRN three-way + effect-size + δ
   rule to (smf-adam − smf-localpc) on retention and new-knowledge
   metrics. Report straight; honest prior = Local-PC inert (≈ adam).

## Free pre-GPU validation (does NOT need budget)

`python -m py_compile local_pc.py` (done) and a CPU/MPS *integration
smoke*: instantiate Qwen-0.5B-Instruct config-only or a tiny stub, wrap a
2-param module, run 3 `LocalPC` steps — proves the optimizer + trainer
hook execute. This validates plumbing only; it is explicitly **not** a
result and is labelled as such if recorded.

## Honest note

This is real engineering on a real 0.5B model: env setup + two-stage
training + eval × arms × seeds = GPU hours and real $ (Vast.ai-class),
competing with the DiffuZamba budget. The integration above is the
de-risked, free part; the runs are the paid part and remain a deliberate
budget decision.
