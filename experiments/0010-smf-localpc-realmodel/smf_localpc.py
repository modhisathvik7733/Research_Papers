"""Runtime wrapper: run the official SMF sparse stage with Local-PC, no fork.

Usage (cwd = SMF-ICML repo root, set by run_0010.sh):
  python smf_localpc.py --optimizer adamw   -- <train_sparse args...>
  python smf_localpc.py --optimizer localpc -- <train_sparse args...>

`adamw`  = the official baseline, unmodified (their TrainingArguments.optim).
`localpc`= monkeypatch SparseMemoryTrainer.create_optimizer to build the
           deployable Local-PC (local_pc.LocalPC) over trainable params.
Everything else (model, PKM, data, TF-IDF slot selection, eval) is their
code, identical across arms — the optimiser is the single variable.
"""
import sys


def _patch_localpc():
    from smf_rebuild.trainer_utils import SparseMemoryTrainer
    from local_pc import LocalPC

    def create_optimizer(self):
        if self.optimizer is None:
            decay, no_decay = [], []
            for _, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim < 2 else decay).append(p)
            wd = getattr(self.args, "weight_decay", 0.0)
            self.optimizer = LocalPC(
                [{"params": decay, "weight_decay": wd},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=self.args.learning_rate)
            n = sum(p.numel() for g in self.optimizer.param_groups
                    for p in g["params"])
            print(f"[smf_localpc] Local-PC optimiser active over "
                  f"{n:,} trainable params (lr={self.args.learning_rate})")
        return self.optimizer

    SparseMemoryTrainer.create_optimizer = create_optimizer
    print("[smf_localpc] PATCHED SparseMemoryTrainer.create_optimizer "
          "-> LocalPC")


def main():
    argv = sys.argv[1:]
    if "--optimizer" not in argv:
        raise SystemExit("need --optimizer {adamw,localpc} -- <args>")
    i = argv.index("--optimizer")
    opt = argv[i + 1]
    rest = argv[:i] + argv[i + 2:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    if opt == "localpc":
        _patch_localpc()
    elif opt != "adamw":
        raise SystemExit(f"unknown --optimizer {opt}")
    # hand the remaining args to their entrypoint via argv
    sys.argv = ["train_sparse"] + rest
    import smf_rebuild.train_sparse as t
    t.main()


if __name__ == "__main__":
    main()
