"""Deployable Local-PC as a drop-in torch.optim.Optimizer.

This is the SAME deployable form used in exp-0004/0008/0009: a
gain-normalised parallel K-level multi-timescale momentum on the gradient
(unit DC gain per level, mean over K), O(1)-in-unroll. On a non-nested
model this is what Local-PC reduces to (its distinctive O(1)-CREDIT only
differs from the hypergradient in the nested/unrolled regime, exp-0005 —
NOT exercised here; see experiment 0010 README, prior R-3).

Drop-in for AdamW in the official SMF HuggingFace Trainer. Integration:
subclass their SparseMemoryTrainer and override create_optimizer (see
INTEGRATION.md). Pure-PyTorch, no extra deps.
"""
import torch
from torch.optim import Optimizer


class LocalPC(Optimizer):
    def __init__(self, params, lr=1e-3, K=4, weight_decay=0.0):
        if lr <= 0:
            raise ValueError(f"invalid lr {lr}")
        betas = [1.0 - 0.5 ** (k + 1) for k in range(K)]  # 0.5,0.75,..,~0.94
        super().__init__(params, dict(lr=lr, K=K, betas=betas,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            K = group["K"]
            betas = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("LocalPC does not support sparse grads")
                if wd != 0.0:
                    g = g.add(p, alpha=wd)
                st = self.state[p]
                if "e" not in st:
                    st["e"] = [torch.zeros_like(p) for _ in range(K)]
                e = st["e"]
                upd = torch.zeros_like(p)
                for k in range(K):
                    b = betas[k]
                    e[k].mul_(b).add_(g, alpha=1.0 - b)   # unit-DC EMA
                    upd.add_(e[k], alpha=1.0 / K)          # mean over levels
                p.add_(upd, alpha=-lr)
        return loss
