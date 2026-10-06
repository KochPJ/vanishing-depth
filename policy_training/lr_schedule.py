"""
Cosine annealing with linear warmup.

Schedule shape
──────────────
  step ∈ [0, warmup_steps)      : lr = base_lr * step / warmup_steps
  step ∈ [warmup_steps, total)  : lr = base_lr * (min_ratio + (1 - min_ratio)
                                              * 0.5 * (1 + cos(π * progress)))

Works with any number of param groups; multiplier is applied uniformly.
"""
from __future__ import annotations
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer:     Optimizer,
    warmup_steps:  int,
    total_steps:   int,
    min_lr_ratio:  float = 0.0,   # floor as a fraction of base_lr
    last_epoch:    int   = -1,
) -> LambdaLR:
    """
    Args
    ────
    warmup_steps  : number of linear warm-up steps
    total_steps   : total training steps (epochs × steps_per_epoch)
    min_lr_ratio  : lr at end of cosine = base_lr * min_lr_ratio
                    Pass args.min_lr / args.lr  if you have explicit values.
    last_epoch    : pass the last completed step when resuming from a checkpoint
                    so the scheduler fast-forwards correctly.
    """
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)