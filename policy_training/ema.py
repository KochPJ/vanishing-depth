"""
EMA of TRAINABLE parameters only, running on the training device.
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class EMAModel:
    """
    Exponential Moving Average over trainable (requires_grad=True) parameters only.

    Design
    ──────
    • Shadow lives on the SAME device as the model (no CPU round-trips).
    • Only parameters with requires_grad=True are tracked and updated.
    • copy_to() ONLY overwrites trainable params; frozen params and all
      buffers (action_min, action_max, diffusion schedule …) are left untouched.

    This solves the two EMA bugs:
      1. Tracking frozen params was wasting memory and polluting copy_to.
      2. copy_to with strict=True was overwriting buffers with EMA-smoothed
         (but effectively random for early training) values.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        raw = _unwrap(model)
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in raw.named_parameters():
            if param.requires_grad:
                # clone onto the SAME device as the param (GPU)
                self.shadow[name] = param.data.detach().clone()

        if not self.shadow:
            raise RuntimeError(
                "[EMAModel] No trainable parameters found. "
                "Call build_optimizer() (which sets requires_grad) BEFORE EMAModel()."
            )
        n      = len(self.shadow)
        elems  = sum(v.numel() for v in self.shadow.values())
        print(f"[EMAModel] Tracking {n} trainable tensors "
              f"({elems / 1e6:.2f} M params, decay={decay})")

    # ── core update ────────────────────────────────────────────────────────
    @torch.no_grad()
    def step(self, model: nn.Module) -> None:
        """
        shadow ← decay · shadow + (1 − decay) · param
        Called once per optimizer step, after optimizer.step().
        """
        raw = _unwrap(model)
        for name, param in raw.named_parameters():
            if name not in self.shadow:
                continue                            # frozen → skip
            s = self.shadow[name]
            # lazy device / dtype migration (handles first step after load_state_dict)
            if s.device != param.device or s.dtype != param.dtype:
                self.shadow[name] = s = s.to(device=param.device, dtype=param.dtype)
            # lerp_: in-place  s = s + (1-decay) * (param - s)
            s.lerp_(param.data, 1.0 - self.decay)

    # ── evaluation ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """
        Copy EMA shadow values into the TRAINABLE params of *model* only.
        Frozen params, action_min/max buffers, diffusion schedule buffers
        are ALL left untouched — they never need to be EMA'd.
        """
        raw = _unwrap(model)
        for name, param in raw.named_parameters():
            if name in self.shadow:
                param.data.copy_(
                    self.shadow[name].to(device=param.device, dtype=param.dtype)
                )
        return raw

    # ── serialisation ───────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {
            "decay":  self.decay,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self.decay  = float(sd["decay"])
        # keep on CPU until first step() call, which migrates lazily
        self.shadow = {k: v.clone() for k, v in sd['shadow'].items()}