"""
FixedStepsSampler
─────────────────
Yields exactly `steps_per_epoch * batch_size` indices per epoch on every
DDP rank, so the DataLoader always produces exactly `steps_per_epoch` batches
regardless of dataset size or batch size.

Why this matters
────────────────
With the default sampler the number of steps depends on
    steps = ceil( len(dataset) / (batch_size * world_size) )
which changes whenever you switch task, batch size, or GPU count.
This sampler fixes steps_per_epoch as a first-class hyper-parameter,
making loss curves directly comparable across ablations.

Dataset size vs. steps
──────────────────────
  dataset > needed : random subset drawn each epoch (no repeats within epoch)
  dataset < needed : cycles through shuffled dataset as many times as needed
                     (sampling with replacement across repetitions)

DDP behaviour
─────────────
Each rank draws *independent* random indices using a rank-specific seed.
All ranks always emit the same __len__, so DDP stays synchronised without
any extra communication.  Gradients are all-reduced after every step as
usual, so the effective batch size is batch_size * world_size.
"""
from __future__ import annotations

import math

import torch
from torch.utils.data import Sampler


class FixedStepsSampler(Sampler):
    """
    Parameters
    ----------
    dataset_size    : len(dataset)
    steps_per_epoch : target number of DataLoader batches per epoch
    batch_size      : per-rank batch size passed to DataLoader
    rank            : this rank's index  (0 for single-GPU / non-DDP)
    world_size      : total DDP ranks    (1 for single-GPU / non-DDP)
    shuffle         : randomise order each epoch
    seed            : base RNG seed; epoch + rank offsets are added to it
    """

    def __init__(
        self,
        dataset_size:    int,
        steps_per_epoch: int,
        batch_size:      int,
        rank:            int  = 0,
        world_size:      int  = 1,
        shuffle:         bool = True,
        seed:            int  = 0,
    ) -> None:
        if dataset_size    <= 0: raise ValueError(f"dataset_size={dataset_size}")
        if steps_per_epoch <= 0: raise ValueError(f"steps_per_epoch={steps_per_epoch}")
        if batch_size      <= 0: raise ValueError(f"batch_size={batch_size}")

        self.dataset_size    = dataset_size
        self.steps_per_epoch = steps_per_epoch
        self.batch_size      = batch_size
        self.rank            = rank
        self.world_size      = world_size
        self.shuffle         = shuffle
        self.seed            = seed
        self.epoch           = 0

        # Indices this rank emits per epoch; divisible by batch_size by construction
        self._num_samples = steps_per_epoch * batch_size

    # ── mirrors DistributedSampler API ────────────────────────────────────
    def set_epoch(self, epoch: int) -> None:
        """Call at the start of every epoch (same as DistributedSampler)."""
        self.epoch = epoch

    # ── index generation ──────────────────────────────────────────────────
    def __iter__(self):
        # Independent RNG per (rank, epoch) → each rank sees different data
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 65_537 + self.rank * 31_337)

        needed  = self._num_samples
        indices: list[int] = []

        # Fill by cycling (shuffled) full-dataset permutations
        while len(indices) < needed:
            if self.shuffle:
                perm = torch.randperm(self.dataset_size, generator=g).tolist()
            else:
                perm = list(range(self.dataset_size))
            indices.extend(perm)

        return iter(indices[:needed])   # exact count, divisible by batch_size

    def __len__(self) -> int:
        return self._num_samples