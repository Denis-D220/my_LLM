"""Deterministic batch ordering for resumable training.

The example order a run sees must be reconstructible from the step number
alone.  If it is not, resuming from a checkpoint either replays examples the
model has already seen or skips others, and neither shows up in the loss curve
clearly enough to notice.

The usual fix is to serialize sampler or DataLoader state into the checkpoint.
This module takes the other route: the order is a *pure function* of
``(seed, dataset_size, batch_size, step)``, so nothing about it needs saving.
Resume restores a step counter and the order follows.

    epoch          = step // steps_per_epoch
    position       = step %  steps_per_epoch
    permutation    = randperm(dataset_size, generator=seed + epoch)
    batch indices  = permutation[position * batch : (position + 1) * batch]

A fresh permutation per epoch means examples are reshuffled between passes,
while any given step always maps to the same examples no matter how many times
the process restarts.  The trailing partial batch of each epoch is dropped so
every step sees exactly ``batch_size`` examples -- an uneven final batch would
make the loss at one step per epoch quietly incomparable to its neighbours.
"""

from __future__ import annotations

import torch


class DeterministicBatchSampler:
    """Map a global step to the example indices that step must train on."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        seed: int = 2026,
    ) -> None:
        if not isinstance(dataset_size, int) or isinstance(dataset_size, bool):
            raise TypeError("dataset_size must be an integer")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if dataset_size <= 0:
            raise ValueError(f"dataset_size must be > 0, got {dataset_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if batch_size > dataset_size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds dataset_size ({dataset_size})"
            )

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.steps_per_epoch = dataset_size // batch_size

        # Cached because a step usually asks for the same epoch as the last one,
        # and randperm over 529,227 windows is not free.
        self._cached_epoch: int | None = None
        self._cached_permutation: torch.Tensor | None = None

    def permutation(self, epoch: int) -> torch.Tensor:
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise TypeError("epoch must be an integer")
        if epoch < 0:
            raise ValueError("epoch must be >= 0")

        if self._cached_epoch == epoch and self._cached_permutation is not None:
            return self._cached_permutation

        generator = torch.Generator().manual_seed(self.seed + epoch)
        permutation = torch.randperm(self.dataset_size, generator=generator)

        self._cached_epoch = epoch
        self._cached_permutation = permutation
        return permutation

    def epoch_for_step(self, step: int) -> int:
        return step // self.steps_per_epoch

    def indices_for_step(self, step: int) -> list[int]:
        """Return the example indices for a global optimizer micro-step."""

        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError("step must be an integer")
        if step < 0:
            raise ValueError("step must be >= 0")

        epoch = step // self.steps_per_epoch
        position = step % self.steps_per_epoch
        permutation = self.permutation(epoch)

        start = position * self.batch_size
        return permutation[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        return self.steps_per_epoch


def collate_windows(
    dataset, indices: list[int], device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ``(input_ids, target_ids)`` pairs into a batch."""

    inputs = []
    targets = []
    for index in indices:
        window_inputs, window_targets = dataset[index]
        inputs.append(window_inputs)
        targets.append(window_targets)

    batch_inputs = torch.stack(inputs)
    batch_targets = torch.stack(targets)

    if device is not None:
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)

    return batch_inputs, batch_targets
