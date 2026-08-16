"""Learning-rate schedule: linear warmup into cosine decay.

Kept as a pure function of the step number rather than a stateful PyTorch
scheduler.  Resuming a run then needs no scheduler state at all: the step
counter is already in the checkpoint, and ``schedule(step)`` reconstructs the
exact learning rate from it.  A stateful scheduler is one more thing that can
be restored incorrectly and produce a silently different run.

Shape::

    lr
     ^
     |      ____
     |     /    ----____
     |    /             ----____
     |   /                      ---___
     |  /                             ---
     | /
     +--------------------------------------> step
       warmup            cosine decay

Warmup exists because the first updates are taken against a model whose
attention patterns are still noise; a large step there can move weights into a
region the run never recovers from.  Cosine decay to a small non-zero floor
lets the model keep making progress late in training without the final steps
undoing earlier ones.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any


@dataclass(frozen=True)
class CosineWithWarmup:
    """Learning rate as a function of optimizer step.

    Parameters
    ----------
    peak_learning_rate:
        Value reached at the end of warmup.
    min_learning_rate:
        Floor held after ``total_steps``.  Must not exceed the peak.
    warmup_steps:
        Steps spent ramping linearly from ``peak/warmup_steps`` to the peak.
        Zero disables warmup entirely.
    total_steps:
        Step at which decay reaches the floor.  Steps beyond this return the
        floor rather than continuing to decay or turning back upward.
    """

    peak_learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    total_steps: int

    def __post_init__(self) -> None:
        for name in ("peak_learning_rate", "min_learning_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        for name in ("warmup_steps", "total_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if self.peak_learning_rate <= 0.0:
            raise ValueError("peak_learning_rate must be > 0")
        if self.min_learning_rate > self.peak_learning_rate:
            raise ValueError(
                f"min_learning_rate ({self.min_learning_rate}) must not exceed "
                f"peak_learning_rate ({self.peak_learning_rate})"
            )
        if self.total_steps < self.warmup_steps:
            raise ValueError(
                f"total_steps ({self.total_steps}) must be >= warmup_steps "
                f"({self.warmup_steps})"
            )

    def __call__(self, step: int) -> float:
        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError(f"step must be an integer, got {type(step).__name__}")
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        # Warmup counts from 1 so the very first step takes a non-zero update;
        # starting at exactly zero wastes a step and hides bugs that only show
        # up once weights begin moving.
        if step < self.warmup_steps:
            return self.peak_learning_rate * (step + 1) / self.warmup_steps

        if step >= self.total_steps:
            return self.min_learning_rate

        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps == 0:
            return self.min_learning_rate

        progress = (step - self.warmup_steps) / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_learning_rate + cosine * (
            self.peak_learning_rate - self.min_learning_rate
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CosineWithWarmup":
        return cls(**payload)
