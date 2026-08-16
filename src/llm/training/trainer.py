"""The pretraining loop.

Deliberately small.  Everything that could be stateful has been pushed
elsewhere so the loop itself has almost nothing to get wrong on resume:

* the learning rate is a pure function of the step (:mod:`llm.training.schedule`)
* the batch order is a pure function of the step (:mod:`llm.training.data`)
* saving and restoring is one module (:mod:`llm.training.checkpoint`)

What remains here is: fetch a batch, forward, loss, backward, accumulate, clip,
step, log.

Optimizer parameter groups
--------------------------
Weight decay applies to matrices and not to vectors.  RMSNorm scales are the
only 1-D parameters in this model, and decaying them pulls the normalization
gain toward zero for no benefit -- it is a regularizer aimed at redundant
directions in a weight matrix, and a per-channel gain has none.  The tied
embedding is 2-D and does receive decay, which is the conventional choice.

Gradient accumulation
---------------------
Each micro-batch's loss is divided by the accumulation count before backward,
so the accumulated gradient equals the gradient of the mean loss over the full
effective batch rather than its sum.  Without that division, the effective
learning rate silently scales with the accumulation factor and a configuration
change that should be numerically neutral is not.

Precision
---------
FP32 is the default.  BF16 runs under ``torch.autocast`` with no gradient
scaler, which BF16 does not need because it keeps FP32's exponent range.
Parameters and optimizer state stay FP32 in both modes; only the forward and
backward math is reduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any, Iterator

import torch
from torch import nn

from llm.model.transformer import causal_lm_loss
from llm.training.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    restore_into,
    save_checkpoint,
)
from llm.training.data import DeterministicBatchSampler, collate_windows
from llm.training.schedule import CosineWithWarmup


PRECISIONS = ("fp32", "bf16", "fp16")


@dataclass(frozen=True)
class TrainingConfig:
    """Everything that defines a training run apart from the model itself."""

    max_steps: int = 1_000

    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 1

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    grad_clip: float = 1.0

    eval_interval: int = 200
    eval_batches: int = 20
    checkpoint_interval: int = 500
    log_interval: int = 10

    seed: int = 2026
    precision: str = "fp32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "eval_batches",
            "log_interval",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        for name in ("warmup_steps", "eval_interval", "checkpoint_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.min_learning_rate < 0.0:
            raise ValueError("min_learning_rate must be >= 0")
        if self.grad_clip < 0.0:
            raise ValueError("grad_clip must be >= 0")
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"precision must be one of {PRECISIONS}, got {self.precision!r}"
            )
        # Caught here rather than inside the schedule so the message names the
        # two fields the caller actually set. A warmup longer than the run means
        # the peak learning rate is never reached.
        if self.warmup_steps > self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) exceeds max_steps "
                f"({self.max_steps}); the run would end before warmup finished"
            )

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepMetrics:
    """One optimizer step's observable behaviour."""

    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    tokens_processed: int
    tokens_per_second: float
    seconds: float
    validation_loss: float | None = None
    peak_memory_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_optimizer(
    model: nn.Module, config: TrainingConfig
) -> torch.optim.AdamW:
    """AdamW with decay on matrices only."""

    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )


class Trainer:
    """Drive a :class:`~llm.model.transformer.Transformer` over token windows."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        validation_dataset=None,
        *,
        config: TrainingConfig | None = None,
        checkpoint_dir: str | Path | None = None,
        dataset_identity: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        self.optimizer = build_optimizer(self.model, self.config)

        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.config.precision == "fp16"
                and self.device.type == "cuda"
            ),
        )

        self.schedule = CosineWithWarmup(
            peak_learning_rate=self.config.learning_rate,
            min_learning_rate=self.config.min_learning_rate,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.max_steps,
        )

        self.sampler = DeterministicBatchSampler(
            len(train_dataset),
            self.config.micro_batch_size,
            seed=self.config.seed,
        )
        self.validation_sampler = (
            DeterministicBatchSampler(
                len(validation_dataset),
                self.config.micro_batch_size,
                seed=self.config.seed + 1,
            )
            if validation_dataset is not None
            else None
        )

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.dataset_identity = dict(dataset_identity or {})

        self.step = 0
        self.tokens_processed = 0
        self.micro_step = 0
        self.history: list[StepMetrics] = []

    # ------------------------------------------------------------------
    # precision
    # ------------------------------------------------------------------

    def _autocast(self):
        if self.config.precision == "fp32":
            return torch.autocast(device_type=self.device.type, enabled=False)
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------

    def _micro_batch(self, micro_step: int):
        indices = self.sampler.indices_for_step(micro_step)
        return collate_windows(self.train_dataset, indices, self.device)

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, batches: int | None = None) -> float:
        """Mean validation loss over a fixed, reproducible set of batches."""

        if self.validation_dataset is None or self.validation_sampler is None:
            raise RuntimeError("no validation dataset was provided")

        count = batches if batches is not None else self.config.eval_batches
        count = min(count, len(self.validation_sampler))

        was_training = self.model.training
        self.model.eval()
        total = 0.0

        try:
            for index in range(count):
                inputs, targets = collate_windows(
                    self.validation_dataset,
                    self.validation_sampler.indices_for_step(index),
                    self.device,
                )
                with self._autocast():
                    loss = causal_lm_loss(self.model(inputs), targets)
                total += float(loss)
        finally:
            if was_training:
                self.model.train()

        return total / count if count else float("nan")

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def train_step(self) -> StepMetrics:
        """One optimizer step, including any gradient accumulation."""

        self.model.train()
        started = time.perf_counter()

        learning_rate = self.schedule(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        self.optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        tokens = 0
        for _ in range(self.config.gradient_accumulation_steps):
            inputs, targets = self._micro_batch(self.micro_step)
            self.micro_step += 1

            with self._autocast():
                loss = causal_lm_loss(self.model(inputs), targets)

            # Mean over the effective batch, not the sum.
            loss_for_backward = (
                loss / self.config.gradient_accumulation_steps
            )
            self.grad_scaler.scale(loss_for_backward).backward()

            accumulated_loss += (
                float(loss.detach()) / self.config.gradient_accumulation_steps
            )
            tokens += inputs.numel()

        # Unscale once after all micro-batches have accumulated.
        self.grad_scaler.unscale_(self.optimizer)

        if self.config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float("inf")
            )

        scale_before = self.grad_scaler.get_scale()

        # Performs optimizer.step() only when gradients are finite.
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        scale_after = self.grad_scaler.get_scale()

        optimizer_step_skipped = (
            self.grad_scaler.is_enabled()
            and scale_after < scale_before
        )

        if optimizer_step_skipped:
            print(
                "WARNING: FP16 overflow detected; optimizer update skipped; "
                f"GradScaler reduced scale "
                f"{scale_before:g} -> {scale_after:g}",
                flush=True,
            )
        else:
            # step counts successful optimizer updates.
            self.step += 1

        # Tokens were still processed even if the update was skipped.
        self.tokens_processed += tokens
        seconds = time.perf_counter() - started

        peak_memory = (
            torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else None
        )

        return StepMetrics(
            step=self.step,
            loss=accumulated_loss,
            learning_rate=learning_rate,
            grad_norm=float(grad_norm),
            tokens_processed=self.tokens_processed,
            tokens_per_second=tokens / seconds if seconds > 0 else float("inf"),
            seconds=seconds,
            peak_memory_bytes=peak_memory,
        )

    def train(
        self,
        steps: int | None = None,
        *,
        progress: bool = False,
    ) -> list[StepMetrics]:
        """Run until ``max_steps``, or for ``steps`` more steps."""

        target = self.config.max_steps if steps is None else self.step + steps
        produced: list[StepMetrics] = []

        while self.step < target:
            metrics = self.train_step()

            if (
                self.validation_dataset is not None
                and self.config.eval_interval > 0
                and self.step % self.config.eval_interval == 0
            ):
                metrics.validation_loss = self.evaluate()

            self.history.append(metrics)
            produced.append(metrics)

            if progress and (
                self.step % self.config.log_interval == 0 or self.step == target
            ):
                self._log(metrics)

            if (
                self.checkpoint_dir is not None
                and self.config.checkpoint_interval > 0
                and self.step % self.config.checkpoint_interval == 0
            ):
                self.save(self.checkpoint_dir / f"step-{self.step:08d}.pt")

        return produced

    def _log(self, metrics: StepMetrics) -> None:
        validation = (
            f"{metrics.validation_loss:>10.4f}"
            if metrics.validation_loss is not None
            else f"{'-':>10}"
        )
        print(
            f"{metrics.step:>8}"
            f"{metrics.loss:>11.4f}"
            f"{validation}"
            f"{metrics.learning_rate:>12.2e}"
            f"{metrics.grad_norm:>11.3f}"
            f"{metrics.tokens_per_second:>12,.0f}"
            f"{metrics.tokens_processed:>16,}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        model_config = (
            self.model.config.to_dict()
            if hasattr(self.model, "config") and hasattr(self.model.config, "to_dict")
            else {}
        )
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            grad_scaler=self.grad_scaler,
            step=self.step,
            tokens_processed=self.tokens_processed,
            model_config=model_config,
            training_config=self.config.to_dict(),
            schedule_config=self.schedule.to_dict(),
            dataset_identity=self.dataset_identity,
            metrics={
                "last_loss": self.history[-1].loss if self.history else None,
                "micro_step": self.micro_step,
            },
        )

    def load(self, path: str | Path, *, strict_dataset: bool = True) -> None:
        payload = load_checkpoint(path, map_location=self.device)
        model_config = (
            self.model.config.to_dict()
            if hasattr(self.model, "config") and hasattr(self.model.config, "to_dict")
            else None
        )

        step, tokens = restore_into(
            payload,
            model=self.model,
            optimizer=self.optimizer,
            grad_scaler=self.grad_scaler,
            model_config=model_config,
            dataset_identity=self.dataset_identity or None,
            strict_dataset=strict_dataset,
        )

        self.step = step
        self.tokens_processed = tokens
        # micro_step is what indexes the batch order; recovering it is what
        # makes the resumed run see the examples it would otherwise have seen.
        recorded = payload.get("metrics", {}).get("micro_step")
        self.micro_step = (
            int(recorded)
            if recorded is not None
            else step * self.config.gradient_accumulation_steps
        )

    def resume_latest(self, directory: str | Path | None = None) -> Path | None:
        target = Path(directory) if directory else self.checkpoint_dir
        if target is None:
            return None
        latest = find_latest_checkpoint(target)
        if latest is not None:
            self.load(latest)
        return latest

    @staticmethod
    def log_header() -> str:
        return (
            f"{'step':>8}{'loss':>11}{'val':>10}{'lr':>12}"
            f"{'grad':>11}{'tok/s':>12}{'tokens':>16}"
        )
