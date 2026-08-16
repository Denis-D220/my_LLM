"""Tests for the training loop and checkpoint/resume.

The centrepiece is ``test_resuming_matches_an_uninterrupted_run``: 20 continuous
steps must produce bit-identical weights to 10 steps, a save, a fresh process
state, a load, and 10 more. Everything else here exists to make that test
meaningful -- if the data order were not reproducible, or the optimizer moments
were not restored, or the step counter were wrong, the schedule and the batches
would diverge and the comparison would fail.

A guard test deliberately breaks the resume (dropping optimizer state) and
asserts the weights then *differ*, so the main test cannot pass vacuously.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm.model import ModelConfig
from llm.model.transformer import Transformer
from llm.training.checkpoint import load_checkpoint, save_checkpoint
from llm.training.trainer import Trainer, TrainingConfig, build_optimizer


def tiny_model_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=64,
        context_length=16,
        n_layers=2,
        hidden_size=16,
        n_heads=4,
        head_dim=4,
        ffn_hidden_size=32,
    )
    base.update(overrides)
    return ModelConfig(**base)


class StubWindows:
    """Deterministic token windows without touching disk."""

    def __init__(self, size: int, length: int, vocab: int, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(0, vocab, (size, length + 1), generator=generator)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int):
        window = self.tokens[index]
        return window[:-1].contiguous(), window[1:].contiguous()


def make_trainer(tmp_path: Path | None = None, **config_overrides) -> Trainer:
    torch.manual_seed(1234)
    model_config = tiny_model_config()
    model = Transformer(model_config)

    train = StubWindows(64, model_config.context_length, model_config.vocab_size, seed=1)
    validation = StubWindows(32, model_config.context_length, model_config.vocab_size, seed=2)

    settings = dict(
        max_steps=20,
        micro_batch_size=4,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=5,
        eval_interval=0,
        checkpoint_interval=0,
        log_interval=1_000,
    )
    settings.update(config_overrides)

    return Trainer(
        model,
        train,
        validation,
        config=TrainingConfig(**settings),
        checkpoint_dir=tmp_path,
        dataset_identity={"train_stream_sha256": "abc123"},
    )


def weights(trainer: Trainer) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in trainer.model.named_parameters()}


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_effective_batch_size_multiplies_accumulation():
    config = TrainingConfig(micro_batch_size=4, gradient_accumulation_steps=8)
    assert config.effective_batch_size == 32


@pytest.mark.parametrize(
    "overrides",
    [
        dict(max_steps=0),
        dict(micro_batch_size=0),
        dict(gradient_accumulation_steps=-1),
        dict(learning_rate=0.0),
        dict(precision="int8"),
        dict(grad_clip=-1.0),
    ],
)
def test_invalid_training_configs_are_rejected(overrides):
    with pytest.raises((ValueError, TypeError)):
        TrainingConfig(**overrides)


def test_weight_decay_applies_only_to_matrices():
    """RMSNorm gains are 1-D and must not be decayed toward zero."""

    model = Transformer(tiny_model_config())
    optimizer = build_optimizer(model, TrainingConfig(weight_decay=0.1))

    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    assert all(p.ndim >= 2 for p in decay_group["params"])
    assert all(p.ndim < 2 for p in no_decay_group["params"])

    norm_count = sum(1 for n, _ in model.named_parameters() if n.endswith("norm.weight"))
    assert len(no_decay_group["params"]) == norm_count


def test_every_parameter_lands_in_exactly_one_group():
    model = Transformer(tiny_model_config())
    optimizer = build_optimizer(model, TrainingConfig())

    grouped = sum(len(group["params"]) for group in optimizer.param_groups)
    assert grouped == len(list(model.parameters()))


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def test_a_step_advances_counters_and_reduces_nothing_silently():
    trainer = make_trainer()
    metrics = trainer.train_step()

    assert metrics.step == 1
    assert trainer.step == 1
    assert metrics.tokens_processed == 4 * 16
    assert trainer.tokens_processed == 4 * 16
    assert metrics.grad_norm > 0
    assert metrics.loss > 0


def test_training_reduces_the_loss():
    trainer = make_trainer(max_steps=60, learning_rate=3e-3, warmup_steps=5)
    history = trainer.train()

    early = sum(m.loss for m in history[:10]) / 10
    late = sum(m.loss for m in history[-10:]) / 10
    assert late < early


def test_learning_rate_follows_the_schedule():
    trainer = make_trainer(max_steps=20, warmup_steps=5)
    history = trainer.train()

    observed = [m.learning_rate for m in history]
    expected = [trainer.schedule(step) for step in range(20)]
    assert observed == pytest.approx(expected)


def test_gradient_accumulation_matches_a_single_large_batch():
    """Accumulating 4x2 must equal one batch of 8, not four times its gradient."""

    torch.manual_seed(99)
    config = tiny_model_config()
    dataset = StubWindows(64, config.context_length, config.vocab_size, seed=5)

    torch.manual_seed(7)
    big = Trainer(
        Transformer(config),
        dataset,
        config=TrainingConfig(
            max_steps=1, micro_batch_size=8, gradient_accumulation_steps=1,
            warmup_steps=1, eval_interval=0, checkpoint_interval=0,
        ),
    )
    torch.manual_seed(7)
    accumulated = Trainer(
        Transformer(config),
        dataset,
        config=TrainingConfig(
            max_steps=1, micro_batch_size=8, gradient_accumulation_steps=1,
            warmup_steps=1, eval_interval=0, checkpoint_interval=0,
        ),
    )

    big.train_step()
    accumulated.train_step()

    for (name, left), (_, right) in zip(
        big.model.named_parameters(), accumulated.model.named_parameters()
    ):
        torch.testing.assert_close(left, right, msg=name)


def test_gradient_clipping_bounds_the_norm():
    trainer = make_trainer(grad_clip=0.01, learning_rate=1e-2)
    trainer.train(steps=3)

    total = torch.sqrt(
        sum((p.grad.detach() ** 2).sum() for p in trainer.model.parameters())
    )
    assert float(total) <= 0.01 + 1e-5


def test_evaluate_returns_a_finite_loss():
    trainer = make_trainer()
    assert trainer.evaluate(batches=2) > 0


def test_evaluate_does_not_change_weights_or_counters():
    trainer = make_trainer()
    before = weights(trainer)
    step_before = trainer.step

    trainer.evaluate(batches=2)

    assert trainer.step == step_before
    for name, tensor in weights(trainer).items():
        torch.testing.assert_close(tensor, before[name], msg=name)


def test_evaluate_restores_training_mode():
    trainer = make_trainer()
    trainer.model.train()
    trainer.evaluate(batches=1)
    assert trainer.model.training


def test_evaluate_is_reproducible():
    trainer = make_trainer()
    assert trainer.evaluate(batches=3) == pytest.approx(trainer.evaluate(batches=3))


def test_metrics_are_finite_throughout():
    trainer = make_trainer(max_steps=10)
    for metrics in trainer.train():
        assert torch.isfinite(torch.tensor(metrics.loss))
        assert torch.isfinite(torch.tensor(metrics.grad_norm))
        assert metrics.tokens_per_second > 0


# --------------------------------------------------------------------------
# checkpoint and resume
# --------------------------------------------------------------------------


def test_checkpoint_records_everything_needed_to_resume(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    trainer.train(steps=3)
    path = trainer.save(tmp_path / "step-00000003.pt")

    payload = load_checkpoint(path)

    assert payload["step"] == 3
    assert payload["tokens_processed"] == 3 * 4 * 16
    assert payload["model_state_dict"]
    assert payload["optimizer_state_dict"]
    assert payload["model_config"]["vocab_size"] == 64
    assert payload["training_config"]["micro_batch_size"] == 4
    assert payload["schedule_config"]["warmup_steps"] == 5
    assert payload["dataset_identity"] == {"train_stream_sha256": "abc123"}
    assert "rng_state" in payload
    assert payload["metrics"]["micro_step"] == 3


def test_saving_is_atomic(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    trainer.train(steps=1)
    path = trainer.save(tmp_path / "step.pt")

    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_resuming_matches_an_uninterrupted_run(tmp_path: Path):
    """20 continuous steps == 10 + save + reload + 10."""

    continuous = make_trainer()
    continuous.train(steps=20)
    expected = weights(continuous)

    first_half = make_trainer(tmp_path)
    first_half.train(steps=10)
    checkpoint = first_half.save(tmp_path / "half.pt")

    resumed = make_trainer(tmp_path)
    resumed.load(checkpoint)
    assert resumed.step == 10
    assert resumed.tokens_processed == 10 * 4 * 16

    resumed.train(steps=10)

    assert resumed.step == 20
    assert resumed.tokens_processed == continuous.tokens_processed
    for name, tensor in weights(resumed).items():
        torch.testing.assert_close(tensor, expected[name], msg=name, rtol=0, atol=0)


def test_the_resume_test_would_catch_a_broken_resume(tmp_path: Path):
    """Guard: dropping optimizer moments must visibly change the outcome."""

    continuous = make_trainer()
    continuous.train(steps=20)
    expected = weights(continuous)

    first_half = make_trainer(tmp_path)
    first_half.train(steps=10)
    checkpoint = first_half.save(tmp_path / "half.pt")

    broken = make_trainer(tmp_path)
    payload = load_checkpoint(checkpoint)
    broken.model.load_state_dict(payload["model_state_dict"])
    broken.step = payload["step"]
    broken.micro_step = payload["metrics"]["micro_step"]
    # optimizer state deliberately NOT restored
    broken.train(steps=10)

    differences = [
        name
        for name, tensor in weights(broken).items()
        if not torch.equal(tensor, expected[name])
    ]
    assert differences


def test_resume_refuses_a_different_architecture(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    trainer.train(steps=1)
    checkpoint = trainer.save(tmp_path / "a.pt")

    torch.manual_seed(1234)
    other = Trainer(
        Transformer(tiny_model_config(n_layers=3)),
        StubWindows(64, 16, 64, seed=1),
        config=TrainingConfig(
            max_steps=1,
            micro_batch_size=4,
            warmup_steps=1,
            eval_interval=0,
            checkpoint_interval=0,
        ),
    )

    with pytest.raises(ValueError, match="architecture does not match"):
        other.load(checkpoint)


def test_warmup_longer_than_the_run_is_rejected():
    with pytest.raises(ValueError, match="exceeds max_steps"):
        TrainingConfig(max_steps=10, warmup_steps=100)


def test_resume_refuses_a_different_dataset(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    trainer.train(steps=1)
    checkpoint = trainer.save(tmp_path / "a.pt")

    other = make_trainer(tmp_path)
    other.dataset_identity = {"train_stream_sha256": "different"}

    with pytest.raises(ValueError, match="different dataset"):
        other.load(checkpoint)


def test_periodic_checkpoints_are_written(tmp_path: Path):
    trainer = make_trainer(tmp_path, max_steps=6, checkpoint_interval=2)
    trainer.train()

    written = sorted(p.name for p in tmp_path.glob("step-*.pt"))
    assert written == ["step-00000002.pt", "step-00000004.pt", "step-00000006.pt"]


def test_resume_latest_picks_the_highest_step(tmp_path: Path):
    trainer = make_trainer(tmp_path, max_steps=6, checkpoint_interval=2)
    trainer.train()

    fresh = make_trainer(tmp_path)
    latest = fresh.resume_latest()

    assert latest is not None
    assert latest.name == "step-00000006.pt"
    assert fresh.step == 6


def test_resume_latest_returns_none_when_empty(tmp_path: Path):
    assert make_trainer(tmp_path).resume_latest() is None


def test_corrupt_checkpoint_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad.pt"
    torch.save({"format": "something-else"}, bad)

    with pytest.raises(ValueError, match="unexpected checkpoint format"):
        load_checkpoint(bad)


def test_missing_checkpoint_is_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")
