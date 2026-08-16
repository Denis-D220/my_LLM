"""Tiny-overfit proof: can the architecture memorize at all?

Every other test in this suite checks that a component computes what it claims.
None of them establish that the assembled model can *learn* -- an architecture
can be individually correct at every layer and still be unable to fit anything,
because learning depends on gradient magnitudes, initialization scale, residual
wiring and the loss orientation all being right together.

The check is deliberately crude: take a handful of fixed sequences and train on
exactly those, repeatedly. A model with 32.7M parameters memorizing 8 short
sequences is not an achievement; failing to is proof of a bug. Cross-entropy
must fall from roughly ``ln(vocab_size)`` to near zero.

This module uses a small architecture so it stays a fast, always-on regression
guard. ``scripts/tiny_overfit.py`` runs the same experiment at the real frozen
scale.
"""

from __future__ import annotations

import math

import pytest
import torch

from llm.model import ModelConfig
from llm.model.transformer import Transformer, causal_lm_loss


def overfit_config(**overrides) -> ModelConfig:
    """Small, but structurally the real architecture: 2 blocks, RoPE, SwiGLU."""

    base = dict(
        vocab_size=128,
        context_length=32,
        n_layers=2,
        hidden_size=32,
        n_heads=4,
        head_dim=8,
        ffn_hidden_size=96,
    )
    base.update(overrides)
    return ModelConfig(**base)


def fixed_batch(config: ModelConfig, *, examples: int, length: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        0, config.vocab_size, (examples, length + 1), generator=generator
    )
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def train_to_memorize(
    model: Transformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    history: list[float] = []

    for _ in range(steps):
        loss = causal_lm_loss(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return history


@pytest.fixture(scope="module")
def overfit_run():
    """One shared 400-step run; several tests inspect different aspects."""

    torch.manual_seed(20260815)
    config = overfit_config()
    model = Transformer(config)
    inputs, targets = fixed_batch(config, examples=8, length=16, seed=11)

    history = train_to_memorize(model, inputs, targets, steps=400, lr=3e-3)
    return config, model, inputs, targets, history


def test_initial_loss_is_near_uniform_cross_entropy(overfit_run):
    config, _, _, _, history = overfit_run
    uniform = math.log(config.vocab_size)

    assert history[0] == pytest.approx(uniform, abs=0.6), (history[0], uniform)


def test_the_model_memorizes_the_batch(overfit_run):
    """The decisive assertion: loss must collapse toward zero."""

    _, _, _, _, history = overfit_run

    assert history[-1] < 0.05, history[-1]
    assert history[-1] < history[0] / 50


def test_the_loss_decreases_monotonically_in_aggregate(overfit_run):
    """Step-to-step noise is fine; the trend must be down."""

    _, _, _, _, history = overfit_run
    window = len(history) // 8
    chunks = [
        sum(history[i : i + window]) / window
        for i in range(0, len(history) - window + 1, window)
    ]

    for earlier, later in zip(chunks, chunks[1:]):
        assert later < earlier, chunks


def test_every_loss_value_is_finite(overfit_run):
    _, _, _, _, history = overfit_run
    assert all(math.isfinite(value) for value in history)


def test_the_memorized_batch_is_predicted_correctly(overfit_run):
    """Loss near zero should mean argmax actually recovers the targets."""

    _, model, inputs, targets, _ = overfit_run

    with torch.no_grad():
        predictions = model(inputs).argmax(dim=-1)

    accuracy = (predictions == targets).float().mean().item()
    assert accuracy > 0.99, accuracy


def test_memorization_does_not_generalize_to_unseen_tokens(overfit_run):
    """Guard: a model that scores ~0 on everything has collapsed, not learned."""

    config, model, _, _, _ = overfit_run
    other_inputs, other_targets = fixed_batch(config, examples=8, length=16, seed=99)

    with torch.no_grad():
        loss = causal_lm_loss(model(other_inputs), other_targets).item()

    assert loss > 1.0, loss


def test_an_untrained_model_does_not_already_fit_the_batch():
    """Guard: the overfit result must come from training, not initialization."""

    torch.manual_seed(4)
    config = overfit_config()
    model = Transformer(config)
    inputs, targets = fixed_batch(config, examples=8, length=16, seed=11)

    with torch.no_grad():
        loss = causal_lm_loss(model(inputs), targets).item()

    assert loss > math.log(config.vocab_size) - 0.6


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_memorization_is_robust_across_seeds(seed):
    """Not a lucky initialization."""

    torch.manual_seed(seed)
    config = overfit_config()
    model = Transformer(config)
    inputs, targets = fixed_batch(config, examples=4, length=12, seed=seed + 100)

    history = train_to_memorize(model, inputs, targets, steps=250, lr=3e-3)
    assert history[-1] < 0.1, (seed, history[-1])
