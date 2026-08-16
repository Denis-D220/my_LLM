"""Tests for the learning-rate schedule."""

from __future__ import annotations

import math

import pytest

from llm.training.schedule import CosineWithWarmup


@pytest.fixture
def schedule() -> CosineWithWarmup:
    return CosineWithWarmup(
        peak_learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=100,
        total_steps=1_000,
    )


def test_warmup_starts_above_zero(schedule):
    """A zero first step wastes an update and hides early bugs."""

    assert schedule(0) > 0
    assert schedule(0) == pytest.approx(1e-3 / 100)


def test_warmup_is_linear(schedule):
    values = [schedule(step) for step in range(100)]
    deltas = [b - a for a, b in zip(values, values[1:])]

    for delta in deltas:
        assert delta == pytest.approx(deltas[0])


def test_peak_is_reached_at_the_end_of_warmup(schedule):
    assert schedule(99) == pytest.approx(1e-3)


def test_decay_starts_from_the_peak_without_a_discontinuity(schedule):
    """cos(0) == 1, so the first decay step sits exactly at the peak."""

    assert schedule(100) == pytest.approx(schedule(99)) == pytest.approx(1e-3)
    assert schedule(101) < schedule(100)


def test_decay_is_monotonically_decreasing(schedule):
    values = [schedule(step) for step in range(100, 1_000)]
    for earlier, later in zip(values, values[1:]):
        assert later <= earlier


def test_midpoint_of_decay_is_the_average(schedule):
    """cos(pi/2) == 0, so halfway through decay sits at the mean."""

    midpoint = schedule(100 + (1_000 - 100) // 2)
    assert midpoint == pytest.approx((1e-3 + 1e-4) / 2, rel=1e-3)


def test_floor_is_reached_and_held(schedule):
    assert schedule(1_000) == pytest.approx(1e-4)
    assert schedule(5_000) == pytest.approx(1e-4)
    assert schedule(10**9) == pytest.approx(1e-4)


def test_never_exceeds_the_peak(schedule):
    for step in range(0, 2_000, 7):
        assert 0.0 < schedule(step) <= 1e-3 + 1e-12


def test_the_floor_applies_to_decay_not_to_warmup(schedule):
    """min_learning_rate bounds the decay phase; warmup ramps up from near zero."""

    assert schedule(0) < 1e-4
    for step in range(100, 2_000, 7):
        assert schedule(step) >= 1e-4 - 1e-12


def test_zero_warmup_starts_at_the_peak():
    schedule = CosineWithWarmup(1e-3, 1e-4, warmup_steps=0, total_steps=100)
    assert schedule(0) == pytest.approx(1e-3)


def test_is_a_pure_function_of_the_step(schedule):
    """No hidden state: resume needs only the step counter."""

    first = [schedule(s) for s in range(0, 1_000, 13)]
    _ = [schedule(s) for s in range(1_000)]
    second = [schedule(s) for s in range(0, 1_000, 13)]
    assert first == second


def test_round_trips_through_a_dict(schedule):
    assert CosineWithWarmup.from_dict(schedule.to_dict()) == schedule


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(peak_learning_rate=0.0, min_learning_rate=0.0, warmup_steps=1, total_steps=2), "peak"),
        (dict(peak_learning_rate=-1.0, min_learning_rate=0.0, warmup_steps=1, total_steps=2), ">= 0"),
        (dict(peak_learning_rate=1e-3, min_learning_rate=1e-2, warmup_steps=1, total_steps=2), "must not exceed"),
        (dict(peak_learning_rate=1e-3, min_learning_rate=0.0, warmup_steps=10, total_steps=5), ">= warmup_steps"),
        (dict(peak_learning_rate=1e-3, min_learning_rate=0.0, warmup_steps=-1, total_steps=5), ">= 0"),
    ],
)
def test_invalid_configurations_are_rejected(kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        CosineWithWarmup(**kwargs)


@pytest.mark.parametrize("step", [-1, 1.5, True, None])
def test_invalid_steps_are_rejected(schedule, step):
    with pytest.raises((TypeError, ValueError)):
        schedule(step)


def test_matches_the_closed_form(schedule):
    """Independent transcription of the cosine formula."""

    for step in (100, 250, 500, 750, 999):
        progress = (step - 100) / (1_000 - 100)
        expected = 1e-4 + 0.5 * (1 + math.cos(math.pi * progress)) * (1e-3 - 1e-4)
        assert schedule(step) == pytest.approx(expected)
