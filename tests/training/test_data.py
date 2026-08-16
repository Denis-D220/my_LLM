"""Tests for deterministic batch ordering.

Resume correctness rests entirely on this being a pure function of the step, so
these tests attack that property from several directions: reproducibility
across instances, independence from access order, coverage of every example
within an epoch, and reshuffling between epochs.
"""

from __future__ import annotations

import pytest
import torch

from llm.training.data import DeterministicBatchSampler, collate_windows


class StubDataset:
    """Windows whose contents encode their own index, so batches are checkable."""

    def __init__(self, size: int, length: int = 4) -> None:
        self.size = size
        self.length = length

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        base = torch.full((self.length,), index, dtype=torch.long)
        return base, base + 1


@pytest.fixture
def sampler() -> DeterministicBatchSampler:
    return DeterministicBatchSampler(100, 8, seed=2026)


def test_steps_per_epoch_drops_the_partial_batch(sampler):
    assert sampler.steps_per_epoch == 12  # 100 // 8
    assert len(sampler) == 12


def test_every_batch_is_full(sampler):
    for step in range(24):
        assert len(sampler.indices_for_step(step)) == 8


def test_indices_are_reproducible_across_instances():
    a = DeterministicBatchSampler(100, 8, seed=7)
    b = DeterministicBatchSampler(100, 8, seed=7)

    for step in (0, 1, 5, 11, 12, 40, 137):
        assert a.indices_for_step(step) == b.indices_for_step(step)


def test_indices_do_not_depend_on_access_order():
    """The crux of resume: step 50 is the same whether or not 0..49 ran first."""

    fresh = DeterministicBatchSampler(100, 8, seed=7)
    expected = fresh.indices_for_step(50)

    sequential = DeterministicBatchSampler(100, 8, seed=7)
    for step in range(51):
        actual = sequential.indices_for_step(step)

    assert actual == expected


def test_an_epoch_covers_every_example_at_most_once(sampler):
    seen: list[int] = []
    for step in range(sampler.steps_per_epoch):
        seen.extend(sampler.indices_for_step(step))

    assert len(seen) == len(set(seen))
    assert len(seen) == 96  # 12 full batches, 4 examples dropped
    assert set(seen) <= set(range(100))


def test_epochs_reshuffle(sampler):
    first = [i for s in range(sampler.steps_per_epoch) for i in sampler.indices_for_step(s)]
    second = [
        i
        for s in range(sampler.steps_per_epoch, 2 * sampler.steps_per_epoch)
        for i in sampler.indices_for_step(s)
    ]

    assert first != second
    assert set(first) <= set(range(100))
    assert set(second) <= set(range(100))


def test_different_seeds_give_different_orders():
    a = DeterministicBatchSampler(100, 8, seed=1)
    b = DeterministicBatchSampler(100, 8, seed=2)

    assert a.indices_for_step(0) != b.indices_for_step(0)


def test_epoch_for_step(sampler):
    assert sampler.epoch_for_step(0) == 0
    assert sampler.epoch_for_step(11) == 0
    assert sampler.epoch_for_step(12) == 1
    assert sampler.epoch_for_step(25) == 2


def test_permutation_is_cached_but_correct(sampler):
    first = sampler.permutation(3).clone()
    sampler.permutation(4)
    second = sampler.permutation(3)

    torch.testing.assert_close(first, second)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dataset_size=0, batch_size=4),
        dict(dataset_size=10, batch_size=0),
        dict(dataset_size=4, batch_size=8),
        dict(dataset_size=-1, batch_size=1),
    ],
)
def test_invalid_configurations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DeterministicBatchSampler(**kwargs)


@pytest.mark.parametrize("step", [-1, 1.5, True, None])
def test_invalid_steps_are_rejected(sampler, step):
    with pytest.raises((TypeError, ValueError)):
        sampler.indices_for_step(step)


# --------------------------------------------------------------------------
# collation
# --------------------------------------------------------------------------


def test_collate_stacks_in_index_order():
    dataset = StubDataset(20)
    inputs, targets = collate_windows(dataset, [3, 7, 1])

    assert inputs.shape == (3, 4)
    assert targets.shape == (3, 4)
    assert inputs[:, 0].tolist() == [3, 7, 1]
    torch.testing.assert_close(targets, inputs + 1)


def test_collate_preserves_dtype():
    dataset = StubDataset(20)
    inputs, _ = collate_windows(dataset, [0, 1])
    assert inputs.dtype == torch.long


def test_collate_respects_the_sampler():
    dataset = StubDataset(64)
    sampler = DeterministicBatchSampler(64, 4, seed=3)

    indices = sampler.indices_for_step(2)
    inputs, _ = collate_windows(dataset, indices)

    assert inputs[:, 0].tolist() == indices
