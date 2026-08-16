"""Tests for rotary position embedding.

The reference below rotates one coordinate pair at a time with an explicit 2x2
matrix, written from the definition rather than from the implementation's
vectorized form.  The two agree only if both compute the same rotation.

The relative-position test and the convention-lock test are the two that
matter most: the first is the property RoPE exists to provide, and the second
prevents a future refactor from silently invalidating every checkpoint.
"""

from __future__ import annotations

import math

import pytest
import torch

from llm.model import ModelConfig
from llm.model.rope import RotaryEmbedding, build_rope_tables


def reference_angle(pair_index: int, position: int, head_dim: int, theta: float) -> float:
    """theta_i = base ** (-2i / head_dim), scaled by absolute position."""

    return position * (theta ** (-2.0 * pair_index / head_dim))


def reference_rope_vector(
    vector: list[float], position: int, head_dim: int, theta: float
) -> list[float]:
    """Rotate one head vector, one adjacent pair at a time."""

    out = list(vector)
    for pair_index in range(head_dim // 2):
        angle = reference_angle(pair_index, position, head_dim, theta)
        cos, sin = math.cos(angle), math.sin(angle)
        low, high = 2 * pair_index, 2 * pair_index + 1
        x_even, x_odd = vector[low], vector[high]
        out[low] = x_even * cos - x_odd * sin
        out[high] = x_even * sin + x_odd * cos
    return out


def reference_rope(x: torch.Tensor, start_pos: int, theta: float) -> torch.Tensor:
    """Apply the per-vector reference across a (batch, seq, heads, dim) tensor."""

    batch, sequence, heads, head_dim = x.shape
    out = torch.empty_like(x)
    for b in range(batch):
        for s in range(sequence):
            for h in range(heads):
                out[b, s, h] = torch.tensor(
                    reference_rope_vector(
                        x[b, s, h].tolist(), start_pos + s, head_dim, theta
                    ),
                    dtype=x.dtype,
                )
    return out


def differentiable_reference_rope(
    x: torch.Tensor, start_pos: int, theta: float
) -> torch.Tensor:
    """Same rotation, expressed with graph-preserving tensor ops.

    ``reference_rope`` above goes through ``.tolist()``, which is what makes it
    a genuinely independent check of the forward pass -- and also what makes it
    useless for gradients, since it severs the autograd graph.  This variant
    still indexes one adjacent pair at a time from the definition, but keeps
    every operation differentiable.
    """

    head_dim = x.shape[-1]
    sequence = x.shape[1]
    columns = []

    for pair_index in range(head_dim // 2):
        angles = torch.tensor(
            [
                reference_angle(pair_index, start_pos + step, head_dim, theta)
                for step in range(sequence)
            ],
            dtype=x.dtype,
        )
        cos = angles.cos().view(1, sequence, 1)
        sin = angles.sin().view(1, sequence, 1)

        even = x[..., 2 * pair_index]
        odd = x[..., 2 * pair_index + 1]
        columns.append(even * cos - odd * sin)
        columns.append(even * sin + odd * cos)

    return torch.stack(columns, dim=-1)


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(20260815)


@pytest.fixture
def rope() -> RotaryEmbedding:
    return RotaryEmbedding(head_dim=8, context_length=64, theta=10_000.0)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_has_no_trainable_parameters(rope):
    assert list(rope.parameters()) == []
    assert sum(p.numel() for p in rope.parameters()) == 0


def test_buffer_shapes_and_precision(rope):
    assert rope.cos_table.shape == (64, 4)
    assert rope.sin_table.shape == (64, 4)
    assert rope.cos_table.dtype == torch.float32
    assert rope.sin_table.dtype == torch.float32
    assert rope.pairs == 4


def test_tables_are_not_persistent(rope):
    """Derived constants belong in the config, not in the checkpoint."""

    assert rope.state_dict() == {}
    assert "cos_table" not in rope.state_dict()
    assert "sin_table" not in rope.state_dict()


def test_from_config_uses_the_frozen_architecture():
    config = ModelConfig()
    module = RotaryEmbedding.from_config(config)

    assert module.head_dim == config.head_dim == 64
    assert module.context_length == config.context_length == 2048
    assert module.theta == pytest.approx(config.rope_theta) == pytest.approx(10_000.0)
    assert module.cos_table.shape == (2048, 32)
    assert list(module.parameters()) == []


def test_from_config_rejects_non_config():
    with pytest.raises(TypeError):
        RotaryEmbedding.from_config({"head_dim": 64})


@pytest.mark.parametrize("head_dim", [0, -2, 2.0, True, None])
def test_invalid_head_dim_is_rejected(head_dim):
    with pytest.raises((TypeError, ValueError)):
        RotaryEmbedding(head_dim=head_dim, context_length=16)


def test_odd_head_dim_is_rejected():
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(head_dim=7, context_length=16)


@pytest.mark.parametrize("context_length", [0, -1, 4.0, True, None])
def test_invalid_context_length_is_rejected(context_length):
    with pytest.raises((TypeError, ValueError)):
        RotaryEmbedding(head_dim=8, context_length=context_length)


@pytest.mark.parametrize("theta", [0.0, -1.0, "big", True, None])
def test_invalid_theta_is_rejected(theta):
    with pytest.raises((TypeError, ValueError)):
        RotaryEmbedding(head_dim=8, context_length=16, theta=theta)


def test_config_rejects_non_positive_rope_theta():
    with pytest.raises(ValueError, match="rope_theta"):
        ModelConfig(rope_theta=0.0)


# --------------------------------------------------------------------------
# the mathematics
# --------------------------------------------------------------------------


def test_position_zero_is_the_identity(rope):
    x = torch.randn(2, 1, 3, 8)
    torch.testing.assert_close(rope(x, start_pos=0), x)


def test_first_row_of_the_tables_is_cos_1_sin_0(rope):
    torch.testing.assert_close(rope.cos_table[0], torch.ones(4))
    torch.testing.assert_close(rope.sin_table[0], torch.zeros(4))


@pytest.mark.parametrize("start_pos", [0, 1, 5, 17])
def test_matches_the_manual_two_by_two_reference(rope, start_pos):
    x = torch.randn(2, 6, 3, 8)
    torch.testing.assert_close(
        rope(x, start_pos=start_pos),
        reference_rope(x, start_pos, rope.theta),
        atol=1e-5,
        rtol=1e-5,
    )


def test_the_two_references_agree_with_each_other(rope):
    """The differentiable reference is only useful if it matches the literal one."""

    x = torch.randn(2, 5, 3, 8)
    torch.testing.assert_close(
        differentiable_reference_rope(x, 4, rope.theta),
        reference_rope(x, 4, rope.theta),
        atol=1e-5,
        rtol=1e-5,
    )


def test_tables_match_the_analytic_frequencies(rope):
    for pair_index in range(rope.pairs):
        for position in (0, 1, 9, 63):
            angle = reference_angle(pair_index, position, rope.head_dim, rope.theta)
            assert rope.cos_table[position, pair_index].item() == pytest.approx(
                math.cos(angle), abs=1e-6
            )
            assert rope.sin_table[position, pair_index].item() == pytest.approx(
                math.sin(angle), abs=1e-6
            )


def test_each_coordinate_pair_keeps_its_norm(rope):
    x = torch.randn(1, 12, 2, 8)
    y = rope(x, start_pos=3)

    before = x.reshape(1, 12, 2, 4, 2).norm(dim=-1)
    after = y.reshape(1, 12, 2, 4, 2).norm(dim=-1)
    torch.testing.assert_close(before, after)


def test_total_vector_norm_is_preserved(rope):
    x = torch.randn(2, 10, 3, 8)
    y = rope(x, start_pos=7)
    torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1))


def test_dot_product_is_preserved_when_both_sides_rotate_equally(rope):
    """A rotation is orthogonal: equal rotation cannot change an inner product."""

    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 2, 8)

    before = (q * k).sum(dim=-1)
    after = (rope(q, start_pos=5) * rope(k, start_pos=5)).sum(dim=-1)
    torch.testing.assert_close(before, after)


@pytest.mark.parametrize(
    "m_a, n_a, m_b, n_b",
    [
        (3, 8, 11, 16),   # both differences +5
        (0, 4, 20, 24),   # both differences +4
        (9, 2, 30, 23),   # both differences -7
    ],
)
def test_dot_product_depends_only_on_relative_position(rope, m_a, n_a, m_b, n_b):
    """(R_m q)·(R_n k) must be a function of n - m alone."""

    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    first = (rope(q, start_pos=m_a) * rope(k, start_pos=n_a)).sum()
    second = (rope(q, start_pos=m_b) * rope(k, start_pos=n_b)).sum()

    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)


def test_different_relative_positions_give_different_scores(rope):
    """Guards the test above: the score must not be constant in the offset."""

    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    scores = {
        offset: (rope(q, start_pos=0) * rope(k, start_pos=offset)).sum().item()
        for offset in (0, 1, 2, 5, 9)
    }
    assert len(set(round(v, 6) for v in scores.values())) > 1


def test_adjacent_pairs_rotate_independently(rope):
    """Changing pair 1 must not disturb pair 0."""

    x = torch.zeros(1, 1, 1, 8)
    x[0, 0, 0, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    modified = x.clone()
    modified[0, 0, 0, 4:] = torch.tensor([9.0, -5.0, 7.0, 1.0])

    base = rope(x, start_pos=4)
    changed = rope(modified, start_pos=4)

    torch.testing.assert_close(base[..., :4], changed[..., :4])


# --------------------------------------------------------------------------
# the convention lock
# --------------------------------------------------------------------------


def test_convention_is_interleaved_not_split_halves():
    """Pin (0,1),(2,3) pairing. Split-halves would pair (0,2),(1,3)."""

    head_dim, theta, position = 4, 10_000.0, 1
    module = RotaryEmbedding(head_dim=head_dim, context_length=8, theta=theta)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 1, 4)
    actual = module(x, start_pos=position).flatten()

    angle_0 = reference_angle(0, position, head_dim, theta)  # 1.0
    angle_1 = reference_angle(1, position, head_dim, theta)  # 0.01
    cos_0, sin_0 = math.cos(angle_0), math.sin(angle_0)
    cos_1, sin_1 = math.cos(angle_1), math.sin(angle_1)

    interleaved = torch.tensor(
        [
            1.0 * cos_0 - 2.0 * sin_0,
            1.0 * sin_0 + 2.0 * cos_0,
            3.0 * cos_1 - 4.0 * sin_1,
            3.0 * sin_1 + 4.0 * cos_1,
        ]
    )
    split_halves = torch.tensor(
        [
            1.0 * cos_0 - 3.0 * sin_0,
            2.0 * cos_1 - 4.0 * sin_1,
            1.0 * sin_0 + 3.0 * cos_0,
            2.0 * sin_1 + 4.0 * cos_1,
        ]
    )

    torch.testing.assert_close(actual, interleaved, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(actual, split_halves, atol=1e-3)


def test_first_two_coordinates_are_a_rotation_of_each_other():
    """A direct 2-D reading of the output's leading pair."""

    module = RotaryEmbedding(head_dim=2, context_length=8, theta=10_000.0)
    x = torch.tensor([3.0, 4.0]).reshape(1, 1, 1, 2)

    for position in range(8):
        out = module(x, start_pos=position).flatten()
        angle = float(position)  # theta_0 == 1 for pair 0
        torch.testing.assert_close(
            out,
            torch.tensor(
                [
                    3.0 * math.cos(angle) - 4.0 * math.sin(angle),
                    3.0 * math.sin(angle) + 4.0 * math.cos(angle),
                ]
            ),
            atol=1e-5,
            rtol=1e-5,
        )


# --------------------------------------------------------------------------
# position handling
# --------------------------------------------------------------------------


def test_start_pos_shifts_the_rotation(rope):
    x = torch.randn(1, 3, 1, 8)

    shifted = rope(x, start_pos=5)
    manual = reference_rope(x, 5, rope.theta)
    torch.testing.assert_close(shifted, manual, atol=1e-5, rtol=1e-5)

    assert not torch.allclose(shifted, rope(x, start_pos=0), atol=1e-3)


def test_a_single_token_at_start_pos_matches_that_row_of_a_full_sequence(rope):
    """The KV-cache path must agree with the full-sequence path."""

    full = torch.randn(1, 20, 2, 8)
    rotated_full = rope(full, start_pos=0)

    for position in (0, 1, 13, 19):
        single = full[:, position : position + 1]
        torch.testing.assert_close(
            rope(single, start_pos=position),
            rotated_full[:, position : position + 1],
        )


def test_the_last_legal_position_is_accepted():
    module = RotaryEmbedding(head_dim=8, context_length=2048)
    x = torch.randn(1, 1, 1, 8)
    assert module(x, start_pos=2047).shape == (1, 1, 1, 8)


def test_position_beyond_the_context_is_rejected():
    module = RotaryEmbedding(head_dim=8, context_length=2048)
    x = torch.randn(1, 1, 1, 8)

    with pytest.raises(ValueError, match="exceed context_length"):
        module(x, start_pos=2048)


def test_a_sequence_crossing_the_context_end_is_rejected(rope):
    x = torch.randn(1, 10, 1, 8)
    with pytest.raises(ValueError, match="exceed context_length"):
        rope(x, start_pos=60)  # 60 + 10 > 64


def test_full_context_sequence_is_accepted():
    config = ModelConfig()
    module = RotaryEmbedding.from_config(config)
    x = torch.randn(1, config.context_length, 1, config.head_dim)

    assert module(x, start_pos=0).shape == x.shape


@pytest.mark.parametrize("start_pos", [-1, 1.0, True, None])
def test_invalid_start_pos_is_rejected(rope, start_pos):
    x = torch.randn(1, 2, 1, 8)
    with pytest.raises((TypeError, ValueError)):
        rope(x, start_pos=start_pos)


# --------------------------------------------------------------------------
# tensor contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("heads", [1, 3, 8])
def test_arbitrary_batch_and_head_counts(rope, batch, heads):
    x = torch.randn(batch, 4, heads, 8)
    assert rope(x, start_pos=2).shape == x.shape


def test_sequence_length_one(rope):
    x = torch.randn(3, 1, 2, 8)
    assert rope(x, start_pos=9).shape == x.shape


def test_heads_are_rotated_identically(rope):
    """Position is a property of the token, not of the head."""

    single = torch.randn(1, 5, 1, 8)
    repeated = single.repeat(1, 1, 4, 1)

    out = rope(repeated, start_pos=3)
    for head in range(1, 4):
        torch.testing.assert_close(out[:, :, head], out[:, :, 0])


def test_wrong_head_dim_is_rejected(rope):
    with pytest.raises(ValueError, match="head_dim"):
        rope(torch.randn(1, 2, 1, 16), start_pos=0)


@pytest.mark.parametrize("shape", [(8,), (2, 8), (2, 3, 8), (1, 2, 3, 4, 8)])
def test_non_four_dimensional_input_is_rejected(rope, shape):
    with pytest.raises(ValueError, match="batch, sequence, heads, head_dim"):
        rope(torch.randn(shape), start_pos=0)


def test_non_tensor_input_is_rejected(rope):
    with pytest.raises(TypeError):
        rope([1.0] * 8, start_pos=0)


def test_input_is_not_modified_in_place(rope):
    x = torch.randn(1, 3, 1, 8)
    original = x.clone()
    rope(x, start_pos=4)
    torch.testing.assert_close(x, original)


# --------------------------------------------------------------------------
# dtype and device policy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    module = RotaryEmbedding(head_dim=8, context_length=16).to(dtype)
    x = torch.randn(2, 3, 1, 8, dtype=dtype)
    assert module(x, start_pos=1).dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_activations_keep_float32_tables(dtype):
    """BF16 training must not silently degrade the position signal."""

    module = RotaryEmbedding(head_dim=8, context_length=16).to(dtype)

    assert module.cos_table.dtype == torch.float32
    assert module.sin_table.dtype == torch.float32

    x = torch.randn(1, 3, 1, 8).to(dtype)
    output = module(x, start_pos=2)
    assert output.dtype == dtype
    assert bool(torch.isfinite(output).all())


def test_tables_survive_a_round_trip_through_half_precision():
    """Rebuilt, not cast back: downcasting loses bits that cannot return."""

    module = RotaryEmbedding(head_dim=8, context_length=16)
    expected_cos, expected_sin = build_rope_tables(8, 16, 10_000.0)

    module.to(torch.bfloat16).to(torch.float32)

    torch.testing.assert_close(module.cos_table, expected_cos)
    torch.testing.assert_close(module.sin_table, expected_sin)


def test_device_is_preserved(rope):
    x = torch.randn(1, 2, 1, 8)
    assert rope(x, start_pos=0).device == x.device


def test_is_deterministic(rope):
    x = torch.randn(2, 4, 2, 8)
    torch.testing.assert_close(rope(x, start_pos=6), rope(x, start_pos=6))


# --------------------------------------------------------------------------
# autograd
# --------------------------------------------------------------------------


def test_gradients_reach_the_input(rope):
    x = torch.randn(2, 4, 2, 8, requires_grad=True)
    rope(x, start_pos=3).pow(2).sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert bool(torch.isfinite(x.grad).all())
    assert x.grad.abs().sum() > 0


def test_there_are_no_parameter_gradients(rope):
    x = torch.randn(1, 2, 1, 8, requires_grad=True)
    rope(x, start_pos=1).sum().backward()

    assert list(rope.parameters()) == []


def test_gradcheck_against_numerical_derivatives():
    module = RotaryEmbedding(head_dim=6, context_length=16, theta=100.0).to(
        torch.float64
    )
    x = torch.randn(2, 3, 2, 6, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda inp: module(inp, start_pos=2), (x,), eps=1e-6, atol=1e-8
    )


def test_input_gradient_matches_the_reference_gradient(rope):
    x = torch.randn(1, 3, 1, 8)

    actual = x.clone().requires_grad_(True)
    rope(actual, start_pos=5).pow(2).sum().backward()

    expected = x.clone().requires_grad_(True)
    differentiable_reference_rope(expected, 5, rope.theta).pow(2).sum().backward()

    torch.testing.assert_close(actual.grad, expected.grad, atol=1e-5, rtol=1e-5)
