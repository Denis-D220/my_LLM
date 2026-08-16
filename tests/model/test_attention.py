"""Tests for causal multi-head self-attention.

The reference below is the textbook formulation written out in full: build the
score matrix, fill the upper triangle with -inf, softmax, multiply by V.  It is
deliberately the slow, materialized version that
``scaled_dot_product_attention`` exists to avoid, which is what makes it a
useful check on the fused kernel.

RoPE is reused from :mod:`llm.model.rope` rather than re-derived here.  That
module has its own 79 tests including a convention lock and a relative-position
proof, so re-implementing the rotation in this file would test it twice and
prove nothing new about attention.
"""

from __future__ import annotations

import math

import pytest
import torch

from llm.model import ModelConfig
from llm.model.attention import CausalSelfAttention
from llm.model.rope import RotaryEmbedding


def small_config(**overrides) -> ModelConfig:
    """A tiny but structurally identical architecture, for fast exact tests."""

    base = dict(
        vocab_size=128,
        context_length=32,
        n_layers=1,
        hidden_size=16,
        n_heads=4,
        head_dim=4,
        ffn_hidden_size=48,
    )
    base.update(overrides)
    return ModelConfig(**base)


def reference_attention(
    module: CausalSelfAttention, x: torch.Tensor, start_pos: int = 0
) -> torch.Tensor:
    """Explicit causal attention, materializing the full score matrix."""

    batch, sequence, _ = x.shape
    heads, head_dim = module.n_heads, module.head_dim

    queries = module.q_proj(x).view(batch, sequence, heads, head_dim)
    keys = module.k_proj(x).view(batch, sequence, heads, head_dim)
    values = module.v_proj(x).view(batch, sequence, heads, head_dim)

    rope = RotaryEmbedding.from_config(module.config)
    queries = rope(queries, start_pos=start_pos)
    keys = rope(keys, start_pos=start_pos)
    # values intentionally NOT rotated

    queries = queries.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    scores = queries @ keys.transpose(-2, -1) / math.sqrt(head_dim)

    causal = torch.triu(
        torch.ones(sequence, sequence, dtype=torch.bool, device=x.device), diagonal=1
    )
    scores = scores.masked_fill(causal, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    attended = weights @ values

    merged = attended.transpose(1, 2).reshape(batch, sequence, module.hidden_size)
    return module.o_proj(merged)


def reference_attention_weights(
    module: CausalSelfAttention, x: torch.Tensor
) -> torch.Tensor:
    """The post-softmax weight matrix, for inspecting what attends to what."""

    batch, sequence, _ = x.shape
    heads, head_dim = module.n_heads, module.head_dim

    queries = module.q_proj(x).view(batch, sequence, heads, head_dim)
    keys = module.k_proj(x).view(batch, sequence, heads, head_dim)

    rope = RotaryEmbedding.from_config(module.config)
    queries = rope(queries, start_pos=0).transpose(1, 2)
    keys = rope(keys, start_pos=0).transpose(1, 2)

    scores = queries @ keys.transpose(-2, -1) / math.sqrt(head_dim)
    causal = torch.triu(
        torch.ones(sequence, sequence, dtype=torch.bool, device=x.device), diagonal=1
    )
    return torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(20260815)


@pytest.fixture
def attention() -> CausalSelfAttention:
    module = CausalSelfAttention(small_config())
    with torch.no_grad():
        for projection in (module.q_proj, module.k_proj, module.v_proj, module.o_proj):
            projection.weight.normal_(std=0.4)
    return module


# --------------------------------------------------------------------------
# construction and parameters
# --------------------------------------------------------------------------


def test_parameter_count_matches_the_config_for_the_frozen_architecture():
    config = ModelConfig()
    module = CausalSelfAttention(config)

    total = sum(p.numel() for p in module.parameters())
    assert total == config.attention_parameters() == 1_048_576
    assert total == 4 * 512 * 512


def test_projections_have_the_expected_shapes():
    module = CausalSelfAttention(ModelConfig())

    for name in ("q_proj", "k_proj", "v_proj"):
        assert getattr(module, name).weight.shape == (512, 512)
    assert module.o_proj.weight.shape == (512, 512)


def test_there_are_no_biases_by_default():
    module = CausalSelfAttention(ModelConfig())

    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(module, name).bias is None
    assert not any(name.endswith("bias") for name, _ in module.named_parameters())


def test_enabling_bias_adds_exactly_the_configured_parameters():
    config = ModelConfig(attention_bias=True)
    module = CausalSelfAttention(config)

    total = sum(p.numel() for p in module.parameters())
    assert total == config.attention_parameters()
    assert total - 4 * 512 * 512 == 4 * 512


def test_rope_contributes_no_parameters():
    module = CausalSelfAttention(ModelConfig())

    assert sum(p.numel() for p in module.rope.parameters()) == 0
    assert module.rope.head_dim == 64
    assert module.rope.context_length == 2048


def test_rope_tables_are_not_in_the_state_dict():
    module = CausalSelfAttention(ModelConfig())
    keys = set(module.state_dict())

    assert not any("cos_table" in k or "sin_table" in k for k in keys)
    assert keys == {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
    }


def test_from_config_and_constructor_agree():
    config = small_config()
    assert isinstance(CausalSelfAttention.from_config(config), CausalSelfAttention)


def test_non_config_is_rejected():
    with pytest.raises(TypeError):
        CausalSelfAttention({"hidden_size": 512})


# --------------------------------------------------------------------------
# agreement with the explicit reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("sequence", [1, 2, 7, 32])
def test_matches_the_explicit_reference(attention, batch, sequence):
    x = torch.randn(batch, sequence, attention.hidden_size)

    torch.testing.assert_close(
        attention(x), reference_attention(attention, x), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("start_pos", [0, 1, 9, 24])
def test_matches_the_reference_at_a_nonzero_start_pos(attention, start_pos):
    x = torch.randn(2, 5, attention.hidden_size)

    torch.testing.assert_close(
        attention(x, start_pos=start_pos),
        reference_attention(attention, x, start_pos),
        atol=1e-5,
        rtol=1e-5,
    )


def test_matches_the_reference_at_the_frozen_architecture_scale():
    config = ModelConfig()
    module = CausalSelfAttention(config)
    x = torch.randn(1, 24, config.hidden_size)

    torch.testing.assert_close(
        module(x), reference_attention(module, x), atol=1e-4, rtol=1e-4
    )


def test_is_translation_invariant_in_absolute_position(attention):
    """Shifting start_pos must NOT change a full window's output.

    This looks at first like proof that start_pos is ignored.  It is the
    opposite: RoPE rotates Q and K by the same offset, and scores depend only
    on the difference between positions, so translating the whole window
    cancels exactly.  See test_dot_product_depends_only_on_relative_position
    in test_rope.py -- this is that property surfacing one layer up.

    The consequence is that start_pos is a no-op for stateless full-window
    attention.  It earns its keep only once a KV cache exists, where a query
    at position n attends to keys spanning 0..n and the offsets genuinely
    differ.
    """

    x = torch.randn(1, 4, attention.hidden_size)

    baseline = attention(x, start_pos=0)
    for start_pos in (1, 6, 20):
        torch.testing.assert_close(
            attention(x, start_pos=start_pos), baseline, atol=1e-5, rtol=1e-5
        )


def test_rope_is_actually_applied(attention):
    """Guard: the reference tests would pass if RoPE were skipped in both."""

    x = torch.randn(1, 8, attention.hidden_size)

    batch, sequence, _ = x.shape
    heads, head_dim = attention.n_heads, attention.head_dim

    queries = attention.q_proj(x).view(batch, sequence, heads, head_dim).transpose(1, 2)
    keys = attention.k_proj(x).view(batch, sequence, heads, head_dim).transpose(1, 2)
    values = attention.v_proj(x).view(batch, sequence, heads, head_dim).transpose(1, 2)

    scores = queries @ keys.transpose(-2, -1) / math.sqrt(head_dim)
    causal = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool), diagonal=1)
    weights = torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)
    without_rope = attention.o_proj(
        (weights @ values)
        .transpose(1, 2)
        .reshape(batch, sequence, attention.hidden_size)
    )

    assert not torch.allclose(attention(x), without_rope, atol=1e-4)


def test_values_are_not_rotated(attention):
    """A reference that also rotated V would disagree with the module."""

    x = torch.randn(2, 6, attention.hidden_size)

    batch, sequence, _ = x.shape
    heads, head_dim = attention.n_heads, attention.head_dim
    rope = RotaryEmbedding.from_config(attention.config)

    queries = rope(
        attention.q_proj(x).view(batch, sequence, heads, head_dim), start_pos=0
    ).transpose(1, 2)
    keys = rope(
        attention.k_proj(x).view(batch, sequence, heads, head_dim), start_pos=0
    ).transpose(1, 2)
    rotated_values = rope(
        attention.v_proj(x).view(batch, sequence, heads, head_dim), start_pos=0
    ).transpose(1, 2)

    scores = queries @ keys.transpose(-2, -1) / math.sqrt(head_dim)
    causal = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool), diagonal=1)
    weights = torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)
    wrong = attention.o_proj(
        (weights @ rotated_values)
        .transpose(1, 2)
        .reshape(batch, sequence, attention.hidden_size)
    )

    assert not torch.allclose(attention(x), wrong, atol=1e-4)


# --------------------------------------------------------------------------
# causality
# --------------------------------------------------------------------------


def test_changing_a_later_token_cannot_change_an_earlier_output(attention):
    """The single most important property of the whole model."""

    x = torch.randn(1, 32, attention.hidden_size)
    baseline = attention(x)

    modified = x.clone()
    modified[0, 20:] = torch.randn(12, attention.hidden_size)
    changed = attention(modified)

    torch.testing.assert_close(baseline[:, :20], changed[:, :20], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(baseline[:, 20:], changed[:, 20:], atol=1e-4)


@pytest.mark.parametrize("cut", [1, 5, 16, 31])
def test_every_prefix_is_independent_of_its_suffix(attention, cut):
    x = torch.randn(1, 32, attention.hidden_size)

    modified = x.clone()
    modified[0, cut:] += 10.0

    torch.testing.assert_close(
        attention(x)[:, :cut], attention(modified)[:, :cut], atol=1e-6, rtol=1e-6
    )


def test_a_prefix_computed_alone_matches_the_same_prefix_in_context(attention):
    """Truncating the input must not change the outputs that remain."""

    x = torch.randn(1, 16, attention.hidden_size)
    full = attention(x)

    for cut in (1, 3, 9, 16):
        torch.testing.assert_close(
            attention(x[:, :cut]), full[:, :cut], atol=1e-5, rtol=1e-5
        )


def test_attention_weights_are_lower_triangular(attention):
    x = torch.randn(1, 12, attention.hidden_size)
    weights = reference_attention_weights(attention, x)

    upper = torch.triu(torch.ones(12, 12, dtype=torch.bool), diagonal=1)
    assert bool((weights[..., upper] == 0).all())


def test_attention_weights_form_a_distribution_over_the_visible_past(attention):
    x = torch.randn(2, 10, attention.hidden_size)
    weights = reference_attention_weights(attention, x)

    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(2, attention.n_heads, 10)
    )
    assert bool((weights >= 0).all())


def test_the_first_token_attends_only_to_itself(attention):
    x = torch.randn(1, 8, attention.hidden_size)
    weights = reference_attention_weights(attention, x)

    torch.testing.assert_close(
        weights[:, :, 0, 0], torch.ones(1, attention.n_heads)
    )
    assert bool((weights[:, :, 0, 1:] == 0).all())


def test_gradient_of_an_earlier_output_never_reaches_a_later_input(attention):
    """Causality stated in the backward direction."""

    x = torch.randn(1, 16, attention.hidden_size, requires_grad=True)
    attention(x)[0, 5].sum().backward()

    assert x.grad is not None
    assert bool((x.grad[0, 6:] == 0).all())
    assert x.grad[0, :6].abs().sum() > 0


# --------------------------------------------------------------------------
# tensor contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("sequence", [1, 4, 32])
def test_output_shape_matches_input_shape(attention, batch, sequence):
    x = torch.randn(batch, sequence, attention.hidden_size)
    assert attention(x).shape == x.shape


def test_full_context_sequence_at_the_frozen_scale():
    config = ModelConfig()
    module = CausalSelfAttention(config)
    x = torch.randn(1, config.context_length, config.hidden_size)

    assert module(x).shape == x.shape


def test_sequence_beyond_the_context_is_rejected():
    config = ModelConfig()
    module = CausalSelfAttention(config)
    x = torch.randn(1, config.context_length + 1, config.hidden_size)

    with pytest.raises(ValueError, match="exceed context_length"):
        module(x)


def test_wrong_hidden_size_is_rejected(attention):
    with pytest.raises(ValueError, match="hidden_size"):
        attention(torch.randn(1, 4, attention.hidden_size + 1))


@pytest.mark.parametrize("shape", [(16,), (2, 16), (1, 2, 3, 16)])
def test_non_three_dimensional_input_is_rejected(attention, shape):
    with pytest.raises(ValueError, match="batch, sequence, hidden_size"):
        attention(torch.randn(shape))


def test_non_tensor_input_is_rejected(attention):
    with pytest.raises(TypeError):
        attention([0.0] * 16)


@pytest.mark.parametrize("start_pos", [-1, 2.0, True, None])
def test_invalid_start_pos_is_rejected(attention, start_pos):
    with pytest.raises((TypeError, ValueError)):
        attention(torch.randn(1, 2, attention.hidden_size), start_pos=start_pos)


def test_input_is_not_modified_in_place(attention):
    x = torch.randn(1, 6, attention.hidden_size)
    original = x.clone()
    attention(x)
    torch.testing.assert_close(x, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    module = CausalSelfAttention(small_config()).to(dtype)
    x = torch.randn(1, 5, module.hidden_size, dtype=dtype)
    assert module(x).dtype == dtype


def test_device_is_preserved(attention):
    x = torch.randn(1, 4, attention.hidden_size)
    assert attention(x).device == x.device


def test_is_deterministic(attention):
    x = torch.randn(2, 6, attention.hidden_size)
    torch.testing.assert_close(attention(x), attention(x))


def test_examples_in_a_batch_do_not_interact(attention):
    a = torch.randn(1, 8, attention.hidden_size)
    b = torch.randn(1, 8, attention.hidden_size)

    together = attention(torch.cat([a, b], dim=0))
    torch.testing.assert_close(together[0:1], attention(a), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(together[1:2], attention(b), atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------
# autograd
# --------------------------------------------------------------------------


def test_gradients_reach_every_projection(attention):
    x = torch.randn(2, 8, attention.hidden_size, requires_grad=True)
    attention(x).pow(2).sum().backward()

    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert parameter.grad.abs().sum() > 0, name

    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())


def test_gradcheck_against_numerical_derivatives():
    module = CausalSelfAttention(small_config(hidden_size=8, n_heads=2, head_dim=4)).to(
        torch.float64
    )
    x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-7)


def test_gradients_match_the_reference_implementation():
    """Run in float64: the fused kernel and the materialized reference sum in
    different orders, and in float32 that drift (~4e-5 on one element in 256)
    is larger than any tolerance worth asserting.  Double precision separates a
    real disagreement from floating-point noise instead of hiding both."""

    module = CausalSelfAttention(small_config()).to(torch.float64)
    with torch.no_grad():
        for projection in (
            module.q_proj,
            module.k_proj,
            module.v_proj,
            module.o_proj,
        ):
            projection.weight.normal_(std=0.4)

    x = torch.randn(1, 6, module.hidden_size, dtype=torch.float64)

    actual = x.clone().requires_grad_(True)
    module(actual).pow(2).sum().backward()
    actual_weight_grads = {
        name: p.grad.clone() for name, p in module.named_parameters()
    }

    module.zero_grad(set_to_none=True)
    expected = x.clone().requires_grad_(True)
    reference_attention(module, expected).pow(2).sum().backward()

    torch.testing.assert_close(actual.grad, expected.grad, atol=1e-10, rtol=1e-10)
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(
            actual_weight_grads[name], parameter.grad, atol=1e-10, rtol=1e-10, msg=name
        )


def test_loss_and_gradients_are_finite_for_extreme_inputs(attention):
    for scale in (1e-6, 1e3):
        attention.zero_grad(set_to_none=True)
        x = (torch.randn(1, 8, attention.hidden_size) * scale).requires_grad_(True)
        output = attention(x)

        assert bool(torch.isfinite(output).all()), scale
        output.pow(2).sum().backward()
        assert bool(torch.isfinite(x.grad).all()), scale
