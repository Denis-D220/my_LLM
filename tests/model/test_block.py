"""Tests for the pre-norm Transformer block.

The block introduces no arithmetic of its own, so the tests here are about
*wiring*: that both residual adds exist, that the two norms are separate, that
the branches run in the right order, and -- most importantly -- that causality
survives composition.

The identity tests are the sharpest tool available for residual wiring. With
``attention.o_proj`` and ``feedforward.down_proj`` zeroed, both branches emit
exactly zero, so a correctly wired block must return its input untouched. A
missing residual returns zeros instead, which no shape or dtype check would
catch.
"""

from __future__ import annotations

import pytest
import torch

from llm.model import ModelConfig
from llm.model.block import TransformerBlock
from llm.model.rmsnorm import RMSNorm


def small_config(**overrides) -> ModelConfig:
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


def randomize(block: TransformerBlock, std: float = 0.4) -> TransformerBlock:
    """Break away from default init so no branch is accidentally near-zero."""

    with torch.no_grad():
        for projection in (
            block.attention.q_proj,
            block.attention.k_proj,
            block.attention.v_proj,
            block.attention.o_proj,
            block.feedforward.gate_proj,
            block.feedforward.up_proj,
            block.feedforward.down_proj,
        ):
            projection.weight.normal_(std=std)
        block.attention_norm.weight.normal_(mean=1.0, std=0.2)
        block.ffn_norm.weight.normal_(mean=1.0, std=0.2)
    return block


def silence_branches(block: TransformerBlock, *, attention: bool, feedforward: bool):
    """Zero a branch's output projection so it contributes exactly nothing."""

    with torch.no_grad():
        if attention:
            block.attention.o_proj.weight.zero_()
        if feedforward:
            block.feedforward.down_proj.weight.zero_()
    return block


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(20260815)


@pytest.fixture
def block() -> TransformerBlock:
    return randomize(TransformerBlock(small_config()))


# --------------------------------------------------------------------------
# parameters and structure
# --------------------------------------------------------------------------


def test_parameter_count_is_exactly_the_frozen_block_budget():
    config = ModelConfig()
    module = TransformerBlock(config)

    total = sum(p.numel() for p in module.parameters())
    assert total == config.block_parameters() == 3_408_896
    assert total == 1_048_576 + 2_359_296 + 512 + 512


def test_the_block_adds_no_parameters_of_its_own():
    config = ModelConfig()
    module = TransformerBlock(config)

    children = (
        sum(p.numel() for p in module.attention.parameters())
        + sum(p.numel() for p in module.feedforward.parameters())
        + sum(p.numel() for p in module.attention_norm.parameters())
        + sum(p.numel() for p in module.ffn_norm.parameters())
    )
    assert sum(p.numel() for p in module.parameters()) == children


def test_state_dict_contains_exactly_the_expected_tensors():
    module = TransformerBlock(ModelConfig())

    assert sorted(module.state_dict()) == [
        "attention.k_proj.weight",
        "attention.o_proj.weight",
        "attention.q_proj.weight",
        "attention.v_proj.weight",
        "attention_norm.weight",
        "feedforward.down_proj.weight",
        "feedforward.gate_proj.weight",
        "feedforward.up_proj.weight",
        "ffn_norm.weight",
    ]


def test_the_two_norms_are_distinct_modules():
    """Sharing one norm would halve the scales and couple the branches."""

    module = TransformerBlock(ModelConfig())

    assert isinstance(module.attention_norm, RMSNorm)
    assert isinstance(module.ffn_norm, RMSNorm)
    assert module.attention_norm is not module.ffn_norm
    assert module.attention_norm.weight is not module.ffn_norm.weight
    assert (
        module.attention_norm.weight.data_ptr() != module.ffn_norm.weight.data_ptr()
    )


def test_the_two_norms_train_independently(block):
    x = torch.randn(1, 6, block.hidden_size)
    block(x).pow(2).sum().backward()

    assert not torch.allclose(
        block.attention_norm.weight.grad, block.ffn_norm.weight.grad
    )


def test_from_config_and_constructor_agree():
    assert isinstance(TransformerBlock.from_config(small_config()), TransformerBlock)


def test_non_config_is_rejected():
    with pytest.raises(TypeError):
        TransformerBlock({"hidden_size": 512})


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("start_pos", [0, 3, 11])
def test_forward_equals_the_manual_composition(block, start_pos):
    x = torch.randn(2, 5, block.hidden_size)

    hidden = x + block.attention(block.attention_norm(x), start_pos=start_pos)
    expected = hidden + block.feedforward(block.ffn_norm(hidden))

    torch.testing.assert_close(block(x, start_pos=start_pos), expected)


def test_residual_deltas_are_the_two_branch_outputs(block):
    """block(x) - x must equal attention branch + feed-forward branch."""

    x = torch.randn(1, 6, block.hidden_size)

    attention_delta = block.attention(block.attention_norm(x))
    hidden = x + attention_delta
    ffn_delta = block.feedforward(block.ffn_norm(hidden))

    torch.testing.assert_close(block(x) - x, attention_delta + ffn_delta)


def test_the_feedforward_branch_sees_the_post_attention_stream(block):
    """The second branch must read x + attention(...), not the raw input."""

    x = torch.randn(1, 6, block.hidden_size)

    hidden = x + block.attention(block.attention_norm(x))
    wrong = hidden + block.feedforward(block.ffn_norm(x))  # normed raw input

    assert not torch.allclose(block(x), wrong, atol=1e-4)


def test_branch_order_matters(block):
    """Feed-forward first, then attention, is a different function."""

    x = torch.randn(1, 6, block.hidden_size)

    swapped = x + block.feedforward(block.ffn_norm(x))
    swapped = swapped + block.attention(block.attention_norm(swapped))

    assert not torch.allclose(block(x), swapped, atol=1e-4)


# --------------------------------------------------------------------------
# residual wiring, proved by silencing branches
# --------------------------------------------------------------------------


def test_zeroing_both_branches_makes_the_block_an_identity(block):
    silence_branches(block, attention=True, feedforward=True)
    x = torch.randn(2, 7, block.hidden_size)

    torch.testing.assert_close(block(x), x)


def test_zeroing_attention_leaves_only_the_feedforward_branch(block):
    silence_branches(block, attention=True, feedforward=False)
    x = torch.randn(2, 7, block.hidden_size)

    torch.testing.assert_close(block(x), x + block.feedforward(block.ffn_norm(x)))


def test_zeroing_feedforward_leaves_only_the_attention_branch(block):
    silence_branches(block, attention=False, feedforward=True)
    x = torch.randn(2, 7, block.hidden_size)

    torch.testing.assert_close(block(x), x + block.attention(block.attention_norm(x)))


def test_with_attention_silenced_positions_stop_interacting(block):
    """Attention is the only sublayer that moves information sideways."""

    silence_branches(block, attention=True, feedforward=False)

    x = torch.randn(1, 20, block.hidden_size)
    baseline = block(x)

    modified = x.clone()
    modified[:, 10, :] = torch.randn(block.hidden_size) * 20.0
    changed = block(modified)

    torch.testing.assert_close(changed[:, :10], baseline[:, :10])
    torch.testing.assert_close(changed[:, 11:], baseline[:, 11:])


def test_a_live_block_does_mix_positions(block):
    """Guard for the test above: with attention active, positions must couple."""

    x = torch.randn(1, 20, block.hidden_size)
    baseline = block(x)

    modified = x.clone()
    modified[:, 5, :] = torch.randn(block.hidden_size) * 20.0
    changed = block(modified)

    assert not torch.allclose(changed[:, 10, :], baseline[:, 10, :], atol=1e-4)


# --------------------------------------------------------------------------
# causality, re-proved after composition
# --------------------------------------------------------------------------


def test_changing_a_later_token_cannot_change_an_earlier_output(block):
    x = torch.randn(1, 32, block.hidden_size)
    baseline = block(x)

    modified = x.clone()
    modified[0, 20:] = torch.randn(12, block.hidden_size)
    changed = block(modified)

    torch.testing.assert_close(baseline[:, :20], changed[:, :20])
    assert not torch.allclose(baseline[:, 20:], changed[:, 20:], atol=1e-4)


@pytest.mark.parametrize("cut", [1, 4, 17, 31])
def test_every_prefix_is_independent_of_its_suffix(block, cut):
    x = torch.randn(1, 32, block.hidden_size)

    modified = x.clone()
    modified[0, cut:] += 8.0

    torch.testing.assert_close(block(x)[:, :cut], block(modified)[:, :cut])


def test_a_prefix_computed_alone_matches_the_same_prefix_in_context(block):
    x = torch.randn(1, 16, block.hidden_size)
    full = block(x)

    for cut in (1, 5, 16):
        torch.testing.assert_close(block(x[:, :cut]), full[:, :cut], atol=1e-5, rtol=1e-5)


def test_gradient_of_an_earlier_output_never_reaches_a_later_input(block):
    x = torch.randn(1, 16, block.hidden_size, requires_grad=True)
    block(x)[0, 5].sum().backward()

    assert bool((x.grad[0, 6:] == 0).all())
    assert x.grad[0, :6].abs().sum() > 0


def test_is_translation_invariant_in_absolute_position(block):
    """Inherited from RoPE: a full window's output does not depend on start_pos."""

    x = torch.randn(1, 5, block.hidden_size)
    baseline = block(x, start_pos=0)

    for start_pos in (2, 9, 21):
        torch.testing.assert_close(
            block(x, start_pos=start_pos), baseline, atol=1e-5, rtol=1e-5
        )


# --------------------------------------------------------------------------
# tensor contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("sequence", [1, 8, 32])
def test_output_shape_matches_input_shape(block, batch, sequence):
    x = torch.randn(batch, sequence, block.hidden_size)
    assert block(x).shape == x.shape


def test_full_context_sequence_at_the_frozen_scale():
    module = TransformerBlock(ModelConfig())
    x = torch.randn(1, 2048, 512)
    assert module(x).shape == x.shape


def test_sequence_beyond_the_context_is_rejected():
    module = TransformerBlock(ModelConfig())
    with pytest.raises(ValueError, match="exceed context_length"):
        module(torch.randn(1, 2049, 512))


def test_wrong_hidden_size_is_rejected(block):
    with pytest.raises(ValueError, match="hidden_size"):
        block(torch.randn(1, 4, block.hidden_size + 1))


@pytest.mark.parametrize("shape", [(16,), (2, 16), (1, 2, 3, 16)])
def test_non_three_dimensional_input_is_rejected(block, shape):
    with pytest.raises(ValueError, match="batch, sequence, hidden_size"):
        block(torch.randn(shape))


def test_non_tensor_input_is_rejected(block):
    with pytest.raises(TypeError):
        block([0.0] * 16)


def test_input_is_not_modified_in_place(block):
    x = torch.randn(1, 6, block.hidden_size)
    original = x.clone()
    block(x)
    torch.testing.assert_close(x, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    module = randomize(TransformerBlock(small_config())).to(dtype)
    x = torch.randn(1, 5, module.hidden_size, dtype=dtype)
    assert module(x).dtype == dtype


def test_device_is_preserved(block):
    x = torch.randn(1, 4, block.hidden_size)
    assert block(x).device == x.device


def test_is_deterministic(block):
    x = torch.randn(2, 6, block.hidden_size)
    torch.testing.assert_close(block(x), block(x))


def test_examples_in_a_batch_do_not_interact(block):
    a = torch.randn(1, 8, block.hidden_size)
    b = torch.randn(1, 8, block.hidden_size)

    together = block(torch.cat([a, b], dim=0))
    torch.testing.assert_close(together[0:1], block(a))
    torch.testing.assert_close(together[1:2], block(b))


def test_output_is_finite_for_extreme_inputs(block):
    for scale in (1e-6, 1e3):
        assert bool(torch.isfinite(block(torch.randn(1, 8, block.hidden_size) * scale)).all())


# --------------------------------------------------------------------------
# autograd
# --------------------------------------------------------------------------


def test_gradients_reach_every_parameter_and_the_input(block):
    x = torch.randn(2, 8, block.hidden_size, requires_grad=True)
    block(x).pow(2).sum().backward()

    expected = {
        "attention.q_proj.weight",
        "attention.k_proj.weight",
        "attention.v_proj.weight",
        "attention.o_proj.weight",
        "attention_norm.weight",
        "feedforward.gate_proj.weight",
        "feedforward.up_proj.weight",
        "feedforward.down_proj.weight",
        "ffn_norm.weight",
    }
    assert {name for name, _ in block.named_parameters()} == expected

    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert parameter.grad.abs().sum() > 0, name

    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())
    assert x.grad.abs().sum() > 0


def test_gradcheck_against_numerical_derivatives():
    module = randomize(
        TransformerBlock(small_config(hidden_size=8, n_heads=2, head_dim=4, ffn_hidden_size=12)),
        std=0.3,
    ).to(torch.float64)
    x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-7)


def test_the_residual_path_carries_gradient_even_with_dead_branches(block):
    """The identity path is what keeps a deep stack trainable."""

    silence_branches(block, attention=True, feedforward=True)
    x = torch.randn(1, 6, block.hidden_size, requires_grad=True)

    block(x).sum().backward()

    torch.testing.assert_close(x.grad, torch.ones_like(x))
