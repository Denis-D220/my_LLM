"""Tests for RMSNorm.

The reference below is a literal transcription of the formula, written with
``/ torch.sqrt(...)`` where the implementation uses ``* torch.rsqrt(...)``.
That divergence is intentional: a reference that reuses the implementation's
own expression proves only that the code equals itself.  These two agree only
if both compute the same mathematics.
"""

from __future__ import annotations

import pytest
import torch

from llm.model import ModelConfig
from llm.model.rmsnorm import RMSNorm


def reference_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm written directly from the definition."""

    mean_square = torch.mean(x * x, dim=-1, keepdim=True)
    return x / torch.sqrt(mean_square + eps) * weight


def reference_layernorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm, used only to prove RMSNorm is not this."""

    mean = x.mean(dim=-1, keepdim=True)
    centred = x - mean
    variance = centred.pow(2).mean(dim=-1, keepdim=True)
    return centred / torch.sqrt(variance + eps)


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(20260815)


# --------------------------------------------------------------------------
# construction and parameters
# --------------------------------------------------------------------------


def test_has_exactly_dim_trainable_parameters():
    norm = RMSNorm(512)

    parameters = dict(norm.named_parameters())
    assert list(parameters) == ["weight"]
    assert parameters["weight"].shape == (512,)
    assert sum(p.numel() for p in norm.parameters()) == 512
    assert all(p.requires_grad for p in norm.parameters())


def test_weight_is_initialized_to_ones():
    norm = RMSNorm(64)
    torch.testing.assert_close(norm.weight, torch.ones(64))


def test_there_is_no_bias():
    norm = RMSNorm(32)

    assert "bias" not in dict(norm.named_parameters())
    assert "bias" not in dict(norm.named_buffers())
    assert getattr(norm, "bias", None) is None


def test_reset_parameters_restores_ones():
    norm = RMSNorm(16)
    with torch.no_grad():
        norm.weight.uniform_(-3.0, 3.0)

    norm.reset_parameters()
    torch.testing.assert_close(norm.weight, torch.ones(16))


def test_from_config_uses_the_frozen_architecture():
    config = ModelConfig()
    norm = RMSNorm.from_config(config)

    assert norm.dim == config.hidden_size == 512
    assert norm.eps == pytest.approx(config.rms_norm_eps)
    assert sum(p.numel() for p in norm.parameters()) == config.norm_parameters()


def test_from_config_rejects_non_config():
    with pytest.raises(TypeError):
        RMSNorm.from_config({"hidden_size": 512})


@pytest.mark.parametrize("dim", [0, -1, 2.0, True, None])
def test_invalid_dim_is_rejected(dim):
    with pytest.raises((TypeError, ValueError)):
        RMSNorm(dim)


@pytest.mark.parametrize("eps", [0.0, -1e-6, "small", True, None])
def test_invalid_eps_is_rejected(eps):
    with pytest.raises((TypeError, ValueError)):
        RMSNorm(8, eps=eps)


# --------------------------------------------------------------------------
# the mathematical contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (16,),
        (4, 16),
        (2, 8, 16),
        (2, 3, 4, 16),
    ],
)
def test_matches_the_reference_for_every_rank(shape):
    norm = RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.3)

    x = torch.randn(shape)
    expected = reference_rmsnorm(x, norm.weight, norm.eps)

    torch.testing.assert_close(norm(x), expected)


def test_matches_the_reference_with_an_arbitrary_learned_weight():
    norm = RMSNorm(6, eps=1e-6)
    weight = torch.tensor([-2.0, 0.0, 0.5, 1.0, 3.25, -0.75])
    with torch.no_grad():
        norm.weight.copy_(weight)

    x = torch.randn(5, 6) * 12.0
    torch.testing.assert_close(norm(x), reference_rmsnorm(x, weight, norm.eps))


def test_output_shape_matches_input_shape():
    norm = RMSNorm(16)
    for shape in [(16,), (3, 16), (2, 5, 16)]:
        assert norm(torch.randn(shape)).shape == torch.Size(shape)


def test_constant_vector_maps_to_the_weight():
    """x = k * ones has RMS |k|, so the output is sign(k) * weight."""

    norm = RMSNorm(8, eps=1e-12)
    weight = torch.linspace(0.5, 2.0, 8)
    with torch.no_grad():
        norm.weight.copy_(weight)

    torch.testing.assert_close(norm(torch.full((8,), 3.0)), weight)
    torch.testing.assert_close(norm(torch.full((8,), -3.0)), -weight)


def test_normalization_is_over_the_last_dimension_only():
    """Rows must not influence each other."""

    norm = RMSNorm(4)
    x = torch.randn(6, 4)

    rows = torch.stack([norm(row) for row in x])
    torch.testing.assert_close(norm(x), rows)


def test_changing_one_row_leaves_the_others_untouched():
    norm = RMSNorm(4)
    x = torch.randn(3, 4)
    before = norm(x)

    # The change must alter row 1's *direction*. Merely rescaling it would
    # leave the output identical, because RMSNorm discards magnitude -- see
    # test_is_approximately_scale_invariant.
    modified = x.clone()
    modified[1] = torch.tensor([5.0, -1.0, 0.25, 3.0])
    after = norm(modified)

    torch.testing.assert_close(after[0], before[0])
    torch.testing.assert_close(after[2], before[2])
    assert not torch.allclose(after[1], before[1])


def test_batched_input_equals_per_example_application():
    norm = RMSNorm(16)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.2)

    batch = torch.randn(4, 7, 16)
    stacked = torch.stack([norm(example) for example in batch])
    torch.testing.assert_close(norm(batch), stacked)


def test_is_approximately_scale_invariant():
    """RMSNorm removes magnitude: only direction survives (up to eps)."""

    norm = RMSNorm(32, eps=1e-12)
    x = torch.randn(4, 32)

    torch.testing.assert_close(norm(x * 50.0), norm(x))
    torch.testing.assert_close(norm(x * 0.02), norm(x))


# --------------------------------------------------------------------------
# not LayerNorm
# --------------------------------------------------------------------------


def test_does_not_subtract_the_mean():
    """An offset row must keep a non-zero mean; LayerNorm would zero it."""

    norm = RMSNorm(4, eps=1e-6)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    output = norm(x)

    assert output.mean().abs() > 0.5
    assert not torch.allclose(output, reference_layernorm(x, norm.eps), atol=1e-3)


def test_all_signs_are_preserved():
    """Rescaling by a positive factor cannot flip a sign; re-centring can."""

    norm = RMSNorm(5, eps=1e-6)
    x = torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0]])

    output = norm(x)

    assert bool((output > 0).all())
    assert bool((reference_layernorm(x, norm.eps) < 0).any())


def test_differs_from_torch_layernorm_on_offset_input():
    norm = RMSNorm(8, eps=1e-6)
    layer_norm = torch.nn.LayerNorm(8, eps=1e-6, elementwise_affine=False)
    x = torch.randn(3, 8) + 7.0

    assert not torch.allclose(norm(x), layer_norm(x), atol=1e-3)


# --------------------------------------------------------------------------
# numerical behaviour
# --------------------------------------------------------------------------


def test_zero_input_is_finite_and_zero():
    norm = RMSNorm(8, eps=1e-6)
    output = norm(torch.zeros(2, 8))

    assert bool(torch.isfinite(output).all())
    torch.testing.assert_close(output, torch.zeros(2, 8))


def test_large_and_small_magnitudes_stay_finite():
    norm = RMSNorm(16, eps=1e-6)

    for scale in (1e-8, 1e-4, 1e4, 1e8):
        output = norm(torch.randn(3, 16) * scale)
        assert bool(torch.isfinite(output).all()), scale


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    norm = RMSNorm(16).to(dtype)
    x = torch.randn(2, 16, dtype=dtype)
    output = norm(x)

    assert output.dtype == dtype
    torch.testing.assert_close(output, reference_rmsnorm(x, norm.weight, norm.eps))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_dtype_is_preserved_without_internal_upcast(dtype):
    """v0.1 policy: compute in the input dtype, return the input dtype."""

    norm = RMSNorm(16).to(dtype)
    output = norm(torch.randn(2, 16).to(dtype))

    assert output.dtype == dtype
    assert bool(torch.isfinite(output).all())


def test_device_is_preserved():
    norm = RMSNorm(8)
    x = torch.randn(2, 8)
    assert norm(x).device == x.device


def test_is_deterministic():
    norm = RMSNorm(16)
    x = torch.randn(3, 16)
    torch.testing.assert_close(norm(x), norm(x))


# --------------------------------------------------------------------------
# shape guarding
# --------------------------------------------------------------------------


@pytest.mark.parametrize("last_dim", [1, 4, 15, 17, 32])
def test_wrong_last_dimension_is_refused(last_dim):
    """Especially last_dim == 1, which would otherwise broadcast silently."""

    norm = RMSNorm(16)
    with pytest.raises(ValueError, match="last dimension"):
        norm(torch.randn(2, last_dim))


def test_scalar_input_is_refused():
    with pytest.raises(ValueError):
        RMSNorm(4)(torch.tensor(1.0))


def test_non_tensor_input_is_refused():
    with pytest.raises(TypeError):
        RMSNorm(4)([1.0, 2.0, 3.0, 4.0])


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------


def test_gradients_reach_both_the_input_and_the_weight():
    norm = RMSNorm(16)
    x = torch.randn(4, 16, requires_grad=True)

    norm(x).sum().backward()

    assert x.grad is not None
    assert norm.weight.grad is not None
    assert x.grad.shape == x.shape
    assert norm.weight.grad.shape == norm.weight.shape
    assert bool(torch.isfinite(x.grad).all())
    assert bool(torch.isfinite(norm.weight.grad).all())
    assert x.grad.abs().sum() > 0
    assert norm.weight.grad.abs().sum() > 0


def test_weight_gradient_matches_the_analytic_value():
    """For L = sum(output), dL/dw is the sum of normalized inputs."""

    norm = RMSNorm(6, eps=1e-6)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.4)

    x = torch.randn(5, 6)
    norm(x).sum().backward()

    normalized = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + norm.eps)
    torch.testing.assert_close(norm.weight.grad, normalized.sum(dim=0))


def test_gradcheck_against_numerical_derivatives():
    """Full analytic-vs-numerical Jacobian check in double precision."""

    norm = RMSNorm(5, eps=1e-6).to(torch.float64)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.3)

    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(norm, (x,), eps=1e-6, atol=1e-8)


def test_gradients_are_finite_for_a_zero_input():
    """eps must keep the backward pass finite where the RMS vanishes."""

    norm = RMSNorm(8, eps=1e-6)
    x = torch.zeros(2, 8, requires_grad=True)

    norm(x).sum().backward()

    assert bool(torch.isfinite(x.grad).all())
    assert bool(torch.isfinite(norm.weight.grad).all())


def test_gradient_matches_reference_implementation_gradient():
    norm = RMSNorm(7, eps=1e-6)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.5)

    x = torch.randn(4, 7)

    actual_input = x.clone().requires_grad_(True)
    norm(actual_input).pow(2).sum().backward()

    reference_input = x.clone().requires_grad_(True)
    reference_weight = norm.weight.detach().clone().requires_grad_(True)
    reference_rmsnorm(reference_input, reference_weight, norm.eps).pow(2).sum().backward()

    torch.testing.assert_close(actual_input.grad, reference_input.grad)
    torch.testing.assert_close(norm.weight.grad, reference_weight.grad)
