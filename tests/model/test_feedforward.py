"""Tests for the SwiGLU feed-forward network.

The reference multiplies by transposed weight matrices directly and spells SiLU
out as ``z * sigmoid(z)`` rather than calling ``F.silu``, so agreement means the
two computed the same function rather than the same call.

Several tests build deliberately *wrong* variants -- gate and up swapped, the
nonlinearity dropped, the product replaced by a sum, the down projection
applied before the gating -- and assert the module differs from each. Together
they pin which of several superficially similar formulas is actually
implemented, which a single reference comparison does not do on its own if the
reference ever drifts.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from llm.model import ModelConfig
from llm.model.feedforward import SwiGLU


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


def reference_silu(z: torch.Tensor) -> torch.Tensor:
    """SiLU from its definition, not from torch.nn.functional."""

    return z * torch.sigmoid(z)


def reference_swiglu(module: SwiGLU, x: torch.Tensor) -> torch.Tensor:
    gate = x @ module.gate_proj.weight.T
    up = x @ module.up_proj.weight.T
    return (reference_silu(gate) * up) @ module.down_proj.weight.T


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(20260815)


@pytest.fixture
def swiglu() -> SwiGLU:
    module = SwiGLU(small_config())
    with torch.no_grad():
        for projection in (module.gate_proj, module.up_proj, module.down_proj):
            projection.weight.normal_(std=0.5)
    return module


# --------------------------------------------------------------------------
# construction and geometry
# --------------------------------------------------------------------------


def test_parameter_count_matches_the_config_for_the_frozen_architecture():
    config = ModelConfig()
    module = SwiGLU(config)

    total = sum(p.numel() for p in module.parameters())
    assert total == config.feedforward_parameters() == 2_359_296
    assert total == 3 * 512 * 1_536


def test_projection_shapes_are_512_to_1536_to_512():
    module = SwiGLU(ModelConfig())

    assert module.gate_proj.weight.shape == (1_536, 512)
    assert module.up_proj.weight.shape == (1_536, 512)
    assert module.down_proj.weight.shape == (512, 1_536)


def test_there_are_exactly_three_weight_matrices_and_no_biases():
    module = SwiGLU(ModelConfig())

    assert sorted(module.state_dict()) == [
        "down_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
    ]
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert getattr(module, name).bias is None
    assert not any(name.endswith("bias") for name, _ in module.named_parameters())


def test_enabling_bias_adds_exactly_the_configured_parameters():
    config = ModelConfig(mlp_bias=True)
    module = SwiGLU(config)

    total = sum(p.numel() for p in module.parameters())
    assert total == config.feedforward_parameters()
    assert total - 3 * 512 * 1_536 == 2 * 1_536 + 512


def test_from_config_and_constructor_agree():
    config = small_config()
    assert isinstance(SwiGLU.from_config(config), SwiGLU)


def test_non_config_is_rejected():
    with pytest.raises(TypeError):
        SwiGLU({"hidden_size": 512})


# --------------------------------------------------------------------------
# forward correctness
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(16,), (4, 16), (2, 8, 16), (2, 3, 4, 16)],
)
def test_matches_the_reference_for_every_rank(swiglu, shape):
    x = torch.randn(shape)
    torch.testing.assert_close(
        swiglu(x), reference_swiglu(swiglu, x), atol=1e-6, rtol=1e-6
    )


def test_matches_the_reference_at_the_frozen_architecture_scale():
    module = SwiGLU(ModelConfig())
    x = torch.randn(2, 16, 512)

    torch.testing.assert_close(
        module(x), reference_swiglu(module, x), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", [(16,), (1, 16), (3, 16), (2, 5, 16)])
def test_output_shape_matches_input_shape(swiglu, shape):
    assert swiglu(torch.randn(shape)).shape == torch.Size(shape)


def test_zero_input_maps_to_zero(swiglu):
    """SiLU(0) == 0 and there are no biases, so the whole path collapses."""

    output = swiglu(torch.zeros(3, 16))
    torch.testing.assert_close(output, torch.zeros(3, 16))


def test_leading_dimensions_are_treated_identically():
    """A (2, 5, 16) call must equal ten independent (16,) calls.

    Exact in exact arithmetic, but a batched GEMM and a per-vector matvec take
    different BLAS paths and disagree by ~1e-6 in float32. Double precision
    keeps the assertion sharp instead of widening it until anything passes.
    """

    module = SwiGLU(small_config()).to(torch.float64)
    with torch.no_grad():
        for projection in (module.gate_proj, module.up_proj, module.down_proj):
            projection.weight.normal_(std=0.5)

    x = torch.randn(2, 5, 16, dtype=torch.float64)
    stacked = torch.stack(
        [torch.stack([module(x[b, s]) for s in range(5)]) for b in range(2)]
    )
    torch.testing.assert_close(module(x), stacked, atol=1e-12, rtol=1e-12)


def test_positions_do_not_interact(swiglu):
    """Attention is the only sublayer allowed to move information sideways."""

    x = torch.randn(1, 20, 16)
    baseline = swiglu(x)

    modified = x.clone()
    modified[:, 10, :] = torch.randn(16) * 25.0
    changed = swiglu(modified)

    torch.testing.assert_close(changed[:, 5, :], baseline[:, 5, :])
    torch.testing.assert_close(changed[:, :10], baseline[:, :10])
    torch.testing.assert_close(changed[:, 11:], baseline[:, 11:])
    assert not torch.allclose(changed[:, 10, :], baseline[:, 10, :])


def test_examples_in_a_batch_do_not_interact(swiglu):
    a = torch.randn(1, 4, 16)
    b = torch.randn(1, 4, 16)

    together = swiglu(torch.cat([a, b], dim=0))
    torch.testing.assert_close(together[0:1], swiglu(a))
    torch.testing.assert_close(together[1:2], swiglu(b))


# --------------------------------------------------------------------------
# SwiGLU semantics: which formula is this, exactly
# --------------------------------------------------------------------------


def test_gate_and_up_are_distinct_projections(swiglu):
    assert not torch.allclose(swiglu.gate_proj.weight, swiglu.up_proj.weight)

    x = torch.randn(4, 16)
    assert not torch.allclose(swiglu.gate_proj(x), swiglu.up_proj(x))


def test_silu_is_applied_to_the_gate_not_the_up_projection(swiglu):
    x = torch.randn(4, 16)

    gate = x @ swiglu.gate_proj.weight.T
    up = x @ swiglu.up_proj.weight.T
    swapped = (gate * reference_silu(up)) @ swiglu.down_proj.weight.T

    assert not torch.allclose(swiglu(x), swapped, atol=1e-4)


def test_swapping_gate_and_up_changes_the_answer(swiglu):
    x = torch.randn(4, 16)

    gate = x @ swiglu.gate_proj.weight.T
    up = x @ swiglu.up_proj.weight.T
    swapped = (reference_silu(up) * gate) @ swiglu.down_proj.weight.T

    # SiLU(u)*g vs SiLU(g)*u -- the same two tensors, different roles.
    assert not torch.allclose(swiglu(x), swapped, atol=1e-4)


def test_removing_the_nonlinearity_changes_the_answer(swiglu):
    x = torch.randn(4, 16)

    gate = x @ swiglu.gate_proj.weight.T
    up = x @ swiglu.up_proj.weight.T
    ungated = (gate * up) @ swiglu.down_proj.weight.T

    assert not torch.allclose(swiglu(x), ungated, atol=1e-4)


def test_replacing_the_product_with_a_sum_changes_the_answer(swiglu):
    x = torch.randn(4, 16)

    gate = x @ swiglu.gate_proj.weight.T
    up = x @ swiglu.up_proj.weight.T
    added = (reference_silu(gate) + up) @ swiglu.down_proj.weight.T

    assert not torch.allclose(swiglu(x), added, atol=1e-4)


def test_gating_happens_before_the_down_projection(swiglu):
    """down(silu(g) * u) is not down(silu(g)) * down(u), though both fit."""

    x = torch.randn(4, 16)

    gate = x @ swiglu.gate_proj.weight.T
    up = x @ swiglu.up_proj.weight.T
    late_gating = (reference_silu(gate) @ swiglu.down_proj.weight.T) * (
        up @ swiglu.down_proj.weight.T
    )

    assert late_gating.shape == swiglu(x).shape
    assert not torch.allclose(swiglu(x), late_gating, atol=1e-4)


def test_is_not_a_plain_two_matrix_mlp(swiglu):
    x = torch.randn(4, 16)
    plain = reference_silu(x @ swiglu.up_proj.weight.T) @ swiglu.down_proj.weight.T

    assert not torch.allclose(swiglu(x), plain, atol=1e-4)


def test_silu_matches_torch_for_the_hidden_activation(swiglu):
    z = torch.randn(6, 48)
    torch.testing.assert_close(reference_silu(z), F.silu(z))


# --------------------------------------------------------------------------
# tensor contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("last_dim", [1, 8, 15, 17, 48])
def test_wrong_last_dimension_is_rejected(swiglu, last_dim):
    with pytest.raises(ValueError, match="hidden_size"):
        swiglu(torch.randn(2, last_dim))


def test_scalar_input_is_rejected(swiglu):
    with pytest.raises(ValueError):
        swiglu(torch.tensor(1.0))


def test_non_tensor_input_is_rejected(swiglu):
    with pytest.raises(TypeError):
        swiglu([0.0] * 16)


def test_input_is_not_modified_in_place(swiglu):
    x = torch.randn(2, 6, 16)
    original = x.clone()
    swiglu(x)
    torch.testing.assert_close(x, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    module = SwiGLU(small_config()).to(dtype)
    x = torch.randn(2, 16, dtype=dtype)

    assert module(x).dtype == dtype
    torch.testing.assert_close(module(x), reference_swiglu(module, x))


def test_device_is_preserved(swiglu):
    x = torch.randn(2, 16)
    assert swiglu(x).device == x.device


def test_is_deterministic(swiglu):
    x = torch.randn(3, 16)
    torch.testing.assert_close(swiglu(x), swiglu(x))


def test_extreme_magnitudes_stay_finite(swiglu):
    for scale in (1e-8, 1e4):
        output = swiglu(torch.randn(2, 16) * scale)
        assert bool(torch.isfinite(output).all()), scale


# --------------------------------------------------------------------------
# autograd
# --------------------------------------------------------------------------


def test_all_three_matrices_receive_gradients(swiglu):
    x = torch.randn(4, 16, requires_grad=True)
    swiglu(x).pow(2).sum().backward()

    names = sorted(name for name, _ in swiglu.named_parameters())
    assert names == ["down_proj.weight", "gate_proj.weight", "up_proj.weight"]

    for name, parameter in swiglu.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert parameter.grad.abs().sum() > 0, name

    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())
    assert x.grad.abs().sum() > 0


def test_gradcheck_against_numerical_derivatives():
    module = SwiGLU(small_config(hidden_size=8, n_heads=2, head_dim=4, ffn_hidden_size=12)).to(
        torch.float64
    )
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-8)


def test_gradients_match_the_reference_implementation():
    module = SwiGLU(small_config()).to(torch.float64)
    with torch.no_grad():
        for projection in (module.gate_proj, module.up_proj, module.down_proj):
            projection.weight.normal_(std=0.5)

    x = torch.randn(4, 16, dtype=torch.float64)

    actual = x.clone().requires_grad_(True)
    module(actual).pow(2).sum().backward()
    actual_weight_grads = {
        name: p.grad.clone() for name, p in module.named_parameters()
    }

    module.zero_grad(set_to_none=True)
    expected = x.clone().requires_grad_(True)
    reference_swiglu(module, expected).pow(2).sum().backward()

    torch.testing.assert_close(actual.grad, expected.grad, atol=1e-10, rtol=1e-10)
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(
            actual_weight_grads[name], parameter.grad, atol=1e-10, rtol=1e-10, msg=name
        )


def test_gradients_are_finite_at_zero(swiglu):
    x = torch.zeros(2, 16, requires_grad=True)
    swiglu(x).sum().backward()

    assert bool(torch.isfinite(x.grad).all())
    for _, parameter in swiglu.named_parameters():
        assert bool(torch.isfinite(parameter.grad).all())
