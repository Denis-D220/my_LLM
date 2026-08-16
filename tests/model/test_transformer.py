"""Full-model correctness gate.

This is the last set of tests before the model is allowed near real data, so
it re-proves at whole-model scale the properties already established for the
components. Causality holding inside one attention module is necessary;
causality surviving an embedding, six residual blocks, a normalizer and a tied
projection is what actually matters.

The tied-head tests are structural rather than numerical. A head implemented
by copying the embedding passes any test that compares values at
initialization and fails silently after the first optimizer step, so these
assert tensor identity and count vocabulary-sized matrices instead.
"""

from __future__ import annotations

import math

import pytest
import torch

from llm.model import ModelConfig
from llm.model.rmsnorm import RMSNorm
from llm.model.transformer import INIT_STD, Transformer, causal_lm_loss


def small_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=128,
        context_length=32,
        n_layers=3,
        hidden_size=16,
        n_heads=4,
        head_dim=4,
        ffn_hidden_size=48,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(20260815)


@pytest.fixture
def model() -> Transformer:
    return Transformer(small_config())


def random_ids(model: Transformer, batch: int, sequence: int) -> torch.Tensor:
    return torch.randint(0, model.config.vocab_size, (batch, sequence))


# --------------------------------------------------------------------------
# the parameter budget
# --------------------------------------------------------------------------


def test_parameter_count_is_exactly_the_frozen_total():
    model = Transformer(ModelConfig())

    assert model.parameter_count() == 32_741_888
    assert sum(p.numel() for p in model.parameters()) == 32_741_888
    assert model.parameter_count() == ModelConfig().parameter_count()


def test_the_budget_decomposes_as_designed():
    config = ModelConfig()
    model = Transformer(config)

    embedding = model.token_embedding.weight.numel()
    blocks = sum(p.numel() for block in model.blocks for p in block.parameters())
    final = sum(p.numel() for p in model.final_norm.parameters())

    assert embedding == 12_288_000
    assert blocks == 6 * 3_408_896 == 20_453_376
    assert final == 512
    assert embedding + blocks + final == model.parameter_count() == 32_741_888


def test_there_are_exactly_six_distinct_blocks():
    model = Transformer(ModelConfig())

    assert len(model.blocks) == 6
    assert len({id(block) for block in model.blocks}) == 6

    pointers = [block.attention.q_proj.weight.data_ptr() for block in model.blocks]
    assert len(set(pointers)) == 6


def test_blocks_hold_independent_weights():
    model = Transformer(small_config())
    ids = random_ids(model, 1, 8)
    model(ids).pow(2).sum().backward()

    grads = [block.attention.q_proj.weight.grad for block in model.blocks]
    for earlier, later in zip(grads, grads[1:]):
        assert not torch.allclose(earlier, later)


# --------------------------------------------------------------------------
# weight tying
# --------------------------------------------------------------------------


def test_the_output_projection_is_the_embedding_tensor():
    model = Transformer(ModelConfig())

    assert model.lm_head is None
    assert model.output_weight is model.token_embedding.weight
    assert model.output_weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_exactly_one_vocabulary_sized_matrix_exists():
    model = Transformer(ModelConfig())

    big = [p for p in model.parameters() if p.shape == (24_000, 512)]
    assert len(big) == 1
    assert big[0] is model.token_embedding.weight


def test_state_dict_has_no_separate_head():
    model = Transformer(ModelConfig())
    keys = set(model.state_dict())

    assert "token_embedding.weight" in keys
    assert not any(key.startswith("lm_head") for key in keys)


def test_editing_the_embedding_changes_the_logits(model):
    """The tie is live, not a snapshot taken at construction."""

    ids = random_ids(model, 1, 6)
    before = model(ids).clone()

    with torch.no_grad():
        model.token_embedding.weight.mul_(1.5)

    assert not torch.allclose(before, model(ids), atol=1e-4)


def test_untying_adds_exactly_one_embedding_matrix():
    config = ModelConfig(tie_embeddings=False)
    model = Transformer(config)

    assert model.lm_head is not None
    assert model.output_weight is model.lm_head.weight
    assert model.output_weight is not model.token_embedding.weight
    assert model.parameter_count() == 32_741_888 + 12_288_000
    assert model.parameter_count() == config.parameter_count()


# --------------------------------------------------------------------------
# forward contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("sequence", [1, 7, 32])
def test_logits_shape(model, batch, sequence):
    ids = random_ids(model, batch, sequence)
    assert model(ids).shape == (batch, sequence, model.config.vocab_size)


def test_logits_shape_at_the_frozen_scale():
    model = Transformer(ModelConfig())
    ids = torch.randint(0, 24_000, (1, 2048))

    logits = model(ids)
    assert logits.shape == (1, 2048, 24_000)
    assert bool(torch.isfinite(logits).all())


def test_sequence_length_one_works(model):
    assert model(random_ids(model, 1, 1)).shape == (1, 1, model.config.vocab_size)


def test_highest_valid_token_id_is_accepted():
    model = Transformer(ModelConfig())
    ids = torch.full((1, 4), 23_999, dtype=torch.long)

    assert bool(torch.isfinite(model(ids)).all())


def test_out_of_range_token_id_is_rejected():
    model = Transformer(ModelConfig())

    with pytest.raises(ValueError, match=r"outside \[0, 24000\)"):
        model(torch.full((1, 4), 24_000, dtype=torch.long))


def test_negative_token_id_is_rejected(model):
    ids = random_ids(model, 1, 4)
    ids[0, 2] = -1

    with pytest.raises(ValueError, match="outside"):
        model(ids)


def test_sequence_beyond_the_context_is_rejected():
    model = Transformer(ModelConfig())
    with pytest.raises(ValueError, match="exceed context_length"):
        model(torch.zeros(1, 2049, dtype=torch.long))


@pytest.mark.parametrize("shape", [(4,), (1, 2, 3)])
def test_wrong_input_rank_is_rejected(model, shape):
    with pytest.raises(ValueError, match=r"\(batch, sequence\)"):
        model(torch.zeros(shape, dtype=torch.long))


def test_float_input_is_rejected(model):
    with pytest.raises(TypeError, match="integer"):
        model(torch.zeros(1, 4))


def test_empty_input_is_rejected(model):
    with pytest.raises(ValueError, match="empty"):
        model(torch.zeros(1, 0, dtype=torch.long))


def test_is_deterministic(model):
    ids = random_ids(model, 2, 8)
    torch.testing.assert_close(model(ids), model(ids))


def test_examples_in_a_batch_do_not_interact(model):
    a = random_ids(model, 1, 8)
    b = random_ids(model, 1, 8)

    together = model(torch.cat([a, b], dim=0))
    torch.testing.assert_close(together[0:1], model(a), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(together[1:2], model(b), atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# causality through the whole stack
# --------------------------------------------------------------------------


def test_changing_a_later_token_cannot_change_earlier_logits(model):
    """The property the entire pretraining objective rests on."""

    ids = random_ids(model, 1, 32)
    baseline = model(ids)

    modified = ids.clone()
    modified[0, 20:] = random_ids(model, 1, 12)[0]
    changed = model(modified)

    torch.testing.assert_close(baseline[:, :20], changed[:, :20], atol=1e-5, rtol=1e-5)


def test_changing_token_100_cannot_affect_logits_at_token_50():
    model = Transformer(small_config(context_length=128))
    ids = random_ids(model, 1, 128)

    baseline = model(ids)
    modified = ids.clone()
    modified[0, 100] = (int(ids[0, 100]) + 37) % model.config.vocab_size

    torch.testing.assert_close(
        baseline[:, 50], model(modified)[:, 50], atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("cut", [1, 5, 16, 31])
def test_every_prefix_is_independent_of_its_suffix(model, cut):
    ids = random_ids(model, 1, 32)

    modified = ids.clone()
    modified[0, cut:] = random_ids(model, 1, 32 - cut)[0]

    torch.testing.assert_close(
        model(ids)[:, :cut], model(modified)[:, :cut], atol=1e-5, rtol=1e-5
    )


def test_a_prefix_computed_alone_matches_the_same_prefix_in_context(model):
    """Generation depends on this: appending a token must not rewrite history."""

    ids = random_ids(model, 1, 16)
    full = model(ids)

    for cut in (1, 4, 16):
        torch.testing.assert_close(
            model(ids[:, :cut]), full[:, :cut], atol=1e-5, rtol=1e-5
        )


def test_gradient_of_earlier_logits_never_reaches_later_embeddings(model):
    """Causality in the backward direction, through all layers."""

    ids = random_ids(model, 1, 16)
    embedded = model.token_embedding(ids).detach().requires_grad_(True)

    hidden = embedded
    for block in model.blocks:
        hidden = block(hidden)
    logits = torch.nn.functional.linear(model.final_norm(hidden), model.output_weight)
    logits[0, 5].sum().backward()

    assert bool((embedded.grad[0, 6:] == 0).all())
    assert embedded.grad[0, :6].abs().sum() > 0


# --------------------------------------------------------------------------
# initialization policy
# --------------------------------------------------------------------------


def test_linear_and_embedding_weights_use_the_policy_std():
    model = Transformer(ModelConfig())

    for name, parameter in model.named_parameters():
        if name.endswith("norm.weight"):
            continue
        assert parameter.std().item() == pytest.approx(INIT_STD, rel=0.05), name
        assert parameter.mean().item() == pytest.approx(0.0, abs=1e-3), name


def test_all_rmsnorm_scales_start_at_one():
    model = Transformer(ModelConfig())

    norms = [m for m in model.modules() if isinstance(m, RMSNorm)]
    assert len(norms) == 6 * 2 + 1
    for norm in norms:
        torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))


def test_reset_parameters_is_reproducible_under_a_seed():
    torch.manual_seed(7)
    a = Transformer(small_config())
    torch.manual_seed(7)
    b = Transformer(small_config())

    for (name, left), (_, right) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(left, right, msg=name)


def test_reset_parameters_can_be_reapplied(model):
    with torch.no_grad():
        model.token_embedding.weight.fill_(9.0)
        model.blocks[0].attention_norm.weight.fill_(4.0)

    model.reset_parameters()

    assert model.token_embedding.weight.std().item() == pytest.approx(INIT_STD, rel=0.1)
    torch.testing.assert_close(
        model.blocks[0].attention_norm.weight,
        torch.ones_like(model.blocks[0].attention_norm.weight),
    )


def test_initial_loss_is_close_to_uniform_cross_entropy():
    """A randomly initialized 24k-way predictor should score near ln(24000)."""

    model = Transformer(ModelConfig())
    ids = torch.randint(0, 24_000, (2, 64))
    targets = torch.randint(0, 24_000, (2, 64))

    loss = causal_lm_loss(model(ids), targets).item()
    uniform = math.log(24_000)

    assert uniform - 0.6 < loss < uniform + 0.2, (loss, uniform)


# --------------------------------------------------------------------------
# the loss
# --------------------------------------------------------------------------


def test_loss_matches_manual_cross_entropy(model):
    ids = random_ids(model, 2, 8)
    targets = random_ids(model, 2, 8)
    logits = model(ids)

    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.config.vocab_size), targets.reshape(-1)
    )
    torch.testing.assert_close(causal_lm_loss(logits, targets), expected)


def test_loss_is_a_finite_scalar(model):
    ids = random_ids(model, 2, 8)
    loss = causal_lm_loss(model(ids), random_ids(model, 2, 8))

    assert loss.ndim == 0
    assert bool(torch.isfinite(loss))
    assert loss.item() > 0


def test_perfect_prediction_gives_near_zero_loss(model):
    """Sanity on orientation: confident correct logits must score ~0."""

    targets = random_ids(model, 1, 5)
    logits = torch.zeros(1, 5, model.config.vocab_size)
    logits.scatter_(2, targets.unsqueeze(-1), 40.0)

    assert causal_lm_loss(logits, targets).item() == pytest.approx(0.0, abs=1e-5)


def test_loss_does_not_shift_the_targets(model):
    """The dataset already shifted them; shifting again would lose a token."""

    targets = random_ids(model, 1, 4)
    logits = torch.zeros(1, 4, model.config.vocab_size)
    logits.scatter_(2, targets.unsqueeze(-1), 40.0)

    # Scoring against the targets as given is ~0; scoring against a shifted
    # copy is not.
    assert causal_lm_loss(logits, targets).item() < 1e-4
    shifted = targets.roll(1, dims=1)
    assert causal_lm_loss(logits, shifted).item() > 1.0


def test_ignore_index_masks_positions(model):
    targets = random_ids(model, 1, 4)
    logits = torch.zeros(1, 4, model.config.vocab_size)
    logits.scatter_(2, targets.unsqueeze(-1), 40.0)

    masked = targets.clone()
    masked[0, 2] = -100

    assert causal_lm_loss(logits, masked).item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    "logits_shape, targets_shape",
    [((2, 4), (2, 4)), ((2, 4, 8), (2, 5)), ((2, 4, 8), (2, 4, 8))],
)
def test_mismatched_loss_shapes_are_rejected(logits_shape, targets_shape):
    with pytest.raises(ValueError):
        causal_lm_loss(
            torch.randn(logits_shape), torch.zeros(targets_shape, dtype=torch.long)
        )


def test_loss_rejects_non_tensors():
    with pytest.raises(TypeError):
        causal_lm_loss([1.0], torch.zeros(1, 1, dtype=torch.long))


# --------------------------------------------------------------------------
# autograd over the whole model
# --------------------------------------------------------------------------


def test_backward_reaches_every_parameter(model):
    ids = random_ids(model, 2, 8)
    targets = random_ids(model, 2, 8)

    loss = causal_lm_loss(model(ids), targets)
    loss.backward()

    assert bool(torch.isfinite(loss))
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert parameter.grad.abs().sum() > 0, name


def test_no_gradient_is_nan_or_inf_at_the_frozen_scale():
    model = Transformer(ModelConfig())
    ids = torch.randint(0, 24_000, (1, 64))
    targets = torch.randint(0, 24_000, (1, 64))

    causal_lm_loss(model(ids), targets).backward()

    for name, parameter in model.named_parameters():
        assert not bool(torch.isnan(parameter.grad).any()), name
        assert not bool(torch.isinf(parameter.grad).any()), name


def test_the_tied_embedding_receives_gradient_from_both_roles(model):
    """One tensor, two jobs: lookup and projection must both contribute."""

    ids = random_ids(model, 1, 6)
    targets = random_ids(model, 1, 6)

    causal_lm_loss(model(ids), targets).backward()
    grad = model.token_embedding.weight.grad

    # The projection role touches every vocabulary row; the lookup role only
    # the rows that appear in the batch. A grad that is non-zero far outside
    # the used ids can only have come through the output projection.
    used = set(ids.flatten().tolist())
    unused = [i for i in range(model.config.vocab_size) if i not in used]
    assert grad[list(used)].abs().sum() > 0
    assert grad[unused].abs().sum() > 0


def test_a_single_optimizer_step_reduces_the_loss_on_one_batch(model):
    """The smallest possible end-to-end proof that learning is wired up."""

    ids = random_ids(model, 2, 8)
    targets = random_ids(model, 2, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    before = causal_lm_loss(model(ids), targets)
    optimizer.zero_grad(set_to_none=True)
    before.backward()
    optimizer.step()
    after = causal_lm_loss(model(ids), targets)

    assert after.item() < before.item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    model = Transformer(small_config()).to(dtype)
    logits = model(random_ids(model, 1, 5))
    assert logits.dtype == dtype
