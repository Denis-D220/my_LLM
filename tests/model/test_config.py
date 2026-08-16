"""Tests for the frozen architecture description.

The parameter-count tests are deliberately written against hand-computed
numbers rather than against the implementation's own arithmetic.  A test that
asserts ``config.parameter_count() == config.parameter_count()`` proves
nothing; these assert the specific totals the v0.1 design implies, so a change
to the accounting has to be argued for rather than absorbed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.model import FROZEN_CONTEXT_LENGTH, FROZEN_VOCAB_SIZE, ModelConfig


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_MANIFEST = (
    REPOSITORY_ROOT / "data" / "tokenized" / "v0.1" / "dataset_manifest.json"
)

# 24,000 x 512 embedding, six blocks of (4 x 512^2 attention + 3 x 512 x 1536
# SwiGLU + 2 x 512 norm), one final norm, tied output projection.
EXPECTED_V01_PARAMETERS = 32_741_888


def test_defaults_are_the_frozen_architecture():
    config = ModelConfig()

    assert config.vocab_size == FROZEN_VOCAB_SIZE == 24_000
    assert config.context_length == FROZEN_CONTEXT_LENGTH == 2_048
    assert config.n_layers == 6
    assert config.hidden_size == 512
    assert config.n_heads == 8
    assert config.head_dim == 64
    assert config.ffn_hidden_size == 1_536
    assert config.rms_norm_eps == pytest.approx(1e-6)
    assert config.tie_embeddings is True
    assert config.attention_bias is False
    assert config.mlp_bias is False


def test_head_geometry_invariants_hold():
    config = ModelConfig()

    assert config.hidden_size % config.n_heads == 0
    assert config.head_dim == config.hidden_size // config.n_heads
    assert config.attention_output_size == config.hidden_size
    assert config.head_dim % 2 == 0


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"hidden_size": 500}, "divisible"),
        ({"head_dim": 32}, "head_dim"),
        ({"n_layers": 0}, "> 0"),
        ({"vocab_size": -1}, "> 0"),
        ({"rms_norm_eps": 0.0}, "> 0"),
        # 8 // 8 == 1: divisibility and head_dim both hold, so the odd-head_dim
        # guard is what has to reject this one.
        ({"hidden_size": 8, "n_heads": 8, "head_dim": 1}, "even"),
    ],
)
def test_invalid_configurations_are_rejected(overrides, message):
    with pytest.raises((ValueError, TypeError), match=message):
        ModelConfig(**overrides)


@pytest.mark.parametrize("field", ["tie_embeddings", "attention_bias", "mlp_bias"])
def test_flag_fields_must_be_booleans(field):
    with pytest.raises(TypeError):
        ModelConfig(**{field: 1})


def test_non_integer_dimensions_are_rejected():
    with pytest.raises(TypeError):
        ModelConfig(hidden_size=512.0)


# --------------------------------------------------------------------------
# parameter accounting
# --------------------------------------------------------------------------


def test_component_counts_match_hand_computation():
    config = ModelConfig()

    assert config.embedding_parameters() == 24_000 * 512 == 12_288_000
    assert config.attention_parameters() == 4 * 512 * 512 == 1_048_576
    assert config.feedforward_parameters() == 3 * 512 * 1_536 == 2_359_296
    assert config.norm_parameters() == 512
    assert config.block_parameters() == 1_048_576 + 2_359_296 + 1_024 == 3_408_896
    assert config.lm_head_parameters() == 0


def test_total_parameter_count_is_the_frozen_v01_number():
    assert ModelConfig().parameter_count() == EXPECTED_V01_PARAMETERS

    # Stated the other way round, so a change to any single term is visible.
    assert EXPECTED_V01_PARAMETERS == 12_288_000 + 6 * 3_408_896 + 512


def test_breakdown_sums_to_the_total():
    config = ModelConfig()
    breakdown = config.parameter_breakdown()

    assert (
        breakdown["embedding"]
        + breakdown["all_blocks"]
        + breakdown["final_norm"]
        + breakdown["lm_head"]
        == breakdown["total"]
        == config.parameter_count()
    )
    assert breakdown["all_blocks"] == config.n_layers * breakdown["block_total"]


def test_untying_embeddings_adds_exactly_one_embedding_matrix():
    tied = ModelConfig()
    untied = ModelConfig(tie_embeddings=False)

    assert untied.parameter_count() - tied.parameter_count() == 24_000 * 512


def test_biases_add_exactly_the_expected_vectors():
    base = ModelConfig()

    with_attention_bias = ModelConfig(attention_bias=True)
    assert (
        with_attention_bias.attention_parameters() - base.attention_parameters()
        == 4 * 512
    )

    with_mlp_bias = ModelConfig(mlp_bias=True)
    assert (
        with_mlp_bias.feedforward_parameters() - base.feedforward_parameters()
        == 2 * 1_536 + 512
    )


# --------------------------------------------------------------------------
# agreement with the frozen dataset
# --------------------------------------------------------------------------


def synthetic_manifest(
    *, vocab_size: int = 24_000, context_length: int = 2_048
) -> dict:
    return {
        "tokenizer": {"vocab_size": vocab_size},
        "training_geometry": {
            "context_length": context_length,
            "window_tokens": context_length + 1,
        },
    }


def test_matching_dataset_manifest_is_accepted():
    ModelConfig().validate_against_dataset(synthetic_manifest())


@pytest.mark.parametrize(
    "manifest, message",
    [
        (synthetic_manifest(vocab_size=32_000), "vocab_size mismatch"),
        (synthetic_manifest(context_length=1_024), "context_length mismatch"),
        ({"tokenizer": {"vocab_size": 24_000}}, "missing"),
    ],
)
def test_mismatched_dataset_manifest_is_rejected(manifest, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig().validate_against_dataset(manifest)


def test_inconsistent_window_tokens_are_rejected():
    manifest = synthetic_manifest()
    manifest["training_geometry"]["window_tokens"] = 2_048

    with pytest.raises(ValueError, match="window_tokens"):
        ModelConfig().validate_against_dataset(manifest)


@pytest.mark.skipif(
    not DATASET_MANIFEST.is_file(),
    reason="tokenized dataset v0.1 is not present in this checkout",
)
def test_default_config_matches_the_real_frozen_dataset():
    manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    ModelConfig().validate_against_dataset(manifest)


# --------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------


def test_round_trips_through_a_dict():
    config = ModelConfig()
    assert ModelConfig.from_dict(config.to_dict()) == config


def test_unknown_fields_are_rejected():
    payload = ModelConfig().to_dict()
    payload["n_kv_heads"] = 4

    with pytest.raises(ValueError, match="unknown ModelConfig fields"):
        ModelConfig.from_dict(payload)


def test_config_is_immutable():
    config = ModelConfig()
    with pytest.raises(Exception):
        config.hidden_size = 1_024  # type: ignore[misc]
