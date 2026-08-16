"""Frozen architecture description for the decoder-only Transformer.

This module contains no tensors and no ``torch`` import.  It is the one place
that states what the model *is*, so that every component built after it --
RMSNorm, RoPE, attention, SwiGLU, the block, the full stack -- reads its shape
from a single object rather than from a constructor argument someone typed
twice.

Why a config module comes before any layer
------------------------------------------
The architecture is already constrained by decisions that are now immutable:

* the tokenizer is frozen at 24,000 ids, so ``vocab_size`` is not a
  hyperparameter any more;
* the dataset was written as 2048-token windows, so ``context_length`` is not
  either.

Getting either wrong does not raise.  A model built with the wrong vocabulary
trains happily and emits garbage for the ids it never allocated; a model built
with the wrong context silently mismatches the data loader's window geometry.
:meth:`ModelConfig.validate_against_dataset` exists so those two numbers are
checked against the frozen dataset manifest instead of trusted.

Parameter count as a design assertion
-------------------------------------
:meth:`ModelConfig.parameter_count` derives the total analytically from the
config alone.  The intended use is to assert the *implementation* matches the
*specification*: once ``Transformer`` exists, its real
``sum(p.numel() for p in model.parameters())`` must equal this number.  A
mismatch means a layer was wired differently from the design -- an untied
head, an unexpected bias, a wrong FFN width -- which is otherwise a very quiet
class of bug.

For the frozen v0.1 architecture that total is 32,741,888.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


#: The tokenizer is frozen at this vocabulary; see artifacts/tokenizer-E011.
FROZEN_VOCAB_SIZE = 24_000

#: The tokenized dataset was written as windows of this many input positions.
FROZEN_CONTEXT_LENGTH = 2_048


def _require_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


@dataclass(frozen=True)
class ModelConfig:
    """Complete architecture of the v0.1 base language model.

    Defaults are the frozen v0.1 architecture.  Every field is validated on
    construction, so an invalid configuration cannot reach a layer.
    """

    vocab_size: int = FROZEN_VOCAB_SIZE
    context_length: int = FROZEN_CONTEXT_LENGTH

    n_layers: int = 6
    hidden_size: int = 512
    n_heads: int = 8
    head_dim: int = 64
    ffn_hidden_size: int = 1_536

    rms_norm_eps: float = 1e-6

    #: RoPE angular-frequency base.  Architectural, not an implementation
    #: detail: it sets how fast each coordinate pair rotates with position, so
    #: a model trained at one theta cannot read positions encoded at another.
    #: v0.1 rotates the entire head (rotary_dim == head_dim).
    rope_theta: float = 10_000.0

    tie_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "context_length",
            "n_layers",
            "hidden_size",
            "n_heads",
            "head_dim",
            "ffn_hidden_size",
        ):
            _require_positive_int(name, getattr(self, name))

        for name in ("tie_embeddings", "attention_bias", "mlp_bias"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

        for name in ("rms_norm_eps", "rope_theta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not value > 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")

        if self.hidden_size % self.n_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by n_heads "
                f"({self.n_heads})"
            )

        expected_head_dim = self.hidden_size // self.n_heads
        if self.head_dim != expected_head_dim:
            raise ValueError(
                f"head_dim ({self.head_dim}) must equal hidden_size // n_heads "
                f"({expected_head_dim})"
            )

        # RoPE rotates coordinate pairs, so an odd head dimension would leave a
        # channel with no partner and silently drop it from the rotation.
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even for RoPE pairing"
            )

    # ----------------------------------------------------------------
    # derived geometry
    # ----------------------------------------------------------------

    @property
    def attention_output_size(self) -> int:
        """Concatenated head width; equals ``hidden_size`` without GQA/MQA."""

        return self.n_heads * self.head_dim

    @property
    def max_position(self) -> int:
        return self.context_length

    # ----------------------------------------------------------------
    # parameter accounting
    # ----------------------------------------------------------------

    def embedding_parameters(self) -> int:
        return self.vocab_size * self.hidden_size

    def attention_parameters(self) -> int:
        """Q, K, V and output projections for one block."""

        projection = self.hidden_size * self.attention_output_size
        total = 3 * projection + self.attention_output_size * self.hidden_size
        if self.attention_bias:
            total += 3 * self.attention_output_size + self.hidden_size
        return total

    def feedforward_parameters(self) -> int:
        """SwiGLU: gate and up projections in, down projection out."""

        total = 3 * self.hidden_size * self.ffn_hidden_size
        if self.mlp_bias:
            total += 2 * self.ffn_hidden_size + self.hidden_size
        return total

    def norm_parameters(self) -> int:
        """RMSNorm has a scale per channel and no bias."""

        return self.hidden_size

    def block_parameters(self) -> int:
        """One TransformerBlock: attention, feed-forward, two norms."""

        return (
            self.attention_parameters()
            + self.feedforward_parameters()
            + 2 * self.norm_parameters()
        )

    def lm_head_parameters(self) -> int:
        """Zero when the output projection reuses the embedding matrix."""

        return 0 if self.tie_embeddings else self.vocab_size * self.hidden_size

    def parameter_count(self) -> int:
        """Total trainable parameters implied by this configuration."""

        return (
            self.embedding_parameters()
            + self.n_layers * self.block_parameters()
            + self.norm_parameters()
            + self.lm_head_parameters()
        )

    def parameter_breakdown(self) -> dict[str, int]:
        """Per-component counts, for reporting and for locating a mismatch."""

        blocks = self.n_layers * self.block_parameters()
        return {
            "embedding": self.embedding_parameters(),
            "attention_per_block": self.attention_parameters(),
            "feedforward_per_block": self.feedforward_parameters(),
            "norms_per_block": 2 * self.norm_parameters(),
            "block_total": self.block_parameters(),
            "all_blocks": blocks,
            "final_norm": self.norm_parameters(),
            "lm_head": self.lm_head_parameters(),
            "total": self.parameter_count(),
        }

    # ----------------------------------------------------------------
    # agreement with frozen upstream artifacts
    # ----------------------------------------------------------------

    def validate_against_dataset(self, dataset_manifest: dict[str, Any]) -> None:
        """Raise unless this config matches a tokenized dataset manifest.

        Checks the two fields the data has already decided: the tokenizer
        vocabulary the shards were written against, and the window geometry the
        shards were cut to.  Both are silent failures if wrong, which is why
        they are checked rather than assumed.
        """

        if not isinstance(dataset_manifest, dict):
            raise TypeError("dataset_manifest must be a dict")

        tokenizer = dataset_manifest.get("tokenizer")
        geometry = dataset_manifest.get("training_geometry")
        if not isinstance(tokenizer, dict) or not isinstance(geometry, dict):
            raise ValueError(
                "dataset manifest is missing 'tokenizer' or 'training_geometry'"
            )

        dataset_vocab = tokenizer.get("vocab_size")
        if dataset_vocab != self.vocab_size:
            raise ValueError(
                f"vocab_size mismatch: model {self.vocab_size}, dataset {dataset_vocab!r}"
            )

        dataset_context = geometry.get("context_length")
        if dataset_context != self.context_length:
            raise ValueError(
                f"context_length mismatch: model {self.context_length}, "
                f"dataset {dataset_context!r}"
            )

        window_tokens = geometry.get("window_tokens")
        if window_tokens != self.context_length + 1:
            raise ValueError(
                f"dataset window_tokens is {window_tokens!r}, expected "
                f"{self.context_length + 1} for {self.context_length} predictions"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        known = {f for f in cls().to_dict()}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown ModelConfig fields: {unknown}")
        return cls(**payload)

    def __repr__(self) -> str:
        return (
            f"ModelConfig(vocab_size={self.vocab_size:,}, "
            f"context_length={self.context_length:,}, "
            f"n_layers={self.n_layers}, hidden_size={self.hidden_size}, "
            f"n_heads={self.n_heads}, head_dim={self.head_dim}, "
            f"ffn_hidden_size={self.ffn_hidden_size}, "
            f"tie_embeddings={self.tie_embeddings}, "
            f"parameters={self.parameter_count():,})"
        )
