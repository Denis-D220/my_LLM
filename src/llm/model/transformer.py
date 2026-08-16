"""The complete decoder-only Transformer.

::

    input_ids                (batch, sequence)
      -> token embedding     (batch, sequence, 512)
      -> 6 x TransformerBlock
      -> final RMSNorm
      -> tied output projection
      -> logits              (batch, sequence, 24000)

Frozen v0.1 budget
------------------
============================  ==========
token embedding               12,288,000
6 x TransformerBlock          20,453,376
final RMSNorm                        512
output projection (tied)               0
----------------------------  ----------
total                         32,741,888
============================  ==========

Weight tying is structural, not numerical
-----------------------------------------
The output projection is computed as ``F.linear(hidden, token_embedding.weight)``.
There is no second matrix anywhere -- not one initialized to the same values,
not one kept in sync.  A tied head implemented by copying weights drifts apart
on the first optimizer step and produces a model that trains, converges, and is
quietly 12.3M parameters larger than its own documentation claims.  Here the
tie cannot come undone because there is nothing to untie.

``test_the_output_projection_is_the_embedding_tensor`` asserts identity by
``data_ptr``, and ``test_exactly_one_vocabulary_sized_matrix_exists`` counts
matrices of shape ``(24000, 512)`` and requires exactly one.

Initialization
--------------
:meth:`Transformer.reset_parameters` is the single model-wide policy, replacing
the per-module defaults that the components deliberately did not invent for
themselves.  v0.1 uses:

* every ``nn.Linear`` weight and the token embedding: ``N(0, 0.02)``
* every RMSNorm scale: ``1.0``
* no depth-dependent residual scaling

That last omission is deliberate and worth revisiting before the full
pretraining run: at 6 layers the residual stream grows slowly enough that
depth scaling is not required for stability, but it is the first thing to try
if a deeper configuration ever misbehaves.

The 0.02 standard deviation interacts with weight tying.  The embedding is
simultaneously a lookup table and the output projection, so its scale sets the
initial logit spread directly.  With a unit-RMS hidden state and 512
contracted dimensions, logits start with a standard deviation near
``sqrt(512) * 0.02 = 0.45``, which keeps the initial distribution close to
uniform and the initial loss just under ``ln(24000) = 10.09``.  Retuning this
value moves the tiny-overfit baseline with it.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from llm.model.block import TransformerBlock
from llm.model.config import ModelConfig
from llm.model.rmsnorm import RMSNorm


#: Standard deviation for the v0.1 model-wide initialization policy.
INIT_STD = 0.02


def causal_lm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean next-token cross-entropy.

    Kept outside :class:`Transformer` on purpose.  The model's job is to turn
    ids into logits; scoring those logits against targets is a separate
    concern, and separating them means each can be tested without the other.

    ``logits`` is ``(batch, sequence, vocab_size)`` and ``targets`` is
    ``(batch, sequence)``.  The dataset already supplies targets shifted by one
    position, so no shifting happens here -- doing it in both places is a
    classic off-by-one that costs one token of context per example and is
    almost invisible in the loss curve.
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a Tensor, got {type(logits).__name__}")
    if not isinstance(targets, torch.Tensor):
        raise TypeError(f"targets must be a Tensor, got {type(targets).__name__}")
    if logits.ndim != 3:
        raise ValueError(
            f"logits must be (batch, sequence, vocab_size), got {tuple(logits.shape)}"
        )
    if targets.ndim != 2:
        raise ValueError(
            f"targets must be (batch, sequence), got {tuple(targets.shape)}"
        )
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)[:2]} and targets {tuple(targets.shape)} "
            "disagree on batch/sequence"
        )

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


class Transformer(nn.Module):
    """Decoder-only language model over the frozen v0.1 architecture."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"config must be a ModelConfig, got {type(config).__name__}"
            )

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm.from_config(config)

        # Only materialized when the head is deliberately untied; the tied path
        # has no second matrix at all.
        self.lm_head = (
            None
            if config.tie_embeddings
            else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )

        self.reset_parameters()

    @classmethod
    def from_config(cls, config: ModelConfig) -> "Transformer":
        return cls(config)

    @property
    def output_weight(self) -> torch.Tensor:
        """The matrix used to project hidden states onto the vocabulary."""

        if self.lm_head is None:
            return self.token_embedding.weight
        return self.lm_head.weight

    def reset_parameters(self) -> None:
        """Apply the single model-wide initialization policy."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            elif isinstance(module, RMSNorm):
                module.reset_parameters()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _validate_input(self, input_ids: torch.Tensor, start_pos: int) -> None:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                f"input_ids must be a Tensor, got {type(input_ids).__name__}"
            )
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be (batch, sequence), got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype not in (torch.long, torch.int32, torch.int64):
            raise TypeError(
                f"input_ids must be an integer tensor, got {input_ids.dtype}"
            )
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")

        # Checked explicitly: nn.Embedding's own out-of-range failure is an
        # opaque index error, and on some devices it is undefined behaviour
        # rather than an error at all.
        minimum = int(input_ids.min())
        maximum = int(input_ids.max())
        if minimum < 0 or maximum >= self.config.vocab_size:
            raise ValueError(
                f"input_ids contains ids outside [0, {self.config.vocab_size}): "
                f"observed [{minimum}, {maximum}]"
            )

        if not isinstance(start_pos, int) or isinstance(start_pos, bool):
            raise TypeError(
                f"start_pos must be an integer, got {type(start_pos).__name__}"
            )
        if start_pos < 0:
            raise ValueError(f"start_pos must be >= 0, got {start_pos}")

    def forward(
        self, input_ids: torch.Tensor, *, start_pos: int = 0
    ) -> torch.Tensor:
        """Map token ids to next-token logits.

        Returns ``(batch, sequence, vocab_size)``.
        """

        self._validate_input(input_ids, start_pos)

        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, start_pos=start_pos)
        hidden = self.final_norm(hidden)

        return F.linear(hidden, self.output_weight)

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.config.vocab_size}, "
            f"n_layers={self.config.n_layers}, "
            f"hidden_size={self.config.hidden_size}, "
            f"tied={self.config.tie_embeddings}, "
            f"parameters={self.parameter_count():,}"
        )
