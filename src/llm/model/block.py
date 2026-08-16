"""One pre-norm decoder block.

Composition only: this module owns no parameters of its own and adds no
operation beyond two normalizations, two sublayers, and two additions::

    x = x + attention(attention_norm(x))
    x = x + feedforward(ffn_norm(x))

Frozen v0.1 budget
------------------
============================  =========
attention                     1,048,576
SwiGLU                        2,359,296
attention RMSNorm                   512
feed-forward RMSNorm                512
----------------------------  ---------
TransformerBlock              3,408,896
============================  =========

Pre-norm, not post-norm
-----------------------
The normalization sits *inside* each branch, so the residual stream from input
to output is a clean sum of identity plus sublayer contributions, never passed
through a normalizer.  A gradient can therefore reach layer 0 without being
rescaled six times on the way, which is what makes a deep stack trainable
without a warmup-dependent schedule.  Post-norm would place a normalizer on
the residual path itself and is a different architecture, not a rearrangement.

Two norms, never one
--------------------
``attention_norm`` and ``ffn_norm`` are separate modules with separate learned
scales.  Sharing one instance between the branches would halve their
parameters and silently couple two unrelated normalizations;
``test_the_two_norms_are_distinct_modules`` rules that out.

What the tests pin
------------------
The residual wiring is asserted directly rather than inferred: zeroing
``attention.o_proj`` and ``feedforward.down_proj`` drives both branch outputs
to exactly zero, at which point the block must be the *identity*.  If either
residual add were missing, that test returns zeros instead of ``x``.

Causality is re-asserted here rather than assumed from
:mod:`llm.model.attention`.  A residual stream is an easy place to leak the
future -- one misplaced shift or a sublayer applied to the wrong tensor -- and
a leak that survives composition is invisible in the component tests.
"""

from __future__ import annotations

import torch
from torch import nn

from llm.model.attention import CausalSelfAttention
from llm.model.config import ModelConfig
from llm.model.feedforward import SwiGLU
from llm.model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm attention + feed-forward with residual connections."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"config must be a ModelConfig, got {type(config).__name__}"
            )

        self.config = config
        self.hidden_size = config.hidden_size

        self.attention_norm = RMSNorm.from_config(config)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm.from_config(config)
        self.feedforward = SwiGLU(config)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "TransformerBlock":
        return cls(config)

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a Tensor, got {type(x).__name__}")
        if x.ndim != 3:
            raise ValueError(
                f"x must be (batch, sequence, hidden_size), got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"x last dimension is {x.shape[-1]}, expected hidden_size "
                f"{self.hidden_size}"
            )

    def forward(self, x: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        """Transform ``(batch, sequence, hidden_size)`` into the same shape."""

        self._validate_input(x)

        x = x + self.attention(self.attention_norm(x), start_pos=start_pos)
        x = x + self.feedforward(self.ffn_norm(x))
        return x

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}"
