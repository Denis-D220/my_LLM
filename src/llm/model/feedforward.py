"""SwiGLU position-wise feed-forward network.

The second half of a Transformer block.  Where attention mixes information
*across* positions, this transforms each position on its own -- and that
division of labour is worth stating as a property rather than a comment:
``test_positions_do_not_interact`` asserts that changing one position's input
cannot alter another position's output.  When the block is assembled, attention
must be the only sublayer capable of moving information along the sequence.

Definition
----------
::

    g = W_gate x
    u = W_up   x
    h = SiLU(g) * u          elementwise
    y = W_down h

with ``SiLU(z) = z * sigmoid(z)``.

Two projections go up, one comes down.  ``g`` passes through the nonlinearity
and acts as a multiplicative gate on the linear path ``u``; the gate is what
distinguishes SwiGLU from a plain ``W_down(SiLU(W_up x))`` MLP, and it is why
the parameter budget is three matrices rather than two.

Frozen v0.1 shape
-----------------
============  =========================================
input          512
hidden        1536
output         512
activation    SiLU
bias          none
dropout       none
parameters    3 x 512 x 1536 = 2,359,296
============  =========================================

Rank independence
-----------------
Unlike :class:`~llm.model.attention.CausalSelfAttention`, which needs
``(batch, sequence, hidden)`` because it reasons about the sequence axis, this
module only ever touches the last dimension.  It accepts any rank of at least
one and returns the same shape, so it can be tested on a bare ``(512,)`` vector
without inventing batch and sequence axes that mean nothing to it.

Initialization
--------------
The projections use PyTorch's default ``nn.Linear`` initialization on purpose.
Transformer-specific initialization -- residual scaling by depth, tied
embedding variance, and so on -- is a *model-level* training policy, and having
each component invent its own would make the eventual policy impossible to
audit. That decision belongs with the full model, before the tiny-overfit run.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from llm.model.config import ModelConfig


class SwiGLU(nn.Module):
    """Gated feed-forward network with a SiLU-activated gate.

    Built only from :class:`~llm.model.config.ModelConfig`, so its widths
    cannot drift from the rest of the architecture.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"config must be a ModelConfig, got {type(config).__name__}"
            )

        self.config = config
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.ffn_hidden_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.ffn_hidden_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.ffn_hidden_size, self.hidden_size, bias=config.mlp_bias
        )

    @classmethod
    def from_config(cls, config: ModelConfig) -> "SwiGLU":
        return cls(config)

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a Tensor, got {type(x).__name__}")
        if x.ndim == 0:
            raise ValueError("x must have at least one dimension")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"x last dimension is {x.shape[-1]}, expected hidden_size "
                f"{self.hidden_size}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward transform to the last dimension."""

        self._validate_input(x)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"ffn_hidden_size={self.ffn_hidden_size}, "
            f"bias={self.config.mlp_bias}"
        )
