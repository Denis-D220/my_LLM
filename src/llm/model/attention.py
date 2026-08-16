"""Causal multi-head self-attention.

The first component with learned weights, and the one place in the model where
information moves between positions.  Everything else -- norms, the
feed-forward network -- acts on each position independently.

Shape flow for the frozen v0.1 architecture::

    x                       (batch, sequence, 512)
      -> Wq, Wk, Wv         three (batch, sequence, 512)
      -> split heads        (batch, sequence, 8, 64)
      -> RoPE on Q and K    (batch, sequence, 8, 64)      V is NOT rotated
      -> transpose          (batch, 8, sequence, 64)
      -> softmax(QK^T / 8 + causal mask) V
      -> merge heads        (batch, sequence, 512)
      -> Wo                 (batch, sequence, 512)

Why V is not rotated
--------------------
RoPE encodes position by making the *score* between two tokens depend on their
separation.  That score comes from the Q-K inner product, so rotating Q and K
is what injects position.  V carries the content that gets mixed once the
weights are decided; rotating it would corrupt that content with a
position-dependent spin for no benefit.  The reference implementation in the
tests rotates only Q and K, so rotating V here would fail immediately.

Causality
---------
Position ``t`` may attend to positions ``0..t`` and never beyond.  This is the
property the entire pretraining objective rests on: the target at position
``t`` is the input at ``t+1``, so a single leak from the future turns
next-token prediction into copying, and the loss curve looks *better* while the
model learns nothing.  ``test_changing_a_later_token_cannot_change_an_earlier_output``
asserts it directly.

Masking strategy
----------------
``forward`` uses :func:`torch.nn.functional.scaled_dot_product_attention` with
``is_causal=True``, which selects fused kernels (Flash attention on CUDA) and
never materializes the ``sequence x sequence`` score matrix.

The tests carry an explicit implementation -- build scores, fill the upper
triangle with ``-inf``, softmax, multiply by V -- and assert the two agree.
That keeps a transparent definition to check against while the fast path is
what actually runs.

Statelessness
-------------
v0.1 has no KV cache.  ``start_pos`` shifts the RoPE positions only, which is
what generation will need; the attention itself always sees a complete window,
so ``is_causal=True`` remains correct for every value of ``start_pos``.  The
cache can be added later without changing the rotation or this signature.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from llm.model.config import ModelConfig
from llm.model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with rotary positions and a causal mask.

    Parameters come from :class:`~llm.model.config.ModelConfig` rather than
    loose arguments, so the head geometry cannot disagree with the rest of the
    model.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"config must be a ModelConfig, got {type(config).__name__}"
            )

        self.config = config
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        projection_size = config.attention_output_size
        self.q_proj = nn.Linear(
            self.hidden_size, projection_size, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, projection_size, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, projection_size, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            projection_size, self.hidden_size, bias=config.attention_bias
        )

        self.rope = RotaryEmbedding.from_config(config)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "CausalSelfAttention":
        return cls(config)

    def _validate_input(self, x: torch.Tensor, start_pos: int) -> None:
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
        if not isinstance(start_pos, int) or isinstance(start_pos, bool):
            raise TypeError(
                f"start_pos must be an integer, got {type(start_pos).__name__}"
            )
        if start_pos < 0:
            raise ValueError(f"start_pos must be >= 0, got {start_pos}")

    def forward(self, x: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        """Attend over ``x`` causally.

        ``x`` is ``(batch, sequence, hidden_size)``; the output has the same
        shape, dtype and device.
        """

        self._validate_input(x, start_pos)
        batch, sequence, _ = x.shape

        # (batch, sequence, heads, head_dim) is RoPE's expected layout.
        queries = self.q_proj(x).view(batch, sequence, self.n_heads, self.head_dim)
        keys = self.k_proj(x).view(batch, sequence, self.n_heads, self.head_dim)
        values = self.v_proj(x).view(batch, sequence, self.n_heads, self.head_dim)

        # Position enters here and nowhere else.  V is deliberately untouched.
        queries = self.rope(queries, start_pos=start_pos)
        keys = self.rope(keys, start_pos=start_pos)

        # SDPA wants (batch, heads, sequence, head_dim).
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.scale,
        )

        merged = attended.transpose(1, 2).reshape(batch, sequence, self.hidden_size)
        return self.o_proj(merged)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, bias={self.config.attention_bias}"
        )
