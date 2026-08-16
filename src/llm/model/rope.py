"""Rotary position embedding (RoPE).

Position enters this model in exactly one place: a rotation applied to the
query and key vectors before attention scores are computed.  There is no
learned position embedding and no additive position signal anywhere else.

The mechanism
-------------
Each head vector is read as ``head_dim / 2`` adjacent coordinate pairs, and
pair ``i`` at position ``m`` is rotated by ``m * theta_i``::

    [x'_2i  ]   [cos(m*theta_i)  -sin(m*theta_i)] [x_2i  ]
    [x'_2i+1] = [sin(m*theta_i)   cos(m*theta_i)] [x_2i+1]

    theta_i = base ** (-2i / head_dim)

The reason this works is that a rotation by ``m`` applied to the query and a
rotation by ``n`` applied to the key leave the dot product depending only on
``n - m``.  Absolute position is injected; relative position is what attention
actually sees.  ``test_dot_product_depends_only_on_relative_position`` asserts
exactly that.

Frozen v0.1 decisions
---------------------
============================  ==========================================
head_dim / rotary_dim         64 / 64 -- the entire head is rotated
context_length                2048
theta                         ``ModelConfig.rope_theta`` (10,000)
pair convention               adjacent/interleaved (0,1), (2,3), ...
tables                        precomputed, non-persistent buffers
table precision               float32, regardless of activation dtype
applied to                    Q and K.  **Never V.**
trainable parameters          0
positions beyond 2047         rejected, never extrapolated
============================  ==========================================

Why interleaved and not split-halves
------------------------------------
Both conventions describe the same family of rotations up to a permutation of
coordinates, and either would train correctly.  They are *not* interchangeable
in a checkpoint: weights trained under one convention produce nonsense under
the other, silently, because nothing about the tensor shapes changes.

Adjacent pairing is chosen because it is the literal reading of the original
2-D rotation formulation, which makes the implementation directly checkable
against the definition.  ``test_convention_is_interleaved_not_split_halves``
pins it so an "equivalent" refactor cannot quietly invalidate every checkpoint
ever trained by this project.

Why the tables stay float32
---------------------------
The tables are deterministic constants; computing transcendental functions in
reduced precision buys nothing and costs accuracy that then multiplies into
every query and key. They are built in float32 and cast to the activation
dtype only at application time, so the output dtype always matches the input.

Calling ``.to(torch.bfloat16)`` on this module -- which is what BF16 training
does -- would otherwise silently downcast the tables.  :meth:`_apply`
regenerates them instead, because casting back up cannot recover the lost
bits.  This differs from :mod:`llm.model.rmsnorm`, which deliberately computes
in the input dtype: there the reduction is over live activations, here the
values are constants known exactly in advance.
"""

from __future__ import annotations

import torch
from torch import nn

from llm.model.config import ModelConfig


#: Dtypes that must never be used to *store* the trigonometric tables.
_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)

#: Tables are always built and held at this precision.
_TABLE_DTYPE = torch.float32


def _require_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def build_rope_tables(
    head_dim: int,
    context_length: int,
    theta: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` of shape ``(context_length, head_dim // 2)``.

    Free function rather than a method so tests can build the tables
    independently of any module instance.
    """

    pairs = head_dim // 2
    exponents = torch.arange(0, pairs, dtype=_TABLE_DTYPE, device=device)
    inverse_frequencies = theta ** (-2.0 * exponents / head_dim)

    positions = torch.arange(context_length, dtype=_TABLE_DTYPE, device=device)
    angles = torch.outer(positions, inverse_frequencies)

    return torch.cos(angles), torch.sin(angles)


class RotaryEmbedding(nn.Module):
    """Apply RoPE to a ``(batch, sequence, heads, head_dim)`` tensor.

    Parameters
    ----------
    head_dim:
        Size of one attention head.  Must be even: every rotated coordinate
        needs a partner.
    context_length:
        Number of positions to precompute.  Positions at or beyond this are
        rejected rather than extrapolated.
    theta:
        Angular-frequency base.

    Notes
    -----
    ``start_pos`` exists for autoregressive generation with a KV cache, where
    a single new token sits at absolute position ``prompt_length + step``.
    Training always passes ``start_pos=0``, but designing it in now avoids
    reworking this module when caching arrives.

    This module has no parameters and rotates nothing in place.
    """

    def __init__(
        self,
        head_dim: int,
        context_length: int,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()

        self.head_dim = _require_positive_int("head_dim", head_dim)
        self.context_length = _require_positive_int("context_length", context_length)

        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even: RoPE rotates adjacent "
                "coordinate pairs, so an odd dimension would leave one channel "
                "unrotated."
            )
        if isinstance(theta, bool) or not isinstance(theta, (int, float)):
            raise TypeError(f"theta must be a number, got {type(theta).__name__}")
        if not theta > 0.0:
            raise ValueError(f"theta must be > 0, got {theta}")
        self.theta = float(theta)

        cos, sin = build_rope_tables(self.head_dim, self.context_length, self.theta)
        # Non-persistent: these are reproducible from head_dim, context_length
        # and theta, so a checkpoint should carry the config, not 128K floats.
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "RotaryEmbedding":
        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"config must be a ModelConfig, got {type(config).__name__}"
            )
        return cls(
            head_dim=config.head_dim,
            context_length=config.context_length,
            theta=config.rope_theta,
        )

    @property
    def pairs(self) -> int:
        """Number of rotated 2-D coordinate pairs per head."""

        return self.head_dim // 2

    def _apply(self, fn, recurse: bool = True):
        """Keep the tables at float32 through ``.to()`` / ``.half()`` calls.

        ``super()._apply`` casts buffers along with parameters, so a BF16
        training setup would silently reduce table precision.  Casting back is
        not enough -- the bits are gone -- so the tables are rebuilt from the
        exact same deterministic definition on whatever device they landed on.
        """

        module = super()._apply(fn, recurse=recurse)

        if module.cos_table.dtype in _LOW_PRECISION_DTYPES:
            cos, sin = build_rope_tables(
                module.head_dim,
                module.context_length,
                module.theta,
                device=module.cos_table.device,
            )
            module.cos_table = cos
            module.sin_table = sin

        return module

    def _validate_input(self, x: torch.Tensor, start_pos: int) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a Tensor, got {type(x).__name__}")
        if x.ndim != 4:
            raise ValueError(
                f"x must be (batch, sequence, heads, head_dim), got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"x last dimension is {x.shape[-1]}, expected head_dim {self.head_dim}"
            )
        if not isinstance(start_pos, int) or isinstance(start_pos, bool):
            raise TypeError(
                f"start_pos must be an integer, got {type(start_pos).__name__}"
            )
        if start_pos < 0:
            raise ValueError(f"start_pos must be >= 0, got {start_pos}")

        end = start_pos + x.shape[1]
        if end > self.context_length:
            raise ValueError(
                f"positions [{start_pos}, {end}) exceed context_length "
                f"{self.context_length}. RoPE is not extrapolated beyond the "
                "positions the model was trained on."
            )

    def forward(self, x: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        """Rotate ``x`` by its absolute position.

        ``x`` is ``(batch, sequence, heads, head_dim)``; the output has the
        same shape, dtype and device.  Apply to queries and keys only.
        """

        self._validate_input(x, start_pos)

        sequence_length = x.shape[1]
        cos = self.cos_table[start_pos : start_pos + sequence_length]
        sin = self.sin_table[start_pos : start_pos + sequence_length]

        # (sequence, pairs) -> (1, sequence, 1, pairs) so it broadcasts across
        # batch and heads, both of which are position-independent.
        cos = cos.to(dtype=x.dtype).unsqueeze(0).unsqueeze(2)
        sin = sin.to(dtype=x.dtype).unsqueeze(0).unsqueeze(2)

        even = x[..., 0::2]
        odd = x[..., 1::2]

        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        # Re-interleave: stacking on a new trailing axis and flattening yields
        # [e0, o0, e1, o1, ...], restoring the original coordinate order.
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, context_length={self.context_length}, "
            f"theta={self.theta}, pairs={self.pairs}"
        )
