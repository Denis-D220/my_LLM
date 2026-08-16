"""Root-mean-square layer normalization.

The first tensor-carrying component of the model.  It is deliberately the
smallest one: it has a closed-form definition that can be checked against a
literal transcription of the formula, so it is where the habit of proving a
layer rather than eyeballing it gets established.

Definition
----------
For the last dimension of ``x`` with ``d`` channels::

    RMSNorm(x) = x / sqrt(mean(x**2) + eps) * weight

There is **no mean subtraction and no bias**.  That is the whole difference
from LayerNorm, and it is not cosmetic:

* LayerNorm re-centres, so its output always has zero mean per row.  RMSNorm
  only rescales, so a row's direction -- including any offset from zero -- is
  preserved and only its magnitude is normalized.
* Dropping the mean removes one reduction and one subtraction per row, which
  is why decoder-only models generally use it.

A test in ``tests/model/test_rmsnorm.py`` asserts the output of an offset
vector keeps a non-zero mean, specifically so that a future edit cannot
silently turn this into LayerNorm.

Numerical policy for v0.1
-------------------------
The reduction is computed in the **input dtype**, and the output dtype always
matches the input.  No internal upcast to FP32 happens here.

That is a deliberate choice, not an oversight.  Upcasting the sum of squares is
the right thing to do under BF16 training -- BF16 has ~8 bits of mantissa, and
a 512-term sum of squares loses precision fast -- but adding that behaviour now
would mean shipping a numerical policy with no test that exercises it and no
training run that motivates it.  When BF16 training arrives, the upcast gets
added together with tests that measure the difference it makes.

Shape contract
--------------
``x.shape[-1]`` must equal ``dim``.  This is checked rather than left to
broadcasting: a tensor whose last dimension is 1 would otherwise broadcast
silently against the weight vector and expand to ``dim`` channels, producing a
plausible-looking tensor of the wrong width.
"""

from __future__ import annotations

import torch
from torch import nn

from llm.model.config import ModelConfig


def _require_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _require_positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not value > 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return float(value)


class RMSNorm(nn.Module):
    """Scale a tensor's last dimension by its root mean square.

    Parameters
    ----------
    dim:
        Size of the normalized (last) dimension.  One learned scale per
        channel, initialized to one, so the layer starts as a pure normalizer.
    eps:
        Added inside the square root to keep an all-zero row finite.

    The only parameter is ``weight`` of shape ``(dim,)``.  There is no bias.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = _require_positive_int("dim", dim)
        self.eps = _require_positive_float("eps", eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    @classmethod
    def from_config(cls, config: ModelConfig) -> "RMSNorm":
        """Build the norm the frozen architecture calls for."""

        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"config must be a ModelConfig, got {type(config).__name__}"
            )
        return cls(config.hidden_size, eps=config.rms_norm_eps)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a Tensor, got {type(x).__name__}")
        if x.ndim == 0:
            raise ValueError("x must have at least one dimension")
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"x last dimension is {x.shape[-1]}, expected {self.dim}. "
                "Broadcasting would silently reshape the input, so it is refused."
            )

        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean_square + self.eps)
        return normalized * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
