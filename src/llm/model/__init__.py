# src/llm/model/__init__.py
#
# Importing this package imports torch, because every component below the
# config carries tensors. ``llm.model.config`` stays torch-free and can be
# imported directly when only the architecture description is needed.

from .config import FROZEN_CONTEXT_LENGTH, FROZEN_VOCAB_SIZE, ModelConfig
from .attention import CausalSelfAttention
from .block import TransformerBlock
from .feedforward import SwiGLU
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding, build_rope_tables
from .transformer import Transformer, causal_lm_loss

__all__ = [
    "ModelConfig",
    "Transformer",
    "causal_lm_loss",
    "RMSNorm",
    "RotaryEmbedding",
    "CausalSelfAttention",
    "SwiGLU",
    "TransformerBlock",
    "build_rope_tables",
    "FROZEN_VOCAB_SIZE",
    "FROZEN_CONTEXT_LENGTH",
]
