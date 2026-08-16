"""Dense document packing and causal next-token window construction.

This module sits immediately after :mod:`llm.data.document` in the base-LM
pretraining pipeline.

The document layer produces independent sequences of the form::

    <|bos|> document content <|eos|>

This module validates those sequences, concatenates them without adding or
removing tokens, and divides the resulting stream into causal-LM windows.

For a context length ``S`` each full training window contains ``S + 1`` token
ids::

    window = [t0, t1, ..., tS]
    input  = [t0, t1, ..., t(S-1)]
    target = [t1, t2, ..., tS]

Adjacent windows advance by ``S`` tokens rather than ``S + 1``.  Therefore the
last token of one window is the first token of the next window.  This one-token
overlap preserves next-token alignment at window boundaries without predicting
any token twice.

No padding, tensors, binary shards, train/validation splitting, or randomization
happens here.  Those belong to later stages.
"""

from __future__ import annotations

from collections.abc import Iterable

from llm.data.document import validate_pretraining_document_tokens
from llm.tokenizer.tokenizer import Tokenizer


DEFAULT_CONTEXT_LENGTH = 2048


def _require_tokenizer(tokenizer: Tokenizer) -> None:
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must be a Tokenizer, got {type(tokenizer).__name__}"
        )


def _validate_context_length(context_length: int) -> None:
    if not isinstance(context_length, int) or isinstance(context_length, bool):
        raise TypeError("context_length must be an integer")
    if context_length <= 0:
        raise ValueError("context_length must be > 0")


def _materialize_token_stream(
    token_ids: Iterable[int],
    tokenizer: Tokenizer,
) -> list[int]:
    """Materialize a token stream and validate integer/range invariants."""

    _require_tokenizer(tokenizer)

    ids = list(token_ids)
    for token_id in ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError(
                f"token ids must be integers, got {type(token_id).__name__}"
            )
        if token_id < 0 or token_id >= tokenizer.vocab_size:
            raise ValueError(
                f"token id {token_id} is outside tokenizer vocabulary "
                f"[0, {tokenizer.vocab_size})"
            )

    return ids


def concatenate_pretraining_documents(
    documents: Iterable[Iterable[int]],
    tokenizer: Tokenizer,
) -> list[int]:
    """Validate and densely concatenate independent pretraining documents.

    Each input document must already satisfy the document-stage contract:

    ``<|bos|> ordinary-content-ids <|eos|>``

    Documents are concatenated exactly as provided.  In particular, this
    function does **not** insert padding, separators, extra EOS tokens, or extra
    BOS tokens.  A boundary therefore naturally appears as::

        ... <|eos|> <|bos|> ...

    Parameters
    ----------
    documents:
        Iterable of independently encoded pretraining-document token sequences.
    tokenizer:
        Project tokenizer used to validate boundaries and token-id ranges.

    Returns
    -------
    list[int]
        One deterministic dense token stream.  An empty document iterable
        produces an empty stream.
    """

    _require_tokenizer(tokenizer)

    stream: list[int] = []
    for token_ids in documents:
        validated = validate_pretraining_document_tokens(token_ids, tokenizer)
        stream.extend(validated)

    return stream


def build_causal_windows(
    token_stream: Iterable[int],
    tokenizer: Tokenizer,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> tuple[list[list[int]], list[int]]:
    """Split a token stream into full ``context_length + 1`` causal windows.

    Window starts advance by exactly ``context_length`` tokens.  For example,
    with ``context_length=4``::

        window 0 = stream[0:5]
        window 1 = stream[4:9]
        window 2 = stream[8:13]

    The returned ``tail`` begins at the *next* window start.  Consequently it
    intentionally includes the final token of the last full window as a carry
    token.  That carry token is needed if more tokens are appended later.

    Examples with ``context_length=4``:

    * 10 stream tokens -> two full windows and tail ``stream[8:]`` (2 tokens).
    * 9 stream tokens  -> two full windows and tail ``stream[8:]`` (1 carry).
    * 4 stream tokens  -> no full window and the entire stream is the tail.

    Returning the tail explicitly avoids silently discarding data.  A later
    final-dataset builder may choose to drop the final incomplete tail, while a
    shard writer may carry it into the next shard.
    """

    _validate_context_length(context_length)
    ids = _materialize_token_stream(token_stream, tokenizer)

    if len(ids) <= context_length:
        return [], ids

    # A full causal example needs S input positions plus one final target token.
    full_window_count = (len(ids) - 1) // context_length

    windows: list[list[int]] = []
    for window_index in range(full_window_count):
        start = window_index * context_length
        end = start + context_length + 1
        window = ids[start:end]

        # Defensive invariant: the arithmetic above must only emit full windows.
        if len(window) != context_length + 1:
            raise RuntimeError("internal packing error produced an incomplete window")

        windows.append(window)

    # The next possible window would start here.  Keep that token as carryover
    # even when it was the final target of the preceding full window.
    tail_start = full_window_count * context_length
    tail = ids[tail_start:]

    return windows, tail


def split_causal_window(
    window: Iterable[int],
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> tuple[list[int], list[int]]:
    """Convert one ``S + 1`` token window into ``S`` inputs and targets.

    This function is intentionally independent of PyTorch.  Tensor conversion
    belongs in the later dataset/DataLoader layer.
    """

    _validate_context_length(context_length)

    ids = list(window)
    expected_length = context_length + 1
    if len(ids) != expected_length:
        raise ValueError(
            f"causal window must contain exactly {expected_length} tokens for "
            f"context_length={context_length}; got {len(ids)}"
        )

    for token_id in ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError(
                f"token ids must be integers, got {type(token_id).__name__}"
            )

    input_ids = ids[:-1]
    target_ids = ids[1:]

    # This is the defining next-token alignment invariant.
    if input_ids[1:] != target_ids[:-1]:
        raise RuntimeError("internal causal input/target alignment invariant failed")

    return input_ids, target_ids


def pack_pretraining_documents(
    documents: Iterable[Iterable[int]],
    tokenizer: Tokenizer,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> tuple[list[list[int]], list[int]]:
    """Validate documents, densely concatenate them, and build causal windows.

    This convenience function is exactly equivalent to calling
    :func:`concatenate_pretraining_documents` followed by
    :func:`build_causal_windows`.
    """

    _validate_context_length(context_length)
    stream = concatenate_pretraining_documents(documents, tokenizer)
    return build_causal_windows(
        stream,
        tokenizer,
        context_length=context_length,
    )