"""Document-level tokenization for base-language-model pretraining.

This module is the first boundary between cleaned text documents and the token
stream consumed by the pretraining data pipeline.

Policy
------
* Each independent pretraining document is wrapped as::

      <|bos|> content tokens <|eos|>

* Source text is always encoded with ``parse_special_tokens=False``.  Therefore
  literal strings such as ``<|assistant|>`` inside ordinary pretraining text are
  represented as normal UTF-8/BPE content, never as control-token ids.
* Normalization is delegated to :class:`llm.tokenizer.tokenizer.Tokenizer`, so
  this layer cannot silently diverge from the tokenizer's UTF-8/NFC/LF policy.
* No padding, truncation, packing, or train/validation splitting happens here.
  Those belong to later data-pipeline stages.
"""

from __future__ import annotations

from collections.abc import Iterable

from llm.tokenizer.tokenizer import Tokenizer


BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"


def _require_boundary_token_id(tokenizer: Tokenizer, token: str) -> int:
    """Return a required boundary-token id using the public tokenizer API."""

    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(
            f"tokenizer does not reserve required pretraining boundary token {token!r}"
        )
    return token_id


def encode_pretraining_document(text: str, tokenizer: Tokenizer) -> list[int]:
    """Encode one independent text document for causal-LM pretraining.

    The returned sequence is always::

        [BOS, *ordinary_content_ids, EOS]

    ``Tokenizer.encode`` performs the project's canonical text normalization.
    Special-token parsing is explicitly disabled for document content, while
    BOS/EOS are added by the tokenizer itself.

    Empty text is valid at this layer and produces ``[BOS, EOS]``.  Upstream
    corpus-quality filters may choose to remove empty documents before calling
    this function.
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must be a Tokenizer, got {type(tokenizer).__name__}"
        )

    # Fail early with a clear data-pipeline error instead of relying only on the
    # Tokenizer.encode error path.
    bos_id = _require_boundary_token_id(tokenizer, BOS_TOKEN)
    eos_id = _require_boundary_token_id(tokenizer, EOS_TOKEN)

    token_ids = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
        parse_special_tokens=False,
    )

    # Defensive invariants.  These should always hold if Tokenizer.encode keeps
    # its public contract, and make a future tokenizer regression fail close to
    # the document boundary instead of later during packing/training.
    if len(token_ids) < 2:
        raise RuntimeError("encoded pretraining document is missing BOS/EOS boundaries")
    if token_ids[0] != bos_id:
        raise RuntimeError("encoded pretraining document does not start with BOS")
    if token_ids[-1] != eos_id:
        raise RuntimeError("encoded pretraining document does not end with EOS")

    special_ids = set(tokenizer.id_to_special_token)
    content_special_ids = [
        token_id
        for token_id in token_ids[1:-1]
        if token_id in special_ids
    ]
    if content_special_ids:
        raise RuntimeError(
            "ordinary pretraining content unexpectedly contains reserved special-token ids"
        )

    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise RuntimeError("tokenizer returned a non-integer token id")
        if token_id < 0 or token_id >= tokenizer.vocab_size:
            raise RuntimeError(
                f"tokenizer returned out-of-range token id {token_id}; "
                f"expected [0, {tokenizer.vocab_size})"
            )

    return token_ids


def validate_pretraining_document_tokens(
    token_ids: Iterable[int],
    tokenizer: Tokenizer,
) -> list[int]:
    """Validate and materialize one BOS/content/EOS token sequence.

    This helper is useful at serialization/shard boundaries where token ids may
    have been read from an iterator or binary buffer.  It rejects malformed
    boundaries, out-of-range ids, and reserved special-token ids inside base
    pretraining content.
    """

    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must be a Tokenizer, got {type(tokenizer).__name__}"
        )

    ids = list(token_ids)
    bos_id = _require_boundary_token_id(tokenizer, BOS_TOKEN)
    eos_id = _require_boundary_token_id(tokenizer, EOS_TOKEN)

    if len(ids) < 2:
        raise ValueError("pretraining document must contain at least BOS and EOS")

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

    if ids[0] != bos_id:
        raise ValueError("pretraining document must start with <|bos|>")
    if ids[-1] != eos_id:
        raise ValueError("pretraining document must end with <|eos|>")

    special_ids = set(tokenizer.id_to_special_token)
    for token_id in ids[1:-1]:
        if token_id in special_ids:
            token = tokenizer.id_to_special_token[token_id]
            raise ValueError(
                "base pretraining content must not contain reserved special-token "
                f"ids; found {token!r}"
            )

    return ids


def decode_pretraining_document(
    token_ids: Iterable[int],
    tokenizer: Tokenizer,
) -> str:
    """Decode one validated pretraining document back to normalized content.

    BOS and EOS are structural boundaries and are removed before decoding, so
    the returned value is exactly the tokenizer-normalized document text.
    """

    ids = validate_pretraining_document_tokens(token_ids, tokenizer)
    return tokenizer.decode(ids[1:-1], skip_special_tokens=False)
