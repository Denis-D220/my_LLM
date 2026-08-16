"""PyTorch dataset over deterministic pretraining token shards.

This module is the read-side bridge between the binary shard format in
:mod:`llm.data.shards` and the decoder-only Transformer training loop.

The shard files store one flat token stream.  They do *not* duplicate the
one-token overlap needed by causal language-model examples.  This dataset
constructs examples dynamically from global token offsets::

    example 0 -> stream[0 : context_length + 1]
    example 1 -> stream[context_length : 2 * context_length + 1]
    ...

For each window, the returned tensors are::

    input_ids  = window[:-1]
    target_ids = window[1:]

so every item contains exactly ``context_length`` next-token predictions.
A window is allowed to cross binary shard boundaries; storage boundaries are
invisible to the model.

Correctness-first policy
------------------------
* Manifests are structurally validated by :func:`load_shard_manifest`.
* File byte counts are always checked at dataset construction.
* SHA-256 verification is enabled by default and can be disabled explicitly
  for faster repeated startup once a dataset has already been validated.
* Token ids are checked against the manifest vocabulary when read.
* Incomplete final tails are never exposed as padded examples.
* Negative indexes are rejected instead of silently wrapping.
"""

from __future__ import annotations

from array import array
from bisect import bisect_right
import hashlib
from pathlib import Path
import sys

import torch
from torch.utils.data import Dataset

from llm.data.packing import DEFAULT_CONTEXT_LENGTH
from llm.data.shards import (
    BYTE_ORDER,
    TOKEN_ITEM_BYTES,
    ShardInfo,
    ShardManifest,
    load_shard_manifest,
)


def _validate_context_length(context_length: int) -> None:
    if not isinstance(context_length, int) or isinstance(context_length, bool):
        raise TypeError("context_length must be an integer")
    if context_length <= 0:
        raise ValueError("context_length must be > 0")


def _validate_expected_vocab_size(expected_vocab_size: int | None) -> None:
    if expected_vocab_size is None:
        return
    if not isinstance(expected_vocab_size, int) or isinstance(expected_vocab_size, bool):
        raise TypeError("expected_vocab_size must be an integer or None")
    if expected_vocab_size <= 0:
        raise ValueError("expected_vocab_size must be > 0")


def _decode_uint16(data: bytes) -> list[int]:
    if len(data) % TOKEN_ITEM_BYTES != 0:
        raise ValueError("uint16 token data must have an even byte length")

    values = array("H")
    values.frombytes(data)
    if values.itemsize != TOKEN_ITEM_BYTES:
        raise RuntimeError("platform unsigned-short size is not 2 bytes")
    if sys.byteorder != BYTE_ORDER:
        values.byteswap()
    return list(values)


class PretrainingDataset(Dataset):
    """Random-access causal-LM examples backed by uint16 token shards.

    Parameters
    ----------
    manifest_path:
        Path to one split's ``manifest.json`` written by
        :func:`llm.data.shards.write_token_shards`.
    context_length:
        Number of model input positions per example.  The default is 2048.
    expected_vocab_size:
        Optional model/tokenizer vocabulary size.  Supplying it makes dataset
        construction fail immediately if the shard manifest was produced with
        a different vocabulary.
    verify_checksums:
        When true, verify every shard SHA-256 and the complete stream SHA-256
        during construction.  File sizes are checked regardless of this flag.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        expected_vocab_size: int | None = None,
        verify_checksums: bool = True,
    ) -> None:
        _validate_context_length(context_length)
        _validate_expected_vocab_size(expected_vocab_size)
        if not isinstance(verify_checksums, bool):
            raise TypeError("verify_checksums must be a bool")

        self.manifest_path = Path(manifest_path)
        self.manifest: ShardManifest = load_shard_manifest(self.manifest_path)
        self.context_length = context_length
        self.verify_checksums = verify_checksums

        if (
            expected_vocab_size is not None
            and expected_vocab_size != self.manifest.tokenizer_vocab_size
        ):
            raise ValueError(
                "expected vocabulary size does not match shard manifest: "
                f"{expected_vocab_size} != {self.manifest.tokenizer_vocab_size}"
            )

        self._shard_ends = tuple(shard.token_end for shard in self.manifest.shards)
        self._validate_storage(verify_checksums=verify_checksums)

        if self.manifest.total_tokens <= 1:
            self._num_examples = 0
        else:
            self._num_examples = (
                self.manifest.total_tokens - 1
            ) // self.context_length

    @property
    def split(self) -> str:
        return self.manifest.split

    @property
    def vocab_size(self) -> int:
        return self.manifest.tokenizer_vocab_size

    @property
    def total_tokens(self) -> int:
        return self.manifest.total_tokens

    @property
    def prediction_token_count(self) -> int:
        """Number of next-token targets exposed by complete examples."""

        return self._num_examples * self.context_length

    @property
    def tail_token_count(self) -> int:
        """Tokens retained after the next complete-window start.

        For a non-empty stream this includes the one carry token that would be
        the first input token of a future example if more data were appended.
        """

        if self.total_tokens == 0:
            return 0
        return self.total_tokens - self.prediction_token_count

    def _validate_storage(self, *, verify_checksums: bool) -> None:
        stream_digest = hashlib.sha256()

        for shard in self.manifest.shards:
            path = self.manifest_path.parent / shard.filename
            if not path.is_file():
                raise FileNotFoundError(f"missing token shard: {path}")

            actual_size = path.stat().st_size
            if actual_size != shard.byte_count:
                raise ValueError(
                    f"shard byte count mismatch for {shard.filename}: "
                    f"expected {shard.byte_count}, got {actual_size}"
                )

            if verify_checksums:
                shard_digest = hashlib.sha256()
                with path.open("rb") as handle:
                    while True:
                        block = handle.read(1024 * 1024)
                        if not block:
                            break
                        shard_digest.update(block)
                        stream_digest.update(block)

                actual_sha256 = shard_digest.hexdigest()
                if actual_sha256 != shard.sha256:
                    raise ValueError(
                        f"shard checksum mismatch for {shard.filename}: "
                        f"expected {shard.sha256}, got {actual_sha256}"
                    )

        if verify_checksums:
            actual_stream_sha256 = stream_digest.hexdigest()
            if actual_stream_sha256 != self.manifest.stream_sha256:
                raise ValueError(
                    "token stream checksum mismatch: "
                    f"expected {self.manifest.stream_sha256}, "
                    f"got {actual_stream_sha256}"
                )

    def __len__(self) -> int:
        return self._num_examples

    def _validate_index(self, index: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            raise IndexError("negative dataset indexes are not supported")
        if index >= self._num_examples:
            raise IndexError(
                f"dataset index {index} is outside [0, {self._num_examples})"
            )

    def _find_shard_index(self, global_token_index: int) -> int:
        shard_index = bisect_right(self._shard_ends, global_token_index)
        if shard_index >= len(self.manifest.shards):
            raise IndexError(
                f"global token index {global_token_index} is outside token stream"
            )
        return shard_index

    def read_token_range(self, start: int, count: int) -> list[int]:
        """Read an exact global token range, transparently crossing shards."""

        if not isinstance(start, int) or isinstance(start, bool):
            raise TypeError("start must be an integer")
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError("count must be an integer")
        if start < 0:
            raise ValueError("start must be >= 0")
        if count < 0:
            raise ValueError("count must be >= 0")
        if start + count > self.total_tokens:
            raise ValueError(
                f"requested token range [{start}, {start + count}) exceeds "
                f"stream length {self.total_tokens}"
            )
        if count == 0:
            return []

        result: list[int] = []
        position = start
        remaining = count
        shard_index = self._find_shard_index(position)

        while remaining > 0:
            shard: ShardInfo = self.manifest.shards[shard_index]
            local_start = position - shard.token_start
            available = shard.token_count - local_start
            take = min(remaining, available)

            path = self.manifest_path.parent / shard.filename
            with path.open("rb") as handle:
                handle.seek(local_start * TOKEN_ITEM_BYTES)
                data = handle.read(take * TOKEN_ITEM_BYTES)

            expected_bytes = take * TOKEN_ITEM_BYTES
            if len(data) != expected_bytes:
                raise ValueError(
                    f"short read from {shard.filename}: "
                    f"expected {expected_bytes} bytes, got {len(data)}"
                )

            ids = _decode_uint16(data)
            if len(ids) != take:
                raise RuntimeError("internal uint16 decoding length mismatch")

            for token_id in ids:
                if token_id >= self.vocab_size:
                    raise ValueError(
                        f"token id {token_id} in {shard.filename} is outside "
                        f"manifest vocabulary [0, {self.vocab_size})"
                    )

            result.extend(ids)
            position += take
            remaining -= take
            shard_index += 1

        if len(result) != count:
            raise RuntimeError("internal token-range read length mismatch")
        return result

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_index(index)

        start = index * self.context_length
        window = self.read_token_range(start, self.context_length + 1)

        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        target_ids = torch.tensor(window[1:], dtype=torch.long)

        if input_ids.shape != (self.context_length,):
            raise RuntimeError("internal input tensor shape invariant failed")
        if target_ids.shape != (self.context_length,):
            raise RuntimeError("internal target tensor shape invariant failed")
        if not torch.equal(input_ids[1:], target_ids[:-1]):
            raise RuntimeError("internal causal alignment invariant failed")

        return input_ids, target_ids

    def __repr__(self) -> str:
        return (
            f"PretrainingDataset(split={self.split!r}, "
            f"examples={len(self):,}, context_length={self.context_length}, "
            f"tokens={self.total_tokens:,}, vocab_size={self.vocab_size:,})"
        )
