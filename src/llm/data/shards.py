"""Deterministic uint16 binary shard storage for pretraining token streams.

This module is the persistence layer immediately after document encoding and
packing.  It stores a *contiguous token stream* in deterministic ``.bin``
files plus a JSON manifest.

Why raw token-stream shards instead of serialized Python/JSON lists?

* Token ids are integers and the frozen tokenizer vocabulary is 24,000, so
  every valid id fits in unsigned 16-bit storage.
* ``uint16`` needs exactly two bytes per token on disk.
* Binary shards are compact, fast to read, and preserve token order exactly.
* Shard boundaries are storage boundaries only: no padding, BOS/EOS, overlap,
  or separator token is inserted by this module.

File format
-----------
Each ``.bin`` file is a flat sequence of little-endian unsigned 16-bit token
ids.  If a stream is split as::

    [t0, t1, ..., tN]

then concatenating the decoded shards reconstructs that exact sequence.  This
module deliberately does not construct causal windows or PyTorch tensors; those
remain responsibilities of :mod:`llm.data.packing` and the later dataset layer.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

from llm.tokenizer.tokenizer import Tokenizer


SHARD_FORMAT = "llm_uint16_token_shards"
SHARD_FORMAT_VERSION = 1
TOKEN_DTYPE = "uint16"
BYTE_ORDER = "little"
TOKEN_ITEM_BYTES = 2
UINT16_MAX = 65_535
UINT16_VOCAB_CAPACITY = UINT16_MAX + 1
DEFAULT_TOKENS_PER_SHARD = 1_000_000
MANIFEST_FILENAME = "manifest.json"
_SPLIT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class ShardInfo:
    """Metadata for one binary token shard."""

    filename: str
    index: int
    token_start: int
    token_end: int
    token_count: int
    byte_count: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "index": self.index,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_count": self.token_count,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ShardInfo":
        if not isinstance(payload, dict):
            raise ValueError("shard manifest entry must be an object")

        try:
            return cls(
                filename=str(payload["filename"]),
                index=int(payload["index"]),
                token_start=int(payload["token_start"]),
                token_end=int(payload["token_end"]),
                token_count=int(payload["token_count"]),
                byte_count=int(payload["byte_count"]),
                sha256=str(payload["sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid shard manifest entry") from exc


@dataclass(frozen=True)
class ShardManifest:
    """Complete description of one split's binary token shards."""

    split: str
    tokenizer_vocab_size: int
    tokens_per_shard: int
    total_tokens: int
    total_bytes: int
    stream_sha256: str
    shards: tuple[ShardInfo, ...]

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": SHARD_FORMAT,
            "format_version": SHARD_FORMAT_VERSION,
            "split": self.split,
            "dtype": TOKEN_DTYPE,
            "byte_order": BYTE_ORDER,
            "token_item_bytes": TOKEN_ITEM_BYTES,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "tokens_per_shard": self.tokens_per_shard,
            "total_tokens": self.total_tokens,
            "total_bytes": self.total_bytes,
            "stream_sha256": self.stream_sha256,
            "shard_count": self.shard_count,
            "shards": [shard.to_dict() for shard in self.shards],
        }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_tokenizer(tokenizer: Tokenizer) -> None:
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must be a Tokenizer, got {type(tokenizer).__name__}"
        )
    if tokenizer.vocab_size > UINT16_VOCAB_CAPACITY:
        raise ValueError(
            "uint16 shards require tokenizer.vocab_size <= 65536; "
            f"got {tokenizer.vocab_size}"
        )


def _validate_split(split: str) -> None:
    if not isinstance(split, str):
        raise TypeError("split must be a string")
    if not _SPLIT_PATTERN.fullmatch(split):
        raise ValueError(
            "split must contain only letters, digits, '_' or '-', and must "
            "start with a letter or digit"
        )


def _validate_tokens_per_shard(tokens_per_shard: int) -> None:
    if not isinstance(tokens_per_shard, int) or isinstance(tokens_per_shard, bool):
        raise TypeError("tokens_per_shard must be an integer")
    if tokens_per_shard <= 0:
        raise ValueError("tokens_per_shard must be > 0")


def _validate_token_id(token_id: int, tokenizer: Tokenizer) -> None:
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        raise TypeError(
            f"token ids must be integers, got {type(token_id).__name__}"
        )
    if token_id < 0 or token_id >= tokenizer.vocab_size:
        raise ValueError(
            f"token id {token_id} is outside tokenizer vocabulary "
            f"[0, {tokenizer.vocab_size})"
        )
    if token_id > UINT16_MAX:
        # This should already be impossible after the tokenizer-vocab check, but
        # keep the storage invariant explicit at the token boundary.
        raise ValueError(f"token id {token_id} cannot be represented as uint16")


def _validate_safe_filename(filename: str) -> None:
    path = Path(filename)
    if path.name != filename or filename in {"", ".", ".."}:
        raise ValueError(f"invalid shard filename in manifest: {filename!r}")


# ---------------------------------------------------------------------------
# uint16 codec
# ---------------------------------------------------------------------------


def _uint16_bytes(token_ids: Iterable[int]) -> bytes:
    """Encode ids as portable little-endian uint16 bytes."""

    values = array("H", token_ids)
    if values.itemsize != TOKEN_ITEM_BYTES:
        raise RuntimeError("platform unsigned-short size is not 2 bytes")
    if sys.byteorder != BYTE_ORDER:
        values.byteswap()
    return values.tobytes()


def _decode_uint16_bytes(data: bytes) -> list[int]:
    if len(data) % TOKEN_ITEM_BYTES != 0:
        raise ValueError(
            "uint16 shard byte length must be divisible by 2; "
            f"got {len(data)} bytes"
        )

    values = array("H")
    values.frombytes(data)
    if values.itemsize != TOKEN_ITEM_BYTES:
        raise RuntimeError("platform unsigned-short size is not 2 bytes")
    if sys.byteorder != BYTE_ORDER:
        values.byteswap()
    return list(values)


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def _validate_manifest(manifest: ShardManifest) -> None:
    _validate_split(manifest.split)
    _validate_tokens_per_shard(manifest.tokens_per_shard)

    if manifest.tokenizer_vocab_size <= 0:
        raise ValueError("manifest tokenizer_vocab_size must be > 0")
    if manifest.tokenizer_vocab_size > UINT16_VOCAB_CAPACITY:
        raise ValueError("manifest tokenizer vocabulary does not fit uint16")
    if manifest.total_tokens < 0 or manifest.total_bytes < 0:
        raise ValueError("manifest totals must be non-negative")
    if manifest.total_bytes != manifest.total_tokens * TOKEN_ITEM_BYTES:
        raise ValueError("manifest total_bytes does not equal total_tokens * 2")
    if len(manifest.stream_sha256) != 64:
        raise ValueError("manifest stream_sha256 must be a SHA-256 hex digest")

    expected_start = 0
    total_tokens = 0
    total_bytes = 0

    for expected_index, shard in enumerate(manifest.shards):
        _validate_safe_filename(shard.filename)
        expected_filename = f"{manifest.split}-{expected_index:05d}.bin"
        if shard.filename != expected_filename:
            raise ValueError(
                f"unexpected shard filename {shard.filename!r}; "
                f"expected {expected_filename!r}"
            )
        if shard.index != expected_index:
            raise ValueError("shard indexes must be contiguous starting at zero")
        if shard.token_start != expected_start:
            raise ValueError("shard token ranges must be contiguous")
        if shard.token_count <= 0:
            raise ValueError("manifest must not contain empty shards")
        if shard.token_count > manifest.tokens_per_shard:
            raise ValueError("shard token_count exceeds tokens_per_shard")
        if shard.token_end != shard.token_start + shard.token_count:
            raise ValueError("shard token_end is inconsistent with token_count")
        if shard.byte_count != shard.token_count * TOKEN_ITEM_BYTES:
            raise ValueError("shard byte_count does not equal token_count * 2")
        if len(shard.sha256) != 64:
            raise ValueError("shard sha256 must be a SHA-256 hex digest")

        expected_start = shard.token_end
        total_tokens += shard.token_count
        total_bytes += shard.byte_count

    if total_tokens != manifest.total_tokens:
        raise ValueError("manifest shard token counts do not match total_tokens")
    if total_bytes != manifest.total_bytes:
        raise ValueError("manifest shard byte counts do not match total_bytes")

    if manifest.total_tokens == 0 and manifest.shards:
        raise ValueError("empty token stream must not contain shard entries")
    if manifest.total_tokens > 0 and not manifest.shards:
        raise ValueError("non-empty token stream must contain shard entries")


# ---------------------------------------------------------------------------
# Public write/read API
# ---------------------------------------------------------------------------


def write_token_shards(
    token_ids: Iterable[int],
    output_dir: str | Path,
    tokenizer: Tokenizer,
    *,
    split: str,
    tokens_per_shard: int = DEFAULT_TOKENS_PER_SHARD,
    overwrite: bool = False,
) -> ShardManifest:
    """Write a contiguous token stream to deterministic uint16 shards.

    No token is added, removed, duplicated, or reordered.  Shard boundaries are
    purely storage boundaries.

    One output directory represents exactly one dataset split (for example,
    ``data/tokenized/train`` or ``data/tokenized/validation``).

    ``overwrite=False`` rejects an existing manifest or existing shard files for
    the same split, preventing stale data from being silently mixed into a new
    dataset build.  With ``overwrite=True``, the new split is fully encoded and
    validated into temporary files before existing split files are replaced.
    Unrelated files are left untouched.
    """

    _require_tokenizer(tokenizer)
    _validate_split(split)
    _validate_tokens_per_shard(tokens_per_shard)
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / MANIFEST_FILENAME
    existing_shards = sorted(output_path.glob(f"{split}-*.bin"))

    if not overwrite and (manifest_path.exists() or existing_shards):
        raise FileExistsError(
            f"output already contains a manifest or {split!r} shard files: "
            f"{output_path}"
        )

    # Do not delete an existing valid dataset yet.  Even when overwrite=True,
    # first build and validate the replacement into temporary files.
    temp_shard_paths: list[tuple[Path, Path]] = []
    temp_manifest = output_path / f".{MANIFEST_FILENAME}.buildtmp"
    if temp_manifest.exists():
        temp_manifest.unlink()

    shard_infos: list[ShardInfo] = []
    stream_digest = hashlib.sha256()
    buffer: list[int] = []
    total_tokens = 0

    def flush_buffer() -> None:
        nonlocal buffer, total_tokens
        if not buffer:
            return

        index = len(shard_infos)
        filename = f"{split}-{index:05d}.bin"
        final_path = output_path / filename
        temp_path = output_path / f".{filename}.buildtmp"

        data = _uint16_bytes(buffer)
        shard_digest = hashlib.sha256(data).hexdigest()

        temp_path.write_bytes(data)
        temp_shard_paths.append((temp_path, final_path))

        token_start = total_tokens
        token_count = len(buffer)
        token_end = token_start + token_count

        shard_infos.append(
            ShardInfo(
                filename=filename,
                index=index,
                token_start=token_start,
                token_end=token_end,
                token_count=token_count,
                byte_count=len(data),
                sha256=shard_digest,
            )
        )
        stream_digest.update(data)
        total_tokens = token_end
        buffer = []

    try:
        for token_id in token_ids:
            _validate_token_id(token_id, tokenizer)
            buffer.append(token_id)
            if len(buffer) == tokens_per_shard:
                flush_buffer()

        flush_buffer()

        manifest = ShardManifest(
            split=split,
            tokenizer_vocab_size=tokenizer.vocab_size,
            tokens_per_shard=tokens_per_shard,
            total_tokens=total_tokens,
            total_bytes=total_tokens * TOKEN_ITEM_BYTES,
            stream_sha256=stream_digest.hexdigest(),
            shards=tuple(shard_infos),
        )
        _validate_manifest(manifest)

        serialized = json.dumps(
            manifest.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        ) + "\n"
        temp_manifest.write_text(serialized, encoding="utf-8", newline="\n")

        # Commit only after the complete replacement has been encoded and
        # structurally validated.  Replace same-name shards first, remove any
        # stale extra shards from an older build, and switch the manifest last.
        new_filenames = {final.name for _, final in temp_shard_paths}
        for temp_path, final_path in temp_shard_paths:
            temp_path.replace(final_path)
        if overwrite:
            for old_path in existing_shards:
                if old_path.name not in new_filenames and old_path.exists():
                    old_path.unlink()
        temp_manifest.replace(manifest_path)

        return manifest

    except Exception:
        # Failed encoding/validation must not leave temporary build products.
        # Existing final files and the existing manifest are intentionally not
        # removed here.
        for temp_path, _ in temp_shard_paths:
            if temp_path.exists():
                temp_path.unlink()
        if temp_manifest.exists():
            temp_manifest.unlink()
        raise


def load_shard_manifest(path: str | Path) -> ShardManifest:
    """Load and structurally validate a shard manifest."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shard manifest must contain a JSON object")

    if payload.get("format") != SHARD_FORMAT:
        raise ValueError("unsupported or missing shard format")
    if payload.get("format_version") != SHARD_FORMAT_VERSION:
        raise ValueError(
            f"unsupported shard format version: {payload.get('format_version')!r}"
        )
    if payload.get("dtype") != TOKEN_DTYPE:
        raise ValueError("unsupported shard dtype")
    if payload.get("byte_order") != BYTE_ORDER:
        raise ValueError("unsupported shard byte order")
    if payload.get("token_item_bytes") != TOKEN_ITEM_BYTES:
        raise ValueError("unsupported token item width")

    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("manifest shards must be a list")

    try:
        manifest = ShardManifest(
            split=str(payload["split"]),
            tokenizer_vocab_size=int(payload["tokenizer_vocab_size"]),
            tokens_per_shard=int(payload["tokens_per_shard"]),
            total_tokens=int(payload["total_tokens"]),
            total_bytes=int(payload["total_bytes"]),
            stream_sha256=str(payload["stream_sha256"]),
            shards=tuple(ShardInfo.from_dict(item) for item in raw_shards),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid shard manifest") from exc

    declared_shard_count = payload.get("shard_count")
    if declared_shard_count != len(manifest.shards):
        raise ValueError("manifest shard_count does not match shard entries")

    _validate_manifest(manifest)
    return manifest


def read_token_shard(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> list[int]:
    """Read one little-endian uint16 shard and optionally verify its checksum."""

    shard_path = Path(path)
    data = shard_path.read_bytes()

    if expected_sha256 is not None:
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"shard checksum mismatch for {shard_path.name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    return _decode_uint16_bytes(data)


def read_token_shards(
    manifest_path: str | Path,
    tokenizer: Tokenizer | None = None,
    *,
    verify_checksums: bool = True,
) -> list[int]:
    """Read all shards in manifest order and reconstruct the exact token stream."""

    if not isinstance(verify_checksums, bool):
        raise TypeError("verify_checksums must be a bool")

    manifest_file = Path(manifest_path)
    manifest = load_shard_manifest(manifest_file)

    if tokenizer is not None:
        _require_tokenizer(tokenizer)
        if tokenizer.vocab_size != manifest.tokenizer_vocab_size:
            raise ValueError(
                "tokenizer vocabulary size does not match shard manifest: "
                f"{tokenizer.vocab_size} != {manifest.tokenizer_vocab_size}"
            )

    stream: list[int] = []
    stream_digest = hashlib.sha256()

    for shard in manifest.shards:
        shard_path = manifest_file.parent / shard.filename
        data = shard_path.read_bytes()

        if len(data) != shard.byte_count:
            raise ValueError(
                f"shard byte count mismatch for {shard.filename}: "
                f"expected {shard.byte_count}, got {len(data)}"
            )

        if verify_checksums:
            actual_sha256 = hashlib.sha256(data).hexdigest()
            if actual_sha256 != shard.sha256:
                raise ValueError(
                    f"shard checksum mismatch for {shard.filename}: "
                    f"expected {shard.sha256}, got {actual_sha256}"
                )

        ids = _decode_uint16_bytes(data)
        if len(ids) != shard.token_count:
            raise ValueError(
                f"shard token count mismatch for {shard.filename}: "
                f"expected {shard.token_count}, got {len(ids)}"
            )

        if tokenizer is not None:
            for token_id in ids:
                _validate_token_id(token_id, tokenizer)

        stream.extend(ids)
        stream_digest.update(data)

    if len(stream) != manifest.total_tokens:
        raise ValueError("reconstructed token count does not match manifest")

    actual_stream_sha256 = stream_digest.hexdigest()
    if verify_checksums and actual_stream_sha256 != manifest.stream_sha256:
        raise ValueError(
            "reconstructed stream checksum does not match manifest: "
            f"expected {manifest.stream_sha256}, got {actual_stream_sha256}"
        )

    return stream