"""Tests for deterministic uint16 pretraining token shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llm.data.shards import (
    BYTE_ORDER,
    MANIFEST_FILENAME,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    TOKEN_DTYPE,
    TOKEN_ITEM_BYTES,
    UINT16_MAX,
    load_shard_manifest,
    read_token_shard,
    read_token_shards,
    write_token_shards,
)
from llm.tokenizer.bpe import BPETrainer
from llm.tokenizer.tokenizer import DEFAULT_SPECIAL_TOKENS, Tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """Use the real project tokenizer with the same public API as production."""

    return Tokenizer.train(
        [
            "The transformer predicts the next token.\n",
            "HTTP != http; MHz != mHz; R = 4.7 kΩ ± 5%.\n",
            "def f(x):\n    return x * 2\n",
        ],
        vocab_size=512,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        min_pair_frequency=1,
    )


def test_single_shard_roundtrip_is_exact(tmp_path: Path, tokenizer: Tokenizer) -> None:
    tokens = [0, 1, 2, 255, 256, 300, 400, 500]

    manifest = write_token_shards(
        tokens,
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=100,
    )

    assert manifest.shard_count == 1
    assert read_token_shards(tmp_path / MANIFEST_FILENAME, tokenizer) == tokens


def test_binary_encoding_is_little_endian_uint16(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    write_token_shards(
        [0x0001, 0x0102, 0x01FF],
        tmp_path,
        tokenizer,
        split="train",
    )

    data = (tmp_path / "train-00000.bin").read_bytes()
    assert data == b"\x01\x00\x02\x01\xff\x01"


def test_each_token_occupies_exactly_two_bytes(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    tokens = list(range(100))
    manifest = write_token_shards(tokens, tmp_path, tokenizer, split="train")

    assert manifest.total_bytes == len(tokens) * 2
    assert (tmp_path / "train-00000.bin").stat().st_size == len(tokens) * 2


def test_multiple_shards_reconstruct_original_stream_without_loss(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    tokens = list(range(37))

    manifest = write_token_shards(
        tokens,
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=8,
    )

    assert manifest.shard_count == 5
    assert [s.token_count for s in manifest.shards] == [8, 8, 8, 8, 5]
    assert read_token_shards(tmp_path / MANIFEST_FILENAME, tokenizer) == tokens


def test_shard_boundaries_do_not_insert_duplicate_or_drop_tokens(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    tokens = list(range(21))
    manifest = write_token_shards(
        tokens,
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=5,
    )

    reconstructed: list[int] = []
    for shard in manifest.shards:
        reconstructed.extend(read_token_shard(tmp_path / shard.filename))

    assert reconstructed == tokens
    assert len(reconstructed) == len(tokens)


def test_shard_names_are_deterministic_and_zero_padded(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    manifest = write_token_shards(
        list(range(7)),
        tmp_path,
        tokenizer,
        split="validation",
        tokens_per_shard=3,
    )

    assert [s.filename for s in manifest.shards] == [
        "validation-00000.bin",
        "validation-00001.bin",
        "validation-00002.bin",
    ]


def test_manifest_records_format_and_storage_contract(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    write_token_shards([1, 2, 3], tmp_path, tokenizer, split="train")

    payload = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert payload["format"] == SHARD_FORMAT
    assert payload["format_version"] == SHARD_FORMAT_VERSION
    assert payload["dtype"] == TOKEN_DTYPE == "uint16"
    assert payload["byte_order"] == BYTE_ORDER == "little"
    assert payload["token_item_bytes"] == TOKEN_ITEM_BYTES == 2
    assert payload["tokenizer_vocab_size"] == tokenizer.vocab_size


def test_manifest_token_and_byte_totals_are_exact(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    tokens = list(range(19))
    manifest = write_token_shards(
        tokens,
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=6,
    )

    assert manifest.total_tokens == 19
    assert manifest.total_bytes == 38
    assert sum(s.token_count for s in manifest.shards) == 19
    assert sum(s.byte_count for s in manifest.shards) == 38


def test_manifest_global_token_ranges_are_contiguous(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    manifest = write_token_shards(
        list(range(11)),
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=4,
    )

    assert [(s.token_start, s.token_end) for s in manifest.shards] == [
        (0, 4),
        (4, 8),
        (8, 11),
    ]


def test_shard_sha256_matches_exact_binary_bytes(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    manifest = write_token_shards(
        list(range(10)),
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=4,
    )

    for shard in manifest.shards:
        data = (tmp_path / shard.filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == shard.sha256


def test_stream_sha256_hashes_shards_in_manifest_order(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    manifest = write_token_shards(
        list(range(13)),
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=5,
    )

    digest = hashlib.sha256()
    for shard in manifest.shards:
        digest.update((tmp_path / shard.filename).read_bytes())

    assert digest.hexdigest() == manifest.stream_sha256


def test_same_stream_and_settings_produce_identical_binary_files_and_manifest(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    tokens = list(range(31))
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_token_shards(
        tokens,
        first,
        tokenizer,
        split="train",
        tokens_per_shard=7,
    )
    write_token_shards(
        tokens,
        second,
        tokenizer,
        split="train",
        tokens_per_shard=7,
    )

    first_manifest = (first / MANIFEST_FILENAME).read_bytes()
    second_manifest = (second / MANIFEST_FILENAME).read_bytes()
    assert first_manifest == second_manifest

    first_bins = sorted(first.glob("*.bin"))
    second_bins = sorted(second.glob("*.bin"))
    assert [p.name for p in first_bins] == [p.name for p in second_bins]
    assert [p.read_bytes() for p in first_bins] == [p.read_bytes() for p in second_bins]


def test_generator_input_is_supported(tmp_path: Path, tokenizer: Tokenizer) -> None:
    manifest = write_token_shards(
        (token for token in range(17)),
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=5,
    )

    assert manifest.total_tokens == 17
    assert read_token_shards(tmp_path / MANIFEST_FILENAME, tokenizer) == list(range(17))


def test_empty_input_writes_manifest_but_no_binary_shards(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    manifest = write_token_shards([], tmp_path, tokenizer, split="train")

    assert manifest.total_tokens == 0
    assert manifest.total_bytes == 0
    assert manifest.shard_count == 0
    assert list(tmp_path.glob("*.bin")) == []
    assert (tmp_path / MANIFEST_FILENAME).exists()
    assert read_token_shards(tmp_path / MANIFEST_FILENAME, tokenizer) == []


@pytest.mark.parametrize("bad_token", [-1, 512, 65_535])
def test_writer_rejects_token_outside_tokenizer_vocabulary(
    tmp_path: Path,
    tokenizer: Tokenizer,
    bad_token: int,
) -> None:
    with pytest.raises(ValueError, match="outside tokenizer vocabulary"):
        write_token_shards([1, bad_token, 2], tmp_path, tokenizer, split="train")

    assert list(tmp_path.glob("*.bin")) == []
    assert not (tmp_path / MANIFEST_FILENAME).exists()


@pytest.mark.parametrize("bad_token", [True, 1.5, "2", None])
def test_writer_rejects_non_integer_and_bool_tokens(
    tmp_path: Path,
    tokenizer: Tokenizer,
    bad_token,
) -> None:
    with pytest.raises(TypeError, match="token ids must be integers"):
        write_token_shards([1, bad_token, 2], tmp_path, tokenizer, split="train")

    assert list(tmp_path.glob("*.bin")) == []


@pytest.mark.parametrize("tokens_per_shard", [0, -1])
def test_writer_rejects_non_positive_tokens_per_shard(
    tmp_path: Path,
    tokenizer: Tokenizer,
    tokens_per_shard: int,
) -> None:
    with pytest.raises(ValueError, match="tokens_per_shard must be > 0"):
        write_token_shards(
            [1, 2],
            tmp_path,
            tokenizer,
            split="train",
            tokens_per_shard=tokens_per_shard,
        )


@pytest.mark.parametrize("tokens_per_shard", [True, 4.0, "4"])
def test_writer_rejects_non_integer_tokens_per_shard(
    tmp_path: Path,
    tokenizer: Tokenizer,
    tokens_per_shard,
) -> None:
    with pytest.raises(TypeError, match="tokens_per_shard must be an integer"):
        write_token_shards(
            [1, 2],
            tmp_path,
            tokenizer,
            split="train",
            tokens_per_shard=tokens_per_shard,
        )


@pytest.mark.parametrize("split", ["", "train/data", "../train", "train data", "_train"])
def test_writer_rejects_unsafe_split_names(
    tmp_path: Path,
    tokenizer: Tokenizer,
    split: str,
) -> None:
    with pytest.raises(ValueError, match="split must contain only"):
        write_token_shards([1, 2], tmp_path, tokenizer, split=split)


def test_writer_rejects_non_string_split(tmp_path: Path, tokenizer: Tokenizer) -> None:
    with pytest.raises(TypeError, match="split must be a string"):
        write_token_shards([1, 2], tmp_path, tokenizer, split=123)  # type: ignore[arg-type]


def test_writer_rejects_existing_output_by_default(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    write_token_shards([1, 2, 3], tmp_path, tokenizer, split="train")

    with pytest.raises(FileExistsError, match="output already contains"):
        write_token_shards([4, 5, 6], tmp_path, tokenizer, split="train")


def test_overwrite_replaces_previous_split_files(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    write_token_shards(
        list(range(10)),
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=3,
    )
    assert len(list(tmp_path.glob("train-*.bin"))) == 4

    write_token_shards(
        [9, 8, 7],
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=10,
        overwrite=True,
    )

    assert [p.name for p in tmp_path.glob("train-*.bin")] == ["train-00000.bin"]
    assert read_token_shards(tmp_path / MANIFEST_FILENAME, tokenizer) == [9, 8, 7]


def test_unrelated_files_are_not_deleted_by_overwrite(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    write_token_shards([1, 2], tmp_path, tokenizer, split="train")

    write_token_shards([3, 4], tmp_path, tokenizer, split="train", overwrite=True)

    assert unrelated.read_text(encoding="utf-8") == "keep me"


def test_load_manifest_roundtrip_preserves_metadata(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    written = write_token_shards(
        list(range(15)),
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=4,
    )

    loaded = load_shard_manifest(tmp_path / MANIFEST_FILENAME)
    assert loaded == written


def test_read_one_shard_rejects_odd_byte_count(tmp_path: Path) -> None:
    path = tmp_path / "broken.bin"
    path.write_bytes(b"\x01\x00\xff")

    with pytest.raises(ValueError, match="divisible by 2"):
        read_token_shard(path)


def test_read_one_shard_can_verify_checksum(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    path.write_bytes(b"\x01\x00\x02\x00")

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_token_shard(path, expected_sha256="0" * 64)


def test_reader_detects_corrupted_shard_checksum(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    manifest = write_token_shards(
        list(range(12)),
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=5,
    )

    first_path = tmp_path / manifest.shards[0].filename
    data = bytearray(first_path.read_bytes())
    data[0] ^= 0x01
    first_path.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_token_shards(tmp_path / MANIFEST_FILENAME, tokenizer)


def test_reader_detects_truncated_shard_byte_count(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    manifest = write_token_shards(
        list(range(12)),
        tmp_path,
        tokenizer,
        split="train",
        tokens_per_shard=5,
    )

    first_path = tmp_path / manifest.shards[0].filename
    first_path.write_bytes(first_path.read_bytes()[:-2])

    with pytest.raises(ValueError, match="byte count mismatch"):
        read_token_shards(tmp_path / MANIFEST_FILENAME, tokenizer)


def test_reader_rejects_tokenizer_vocab_mismatch(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    write_token_shards([1, 2, 3], tmp_path, tokenizer, split="train")
    other = Tokenizer.train(
        ["abc abc abc"],
        vocab_size=513,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
    )

    with pytest.raises(ValueError, match="vocabulary size does not match"):
        read_token_shards(tmp_path / MANIFEST_FILENAME, other)


def test_uint16_storage_rejects_tokenizer_vocab_larger_than_65536(
    tmp_path: Path,
) -> None:
    vocab_size = 65_537
    content_vocab = vocab_size - len(DEFAULT_SPECIAL_TOKENS)
    too_large = Tokenizer(
        bpe=BPETrainer(vocab_size=content_vocab),
        vocab_size=vocab_size,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
    )

    with pytest.raises(ValueError, match="vocab_size <= 65536"):
        write_token_shards([1, 2], tmp_path, too_large, split="train")


def test_reader_without_tokenizer_still_roundtrips_valid_binary_data(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    tokens = [1, 255, 256, 400, 511]
    write_token_shards(tokens, tmp_path, tokenizer, split="train")

    assert read_token_shards(tmp_path / MANIFEST_FILENAME) == tokens


def test_manifest_rejects_tampered_total_token_count(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    write_token_shards([1, 2, 3], tmp_path, tokenizer, split="train")
    path = tmp_path / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["total_tokens"] = 999
    payload["total_bytes"] = 1998
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="token counts do not match"):
        load_shard_manifest(path)


def test_manifest_rejects_tampered_shard_filename(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    write_token_shards([1, 2, 3], tmp_path, tokenizer, split="train")
    path = tmp_path / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["shards"][0]["filename"] = "../escape.bin"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid shard filename"):
        load_shard_manifest(path)


def test_maximum_uint16_value_codec_is_lossless(tmp_path: Path) -> None:
    # read_token_shard is a raw storage-codec test and intentionally does not
    # require a tokenizer, so exercise the complete uint16 representable range.
    path = tmp_path / "max.bin"
    path.write_bytes(b"\xff\xff")
    assert read_token_shard(path) == [UINT16_MAX]


def test_failed_overwrite_preserves_previous_valid_dataset(
    tmp_path: Path,
    tokenizer: Tokenizer,
) -> None:
    original = [1, 2, 3, 4]
    write_token_shards(original, tmp_path, tokenizer, split="train")
    original_manifest_bytes = (tmp_path / MANIFEST_FILENAME).read_bytes()
    original_shard_bytes = (tmp_path / "train-00000.bin").read_bytes()

    with pytest.raises(ValueError, match="outside tokenizer vocabulary"):
        write_token_shards(
            [5, 6, tokenizer.vocab_size, 7],
            tmp_path,
            tokenizer,
            split="train",
            overwrite=True,
            tokens_per_shard=2,
        )

    assert (tmp_path / MANIFEST_FILENAME).read_bytes() == original_manifest_bytes
    assert (tmp_path / "train-00000.bin").read_bytes() == original_shard_bytes
    assert read_token_shards(tmp_path / MANIFEST_FILENAME, tokenizer) == original
    assert list(tmp_path.glob("*.buildtmp")) == []