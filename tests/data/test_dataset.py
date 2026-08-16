from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from llm.data.dataset import PretrainingDataset
from llm.data.shards import write_token_shards
from llm.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    corpus = [
        "The robot turns left at corners.",
        "HTTP != http; MHz != mHz.",
        "R = 4.7 kΩ ± 5%; C = 10 µF.",
        "def f(x):\n    return x * 2\n",
    ]
    return Tokenizer.train(corpus, vocab_size=512, min_pair_frequency=1)


def make_dataset(
    tmp_path: Path,
    tokenizer: Tokenizer,
    stream: list[int],
    *,
    context_length: int = 4,
    tokens_per_shard: int = 5,
    verify_checksums: bool = True,
) -> PretrainingDataset:
    split_dir = tmp_path / "train"
    write_token_shards(
        stream,
        split_dir,
        tokenizer,
        split="train",
        tokens_per_shard=tokens_per_shard,
    )
    return PretrainingDataset(
        split_dir / "manifest.json",
        context_length=context_length,
        expected_vocab_size=tokenizer.vocab_size,
        verify_checksums=verify_checksums,
    )


def test_length_formula_is_exact(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, list(range(18)))
    assert len(dataset) == (18 - 1) // 4 == 4


def test_item_shapes_and_dtypes(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, list(range(18)))
    x, y = dataset[0]
    assert x.shape == (4,)
    assert y.shape == (4,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_first_example_is_exact(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, list(range(18)))
    x, y = dataset[0]
    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 4]


def test_middle_example_is_exact(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, list(range(18)))
    x, y = dataset[2]
    assert x.tolist() == [8, 9, 10, 11]
    assert y.tolist() == [9, 10, 11, 12]


def test_final_complete_example_is_exact_and_tail_is_hidden(
    tmp_path: Path, tokenizer: Tokenizer
):
    dataset = make_dataset(tmp_path, tokenizer, list(range(18)))
    x, y = dataset[3]
    assert x.tolist() == [12, 13, 14, 15]
    assert y.tolist() == [13, 14, 15, 16]
    assert dataset.tail_token_count == 2
    with pytest.raises(IndexError):
        _ = dataset[4]


def test_alignment_invariant_for_every_example(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, list(range(30)))
    for index in range(len(dataset)):
        x, y = dataset[index]
        assert torch.equal(x[1:], y[:-1])


def test_example_crosses_binary_shard_boundary_exactly(
    tmp_path: Path, tokenizer: Tokenizer
):
    # 5 tokens per shard; example 1 is stream[4:9], crossing both boundaries.
    dataset = make_dataset(
        tmp_path,
        tokenizer,
        list(range(20)),
        context_length=4,
        tokens_per_shard=5,
    )
    x, y = dataset[1]
    assert x.tolist() == [4, 5, 6, 7]
    assert y.tolist() == [5, 6, 7, 8]


def test_read_token_range_can_cross_multiple_shards(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, list(range(30)), tokens_per_shard=3)
    assert dataset.read_token_range(2, 10) == list(range(2, 12))


def test_prediction_targets_have_no_gaps_or_duplicates(tmp_path: Path, tokenizer: Tokenizer):
    stream = list(range(22))
    dataset = make_dataset(tmp_path, tokenizer, stream)
    targets: list[int] = []
    for index in range(len(dataset)):
        _, y = dataset[index]
        targets.extend(y.tolist())
    assert targets == stream[1 : 1 + len(dataset) * 4]


def test_empty_stream_has_zero_examples(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, [])
    assert len(dataset) == 0
    assert dataset.total_tokens == 0
    assert dataset.tail_token_count == 0


def test_single_token_stream_has_zero_examples(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, [1])
    assert len(dataset) == 0
    assert dataset.tail_token_count == 1


def test_exact_window_stream_has_one_carry_tail(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, list(range(9)), context_length=4)
    assert len(dataset) == 2
    assert dataset.tail_token_count == 1


def test_negative_index_is_explicitly_rejected(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, list(range(10)))
    with pytest.raises(IndexError, match="negative"):
        _ = dataset[-1]


@pytest.mark.parametrize("index", [True, 1.5, "1"])
def test_non_integer_index_is_rejected(tmp_path: Path, tokenizer: Tokenizer, index):
    dataset = make_dataset(tmp_path, tokenizer, list(range(10)))
    with pytest.raises(TypeError):
        _ = dataset[index]


def test_out_of_range_index_is_rejected(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, list(range(10)))
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


@pytest.mark.parametrize("context_length", [0, -1])
def test_non_positive_context_length_is_rejected(
    tmp_path: Path, tokenizer: Tokenizer, context_length: int
):
    split_dir = tmp_path / "train"
    write_token_shards([1, 2, 3], split_dir, tokenizer, split="train")
    with pytest.raises(ValueError):
        PretrainingDataset(split_dir / "manifest.json", context_length=context_length)


@pytest.mark.parametrize("context_length", [True, 4.0, "4"])
def test_non_integer_context_length_is_rejected(
    tmp_path: Path, tokenizer: Tokenizer, context_length
):
    split_dir = tmp_path / "train"
    write_token_shards([1, 2, 3], split_dir, tokenizer, split="train")
    with pytest.raises(TypeError):
        PretrainingDataset(split_dir / "manifest.json", context_length=context_length)


def test_vocab_mismatch_is_rejected(tmp_path: Path, tokenizer: Tokenizer):
    split_dir = tmp_path / "train"
    write_token_shards([1, 2, 3], split_dir, tokenizer, split="train")
    with pytest.raises(ValueError, match="vocabulary"):
        PretrainingDataset(
            split_dir / "manifest.json",
            expected_vocab_size=tokenizer.vocab_size + 1,
        )


def test_missing_shard_is_detected_at_construction(tmp_path: Path, tokenizer: Tokenizer):
    split_dir = tmp_path / "train"
    manifest = write_token_shards([1, 2, 3], split_dir, tokenizer, split="train")
    (split_dir / manifest.shards[0].filename).unlink()
    with pytest.raises(FileNotFoundError):
        PretrainingDataset(split_dir / "manifest.json")


def test_corrupted_shard_checksum_is_detected(tmp_path: Path, tokenizer: Tokenizer):
    split_dir = tmp_path / "train"
    manifest = write_token_shards([1, 2, 3, 4], split_dir, tokenizer, split="train")
    shard_path = split_dir / manifest.shards[0].filename
    data = bytearray(shard_path.read_bytes())
    data[0] ^= 0x01
    shard_path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        PretrainingDataset(split_dir / "manifest.json", verify_checksums=True)


def test_checksum_verification_can_be_disabled_explicitly(
    tmp_path: Path, tokenizer: Tokenizer
):
    split_dir = tmp_path / "train"
    manifest = write_token_shards([1, 2, 3, 4, 5], split_dir, tokenizer, split="train")
    shard_path = split_dir / manifest.shards[0].filename
    data = bytearray(shard_path.read_bytes())
    # Change token 1 -> 2 but keep it inside vocabulary and same byte length.
    data[0:2] = (2).to_bytes(2, "little")
    shard_path.write_bytes(data)
    dataset = PretrainingDataset(
        split_dir / "manifest.json",
        context_length=2,
        verify_checksums=False,
    )
    x, _ = dataset[0]
    assert x[0].item() == 2


def test_out_of_vocab_token_is_detected_on_read_when_checksums_disabled(
    tmp_path: Path, tokenizer: Tokenizer
):
    split_dir = tmp_path / "train"
    manifest = write_token_shards([1, 2, 3, 4, 5], split_dir, tokenizer, split="train")
    shard_path = split_dir / manifest.shards[0].filename
    data = bytearray(shard_path.read_bytes())
    invalid = tokenizer.vocab_size
    data[0:2] = invalid.to_bytes(2, "little")
    shard_path.write_bytes(data)
    dataset = PretrainingDataset(
        split_dir / "manifest.json",
        context_length=2,
        verify_checksums=False,
    )
    with pytest.raises(ValueError, match="outside"):
        _ = dataset[0]


def test_dataset_reads_are_deterministic(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, list(range(20)))
    first = dataset[2]
    second = dataset[2]
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_dataloader_batches_have_model_ready_shape(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, list(range(30)), context_length=4)
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    x, y = next(iter(loader))
    assert x.shape == (3, 4)
    assert y.shape == (3, 4)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_repr_contains_split_geometry(tmp_path: Path, tokenizer: Tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, list(range(18)))
    value = repr(dataset)
    assert "PretrainingDataset" in value
    assert "train" in value
    assert "context_length=4" in value
