from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.data.dataset import PretrainingDataset
from llm.data.dataset_training import (
    DATASET_MANIFEST_FILENAME,
    TrainingDocument,
    build_pretraining_dataset,
    split_training_documents,
)
from llm.data.shards import read_token_shards
from llm.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    corpus = [
        "The robot turns left at corners.",
        "HTTP != http; MHz != mHz.",
        "R = 4.7 kΩ ± 5%; C = 10 µF.",
        "Software systems require tests and deterministic data pipelines.",
    ]
    return Tokenizer.train(corpus, vocab_size=512, min_pair_frequency=1)


def sample_documents(count: int = 20) -> list[TrainingDocument]:
    return [
        TrainingDocument(
            document_id=f"doc-{i:03d}",
            text=(f"Document {i}. " + "engineering software science " * (i % 4 + 1)),
        )
        for i in range(count)
    ]


def test_split_is_deterministic():
    docs = sample_documents()
    a_train, a_val = split_training_documents(docs, validation_fraction=0.2, seed=42)
    b_train, b_val = split_training_documents(docs, validation_fraction=0.2, seed=42)
    assert [d.document_id for d in a_train] == [d.document_id for d in b_train]
    assert [d.document_id for d in a_val] == [d.document_id for d in b_val]


def test_split_preserves_original_order_within_each_split():
    docs = sample_documents()
    train, val = split_training_documents(docs, validation_fraction=0.2, seed=42)
    positions = {d.document_id: i for i, d in enumerate(docs)}
    assert [positions[d.document_id] for d in train] == sorted(
        positions[d.document_id] for d in train
    )
    assert [positions[d.document_id] for d in val] == sorted(
        positions[d.document_id] for d in val
    )


def test_no_document_id_appears_in_both_splits():
    train, val = split_training_documents(
        sample_documents(), validation_fraction=0.2, seed=42
    )
    assert {d.document_id for d in train}.isdisjoint({d.document_id for d in val})


def test_split_group_keeps_chunks_together():
    docs = [
        TrainingDocument("a-0", "alpha zero", split_group="source-a"),
        TrainingDocument("a-1", "alpha one", split_group="source-a"),
        TrainingDocument("b-0", "beta zero", split_group="source-b"),
        TrainingDocument("c-0", "gamma zero", split_group="source-c"),
    ]
    train, val = split_training_documents(docs, validation_fraction=0.4, seed=3)
    assignment = {}
    for split_name, members in (("train", train), ("validation", val)):
        for member in members:
            assignment[member.document_id] = split_name
    assert assignment["a-0"] == assignment["a-1"]


def test_validation_zero_puts_everything_in_train():
    docs = sample_documents(5)
    train, val = split_training_documents(docs, validation_fraction=0.0, seed=42)
    assert train == docs
    assert val == []


def test_positive_validation_requires_two_groups():
    docs = [TrainingDocument("doc", "text", split_group="one")]
    with pytest.raises(ValueError, match="two distinct"):
        split_training_documents(docs, validation_fraction=0.1)


def test_duplicate_document_id_is_rejected():
    docs = [TrainingDocument("same", "one"), TrainingDocument("same", "two")]
    with pytest.raises(ValueError, match="duplicate document_id"):
        split_training_documents(docs, validation_fraction=0.0)


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.1])
def test_invalid_validation_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        split_training_documents(sample_documents(), validation_fraction=fraction)


def test_build_creates_both_split_manifests_and_top_manifest(
    tmp_path: Path, tokenizer: Tokenizer
):
    output = tmp_path / "dataset"
    result = build_pretraining_dataset(
        sample_documents(),
        tokenizer,
        output,
        validation_fraction=0.2,
        seed=42,
        context_length=8,
        tokens_per_shard=13,
    )
    assert (output / "train" / "manifest.json").is_file()
    assert (output / "validation" / "manifest.json").is_file()
    assert (output / DATASET_MANIFEST_FILENAME).is_file()
    assert result.dataset_manifest_path == output / DATASET_MANIFEST_FILENAME


def test_built_shards_are_readable_and_dataset_ready(tmp_path: Path, tokenizer: Tokenizer):
    output = tmp_path / "dataset"
    build_pretraining_dataset(
        sample_documents(),
        tokenizer,
        output,
        validation_fraction=0.2,
        seed=42,
        context_length=8,
        tokens_per_shard=17,
    )
    train = PretrainingDataset(
        output / "train" / "manifest.json",
        context_length=8,
        expected_vocab_size=tokenizer.vocab_size,
    )
    validation = PretrainingDataset(
        output / "validation" / "manifest.json",
        context_length=8,
        expected_vocab_size=tokenizer.vocab_size,
    )
    assert len(train) > 0
    assert len(validation) > 0
    x, y = train[0]
    assert x.shape == (8,)
    assert y.shape == (8,)


def test_document_boundaries_survive_in_flat_split_stream(
    tmp_path: Path, tokenizer: Tokenizer
):
    output = tmp_path / "dataset"
    result = build_pretraining_dataset(
        sample_documents(10),
        tokenizer,
        output,
        validation_fraction=0.2,
        seed=42,
        context_length=8,
        tokens_per_shard=11,
    )
    stream = read_token_shards(output / "train" / "manifest.json", tokenizer)
    bos = tokenizer.token_to_id("<|bos|>")
    eos = tokenizer.token_to_id("<|eos|>")
    assert bos is not None and eos is not None
    assert stream[0] == bos
    assert stream[-1] == eos
    # Every interior EOS should be followed immediately by the next BOS.
    for i, token_id in enumerate(stream[:-1]):
        if token_id == eos:
            assert stream[i + 1] == bos
    assert result.train_manifest.total_tokens == len(stream)


def test_top_manifest_records_geometry_and_split_policy(tmp_path: Path, tokenizer: Tokenizer):
    output = tmp_path / "dataset"
    build_pretraining_dataset(
        sample_documents(),
        tokenizer,
        output,
        validation_fraction=0.2,
        seed=99,
        context_length=8,
        tokens_per_shard=17,
    )
    payload = json.loads((output / DATASET_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert payload["format"] == "llm_pretraining_dataset"
    assert payload["tokenizer"]["vocab_size"] == tokenizer.vocab_size
    assert payload["split_policy"]["seed"] == 99
    assert payload["split_policy"]["split_before_tokenization"] is True
    assert payload["split_policy"]["group_safe"] is True
    assert payload["training_geometry"]["context_length"] == 8
    assert payload["training_geometry"]["window_tokens"] == 9
    assert payload["training_geometry"]["window_stride"] == 8
    assert payload["training_geometry"]["incomplete_final_tail_padded"] is False


def test_same_documents_and_settings_produce_identical_dataset_manifest(
    tmp_path: Path, tokenizer: Tokenizer
):
    a = tmp_path / "a"
    b = tmp_path / "b"
    kwargs = dict(
        validation_fraction=0.2,
        seed=42,
        context_length=8,
        tokens_per_shard=17,
    )
    build_pretraining_dataset(sample_documents(), tokenizer, a, **kwargs)
    build_pretraining_dataset(sample_documents(), tokenizer, b, **kwargs)
    assert (a / DATASET_MANIFEST_FILENAME).read_bytes() == (
        b / DATASET_MANIFEST_FILENAME
    ).read_bytes()


def test_existing_output_is_rejected_by_default(tmp_path: Path, tokenizer: Tokenizer):
    output = tmp_path / "dataset"
    build_pretraining_dataset(
        sample_documents(), tokenizer, output, validation_fraction=0.2
    )
    with pytest.raises(FileExistsError):
        build_pretraining_dataset(
            sample_documents(), tokenizer, output, validation_fraction=0.2
        )


def test_overwrite_replaces_complete_dataset(tmp_path: Path, tokenizer: Tokenizer):
    output = tmp_path / "dataset"
    first = sample_documents(10)
    second = sample_documents(14)
    build_pretraining_dataset(first, tokenizer, output, validation_fraction=0.2)
    old = (output / DATASET_MANIFEST_FILENAME).read_bytes()
    build_pretraining_dataset(
        second,
        tokenizer,
        output,
        validation_fraction=0.2,
        overwrite=True,
    )
    new = (output / DATASET_MANIFEST_FILENAME).read_bytes()
    assert new != old
    assert not (tmp_path / ".dataset.backup").exists()
    assert not (tmp_path / ".dataset.buildtmp").exists()


def test_literal_special_token_text_is_not_structural_inside_training_document(
    tmp_path: Path, tokenizer: Tokenizer
):
    docs = [
        TrainingDocument("a", "ordinary <|assistant|> literal text"),
        TrainingDocument("b", "second independent document"),
        TrainingDocument("c", "third independent document"),
    ]
    output = tmp_path / "dataset"
    build_pretraining_dataset(
        docs,
        tokenizer,
        output,
        validation_fraction=0.34,
        seed=1,
        context_length=2,
        tokens_per_shard=7,
    )
    special_id = tokenizer.token_to_id("<|assistant|>")
    assert special_id is not None
    train_stream = read_token_shards(output / "train" / "manifest.json", tokenizer)
    val_stream = read_token_shards(output / "validation" / "manifest.json", tokenizer)
    assert special_id not in train_stream
    assert special_id not in val_stream
