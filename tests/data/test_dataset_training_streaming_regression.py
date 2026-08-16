"""Regression tests for the streaming pretraining-dataset build.

These tests intentionally contain a full copy of the original list-based
implementation.  The streaming rewrite exists to remove a memory ceiling, not
to change a single token, so the only convincing evidence is that both
implementations produce byte-identical output.

The reference below is the pre-refactor code verbatim: it materialises every
document, splits the list, and writes the two token streams.  It calls the same
private helpers the original called, so if one of those helpers ever changes
semantics, this reference changes with it and the comparison stays honest about
what it is actually testing.

Run:

    python -m pytest tests/data/test_dataset_training_streaming_regression.py -q
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import pytest

from llm.data import dataset_training as dt
from llm.data.dataset_training import (
    DATASET_MANIFEST_FILENAME,
    TrainingDocument,
    build_pretraining_dataset,
    plan_training_split,
    split_training_documents,
)
from llm.data.shards import write_token_shards
from llm.tokenizer import Tokenizer


# --------------------------------------------------------------------------
# reference implementation (pre-refactor, verbatim)
# --------------------------------------------------------------------------


def reference_split(documents, *, validation_fraction, seed):
    """The original in-memory split."""

    materialized = [dt._validate_document(document) for document in documents]

    seen_ids: set[str] = set()
    groups: dict[str, list[TrainingDocument]] = {}
    for document in materialized:
        if document.document_id in seen_ids:
            raise ValueError(f"duplicate document_id: {document.document_id!r}")
        seen_ids.add(document.document_id)
        groups.setdefault(document.effective_split_group, []).append(document)

    if not materialized or float(validation_fraction) == 0.0:
        return list(materialized), []

    if len(groups) < 2:
        raise ValueError(
            "validation_fraction > 0 requires at least two distinct split groups"
        )

    ranked_groups: list[tuple[bytes, str, int]] = []
    total_bytes = 0
    for group, members in groups.items():
        group_bytes = sum(dt._normalized_utf8_bytes(document) for document in members)
        total_bytes += group_bytes
        ranked_groups.append((dt._group_rank(group, seed), group, group_bytes))

    ranked_groups.sort(key=lambda item: (item[0], item[1]))
    target_validation_bytes = total_bytes * float(validation_fraction)

    cumulative = 0
    candidates: list[tuple[float, int]] = []
    for prefix_count, (_, _, group_bytes) in enumerate(ranked_groups[:-1], start=1):
        cumulative += group_bytes
        candidates.append((abs(cumulative - target_validation_bytes), prefix_count))

    _, chosen_prefix_count = min(candidates, key=lambda item: (item[0], item[1]))
    validation_groups = {group for _, group, _ in ranked_groups[:chosen_prefix_count]}

    train: list[TrainingDocument] = []
    validation: list[TrainingDocument] = []
    for document in materialized:
        if document.effective_split_group in validation_groups:
            validation.append(document)
        else:
            train.append(document)

    if not train or not validation:
        raise RuntimeError("internal split invariant produced an empty required split")

    return train, validation


def reference_split_summary(documents, manifest, *, context_length):
    """The original per-split summary, computed from a document list."""

    characters = 0
    utf8_bytes = 0
    groups: set[str] = set()

    for document in documents:
        normalized = dt.normalize_text(document.text)
        characters += len(normalized)
        utf8_bytes += len(normalized.encode("utf-8", errors="strict"))
        groups.add(document.effective_split_group)

    examples = (
        (manifest.total_tokens - 1) // context_length
        if manifest.total_tokens > 1
        else 0
    )

    return {
        "document_count": len(documents),
        "split_group_count": len(groups),
        "normalized_characters": characters,
        "normalized_utf8_bytes": utf8_bytes,
        "document_identity_sha256": dt._document_identity_sha256(documents),
        "total_tokens": manifest.total_tokens,
        "shard_count": manifest.shard_count,
        "complete_examples": examples,
        "prediction_tokens": examples * context_length,
        "tail_tokens": (
            manifest.total_tokens - examples * context_length
            if manifest.total_tokens > 0
            else 0
        ),
        "manifest": f"{manifest.split}/manifest.json",
        "stream_sha256": manifest.stream_sha256,
    }


def reference_build(
    documents,
    tokenizer,
    output_dir,
    *,
    validation_fraction,
    seed,
    context_length,
    tokens_per_shard,
):
    """The original build, minus the staging dance that final bytes ignore."""

    materialized = [dt._validate_document(document) for document in documents]
    train_documents, validation_documents = reference_split(
        materialized, validation_fraction=validation_fraction, seed=seed
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True)

    train_manifest = write_token_shards(
        dt._encoded_token_stream(train_documents, tokenizer),
        output_path / "train",
        tokenizer,
        split="train",
        tokens_per_shard=tokens_per_shard,
    )
    validation_manifest = write_token_shards(
        dt._encoded_token_stream(validation_documents, tokenizer),
        output_path / "validation",
        tokenizer,
        split="validation",
        tokens_per_shard=tokens_per_shard,
    )

    manifest_payload = {
        "format": dt.DATASET_FORMAT,
        "format_version": dt.DATASET_FORMAT_VERSION,
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "state_sha256": dt._tokenizer_state_sha256(tokenizer),
        },
        "split_policy": {
            "method": "seeded_sha256_group_prefix_by_normalized_utf8_bytes",
            "validation_fraction": float(validation_fraction),
            "seed": seed,
            "split_before_tokenization": True,
            "group_safe": True,
        },
        "training_geometry": {
            "context_length": context_length,
            "window_tokens": context_length + 1,
            "window_stride": context_length,
            "incomplete_final_tail_padded": False,
        },
        "storage": {
            "dtype": "uint16",
            "byte_order": "little",
            "tokens_per_shard": tokens_per_shard,
        },
        "splits": {
            "train": reference_split_summary(
                train_documents, train_manifest, context_length=context_length
            ),
            "validation": reference_split_summary(
                validation_documents,
                validation_manifest,
                context_length=context_length,
            ),
        },
    }

    (output_path / DATASET_MANIFEST_FILENAME).write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return train_documents, validation_documents


# --------------------------------------------------------------------------
# fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    corpus = [
        "The robot turns left at corners.",
        "HTTP != http; MHz != mHz.",
        "R = 4.7 kΩ ± 5%; C = 10 µF.",
        "Software systems require tests and deterministic data pipelines.",
    ]
    return Tokenizer.train(corpus, vocab_size=512, min_pair_frequency=1)


def sample_documents(count: int = 40) -> list[TrainingDocument]:
    return [
        TrainingDocument(
            document_id=f"doc-{i:03d}",
            text=(f"Document {i}. " + "engineering software science " * (i % 5 + 1)),
            split_group=f"group-{i % 11:02d}",
        )
        for i in range(count)
    ]


def digest_tree(root: Path) -> dict[str, str]:
    """SHA-256 of every file under ``root``, keyed by relative path."""

    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            key = path.relative_to(root).as_posix()
            digests[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def ids(documents) -> list[str]:
    return [document.document_id for document in documents]


# --------------------------------------------------------------------------
# the split itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("validation_fraction", [0.0, 0.05, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("seed", [0, 42, 2026])
def test_streaming_split_matches_reference(validation_fraction, seed):
    documents = sample_documents()

    expected_train, expected_validation = reference_split(
        documents, validation_fraction=validation_fraction, seed=seed
    )
    actual_train, actual_validation = split_training_documents(
        documents, validation_fraction=validation_fraction, seed=seed
    )

    assert ids(actual_train) == ids(expected_train)
    assert ids(actual_validation) == ids(expected_validation)


def test_plan_selects_the_same_validation_groups_as_the_reference():
    documents = sample_documents()

    _, expected_validation = reference_split(
        documents, validation_fraction=0.2, seed=7
    )
    expected_groups = {
        document.effective_split_group for document in expected_validation
    }

    plan = plan_training_split(documents, validation_fraction=0.2, seed=7)

    assert set(plan.validation_groups) == expected_groups
    assert plan.document_count == len(documents)
    assert plan.train_group_count + plan.validation_group_count == plan.group_count


def test_random_corpora_split_identically():
    rng = random.Random(20260815)

    for _ in range(60):
        count = rng.randint(2, 60)
        group_count = rng.randint(2, 12)
        documents = [
            TrainingDocument(
                document_id=f"d{i}",
                text="token " * rng.randint(1, 40),
                split_group=(
                    f"g{rng.randrange(group_count)}" if rng.random() < 0.8 else None
                ),
            )
            for i in range(count)
        ]
        fraction = rng.choice([0.0, 0.1, 0.25, 0.4])
        seed = rng.randint(0, 10_000)

        try:
            expected = reference_split(
                documents, validation_fraction=fraction, seed=seed
            )
        except (ValueError, RuntimeError) as exc:
            with pytest.raises(type(exc)):
                split_training_documents(
                    documents, validation_fraction=fraction, seed=seed
                )
            continue

        actual = split_training_documents(
            documents, validation_fraction=fraction, seed=seed
        )
        assert ids(actual[0]) == ids(expected[0])
        assert ids(actual[1]) == ids(expected[1])


# --------------------------------------------------------------------------
# the built dataset, byte for byte
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "validation_fraction, tokens_per_shard, context_length",
    [
        (0.2, 64, 16),
        (0.05, 1_000_000, 2048),
        (0.5, 7, 4),
        (0.0, 32, 8),
    ],
)
def test_built_dataset_is_byte_identical_to_the_reference(
    tmp_path: Path,
    tokenizer: Tokenizer,
    validation_fraction: float,
    tokens_per_shard: int,
    context_length: int,
):
    documents = sample_documents()
    expected_dir = tmp_path / "reference"
    actual_dir = tmp_path / "streaming"

    reference_train, reference_validation = reference_build(
        documents,
        tokenizer,
        expected_dir,
        validation_fraction=validation_fraction,
        seed=2026,
        context_length=context_length,
        tokens_per_shard=tokens_per_shard,
    )
    result = build_pretraining_dataset(
        documents,
        tokenizer,
        actual_dir,
        validation_fraction=validation_fraction,
        seed=2026,
        context_length=context_length,
        tokens_per_shard=tokens_per_shard,
    )

    expected_digests = digest_tree(expected_dir)
    actual_digests = digest_tree(actual_dir)

    # Named explicitly so a failure says which file diverged, not just "a dict
    # differs"; the .bin digests are the claim that matters most.
    assert sorted(actual_digests) == sorted(expected_digests)
    for name in sorted(expected_digests):
        assert actual_digests[name] == expected_digests[name], name

    assert result.train_document_count == len(reference_train)
    assert result.validation_document_count == len(reference_validation)


def test_document_counts_and_token_counts_match_the_reference(
    tmp_path: Path, tokenizer: Tokenizer
):
    documents = sample_documents(75)
    expected_dir = tmp_path / "reference"
    actual_dir = tmp_path / "streaming"

    reference_build(
        documents,
        tokenizer,
        expected_dir,
        validation_fraction=0.15,
        seed=99,
        context_length=32,
        tokens_per_shard=128,
    )
    build_pretraining_dataset(
        documents,
        tokenizer,
        actual_dir,
        validation_fraction=0.15,
        seed=99,
        context_length=32,
        tokens_per_shard=128,
    )

    expected = json.loads(
        (expected_dir / DATASET_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    actual = json.loads(
        (actual_dir / DATASET_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )

    assert actual == expected
    for split in ("train", "validation"):
        for key in (
            "document_count",
            "split_group_count",
            "normalized_characters",
            "normalized_utf8_bytes",
            "document_identity_sha256",
            "total_tokens",
            "shard_count",
            "complete_examples",
            "stream_sha256",
        ):
            assert actual["splits"][split][key] == expected["splits"][split][key], (
                f"{split}.{key}"
            )


def test_grouped_documents_stay_together_and_match_the_reference(
    tmp_path: Path, tokenizer: Tokenizer
):
    documents = [
        TrainingDocument(f"chunk-{i}", f"chunk {i} of a page. " * 4, split_group=f"page-{i // 3}")
        for i in range(30)
    ]

    reference_build(
        documents,
        tokenizer,
        tmp_path / "reference",
        validation_fraction=0.3,
        seed=5,
        context_length=16,
        tokens_per_shard=64,
    )
    build_pretraining_dataset(
        documents,
        tokenizer,
        tmp_path / "streaming",
        validation_fraction=0.3,
        seed=5,
        context_length=16,
        tokens_per_shard=64,
    )

    assert digest_tree(tmp_path / "streaming") == digest_tree(tmp_path / "reference")


# --------------------------------------------------------------------------
# streaming-specific behaviour the reference could not have
# --------------------------------------------------------------------------


def test_a_callable_source_produces_the_same_bytes_as_a_sequence(
    tmp_path: Path, tokenizer: Tokenizer
):
    documents = sample_documents(24)

    build_pretraining_dataset(
        documents,
        tokenizer,
        tmp_path / "from-sequence",
        validation_fraction=0.25,
        seed=11,
        context_length=16,
        tokens_per_shard=64,
    )
    build_pretraining_dataset(
        lambda: iter(documents),
        tokenizer,
        tmp_path / "from-callable",
        validation_fraction=0.25,
        seed=11,
        context_length=16,
        tokens_per_shard=64,
    )

    assert digest_tree(tmp_path / "from-callable") == digest_tree(
        tmp_path / "from-sequence"
    )


def test_one_shot_iterator_is_rejected(tmp_path: Path, tokenizer: Tokenizer):
    documents = iter(sample_documents(5))

    with pytest.raises(TypeError, match="one-shot iterator"):
        build_pretraining_dataset(
            documents,
            tokenizer,
            tmp_path / "dataset",
            validation_fraction=0.2,
        )


def test_the_corpus_is_never_held_in_memory(tmp_path: Path, tokenizer: Tokenizer):
    """The source must be re-entered per pass, never buffered on first use."""

    starts = 0

    def source():
        nonlocal starts
        starts += 1
        return iter(sample_documents(12))

    build_pretraining_dataset(
        source,
        tokenizer,
        tmp_path / "dataset",
        validation_fraction=0.25,
        seed=3,
        context_length=16,
        tokens_per_shard=64,
    )

    # One planning pass plus one pass per split.  Anything less means something
    # kept a copy.
    assert starts == 3


def test_a_source_that_changes_between_passes_is_rejected(
    tmp_path: Path, tokenizer: Tokenizer
):
    calls = 0

    def unstable_source():
        nonlocal calls
        calls += 1
        documents = sample_documents(20)
        if calls > 1:
            documents = documents[:-1]
        return iter(documents)

    with pytest.raises(RuntimeError, match="not stable across passes"):
        build_pretraining_dataset(
            unstable_source,
            tokenizer,
            tmp_path / "dataset",
            validation_fraction=0.25,
            seed=3,
            context_length=16,
            tokens_per_shard=64,
        )


def test_a_source_that_swaps_documents_between_passes_is_rejected(
    tmp_path: Path, tokenizer: Tokenizer
):
    """Same document count, different identities: counters alone would miss it."""

    calls = 0

    def swapping_source():
        nonlocal calls
        calls += 1
        documents = sample_documents(20)
        if calls > 1:
            documents[7] = TrainingDocument(
                document_id="doc-999",
                text=documents[7].text,
                split_group=documents[7].split_group,
            )
        return iter(documents)

    with pytest.raises(RuntimeError, match="not stable across passes"):
        build_pretraining_dataset(
            swapping_source,
            tokenizer,
            tmp_path / "dataset",
            validation_fraction=0.25,
            seed=3,
            context_length=16,
            tokens_per_shard=64,
        )
