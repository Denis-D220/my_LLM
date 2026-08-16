"""Regression tests for the optimized byte-level BPE kernel.

These tests intentionally contain a tiny, slow reference trainer. The goal is
to prove that future performance work does not change BPE semantics.

Place this file at:

    tests/tokenizer/test_bpe_optimized_regression.py

Run:

    python -m pytest tests/tokenizer/test_bpe_optimized_regression.py -q
"""

from __future__ import annotations

from collections import Counter
import random

from llm.tokenizer.bpe import BPETrainer, BYTE_VOCAB_SIZE


def reference_merge_one(sequence, pair, new_token_id):
    left, right = pair
    output = []
    i = 0

    while i < len(sequence):
        if (
            i + 1 < len(sequence)
            and sequence[i] == left
            and sequence[i + 1] == right
        ):
            output.append(new_token_id)
            i += 2
        else:
            output.append(sequence[i])
            i += 1

    return output


def reference_train(texts, vocab_size, min_pair_frequency):
    """Original clear-but-slow BPE training semantics."""

    sequences = []
    for item in texts:
        raw = item.encode("utf-8") if isinstance(item, str) else bytes(item)
        if raw:
            sequences.append(list(raw))

    merges = {}
    next_token_id = BYTE_VOCAB_SIZE

    while next_token_id < vocab_size:
        counts = Counter()

        for sequence in sequences:
            counts.update(zip(sequence, sequence[1:]))

        if not counts:
            break

        max_count = max(counts.values())
        if max_count < min_pair_frequency:
            break

        best_pair = min(
            pair
            for pair, count in counts.items()
            if count == max_count
        )

        sequences = [
            reference_merge_one(sequence, best_pair, next_token_id)
            for sequence in sequences
        ]

        merges[best_pair] = next_token_id
        next_token_id += 1

    return merges


def test_overlapping_self_pair_matches_reference():
    corpus = [b"aaaaaa", b"aaa", b"aaaaaaaaa"]

    expected = reference_train(
        corpus,
        vocab_size=280,
        min_pair_frequency=1,
    )
    actual = BPETrainer(
        vocab_size=280,
        min_pair_frequency=1,
    ).train(corpus)

    assert actual.merges == expected


def test_multiple_documents_do_not_merge_across_boundaries():
    corpus = [b"abc", b"def", b"abc"]

    expected = reference_train(
        corpus,
        vocab_size=280,
        min_pair_frequency=1,
    )
    actual = BPETrainer(
        vocab_size=280,
        min_pair_frequency=1,
    ).train(corpus)

    assert actual.merges == expected


def test_deterministic_tie_break_matches_reference():
    corpus = [
        b"ab cd ab cd",
        b"cd ab cd ab",
        b"xy xy yz yz",
    ]

    expected = reference_train(
        corpus,
        vocab_size=310,
        min_pair_frequency=1,
    )
    actual = BPETrainer(
        vocab_size=310,
        min_pair_frequency=1,
    ).train(corpus)

    assert actual.merges == expected


def test_random_small_corpora_match_reference_exactly():
    rng = random.Random(20260810)
    alphabet = b"abcde 0123_-\n"

    for _ in range(100):
        corpus = []

        for _ in range(rng.randint(1, 5)):
            length = rng.randint(0, 80)
            corpus.append(
                bytes(rng.choice(alphabet) for _ in range(length))
            )

        vocab_size = rng.randint(256, 340)
        min_pair_frequency = rng.randint(1, 4)

        expected = reference_train(
            corpus,
            vocab_size=vocab_size,
            min_pair_frequency=min_pair_frequency,
        )
        actual = BPETrainer(
            vocab_size=vocab_size,
            min_pair_frequency=min_pair_frequency,
        ).train(corpus)

        assert actual.merges == expected


def test_encode_decode_and_serialization_are_stable():
    corpus = [
        "Engineering systems use deterministic interfaces.",
        "R = 4.7 kΩ ± 5%; VCC = 3.3 V.",
        "for i in range(10): print(i)",
        "A AND B OR NOT C",
    ]

    trainer = BPETrainer(
        vocab_size=400,
        min_pair_frequency=1,
    ).train(corpus)

    payload = trainer.to_dict()
    loaded = BPETrainer.from_dict(payload)

    probes = [
        "",
        "Hello, world!",
        "STM32F411 communicates over I²C.",
        "x = (a + b) * 3.14159",
        "café λ ∇ Ω",
    ]

    for text in probes:
        raw = text.encode("utf-8")
        ids_a = trainer.encode_bytes(raw)
        ids_b = loaded.encode_bytes(raw)

        assert ids_a == ids_b
        assert trainer.decode_ids(ids_a) == raw
        assert loaded.decode_ids(ids_b) == raw