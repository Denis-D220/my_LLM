"""Tests for dense pretraining-document packing and causal window alignment."""

from __future__ import annotations

import pytest

from llm.data.document import encode_pretraining_document
from src.llm.data.packing import (
    DEFAULT_CONTEXT_LENGTH,
    build_causal_windows,
    concatenate_pretraining_documents,
    pack_pretraining_documents,
    split_causal_window,
)
from llm.tokenizer.tokenizer import DEFAULT_SPECIAL_TOKENS, Tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """Use the real project Tokenizer API rather than a mock."""

    training_texts = [
        "The transformer predicts the next token from previous tokens.\n",
        "HTTP != http; MHz != mHz; R = 4.7 kΩ ± 5%.\n",
        "ΔV = I × R; C = 10 µF; T = 25 °C.\n",
        "def f(x):\n    return x * 2\n",
        "Document boundaries are explicit and deterministic.\n",
    ]
    return Tokenizer.train(
        training_texts,
        vocab_size=512,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        min_pair_frequency=1,
    )


def _encoded_documents(tokenizer: Tokenizer) -> list[list[int]]:
    return [
        encode_pretraining_document("alpha", tokenizer),
        encode_pretraining_document("beta", tokenizer),
        encode_pretraining_document("gamma", tokenizer),
    ]


def test_concatenation_is_exact(tokenizer: Tokenizer) -> None:
    documents = _encoded_documents(tokenizer)

    stream = concatenate_pretraining_documents(documents, tokenizer)

    assert stream == documents[0] + documents[1] + documents[2]


def test_document_boundary_is_eos_followed_immediately_by_bos(
    tokenizer: Tokenizer,
) -> None:
    first, second, third = _encoded_documents(tokenizer)
    stream = concatenate_pretraining_documents([first, second, third], tokenizer)

    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")

    first_boundary = len(first)
    second_boundary = len(first) + len(second)

    assert stream[first_boundary - 1 : first_boundary + 1] == [eos_id, bos_id]
    assert stream[second_boundary - 1 : second_boundary + 1] == [eos_id, bos_id]


def test_concatenation_inserts_no_padding_or_separator_tokens(
    tokenizer: Tokenizer,
) -> None:
    documents = _encoded_documents(tokenizer)
    stream = concatenate_pretraining_documents(documents, tokenizer)

    assert len(stream) == sum(len(document) for document in documents)


def test_empty_document_iterable_produces_empty_stream(tokenizer: Tokenizer) -> None:
    assert concatenate_pretraining_documents([], tokenizer) == []


def test_document_generator_is_supported_and_deterministic(tokenizer: Tokenizer) -> None:
    documents = _encoded_documents(tokenizer)

    first = concatenate_pretraining_documents((doc for doc in documents), tokenizer)
    second = concatenate_pretraining_documents((doc for doc in documents), tokenizer)

    assert first == second


def test_concatenation_rejects_document_missing_bos(tokenizer: Tokenizer) -> None:
    document = encode_pretraining_document("broken", tokenizer)[1:]

    with pytest.raises(ValueError, match="start with <\\|bos\\|>"):
        concatenate_pretraining_documents([document], tokenizer)


def test_concatenation_rejects_document_missing_eos(tokenizer: Tokenizer) -> None:
    document = encode_pretraining_document("broken", tokenizer)[:-1]

    with pytest.raises(ValueError, match="end with <\\|eos\\|>"):
        concatenate_pretraining_documents([document], tokenizer)


def test_concatenation_rejects_internal_reserved_special_id(
    tokenizer: Tokenizer,
) -> None:
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    assistant_id = tokenizer.token_to_id("<|assistant|>")

    with pytest.raises(ValueError, match="reserved special-token ids"):
        concatenate_pretraining_documents(
            [[bos_id, assistant_id, eos_id]],
            tokenizer,
        )


def test_concatenation_rejects_out_of_range_document_token(
    tokenizer: Tokenizer,
) -> None:
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")

    with pytest.raises(ValueError, match="outside tokenizer vocabulary"):
        concatenate_pretraining_documents(
            [[bos_id, tokenizer.vocab_size, eos_id]],
            tokenizer,
        )


def test_context_four_builds_expected_windows_and_tail(tokenizer: Tokenizer) -> None:
    stream = list(range(10))

    windows, tail = build_causal_windows(
        stream,
        tokenizer,
        context_length=4,
    )

    assert windows == [
        [0, 1, 2, 3, 4],
        [4, 5, 6, 7, 8],
    ]
    assert tail == [8, 9]


def test_adjacent_windows_overlap_by_exactly_one_boundary_token(
    tokenizer: Tokenizer,
) -> None:
    stream = list(range(14))
    windows, _ = build_causal_windows(stream, tokenizer, context_length=4)

    assert windows[0][-1] == windows[1][0]
    assert windows[1][-1] == windows[2][0]
    assert windows[0] == stream[0:5]
    assert windows[1] == stream[4:9]
    assert windows[2] == stream[8:13]


def test_window_starts_advance_by_context_length(tokenizer: Tokenizer) -> None:
    stream = list(range(18))
    windows, _ = build_causal_windows(stream, tokenizer, context_length=4)

    assert [window[0] for window in windows] == [0, 4, 8, 12]


def test_exact_full_windows_leave_single_carry_token(tokenizer: Tokenizer) -> None:
    # 9 tokens with S=4 form windows [0:5] and [4:9].  Token 8 is retained as
    # the next-window carry token even though there is no remaining prediction.
    stream = list(range(9))

    windows, tail = build_causal_windows(stream, tokenizer, context_length=4)

    assert len(windows) == 2
    assert tail == [8]


def test_one_full_window_leaves_its_last_token_as_carry(tokenizer: Tokenizer) -> None:
    stream = list(range(5))

    windows, tail = build_causal_windows(stream, tokenizer, context_length=4)

    assert windows == [[0, 1, 2, 3, 4]]
    assert tail == [4]


def test_incomplete_stream_produces_no_window_and_preserves_tail(
    tokenizer: Tokenizer,
) -> None:
    stream = [10, 11, 12, 13]

    windows, tail = build_causal_windows(stream, tokenizer, context_length=4)

    assert windows == []
    assert tail == stream


def test_empty_stream_produces_no_windows_and_empty_tail(tokenizer: Tokenizer) -> None:
    windows, tail = build_causal_windows([], tokenizer, context_length=4)

    assert windows == []
    assert tail == []


def test_all_full_windows_have_context_plus_one_tokens(tokenizer: Tokenizer) -> None:
    windows, _ = build_causal_windows(
        list(range(30)),
        tokenizer,
        context_length=8,
    )

    assert windows
    assert all(len(window) == 9 for window in windows)


def test_split_causal_window_produces_correct_input_and_target_alignment() -> None:
    window = [10, 11, 12, 13, 14]

    input_ids, target_ids = split_causal_window(window, context_length=4)

    assert input_ids == [10, 11, 12, 13]
    assert target_ids == [11, 12, 13, 14]
    assert input_ids[1:] == target_ids[:-1]


def test_each_full_window_yields_exactly_context_length_predictions(
    tokenizer: Tokenizer,
) -> None:
    windows, _ = build_causal_windows(
        list(range(18)),
        tokenizer,
        context_length=4,
    )

    for window in windows:
        input_ids, target_ids = split_causal_window(window, context_length=4)
        assert len(input_ids) == 4
        assert len(target_ids) == 4


def test_full_window_predictions_cover_stream_without_duplicate_targets(
    tokenizer: Tokenizer,
) -> None:
    stream = list(range(13))
    windows, tail = build_causal_windows(stream, tokenizer, context_length=4)

    # Three full windows predict token positions 1..12 exactly once.
    predicted_tokens: list[int] = []
    for window in windows:
        _, targets = split_causal_window(window, context_length=4)
        predicted_tokens.extend(targets)

    assert predicted_tokens == stream[1:13]
    assert tail == [12]


def test_build_causal_windows_does_not_modify_input_stream(tokenizer: Tokenizer) -> None:
    stream = list(range(10))
    original = list(stream)

    build_causal_windows(stream, tokenizer, context_length=4)

    assert stream == original


def test_build_causal_windows_is_deterministic(tokenizer: Tokenizer) -> None:
    stream = list(range(25))

    first = build_causal_windows(stream, tokenizer, context_length=6)
    second = build_causal_windows(stream, tokenizer, context_length=6)

    assert first == second


def test_token_stream_generator_is_supported(tokenizer: Tokenizer) -> None:
    windows, tail = build_causal_windows(
        (token_id for token_id in range(10)),
        tokenizer,
        context_length=4,
    )

    assert windows[0] == [0, 1, 2, 3, 4]
    assert tail == [8, 9]


def test_build_causal_windows_rejects_non_integer_token(tokenizer: Tokenizer) -> None:
    with pytest.raises(TypeError, match="token ids must be integers"):
        build_causal_windows([1, 2, "3", 4, 5], tokenizer, context_length=4)


def test_build_causal_windows_rejects_bool_token(tokenizer: Tokenizer) -> None:
    with pytest.raises(TypeError, match="token ids must be integers"):
        build_causal_windows([1, 2, True, 4, 5], tokenizer, context_length=4)


def test_build_causal_windows_rejects_out_of_range_token(tokenizer: Tokenizer) -> None:
    with pytest.raises(ValueError, match="outside tokenizer vocabulary"):
        build_causal_windows(
            [1, 2, tokenizer.vocab_size, 4, 5],
            tokenizer,
            context_length=4,
        )


@pytest.mark.parametrize("context_length", [0, -1, -2048])
def test_non_positive_context_length_is_rejected(
    tokenizer: Tokenizer,
    context_length: int,
) -> None:
    with pytest.raises(ValueError, match="context_length must be > 0"):
        build_causal_windows([1, 2], tokenizer, context_length=context_length)


@pytest.mark.parametrize("context_length", [True, 4.0, "4"])
def test_non_integer_context_length_is_rejected(
    tokenizer: Tokenizer,
    context_length,
) -> None:
    with pytest.raises(TypeError, match="context_length must be an integer"):
        build_causal_windows([1, 2], tokenizer, context_length=context_length)


def test_split_rejects_wrong_window_length() -> None:
    with pytest.raises(ValueError, match="exactly 5 tokens"):
        split_causal_window([1, 2, 3, 4], context_length=4)


def test_split_rejects_non_integer_token() -> None:
    with pytest.raises(TypeError, match="token ids must be integers"):
        split_causal_window([1, 2, "3", 4, 5], context_length=4)


def test_pack_pretraining_documents_matches_explicit_two_step_pipeline(
    tokenizer: Tokenizer,
) -> None:
    documents = _encoded_documents(tokenizer)

    stream = concatenate_pretraining_documents(documents, tokenizer)
    expected = build_causal_windows(stream, tokenizer, context_length=4)
    actual = pack_pretraining_documents(documents, tokenizer, context_length=4)

    assert actual == expected


def test_pack_pretraining_documents_preserves_cross_document_learning_boundary(
    tokenizer: Tokenizer,
) -> None:
    first = encode_pretraining_document("A", tokenizer)
    second = encode_pretraining_document("B", tokenizer)
    stream = concatenate_pretraining_documents([first, second], tokenizer)

    eos_id = tokenizer.token_to_id("<|eos|>")
    bos_id = tokenizer.token_to_id("<|bos|>")
    boundary = len(first) - 1

    assert stream[boundary] == eos_id
    assert stream[boundary + 1] == bos_id

    # Packing never masks this transition: causal LM pretraining can learn that
    # EOS is followed by BOS when one independent document ends and the next
    # begins in the dense stream.
    windows, _ = pack_pretraining_documents(
        [first, second],
        tokenizer,
        context_length=2,
    )
    flattened_pairs = [
        pair
        for window in windows
        for pair in zip(window[:-1], window[1:])
    ]
    assert (eos_id, bos_id) in flattened_pairs


def test_default_context_length_is_2048() -> None:
    assert DEFAULT_CONTEXT_LENGTH == 2048


def test_default_context_builds_2049_token_windows(tokenizer: Tokenizer) -> None:
    # Use one known-valid content id repeatedly; the test is about geometry, not
    # lexical content.
    stream = [65] * (2 * DEFAULT_CONTEXT_LENGTH + 1)

    windows, tail = build_causal_windows(stream, tokenizer)

    assert len(windows) == 2
    assert all(len(window) == DEFAULT_CONTEXT_LENGTH + 1 for window in windows)
    assert len(tail) == 1

    input_ids, target_ids = split_causal_window(windows[0])
    assert len(input_ids) == DEFAULT_CONTEXT_LENGTH
    assert len(target_ids) == DEFAULT_CONTEXT_LENGTH
    assert input_ids[1:] == target_ids[:-1]