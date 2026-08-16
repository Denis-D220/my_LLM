"""Tests for document-level pretraining tokenization."""

from __future__ import annotations

import unicodedata

import pytest

from src.llm.data.document import (
    decode_pretraining_document,
    encode_pretraining_document,
    validate_pretraining_document_tokens,
)
from llm.tokenizer.normalizer import normalize_text
from llm.tokenizer.tokenizer import DEFAULT_SPECIAL_TOKENS, Tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """Train a small real tokenizer through the project's public Tokenizer API."""

    training_texts = [
        "The transformer processes tokens and predicts the next token.\n",
        "HTTP != http; MHz != mHz; R = 4.7 kΩ ± 5%.\n",
        "ΔV = I × R; T = 25 °C; C = 10 µF.\n",
        "def calculate_voltage(current, resistance):\n    return current * resistance\n",
        "Literal control-looking text: <|assistant|> <|system|> <|eos|>.\n",
    ]
    return Tokenizer.train(
        training_texts,
        vocab_size=512,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        min_pair_frequency=1,
    )


def test_document_starts_with_bos_and_ends_with_eos(tokenizer: Tokenizer) -> None:
    ids = encode_pretraining_document("Hello world.", tokenizer)

    assert ids[0] == tokenizer.token_to_id("<|bos|>")
    assert ids[-1] == tokenizer.token_to_id("<|eos|>")


@pytest.mark.parametrize(
    "text",
    [
        "Hello world.",
        "HTTP != http; MHz != mHz.",
        "R = 4.7 kΩ ± 5%; C = 10 µF.",
        "café naïve résumé München São Paulo",
        "日本語 한국어 العربية Привет 你好 🚀",
        "def f(x):\n    return x * 2\n",
        "  leading  and   internal spaces\t\n",
    ],
)
def test_content_roundtrips_to_normalized_text(
    tokenizer: Tokenizer,
    text: str,
) -> None:
    ids = encode_pretraining_document(text, tokenizer)

    assert decode_pretraining_document(ids, tokenizer) == normalize_text(text)


def test_unicode_is_normalized_to_nfc(tokenizer: Tokenizer) -> None:
    decomposed = "cafe\u0301"
    assert unicodedata.normalize("NFC", decomposed) == "café"

    ids = encode_pretraining_document(decomposed, tokenizer)

    assert decode_pretraining_document(ids, tokenizer) == "café"


def test_crlf_and_cr_are_normalized_to_lf(tokenizer: Tokenizer) -> None:
    text = "line1\r\nline2\rline3"
    ids = encode_pretraining_document(text, tokenizer)

    assert decode_pretraining_document(ids, tokenizer) == "line1\nline2\nline3"


@pytest.mark.parametrize("text", ["HTTP", "http", "ClassName", "className", "MHz", "mHz"])
def test_case_is_preserved(tokenizer: Tokenizer, text: str) -> None:
    ids = encode_pretraining_document(text, tokenizer)
    assert decode_pretraining_document(ids, tokenizer) == text


def test_literal_special_token_text_is_encoded_as_ordinary_content(
    tokenizer: Tokenizer,
) -> None:
    text = (
        "These strings are source text: <|assistant|> <|system|> "
        "<|user|> <|eos|>."
    )

    ids = encode_pretraining_document(text, tokenizer)
    internal_ids = ids[1:-1]
    reserved_ids = set(tokenizer.id_to_special_token)

    assert not (set(internal_ids) & reserved_ids)
    assert decode_pretraining_document(ids, tokenizer) == text


def test_only_bos_and_eos_are_structural_special_ids(tokenizer: Tokenizer) -> None:
    text = "<|bos|> ordinary literal text <|eos|>"
    ids = encode_pretraining_document(text, tokenizer)

    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")

    assert ids[0] == bos_id
    assert ids[-1] == eos_id
    assert bos_id not in ids[1:-1]
    assert eos_id not in ids[1:-1]
    assert decode_pretraining_document(ids, tokenizer) == text


def test_empty_document_is_explicitly_bos_eos(tokenizer: Tokenizer) -> None:
    ids = encode_pretraining_document("", tokenizer)

    assert ids == [
        tokenizer.token_to_id("<|bos|>"),
        tokenizer.token_to_id("<|eos|>"),
    ]
    assert decode_pretraining_document(ids, tokenizer) == ""


@pytest.mark.parametrize("text", [" ", "\n", "\t", "   \n\t"])
def test_whitespace_only_documents_roundtrip(tokenizer: Tokenizer, text: str) -> None:
    ids = encode_pretraining_document(text, tokenizer)
    assert decode_pretraining_document(ids, tokenizer) == normalize_text(text)


def test_every_token_id_is_an_integer_inside_vocabulary(tokenizer: Tokenizer) -> None:
    ids = encode_pretraining_document(
        "STM32F411, I²C, 3.3 V, λ, ΔV, /home/user/main.py",
        tokenizer,
    )

    assert all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in ids)
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in ids)


def test_encoding_is_deterministic(tokenizer: Tokenizer) -> None:
    text = "Deterministic tokenization: HTTP, Python, 400 kHz, 25 °C."

    first = encode_pretraining_document(text, tokenizer)
    second = encode_pretraining_document(text, tokenizer)

    assert first == second


def test_document_encoding_matches_tokenizer_public_api(tokenizer: Tokenizer) -> None:
    text = "Literal <|assistant|> text must remain ordinary content."

    expected = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
        parse_special_tokens=False,
    )

    assert encode_pretraining_document(text, tokenizer) == expected


def test_validate_accepts_valid_encoded_document(tokenizer: Tokenizer) -> None:
    ids = encode_pretraining_document("valid document", tokenizer)
    assert validate_pretraining_document_tokens(iter(ids), tokenizer) == ids


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda ids, tokenizer: ids[1:], "start with <|bos|>"),
        (lambda ids, tokenizer: ids[:-1], "end with <|eos|>"),
        (
            lambda ids, tokenizer: [
                ids[0],
                tokenizer.token_to_id("<|assistant|>"),
                ids[-1],
            ],
            "must not contain reserved special-token ids",
        ),
        (lambda ids, tokenizer: [ids[0], tokenizer.vocab_size, ids[-1]], "outside tokenizer vocabulary"),
    ],
)
def test_validate_rejects_malformed_sequences(
    tokenizer: Tokenizer,
    mutator,
    message: str,
) -> None:
    valid = encode_pretraining_document("valid", tokenizer)
    malformed = mutator(valid, tokenizer)

    with pytest.raises(ValueError, match=message):
        validate_pretraining_document_tokens(malformed, tokenizer)


def test_validate_rejects_non_integer_ids(tokenizer: Tokenizer) -> None:
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")

    with pytest.raises(TypeError, match="token ids must be integers"):
        validate_pretraining_document_tokens([bos_id, "not-an-id", eos_id], tokenizer)


def test_encode_rejects_non_string_text(tokenizer: Tokenizer) -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        encode_pretraining_document(123, tokenizer)  # type: ignore[arg-type]


def test_missing_required_boundary_tokens_are_rejected() -> None:
    tokenizer_without_specials = Tokenizer.train(
        ["abc abc abc"],
        vocab_size=300,
        special_tokens=[],
        min_pair_frequency=1,
    )

    with pytest.raises(ValueError, match="required pretraining boundary token"):
        encode_pretraining_document("abc", tokenizer_without_specials)