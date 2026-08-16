"""
Round-trip tests for the LLM tokenizer.

The most important invariant of Tokenizer v0.1 is:

    decode(encode(normalize(text))) == normalize(text)

The tokenizer must preserve all semantically relevant information, including:

- Upper/lower case
- Technical units
- Unicode symbols
- Mathematical expressions
- Source code
- Whitespace
- Newlines
- URLs
- File paths
- Numbers
- Scientific notation
- Structured data

The tokenizer uses:

- UTF-8
- Unicode NFC normalization
- Case preservation
- Byte-level BPE
- No lowercasing
- No case folding

These tests intentionally use a small vocabulary so that the test suite
can run quickly. The production tokenizer will use a 24,000-token vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm.tokenizer.normalizer import normalize_text
from llm.tokenizer.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

TEST_VOCAB_SIZE = 512

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_turn|>",
    "<|tool|>",
    "<|tool_result|>",
]


# ---------------------------------------------------------------------------
# Small representative tokenizer-training corpus
# ---------------------------------------------------------------------------

TRAINING_TEXTS = [
    # General English
    """
    A transformer is a neural network architecture based on attention
    mechanisms. Large language models learn statistical relationships
    between tokens in sequences of text.
    """,

    # Electronics
    """
    The voltage across a resistor is proportional to the current flowing
    through it. Ohm's law is expressed as V = I × R.
    """,

    # Technical units
    """
    The CPU operates at 3.2 GHz.
    The resistance is 4.7 kΩ ± 5%.
    The capacitor has a value of 10 µF.
    The temperature is 25 °C.
    The signal frequency is 100 MHz.
    """,

    # Case-sensitive technical terminology
    """
    HTTP and HTTPS are application-layer protocols.
    GET and POST are HTTP methods.
    RAM is different from ram.
    MHz is different from mHz.
    Python is different from python.
    """,

    # Python
    """
    def calculate_voltage(current: float, resistance: float) -> float:
        return current * resistance

    class HTTPClient:
        MAX_RETRIES = 5
    """,

    # C / C++
    """
    uint32_t HAL_GetTick(void);

    std::vector<std::string> values;
    GPIO_PIN_13 = 0x2000;
    """,

    # Mathematics
    """
    ΔV = I × R
    E = mc²
    A = πr²
    x ≤ y
    y ≥ z
    √2 ≈ 1.41421356
    """,

    # URLs and networking
    """
    https://example.com/api/v1/devices?id=42
    192.168.1.100
    2001:db8::1
    """,

    # Paths
    """
    /home/user/project/src/main.py
    C:\\Users\\Daniel\\project\\main.py
    """,

    # Structured data
    """
    {
        "temperature": 23.5,
        "enabled": true,
        "device": "STM32F411"
    }
    """,

    # Scientific notation
    """
    1.25e-6
    6.022e23
    -3.1415926535
    0x7F
    0b10101010
    """,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tokenizer() -> Tokenizer:
    """
    Train a small byte-level BPE tokenizer for unit testing.

    The production tokenizer will use vocab_size=24000.

    A much smaller vocabulary is deliberately used here because these tests
    verify correctness, not production compression quality.
    """

    tokenizer = Tokenizer.train(
        texts=TRAINING_TEXTS,
        vocab_size=TEST_VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    return tokenizer


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def assert_roundtrip(tokenizer: Tokenizer, text: str) -> None:
    """
    Verify the central tokenizer invariant.

    Raw text is first normalized according to our tokenizer specification.

    The decoded value must exactly equal the normalized value.
    """

    normalized = normalize_text(text)

    token_ids = tokenizer.encode(normalized)

    decoded = tokenizer.decode(token_ids)

    assert decoded == normalized, (
        "\nTokenizer round-trip failure\n"
        f"Original:   {text!r}\n"
        f"Normalized: {normalized!r}\n"
        f"Token IDs:  {token_ids}\n"
        f"Decoded:    {decoded!r}\n"
    )


# ---------------------------------------------------------------------------
# Basic round-trip tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello",
        "Hello world.",
        "The transformer processes a sequence of tokens.",
        "Artificial intelligence is changing software engineering.",
        "A resistor limits electrical current.",
    ],
)
def test_basic_english_roundtrip(
    tokenizer: Tokenizer,
    text: str,
) -> None:

    assert_roundtrip(tokenizer, text)


# ---------------------------------------------------------------------------
# Case preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "HTTP",
        "http",
        "HTTP != http",
        "RAM != ram",
        "Python != python",
        "GET != get",
        "POST != post",
        "ClassName",
        "className",
        "classname",
        "MAX_CURRENT",
        "max_current",
        "MHz",
        "mHz",
    ],
)
def test_case_is_preserved(
    tokenizer: Tokenizer,
    text: str,
) -> None:

    assert_roundtrip(tokenizer, text)


def test_case_sensitive_strings_encode_differently(
    tokenizer: Tokenizer,
) -> None:
    """
    Case-sensitive strings must not collapse into identical token streams.
    """

    pairs = [
        ("HTTP", "http"),
        ("RAM", "ram"),
        ("GET", "get"),
        ("Python", "python"),
        ("MHz", "mHz"),
        ("ClassName", "classname"),
    ]

    for upper_or_mixed, lower in pairs:

        encoded_a = tokenizer.encode(upper_or_mixed)
        encoded_b = tokenizer.encode(lower)

        assert encoded_a != encoded_b, (
            f"Case distinction lost:\n"
            f"{upper_or_mixed!r} -> {encoded_a}\n"
            f"{lower!r} -> {encoded_b}"
        )


# ---------------------------------------------------------------------------
# Technical units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "3.3 V",
        "250 mA",
        "4.7 kΩ",
        "10 µF",
        "100 nF",
        "2.4 GHz",
        "100 MHz",
        "100 mHz",
        "25 °C",
        "12.6 VDC",
        "50 VAC",
        "1 kHz",
        "20 dBm",
        "-40 dB",
    ],
)
def test_engineering_units_roundtrip(
    tokenizer: Tokenizer,
    text: str,
) -> None:

    assert_roundtrip(tokenizer, text)


# ---------------------------------------------------------------------------
# Unicode technical symbols
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Ω",
        "µ",
        "π",
        "Σ",
        "Δ",
        "√",
        "∞",
        "°",
        "±",
        "×",
        "÷",
        "≤",
        "≥",
        "→",
        "ΔV = I × R",
        "R = 4.7 kΩ ± 5%",
        "A = πr²",
        "H₂O",
        "E = mc²",
        "√2 ≈ 1.41421356",
        "x ≤ y ≤ z",
    ],
)
def test_unicode_technical_symbols_roundtrip(
    tokenizer: Tokenizer,
    text: str,
) -> None:

    assert_roundtrip(tokenizer, text)


# ---------------------------------------------------------------------------
# Unicode outside the target language
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "résumé",
        "naïve",
        "café",
        "São Paulo",
        "München",
        "日本語",
        "한국어",
        "العربية",
        "Привет",
        "你好",
        "🚀",
        "CPU 🚀 GPU",
    ],
)
def test_arbitrary_unicode_roundtrip(
    tokenizer: Tokenizer,
    text: str,
) -> None:
    """
    The model is English-focused, but byte-level tokenization means arbitrary
    valid Unicode must still be representable without data loss.
    """

    assert_roundtrip(tokenizer, text)


# ---------------------------------------------------------------------------
# NFC normalization
# ---------------------------------------------------------------------------

def test_unicode_nfc_normalization(
    tokenizer: Tokenizer,
) -> None:
    """
    Canonically equivalent Unicode strings should normalize to the same form.

    'é' can be represented either as:

        U+00E9

    or:

        U+0065 + U+0301

    NFC converts the decomposed representation into the canonical composed
    representation.
    """

    composed = "café"

    decomposed = "cafe\u0301"

    normalized_composed = normalize_text(composed)
    normalized_decomposed = normalize_text(decomposed)

    assert normalized_composed == normalized_decomposed

    assert_roundtrip(tokenizer, composed)
    assert_roundtrip(tokenizer, decomposed)


# ---------------------------------------------------------------------------
# Source code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        """
def calculate_voltage(current: float, resistance: float) -> float:
    return current * resistance
""",
        """
class HTTPClient:
    MAX_RETRIES = 5

    def get_response(self, url: str):
        return self._request(url)
""",
        """
if voltage > MAX_VOLTAGE:
    shutdown()
else:
    continue_operation()
""",
        """
uint32_t HAL_GetTick(void);
""",
        """
std::vector<std::string> values;
""",
        """
GPIO_WritePin(GPIOA, GPIO_PIN_13, GPIO_PIN_SET);
""",
    ],
)
def test_source_code_roundtrip(
    tokenizer: Tokenizer,
    text: str,
) -> None:

    assert_roundtrip(tokenizer, text)


# ---------------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "one two three",
        "one  two   three",
        "    indented",
        "\tindented_with_tab",
        "line1\nline2",
        "line1\n\nline3",
        "line1\n\n\nline4",
        "trailing spaces   ",
        "   leading spaces",
    ],
)
def test_whitespace_is_preserved(
    tokenizer: Tokenizer,
    text: str,
) -> None:

    assert_roundtrip(tokenizer, text)


# ---------------------------------------------------------------------------
# Line-ending normalization
# ---------------------------------------------------------------------------

def test_crlf_is_normalized_to_lf(
    tokenizer: Tokenizer,
) -> None:

    raw = "line1\r\nline2\r\nline3"

    expected = "line1\nline2\nline3"

    normalized = normalize_text(raw)

    assert normalized == expected

    token_ids = tokenizer.encode(normalized)

    decoded = tokenizer.decode(token_ids)

    assert decoded == expected


def test_cr_is_normalized_to_lf(
    tokenizer: Tokenizer,
) -> None:

    raw = "line1\rline2\rline3"

    expected = "line1\nline2\nline3"

    normalized = normalize_text(raw)

    assert normalized == expected

    assert_roundtrip(tokenizer, raw)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "0",
        "1",
        "-1",
        "123456789",
        "3.141592653589793",
        "-273.15",
        "1.25e-6",
        "6.022e23",
        "1E+10",
        "0x7F",
        "0xDEADBEEF",
        "0b10101010",
        "0o755",
        "192.168.1.100",
        "255.255.255.0",
        "2026-08-08",
        "v1.2.3",
        "STM32F411",
        "VL53L8CX",
    ],
)
def test_numbers_roundtrip(
    tokenizer: Tokenizer,
    text: str,
) -> None:

    assert_roundtrip(tokenizer, text)


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "https://example.com",
        "https://example.com/api/v1/devices",
        "https://example.com/api/v1/devices?id=42",
        "https://example.com/search?q=transformer&page=2",
        "http://localhost:8080",
        "ws://192.168.1.10:9000/socket",
    ],
)
def test_urls_roundtrip(
    tokenizer: Tokenizer,
    text: str,
) -> None:

    assert_roundtrip(tokenizer, text)


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "/home/user/project/main.py",
        "/opt/application/config/settings.yaml",
        "./src/llm/tokenizer/tokenizer.py",
        "../checkpoints/model.pt",
        r"C:\Users\Daniel\project\main.py",
        r"\\server\share\project\data.bin",
    ],
)
def test_file_paths_roundtrip(
    tokenizer: Tokenizer,
    text: str,
) -> None:

    assert_roundtrip(tokenizer, text)


# ---------------------------------------------------------------------------
# Structured data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        """
{
    "temperature": 23.5,
    "enabled": true
}
""",
        """
model:
  hidden_size: 512
  num_layers: 6
  num_heads: 8
""",
        """
<device id="42">
    <temperature>25.4</temperature>
</device>
""",
    ],
)
def test_structured_data_roundtrip(
    tokenizer: Tokenizer,
    text: str,
) -> None:

    assert_roundtrip(tokenizer, text)


# ---------------------------------------------------------------------------
# Long technical identifiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "CUDA_VISIBLE_DEVICES",
        "MAXIMUM_ALLOWED_CURRENT",
        "calculate_average_temperature",
        "TransformerDecoderLayer",
        "VL53L8CX_CONFIGURATION",
        "HAL_I2C_Master_Transmit",
        "torch.nn.functional.scaled_dot_product_attention",
    ],
)
def test_long_identifiers_roundtrip(
    tokenizer: Tokenizer,
    text: str,
) -> None:

    assert_roundtrip(tokenizer, text)


# ---------------------------------------------------------------------------
# Empty and unusual input
# ---------------------------------------------------------------------------

def test_empty_string(
    tokenizer: Tokenizer,
) -> None:

    token_ids = tokenizer.encode("")

    decoded = tokenizer.decode(token_ids)

    assert decoded == ""


def test_single_space(
    tokenizer: Tokenizer,
) -> None:

    assert_roundtrip(tokenizer, " ")


def test_only_newline(
    tokenizer: Tokenizer,
) -> None:

    assert_roundtrip(tokenizer, "\n")


def test_only_tab(
    tokenizer: Tokenizer,
) -> None:

    assert_roundtrip(tokenizer, "\t")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_encoding_is_deterministic(
    tokenizer: Tokenizer,
) -> None:

    text = "The STM32F411 communicates with the VL53L8CX over I²C."

    normalized = normalize_text(text)

    first = tokenizer.encode(normalized)
    second = tokenizer.encode(normalized)
    third = tokenizer.encode(normalized)

    assert first == second == third


def test_decoding_is_deterministic(
    tokenizer: Tokenizer,
) -> None:

    text = "The CPU operates at 3.2 GHz."

    token_ids = tokenizer.encode(normalize_text(text))

    first = tokenizer.decode(token_ids)
    second = tokenizer.decode(token_ids)
    third = tokenizer.decode(token_ids)

    assert first == second == third


# ---------------------------------------------------------------------------
# Token ID validation
# ---------------------------------------------------------------------------

def test_encode_returns_integer_token_ids(
    tokenizer: Tokenizer,
) -> None:

    text = "Technical language model"

    token_ids = tokenizer.encode(text)

    assert isinstance(token_ids, list)

    assert all(
        isinstance(token_id, int)
        for token_id in token_ids
    )


def test_token_ids_are_non_negative(
    tokenizer: Tokenizer,
) -> None:

    token_ids = tokenizer.encode(
        "The voltage is 3.3 V."
    )

    assert all(
        token_id >= 0
        for token_id in token_ids
    )


def test_token_ids_are_inside_vocabulary(
    tokenizer: Tokenizer,
) -> None:

    token_ids = tokenizer.encode(
        "Transformer attention mechanism"
    )

    assert all(
        token_id < tokenizer.vocab_size
        for token_id in token_ids
    )


# ---------------------------------------------------------------------------
# Byte-level Unicode fallback
# ---------------------------------------------------------------------------

def test_unseen_unicode_can_still_be_encoded(
    tokenizer: Tokenizer,
) -> None:
    """
    This text deliberately contains characters unlikely to occur in the
    small tokenizer-training corpus.

    Byte-level fallback must make them representable anyway.
    """

    text = "Rare symbols: ∂ ∇ λ ξ η θ ↔ ⇌ ⚡ 🛰️"

    normalized = normalize_text(text)

    token_ids = tokenizer.encode(normalized)

    assert len(token_ids) > 0

    decoded = tokenizer.decode(token_ids)

    assert decoded == normalized


# ---------------------------------------------------------------------------
# Mixed technical document
# ---------------------------------------------------------------------------

def test_complete_technical_document_roundtrip(
    tokenizer: Tokenizer,
) -> None:

    text = """
STM32F411 Sensor Interface
==========================

The STM32F411 communicates with the VL53L8CX sensor over I²C.

Configuration:

    VCC = 3.3 V
    I_MAX = 250 mA
    frequency = 400 kHz
    R_pullup = 4.7 kΩ

Python example:

def calculate_voltage(current: float, resistance: float) -> float:
    return current * resistance

The relationship is described by Ohm's law:

    V = I × R

For I = 10 mA and R = 1 kΩ:

    V = 10 V

More information:
https://example.com/api/v1/sensors?id=53
"""

    assert_roundtrip(tokenizer, text)


# ---------------------------------------------------------------------------
# Save/load persistence
# ---------------------------------------------------------------------------

def test_tokenizer_roundtrip_after_save_and_reload(
    tokenizer: Tokenizer,
    tmp_path: Path,
) -> None:
    """
    A serialized tokenizer must produce exactly the same token IDs after
    being loaded again.
    """

    tokenizer_path = tmp_path / "tokenizer.json"

    tokenizer.save(tokenizer_path)

    loaded = Tokenizer.load(tokenizer_path)

    text = (
        "The STM32F411 communicates with the "
        "VL53L8CX over I²C at 400 kHz."
    )

    normalized = normalize_text(text)

    original_ids = tokenizer.encode(normalized)
    loaded_ids = loaded.encode(normalized)

    assert loaded_ids == original_ids

    decoded = loaded.decode(loaded_ids)

    assert decoded == normalized