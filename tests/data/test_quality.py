"""Tests for document-quality measurement and filtering.

The central risk this module guards against is *over-rejection*.  A filter that
quietly discards source code, formulas, or datasheet prose would remove exactly
the technical capability the corpus exists to teach, and would do so without
any visible failure.  Most tests below therefore assert that realistic
technical documents are **accepted**.
"""

from __future__ import annotations

import pytest

from llm.data.quality import (
    REASON_CONTROL_CHARS,
    REASON_DUPLICATE_LINES,
    REASON_EMPTY,
    REASON_KEYWORD_WALL,
    REASON_LOW_ALPHA,
    REASON_NAVIGATION,
    REASON_REPLACEMENT_CHARS,
    REASON_SECRET,
    REASON_TOO_FEW_WORDS,
    REASON_TOO_SHORT,
    REASON_URL_DIRECTORY,
    QualityThresholds,
    assess_document,
    compute_metrics,
    find_secret,
)


# Distinct sentences, not one sentence repeated.  A repeated sentence has a
# type-token ratio near 0.05, which the keyword-wall rule correctly rejects, so
# a naive fixture would make a working filter look broken.
_SENTENCES = (
    "The controller samples the input voltage and updates the duty cycle.",
    "Feedback compensation determines how quickly the loop settles.",
    "Ripple current through the inductor sets the minimum capacitance.",
    "Thermal derating becomes significant above eighty degrees ambient.",
    "Layout parasitics dominate switching losses at higher frequencies.",
    "A snubber network damps ringing caused by leakage inductance.",
    "Efficiency peaks near half load and falls sharply under light load.",
    "Soft start limits inrush current while the output capacitor charges.",
    "Overcurrent protection latches off after three consecutive faults.",
    "Firmware reports telemetry through an isolated serial connection.",
    "Calibration constants live in a reserved region of nonvolatile memory.",
    "Field failures traced back to insufficient creepage on the primary side.",
)


def prose(repeat: int = 2) -> str:
    """Build varied English prose long enough to pass length thresholds."""

    return " ".join(_SENTENCES * repeat) + " "


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_metrics_are_pure_measurement_and_never_reject() -> None:
    metrics = compute_metrics("")
    assert metrics.characters == 0
    assert metrics.words == 0
    assert metrics.alphabetic_ratio == 0.0


def test_metrics_counts_basic_structure() -> None:
    text = "alpha beta\ngamma\n\ndelta"
    metrics = compute_metrics(text)

    assert metrics.characters == len(text)
    assert metrics.utf8_bytes == len(text.encode("utf-8"))
    assert metrics.lines == 4
    assert metrics.non_empty_lines == 3
    assert metrics.words == 4
    assert metrics.unique_words == 4


def test_metrics_utf8_bytes_differ_from_characters_for_non_ascii() -> None:
    text = "R = 4.7 kΩ ± 5%"
    metrics = compute_metrics(text)
    assert metrics.utf8_bytes > metrics.characters


def test_duplicate_line_ratio_detects_repetition() -> None:
    text = "\n".join(["same line"] * 10)
    metrics = compute_metrics(text)
    assert metrics.duplicate_line_ratio == pytest.approx(0.9)


def test_type_token_ratio_detects_low_vocabulary() -> None:
    metrics = compute_metrics("buy shoes " * 100)
    assert metrics.type_token_ratio < 0.05


def test_compute_metrics_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        compute_metrics(b"bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Technical material must survive
# ---------------------------------------------------------------------------


def test_source_code_is_accepted() -> None:
    text = (
        "The I2C driver checks the hardware abstraction layer status before "
        "returning control to the caller. A timeout indicates that the "
        "peripheral did not acknowledge the address phase within the "
        "configured window, which usually means the bus is held low.\n\n"
        "static int read_register(uint8_t addr, uint8_t *out)\n"
        "{\n"
        "    if (status != HAL_OK) {\n"
        "        return ERROR_I2C_TIMEOUT;\n"
        "    }\n"
        "    *out = rx_buffer[0];\n"
        "    return 0;\n"
        "}\n\n"
        + prose()
    )
    verdict = assess_document(text)
    assert verdict.accepted, verdict.reason


def test_formulas_and_units_are_accepted() -> None:
    text = (
        "A resistive divider scales the input voltage before it reaches the "
        "converter input. The design must account for tolerance stack-up.\n\n"
        "Vout = Vin × R2 / (R1 + R2)\n"
        "R1 = 10 kΩ ± 1%\n"
        "R2 = 4.7 kΩ ± 5%\n"
        "Vref = 2.500 V\n\n"
        + prose()
    )
    verdict = assess_document(text)
    assert verdict.accepted, verdict.reason


def test_high_digit_density_alone_does_not_reject() -> None:
    text = prose() + "\n" + "\n".join(
        f"channel {i} reading 1024 offset 512 gain 2" for i in range(40)
    )
    verdict = assess_document(text)
    assert verdict.accepted, verdict.reason


def test_ordinary_blog_prose_is_accepted() -> None:
    verdict = assess_document(prose(4))
    assert verdict.accepted, verdict.reason


def test_realistic_prose_has_healthy_type_token_ratio() -> None:
    """Guards the fixture itself, so a bad fixture cannot mask a bad filter."""

    metrics = compute_metrics(prose())
    assert metrics.type_token_ratio > 0.20


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_empty_text_rejected() -> None:
    assert assess_document("   \n  ").reason == REASON_EMPTY


def test_short_text_rejected() -> None:
    assert assess_document("Too short.").reason == REASON_TOO_SHORT


def test_too_few_words_rejected() -> None:
    text = "1234567890 " * 60
    verdict = assess_document(text)
    assert verdict.reason == REASON_TOO_FEW_WORDS


def test_garbled_text_rejected_by_alphabetic_ratio() -> None:
    text = ("@#$%^&*()_+{}|:<>?~`" * 40) + " word " * 60
    verdict = assess_document(text)
    assert verdict.reason == REASON_LOW_ALPHA


def test_replacement_characters_rejected() -> None:
    text = prose() + ("�" * 20)
    assert assess_document(text).reason == REASON_REPLACEMENT_CHARS


def test_control_characters_rejected() -> None:
    text = prose() + ("\x00" * 40)
    assert assess_document(text).reason == REASON_CONTROL_CHARS


def test_massively_duplicated_lines_rejected() -> None:
    text = "\n".join(["Subscribe to our newsletter today please"] * 40)
    assert assess_document(text).reason == REASON_DUPLICATE_LINES


def test_navigation_dump_rejected() -> None:
    nav = [
        "Home", "Products", "Contact", "Privacy", "Accept cookies", "Next",
        "Previous", "Login", "Register", "About us", "Terms", "Search",
        "Categories", "Archives", "Tags", "Comments", "Sitemap", "Help",
        "Support", "Download", "My account", "Checkout", "Newsletter",
        "Follow us", "Share", "Tweet", "Read more", "Click here", "FAQ",
        "Menu",
    ]
    verdict = assess_document("\n".join(nav * 3))
    assert verdict.reason in (REASON_NAVIGATION, REASON_DUPLICATE_LINES)


def test_url_directory_rejected() -> None:
    lines = [f"https://example.com/page/{i}" for i in range(60)]
    text = prose() + "\n" + "\n".join(lines)
    assert assess_document(text).reason == REASON_URL_DIRECTORY


def test_keyword_wall_rejected() -> None:
    text = "cheap shoes buy shoes online shoes " * 80
    assert assess_document(text).reason == REASON_KEYWORD_WALL


# ---------------------------------------------------------------------------
# Regressions found by auditing 10,000 real Common Crawl documents
# ---------------------------------------------------------------------------


def test_repeated_http_error_page_rejected() -> None:
    """Observed accepted in the 10k audit before duplicate_line_min_lines fell.

    Nine identical lines of an origin-server error page, which is pure noise
    but sat just under the previous ten-line gate.
    """

    line = (
        "The server is temporarily unable to service your request due to "
        "maintenance downtime or capacity problems. Please try again later."
    )
    verdict = assess_document("\n".join([line] * 9))
    assert verdict.reason == REASON_DUPLICATE_LINES


def test_short_repeated_promo_banner_rejected() -> None:
    line = (
        "Shop the sale now through July 15th. No checkout required; everything "
        "in your cart will be automatically billed on July 16th."
    )
    verdict = assess_document("\n".join([line] * 4))
    assert verdict.reason == REASON_DUPLICATE_LINES


def test_repeating_a_heading_twice_is_not_a_duplicate_document() -> None:
    """Real pages legitimately echo a heading; that must stay acceptable."""

    text = "Could you be a chaplain?\nCould you be a chaplain?\n" + prose(2)
    verdict = assess_document(text)
    assert verdict.accepted, verdict.reason


def test_low_vocabulary_technical_prose_survives() -> None:
    """Patent and catalogue prose measured 0.122-0.123 type-token ratio.

    The keyword-wall threshold must sit clear of that band, otherwise the
    filter silently deletes exactly the technical writing the corpus wants.
    """

    vocabulary = [
        "telescopic", "rod", "connecting", "sleeve", "fixedly", "connected",
        "utility", "model", "discloses", "device", "construction", "brushing",
        "assembly", "mounted", "rotating", "shaft", "bearing", "housing",
        "spring", "plate",
    ]
    text = " ".join(vocabulary * 9)
    metrics = compute_metrics(text)

    assert 0.10 < metrics.type_token_ratio < 0.12
    assert assess_document(text).accepted


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "snippet",
    [
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg\n-----END PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_" + "a" * 36,
        # Split like the neighbours above: written as one literal, this matches
        # GitHub's push-protection scanner and blocks the push even though the
        # value is synthetic.
        "xoxb-" + "123456789012-abcdefghijklmno",
        "AIza" + "b" * 35,
        "sk_live_" + "c" * 30,
        "Authorization: Bearer " + "d" * 60,
    ],
)
def test_high_confidence_secrets_detected(snippet: str) -> None:
    assert find_secret(snippet) is not None


@pytest.mark.parametrize(
    "snippet",
    [
        "Contact us at user@example.com for details.",
        "The gateway is reachable at 192.168.1.100 on the lab subnet.",
        "Start the dev server on http://localhost:8080 and reload.",
        'Set API_KEY="example" in your local configuration file.',
        "password = your_password_here  # placeholder",
        "The token endpoint returns a bearer token valid for one hour.",
        "Use ssh-keygen to generate a key pair before deploying.",
    ],
)
def test_ordinary_documentation_is_not_flagged_as_secret(snippet: str) -> None:
    assert find_secret(snippet) is None


def test_document_containing_secret_is_rejected() -> None:
    text = prose() + "\n-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg\n"
    assert assess_document(text).reason == REASON_SECRET


def test_secret_check_can_be_disabled() -> None:
    text = prose() + "\n-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg\n"
    assert assess_document(text, check_secrets=False).accepted


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def test_thresholds_are_configurable() -> None:
    text = "Short document with only a handful of words in it here now."
    assert not assess_document(text).accepted

    lenient = QualityThresholds(min_characters=10, min_words=5)
    assert assess_document(text, thresholds=lenient).accepted


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_characters": -1},
        {"min_alphabetic_ratio": 1.5},
        {"max_control_ratio": -0.1},
        {"max_url_line_ratio": 2.0},
    ],
)
def test_invalid_thresholds_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        QualityThresholds(**kwargs)


def test_verdict_carries_metrics_even_when_rejected() -> None:
    verdict = assess_document("tiny")
    assert not verdict.accepted
    assert verdict.metrics.characters == 4
