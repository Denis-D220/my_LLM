"""Regression tests for the conservative English script sanity filter."""

from __future__ import annotations

import pytest

from llm.data.language_sanity import (
    REASON_LANGUAGE_SCRIPT_MISMATCH,
    ScriptSanityThresholds,
    assess_english_script,
    classify_alphabetic_script,
    compute_script_metrics,
)


ENGLISH = (
    "The switching regulator controls output voltage by adjusting duty cycle. "
    "Feedback compensation determines loop stability and transient response. "
    "The inductor current rises during the on interval and falls while the "
    "switch is open. Proper layout minimizes parasitic inductance and noise. "
) * 3

RUSSIAN = (
    "Эта техническая документация описывает работу электронной системы, "
    "методы измерения напряжения, обработку сигналов и настройку оборудования. "
    "Инженеры используют результаты испытаний для проверки надежности схемы. "
) * 5

KOREAN = (
    "이 기술 문서는 전자 시스템의 동작과 전압 측정 방법을 설명합니다. "
    "엔지니어는 시험 결과를 사용하여 회로의 안정성과 성능을 검증합니다. "
    "센서 데이터는 제어 장치에서 처리되고 기록됩니다. "
) * 8

JAPANESE = (
    "この技術文書では電子回路の動作と測定方法について説明します。"
    "制御装置はセンサから取得したデータを処理し結果を記録します。"
    "設計者は試験結果を確認してシステムの信頼性を評価します。"
) * 8


def test_classifies_representative_scripts() -> None:
    assert classify_alphabetic_script("A") == "latin"
    assert classify_alphabetic_script("é") == "latin"
    assert classify_alphabetic_script("Ж") == "cyrillic"
    assert classify_alphabetic_script("λ") == "greek"
    assert classify_alphabetic_script("한") == "east_asian"
    assert classify_alphabetic_script("あ") == "east_asian"
    assert classify_alphabetic_script("界") == "east_asian"


def test_non_alphabetic_character_is_not_classified() -> None:
    with pytest.raises(ValueError):
        classify_alphabetic_script("7")


def test_metrics_use_only_alphabetic_characters() -> None:
    metrics = compute_script_metrics("ABC 123 +-= λλ !!")
    assert metrics.alphabetic_characters == 5
    assert metrics.latin_characters == 3
    assert metrics.script_counts["greek"] == 2
    assert metrics.latin_share == pytest.approx(3 / 5)


def test_metrics_reject_non_string() -> None:
    with pytest.raises(TypeError):
        compute_script_metrics(b"bytes")  # type: ignore[arg-type]


def test_english_prose_is_accepted() -> None:
    verdict = assess_english_script(ENGLISH)
    assert verdict.accepted, verdict.metrics
    assert verdict.metrics.latin_share > 0.99


def test_russian_page_mislabeled_english_is_rejected() -> None:
    """Regression for Cyrillic pages observed in the accepted 100k corpus."""

    verdict = assess_english_script(RUSSIAN)
    assert not verdict.accepted
    assert verdict.reason == REASON_LANGUAGE_SCRIPT_MISMATCH
    assert verdict.metrics.dominant_non_latin_script == "cyrillic"
    assert verdict.metrics.dominant_non_latin_share > 0.90


def test_korean_page_mislabeled_english_is_rejected() -> None:
    """Regression for Korean/CJK pages observed in the accepted 100k corpus."""

    verdict = assess_english_script(KOREAN)
    assert not verdict.accepted
    assert verdict.reason == REASON_LANGUAGE_SCRIPT_MISMATCH
    assert verdict.metrics.dominant_non_latin_script == "east_asian"
    assert verdict.metrics.dominant_non_latin_share > 0.80


def test_japanese_mixed_han_and_hiragana_is_rejected() -> None:
    """Han + Hiragana are grouped so Japanese cannot evade the dominance gate."""

    verdict = assess_english_script(JAPANESE)
    assert not verdict.accepted
    assert verdict.metrics.dominant_non_latin_script == "east_asian"


def test_source_code_and_english_comments_are_accepted() -> None:
    text = ENGLISH + "\n" + "\n".join(
        [
            "static int read_register(uint8_t addr, uint8_t *out) {",
            "    if (status != HAL_OK) return ERROR_I2C_TIMEOUT;",
            "    *out = rx_buffer[0];",
            "    return 0;",
            "}",
        ]
        * 20
    )
    verdict = assess_english_script(text)
    assert verdict.accepted, verdict.metrics


def test_greek_formula_symbols_do_not_trigger_rejection() -> None:
    text = ENGLISH + "\n" + ("α β γ δ ε λ μ Ω Δ Vout Vin Rload " * 20)
    verdict = assess_english_script(text)
    assert verdict.accepted, verdict.metrics


def test_latin_script_foreign_text_is_not_this_filters_job() -> None:
    """Spanish/French/Vietnamese use Latin script; metadata filter handles them."""

    spanish = (
        "Este documento técnico describe el funcionamiento del sistema y los "
        "métodos utilizados para medir la tensión y verificar los resultados. "
    ) * 8
    verdict = assess_english_script(spanish)
    assert verdict.accepted
    assert verdict.metrics.latin_share > 0.95


def test_short_non_latin_snippet_is_accepted_conservatively() -> None:
    text = "Привет мир. Короткая цитата."
    verdict = assess_english_script(text)
    assert verdict.metrics.alphabetic_characters < 100
    assert verdict.accepted


def test_mixed_page_with_latin_majority_is_accepted() -> None:
    text = ENGLISH + "\n" + ("Техническая заметка. " * 20)
    verdict = assess_english_script(text)
    assert verdict.metrics.latin_share >= 0.50
    assert verdict.accepted


def test_thresholds_are_tunable() -> None:
    text = "A" * 40 + "Ж" * 60
    default = assess_english_script(text)
    assert default.accepted  # 60% Cyrillic is below conservative 70% dominance.

    stricter = assess_english_script(
        text,
        thresholds=ScriptSanityThresholds(
            min_alphabetic_characters=100,
            min_latin_share=0.50,
            min_dominant_non_latin_share=0.60,
        ),
    )
    assert not stricter.accepted


# ---------------------------------------------------------------------------
# Regressions found auditing the 100k script-gate run
# ---------------------------------------------------------------------------


ARABIC_WITH_DIACRITICS = (
    "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ "
) * 12


def test_script_shares_never_exceed_one() -> None:
    """Arabic harakat sit inside the Arabic range but are not alphabetic.

    Counting them in the numerator while excluding them from the alphabetic
    denominator produced dominant shares of 1.05 in the first 100k run.
    """

    metrics = compute_script_metrics(ARABIC_WITH_DIACRITICS)

    assert metrics.dominant_non_latin_script == "arabic"
    assert 0.0 <= metrics.dominant_non_latin_share <= 1.0
    assert 0.0 <= metrics.latin_share <= 1.0


def test_script_counts_sum_to_alphabetic_total() -> None:
    """The per-script tally must partition the alphabetic characters exactly."""

    for text in (ENGLISH, RUSSIAN, KOREAN, JAPANESE, ARABIC_WITH_DIACRITICS):
        metrics = compute_script_metrics(text)
        assert sum(metrics.script_counts.values()) == (
            metrics.alphabetic_characters
        )


def test_combining_marks_are_not_counted_as_letters() -> None:
    """A bare combining mark contributes nothing to any script tally."""

    metrics = compute_script_metrics("abcًْ")
    assert metrics.alphabetic_characters == 3
    assert metrics.script_counts == {"latin": 3}


def test_arabic_document_is_still_rejected_after_the_fix() -> None:
    verdict = assess_english_script(ARABIC_WITH_DIACRITICS)
    assert not verdict.accepted
    assert verdict.reason == REASON_LANGUAGE_SCRIPT_MISMATCH


def test_threshold_validation() -> None:
    with pytest.raises(ValueError):
        ScriptSanityThresholds(min_alphabetic_characters=-1)
    with pytest.raises(ValueError):
        ScriptSanityThresholds(min_latin_share=1.1)
    with pytest.raises(ValueError):
        ScriptSanityThresholds(min_dominant_non_latin_share=-0.1)
