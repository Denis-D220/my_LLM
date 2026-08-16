"""Tests for evaluation decontamination.

The asymmetry that matters: a short evaluation passage buried inside a large
training page is fully contaminated even though symmetric Jaccard between the
two documents is near zero.  Several tests below check exactly that, because a
detector built on Jaccard would pass every other test in this file and still
let the benchmark be memorised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.data.decontamination import (
    DEFAULT_MIN_CONTAINMENT,
    REASON_CONTAINMENT,
    REASON_EXACT_MATCH,
    ContaminationVerdict,
    DecontaminationConfig,
    EvaluationIndex,
    EvaluationItem,
    containment,
    load_evaluation_items,
    normalized_fingerprint,
)
from llm.data.near_dedup import hashed_word_shingles


EVAL_PASSAGE = (
    "A switching regulator transfers energy in discrete packets rather than "
    "dissipating the difference between input and output as heat. During the "
    "on interval the inductor current rises and stores magnetic energy, and "
    "during the off interval that stored energy is delivered to the output "
    "through the freewheeling path. The ratio of on time to the full switching "
    "period determines the conversion ratio, and feedback adjusts that ratio "
    "so the output stays at the programmed value as load and input vary. "
    "Efficiency is highest near the middle of the load range because "
    "conduction losses dominate at high current while switching and quiescent "
    "losses dominate at light load."
)

SECOND_EVAL_PASSAGE = (
    "Modular arithmetic partitions the integers into residue classes by their "
    "remainder on division by a fixed modulus. Addition, subtraction, and "
    "multiplication all respect this partition, so arithmetic can be carried "
    "out on the representatives without leaving the system. Division is the "
    "exception: a multiplicative inverse exists only when the divisor shares "
    "no common factor with the modulus, which is why the extended Euclidean "
    "algorithm appears wherever modular inverses are required, including in "
    "public key cryptography and in checksum digit schemes."
)

WEB_BOILERPLATE = (
    "Home Products Services Support Downloads Contact About Us Privacy Policy "
    "Terms of Use Accessibility Statement Careers Newsroom Investor Relations "
    "Sitemap Cookie Preferences Manage Consent Accept All Reject Non Essential "
)


def filler(paragraphs: int = 60) -> str:
    """Varied surrounding text.

    Repeating one boilerplate block would not work: shingle sets are unique, so
    the same block repeated forty times contributes almost no distinct
    shingles and the page stays small in shingle terms.  Real pages carry
    genuinely varied material, and only varied filler produces the size
    asymmetry these tests are about.
    """

    return "\n".join(
        f"Section {n} of the catalogue lists part number QX{n:04d} with a "
        f"lead time of {n % 12 + 1} weeks, a minimum order quantity of "
        f"{n * 7 + 3} units, and a unit price that depends on the finish "
        f"selected at checkout for revision {n % 5}."
        for n in range(paragraphs)
    )


def build_index(
    *items: EvaluationItem,
    config: DecontaminationConfig | None = None,
) -> EvaluationIndex:
    return EvaluationIndex(items or default_items(), config=config)


def default_items() -> list[EvaluationItem]:
    return [
        EvaluationItem("eval-eng-0001", "engineering", EVAL_PASSAGE),
        EvaluationItem("eval-math-0001", "mathematics", SECOND_EVAL_PASSAGE),
    ]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_containment_is_asymmetric() -> None:
    """The property the whole module rests on."""

    small = hashed_word_shingles("alpha beta gamma delta epsilon zeta eta theta")
    large = hashed_word_shingles(
        "prefix words here alpha beta gamma delta epsilon zeta eta theta "
        "and then a great deal of unrelated trailing material follows on"
    )

    assert containment(small, large) == pytest.approx(1.0)
    assert containment(large, small) < 0.5


def test_containment_of_empty_evaluation_is_zero() -> None:
    assert containment((), (1, 2, 3)) == 0.0


def test_fingerprint_is_deterministic_and_normalization_aware() -> None:
    assert normalized_fingerprint("abc") == normalized_fingerprint("abc")
    # CRLF is normalized to LF by the frozen contract, so these agree.
    assert normalized_fingerprint("a\r\nb") == normalized_fingerprint("a\nb")


def test_fingerprint_differs_for_different_text() -> None:
    assert normalized_fingerprint("alpha") != normalized_fingerprint("beta")


# ---------------------------------------------------------------------------
# The eight required cases
# ---------------------------------------------------------------------------


def test_exact_evaluation_document_is_rejected() -> None:
    index = build_index()
    verdict = index.check(EVAL_PASSAGE)

    assert verdict.contaminated
    assert verdict.reason == REASON_EXACT_MATCH
    assert verdict.best is not None
    assert verdict.best.item_id == "eval-eng-0001"
    assert verdict.best.containment == pytest.approx(1.0)


def test_evaluation_passage_embedded_in_huge_page_is_rejected() -> None:
    """The case symmetric Jaccard cannot catch."""

    index = build_index()
    page = filler(60) + "\n\n" + EVAL_PASSAGE + "\n\n" + filler(60)

    verdict = index.check(page)

    assert verdict.contaminated
    assert verdict.reason == REASON_CONTAINMENT
    assert verdict.best is not None
    assert verdict.best.containment >= DEFAULT_MIN_CONTAINMENT

    # Prove the detector is not simply finding a near-duplicate: the page and
    # the evaluation item are overwhelmingly dissimilar by Jaccard.
    page_shingles = hashed_word_shingles(page)
    eval_shingles = hashed_word_shingles(EVAL_PASSAGE)
    shared = len(set(page_shingles) & set(eval_shingles))
    union = len(set(page_shingles) | set(eval_shingles))
    assert shared / union < 0.20


def test_evaluation_with_formatting_changes_is_rejected() -> None:
    index = build_index()
    reformatted = (
        EVAL_PASSAGE.upper()
        .replace(". ", ".\n\n")
        .replace(",", " ;")
        .replace("  ", " ")
    )

    verdict = index.check(reformatted)

    assert verdict.contaminated
    assert verdict.best is not None
    assert verdict.best.containment >= DEFAULT_MIN_CONTAINMENT


def test_same_topic_different_explanation_is_kept() -> None:
    index = build_index()
    different = (
        "Buck converters step voltage down by chopping the supply with a "
        "transistor and smoothing the result with an output filter. Designers "
        "size the filter for acceptable ripple, choose a switching frequency "
        "that balances magnetics size against gate losses, and verify that the "
        "control loop remains stable across the whole operating envelope. "
        "Thermal performance is usually verified last, on hardware, because "
        "board copper area dominates the result and is hard to model well."
    )

    assert not index.check(different).contaminated


def test_same_equations_different_prose_is_kept() -> None:
    index = build_index()
    shared_equations = (
        "Vout = Vin * D\n"
        "D = Ton / (Ton + Toff)\n"
        "IL_ripple = (Vin - Vout) * D / (L * f)\n"
    )
    document = (
        "The relationships below are quoted from a datasheet application note "
        "and are used here to size the magnetics for a laboratory supply.\n"
        + shared_equations
        + "Substituting the worst case input and the maximum load gives the "
        "peak inductor current, from which the saturation rating follows. "
        "The remaining work is mechanical: fitting the chosen inductor into "
        "the available height and keeping the switch node copper compact."
    )

    assert not index.check(document).contaminated


def test_short_generic_phrase_overlap_is_kept() -> None:
    index = build_index()
    document = (
        "During the on interval the inductor current rises, which is a phrase "
        "that appears in a great many power electronics texts. The remainder "
        "of this page concerns warehouse logistics, pallet racking heights, "
        "forklift turning radii, and the scheduling of inbound deliveries "
        "across three distribution centres in different time zones. None of "
        "that material has anything to do with switching converters at all."
    )

    assert not index.check(document).contaminated


def test_source_code_with_coincidental_tokens_is_kept() -> None:
    index = build_index()
    document = (
        "static int regulator_set_output(struct regulator *reg, int uv)\n"
        "{\n"
        "    if (uv < reg->min_uv || uv > reg->max_uv)\n"
        "        return -EINVAL;\n"
        "    reg->target_uv = uv;\n"
        "    return regulator_apply(reg);\n"
        "}\n"
        "\n"
        "/* The inductor current is measured by the ADC on channel three. */\n"
        "static int inductor_current_ua(const struct regulator *reg)\n"
        "{\n"
        "    return adc_read(reg->sense_channel) * reg->sense_gain;\n"
        "}\n"
    ) * 6

    assert not index.check(document).contaminated


def test_hashing_is_deterministic_across_index_instances() -> None:
    page = WEB_BOILERPLATE * 10 + EVAL_PASSAGE

    first = build_index().check(page)
    second = build_index().check(page)

    assert first == second
    assert isinstance(first, ContaminationVerdict)


# ---------------------------------------------------------------------------
# Index behaviour
# ---------------------------------------------------------------------------


def test_clean_document_reports_no_matches() -> None:
    verdict = build_index().check(WEB_BOILERPLATE * 20)
    assert not verdict.contaminated
    assert verdict.matches == ()
    assert verdict.best is None
    assert verdict.reason is None


def test_document_may_match_several_evaluation_items() -> None:
    index = build_index()
    combined = EVAL_PASSAGE + "\n\n" + SECOND_EVAL_PASSAGE
    verdict = index.check(combined)

    assert verdict.contaminated
    assert len(verdict.matches) == 2
    assert {match.item_id for match in verdict.matches} == {
        "eval-eng-0001",
        "eval-math-0001",
    }


def test_exact_match_is_reported_before_containment() -> None:
    index = build_index()
    verdict = index.check(EVAL_PASSAGE + "\n\n" + SECOND_EVAL_PASSAGE)
    # Not an exact match of either item, so both are containment matches.
    assert all(match.reason == REASON_CONTAINMENT for match in verdict.matches)

    exact = index.check(EVAL_PASSAGE)
    assert exact.matches[0].reason == REASON_EXACT_MATCH


def test_short_evaluation_items_are_skipped() -> None:
    index = EvaluationIndex(
        [EvaluationItem("eval-tiny", "general", "far too short to index")]
    )

    assert len(index) == 0
    assert index.skipped_short == ["eval-tiny"]
    assert not index.check("far too short to index").contaminated


def test_empty_index_never_reports_contamination() -> None:
    index = EvaluationIndex([])
    assert not index.check(EVAL_PASSAGE).contaminated


def test_threshold_is_configurable() -> None:
    partial = " ".join(EVAL_PASSAGE.split()[:55])

    lenient = EvaluationIndex(
        default_items(),
        config=DecontaminationConfig(min_containment=0.25),
    )
    strict = EvaluationIndex(
        default_items(),
        config=DecontaminationConfig(min_containment=0.95),
    )

    assert lenient.check(partial).contaminated
    assert not strict.check(partial).contaminated


def test_check_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        build_index().check(b"bytes")  # type: ignore[arg-type]


def test_category_counts() -> None:
    assert build_index().category_counts() == {
        "engineering": 1,
        "mathematics": 1,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shingle_size": 0},
        {"min_containment": 0.0},
        {"min_containment": 1.5},
        {"min_evaluation_tokens": 0},
    ],
)
def test_invalid_config_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DecontaminationConfig(**kwargs)


# ---------------------------------------------------------------------------
# Loading the frozen evaluation file
# ---------------------------------------------------------------------------


def write_eval_file(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        + "\n",
        encoding="utf-8",
    )
    return path


def test_load_evaluation_items(tmp_path: Path) -> None:
    path = write_eval_file(
        tmp_path / "pretraining_eval.jsonl",
        [
            {"id": "eval-a", "category": "engineering", "text": EVAL_PASSAGE},
            {"id": "eval-b", "category": "mathematics", "text": SECOND_EVAL_PASSAGE},
        ],
    )

    items = load_evaluation_items(path)
    assert [item.item_id for item in items] == ["eval-a", "eval-b"]
    assert items[0].category == "engineering"


def test_load_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = write_eval_file(
        tmp_path / "eval.jsonl",
        [
            {"id": "eval-a", "category": "x", "text": EVAL_PASSAGE},
            {"id": "eval-a", "category": "x", "text": SECOND_EVAL_PASSAGE},
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_evaluation_items(path)


def test_load_rejects_missing_text(tmp_path: Path) -> None:
    path = write_eval_file(tmp_path / "eval.jsonl", [{"id": "eval-a"}])
    with pytest.raises(ValueError):
        load_evaluation_items(path)


def test_load_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_evaluation_items(path)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_evaluation_items(tmp_path / "nope.jsonl")
