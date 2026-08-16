"""Tests for first-principles near-duplicate detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm.data.near_dedup import (
    AMBIGUOUS_OVERLAP,
    SAFE_NEAR_DUPLICATE,
    NearDedupConfig,
    NearDuplicateIndex,
    ShingleOverlap,
    build_features,
    classify_overlap,
    comparison_tokens,
    exact_jaccard,
    hashed_word_shingles,
    length_ratio,
    partitioned_signature,
    shingle_overlap,
    signature_similarity,
)


BASE_SENTENCES = [
    "The voltage regulator maintains a stable output as the load current changes.",
    "Feedback compensation determines the transient response of the control loop.",
    "The inductor stores magnetic energy while the switching transistor is active.",
    "Output capacitance reduces ripple and supplies current during switching edges.",
    "Thermal resistance determines the junction temperature at a given power loss.",
    "Layout parasitics become important when switching frequency increases.",
    "The microcontroller samples the converter output through an analog input.",
    "Firmware reports voltage current and temperature over the serial interface.",
    "Current limiting protects the power stage during an accidental short circuit.",
    "Component tolerances must be included when calculating the feedback divider.",
    "The prototype is tested at minimum nominal and maximum input voltage.",
    "Oscilloscope measurements confirm that the compensation network is stable.",
]


def technical_document(repeat: int = 6) -> str:
    lines: list[str] = []
    for cycle in range(repeat):
        for sentence in BASE_SENTENCES:
            lines.append(f"{sentence} Measurement cycle {cycle}.")
    return "\n".join(lines)


def different_technical_document(repeat: int = 6) -> str:
    sentences = [
        "A binary search halves the remaining interval after every comparison.",
        "The array must already be sorted according to the selected ordering.",
        "The lower boundary is inclusive while the upper boundary is exclusive.",
        "Integer overflow can be avoided when calculating the midpoint index.",
        "The algorithm terminates when the search interval becomes empty.",
        "Logarithmic complexity makes the method efficient for large collections.",
        "A comparator can generalize the implementation to structured records.",
        "Unit tests should cover missing values and both ends of the collection.",
        "Duplicate keys require a defined policy for returning the first match.",
        "Iterative implementations avoid recursive call overhead.",
        "Cache behavior is favorable because only a few array entries are touched.",
        "The implementation returns a sentinel when no matching value exists.",
    ]
    return "\n".join(
        f"{sentence} Search pass {cycle}."
        for cycle in range(repeat)
        for sentence in sentences
    )


def index_one(
    db: Path,
    document_id: str,
    text: str,
    *,
    config: NearDedupConfig | None = None,
) -> None:
    with NearDuplicateIndex(db, config=config) as index:
        index.begin()
        features = index.analyze_text(text)
        assert features is not None
        index.add_document(document_id, features, url=f"https://x/{document_id}")
        index.commit()


def test_comparison_tokens_ignore_case_and_punctuation() -> None:
    first = comparison_tokens("HTTP Server: Voltage=5.0V; READY!")
    second = comparison_tokens("http server voltage 5 0v ready")
    assert first == second


def test_hashed_shingles_are_deterministic() -> None:
    text = technical_document()
    assert hashed_word_shingles(text) == hashed_word_shingles(text)


def test_partitioned_signature_is_deterministic() -> None:
    shingles = hashed_word_shingles(technical_document())
    assert partitioned_signature(shingles) == partitioned_signature(shingles)


def test_exact_jaccard_identical_is_one() -> None:
    values = (1, 2, 3, 5)
    assert exact_jaccard(values, values) == 1.0


def test_exact_jaccard_disjoint_is_zero() -> None:
    assert exact_jaccard((1, 2), (3, 4)) == 0.0


def test_length_ratio() -> None:
    assert length_ratio(90, 100) == pytest.approx(0.9)
    assert length_ratio(100, 90) == pytest.approx(0.9)


def test_signature_similarity_requires_equal_lengths() -> None:
    with pytest.raises(ValueError):
        signature_similarity((1, 2), (1,))


def test_identical_document_matches(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    text = technical_document()
    index_one(db, "a", text)

    with NearDuplicateIndex(db) as index:
        features = index.analyze_text(text)
        assert features is not None
        result = index.find_near_duplicates(features)

    assert result.matches
    assert result.matches[0].document_id == "a"
    assert result.matches[0].similarity == pytest.approx(1.0)


def test_case_and_punctuation_variation_matches(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    original = technical_document()
    changed = (
        original.upper()
        .replace(".", " !!!")
        .replace(",", " ; ")
    )

    index_one(db, "original", original)

    with NearDuplicateIndex(db) as index:
        features = index.analyze_text(changed)
        assert features is not None
        result = index.find_near_duplicates(features)

    assert result.matches
    assert result.matches[0].similarity >= 0.90


def test_small_footer_change_matches(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    original = technical_document(repeat=8)
    changed = (
        original
        + "\nCopyright 2026 Example Engineering. All rights reserved."
    )

    index_one(db, "original", original)

    with NearDuplicateIndex(db) as index:
        features = index.analyze_text(changed)
        assert features is not None
        result = index.find_near_duplicates(features)

    assert result.matches
    assert result.matches[0].similarity >= 0.90
    assert result.matches[0].length_ratio >= 0.90


def test_same_general_topic_but_different_writing_does_not_match(
    tmp_path: Path,
) -> None:
    db = tmp_path / "near.sqlite"
    index_one(db, "power", technical_document())

    with NearDuplicateIndex(db) as index:
        features = index.analyze_text(different_technical_document())
        assert features is not None
        result = index.find_near_duplicates(features)

    assert result.matches == ()


def test_short_excerpt_does_not_match_long_document(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    original = technical_document(repeat=10)
    excerpt = "\n".join(original.splitlines()[:15])

    index_one(db, "long", original)

    with NearDuplicateIndex(db) as index:
        features = build_features(excerpt)
        assert features.token_count >= 50
        result = index.find_near_duplicates(features)

    assert result.matches == ()


def test_state_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    original = technical_document()
    index_one(db, "persistent", original)

    with NearDuplicateIndex(db) as reopened:
        assert reopened.count() == 1
        features = reopened.analyze_text(original)
        assert features is not None
        assert reopened.find_near_duplicates(features).matches


def test_rollback_removes_uncommitted_document(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    text = technical_document()

    with NearDuplicateIndex(db) as index:
        index.begin()
        features = index.analyze_text(text)
        assert features is not None
        index.add_document("rolled-back", features)
        assert index.contains("rolled-back")
        index.rollback()
        assert not index.contains("rolled-back")

    with NearDuplicateIndex(db) as reopened:
        assert reopened.count() == 0


def test_close_rolls_back_active_transaction(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    text = technical_document()

    index = NearDuplicateIndex(db)
    index.begin()
    features = index.analyze_text(text)
    assert features is not None
    index.add_document("not-committed", features)
    index.close()

    with NearDuplicateIndex(db) as reopened:
        assert reopened.count() == 0


def test_add_requires_explicit_transaction(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    with NearDuplicateIndex(db) as index:
        features = index.analyze_text(technical_document())
        assert features is not None
        with pytest.raises(RuntimeError):
            index.add_document("a", features)


def test_too_short_document_is_not_analyzed(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    with NearDuplicateIndex(db) as index:
        assert index.analyze_text("short text only") is None


def test_results_are_deterministic(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    original = technical_document(repeat=8)
    variant = original + "\nCopyright 2026 Example Engineering."

    index_one(db, "a", original)

    with NearDuplicateIndex(db) as index:
        features = index.analyze_text(variant)
        assert features is not None
        first = index.find_near_duplicates(features)
        second = index.find_near_duplicates(features)

    assert first == second


# ---------------------------------------------------------------------------
# Unique-information guard
#
# The 100k audit showed that similarity alone cannot separate "a changed
# copyright year" from "a different technical specification behind the same
# site template". Both exceed 0.90 Jaccard. Only the absolute amount of unique
# content distinguishes them.
# ---------------------------------------------------------------------------


SITE_TEMPLATE = "\n".join(
    f"Navigation section {n}: products services support downloads contact "
    f"privacy policy terms of use accessibility statement region {n}."
    for n in range(200)
)


def test_shingle_overlap_counts_both_sides() -> None:
    overlap = shingle_overlap((1, 2, 3, 4), (3, 4, 5))
    assert overlap.shared == 2
    assert overlap.unique_first == 2
    assert overlap.unique_second == 1
    assert overlap.union == 5
    assert overlap.max_unique == 2
    assert overlap.jaccard == pytest.approx(2 / 5)


def test_shingle_overlap_matches_exact_jaccard() -> None:
    first = (1, 3, 5, 7, 9)
    second = (3, 5, 7, 11)
    assert shingle_overlap(first, second).jaccard == exact_jaccard(first, second)


def test_empty_overlap_is_vacuously_identical() -> None:
    assert shingle_overlap((), ()).jaccard == 1.0


def test_classification_boundary_is_inclusive() -> None:
    at_limit = ShingleOverlap(shared=1000, unique_first=16, unique_second=3)
    over_limit = ShingleOverlap(shared=1000, unique_first=17, unique_second=3)

    assert classify_overlap(at_limit) == SAFE_NEAR_DUPLICATE
    assert classify_overlap(over_limit) == AMBIGUOUS_OVERLAP


def test_classification_uses_the_larger_side() -> None:
    """One side carrying unique content is enough to make a pair ambiguous."""

    lopsided = ShingleOverlap(shared=1000, unique_first=0, unique_second=90)
    assert classify_overlap(lopsided) == AMBIGUOUS_OVERLAP


def test_changed_copyright_year_is_a_safe_near_duplicate(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    body = technical_document(repeat=8)
    original = body + "\nCopyright 2025 Example Engineering. All rights reserved."
    revised = body + "\nCopyright 2026 Example Engineering. All rights reserved."

    index_one(db, "original", original)

    with NearDuplicateIndex(db) as index:
        features = index.analyze_text(revised)
        assert features is not None
        result = index.find_near_duplicates(features)

    assert result.matches
    best = result.matches[0]
    assert best.similarity >= 0.90
    assert best.max_unique_shingles <= 16
    assert best.classification == SAFE_NEAR_DUPLICATE
    assert result.best_safe_match is not None


def test_different_specification_behind_shared_template_is_ambiguous(
    tmp_path: Path,
) -> None:
    """The false positive the audit caught: same template, different data.

    Two product pages sharing a large site template but describing different
    machining specifications must survive, however high their similarity.
    """

    db = tmp_path / "near.sqlite"

    milled = SITE_TEMPLATE + "\n" + "\n".join(
        [
            "Process: precision milled on a vertical machining centre.",
            "Flatness tolerance is plus or minus 0.002 inches across the plate.",
            "Material is aluminium 6061 T651 with a clear anodised finish.",
            "Nominal dimensions are 12 by 18 by 0.75 inches before finishing.",
            "Surface roughness is 32 microinches average on the ground faces.",
            "Edges are deburred and the part ships with a protective film.",
        ]
    )
    saw_cut = SITE_TEMPLATE + "\n" + "\n".join(
        [
            "Process: saw cut from plate stock without secondary operations.",
            "Thickness tolerance is plus or minus 0.030 inches as supplied.",
            "Material is cast tooling plate with a mill finish surface.",
            "Nominal dimensions are 10 by 24 by 1.5 inches before machining.",
            "Cut edges show normal saw marks and are not deburred.",
            "Parts are shipped bare and require cleaning before use.",
        ]
    )

    index_one(db, "milled", milled)

    with NearDuplicateIndex(db) as index:
        features = index.analyze_text(saw_cut)
        assert features is not None
        result = index.find_near_duplicates(features)

    assert result.matches, "template overlap should still make these candidates"
    best = result.matches[0]

    # Similarity alone would have deleted one of these.
    assert best.similarity >= 0.90
    assert best.length_ratio >= 0.90

    # The absolute unique content is what saves them.
    assert best.max_unique_shingles > 16
    assert best.classification == AMBIGUOUS_OVERLAP
    assert result.best_safe_match is None
    assert result.ambiguous_matches
    assert result.safe_matches == ()


def test_match_reports_consistent_shingle_arithmetic(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    body = technical_document(repeat=8)
    index_one(db, "a", body)

    with NearDuplicateIndex(db) as index:
        features = index.analyze_text(body + "\nCopyright 2026.")
        assert features is not None
        best = index.find_near_duplicates(features).matches[0]

    assert best.max_unique_shingles == max(
        best.unique_query_shingles, best.unique_candidate_shingles
    )
    union = (
        best.shared_shingles
        + best.unique_query_shingles
        + best.unique_candidate_shingles
    )
    assert best.similarity == pytest.approx(best.shared_shingles / union)


def test_guard_is_configurable(tmp_path: Path) -> None:
    """max_unique_shingles is an audit parameter, not a frozen constant."""

    db = tmp_path / "near.sqlite"
    body = technical_document(repeat=8)
    strict = NearDedupConfig(max_unique_shingles=0)

    index_one(db, "a", body, config=strict)

    with NearDuplicateIndex(db, config=strict) as index:
        features = index.analyze_text(body + "\nCopyright 2026.")
        assert features is not None
        best = index.find_near_duplicates(features).matches[0]

    # Any unique content at all is ambiguous when the guard is zero.
    assert best.classification == AMBIGUOUS_OVERLAP


def test_identical_documents_have_no_unique_content(tmp_path: Path) -> None:
    db = tmp_path / "near.sqlite"
    text = technical_document()
    index_one(db, "a", text)

    with NearDuplicateIndex(db) as index:
        features = index.analyze_text(text)
        assert features is not None
        best = index.find_near_duplicates(features).matches[0]

    assert best.unique_query_shingles == 0
    assert best.unique_candidate_shingles == 0
    assert best.classification == SAFE_NEAR_DUPLICATE


def test_config_rejects_negative_unique_guard() -> None:
    with pytest.raises(ValueError):
        NearDedupConfig(max_unique_shingles=-1)


def test_config_rejects_out_of_range_unique_share() -> None:
    with pytest.raises(ValueError):
        NearDedupConfig(max_unique_share=1.5)


# ---------------------------------------------------------------------------
# Scale-awareness: measurements taken from the real 100k audit
#
# An absolute count alone is scale-blind. These three overlaps are the ones
# that decide the rule, and their absolute counts order them *wrongly*: the
# two records that must be preserved carry more unique shingles (15, 10) than
# the boilerplate that should be removed (8).
# ---------------------------------------------------------------------------


NIST_SPECIES = ShingleOverlap(shared=146, unique_first=15, unique_second=8)
UWM_GEOSPATIAL = ShingleOverlap(shared=207, unique_first=10, unique_second=10)
PHPBB_FAQ = ShingleOverlap(shared=4541, unique_first=8, unique_second=8)


def test_absolute_guard_alone_would_delete_distinct_records() -> None:
    """Documents the defect: absolute-only classification is wrong here."""

    for overlap in (NIST_SPECIES, UWM_GEOSPATIAL):
        assert overlap.max_unique <= 16
        # Absolute guard alone calls these safe to delete. They are not.
        assert (
            classify_overlap(overlap, max_unique_shingles=16, max_unique_share=1.0)
            == SAFE_NEAR_DUPLICATE
        )


def test_short_distinct_records_are_preserved_by_the_relative_guard() -> None:
    assert NIST_SPECIES.unique_share > 0.02
    assert UWM_GEOSPATIAL.unique_share > 0.02

    for overlap in (NIST_SPECIES, UWM_GEOSPATIAL):
        assert classify_overlap(overlap) == AMBIGUOUS_OVERLAP


def test_large_template_boilerplate_is_still_removed() -> None:
    """The relative guard must not make the filter useless."""

    assert PHPBB_FAQ.max_unique <= 16
    assert PHPBB_FAQ.unique_share < 0.02
    assert classify_overlap(PHPBB_FAQ) == SAFE_NEAR_DUPLICATE


def test_relative_and_absolute_guards_are_both_required() -> None:
    """Either guard alone misclassifies one of the decisive cases."""

    # Absolute only: deletes the NIST record.
    assert (
        classify_overlap(
            NIST_SPECIES, max_unique_shingles=16, max_unique_share=1.0
        )
        == SAFE_NEAR_DUPLICATE
    )
    # Relative only: keeps a large-template pair with a genuinely big payload.
    big_payload = ShingleOverlap(shared=100_000, unique_first=900, unique_second=5)
    assert big_payload.unique_share < 0.02
    assert (
        classify_overlap(
            big_payload, max_unique_shingles=10**9, max_unique_share=0.02
        )
        == SAFE_NEAR_DUPLICATE
    )
    assert classify_overlap(big_payload) == AMBIGUOUS_OVERLAP


def test_unique_share_of_identical_documents_is_zero() -> None:
    assert ShingleOverlap(shared=500, unique_first=0, unique_second=0).unique_share == 0.0
    assert ShingleOverlap(shared=0, unique_first=0, unique_second=0).unique_share == 0.0


def test_config_rejects_invalid_band_geometry() -> None:
    with pytest.raises(ValueError):
        NearDedupConfig(signature_components=8, bands=3, rows_per_band=2)


def test_config_rejects_non_power_of_two_signature() -> None:
    with pytest.raises(ValueError):
        NearDedupConfig(signature_components=10, bands=5, rows_per_band=2)
