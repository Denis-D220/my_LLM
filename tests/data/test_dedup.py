"""Tests for persistent exact deduplication.

The property that matters most is persistence *across process restarts*: the
extractor's in-memory deduplication already handles a single run, and this
module exists specifically to cover the gap it leaves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm.data.dedup import (
    DEFAULT_DATABASE_NAME,
    ExactDeduplicator,
    fingerprint_hex,
    text_fingerprint,
)


def test_fingerprint_is_stable_and_32_bytes() -> None:
    digest = text_fingerprint("hello world")
    assert isinstance(digest, bytes)
    assert len(digest) == 32
    assert digest == text_fingerprint("hello world")


def test_fingerprint_differs_for_different_text() -> None:
    assert text_fingerprint("a") != text_fingerprint("b")


def test_fingerprint_hex_matches_digest() -> None:
    assert fingerprint_hex("sample") == text_fingerprint("sample").hex()


def test_fingerprint_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        text_fingerprint(b"bytes")  # type: ignore[arg-type]


def test_check_and_add_reports_first_occurrence_as_new(tmp_path: Path) -> None:
    with ExactDeduplicator(tmp_path / DEFAULT_DATABASE_NAME) as dedup:
        assert dedup.check_and_add("document one") is True
        assert dedup.check_and_add("document one") is False
        assert dedup.check_and_add("document two") is True

        assert dedup.stats.checked == 3
        assert dedup.stats.unique == 2
        assert dedup.stats.duplicates == 1
        assert dedup.stats.duplicate_ratio == pytest.approx(1 / 3)


def test_seen_and_add_are_separable(tmp_path: Path) -> None:
    with ExactDeduplicator(tmp_path / "d.sqlite") as dedup:
        assert dedup.seen("text") is False
        dedup.add("text")
        assert dedup.seen("text") is True


def test_state_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "d.sqlite"

    with ExactDeduplicator(path) as first:
        first.check_and_add("persistent document")

    # Simulates the extractor or corpus builder being restarted.
    with ExactDeduplicator(path) as second:
        assert second.seen("persistent document") is True
        assert second.check_and_add("persistent document") is False
        assert second.count() == 1


def test_commit_interval_does_not_lose_data_on_clean_close(
    tmp_path: Path,
) -> None:
    path = tmp_path / "d.sqlite"

    with ExactDeduplicator(path, commit_interval=1000) as dedup:
        for index in range(50):
            dedup.check_and_add(f"document {index}")

    with ExactDeduplicator(path) as reopened:
        assert reopened.count() == 50


def test_explicit_commit_persists_before_close(tmp_path: Path) -> None:
    path = tmp_path / "d.sqlite"

    dedup = ExactDeduplicator(path, commit_interval=10_000)
    dedup.check_and_add("committed early")
    dedup.commit()

    with ExactDeduplicator(path, read_only=True) as reader:
        assert reader.seen("committed early") is True

    dedup.close()


def test_add_is_idempotent(tmp_path: Path) -> None:
    with ExactDeduplicator(tmp_path / "d.sqlite") as dedup:
        dedup.add("same")
        dedup.add("same")
        assert dedup.count() == 1


def test_metadata_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "d.sqlite"

    with ExactDeduplicator(path) as dedup:
        dedup.set_metadata("corpus_version", "v0.1")

    with ExactDeduplicator(path) as reopened:
        assert reopened.get_metadata("corpus_version") == "v0.1"
        assert reopened.get_metadata("missing") is None


def test_read_only_rejects_writes(tmp_path: Path) -> None:
    path = tmp_path / "d.sqlite"
    with ExactDeduplicator(path) as dedup:
        dedup.add("seed")

    with ExactDeduplicator(path, read_only=True) as reader:
        assert reader.seen("seed") is True
        with pytest.raises(RuntimeError):
            reader.add("new")


def test_closed_deduplicator_rejects_use(tmp_path: Path) -> None:
    dedup = ExactDeduplicator(tmp_path / "d.sqlite")
    dedup.close()
    with pytest.raises(RuntimeError):
        dedup.seen("anything")


def test_close_is_idempotent(tmp_path: Path) -> None:
    dedup = ExactDeduplicator(tmp_path / "d.sqlite")
    dedup.close()
    dedup.close()


def test_invalid_commit_interval_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ExactDeduplicator(tmp_path / "d.sqlite", commit_interval=0)


# ---------------------------------------------------------------------------
# Transactional mode
# ---------------------------------------------------------------------------


def test_transactional_mode_does_not_auto_commit(tmp_path: Path) -> None:
    path = tmp_path / "d.sqlite"

    dedup = ExactDeduplicator(path, commit_interval=2, transactional=True)
    dedup.begin()
    for index in range(10):
        dedup.check_and_add(f"document {index}")

    # commit_interval would have fired five times in batched mode.
    assert dedup.pending == 10

    with ExactDeduplicator(path, read_only=True) as reader:
        assert reader.count() == 0

    dedup.commit()
    with ExactDeduplicator(path, read_only=True) as reader:
        assert reader.count() == 10

    dedup.close()


def test_rollback_discards_uncommitted_fingerprints(tmp_path: Path) -> None:
    path = tmp_path / "d.sqlite"

    with ExactDeduplicator(path, transactional=True) as dedup:
        dedup.begin()
        dedup.add("committed")
        dedup.commit()

        dedup.begin()
        dedup.add("rolled back")
        dedup.rollback()

        assert dedup.seen("committed") is True
        assert dedup.seen("rolled back") is False

    with ExactDeduplicator(path) as reopened:
        assert reopened.count() == 1
        assert reopened.seen("rolled back") is False


def test_transactional_close_rolls_back_instead_of_committing(
    tmp_path: Path,
) -> None:
    """The bug this mode exists to prevent.

    If close() committed pending work, fingerprints would survive for
    documents whose output shard was never finished, and reprocessing that
    input would silently discard them as duplicates.
    """

    path = tmp_path / "d.sqlite"

    dedup = ExactDeduplicator(path, transactional=True)
    dedup.begin()
    dedup.add("never committed")
    dedup.close()

    with ExactDeduplicator(path) as reopened:
        assert reopened.count() == 0
        assert reopened.seen("never committed") is False


def test_batched_close_still_commits(tmp_path: Path) -> None:
    path = tmp_path / "d.sqlite"

    dedup = ExactDeduplicator(path, transactional=False)
    dedup.add("batched")
    dedup.close()

    with ExactDeduplicator(path) as reopened:
        assert reopened.seen("batched") is True


def test_begin_rejects_uncommitted_work(tmp_path: Path) -> None:
    with ExactDeduplicator(tmp_path / "d.sqlite", transactional=True) as dedup:
        dedup.begin()
        dedup.add("pending")
        with pytest.raises(RuntimeError):
            dedup.begin()


def test_rollback_is_safe_when_nothing_pending(tmp_path: Path) -> None:
    with ExactDeduplicator(tmp_path / "d.sqlite", transactional=True) as dedup:
        dedup.rollback()
        assert dedup.count() == 0


def test_normalization_sensitive_texts_are_distinct(tmp_path: Path) -> None:
    """Fingerprints are of *normalized* text; the caller must normalize first.

    This test documents the contract rather than asserting a normalization
    behaviour that lives in the tokenizer module.
    """

    with ExactDeduplicator(tmp_path / "d.sqlite") as dedup:
        assert dedup.check_and_add("line\r\nbreak") is True
        assert dedup.check_and_add("line\nbreak") is True
