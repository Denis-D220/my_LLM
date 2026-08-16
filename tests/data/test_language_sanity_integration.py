"""Corpus-builder regression tests for ``language_script_mismatch``.

These tests are self-contained so they can be dropped into the existing test
suite after the small integration changes described in INTEGRATION.md.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from llm.data.language_sanity import REASON_LANGUAGE_SCRIPT_MISMATCH
from llm.data.pretraining_corpus import (
    build_pretraining_corpus as _build_pretraining_corpus,
)


def build_pretraining_corpus(**kwargs):
    """Script-sanity tests do not involve the evaluation set.

    The builder fails closed on a missing evaluation artifact, so these tests
    opt out explicitly. See test_pretraining_corpus_decontamination.py for the
    decontamination behaviour itself.
    """

    kwargs.setdefault("check_decontamination", False)
    return _build_pretraining_corpus(**kwargs)


ENGLISH = (
    "The controller samples input voltage and adjusts the duty cycle to keep "
    "the regulated output stable. The firmware records telemetry and reports "
    "fault conditions through an isolated serial interface. "
) * 5

RUSSIAN = (
    "Эта техническая документация описывает работу электронной системы, "
    "измерение напряжения, обработку сигналов и настройку оборудования. "
) * 8

KOREAN = (
    "이 기술 문서는 전자 시스템의 동작과 전압 측정 방법을 설명합니다. "
    "엔지니어는 시험 결과를 사용하여 회로의 성능을 검증합니다. "
) * 10


def _write_source(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record(text: str, index: int) -> dict:
    return {
        "text": text,
        "url": f"https://example.com/{index}",
        "date": "2026-08-14T00:00:00Z",
        # Deliberately wrong for the non-Latin fixtures: this is exactly the
        # Common Crawl metadata failure the second sanity gate is meant to catch.
        "lang": "eng",
    }


def test_builder_rejects_obvious_script_mismatches(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(
        source / "cc_text_00000.jsonl.gz",
        [
            _record(ENGLISH, 0),
            _record(RUSSIAN, 1),
            _record(KOREAN, 2),
        ],
    )

    report = build_pretraining_corpus(source=source, output=output)

    assert report["stats"]["documents_accepted"] == 1
    assert (
        report["stats"]["rejected_by_reason"][
            REASON_LANGUAGE_SCRIPT_MISMATCH
        ]
        == 2
    )


def test_builder_can_disable_script_sanity_for_controlled_comparison(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(
        source / "cc_text_00000.jsonl.gz",
        [_record(RUSSIAN, 0)],
    )

    report = build_pretraining_corpus(
        source=source,
        output=output,
        check_script_sanity=False,
    )

    assert report["stats"]["documents_accepted"] == 1
    assert REASON_LANGUAGE_SCRIPT_MISMATCH not in report["stats"][
        "rejected_by_reason"
    ]
