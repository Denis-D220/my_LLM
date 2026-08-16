"""Tests for the Common Crawl -> cleaned pretraining corpus stage.

These tests build synthetic ``cc_text_*.jsonl.gz`` shards in ``tmp_path`` so the
whole pipeline can be exercised without touching the real 1.85 GB corpus.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from llm.data.pretraining_corpus import (
    DEDUP_NAME,
    MANIFEST_NAME,
    PROGRESS_NAME,
    REASON_EXACT_DUPLICATE,
    REASON_INVALID_JSON,
    REASON_MISSING_TEXT,
    REASON_NON_ENGLISH,
    REASON_TEXT_NOT_STRING,
    REPORT_NAME,
    CleanDocument,
    ShardWriter,
    discover_source_shards,
    iter_shard_lines,
    make_split_group,
    primary_language,
    remove_output_shards_from,
    shard_index,
)
from llm.data.pretraining_corpus import (
    build_pretraining_corpus as _build_pretraining_corpus,
)
from llm.data.quality import QualityThresholds


def build_pretraining_corpus(**kwargs):
    """Run the builder with evaluation decontamination explicitly disabled.

    The builder fails closed: a production build without an evaluation
    artifact is an error. These tests cover schema handling, quality,
    crash-safe resume, and provenance, none of which involve the evaluation
    set, so they opt out deliberately rather than carrying a dummy artifact.

    Decontamination itself is covered by
    ``test_pretraining_corpus_decontamination.py``, including the case where
    omitting the artifact must fail.
    """

    kwargs.setdefault("check_decontamination", False)
    return _build_pretraining_corpus(**kwargs)


PROSE = (
    "The converter regulates its output by adjusting the duty cycle of the "
    "switching element in response to the measured feedback voltage. When the "
    "load increases, the control loop raises the on-time so that the average "
    "inductor current rises to meet demand without exceeding the current "
    "limit programmed into the controller. "
)


def document_text(seed: str = "", repeat: int = 6) -> str:
    return seed + (PROSE * repeat)


def write_shard(directory: Path, name: str, records: list[dict]) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def good_record(index: int, **overrides) -> dict:
    record = {
        "text": document_text(f"Document number {index}. "),
        "url": f"https://example.com/page/{index}",
        "date": "2026-07-10T08:02:18Z",
        "lang": "eng",
    }
    record.update(overrides)
    return record


def read_output_documents(output: Path) -> list[dict]:
    documents: list[dict] = []
    for shard in sorted(output.glob("corpus-*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    documents.append(json.loads(line))
    return documents


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("eng", "eng"),
        ("eng,spa", "eng"),
        ("eng,ind", "eng"),
        ("  ENG , spa ", "eng"),
        ("spa", "spa"),
        ("deu,eng", "deu"),
        ("", None),
        (None, None),
        (123, None),
    ],
)
def test_primary_language(value, expected) -> None:
    assert primary_language(value) == expected


def test_split_group_is_shared_by_same_url() -> None:
    first = make_split_group("https://example.com/a", "cc-1")
    second = make_split_group("https://example.com/a", "cc-2")
    assert first == second


def test_split_group_differs_by_url() -> None:
    assert make_split_group("https://a.com", "cc-1") != make_split_group(
        "https://b.com", "cc-1"
    )


def test_split_group_falls_back_to_document_id() -> None:
    assert make_split_group(None, "cc-abc") == "cc-abc"
    assert make_split_group("   ", "cc-abc") == "cc-abc"


def test_discover_source_shards_excludes_state_file(tmp_path: Path) -> None:
    write_shard(tmp_path, "cc_text_00000.jsonl.gz", [good_record(0)])
    write_shard(tmp_path, "cc_text_00001.jsonl.gz", [good_record(1)])
    (tmp_path / "_state.json").write_text("{}", encoding="utf-8")

    shards = discover_source_shards(tmp_path)
    assert [shard.name for shard in shards] == [
        "cc_text_00000.jsonl.gz",
        "cc_text_00001.jsonl.gz",
    ]


def test_discover_source_shards_requires_matches(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_source_shards(tmp_path)


def test_discover_source_shards_requires_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_source_shards(tmp_path / "missing")


def test_iter_shard_lines_reads_concatenated_gzip_members(
    tmp_path: Path,
) -> None:
    """The extractor produces multi-member gzip files after a resumed download."""

    path = tmp_path / "cc_text_00000.jsonl.gz"
    first = gzip.compress(b'{"a": 1}\n')
    second = gzip.compress(b'{"b": 2}\n')
    path.write_bytes(first + second)

    lines = list(iter_shard_lines(path))
    assert [number for number, _ in lines] == [1, 2]
    assert json.loads(lines[1][1]) == {"b": 2}


# ---------------------------------------------------------------------------
# ShardWriter
# ---------------------------------------------------------------------------


def make_document(index: int) -> CleanDocument:
    return CleanDocument(
        document_id=f"cc-{index:024d}",
        text=f"text {index}",
        url=f"https://example.com/{index}",
        date=None,
        lang="eng",
        source_shard="cc_text_00000.jsonl.gz",
        source_line=index,
        text_sha256="0" * 64,
        split_group="group",
    )


def test_shard_writer_rotates_on_size(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, max_bytes=200)
    for index in range(20):
        writer.write(make_document(index))
    writer.close()

    assert len(writer.shards_written) > 1
    assert writer.documents_written == 20
    assert len(read_output_documents(tmp_path)) == 20


def test_shard_writer_produces_valid_gzip_jsonl(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path)
    writer.write(make_document(1))
    writer.close()

    documents = read_output_documents(tmp_path)
    assert len(documents) == 1
    assert documents[0]["document_id"] == make_document(1).document_id


def test_shard_writer_close_is_safe_when_empty(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path)
    writer.close()
    assert writer.shards_written == []
    assert list(tmp_path.glob("corpus-*.jsonl.gz")) == []


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------


def test_build_accepts_good_documents(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(i) for i in range(5)],
    )

    report = build_pretraining_corpus(source=source, output=output)

    assert report["stats"]["documents_accepted"] == 5
    assert report["stats"]["records_seen"] == 5
    assert len(read_output_documents(output)) == 5


def test_build_writes_manifest_report_and_dedup_db(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(source, "cc_text_00000.jsonl.gz", [good_record(0)])

    build_pretraining_corpus(source=source, output=output)

    assert (output / MANIFEST_NAME).exists()
    assert (output / REPORT_NAME).exists()
    assert (output / PROGRESS_NAME).exists()
    assert (output / DEDUP_NAME).exists()

    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["format"] == "llm_pretraining_corpus"
    assert manifest["filters"]["language"] == "eng"
    assert manifest["filters"]["near_deduplication"] is False
    # Now a structured block rather than a bare boolean, so a corpus records
    # exactly which evaluation artifact it was protected against.
    assert manifest["filters"]["evaluation_decontamination"] == {
        "enabled": False
    }
    assert manifest["source"]["shards"] == ["cc_text_00000.jsonl.gz"]


def test_output_records_carry_full_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(source, "cc_text_00000.jsonl.gz", [good_record(7)])

    build_pretraining_corpus(source=source, output=output)
    document = read_output_documents(output)[0]

    assert document["source_shard"] == "cc_text_00000.jsonl.gz"
    assert document["source_line"] == 1
    assert document["url"] == "https://example.com/page/7"
    assert document["lang"] == "eng"
    assert len(document["text_sha256"]) == 64
    assert document["document_id"].startswith("cc-")
    assert document["split_group"]


def test_non_english_primary_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [
            good_record(0, lang="eng"),
            good_record(1, lang="eng,spa"),
            good_record(2, lang="spa"),
            good_record(3, lang="deu,eng"),
        ],
    )

    report = build_pretraining_corpus(source=source, output=output)

    assert report["stats"]["documents_accepted"] == 2
    assert report["stats"]["rejected_by_reason"][REASON_NON_ENGLISH] == 2


def test_schema_violations_are_counted_separately(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    path = source / "cc_text_00000.jsonl.gz"
    source.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write("{not valid json\n")
        handle.write(json.dumps({"url": "u", "lang": "eng"}) + "\n")
        handle.write(json.dumps({"text": 42, "lang": "eng"}) + "\n")
        handle.write(json.dumps(good_record(0)) + "\n")

    report = build_pretraining_corpus(source=source, output=output)
    reasons = report["stats"]["rejected_by_reason"]

    assert reasons[REASON_INVALID_JSON] == 1
    assert reasons[REASON_MISSING_TEXT] == 1
    assert reasons[REASON_TEXT_NOT_STRING] == 1
    assert report["stats"]["documents_accepted"] == 1


def test_exact_duplicates_removed_within_a_run(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    duplicate = good_record(0)
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [duplicate, dict(duplicate), good_record(1)],
    )

    report = build_pretraining_corpus(source=source, output=output)

    assert report["stats"]["documents_accepted"] == 2
    assert report["stats"]["rejected_by_reason"][REASON_EXACT_DUPLICATE] == 1


def test_duplicates_removed_across_separate_builds(tmp_path: Path) -> None:
    """The gap the extractor leaves: duplicates across process runs."""

    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(source, "cc_text_00000.jsonl.gz", [good_record(0)])

    first = build_pretraining_corpus(source=source, output=output)
    assert first["stats"]["documents_accepted"] == 1

    # A second build over the same input, reusing the persisted fingerprints.
    write_shard(source, "cc_text_00001.jsonl.gz", [good_record(0)])
    second = build_pretraining_corpus(source=source, output=output, resume=True)

    assert second["stats"]["documents_accepted"] == 0
    assert second["stats"]["rejected_by_reason"][REASON_EXACT_DUPLICATE] == 1


def test_line_ending_variants_deduplicate_after_normalization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    text = document_text("Line ending test. ")
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [
            good_record(0, text=text.replace(" ", " ", 1)),
            good_record(1, text=text.replace("\n", "\r\n")),
        ],
    )

    report = build_pretraining_corpus(source=source, output=output)
    assert report["stats"]["documents_accepted"] == 1


def test_max_records_limits_input_examined(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(i) for i in range(20)],
    )

    report = build_pretraining_corpus(
        source=source,
        output=output,
        max_records=5,
    )
    assert report["stats"]["records_seen"] == 5


def test_max_documents_limits_accepted(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(i) for i in range(20)],
    )

    report = build_pretraining_corpus(
        source=source,
        output=output,
        max_documents=3,
    )
    assert report["stats"]["documents_accepted"] == 3


def test_resume_skips_completed_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(source, "cc_text_00000.jsonl.gz", [good_record(0)])

    build_pretraining_corpus(source=source, output=output)

    write_shard(source, "cc_text_00001.jsonl.gz", [good_record(1)])
    second = build_pretraining_corpus(source=source, output=output, resume=True)

    # Only the new shard is read.
    assert second["stats"]["shards_read"] == 1
    assert second["stats"]["documents_accepted"] == 1
    assert len(read_output_documents(output)) == 2


def test_quality_thresholds_are_honoured(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(0, text="Tiny document, far too short to keep.")],
    )

    strict = build_pretraining_corpus(source=source, output=output)
    assert strict["stats"]["documents_accepted"] == 0

    lenient = build_pretraining_corpus(
        source=source,
        output=tmp_path / "out2",
        thresholds=QualityThresholds(min_characters=5, min_words=3),
    )
    assert lenient["stats"]["documents_accepted"] == 1


def test_rejected_samples_are_retained_for_audit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(i, lang="spa") for i in range(5)],
    )

    report = build_pretraining_corpus(source=source, output=output)
    samples = report["rejected_samples"]

    assert samples
    assert all(sample["reason"] == REASON_NON_ENGLISH for sample in samples)
    assert all(sample["source_shard"] for sample in samples)


def test_documents_are_normalized_in_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(0, text=document_text("CRLF test. ").replace("\n", "\r\n"))],
    )

    build_pretraining_corpus(source=source, output=output)
    document = read_output_documents(output)[0]

    assert "\r" not in document["text"]


# ---------------------------------------------------------------------------
# Crash safety: one input shard is one transaction
# ---------------------------------------------------------------------------


class InjectedCrash(RuntimeError):
    """Stand-in for a process-level failure part-way through an input shard."""


def crash_after(n: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ShardWriter.write raise once it has accepted ``n`` documents."""

    original = ShardWriter.write
    calls = {"count": 0}

    def wrapped(self, document):  # type: ignore[no-untyped-def]
        if calls["count"] >= n:
            raise InjectedCrash("simulated crash mid input shard")
        calls["count"] += 1
        return original(self, document)

    monkeypatch.setattr(ShardWriter, "write", wrapped)


def test_dedup_rollback_after_interrupted_source_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashed shard must leave no fingerprints behind.

    Committed fingerprints for unwritten documents are the failure mode that
    loses data silently on the next run.
    """

    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(i) for i in range(6)],
    )

    crash_after(3, monkeypatch)
    with pytest.raises(InjectedCrash):
        build_pretraining_corpus(source=source, output=output)

    monkeypatch.undo()

    from llm.data.dedup import ExactDeduplicator

    with ExactDeduplicator(output / DEDUP_NAME, read_only=True) as dedup:
        assert dedup.count() == 0


def test_resume_after_interrupted_input_shard_preserves_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every document must appear exactly once after crash and resume."""

    source = tmp_path / "source"
    output = tmp_path / "out"
    records = [good_record(i) for i in range(8)]
    write_shard(source, "cc_text_00000.jsonl.gz", records)

    crash_after(3, monkeypatch)
    with pytest.raises(InjectedCrash):
        build_pretraining_corpus(source=source, output=output)
    monkeypatch.undo()

    build_pretraining_corpus(source=source, output=output, resume=True)

    documents = read_output_documents(output)
    ids = [doc["document_id"] for doc in documents]

    assert len(ids) == 8
    assert len(set(ids)) == 8, "a document was written twice"

    texts = {doc["text"] for doc in documents}
    assert texts == {
        record["text"].replace("\r\n", "\n") for record in records
    }


def test_crash_removes_output_shards_written_by_that_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(i) for i in range(20)],
    )

    # Small shards force several rotations before the crash.
    crash_after(10, monkeypatch)
    with pytest.raises(InjectedCrash):
        build_pretraining_corpus(
            source=source,
            output=output,
            shard_bytes=2_000,
        )
    monkeypatch.undo()

    assert list(output.glob("corpus-*.jsonl.gz")) == []


def test_resume_removes_orphan_output_shards(tmp_path: Path) -> None:
    """A shard left above next_output_index must not survive into the corpus."""

    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(source, "cc_text_00000.jsonl.gz", [good_record(0)])

    build_pretraining_corpus(source=source, output=output)

    progress = json.loads((output / PROGRESS_NAME).read_text(encoding="utf-8"))
    orphan_index = progress["next_output_index"]
    orphan = output / f"corpus-{orphan_index:05d}.jsonl.gz"
    with gzip.open(orphan, "wt", encoding="utf-8") as handle:
        handle.write('{"document_id":"stale","text":"stale"}\n')
    assert orphan.exists()

    write_shard(source, "cc_text_00001.jsonl.gz", [good_record(1)])
    report = build_pretraining_corpus(source=source, output=output, resume=True)

    assert orphan.name in report["orphan_shards_removed"]
    documents = read_output_documents(output)
    assert all(doc["document_id"] != "stale" for doc in documents)
    assert len(documents) == 2


def test_resume_manifest_contains_all_output_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(i) for i in range(5)],
    )
    build_pretraining_corpus(source=source, output=output, shard_bytes=2_000)

    write_shard(
        source,
        "cc_text_00001.jsonl.gz",
        [good_record(i) for i in range(100, 105)],
    )
    build_pretraining_corpus(
        source=source,
        output=output,
        resume=True,
        shard_bytes=2_000,
    )

    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    on_disk = sorted(path.name for path in output.glob("corpus-*.jsonl.gz"))

    assert manifest["output"]["shards"] == on_disk
    assert len(on_disk) > 1, "test needs more than one shard to be meaningful"


def test_resume_manifest_reports_cumulative_document_count(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(i) for i in range(4)],
    )
    build_pretraining_corpus(source=source, output=output)

    write_shard(
        source,
        "cc_text_00001.jsonl.gz",
        [good_record(i) for i in range(100, 107)],
    )
    report = build_pretraining_corpus(source=source, output=output, resume=True)

    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))

    # The second invocation only accepted 7 documents...
    assert report["stats"]["documents_accepted"] == 7
    # ...but the corpus contains 11, and the manifest must say so.
    assert manifest["output"]["documents"] == 11
    assert manifest["result"]["documents"] == 11
    assert len(read_output_documents(output)) == 11


def test_resume_totals_include_rejections_from_earlier_runs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(i, lang="spa") for i in range(3)],
    )
    build_pretraining_corpus(source=source, output=output)

    write_shard(
        source,
        "cc_text_00001.jsonl.gz",
        [good_record(i, lang="spa") for i in range(100, 102)],
    )
    report = build_pretraining_corpus(source=source, output=output, resume=True)

    assert report["stats"]["rejected_by_reason"][REASON_NON_ENGLISH] == 2
    assert report["totals"]["rejected_by_reason"][REASON_NON_ENGLISH] == 5
    assert report["totals"]["records_seen"] == 5


def test_limit_stop_records_partial_shard_and_resume_continues(
    tmp_path: Path,
) -> None:
    """max_records may halt mid-shard; resuming must not lose or duplicate."""

    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(i) for i in range(10)],
    )

    first = build_pretraining_corpus(
        source=source,
        output=output,
        max_records=4,
    )
    assert first["stats"]["documents_accepted"] == 4

    progress = json.loads((output / PROGRESS_NAME).read_text(encoding="utf-8"))
    assert progress["partial_shard"]["name"] == "cc_text_00000.jsonl.gz"
    assert progress["partial_shard"]["next_line"] == 4

    second = build_pretraining_corpus(source=source, output=output, resume=True)

    assert second["stats"]["documents_accepted"] == 6
    documents = read_output_documents(output)
    assert len(documents) == 10
    assert len({doc["document_id"] for doc in documents}) == 10


def test_progress_file_carries_totals(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(i) for i in range(3)],
    )
    build_pretraining_corpus(source=source, output=output)

    progress = json.loads((output / PROGRESS_NAME).read_text(encoding="utf-8"))
    assert progress["totals"]["documents_accepted"] == 3
    assert progress["totals"]["records_seen"] == 3
    assert progress["totals"]["accepted_utf8_bytes"] > 0


def test_shard_index_parsing() -> None:
    assert shard_index("corpus-00007.jsonl.gz") == 7
    assert shard_index(Path("a/b/corpus-00000.jsonl.gz")) == 0
    assert shard_index("manifest.json") is None


def test_remove_output_shards_from(tmp_path: Path) -> None:
    for index in range(4):
        (tmp_path / f"corpus-{index:05d}.jsonl.gz").write_bytes(b"")

    removed = remove_output_shards_from(tmp_path, from_index=2)

    assert removed == ["corpus-00002.jsonl.gz", "corpus-00003.jsonl.gz"]
    assert sorted(p.name for p in tmp_path.glob("corpus-*")) == [
        "corpus-00000.jsonl.gz",
        "corpus-00001.jsonl.gz",
    ]


def test_accepted_bytes_match_written_documents(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    write_shard(
        source,
        "cc_text_00000.jsonl.gz",
        [good_record(i) for i in range(4)],
    )

    report = build_pretraining_corpus(source=source, output=output)
    documents = read_output_documents(output)

    total = sum(len(doc["text"].encode("utf-8")) for doc in documents)
    assert report["stats"]["accepted_utf8_bytes"] == total
