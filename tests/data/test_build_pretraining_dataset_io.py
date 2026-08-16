"""Reader tests for the pretraining dataset builder.

The cleaned corpus is written as ``corpus-*.jsonl.gz``.  Before this support
existed, ``Path.suffix`` reported ``.gz`` and the corpus looked like an
unsupported file type, so the tokenizer could not consume it at all.

The central property proved here is **transparency**: a plain file and its
gzipped copy must yield identical documents, identical ids, and identical
split groups.  Anything less would make compression a silent variable in the
training data.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from build_pretraining_dataset import (
    SUPPORTED_SUFFIXES,
    discover_input_files,
    is_compressed,
    iter_training_documents,
    logical_suffix,
    open_text_file,
)


RECORDS = [
    {
        "document_id": "cc-000000000000000000000001",
        "text": "The regulator holds its output within tolerance as the load varies.",
        "split_group": "group-a",
        "url": "https://example.com/1",
    },
    {
        "document_id": "cc-000000000000000000000002",
        "text": "Feedback compensation determines how quickly the loop settles.",
        "split_group": "group-b",
        "url": "https://example.com/2",
    },
    {
        "document_id": "cc-000000000000000000000003",
        "text": "Layout parasitics dominate switching losses at higher frequencies.",
        "split_group": "group-a",
        "url": "https://example.com/3",
    },
]


def write_plain(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def write_gzipped(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def as_tuples(documents) -> list[tuple[str, str, str]]:
    return [(d.document_id, d.text, d.split_group) for d in documents]


# ---------------------------------------------------------------------------
# Suffix handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("corpus-00000.jsonl", ".jsonl"),
        ("corpus-00000.jsonl.gz", ".jsonl"),
        ("data.ndjson", ".ndjson"),
        ("data.ndjson.gz", ".ndjson"),
        ("notes.txt", ".txt"),
        ("notes.txt.gz", ".txt"),
        ("readme.md", ".md"),
        ("archive.gz", ""),
        ("plain", ""),
        ("v0.1.jsonl", ".jsonl"),
        ("v0.1.jsonl.gz", ".jsonl"),
    ],
)
def test_logical_suffix(name: str, expected: str) -> None:
    assert logical_suffix(Path(name)) == expected


def test_logical_suffix_is_case_insensitive() -> None:
    assert logical_suffix(Path("CORPUS.JSONL.GZ")) == ".jsonl"


def test_is_compressed() -> None:
    assert is_compressed(Path("a.jsonl.gz"))
    assert not is_compressed(Path("a.jsonl"))


def test_every_supported_suffix_is_also_supported_compressed() -> None:
    for suffix in SUPPORTED_SUFFIXES:
        assert logical_suffix(Path(f"file{suffix}.gz")) == suffix


# ---------------------------------------------------------------------------
# open_text_file
# ---------------------------------------------------------------------------


def test_open_text_file_reads_both_forms(tmp_path: Path) -> None:
    content = "first line\nsecond line\n"

    plain = tmp_path / "a.jsonl"
    plain.write_text(content, encoding="utf-8", newline="\n")

    compressed = tmp_path / "b.jsonl.gz"
    with gzip.open(compressed, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(content)

    with open_text_file(plain) as handle:
        assert handle.read() == content
    with open_text_file(compressed) as handle:
        assert handle.read() == content


def test_open_text_file_rejects_invalid_utf8(tmp_path: Path) -> None:
    """Strict decoding: mojibake must not reach the token stream silently."""

    path = tmp_path / "bad.jsonl.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(b"\xff\xfe not valid utf-8\n")

    with pytest.raises(UnicodeDecodeError):
        with open_text_file(path) as handle:
            handle.read()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_finds_gzipped_corpus_shards(tmp_path: Path) -> None:
    write_gzipped(tmp_path / "corpus-00000.jsonl.gz", RECORDS)
    write_gzipped(tmp_path / "corpus-00001.jsonl.gz", RECORDS)

    found = discover_input_files([str(tmp_path)])
    assert [p.name for p in found] == [
        "corpus-00000.jsonl.gz",
        "corpus-00001.jsonl.gz",
    ]


def test_discover_accepts_a_single_gzipped_file(tmp_path: Path) -> None:
    path = write_gzipped(tmp_path / "corpus-00000.jsonl.gz", RECORDS)
    assert discover_input_files([str(path)]) == [path.resolve()]


def test_discover_ignores_unrelated_gzip_files(tmp_path: Path) -> None:
    write_gzipped(tmp_path / "corpus-00000.jsonl.gz", RECORDS)
    (tmp_path / "dedup.sqlite.gz").write_bytes(b"")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    found = discover_input_files([str(tmp_path)])
    assert [p.name for p in found] == ["corpus-00000.jsonl.gz"]


def test_discover_rejects_unsupported_single_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported input file type"):
        discover_input_files([str(path)])


def test_discover_mixes_compressed_and_plain(tmp_path: Path) -> None:
    write_plain(tmp_path / "a.jsonl", RECORDS)
    write_gzipped(tmp_path / "b.jsonl.gz", RECORDS)

    found = discover_input_files([str(tmp_path)])
    assert [p.name for p in found] == ["a.jsonl", "b.jsonl.gz"]


# ---------------------------------------------------------------------------
# The transparency property
# ---------------------------------------------------------------------------


def test_gzipped_and_plain_jsonl_produce_identical_documents(
    tmp_path: Path,
) -> None:
    plain = write_plain(tmp_path / "plain" / "corpus.jsonl", RECORDS)
    compressed = write_gzipped(tmp_path / "gz" / "corpus.jsonl.gz", RECORDS)

    from_plain = as_tuples(iter_training_documents([plain], text_field="text"))
    from_gzip = as_tuples(
        iter_training_documents([compressed], text_field="text")
    )

    assert from_plain == from_gzip
    assert len(from_plain) == 3


def test_gzipped_and_plain_text_files_produce_identical_documents(
    tmp_path: Path,
) -> None:
    content = "A paragraph of ordinary technical prose used as one document.\n"

    plain = tmp_path / "doc.txt"
    plain.write_text(content, encoding="utf-8", newline="\n")

    compressed = tmp_path / "doc.txt.gz"
    with gzip.open(compressed, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(content)

    [from_plain] = list(iter_training_documents([plain], text_field="text"))
    [from_gzip] = list(iter_training_documents([compressed], text_field="text"))

    assert from_plain.text == from_gzip.text
    # Ids derive from the path, so they legitimately differ between the two.
    assert from_plain.document_id != from_gzip.document_id


def test_provenance_survives_compression(tmp_path: Path) -> None:
    """document_id and split_group must come from the record, not the file."""

    compressed = write_gzipped(tmp_path / "corpus-00000.jsonl.gz", RECORDS)
    documents = list(iter_training_documents([compressed], text_field="text"))

    assert [d.document_id for d in documents] == [
        r["document_id"] for r in RECORDS
    ]
    assert [d.split_group for d in documents] == ["group-a", "group-b", "group-a"]


def test_cleaned_corpus_record_shape_is_consumable(tmp_path: Path) -> None:
    """A record exactly as pretraining_corpus.py writes it."""

    record = {
        "document_id": "cc-abcdef0123456789abcdef01",
        "text": "Cleaned document text from the pretraining corpus.",
        "url": "https://example.com/page",
        "date": "2026-08-14T00:00:00Z",
        "lang": "eng",
        "source_shard": "cc_text_00003.jsonl.gz",
        "source_line": 4821,
        "text_sha256": "0" * 64,
        "split_group": "a1b2c3d4e5f60718",
    }
    path = write_gzipped(tmp_path / "corpus-00000.jsonl.gz", [record])

    [document] = list(iter_training_documents([path], text_field="text"))

    assert document.document_id == record["document_id"]
    assert document.split_group == record["split_group"]
    assert document.text == record["text"]


def test_empty_documents_are_skipped_in_both_forms(tmp_path: Path) -> None:
    records = [
        {"document_id": "a", "text": "", "split_group": "g"},
        {"document_id": "b", "text": "Real content here.", "split_group": "g"},
    ]
    plain = write_plain(tmp_path / "a.jsonl", records)
    compressed = write_gzipped(tmp_path / "b.jsonl.gz", records)

    assert len(list(iter_training_documents([plain], text_field="text"))) == 1
    assert len(list(iter_training_documents([compressed], text_field="text"))) == 1


def test_whitespace_only_documents_are_not_skipped(tmp_path: Path) -> None:
    """Documents the actual contract, which is easy to assume wrongly.

    The skip test is ``normalize_text(text) == ""``, and normalization
    deliberately preserves whitespace, so a whitespace-only record survives
    here.  It never reaches this stage in practice because the corpus builder
    rejects empty and undersized documents upstream.
    """

    record = {"document_id": "a", "text": "   \n  ", "split_group": "g"}
    path = write_gzipped(tmp_path / "a.jsonl.gz", [record])

    assert len(list(iter_training_documents([path], text_field="text"))) == 1


def test_malformed_json_in_gzip_reports_the_line(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(RECORDS[0]) + "\n")
        handle.write("{ not json\n")

    with pytest.raises(ValueError, match="line 2"):
        list(iter_training_documents([path], text_field="text"))


def test_multi_member_gzip_is_read_fully(tmp_path: Path) -> None:
    """Concatenated gzip members must all be read, as with the extractor."""

    path = tmp_path / "corpus.jsonl.gz"
    first = gzip.compress((json.dumps(RECORDS[0]) + "\n").encode("utf-8"))
    second = gzip.compress((json.dumps(RECORDS[1]) + "\n").encode("utf-8"))
    path.write_bytes(first + second)

    documents = list(iter_training_documents([path], text_field="text"))
    assert [d.document_id for d in documents] == [
        RECORDS[0]["document_id"],
        RECORDS[1]["document_id"],
    ]
