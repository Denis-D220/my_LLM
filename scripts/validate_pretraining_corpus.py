r"""Validate a cleaned pretraining corpus, then optionally freeze it.

This script is the gate between corpus construction and tokenization.  It is
read-only with respect to corpus data: it never deletes, rewrites, or reorders
a document.  The only file it may create is the freeze stamp, and only when
every check passed.

    python scripts\validate_pretraining_corpus.py `
        --corpus data\cleaned\pretraining\v0.1 `
        --expected-evaluation-sha256 6c994e255976ea692251cee0d0d43218dff52ae6f958d706e7a084c39dd75452 `
        --report data\audits\corpus-v0.1-validation.json `
        --freeze

What is actually being proved
-----------------------------
The manifest is a *claim* written by the builder.  A claim that agrees only
with itself proves nothing, so every headline number is recomputed from the
shard bytes and then compared against all three independent records of it:

    manifest.json      what the builder declared when it finished
    progress.json      what the builder's resumable counter accumulated
    dedup.sqlite       what the exact-dedup index actually stored

Agreement across those three plus the recomputed value is meaningful, because
they are produced by different code paths at different times.

Derived-field checks matter more than they look
-----------------------------------------------
``document_id`` and ``split_group`` are *derived* values::

    document_id  = make_document_id(text_sha256)
    split_group  = make_split_group(url, document_id)

They are recomputed here rather than trusted.  ``split_group`` decides which
side of the train/validation split a document lands on, so silent drift in
that function would not corrupt the corpus at all -- it would corrupt the
*split*, months later, in a way no downstream check would attribute to this
stage.

Uniqueness without holding the corpus in memory
-----------------------------------------------
``document_id`` is ``cc-`` plus the first 24 hex characters of the text digest,
so a document-id collision is exactly a 12-byte digest-prefix collision.  One
set of 12-byte prefixes therefore proves both uniqueness invariants at once,
and prefix uniqueness implies full-digest uniqueness because the prefix is a
function of the digest.  Peak cost is roughly 80 MB for a 1.2M-document corpus
rather than the corpus text itself.

The dedup cross-check then closes both directions without a second set:

    every document digest is present in dedup.sqlite   (streamed lookups)
    document digests are pairwise distinct             (prefix set)
    len(documents) == len(exact_hashes)                (counts)
        => the two sets are equal

Cost
----
Every byte of every shard is decompressed, decoded, re-hashed, and (unless
--skip-normalization-check) re-normalized.  For the v0.1 corpus that is about
5.5 GB of JSON carrying 5.1 GB of UTF-8 text, so budget tens of minutes.  This
runs once per corpus version.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Iterator, Sequence

from llm.data.pretraining_corpus import make_document_id, make_split_group
from llm.tokenizer.normalizer import normalize_text


VALIDATOR_FORMAT = "llm_pretraining_corpus_validation"
VALIDATOR_FORMAT_VERSION = 1

FREEZE_FORMAT = "llm_pretraining_corpus_freeze"
FREEZE_FORMAT_VERSION = 1
FREEZE_FILENAME = "FROZEN.json"

DEFAULT_SHARD_PREFIX = "corpus"
SHARD_PATTERN = re.compile(r"^corpus-(\d{5})\.jsonl\.gz$")

MANIFEST_FILENAME = "manifest.json"
PROGRESS_FILENAME = "progress.json"
BUILD_REPORT_FILENAME = "build_report.json"
DEDUP_FILENAME = "dedup.sqlite"

# Field order is part of the on-disk contract: CleanDocument.to_record() emits
# these keys in this order, and the canonical-line check below depends on it.
EXPECTED_RECORD_FIELDS = (
    "document_id",
    "text",
    "url",
    "date",
    "lang",
    "source_shard",
    "source_line",
    "text_sha256",
    "split_group",
)

EXPECTED_NORMALIZATION = {
    "module": "llm.tokenizer.normalizer.normalize_text",
    "encoding": "UTF-8",
    "unicode_form": "NFC",
    "preserve_case": True,
    "line_endings": "LF",
}

SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIP"
STATUS_WARN = "WARN"

READ_CHUNK = 1 << 20
PROGRESS_INTERVAL = 100_000
MAX_REPORTED_PROBLEMS = 50


# --------------------------------------------------------------------------
# check bookkeeping
# --------------------------------------------------------------------------


@dataclass
class Check:
    """One named invariant and its outcome."""

    name: str
    status: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "check": self.name,
            "status": self.status,
            "detail": self.detail,
        }
        if self.data:
            payload["data"] = self.data
        return payload


class CheckLog:
    """Ordered checks plus the pass/fail verdict over all of them."""

    def __init__(self, *, quiet: bool = False) -> None:
        self._checks: list[Check] = []
        self._quiet = quiet

    def record(
        self,
        name: str,
        status: str,
        detail: str,
        data: dict[str, Any] | None = None,
    ) -> Check:
        check = Check(name=name, status=status, detail=detail, data=data or {})
        self._checks.append(check)
        if not self._quiet:
            print(f"  [{status:4}] {name}: {detail}")
        return check

    def ok(self, name: str, detail: str, data: dict[str, Any] | None = None) -> Check:
        return self.record(name, STATUS_PASS, detail, data)

    def fail(self, name: str, detail: str, data: dict[str, Any] | None = None) -> Check:
        return self.record(name, STATUS_FAIL, detail, data)

    def skip(self, name: str, detail: str, data: dict[str, Any] | None = None) -> Check:
        return self.record(name, STATUS_SKIP, detail, data)

    def warn(self, name: str, detail: str, data: dict[str, Any] | None = None) -> Check:
        return self.record(name, STATUS_WARN, detail, data)

    def verdict(self, name: str, failures: Sequence[str], detail_ok: str) -> Check:
        if failures:
            return self.fail(
                name,
                f"{len(failures)} problem(s); first: {failures[0]}",
                {"problems": list(failures[:MAX_REPORTED_PROBLEMS])},
            )
        return self.ok(name, detail_ok)

    @property
    def checks(self) -> list[Check]:
        return list(self._checks)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self._checks if c.status == STATUS_FAIL]

    @property
    def skipped(self) -> list[Check]:
        return [c for c in self._checks if c.status == STATUS_SKIP]

    @property
    def warned(self) -> list[Check]:
        return [c for c in self._checks if c.status == STATUS_WARN]

    @property
    def passed(self) -> bool:
        return not self.failed


# --------------------------------------------------------------------------
# streaming helpers
# --------------------------------------------------------------------------


class _HashingReader:
    """Feed a file to gzip while hashing the compressed bytes in one pass.

    Computing the shard's on-disk SHA-256 separately would mean reading 1.9 GB
    twice.  This wrapper sits between the raw file and :class:`gzip.GzipFile`
    so the freeze stamp's digests cost nothing extra.
    """

    def __init__(self, handle: io.BufferedReader) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self._handle.read(size)
        if chunk:
            self._digest.update(chunk)
        return chunk

    def readable(self) -> bool:
        return True

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(READ_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_mapping(value: Any) -> dict[str, Any]:
    """Coerce a manifest slot to a dict.

    Older corpus manifests wrote ``"evaluation_decontamination": false`` where
    later ones write a settings object.  Treating the disabled form as an empty
    mapping keeps the validator usable against every corpus version instead of
    crashing on the trial builds.
    """

    return value if isinstance(value, dict) else {}


def evaluation_settings(manifest: dict[str, Any]) -> dict[str, Any]:
    return as_mapping(as_mapping(manifest.get("filters")).get("evaluation_decontamination"))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def sha256_hex(value: str) -> str:
    lowered = value.strip().lower()
    if not SHA256_HEX.match(lowered):
        raise argparse.ArgumentTypeError("must be 64 lowercase hex characters")
    return lowered


# --------------------------------------------------------------------------
# per-shard streaming statistics
# --------------------------------------------------------------------------


@dataclass
class ShardStats:
    """Everything recomputed from one shard's bytes."""

    name: str
    compressed_bytes: int
    compressed_sha256: str
    uncompressed_bytes: int
    documents: int
    characters: int
    utf8_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "compressed_bytes": self.compressed_bytes,
            "compressed_sha256": self.compressed_sha256,
            "uncompressed_bytes": self.uncompressed_bytes,
            "documents": self.documents,
            "characters": self.characters,
            "utf8_bytes": self.utf8_bytes,
        }


@dataclass
class ScanTotals:
    """Corpus-wide accumulators from the single streaming pass."""

    documents: int = 0
    characters: int = 0
    utf8_bytes: int = 0
    uncompressed_bytes: int = 0
    shards: list[ShardStats] = field(default_factory=list)

    problems_json: list[str] = field(default_factory=list)
    problems_fields: list[str] = field(default_factory=list)
    problems_canonical: list[str] = field(default_factory=list)
    problems_text_hash: list[str] = field(default_factory=list)
    problems_document_id: list[str] = field(default_factory=list)
    problems_split_group: list[str] = field(default_factory=list)
    problems_normalization: list[str] = field(default_factory=list)
    problems_uniqueness: list[str] = field(default_factory=list)
    problems_dedup: list[str] = field(default_factory=list)

    def note(self, bucket: list[str], message: str) -> None:
        if len(bucket) < MAX_REPORTED_PROBLEMS:
            bucket.append(message)


def _check_record_fields(record: Any, location: str, totals: ScanTotals) -> bool:
    """Validate the record shape.  Returns False when it is unusable."""

    if not isinstance(record, dict):
        totals.note(
            totals.problems_fields,
            f"{location}: record is {type(record).__name__}, expected object",
        )
        return False

    keys = list(record.keys())
    if keys != list(EXPECTED_RECORD_FIELDS):
        missing = [f for f in EXPECTED_RECORD_FIELDS if f not in record]
        extra = [f for f in keys if f not in EXPECTED_RECORD_FIELDS]
        if missing or extra:
            totals.note(
                totals.problems_fields,
                f"{location}: field mismatch (missing={missing}, extra={extra})",
            )
            return False
        totals.note(
            totals.problems_fields,
            f"{location}: fields are correct but out of order: {keys}",
        )

    usable = True
    text = record.get("text")
    if not isinstance(text, str):
        totals.note(
            totals.problems_fields,
            f"{location}: text is {type(text).__name__}, expected str",
        )
        usable = False

    for key in ("document_id", "source_shard", "text_sha256", "split_group"):
        value = record.get(key)
        if not isinstance(value, str) or not value:
            totals.note(
                totals.problems_fields,
                f"{location}: {key} must be a non-empty string, got {value!r}",
            )
            usable = False

    for key in ("url", "date", "lang"):
        value = record.get(key)
        if value is not None and not isinstance(value, str):
            totals.note(
                totals.problems_fields,
                f"{location}: {key} must be a string or null, got {type(value).__name__}",
            )

    source_line = record.get("source_line")
    if not isinstance(source_line, int) or isinstance(source_line, bool):
        totals.note(
            totals.problems_fields,
            f"{location}: source_line must be an integer, got {source_line!r}",
        )

    digest = record.get("text_sha256")
    if isinstance(digest, str) and not SHA256_HEX.match(digest):
        totals.note(
            totals.problems_fields,
            f"{location}: text_sha256 is not 64 lowercase hex characters",
        )
        usable = False

    return usable


def scan_shard(
    path: Path,
    totals: ScanTotals,
    seen_prefixes: set[bytes],
    dedup: sqlite3.Connection | None,
    *,
    check_normalization: bool,
    check_canonical: bool,
    max_documents: int | None,
    quiet: bool,
) -> bool:
    """Stream one shard.  Returns False when the document limit was reached."""

    shard_documents = 0
    shard_characters = 0
    shard_utf8_bytes = 0
    shard_uncompressed = 0
    compressed_bytes = path.stat().st_size

    with path.open("rb") as raw:
        reader = _HashingReader(raw)
        with gzip.GzipFile(fileobj=reader, mode="rb") as gz:
            for line_number, raw_line in enumerate(gz, start=1):
                shard_uncompressed += len(raw_line)
                location = f"{path.name}:{line_number}"

                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    totals.note(
                        totals.problems_json, f"{location}: invalid UTF-8 ({exc})"
                    )
                    continue

                stripped = line.rstrip("\n")
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    totals.note(totals.problems_json, f"{location}: {exc}")
                    continue

                shard_documents += 1

                if not _check_record_fields(record, location, totals):
                    continue

                text: str = record["text"]
                declared_digest: str = record["text_sha256"]
                document_id: str = record["document_id"]
                split_group: str = record["split_group"]
                url = record["url"]

                if check_canonical:
                    canonical = json.dumps(
                        record, ensure_ascii=False, separators=(",", ":")
                    )
                    if canonical != stripped:
                        totals.note(
                            totals.problems_canonical,
                            f"{location}: line is not the canonical serialization "
                            "of its own record",
                        )

                encoded = text.encode("utf-8")
                shard_characters += len(text)
                shard_utf8_bytes += len(encoded)

                actual_digest = hashlib.sha256(encoded).hexdigest()
                if actual_digest != declared_digest:
                    totals.note(
                        totals.problems_text_hash,
                        f"{location}: text_sha256 declared {declared_digest}, "
                        f"recomputed {actual_digest}",
                    )

                expected_id = make_document_id(actual_digest)
                if document_id != expected_id:
                    totals.note(
                        totals.problems_document_id,
                        f"{location}: document_id {document_id!r} is not "
                        f"make_document_id(text_sha256) = {expected_id!r}",
                    )

                expected_group = make_split_group(url, document_id)
                if split_group != expected_group:
                    totals.note(
                        totals.problems_split_group,
                        f"{location}: split_group {split_group!r} is not "
                        f"make_split_group(url, document_id) = {expected_group!r}",
                    )

                if check_normalization:
                    # Idempotence is the drift test: stored text must already be
                    # a fixed point of the normalizer the tokenizer will apply.
                    try:
                        if normalize_text(text) != text:
                            totals.note(
                                totals.problems_normalization,
                                f"{location}: stored text is not normalizer-stable",
                            )
                    except (TypeError, ValueError) as exc:
                        totals.note(
                            totals.problems_normalization,
                            f"{location}: normalize_text raised {exc!r}",
                        )

                prefix = bytes.fromhex(declared_digest[:24])
                if prefix in seen_prefixes:
                    totals.note(
                        totals.problems_uniqueness,
                        f"{location}: duplicate document_id/text digest "
                        f"({document_id})",
                    )
                else:
                    seen_prefixes.add(prefix)

                if dedup is not None:
                    row = dedup.execute(
                        "SELECT 1 FROM exact_hashes WHERE sha256 = ? LIMIT 1",
                        (bytes.fromhex(declared_digest),),
                    ).fetchone()
                    if row is None:
                        totals.note(
                            totals.problems_dedup,
                            f"{location}: {document_id} has no fingerprint in "
                            f"{DEDUP_FILENAME}",
                        )

                if (
                    not quiet
                    and (totals.documents + shard_documents) % PROGRESS_INTERVAL == 0
                ):
                    scanned = totals.documents + shard_documents
                    print(f"         ... {scanned:,} documents", flush=True)

                if (
                    max_documents is not None
                    and totals.documents + shard_documents >= max_documents
                ):
                    break

            compressed_sha256 = reader.hexdigest()

    # The digest above covers only the bytes gzip consumed.  When a document
    # limit stops the scan early that is a prefix of the file, so the shard
    # digest is meaningful only for a complete read.
    limited = (
        max_documents is not None
        and totals.documents + shard_documents >= max_documents
    )
    if limited:
        compressed_sha256 = ""

    totals.shards.append(
        ShardStats(
            name=path.name,
            compressed_bytes=compressed_bytes,
            compressed_sha256=compressed_sha256,
            uncompressed_bytes=shard_uncompressed,
            documents=shard_documents,
            characters=shard_characters,
            utf8_bytes=shard_utf8_bytes,
        )
    )
    totals.documents += shard_documents
    totals.characters += shard_characters
    totals.utf8_bytes += shard_utf8_bytes
    totals.uncompressed_bytes += shard_uncompressed

    return not limited


# --------------------------------------------------------------------------
# structural checks
# --------------------------------------------------------------------------


def discover_shards(corpus: Path, log: CheckLog) -> list[Path]:
    """Return shard paths ordered by index, checking contiguity and strays."""

    indexed: list[tuple[int, Path]] = []
    unexpected: list[str] = []
    sqlite_sidecars: list[str] = []

    allowed = {
        MANIFEST_FILENAME,
        PROGRESS_FILENAME,
        BUILD_REPORT_FILENAME,
        DEDUP_FILENAME,
        FREEZE_FILENAME,
    }

    for entry in sorted(corpus.iterdir()):
        if entry.is_dir():
            unexpected.append(f"{entry.name}/ (directory)")
            continue
        match = SHARD_PATTERN.match(entry.name)
        if match:
            indexed.append((int(match.group(1)), entry))
        elif entry.name in allowed:
            continue
        elif entry.name.startswith(f"{DEDUP_FILENAME}-"):
            sqlite_sidecars.append(entry.name)
        else:
            unexpected.append(entry.name)

    indexed.sort(key=lambda item: item[0])
    observed = [index for index, _ in indexed]

    if not observed:
        log.fail("shards.present", f"no corpus-*.jsonl.gz files in {corpus}")
        return []

    expected = list(range(len(observed)))
    if observed == expected:
        log.ok(
            "shards.contiguous",
            f"{len(observed)} shards, indices 00000-{len(observed) - 1:05d}, no gaps",
        )
    else:
        gaps = sorted(set(expected) - set(observed))
        log.fail(
            "shards.contiguous",
            f"indices are not contiguous from 0 (missing {gaps[:10]})",
            {"observed_count": len(observed), "missing": gaps[:MAX_REPORTED_PROBLEMS]},
        )

    if unexpected:
        log.fail(
            "shards.no_stray_files",
            f"{len(unexpected)} unexpected entries in the corpus directory",
            {"entries": unexpected[:MAX_REPORTED_PROBLEMS]},
        )
    else:
        log.ok("shards.no_stray_files", "no unexpected files in the corpus directory")

    if sqlite_sidecars:
        log.warn(
            "dedup.clean_shutdown",
            f"SQLite sidecar files present ({', '.join(sqlite_sidecars)}); the "
            "dedup database may not have been closed cleanly",
            {"files": sqlite_sidecars},
        )

    return [path for _, path in indexed]


def check_inputs_complete(
    manifest: dict[str, Any], progress: dict[str, Any], log: CheckLog
) -> None:
    source = manifest.get("source", {})
    expected = source.get("shards")
    completed = source.get("completed_shards")
    progress_completed = progress.get("completed_shards")

    if not isinstance(expected, list) or not expected:
        log.fail("inputs.declared", "manifest.source.shards is missing or empty")
        return

    if completed == expected:
        log.ok(
            "inputs.all_completed",
            f"all {len(expected)} Common Crawl input shards completed",
        )
    else:
        missing = [s for s in expected if s not in (completed or [])]
        log.fail(
            "inputs.all_completed",
            f"{len(missing)} declared input shard(s) never completed",
            {"missing": missing[:MAX_REPORTED_PROBLEMS]},
        )

    if progress_completed == completed:
        log.ok("inputs.progress_agrees", "progress.json lists the same completed inputs")
    else:
        log.fail(
            "inputs.progress_agrees",
            "progress.completed_shards disagrees with manifest.source.completed_shards",
        )

    partial = progress.get("partial_shard")
    if partial is None:
        log.ok("inputs.no_partial_shard", "progress.partial_shard is null")
    else:
        log.fail(
            "inputs.no_partial_shard",
            f"a partial input shard remains: {partial!r}",
            {"partial_shard": partial},
        )


def check_shard_listing(
    manifest: dict[str, Any],
    progress: dict[str, Any],
    shards: Sequence[Path],
    log: CheckLog,
) -> None:
    declared = manifest.get("output", {}).get("shards")
    observed = [path.name for path in shards]

    if declared == observed:
        log.ok("shards.match_manifest", f"{len(observed)} shards match manifest.output.shards")
    else:
        only_disk = [name for name in observed if name not in (declared or [])]
        only_manifest = [name for name in (declared or []) if name not in observed]
        log.fail(
            "shards.match_manifest",
            "on-disk shards differ from manifest.output.shards",
            {
                "on_disk_only": only_disk[:MAX_REPORTED_PROBLEMS],
                "manifest_only": only_manifest[:MAX_REPORTED_PROBLEMS],
            },
        )

    next_index = progress.get("next_output_index")
    if next_index == len(observed):
        log.ok(
            "shards.next_output_index",
            f"progress.next_output_index == {next_index} == shard count",
        )
    else:
        log.fail(
            "shards.next_output_index",
            f"progress.next_output_index is {next_index!r} but {len(observed)} "
            "shards exist",
        )


def check_configuration(
    manifest: dict[str, Any],
    progress: dict[str, Any],
    build_report: dict[str, Any] | None,
    log: CheckLog,
) -> None:
    normalization = manifest.get("normalization", {})
    drift = {
        key: (value, normalization.get(key))
        for key, value in EXPECTED_NORMALIZATION.items()
        if normalization.get(key) != value
    }
    if drift:
        log.fail(
            "config.normalization_policy",
            f"{len(drift)} normalization field(s) differ from the frozen policy",
            {"expected_vs_actual": {k: list(v) for k, v in drift.items()}},
        )
    else:
        log.ok(
            "config.normalization_policy",
            "NFC, case preserved, LF line endings, project normalizer",
        )

    manifest_decon = manifest.get("filters", {}).get("evaluation_decontamination")
    progress_decon = progress.get("decontamination")
    if manifest_decon == progress_decon:
        log.ok(
            "config.decontamination_agrees",
            "manifest and progress declare identical decontamination settings",
        )
    else:
        log.fail(
            "config.decontamination_agrees",
            "decontamination settings differ between manifest.json and progress.json",
            {"manifest": manifest_decon, "progress": progress_decon},
        )

    if build_report is None:
        log.skip("config.build_report_totals", f"{BUILD_REPORT_FILENAME} not present")
        return

    report_totals = build_report.get("totals")
    progress_totals = progress.get("totals")
    if report_totals == progress_totals:
        log.ok("config.build_report_totals", "build_report totals match progress totals")
    else:
        log.fail(
            "config.build_report_totals",
            "build_report.totals disagrees with progress.totals",
        )


def check_evaluation_digest(
    manifest: dict[str, Any],
    progress: dict[str, Any],
    expected: str | None,
    evaluation_file: Path | None,
    log: CheckLog,
) -> None:
    manifest_digest = evaluation_settings(manifest).get("evaluation_sha256")
    progress_digest = as_mapping(progress.get("decontamination")).get("evaluation_sha256")

    if manifest_digest is None and progress_digest is None:
        log.fail(
            "evaluation.digest_internally_consistent",
            "this corpus records no evaluation decontamination at all; it must "
            "not be used to train a model that will be evaluated",
        )
    elif manifest_digest and manifest_digest == progress_digest:
        log.ok(
            "evaluation.digest_internally_consistent",
            f"manifest and progress agree: {manifest_digest[:16]}...",
        )
    else:
        log.fail(
            "evaluation.digest_internally_consistent",
            f"manifest={manifest_digest!r} progress={progress_digest!r}",
        )

    if expected is None:
        log.skip(
            "evaluation.digest_expected",
            "--expected-evaluation-sha256 not supplied",
        )
    elif manifest_digest == expected:
        log.ok("evaluation.digest_expected", f"matches the pinned digest {expected[:16]}...")
    else:
        log.fail(
            "evaluation.digest_expected",
            f"corpus was decontaminated against {manifest_digest!r}, expected {expected!r}",
        )

    if evaluation_file is None:
        log.skip(
            "evaluation.file_digest",
            "--evaluation not supplied; the evaluation file itself was not hashed",
        )
        return

    if not evaluation_file.is_file():
        log.fail("evaluation.file_digest", f"evaluation file not found: {evaluation_file}")
        return

    actual = sha256_file(evaluation_file)
    if actual == manifest_digest:
        log.ok(
            "evaluation.file_digest",
            f"{evaluation_file.name} still hashes to the digest recorded at build time",
        )
    else:
        log.fail(
            "evaluation.file_digest",
            f"{evaluation_file.name} now hashes to {actual}, but the corpus was "
            f"decontaminated against {manifest_digest!r}",
        )


def check_counts(
    manifest: dict[str, Any],
    progress: dict[str, Any],
    totals: ScanTotals,
    dedup_fingerprints: int | None,
    log: CheckLog,
) -> None:
    sources = {
        "recomputed": totals.documents,
        "manifest.output.documents": manifest.get("output", {}).get("documents"),
        "manifest.result.documents": manifest.get("result", {}).get("documents"),
        "progress.documents_accepted": progress.get("totals", {}).get(
            "documents_accepted"
        ),
    }
    if dedup_fingerprints is not None:
        sources["dedup.fingerprints_stored"] = dedup_fingerprints
        declared_fingerprints = manifest.get("dedup", {}).get("fingerprints")
        if declared_fingerprints is not None:
            sources["manifest.dedup.fingerprints"] = declared_fingerprints

    distinct = {value for value in sources.values()}
    if len(distinct) == 1 and totals.documents > 0:
        log.ok(
            "counts.documents",
            f"{totals.documents:,} documents, agreed by all {len(sources)} sources",
            sources,
        )
    else:
        log.fail(
            "counts.documents",
            f"document count disagrees across sources: {sources}",
            sources,
        )

    byte_sources = {
        "recomputed": totals.utf8_bytes,
        "manifest.result.utf8_bytes": manifest.get("result", {}).get("utf8_bytes"),
        "progress.accepted_utf8_bytes": progress.get("totals", {}).get(
            "accepted_utf8_bytes"
        ),
    }
    if len({v for v in byte_sources.values()}) == 1:
        log.ok(
            "counts.utf8_bytes",
            f"{totals.utf8_bytes:,} UTF-8 text bytes, agreed by all sources",
            byte_sources,
        )
    else:
        log.fail("counts.utf8_bytes", f"UTF-8 byte totals disagree: {byte_sources}", byte_sources)

    char_sources = {
        "recomputed": totals.characters,
        "manifest.result.characters": manifest.get("result", {}).get("characters"),
        "progress.accepted_characters": progress.get("totals", {}).get(
            "accepted_characters"
        ),
    }
    if len({v for v in char_sources.values()}) == 1:
        log.ok(
            "counts.characters",
            f"{totals.characters:,} characters, agreed by all sources",
            char_sources,
        )
    else:
        log.fail("counts.characters", f"character totals disagree: {char_sources}", char_sources)

    jsonl_sources = {
        "recomputed": totals.uncompressed_bytes,
        "manifest.output.uncompressed_jsonl_bytes": manifest.get("output", {}).get(
            "uncompressed_jsonl_bytes"
        ),
        "progress.output_jsonl_bytes": progress.get("totals", {}).get(
            "output_jsonl_bytes"
        ),
    }
    if len({v for v in jsonl_sources.values()}) == 1:
        log.ok(
            "counts.uncompressed_jsonl_bytes",
            f"{totals.uncompressed_bytes:,} decompressed JSONL bytes, agreed by all sources",
            jsonl_sources,
        )
    else:
        log.fail(
            "counts.uncompressed_jsonl_bytes",
            f"decompressed byte totals disagree: {jsonl_sources}",
            jsonl_sources,
        )


def open_dedup(corpus: Path, log: CheckLog) -> tuple[sqlite3.Connection | None, int | None]:
    path = corpus / DEDUP_FILENAME
    if not path.is_file():
        log.skip("dedup.available", f"{DEDUP_FILENAME} not present")
        return None, None

    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        count = next(connection.execute("SELECT COUNT(*) FROM exact_hashes"))[0]
    except sqlite3.Error as exc:
        log.fail("dedup.available", f"cannot read {DEDUP_FILENAME}: {exc}")
        return None, None

    log.ok("dedup.available", f"{count:,} exact fingerprints stored")
    return connection, int(count)


# --------------------------------------------------------------------------
# freeze stamp
# --------------------------------------------------------------------------


def build_freeze_stamp(
    corpus: Path,
    manifest: dict[str, Any],
    totals: ScanTotals,
    report_path: Path | None,
    report_digest: str,
) -> dict[str, Any]:
    return {
        "format": FREEZE_FORMAT,
        "format_version": FREEZE_FORMAT_VERSION,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus.name,
        "status": "FROZEN",
        "policy": (
            "This corpus is immutable. Any change to cleaning, filtering, "
            "deduplication, or decontamination produces a new corpus version; "
            "it never modifies this one."
        ),
        "documents": totals.documents,
        "characters": totals.characters,
        "utf8_bytes": totals.utf8_bytes,
        "uncompressed_jsonl_bytes": totals.uncompressed_bytes,
        "manifest_sha256": sha256_file(corpus / MANIFEST_FILENAME),
        "evaluation_sha256": evaluation_settings(manifest).get("evaluation_sha256"),
        "validation_report": {
            "path": str(report_path) if report_path is not None else None,
            "sha256": report_digest,
        },
        "shards": [shard.as_dict() for shard in totals.shards],
    }


def compare_freeze_stamp(
    existing: dict[str, Any], candidate: dict[str, Any], log: CheckLog
) -> None:
    """Re-validation: an existing stamp must still describe the same bytes."""

    drift: list[str] = []
    for key in ("documents", "characters", "utf8_bytes", "uncompressed_jsonl_bytes"):
        if existing.get(key) != candidate.get(key):
            drift.append(f"{key}: frozen={existing.get(key)!r} now={candidate.get(key)!r}")

    frozen_shards = {
        entry.get("name"): entry.get("compressed_sha256")
        for entry in existing.get("shards", [])
    }
    current_shards = {
        entry.get("name"): entry.get("compressed_sha256")
        for entry in candidate.get("shards", [])
    }
    for name in sorted(set(frozen_shards) | set(current_shards)):
        before = frozen_shards.get(name)
        after = current_shards.get(name)
        if before != after:
            drift.append(f"{name}: frozen={before} now={after}")

    if drift:
        log.fail(
            "freeze.corpus_unchanged",
            f"{len(drift)} difference(s) from the existing freeze stamp; the "
            "corpus has been modified since it was frozen",
            {"drift": drift[:MAX_REPORTED_PROBLEMS]},
        )
    else:
        log.ok(
            "freeze.corpus_unchanged",
            f"every shard still matches the freeze stamp from "
            f"{existing.get('frozen_at_utc', 'unknown time')}",
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute every corpus invariant from the shard bytes and compare "
            "against the manifest, the progress file, and the dedup index."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write the full JSON validation report here.",
    )
    parser.add_argument(
        "--expected-evaluation-sha256",
        type=sha256_hex,
        default=None,
        help="Pin the evaluation set the corpus must have been decontaminated against.",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=None,
        help="Evaluation JSONL to re-hash, proving the pinned digest is still its content.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help=(
            f"On a complete, fully passing run, write {FREEZE_FILENAME} recording "
            "per-shard SHA-256 digests. Refuses if any check failed or was skipped."
        ),
    )
    parser.add_argument(
        "--skip-normalization-check",
        action="store_true",
        help="Skip per-document normalizer idempotence (the slowest check).",
    )
    parser.add_argument(
        "--skip-canonical-check",
        action="store_true",
        help="Skip re-serializing each record to confirm the stored line is canonical.",
    )
    parser.add_argument(
        "--skip-dedup-crosscheck",
        action="store_true",
        help="Skip the per-document fingerprint lookup in dedup.sqlite.",
    )
    parser.add_argument(
        "--max-documents",
        type=positive_int,
        default=None,
        help="Stop after this many documents. Development only; blocks --freeze.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing --report.")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)

    try:
        corpus: Path = args.corpus
        if not corpus.is_dir():
            print(f"ERROR: corpus directory not found: {corpus}", file=sys.stderr)
            return 1

        if (
            args.report is not None
            and args.report.exists()
            and not args.overwrite
        ):
            print(
                f"ERROR: report already exists (use --overwrite): {args.report}",
                file=sys.stderr,
            )
            return 1

        manifest_path = corpus / MANIFEST_FILENAME
        progress_path = corpus / PROGRESS_FILENAME
        for required in (manifest_path, progress_path):
            if not required.is_file():
                print(f"ERROR: required file missing: {required}", file=sys.stderr)
                return 1

        manifest = load_json(manifest_path)
        progress = load_json(progress_path)
        build_report_path = corpus / BUILD_REPORT_FILENAME
        build_report = (
            load_json(build_report_path) if build_report_path.is_file() else None
        )

        log = CheckLog(quiet=args.quiet)

        if not args.quiet:
            print()
            print("PRETRAINING CORPUS VALIDATION")
            print("=" * 72)
            print(f"Corpus: {corpus}")
            print()
            print("Structure")
            print("-" * 72)

        shards = discover_shards(corpus, log)
        if not shards:
            return 2

        check_shard_listing(manifest, progress, shards, log)
        check_inputs_complete(manifest, progress, log)

        if not args.quiet:
            print()
            print("Configuration")
            print("-" * 72)

        check_configuration(manifest, progress, build_report, log)
        check_evaluation_digest(
            manifest,
            progress,
            args.expected_evaluation_sha256,
            args.evaluation,
            log,
        )

        if not args.quiet:
            print()
            print("Content")
            print("-" * 72)

        dedup_connection, dedup_fingerprints = (
            (None, None) if args.skip_dedup_crosscheck else open_dedup(corpus, log)
        )
        if args.skip_dedup_crosscheck:
            log.skip("dedup.available", "--skip-dedup-crosscheck")

        totals = ScanTotals()
        seen_prefixes: set[bytes] = set()
        completed = True

        try:
            for shard in shards:
                if not args.quiet:
                    print(f"         reading {shard.name}", flush=True)
                try:
                    completed = scan_shard(
                        shard,
                        totals,
                        seen_prefixes,
                        dedup_connection,
                        check_normalization=not args.skip_normalization_check,
                        check_canonical=not args.skip_canonical_check,
                        max_documents=args.max_documents,
                        quiet=args.quiet,
                    )
                except (OSError, EOFError, gzip.BadGzipFile) as exc:
                    log.fail(
                        "shards.readable",
                        f"{shard.name} could not be read to the end: {exc}",
                    )
                    completed = False
                    break
                if not completed:
                    break
        finally:
            if dedup_connection is not None:
                dedup_connection.close()

        if completed:
            log.ok(
                "shards.readable",
                f"all {len(shards)} shards decompressed and CRC-verified to EOF",
            )

        log.verdict("records.json_parse", totals.problems_json, f"{totals.documents:,} lines parsed as JSON")
        log.verdict("records.required_fields", totals.problems_fields, "every record has the 9 expected fields with correct types")

        if args.skip_canonical_check:
            log.skip("records.canonical_form", "--skip-canonical-check")
        else:
            log.verdict("records.canonical_form", totals.problems_canonical, "every line is the canonical serialization of its record")

        log.verdict("records.text_sha256", totals.problems_text_hash, "every text_sha256 matches its recomputed digest")
        log.verdict("records.document_id_derivation", totals.problems_document_id, "every document_id is derived from its text digest")
        log.verdict("records.split_group_derivation", totals.problems_split_group, "every split_group is derived from its url and document_id")

        if args.skip_normalization_check:
            log.skip("records.normalizer_stable", "--skip-normalization-check")
        else:
            log.verdict("records.normalizer_stable", totals.problems_normalization, "stored text is a fixed point of the project normalizer")

        log.verdict("uniqueness.document_id_and_text", totals.problems_uniqueness, f"{len(seen_prefixes):,} distinct digests over {totals.documents:,} documents (100% unique)")

        if dedup_fingerprints is None:
            log.skip("dedup.every_document_indexed", "dedup index unavailable or skipped")
        else:
            log.verdict("dedup.every_document_indexed", totals.problems_dedup, "every document's fingerprint is present in dedup.sqlite")

        if not args.quiet:
            print()
            print("Totals")
            print("-" * 72)

        check_counts(manifest, progress, totals, dedup_fingerprints, log)

        if args.max_documents is not None and not completed:
            log.skip(
                "scan.complete",
                f"--max-documents {args.max_documents} stopped the scan early; "
                "totals and digests describe a prefix of the corpus only",
            )
        else:
            log.ok("scan.complete", "the entire corpus was scanned")

        # ------------------------------------------------------------------
        # freeze
        # ------------------------------------------------------------------
        finished = datetime.now(timezone.utc)
        report = {
            "format": VALIDATOR_FORMAT,
            "format_version": VALIDATOR_FORMAT_VERSION,
            "created_at_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "corpus": {
                "path": str(corpus),
                "shard_count": len(shards),
                "manifest_sha256": sha256_file(manifest_path),
            },
            "options": {
                "normalization_check": not args.skip_normalization_check,
                "canonical_check": not args.skip_canonical_check,
                "dedup_crosscheck": not args.skip_dedup_crosscheck,
                "max_documents": args.max_documents,
                "complete_scan": completed and args.max_documents is None,
            },
            "recomputed": {
                "documents": totals.documents,
                "characters": totals.characters,
                "utf8_bytes": totals.utf8_bytes,
                "uncompressed_jsonl_bytes": totals.uncompressed_bytes,
                "distinct_text_digests": len(seen_prefixes),
            },
            "shards": [shard.as_dict() for shard in totals.shards],
            "checks": [check.as_dict() for check in log.checks],
            "summary": {
                "total": len(log.checks),
                "passed": len(log.checks) - len(log.failed) - len(log.skipped) - len(log.warned),
                "failed": len(log.failed),
                "skipped": len(log.skipped),
                "warnings": len(log.warned),
                "result": "PASS" if log.passed else "FAIL",
            },
        }

        report_bytes = (
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        report_digest = hashlib.sha256(report_bytes).hexdigest()

        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_bytes(report_bytes)

        freeze_written = False
        freeze_path = corpus / FREEZE_FILENAME
        if args.freeze:
            candidate = build_freeze_stamp(
                corpus, manifest, totals, args.report, report_digest
            )
            blockers: list[str] = []
            if not log.passed:
                blockers.append(f"{len(log.failed)} check(s) failed")
            if args.max_documents is not None or not completed:
                blockers.append("the scan did not cover the whole corpus")
            if log.skipped:
                blockers.append(
                    f"{len(log.skipped)} check(s) were skipped: "
                    + ", ".join(c.name for c in log.skipped)
                )

            if blockers:
                log.fail("freeze.written", "refused to freeze: " + "; ".join(blockers))
            elif freeze_path.is_file():
                compare_freeze_stamp(load_json(freeze_path), candidate, log)
                if log.passed:
                    log.ok(
                        "freeze.written",
                        f"{FREEZE_FILENAME} already present and still accurate; left unchanged",
                    )
            else:
                freeze_path.write_text(
                    json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                freeze_written = True
                log.ok("freeze.written", f"wrote {freeze_path}")
        elif freeze_path.is_file():
            candidate = build_freeze_stamp(
                corpus, manifest, totals, args.report, report_digest
            )
            if completed and args.max_documents is None:
                compare_freeze_stamp(load_json(freeze_path), candidate, log)
            else:
                log.skip(
                    "freeze.corpus_unchanged",
                    "partial scan cannot be compared against the freeze stamp",
                )

        if not args.quiet:
            print()
            print("=" * 72)
            print(
                f"{len(log.checks)} checks: "
                f"{len(log.checks) - len(log.failed) - len(log.skipped) - len(log.warned)} passed, "
                f"{len(log.failed)} failed, "
                f"{len(log.skipped)} skipped, "
                f"{len(log.warned)} warnings"
            )
            print()
            print(f"Documents:            {totals.documents:,}")
            print(f"Characters:           {totals.characters:,}")
            print(f"UTF-8 text bytes:     {totals.utf8_bytes:,}")
            print(f"Decompressed JSONL:   {totals.uncompressed_bytes:,}")
            print(f"Elapsed:              {report['duration_seconds']:.1f}s")
            print()
            if log.failed:
                print("RESULT: FAIL")
                for check in log.failed:
                    print(f"  FAILED {check.name}: {check.detail}")
            else:
                print("RESULT: PASS")
                if freeze_written:
                    print()
                    print("PRETRAINING CORPUS")
                    print("STATUS: FROZEN")
                    print(f"Stamp:  {freeze_path}")
            if args.report is not None:
                print()
                print(f"Report: {args.report}")

        return 0 if log.passed else 2

    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
