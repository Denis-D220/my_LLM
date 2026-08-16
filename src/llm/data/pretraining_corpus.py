"""Build a cleaned, deduplicated pretraining corpus from Common Crawl text.

This module implements the missing stage between the raw extractor output and
the tokenization pipeline::

    D:/Common Crawl/Data/cc_text_*.jsonl.gz
            |
            v
    [ this module ]
            |
            v
    data/cleaned/pretraining/v0.1/corpus-*.jsonl.gz

Stages, in order
----------------
1. stream gzip JSONL, one record at a time
2. schema validation
3. language filtering (English-primary, from extractor metadata)
4. normalization, using the tokenizer's frozen contract
5. script sanity (second language gate, from the text itself)
6. quality filtering
7. high-confidence secret filtering
8. persistent exact deduplication
9. evaluation decontamination
10. provenance-preserving output

Stage 9 removes any document that republishes or embeds frozen evaluation
material.  It fails closed: a build with decontamination enabled and no
evaluation artifact is an error, because producing the final training corpus
without protecting the evaluation set is silent and unrecoverable - no
downstream stage can detect it, and every benchmark computed afterwards is
meaningless.  The manifest records the evaluation file's SHA-256 so the claim
"corpus vN was decontaminated against exactly this artifact" stays auditable.

Stage 5 exists because stage 3 trusts metadata.  Audits found Cyrillic and
Korean pages labelled ``eng`` in the accepted corpus; the script gate reads the
text rather than the label.  It runs after normalization so it sees exactly the
NFC form that will be stored, and before quality filtering so that obviously
non-English pages never reach the more expensive checks.

Ordering is deliberate.  Normalization happens *before* fingerprinting so that
documents differing only in line endings deduplicate against each other.
Quality filtering happens before deduplication so the fingerprint database is
not polluted with junk that would never have been admitted.

Memory
------
Everything is streamed.  Peak memory is one document plus the SQLite page
cache, so corpus size does not affect resident memory.

Crash safety: one input shard is one transaction
------------------------------------------------
The unit of atomicity is a single Common Crawl input shard::

    remember the output shard index at which this input shard starts
        -> process records, writing output shards
        -> no dedup commit yet
        -> input shard finishes
        -> rotate output (every closed shard is a complete gzip member)
        -> commit dedup fingerprints
        -> write progress.json
        -> checkpoint complete

If anything raises before the checkpoint, the builder rolls the dedup
transaction back and deletes every output shard produced since the remembered
index.  The invariant is:

    an input shard is either completely committed, or completely reprocessed

This matters because the two halves of the state can otherwise disagree.  If
fingerprints were committed for documents whose output shard was never
finished, reprocessing that input shard would reject those documents as exact
duplicates and lose them silently - no error, no warning, just missing
training data.

Resume additionally deletes orphan output shards at or above
``next_output_index`` before writing, covering the case where a crash occurred
after a size-based rotation produced a shard the progress file never recorded.

Early stops
-----------
``max_records`` / ``max_documents`` can halt in the middle of an input shard.
Rather than discarding that partial work, the checkpoint records the physical
line number reached, and a resumed build skips past it.  So limits and
``--resume`` compose correctly instead of quietly dropping documents.

Not implemented here
--------------------
Near-duplicate detection and evaluation decontamination are separate stages,
deliberately excluded until this pipeline has passed a real sample audit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator

from llm.data.dedup import ExactDeduplicator, text_fingerprint
from llm.data.decontamination import (
    REASON_CONTAINMENT,
    REASON_EXACT_MATCH,
    DecontaminationConfig,
    EvaluationIndex,
    load_evaluation_items,
)
from llm.data.language_sanity import (
    REASON_LANGUAGE_SCRIPT_MISMATCH,
    ScriptSanityThresholds,
    assess_english_script,
)
from llm.data.quality import QualityThresholds, assess_document
from llm.tokenizer.normalizer import normalize_text


CORPUS_FORMAT = "llm_pretraining_corpus"
CORPUS_FORMAT_VERSION = 1

DEFAULT_SOURCE_GLOB = "cc_text_*.jsonl.gz"
DEFAULT_SHARD_PREFIX = "corpus"
DEFAULT_SHARD_BYTES = 256_000_000
DEFAULT_LANGUAGE = "eng"
DEFAULT_COMPRESS_LEVEL = 6

MANIFEST_NAME = "manifest.json"
REPORT_NAME = "build_report.json"
PROGRESS_NAME = "progress.json"
DEDUP_NAME = "dedup.sqlite"

# Rejection reasons owned by this module. Quality reasons come from quality.py.
REASON_INVALID_JSON = "invalid_json"
REASON_NOT_OBJECT = "record_not_object"
REASON_MISSING_TEXT = "missing_text"
REASON_TEXT_NOT_STRING = "text_not_string"
REASON_INVALID_UNICODE = "invalid_unicode"
REASON_NON_ENGLISH = "non_english"
REASON_EXACT_DUPLICATE = "exact_duplicate"

# Contamination subtypes are kept distinct in reporting. "This page republished
# an evaluation passage verbatim" and "this page happens to contain most of one"
# are different phenomena, and collapsing them would make future audits harder.
REASON_EVAL_CONTAMINATION_EXACT = "evaluation_contamination_exact"
REASON_EVAL_CONTAMINATION_CONTAINMENT = "evaluation_contamination_containment"

_CONTAMINATION_REASONS = {
    REASON_EXACT_MATCH: REASON_EVAL_CONTAMINATION_EXACT,
    REASON_CONTAINMENT: REASON_EVAL_CONTAMINATION_CONTAINMENT,
}

_SHARD_INDEX_RE = re.compile(r"-(\d+)\.jsonl\.gz$")


def file_sha256(path: Path) -> str:
    """Hex digest of a file's bytes, read in chunks.

    Used to bind a corpus to the exact evaluation artifact it was protected
    against.  A single changed byte in the evaluation file changes this digest,
    which is what makes the claim auditable months later.
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CorpusStats:
    """Counters for the current process invocation."""

    shards_read: int = 0
    records_seen: int = 0
    documents_accepted: int = 0
    accepted_characters: int = 0
    accepted_utf8_bytes: int = 0
    rejected: Counter = field(default_factory=Counter)

    @property
    def documents_rejected(self) -> int:
        return sum(self.rejected.values())

    @property
    def acceptance_rate(self) -> float:
        return (
            self.documents_accepted / self.records_seen
            if self.records_seen
            else 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "shards_read": self.shards_read,
            "records_seen": self.records_seen,
            "documents_accepted": self.documents_accepted,
            "documents_rejected": self.documents_rejected,
            "accepted_characters": self.accepted_characters,
            "accepted_utf8_bytes": self.accepted_utf8_bytes,
            "acceptance_rate": self.acceptance_rate,
            "rejected_by_reason": dict(sorted(self.rejected.items())),
        }


@dataclass
class CorpusTotals:
    """Cumulative counters spanning every invocation of one corpus build.

    Persisted in ``progress.json`` so that a resumed build continues its
    statistics instead of restarting them at zero.
    """

    records_seen: int = 0
    documents_accepted: int = 0
    accepted_characters: int = 0
    accepted_utf8_bytes: int = 0
    output_jsonl_bytes: int = 0
    rejected: Counter = field(default_factory=Counter)

    @property
    def documents_rejected(self) -> int:
        return sum(self.rejected.values())

    @property
    def acceptance_rate(self) -> float:
        return (
            self.documents_accepted / self.records_seen
            if self.records_seen
            else 0.0
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CorpusTotals":
        if not payload:
            return cls()
        return cls(
            records_seen=int(payload.get("records_seen", 0)),
            documents_accepted=int(payload.get("documents_accepted", 0)),
            accepted_characters=int(payload.get("accepted_characters", 0)),
            accepted_utf8_bytes=int(payload.get("accepted_utf8_bytes", 0)),
            output_jsonl_bytes=int(payload.get("output_jsonl_bytes", 0)),
            rejected=Counter(payload.get("rejected_by_reason", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "records_seen": self.records_seen,
            "documents_accepted": self.documents_accepted,
            "documents_rejected": self.documents_rejected,
            "accepted_characters": self.accepted_characters,
            "accepted_utf8_bytes": self.accepted_utf8_bytes,
            "output_jsonl_bytes": self.output_jsonl_bytes,
            "acceptance_rate": self.acceptance_rate,
            "rejected_by_reason": dict(sorted(self.rejected.items())),
        }


@dataclass(frozen=True)
class CleanDocument:
    """One accepted document with full provenance."""

    document_id: str
    text: str
    url: str | None
    date: str | None
    lang: str | None
    source_shard: str
    source_line: int
    text_sha256: str
    split_group: str

    def to_record(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "url": self.url,
            "date": self.date,
            "lang": self.lang,
            "source_shard": self.source_shard,
            "source_line": self.source_line,
            "text_sha256": self.text_sha256,
            "split_group": self.split_group,
        }


@dataclass(frozen=True)
class RejectedSample:
    """A small retained example of a rejected document, for auditing."""

    reason: str
    source_shard: str
    source_line: int
    url: str | None
    characters: int
    excerpt: str


def primary_language(lang: Any) -> str | None:
    """Return the primary language code from an extractor ``lang`` value.

    Common Crawl language fields may list several codes::

        "eng"      -> "eng"
        "eng,spa"  -> "eng"
        "deu,eng"  -> "deu"

    Only the first entry is treated as primary, so ``deu,eng`` is German text
    that happens to contain English, not the reverse.
    """

    if not isinstance(lang, str):
        return None
    first = lang.split(",")[0].strip().lower()
    return first or None


def make_document_id(text_sha256: str) -> str:
    """Content-addressed identifier.

    Exact deduplication guarantees one accepted document per distinct
    normalized text, so a prefix of the text digest is unique within a corpus
    and stable across rebuilds.
    """

    return f"cc-{text_sha256[:24]}"


def make_split_group(url: str | None, document_id: str) -> str:
    """Group key that keeps fragments of one source page together.

    Grouping by source URL means that if this document is later split into
    several chunks, or if the same page contributes more than one record, all
    of them land on the same side of the train/validation split.
    """

    if isinstance(url, str) and url.strip():
        digest = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()
        return digest[:16]
    return document_id


def discover_source_shards(
    source: Path,
    *,
    pattern: str = DEFAULT_SOURCE_GLOB,
) -> list[Path]:
    """Return input shards in deterministic order.

    Only files matching ``pattern`` are returned, which excludes crawler
    bookkeeping such as ``_state.json``.
    """

    resolved = Path(source).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"source directory does not exist: {source}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"source is not a directory: {source}")

    shards = sorted(resolved.glob(pattern), key=lambda item: item.name)
    if not shards:
        raise FileNotFoundError(f"no files matching {pattern!r} under {resolved}")
    return shards


def iter_shard_lines(
    path: Path,
    *,
    skip_until_line: int = 0,
) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, line)`` from a gzip JSONL shard.

    ``gzip.open`` transparently reads concatenated gzip members, which is what
    the extractor produces when a download is resumed.  ``line_number`` is the
    physical line number, so it stays valid as a resume position.
    """

    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number <= skip_until_line:
                continue
            if line.strip():
                yield line_number, line


def shard_index(path: Path | str) -> int | None:
    """Extract the numeric index from an output shard filename."""

    match = _SHARD_INDEX_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def remove_output_shards_from(
    output_dir: Path,
    *,
    from_index: int,
    prefix: str = DEFAULT_SHARD_PREFIX,
) -> list[str]:
    """Delete output shards whose index is >= ``from_index``.

    Used both when rolling back a failed input shard and when resuming, where
    a shard may exist on disk that the progress file never recorded.
    """

    removed: list[str] = []
    for path in sorted(Path(output_dir).glob(f"{prefix}-*.jsonl.gz")):
        index = shard_index(path)
        if index is not None and index >= from_index:
            path.unlink()
            removed.append(path.name)
    return removed


class ShardWriter:
    """Write accepted documents to rotating gzip JSONL shards."""

    def __init__(
        self,
        output_dir: Path,
        *,
        prefix: str = DEFAULT_SHARD_PREFIX,
        max_bytes: int = DEFAULT_SHARD_BYTES,
        compress_level: int = DEFAULT_COMPRESS_LEVEL,
        start_index: int = 0,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")

        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.compress_level = compress_level

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._index = start_index
        self._handle: gzip.GzipFile | None = None
        self._bytes_in_shard = 0
        self.shards_written: list[str] = []
        self.documents_written = 0
        self.bytes_written = 0

    @property
    def next_index(self) -> int:
        return self._index

    def _shard_path(self, index: int) -> Path:
        return self.output_dir / f"{self.prefix}-{index:05d}.jsonl.gz"

    def _open_shard(self) -> None:
        self._handle = gzip.open(
            self._shard_path(self._index),
            "wt",
            encoding="utf-8",
            newline="\n",
            compresslevel=self.compress_level,
        )
        self._bytes_in_shard = 0

    def write(self, document: CleanDocument) -> None:
        if self._handle is None:
            self._open_shard()

        line = json.dumps(
            document.to_record(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        encoded_length = len(line.encode("utf-8")) + 1

        assert self._handle is not None
        self._handle.write(line + "\n")

        self._bytes_in_shard += encoded_length
        self.bytes_written += encoded_length
        self.documents_written += 1

        if self._bytes_in_shard >= self.max_bytes:
            self.rotate()

    def rotate(self) -> None:
        """Close the current shard, if any, and advance the index."""

        if self._handle is None:
            return
        self._handle.close()
        self.shards_written.append(self._shard_path(self._index).name)
        self._handle = None
        self._bytes_in_shard = 0
        self._index += 1

    def abandon(self) -> None:
        """Close the open handle without recording the shard as written.

        Used on the rollback path, where the partially written shard is about
        to be deleted.  On Windows the file must be closed before it can be
        unlinked.
        """

        if self._handle is None:
            return
        self._handle.close()
        self._handle = None
        self._bytes_in_shard = 0

    def close(self) -> None:
        self.rotate()


def _empty_progress() -> dict[str, Any]:
    return {
        "completed_shards": [],
        "next_output_index": 0,
        "partial_shard": None,
        "totals": CorpusTotals().as_dict(),
        "decontamination": None,
    }


def decontamination_identity(state: dict[str, Any]) -> dict[str, Any]:
    """The parts of the decontamination policy that change what is written.

    The file *path* is deliberately excluded: relocating the evaluation
    artifact does not change which documents are rejected, whereas editing it
    does, and the digest already captures that.
    """

    return {
        "enabled": bool(state.get("enabled", False)),
        "evaluation_sha256": state.get("evaluation_sha256"),
        "shingle_size": state.get("shingle_size"),
        "min_containment": state.get("min_containment"),
        "min_evaluation_tokens": state.get("min_evaluation_tokens"),
    }


def _describe_drift(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> str:
    differences = [
        f"{key}: {previous.get(key)!r} -> {current.get(key)!r}"
        for key in sorted(set(previous) | set(current))
        if previous.get(key) != current.get(key)
    ]
    return "; ".join(differences)


def load_progress(path: Path) -> dict[str, Any]:
    """Read a progress file, tolerating absence and older payloads."""

    if not Path(path).exists():
        return _empty_progress()

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    base = _empty_progress()
    base.update(data)
    base.setdefault("partial_shard", None)
    return base


def _write_json(path: Path, payload: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_pretraining_corpus(
    *,
    source: Path,
    output: Path,
    thresholds: QualityThresholds | None = None,
    script_thresholds: ScriptSanityThresholds | None = None,
    check_script_sanity: bool = True,
    evaluation_path: Path | None = None,
    decontamination_config: DecontaminationConfig | None = None,
    check_decontamination: bool = True,
    language: str = DEFAULT_LANGUAGE,
    source_pattern: str = DEFAULT_SOURCE_GLOB,
    shard_prefix: str = DEFAULT_SHARD_PREFIX,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    compress_level: int = DEFAULT_COMPRESS_LEVEL,
    max_documents: int | None = None,
    max_records: int | None = None,
    resume: bool = False,
    keep_rejected_samples: int = 50,
    check_secrets: bool = True,
    progress_every: int = 25_000,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Run the corpus build and return its build report.

    ``max_records`` limits how many *input* records are examined, which is the
    right knob for a trial run; ``max_documents`` limits how many are accepted.
    """

    policy = thresholds or QualityThresholds()
    script_policy = script_thresholds or ScriptSanityThresholds()
    decontamination_policy = decontamination_config or DecontaminationConfig()
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fail closed. Producing the final training corpus without protecting the
    # evaluation set is a silent, unrecoverable mistake: nothing downstream can
    # detect it, and every benchmark computed afterwards is meaningless. The
    # caller must opt out deliberately rather than by omission.
    evaluation_index: EvaluationIndex | None = None
    evaluation_state: dict[str, Any] = {"enabled": False}

    if check_decontamination:
        if evaluation_path is None:
            raise ValueError(
                "evaluation decontamination is enabled but no evaluation "
                "artifact was supplied; pass evaluation_path, or disable it "
                "explicitly with check_decontamination=False"
            )

        evaluation_file = Path(evaluation_path)
        evaluation_items = load_evaluation_items(evaluation_file)
        evaluation_index = EvaluationIndex(
            evaluation_items,
            config=decontamination_policy,
        )

        evaluation_state = {
            "enabled": True,
            "evaluation_file": str(evaluation_file),
            "evaluation_items": len(evaluation_items),
            "evaluation_items_indexed": len(evaluation_index),
            "evaluation_items_skipped_short": list(
                evaluation_index.skipped_short
            ),
            "evaluation_sha256": file_sha256(evaluation_file),
            "shingle_size": decontamination_policy.shingle_size,
            "min_containment": decontamination_policy.min_containment,
            "min_evaluation_tokens": (
                decontamination_policy.min_evaluation_tokens
            ),
        }

    shards = discover_source_shards(Path(source), pattern=source_pattern)

    progress_path = output_dir / PROGRESS_NAME
    progress = load_progress(progress_path) if resume else _empty_progress()

    # A single corpus version must not contain documents processed under
    # different decontamination rules. Half a corpus protected at 0.80 and half
    # at 0.90 is not a coherent artifact, and nothing downstream could tell.
    previous_decontamination = progress.get("decontamination")
    if resume and previous_decontamination is not None:
        previous_identity = decontamination_identity(previous_decontamination)
        current_identity = decontamination_identity(evaluation_state)
        if previous_identity != current_identity:
            raise ValueError(
                "decontamination policy changed since the interrupted build; "
                "resuming would mix documents processed under different rules "
                f"({_describe_drift(previous_identity, current_identity)}). "
                "Start a new corpus version instead of resuming."
            )

    completed: set[str] = set(progress["completed_shards"])
    # Totals carried in from previous invocations. The running total is always
    # this baseline plus the current invocation's stats, never an incremental
    # accumulation, so a shard can never be counted twice.
    baseline = CorpusTotals.from_dict(progress.get("totals"))
    partial = progress.get("partial_shard") or None
    start_index = int(progress["next_output_index"])

    stats = CorpusStats()
    rejected_samples: list[RejectedSample] = []
    sample_quota: Counter = Counter()
    started_at = datetime.now(timezone.utc)

    # A crash can leave shards on disk at or beyond the recorded index, for
    # example when a size-based rotation happened after the last checkpoint.
    orphans_removed = remove_output_shards_from(
        output_dir,
        from_index=start_index,
        prefix=shard_prefix,
    )

    writer = ShardWriter(
        output_dir,
        prefix=shard_prefix,
        max_bytes=shard_bytes,
        compress_level=compress_level,
        start_index=start_index,
    )

    def record_rejection(
        reason: str,
        *,
        shard_name: str,
        line_number: int,
        url: Any,
        text: str | None,
    ) -> None:
        stats.rejected[reason] += 1
        per_reason = keep_rejected_samples // 10 or 1
        if sample_quota[reason] < per_reason:
            sample_quota[reason] += 1
            rejected_samples.append(
                RejectedSample(
                    reason=reason,
                    source_shard=shard_name,
                    source_line=line_number,
                    url=url if isinstance(url, str) else None,
                    characters=len(text or ""),
                    excerpt=(text or "")[:300],
                )
            )

    def current_totals() -> CorpusTotals:
        """Cumulative totals across every invocation of this corpus build."""

        merged = Counter(baseline.rejected)
        merged.update(stats.rejected)
        return CorpusTotals(
            records_seen=baseline.records_seen + stats.records_seen,
            documents_accepted=(
                baseline.documents_accepted + stats.documents_accepted
            ),
            accepted_characters=(
                baseline.accepted_characters + stats.accepted_characters
            ),
            accepted_utf8_bytes=(
                baseline.accepted_utf8_bytes + stats.accepted_utf8_bytes
            ),
            output_jsonl_bytes=(
                baseline.output_jsonl_bytes + writer.bytes_written
            ),
            rejected=merged,
        )

    def save_checkpoint(partial_state: dict[str, Any] | None) -> None:
        _write_json(
            progress_path,
            {
                "completed_shards": sorted(completed),
                "next_output_index": writer.next_index,
                "partial_shard": partial_state,
                "totals": current_totals().as_dict(),
                "decontamination": evaluation_state,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    stop = False

    with ExactDeduplicator(
        output_dir / DEDUP_NAME,
        transactional=True,
    ) as dedup:
        for shard_path in shards:
            if stop:
                break

            shard_name = shard_path.name
            if shard_name in completed:
                continue

            skip_until = 0
            if partial and partial.get("name") == shard_name:
                skip_until = int(partial.get("next_line", 0))

            stats.shards_read += 1
            shard_start_index = writer.next_index
            last_line = skip_until

            dedup.begin()

            try:
                for line_number, line in iter_shard_lines(
                    shard_path,
                    skip_until_line=skip_until,
                ):
                    if (
                        max_records is not None
                        and stats.records_seen >= max_records
                    ):
                        stop = True
                        break
                    if (
                        max_documents is not None
                        and stats.documents_accepted >= max_documents
                    ):
                        stop = True
                        break

                    last_line = line_number
                    stats.records_seen += 1

                    if (
                        on_progress is not None
                        and progress_every
                        and stats.records_seen % progress_every == 0
                    ):
                        on_progress(stats)

                    # -- schema --------------------------------------------
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        record_rejection(
                            REASON_INVALID_JSON,
                            shard_name=shard_name,
                            line_number=line_number,
                            url=None,
                            text=None,
                        )
                        continue

                    if not isinstance(record, dict):
                        record_rejection(
                            REASON_NOT_OBJECT,
                            shard_name=shard_name,
                            line_number=line_number,
                            url=None,
                            text=None,
                        )
                        continue

                    url = record.get("url")

                    if "text" not in record:
                        record_rejection(
                            REASON_MISSING_TEXT,
                            shard_name=shard_name,
                            line_number=line_number,
                            url=url,
                            text=None,
                        )
                        continue

                    raw_text = record["text"]
                    if not isinstance(raw_text, str):
                        record_rejection(
                            REASON_TEXT_NOT_STRING,
                            shard_name=shard_name,
                            line_number=line_number,
                            url=url,
                            text=None,
                        )
                        continue

                    # -- language ------------------------------------------
                    if primary_language(record.get("lang")) != language:
                        record_rejection(
                            REASON_NON_ENGLISH,
                            shard_name=shard_name,
                            line_number=line_number,
                            url=url,
                            text=raw_text,
                        )
                        continue

                    # -- normalization -------------------------------------
                    try:
                        text = normalize_text(raw_text)
                    except (UnicodeEncodeError, TypeError):
                        record_rejection(
                            REASON_INVALID_UNICODE,
                            shard_name=shard_name,
                            line_number=line_number,
                            url=url,
                            text=None,
                        )
                        continue

                    # -- script sanity -------------------------------------
                    # Second language gate. The extractor's lang field is
                    # metadata and is demonstrably wrong for some pages; this
                    # looks at the script the text is actually written in.
                    if check_script_sanity:
                        script_verdict = assess_english_script(
                            text,
                            thresholds=script_policy,
                        )
                        if not script_verdict.accepted:
                            record_rejection(
                                REASON_LANGUAGE_SCRIPT_MISMATCH,
                                shard_name=shard_name,
                                line_number=line_number,
                                url=url,
                                text=text,
                            )
                            continue

                    # -- quality and secrets -------------------------------
                    verdict = assess_document(
                        text,
                        thresholds=policy,
                        check_secrets=check_secrets,
                    )
                    if not verdict.accepted:
                        record_rejection(
                            verdict.reason or "unknown",
                            shard_name=shard_name,
                            line_number=line_number,
                            url=url,
                            text=text,
                        )
                        continue

                    # -- exact deduplication -------------------------------
                    digest = text_fingerprint(text)
                    if dedup.seen_fingerprint(digest):
                        stats.rejected[REASON_EXACT_DUPLICATE] += 1
                        continue

                    # -- evaluation decontamination ------------------------
                    # Runs after dedup so duplicates are discarded first, and
                    # before the write so a contaminated document never has a
                    # fingerprint recorded for it.
                    if evaluation_index is not None:
                        contamination = evaluation_index.check(text)
                        if contamination.contaminated:
                            best = contamination.best
                            assert best is not None
                            record_rejection(
                                _CONTAMINATION_REASONS.get(
                                    best.reason,
                                    REASON_EVAL_CONTAMINATION_CONTAINMENT,
                                ),
                                shard_name=shard_name,
                                line_number=line_number,
                                url=url,
                                text=text,
                            )
                            continue

                    # -- provenance and output -----------------------------
                    text_sha256 = digest.hex()
                    document_id = make_document_id(text_sha256)

                    writer.write(
                        CleanDocument(
                            document_id=document_id,
                            text=text,
                            url=url if isinstance(url, str) else None,
                            date=(
                                record.get("date")
                                if isinstance(record.get("date"), str)
                                else None
                            ),
                            lang=(
                                record.get("lang")
                                if isinstance(record.get("lang"), str)
                                else None
                            ),
                            source_shard=shard_name,
                            source_line=line_number,
                            text_sha256=text_sha256,
                            split_group=make_split_group(url, document_id),
                        )
                    )

                    # Recorded only after the writer has accepted it, and only
                    # made permanent by the commit below.
                    dedup.add_fingerprint(digest)

                    stats.documents_accepted += 1
                    stats.accepted_characters += verdict.metrics.characters
                    stats.accepted_utf8_bytes += verdict.metrics.utf8_bytes

                # -- checkpoint ---------------------------------------------
                # Rotate first so every closed shard is a complete gzip member,
                # then commit fingerprints, then record progress. That order is
                # what makes the two halves of the state agree.
                writer.rotate()
                dedup.commit()

            except BaseException:
                # Roll the whole input shard back: discard its fingerprints and
                # delete every output shard it produced, so a resumed run can
                # reprocess it from a clean slate.
                dedup.rollback()
                writer.abandon()
                remove_output_shards_from(
                    output_dir,
                    from_index=shard_start_index,
                    prefix=shard_prefix,
                )
                raise

            if stop:
                # Stopped by a limit part-way through this shard. Record the
                # line reached so a resumed build continues from it instead of
                # reprocessing (and then discarding as duplicates) the work
                # already committed above.
                partial = {"name": shard_name, "next_line": last_line}
            else:
                completed.add(shard_name)
                partial = None

            save_checkpoint(partial)

        writer.close()
        dedup.commit()
        dedup_count = dedup.count()

    finished_at = datetime.now(timezone.utc)
    totals = current_totals()

    # The manifest must describe the whole physical corpus, not just the shards
    # this invocation happened to produce.
    all_shards = sorted(
        path.name for path in output_dir.glob(f"{shard_prefix}-*.jsonl.gz")
    )

    manifest = {
        "format": CORPUS_FORMAT,
        "format_version": CORPUS_FORMAT_VERSION,
        "created_at_utc": finished_at.isoformat(),
        "source": {
            "path": str(Path(source).expanduser()),
            "pattern": source_pattern,
            "shards": [shard.name for shard in shards],
            "completed_shards": sorted(completed),
        },
        "normalization": {
            "module": "llm.tokenizer.normalizer.normalize_text",
            "encoding": "UTF-8",
            "unicode_form": "NFC",
            "preserve_case": True,
            "line_endings": "LF",
        },
        "filters": {
            "language": language,
            "language_rule": "primary code of comma-separated lang field",
            "script_sanity": {
                "enabled": check_script_sanity,
                "reason": REASON_LANGUAGE_SCRIPT_MISMATCH,
                "thresholds": asdict(script_policy),
            },
            "quality_thresholds": asdict(policy),
            "secret_filtering": check_secrets,
            "exact_deduplication": True,
            "near_deduplication": False,
            "evaluation_decontamination": evaluation_state,
        },
        "output": {
            "shards": all_shards,
            "documents": totals.documents_accepted,
            "uncompressed_jsonl_bytes": totals.output_jsonl_bytes,
            "shard_max_bytes": shard_bytes,
        },
        "dedup": {
            "database": DEDUP_NAME,
            "algorithm": "sha256-of-normalized-utf8",
            "fingerprints": dedup_count,
        },
        "result": {
            "documents": totals.documents_accepted,
            "characters": totals.accepted_characters,
            "utf8_bytes": totals.accepted_utf8_bytes,
        },
    }

    report = {
        "created_at_utc": finished_at.isoformat(),
        "started_at_utc": started_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "limits": {
            "max_records": max_records,
            "max_documents": max_documents,
            "resume": resume,
        },
        "orphan_shards_removed": orphans_removed,
        "stats": stats.as_dict(),
        "totals": totals.as_dict(),
        "rejected_samples": [asdict(sample) for sample in rejected_samples],
    }

    _write_json(output_dir / MANIFEST_NAME, manifest)
    _write_json(output_dir / REPORT_NAME, report)

    return report
