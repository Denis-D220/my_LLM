"""
Build a representative tokenizer-training corpus.

This script prepares a deterministic, deduplicated, category-balanced corpus
for training the project's byte-level BPE tokenizer.

It is intentionally separate from ``train_tokenizer.py``:

    source documents
        -> build_tokenizer_corpus.py
        -> tokenizer_corpus.jsonl
        -> train_tokenizer.py
        -> tokenizer.json

Tokenizer Corpus v0.1 policy
----------------------------
* English-focused source material is expected to be curated upstream.
* Unicode is normalized using the tokenizer's normalizer (NFC).
* Letter case is preserved.
* Meaningful whitespace is preserved.
* Empty and undersized documents are removed.
* Oversized documents are split into deterministic, boundary-aware chunks.
* Exact duplicate normalized chunks/documents are removed using SHA-256.
* Literal reserved special-token strings are rejected by default.
* Category sampling is deterministic for a given random seed.
* An optional deterministic source-document byte cap can limit concentration.
* Sampling targets category proportions by UTF-8 byte count, not document count.
* The output keeps provenance metadata for every selected document.

This script does NOT yet perform:
* Big data extraction, cleaning, or filtering;
* statistical language identification;
* near-duplicate detection;
* PII detection;
* sophisticated quality scoring.

Those belong to the larger LLM data pipeline.  This builder is for producing a
controlled tokenizer-training sample from already curated/clean text sources.

Supported input files
---------------------
* .txt / .text / .md
    One file is treated as one document.

* .jsonl / .ndjson
    One JSON object per line.  The document text is taken from --text-field.

Recommended directory layout
----------------------------
data/tokenizer_sources/
    general/
    logical/
    mathematics/
    science/
    engineering/
    software/
    code/
    networking/
    business_finance/
    humanities/

Example: build a 30 MB balanced corpus with source concentration control
---------------------------------------------------------------------------
python scripts/build_tokenizer_corpus.py ^
    --source general=data/tokenizer_sources/general ^
    --source logical=data/tokenizer_sources/logical ^
    --source mathematics=data/tokenizer_sources/mathematics ^
    --source science=data/tokenizer_sources/science ^
    --source engineering=data/tokenizer_sources/engineering ^
    --source software=data/tokenizer_sources/software ^
    --source code=data/tokenizer_sources/code ^
    --source networking=data/tokenizer_sources/networking ^
    --source business_finance=data/tokenizer_sources/business_finance ^
    --source humanities=data/tokenizer_sources/humanities ^
    --target-bytes 30000000 ^
    --max-source-document-bytes 300000 ^
    --output data/tokenizer_training/v0.3

PowerShell uses backticks instead of ^ for line continuation.


The mixture can be changed with repeated --weight CATEGORY=VALUE options.

Outputs
-------
<output>/
    tokenizer_corpus.jsonl
    manifest.json
    build_report.json

``tokenizer_corpus.jsonl`` can be passed directly to ``train_tokenizer.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable, Iterator, Sequence

from llm.tokenizer.normalizer import normalize_text
from llm.tokenizer.tokenizer import DEFAULT_SPECIAL_TOKENS


SUPPORTED_TEXT_SUFFIXES = {".txt", ".text", ".md"}
SUPPORTED_JSONL_SUFFIXES = {".jsonl", ".ndjson"}
SUPPORTED_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | SUPPORTED_JSONL_SUFFIXES

DEFAULT_TEXT_FIELD = "text"
DEFAULT_MIN_CHARS = 100
DEFAULT_MAX_CHARS = 100_000
DEFAULT_CHUNK_TARGET_CHARS = 75_000
DEFAULT_MIN_TAIL_CHARS = 10_000
DEFAULT_MAX_SOURCE_DOCUMENT_BYTES: int | None = None
DEFAULT_SEED = 42
DEFAULT_OUTPUT_FILENAME = "tokenizer_corpus.jsonl"

DEFAULT_CATEGORY_WEIGHTS = {
    "general": 0.30,
    "logical": 0.12,
    "mathematics": 0.10,
    "science": 0.12,
    "engineering": 0.10,
    "software": 0.08,
    "code": 0.05,
    "networking": 0.04,
    "business_finance": 0.04,
    "humanities": 0.05,
}


@dataclass(frozen=True)
class SourceSpec:
    category: str
    path: Path


@dataclass(frozen=True)
class CandidateDocument:
    document_id: str
    category: str
    source_file: str
    source_record: int | None
    source_chunk: int | None
    sha256: str
    characters: int
    utf8_bytes: int
    lines: int
    text: str


@dataclass(frozen=True)
class RejectedDocument:
    category: str
    source_file: str
    source_record: int | None
    source_chunk: int | None
    reason: str
    characters: int | None = None
    duplicate_of_sha256: str | None = None


@dataclass
class CategoryStats:
    files_discovered: int = 0
    documents_seen: int = 0
    large_documents_chunked: int = 0
    chunks_created: int = 0
    documents_accepted_before_sampling: int = 0
    documents_selected: int = 0
    bytes_accepted_before_sampling: int = 0
    source_documents_capped: int = 0
    source_cap_candidates_rejected: int = 0
    source_cap_bytes_rejected: int = 0
    bytes_selected: int = 0


@dataclass
class BuildStats:
    files_discovered: int = 0
    documents_seen: int = 0
    documents_selected: int = 0
    large_documents_chunked: int = 0
    chunks_created: int = 0
    chunk_tails_merged: int = 0
    chunks_skipped_too_short: int = 0
    skipped_empty: int = 0
    skipped_too_short: int = 0
    # Retained in the report schema for compatibility. With chunking enabled,
    # oversized source documents are split instead of rejected.
    skipped_too_long: int = 0
    skipped_duplicate: int = 0
    skipped_special_token_collision: int = 0
    source_documents_capped: int = 0
    source_cap_candidates_rejected: int = 0
    source_cap_bytes_rejected: int = 0
    selected_characters: int = 0
    selected_utf8_bytes: int = 0
    selected_lines: int = 0


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")

    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")

    return parsed


def parse_source_spec(value: str) -> SourceSpec:
    """Parse ``CATEGORY=PATH`` from the command line."""

    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--source must use CATEGORY=PATH, for example "
            "general=data/tokenizer_sources/general"
        )

    category, raw_path = value.split("=", 1)
    category = category.strip()
    raw_path = raw_path.strip()

    if not category:
        raise argparse.ArgumentTypeError("source category cannot be empty")
    if not raw_path:
        raise argparse.ArgumentTypeError("source path cannot be empty")

    return SourceSpec(category=category, path=Path(raw_path))


def parse_weight(value: str) -> tuple[str, float]:
    """Parse ``CATEGORY=WEIGHT`` from the command line."""

    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--weight must use CATEGORY=VALUE, for example general=0.50"
        )

    category, raw_weight = value.split("=", 1)
    category = category.strip()
    raw_weight = raw_weight.strip()

    if not category:
        raise argparse.ArgumentTypeError("weight category cannot be empty")

    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weight must be numeric") from exc

    if weight < 0:
        raise argparse.ArgumentTypeError("weight must be >= 0")

    return category, weight


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a normalized, deduplicated, category-balanced corpus "
            "for tokenizer training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=parse_source_spec,
        metavar="CATEGORY=PATH",
        help=(
            "Tokenizer source category and file/directory path. "
            "Repeat for multiple categories."
        ),
    )
    parser.add_argument(
        "--weight",
        action="append",
        type=parse_weight,
        default=[],
        metavar="CATEGORY=VALUE",
        help=(
            "Override a category sampling weight. Repeat as needed. "
            "Unspecified canonical categories use the technical-English "
            "defaults; unknown categories default to weight 1.0."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output directory for corpus and reports.",
    )
    parser.add_argument(
        "--target-bytes",
        type=positive_int,
        default=None,
        help=(
            "Approximate target corpus size in normalized UTF-8 bytes. "
            "If omitted, all accepted documents are selected."
        ),
    )
    parser.add_argument(
        "--max-documents",
        type=positive_int,
        default=None,
        help="Optional maximum number of selected documents.",
    )
    parser.add_argument(
        "--text-field",
        default=DEFAULT_TEXT_FIELD,
        help="JSONL field containing document text.",
    )
    parser.add_argument(
        "--min-chars",
        type=non_negative_int,
        default=DEFAULT_MIN_CHARS,
        help="Discard documents shorter than this after normalization.",
    )
    parser.add_argument(
        "--max-chars",
        type=positive_int,
        default=DEFAULT_MAX_CHARS,
        help=(
            "Maximum normalized characters per tokenizer candidate. "
            "Longer source documents are split instead of discarded."
        ),
    )
    parser.add_argument(
        "--chunk-target-chars",
        type=positive_int,
        default=DEFAULT_CHUNK_TARGET_CHARS,
        help=(
            "Preferred chunk size for oversized documents. Splits prefer "
            "paragraph/newline/whitespace boundaries and never exceed --max-chars."
        ),
    )
    parser.add_argument(
        "--min-tail-chars",
        type=non_negative_int,
        default=DEFAULT_MIN_TAIL_CHARS,
        help=(
            "If the final chunk is smaller than this, merge it into the previous "
            "chunk when the merged result still fits --max-chars."
        ),
    )
    parser.add_argument(
        "--max-source-document-bytes",
        type=positive_int,
        default=DEFAULT_MAX_SOURCE_DOCUMENT_BYTES,
        help=(
            "Optional maximum UTF-8 bytes contributed by one original source "
            "document before category sampling. Chunks are grouped by "
            "source_file + source_record; source_chunk is intentionally ignored. "
            "If omitted, no per-source-document cap is applied."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for deterministic sampling and output ordering.",
    )
    parser.add_argument(
        "--allow-special-token-text",
        action="store_true",
        help=(
            "Allow source documents containing literal reserved strings such "
            "as <|system|>. Default behavior rejects those documents."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in an existing output directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress status output except errors.",
    )

    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.text_field:
        raise ValueError("--text-field cannot be empty")

    if args.max_chars < args.min_chars:
        raise ValueError("--max-chars must be >= --min-chars")
    if args.chunk_target_chars > args.max_chars:
        raise ValueError("--chunk-target-chars must be <= --max-chars")
    if args.chunk_target_chars < args.min_chars:
        raise ValueError("--chunk-target-chars must be >= --min-chars")
    if args.min_tail_chars > args.max_chars:
        raise ValueError("--min-tail-chars must be <= --max-chars")

    if (
        args.max_source_document_bytes is not None
        and args.max_source_document_bytes < args.max_chars
    ):
        raise ValueError(
            "--max-source-document-bytes must be >= --max-chars"
        )

    categories = [spec.category for spec in args.source]
    duplicate_categories = sorted(
        category
        for category, count in Counter(categories).items()
        if count > 1
    )
    if duplicate_categories:
        raise ValueError(
            "each category may appear only once in --source; duplicates: "
            + ", ".join(duplicate_categories)
        )

    supplied_weights = [category for category, _ in args.weight]
    duplicate_weights = sorted(
        category
        for category, count in Counter(supplied_weights).items()
        if count > 1
    )
    if duplicate_weights:
        raise ValueError(
            "each category may appear only once in --weight; duplicates: "
            + ", ".join(duplicate_weights)
        )

    source_categories = set(categories)
    unknown_weights = sorted(set(supplied_weights) - source_categories)
    if unknown_weights:
        raise ValueError(
            "--weight supplied for category without matching --source: "
            + ", ".join(unknown_weights)
        )


def resolve_weights(
    sources: Sequence[SourceSpec],
    overrides: Sequence[tuple[str, float]],
) -> dict[str, float]:
    """Build normalized category weights for the selected sources."""

    override_map = dict(overrides)

    raw: dict[str, float] = {}
    for source in sources:
        if source.category in override_map:
            weight = override_map[source.category]
        else:
            weight = DEFAULT_CATEGORY_WEIGHTS.get(source.category, 1.0)

        raw[source.category] = weight

    total = sum(raw.values())
    if total <= 0:
        raise ValueError("at least one category must have a positive sampling weight")

    return {
        category: weight / total
        for category, weight in raw.items()
    }


def discover_files(path: Path) -> list[Path]:
    """Discover supported files under one source path deterministically."""

    resolved = path.expanduser().resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"source path does not exist: {path}")

    if resolved.is_file():
        if resolved.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"unsupported input file: {resolved}; supported suffixes: "
                + ", ".join(sorted(SUPPORTED_SUFFIXES))
            )
        return [resolved]

    if not resolved.is_dir():
        raise ValueError(f"source path is neither file nor directory: {resolved}")

    files = [
        candidate.resolve()
        for candidate in resolved.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in SUPPORTED_SUFFIXES
    ]

    return sorted(files, key=lambda item: str(item).casefold())


def iter_documents(
    path: Path,
    *,
    text_field: str,
) -> Iterator[tuple[str, int | None]]:
    """Yield ``(text, source_record)`` from a supported input file."""

    suffix = path.suffix.lower()

    if suffix in SUPPORTED_TEXT_SUFFIXES:
        yield path.read_text(encoding="utf-8", errors="strict"), None
        return

    if suffix in SUPPORTED_JSONL_SUFFIXES:
        with path.open(
            "r",
            encoding="utf-8",
            errors="strict",
            newline="",
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path} line {line_number}: {exc.msg}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL record must be an object: "
                        f"{path} line {line_number}"
                    )

                if text_field not in record:
                    raise ValueError(
                        f"missing field {text_field!r}: "
                        f"{path} line {line_number}"
                    )

                text = record[text_field]
                if not isinstance(text, str):
                    raise ValueError(
                        f"field {text_field!r} must be a string: "
                        f"{path} line {line_number}"
                    )

                yield text, line_number

        return

    raise ValueError(f"unsupported file: {path}")


def contains_special_token(text: str) -> bool:
    return any(token in text for token in DEFAULT_SPECIAL_TOKENS)



def _nearest_boundary(
    text: str,
    *,
    start: int,
    target_end: int,
    hard_end: int,
) -> int:
    """Choose a deterministic split boundary near ``target_end``.

    Boundary preference is semantic before positional:

    1. paragraph break (``\\n\\n``)
    2. line break (``\\n``)
    3. sentence-ish boundary (``. ``)
    4. other whitespace

    Within one boundary class, the closest location to the target is chosen.
    The returned position is always greater than ``start`` and no greater than
    ``hard_end``.
    """

    if not (start < target_end <= hard_end):
        raise ValueError("invalid chunk boundary search range")

    # Do not accept a boundary extremely close to the start; that would create
    # pathological tiny chunks simply because a heading or blank line happens
    # to occur there.
    lower = start + max(1, (target_end - start) // 2)

    def candidate_for(separator: str) -> int | None:
        before = text.rfind(separator, lower, target_end + 1)
        after = text.find(separator, target_end, hard_end)

        positions: list[int] = []
        if before != -1:
            positions.append(before + len(separator))
        if after != -1:
            positions.append(after + len(separator))

        if not positions:
            return None

        return min(positions, key=lambda pos: (abs(pos - target_end), pos))

    for separator in ("\\n\\n", "\\n", ". ", " "):
        position = candidate_for(separator)
        if position is not None and start < position <= hard_end:
            return position

    return hard_end


def split_oversized_document(
    text: str,
    *,
    target_chars: int,
    max_chars: int,
    min_tail_chars: int,
) -> tuple[list[str], bool]:
    """Split normalized text without losing or reordering any characters.

    Returns ``(chunks, tail_merged)``.  All chunks are at most ``max_chars``.
    Splits are deterministic and prefer natural text boundaries.  Concatenating
    the returned chunks always reconstructs the original input exactly.
    """

    if len(text) <= max_chars:
        return [text], False
    if target_chars <= 0 or max_chars <= 0:
        raise ValueError("chunk sizes must be positive")
    if target_chars > max_chars:
        raise ValueError("target_chars cannot exceed max_chars")

    chunks: list[str] = []
    start = 0
    n = len(text)

    while n - start > max_chars:
        target_end = min(start + target_chars, n)
        hard_end = min(start + max_chars, n)
        end = _nearest_boundary(
            text,
            start=start,
            target_end=target_end,
            hard_end=hard_end,
        )

        # Defensive fallback. This should only matter for highly unusual input.
        if end <= start:
            end = hard_end

        chunks.append(text[start:end])
        start = end

    if start < n:
        chunks.append(text[start:])

    tail_merged = False
    if (
        len(chunks) >= 2
        and len(chunks[-1]) < min_tail_chars
        and len(chunks[-2]) + len(chunks[-1]) <= max_chars
    ):
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks.pop()
        tail_merged = True

    if any(len(chunk) > max_chars for chunk in chunks):
        raise RuntimeError("internal error: chunk exceeds max_chars")
    if "".join(chunks) != text:
        raise RuntimeError("internal error: chunking changed source text")

    return chunks, tail_merged


def make_document_id(
    *,
    category: str,
    source_file: Path,
    source_record: int | None,
    source_chunk: int | None,
    text_sha256: str,
) -> str:
    """Create a stable identifier without embedding the full source path."""

    payload = (
        f"{category}\0{source_file.as_posix()}\0"
        f"{source_record if source_record is not None else ''}\0"
        f"{source_chunk if source_chunk is not None else ''}\0{text_sha256}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:24]


def load_candidates(
    sources: Sequence[SourceSpec],
    *,
    text_field: str,
    min_chars: int,
    max_chars: int,
    chunk_target_chars: int,
    min_tail_chars: int,
    allow_special_token_text: bool,
    stats: BuildStats,
    category_stats: dict[str, CategoryStats],
    rejected_documents: list[RejectedDocument],
) -> dict[str, list[CandidateDocument]]:
    """Normalize, chunk, filter, and exact-deduplicate source documents."""

    candidates: dict[str, list[CandidateDocument]] = defaultdict(list)
    seen_hashes: dict[str, str] = {}

    for source in sources:
        files = discover_files(source.path)

        category_stats[source.category].files_discovered += len(files)
        stats.files_discovered += len(files)

        for file_path in files:
            for raw_text, source_record in iter_documents(
                file_path,
                text_field=text_field,
            ):
                stats.documents_seen += 1
                category_stats[source.category].documents_seen += 1

                normalized = normalize_text(raw_text)

                if not normalized.strip():
                    stats.skipped_empty += 1
                    rejected_documents.append(
                        RejectedDocument(
                            category=source.category,
                            source_file=str(file_path),
                            source_record=source_record,
                            source_chunk=None,
                            reason="empty",
                            characters=len(normalized),
                        )
                    )
                    continue

                source_characters = len(normalized)

                if source_characters < min_chars:
                    stats.skipped_too_short += 1
                    rejected_documents.append(
                        RejectedDocument(
                            category=source.category,
                            source_file=str(file_path),
                            source_record=source_record,
                            source_chunk=None,
                            reason="too_short",
                            characters=source_characters,
                        )
                    )
                    continue

                # Preserve the original document-level collision policy.  We do
                # not silently retain safe chunks from a source document that
                # contains reserved control-token text elsewhere.
                if (
                    not allow_special_token_text
                    and contains_special_token(normalized)
                ):
                    stats.skipped_special_token_collision += 1
                    rejected_documents.append(
                        RejectedDocument(
                            category=source.category,
                            source_file=str(file_path),
                            source_record=source_record,
                            source_chunk=None,
                            reason="special_token_collision",
                            characters=source_characters,
                        )
                    )
                    continue

                was_oversized = source_characters > max_chars
                chunks, tail_merged = split_oversized_document(
                    normalized,
                    target_chars=chunk_target_chars,
                    max_chars=max_chars,
                    min_tail_chars=min_tail_chars,
                )

                if was_oversized:
                    stats.large_documents_chunked += 1
                    stats.chunks_created += len(chunks)
                    category_stats[source.category].large_documents_chunked += 1
                    category_stats[source.category].chunks_created += len(chunks)
                    if tail_merged:
                        stats.chunk_tails_merged += 1

                # Unsplit documents retain source_chunk=None for backward-friendly
                # provenance. Split documents use zero-based chunk indices.
                split_document = len(chunks) > 1 or was_oversized

                for chunk_index, chunk in enumerate(chunks):
                    source_chunk = chunk_index if split_document else None
                    characters = len(chunk)

                    if characters < min_chars:
                        stats.skipped_too_short += 1
                        stats.chunks_skipped_too_short += 1
                        rejected_documents.append(
                            RejectedDocument(
                                category=source.category,
                                source_file=str(file_path),
                                source_record=source_record,
                                source_chunk=source_chunk,
                                reason="chunk_too_short",
                                characters=characters,
                            )
                        )
                        continue

                    # This should be guaranteed by split_oversized_document. Keep
                    # the guard so a future chunker regression is visible.
                    if characters > max_chars:
                        stats.skipped_too_long += 1
                        rejected_documents.append(
                            RejectedDocument(
                                category=source.category,
                                source_file=str(file_path),
                                source_record=source_record,
                                source_chunk=source_chunk,
                                reason="chunk_too_long",
                                characters=characters,
                            )
                        )
                        continue

                    encoded = chunk.encode("utf-8", errors="strict")
                    text_sha256 = hashlib.sha256(encoded).hexdigest()

                    if text_sha256 in seen_hashes:
                        stats.skipped_duplicate += 1
                        rejected_documents.append(
                            RejectedDocument(
                                category=source.category,
                                source_file=str(file_path),
                                source_record=source_record,
                                source_chunk=source_chunk,
                                reason="duplicate",
                                characters=characters,
                                duplicate_of_sha256=text_sha256,
                            )
                        )
                        continue

                    seen_hashes[text_sha256] = str(file_path)

                    candidate = CandidateDocument(
                        document_id=make_document_id(
                            category=source.category,
                            source_file=file_path,
                            source_record=source_record,
                            source_chunk=source_chunk,
                            text_sha256=text_sha256,
                        ),
                        category=source.category,
                        source_file=str(file_path),
                        source_record=source_record,
                        source_chunk=source_chunk,
                        sha256=text_sha256,
                        characters=characters,
                        utf8_bytes=len(encoded),
                        lines=chunk.count("\n") + 1,
                        text=chunk,
                    )

                    candidates[source.category].append(candidate)

                    cat_stats = category_stats[source.category]
                    cat_stats.documents_accepted_before_sampling += 1
                    cat_stats.bytes_accepted_before_sampling += candidate.utf8_bytes

    return dict(candidates)


def canonical_source_document_key(
    source_file: str,
    source_record: int | None,
) -> tuple[str, int | None]:
    """Return a stable key for one original source document.

    ``source_chunk`` is intentionally excluded so every chunk produced from the
    same original text/Markdown file or JSONL record shares one contribution
    budget.
    """

    path = Path(source_file).expanduser()
    try:
        canonical = str(path.resolve())
    except OSError:
        canonical = str(path)

    return canonical.casefold(), source_record


def _source_seed(
    seed: int,
    source_key: tuple[str, int | None],
) -> int:
    """Derive a deterministic per-source random seed."""

    payload = f"{seed}:{source_key[0]}:{source_key[1]}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def cap_source_document_contribution(
    candidates: dict[str, list[CandidateDocument]],
    *,
    max_source_document_bytes: int,
    seed: int,
    stats: BuildStats,
    category_stats: dict[str, CategoryStats],
) -> dict[str, list[CandidateDocument]]:
    """Limit each original source document before category sampling.

    Candidate chunks are grouped by ``source_file + source_record`` and shuffled
    deterministically within the group.  This prevents a large document from
    always contributing only its beginning while preserving reproducibility.
    The cap is strict in UTF-8 bytes: a candidate that would exceed the remaining
    source budget is excluded from the sampling pool.
    """

    result: dict[str, list[CandidateDocument]] = defaultdict(list)

    for category in sorted(candidates):
        grouped: dict[
            tuple[str, int | None],
            list[CandidateDocument],
        ] = defaultdict(list)

        for document in candidates[category]:
            source_key = canonical_source_document_key(
                document.source_file,
                document.source_record,
            )
            grouped[source_key].append(document)

        for source_key in sorted(grouped):
            pool = list(grouped[source_key])
            random.Random(_source_seed(seed, source_key)).shuffle(pool)

            kept_bytes = 0
            source_was_capped = False

            for document in pool:
                if kept_bytes + document.utf8_bytes > max_source_document_bytes:
                    stats.source_cap_candidates_rejected += 1
                    stats.source_cap_bytes_rejected += document.utf8_bytes

                    cat_stats = category_stats[category]
                    cat_stats.source_cap_candidates_rejected += 1
                    cat_stats.source_cap_bytes_rejected += document.utf8_bytes
                    source_was_capped = True
                    continue

                result[category].append(document)
                kept_bytes += document.utf8_bytes

            if source_was_capped:
                stats.source_documents_capped += 1
                category_stats[category].source_documents_capped += 1

    return dict(result)


def deterministic_shuffle(
    documents: Sequence[CandidateDocument],
    *,
    seed: int,
    category: str,
) -> list[CandidateDocument]:
    """Shuffle a category reproducibly without depending on hash randomization."""

    category_seed_bytes = hashlib.sha256(
        f"{seed}:{category}".encode("utf-8")
    ).digest()
    category_seed = int.from_bytes(category_seed_bytes[:8], "big")

    shuffled = list(documents)
    random.Random(category_seed).shuffle(shuffled)
    return shuffled


def select_all(
    candidates: dict[str, list[CandidateDocument]],
    *,
    seed: int,
) -> list[CandidateDocument]:
    """Select every accepted document, still using deterministic ordering."""

    selected: list[CandidateDocument] = []

    for category in sorted(candidates):
        selected.extend(
            deterministic_shuffle(
                candidates[category],
                seed=seed,
                category=category,
            )
        )

    random.Random(seed).shuffle(selected)
    return selected


def select_weighted_by_bytes(
    candidates: dict[str, list[CandidateDocument]],
    *,
    weights: dict[str, float],
    target_bytes: int,
    seed: int,
) -> list[CandidateDocument]:
    """Select documents toward category byte quotas.

    A whole document is always selected; therefore the final total may slightly
    exceed the requested byte target.
    """

    pools = {
        category: deterministic_shuffle(
            documents,
            seed=seed,
            category=category,
        )
        for category, documents in candidates.items()
    }

    quotas = {
        category: target_bytes * weight
        for category, weight in weights.items()
    }

    selected: list[CandidateDocument] = []
    selected_ids: set[str] = set()
    selected_bytes: dict[str, int] = defaultdict(int)

    # First pass: fill each category toward its own quota.
    for category in sorted(weights):
        for document in pools.get(category, []):
            if selected_bytes[category] >= quotas[category]:
                break

            selected.append(document)
            selected_ids.add(document.document_id)
            selected_bytes[category] += document.utf8_bytes

    total_bytes = sum(document.utf8_bytes for document in selected)

    # If one or more categories did not contain enough data to satisfy their
    # quota, redistribute the unused target capacity across all remaining
    # documents.  Prefer categories that are most underrepresented relative
    # to their requested share.
    remaining: dict[str, list[CandidateDocument]] = {
        category: [
            document
            for document in pool
            if document.document_id not in selected_ids
        ]
        for category, pool in pools.items()
    }

    while total_bytes < target_bytes:
        available_categories = [
            category
            for category, pool in remaining.items()
            if pool
        ]

        if not available_categories:
            break

        def deficit_score(category: str) -> tuple[float, str]:
            expected = max(quotas.get(category, 0.0), 1.0)
            ratio = selected_bytes[category] / expected
            return ratio, category

        category = min(available_categories, key=deficit_score)
        document = remaining[category].pop(0)

        selected.append(document)
        selected_ids.add(document.document_id)
        selected_bytes[category] += document.utf8_bytes
        total_bytes += document.utf8_bytes

    random.Random(seed).shuffle(selected)
    return selected


def apply_document_limit(
    documents: list[CandidateDocument],
    *,
    max_documents: int | None,
) -> list[CandidateDocument]:
    if max_documents is None:
        return documents
    return documents[:max_documents]


def update_selected_stats(
    selected: Sequence[CandidateDocument],
    *,
    stats: BuildStats,
    category_stats: dict[str, CategoryStats],
) -> None:
    stats.documents_selected = len(selected)
    stats.selected_characters = sum(doc.characters for doc in selected)
    stats.selected_utf8_bytes = sum(doc.utf8_bytes for doc in selected)
    stats.selected_lines = sum(doc.lines for doc in selected)

    for document in selected:
        cat_stats = category_stats[document.category]
        cat_stats.documents_selected += 1
        cat_stats.bytes_selected += document.utf8_bytes


def ensure_output_directory(path: Path, *, overwrite: bool) -> Path:
    output = path.expanduser().resolve()

    if output.exists() and not output.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output}")

    output.mkdir(parents=True, exist_ok=True)

    managed_files = {
        DEFAULT_OUTPUT_FILENAME,
        "manifest.json",
        "build_report.json",
    }

    existing_managed = [
        output / filename
        for filename in managed_files
        if (output / filename).exists()
    ]

    if existing_managed and not overwrite:
        names = ", ".join(path.name for path in existing_managed)
        raise FileExistsError(
            f"output already contains generated files ({names}); "
            "use --overwrite to replace them"
        )

    return output


def write_jsonl(
    path: Path,
    documents: Sequence[CandidateDocument],
) -> str:
    """Write corpus JSONL and return its SHA-256."""

    digest = hashlib.sha256()

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for document in documents:
            record = {
                "id": document.document_id,
                "category": document.category,
                "source_file": document.source_file,
                "source_record": document.source_record,
                "source_chunk": document.source_chunk,
                "source_sha256": document.sha256,
                "characters": document.characters,
                "utf8_bytes": document.utf8_bytes,
                "text": document.text,
            }

            serialized = json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            line = serialized + "\n"

            handle.write(line)
            digest.update(line.encode("utf-8"))

    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def category_distribution(
    selected: Sequence[CandidateDocument],
) -> dict[str, dict[str, float | int]]:
    by_category: dict[str, list[CandidateDocument]] = defaultdict(list)

    for document in selected:
        by_category[document.category].append(document)

    total_bytes = sum(document.utf8_bytes for document in selected)
    total_documents = len(selected)

    result: dict[str, dict[str, float | int]] = {}

    for category in sorted(by_category):
        documents = by_category[category]
        category_bytes = sum(document.utf8_bytes for document in documents)

        result[category] = {
            "documents": len(documents),
            "utf8_bytes": category_bytes,
            "document_fraction": (
                len(documents) / total_documents if total_documents else 0.0
            ),
            "byte_fraction": (
                category_bytes / total_bytes if total_bytes else 0.0
            ),
        }

    return result


def source_contribution_summary(
    selected: Sequence[CandidateDocument],
    *,
    top_n: int = 10,
) -> dict[str, object]:
    """Summarize final concentration by original source document."""

    total_bytes = sum(document.utf8_bytes for document in selected)
    by_source: dict[tuple[str, int | None], dict[str, object]] = {}
    source_files: set[str] = set()

    for document in selected:
        key = canonical_source_document_key(
            document.source_file,
            document.source_record,
        )
        source_files.add(key[0])

        if key not in by_source:
            by_source[key] = {
                "source_file": document.source_file,
                "source_record": document.source_record,
                "utf8_bytes": 0,
                "chunks": 0,
            }

        item = by_source[key]
        item["utf8_bytes"] = int(item["utf8_bytes"]) + document.utf8_bytes
        item["chunks"] = int(item["chunks"]) + 1

    ranked = sorted(
        by_source.values(),
        key=lambda item: (
            -int(item["utf8_bytes"]),
            str(item["source_file"]).casefold(),
            -1 if item["source_record"] is None else int(item["source_record"]),
        ),
    )

    top_source_documents: list[dict[str, object]] = []
    for item in ranked[:top_n]:
        item_bytes = int(item["utf8_bytes"])
        top_source_documents.append(
            {
                **item,
                "fraction": item_bytes / total_bytes if total_bytes else 0.0,
            }
        )

    largest_bytes = int(ranked[0]["utf8_bytes"]) if ranked else 0

    return {
        "distinct_source_files": len(source_files),
        "distinct_source_documents": len(by_source),
        "largest_source_document_bytes": largest_bytes,
        "largest_source_document_fraction": (
            largest_bytes / total_bytes if total_bytes else 0.0
        ),
        "top_source_documents": top_source_documents,
    }


def print_summary(
    *,
    output: Path,
    weights: dict[str, float],
    stats: BuildStats,
    category_stats: dict[str, CategoryStats],
    corpus_sha256: str,
    max_source_document_bytes: int | None,
    source_summary: dict[str, object],
) -> None:
    print()
    print("Tokenizer Corpus Build")
    print("=" * 80)
    print(f"Output:                  {output}")
    print(f"Files discovered:        {stats.files_discovered:,}")
    print(f"Documents seen:          {stats.documents_seen:,}")
    print(f"Documents selected:      {stats.documents_selected:,}")
    print(f"Characters selected:     {stats.selected_characters:,}")
    print(f"UTF-8 bytes selected:    {stats.selected_utf8_bytes:,}")
    print(f"Lines selected:          {stats.selected_lines:,}")
    print()
    print("Chunking")
    print("-" * 80)
    print(f"Large documents chunked: {stats.large_documents_chunked:,}")
    print(f"Chunks created:          {stats.chunks_created:,}")
    print(f"Small tails merged:      {stats.chunk_tails_merged:,}")
    print()
    print("Filtered")
    print("-" * 80)
    print(f"Empty:                   {stats.skipped_empty:,}")
    print(f"Too short:               {stats.skipped_too_short:,}")
    print(f"  short chunks:          {stats.chunks_skipped_too_short:,}")
    print(f"Too long after chunking: {stats.skipped_too_long:,}")
    print(f"Exact duplicates:        {stats.skipped_duplicate:,}")
    print(
        f"Special-token collision: {stats.skipped_special_token_collision:,}"
    )
    print()
    print("Source contribution cap")
    print("-" * 80)
    if max_source_document_bytes is None:
        print("Maximum/source document: disabled")
    else:
        print(
            f"Maximum/source document: {max_source_document_bytes:,} UTF-8 bytes"
        )
    print(f"Source documents capped: {stats.source_documents_capped:,}")
    print(
        f"Candidate chunks rejected: {stats.source_cap_candidates_rejected:,}"
    )
    print(f"Candidate bytes rejected:  {stats.source_cap_bytes_rejected:,}")
    print()
    print("Final source concentration")
    print("-" * 80)
    print(
        "Distinct source files:      "
        f"{int(source_summary['distinct_source_files']):,}"
    )
    print(
        "Distinct source documents:  "
        f"{int(source_summary['distinct_source_documents']):,}"
    )
    print(
        "Largest source document:    "
        f"{int(source_summary['largest_source_document_bytes']):,} bytes "
        f"({float(source_summary['largest_source_document_fraction']):.2%})"
    )
    print()
    print("Category distribution")
    print("-" * 80)
    print(
        f"{'Category':<18} {'Weight':>9} {'Accepted':>10} "
        f"{'Selected':>10} {'Bytes':>12}"
    )

    for category in sorted(weights):
        cat = category_stats[category]
        print(
            f"{category:<18} "
            f"{weights[category]:>8.1%} "
            f"{cat.documents_accepted_before_sampling:>10,} "
            f"{cat.documents_selected:>10,} "
            f"{cat.bytes_selected:>12,}"
        )

    print()
    print(f"Corpus SHA-256: {corpus_sha256}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        validate_args(args)
        weights = resolve_weights(args.source, args.weight)

        output = ensure_output_directory(
            args.output,
            overwrite=args.overwrite,
        )

        stats = BuildStats()
        rejected_documents: list[RejectedDocument] = []
        category_stats: dict[str, CategoryStats] = {
            source.category: CategoryStats()
            for source in args.source
        }

        if not args.quiet:
            print("Discovering and preparing tokenizer source documents...")

        candidates = load_candidates(
            args.source,
            text_field=args.text_field,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            chunk_target_chars=args.chunk_target_chars,
            min_tail_chars=args.min_tail_chars,
            allow_special_token_text=args.allow_special_token_text,
            stats=stats,
            category_stats=category_stats,
            rejected_documents=rejected_documents,
        )

        if args.max_source_document_bytes is not None:
            candidates = cap_source_document_contribution(
                candidates,
                max_source_document_bytes=args.max_source_document_bytes,
                seed=args.seed,
                stats=stats,
                category_stats=category_stats,
            )

        accepted_count = sum(len(items) for items in candidates.values())
        if accepted_count == 0:
            raise ValueError(
                "no documents remained after normalization/filtering"
            )

        if args.target_bytes is None:
            selected = select_all(
                candidates,
                seed=args.seed,
            )
        else:
            selected = select_weighted_by_bytes(
                candidates,
                weights=weights,
                target_bytes=args.target_bytes,
                seed=args.seed,
            )

        selected = apply_document_limit(
            selected,
            max_documents=args.max_documents,
        )

        if not selected:
            raise ValueError("sampling selected zero documents")

        update_selected_stats(
            selected,
            stats=stats,
            category_stats=category_stats,
        )
        source_summary = source_contribution_summary(selected)

        corpus_path = output / DEFAULT_OUTPUT_FILENAME
        corpus_sha256 = write_jsonl(corpus_path, selected)

        manifest = {
            "format": "llm_tokenizer_training_corpus",
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_file": DEFAULT_OUTPUT_FILENAME,
            "corpus_sha256": corpus_sha256,
            "normalization": {
                "encoding": "UTF-8",
                "unicode_form": "NFC",
                "preserve_case": True,
                "case_folding": False,
                "line_endings": "LF",
            },
            "settings": {
                "target_bytes": args.target_bytes,
                "max_documents": args.max_documents,
                "min_chars": args.min_chars,
                "max_chars": args.max_chars,
                "chunk_target_chars": args.chunk_target_chars,
                "min_tail_chars": args.min_tail_chars,
                "max_source_document_bytes": args.max_source_document_bytes,
                "seed": args.seed,
                "text_field": args.text_field,
                "allow_special_token_text": args.allow_special_token_text,
            },
            "sources": [
                {
                    "category": source.category,
                    "path": str(source.path.expanduser().resolve()),
                    "normalized_weight": weights[source.category],
                }
                for source in args.source
            ],
            "special_tokens_reserved": list(DEFAULT_SPECIAL_TOKENS),
            "result": {
                "documents": stats.documents_selected,
                "characters": stats.selected_characters,
                "utf8_bytes": stats.selected_utf8_bytes,
                "lines": stats.selected_lines,
                "category_distribution": category_distribution(selected),
                "source_contribution": source_summary,
            },
        }

        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_sha256": corpus_sha256,
            "stats": asdict(stats),
            "categories": {
                category: asdict(category_stats[category])
                for category in sorted(category_stats)
            },
            "weights": weights,
            "filters": {
                "min_chars": args.min_chars,
                "max_chars": args.max_chars,
                "chunk_target_chars": args.chunk_target_chars,
                "min_tail_chars": args.min_tail_chars,
                "max_source_document_bytes": args.max_source_document_bytes,
                "oversized_documents_chunked": True,
                "exact_deduplication": True,
                "special_token_collision_rejected": (
                    not args.allow_special_token_text
                ),
            },
            "source_document_cap": {
                "enabled": args.max_source_document_bytes is not None,
                "max_bytes": args.max_source_document_bytes,
                "source_documents_capped": stats.source_documents_capped,
                "candidates_rejected": stats.source_cap_candidates_rejected,
                "bytes_rejected": stats.source_cap_bytes_rejected,
            },
            "source_contribution": source_summary,
            "rejected_documents": [
                asdict(item)
                for item in rejected_documents
            ],
        }

        write_json(output / "manifest.json", manifest)
        write_json(output / "build_report.json", report)

        if not args.quiet:
            print_summary(
                output=output,
                weights=weights,
                stats=stats,
                category_stats=category_stats,
                corpus_sha256=corpus_sha256,
                max_source_document_bytes=args.max_source_document_bytes,
                source_summary=source_summary,
            )
            print()
            print("Generated:")
            print(f"  {corpus_path}")
            print(f"  {output / 'manifest.json'}")
            print(f"  {output / 'build_report.json'}")

        return 0

    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
