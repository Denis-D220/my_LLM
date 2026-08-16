"""Train Tokenizer v0.1 for the LLM project.

This script is the command-line entry point for training the project's
case-preserving, Unicode-safe byte-level BPE tokenizer.

It deliberately keeps corpus ingestion separate from the tokenizer core:

    corpus files -> normalized documents -> Tokenizer.train() -> artifacts

Supported input formats
-----------------------
* .txt / .md / .text : one file is treated as one document by default
* .jsonl              : one JSON object per line; text is read from --text-field

Examples
--------
Train a small development tokenizer::

    python scripts/train_tokenizer.py \
        --input data/samples/tokenizer_sample.txt \
        --vocab-size 512 \
        --output artifacts/tokenizer-dev

Train Tokenizer v0.1::

    python scripts/train_tokenizer.py \
        --input data/tokenizer_training \
        --vocab-size 24000 \
        --output artifacts/tokenizer-v0.1

Notes
-----
The current BPE implementation is intentionally written in pure Python for
clarity and correctness.  It materializes the training documents in memory and
recomputes pair statistics during merge learning.  Therefore, use corpus limits
for early experiments.  Before multi-gigabyte production training, the BPE
training kernel should be optimized while preserving the same public tokenizer
format and semantics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Iterable, Iterator, Sequence

from llm.tokenizer.normalizer import normalize_text
from llm.tokenizer.tokenizer import DEFAULT_SPECIAL_TOKENS, Tokenizer


SUPPORTED_TEXT_SUFFIXES = {".txt", ".text", ".md"}
SUPPORTED_JSONL_SUFFIXES = {".jsonl", ".ndjson"}
SUPPORTED_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | SUPPORTED_JSONL_SUFFIXES

DEFAULT_VOCAB_SIZE = 24_000
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MIN_PAIR_FREQUENCY = 2


@dataclass(frozen=True)
class CorpusStats:
    files_discovered: int
    documents_seen: int
    documents_used: int
    documents_skipped_empty: int
    characters: int
    utf8_bytes: int
    lines: int
    special_token_collisions: int
    corpus_sha256: str


@dataclass(frozen=True)
class TrainingSettings:
    vocab_size: int
    min_pair_frequency: int
    text_field: str
    max_documents: int | None
    max_bytes: int | None
    special_tokens: list[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the LLM project's byte-level BPE tokenizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        type=Path,
        help="Input files and/or directories. Directories are searched recursively.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Artifact directory, e.g. artifacts/tokenizer-v0.1.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="Total vocabulary size including reserved special tokens.",
    )
    parser.add_argument(
        "--min-pair-frequency",
        type=int,
        default=DEFAULT_MIN_PAIR_FREQUENCY,
        help="Minimum corpus frequency required to learn a BPE merge.",
    )
    parser.add_argument(
        "--text-field",
        default=DEFAULT_TEXT_FIELD,
        help="JSONL field containing document text.",
    )
    parser.add_argument(
        "--max-documents",
        type=positive_int,
        default=None,
        help="Optional development limit on the number of non-empty documents.",
    )
    parser.add_argument(
        "--max-bytes",
        type=positive_int,
        default=None,
        help=(
            "Optional development limit on normalized UTF-8 corpus bytes. "
            "The document that would exceed the limit is not included."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress/status messages except errors.",
    )

    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def validate_args(args: argparse.Namespace) -> None:
    if args.vocab_size < 256 + len(DEFAULT_SPECIAL_TOKENS):
        raise ValueError(
            "--vocab-size is too small for 256 byte tokens plus "
            f"{len(DEFAULT_SPECIAL_TOKENS)} special tokens"
        )
    if args.min_pair_frequency < 1:
        raise ValueError("--min-pair-frequency must be >= 1")
    if not args.text_field:
        raise ValueError("--text-field cannot be empty")


def discover_input_files(paths: Iterable[Path]) -> list[Path]:
    """Resolve supported corpus files in deterministic path order."""

    files: set[Path] = set()

    for supplied in paths:
        path = supplied.expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"input path does not exist: {supplied}")

        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(
                    f"unsupported input file type: {path} "
                    f"(supported: {', '.join(sorted(SUPPORTED_SUFFIXES))})"
                )
            files.add(path)
            continue

        if path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                    files.add(candidate.resolve())
            continue

        raise ValueError(f"input path is neither a regular file nor directory: {path}")

    ordered = sorted(files, key=lambda p: str(p).casefold())
    if not ordered:
        raise ValueError("no supported corpus files were found")
    return ordered


def iter_file_documents(path: Path, *, text_field: str) -> Iterator[str]:
    """Yield documents from one corpus file."""

    suffix = path.suffix.lower()

    if suffix in SUPPORTED_TEXT_SUFFIXES:
        # For v0.1 one ordinary text file is one document.  Large files should
        # later be produced as already-sharded corpus documents upstream.
        yield path.read_text(encoding="utf-8", errors="strict")
        return

    if suffix in SUPPORTED_JSONL_SUFFIXES:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path} at line {line_number}: {exc.msg}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL record must be an object in {path} at line {line_number}"
                    )

                if text_field not in record:
                    raise ValueError(
                        f"missing text field {text_field!r} in {path} "
                        f"at line {line_number}"
                    )

                text = record[text_field]
                if not isinstance(text, str):
                    raise ValueError(
                        f"field {text_field!r} must be a string in {path} "
                        f"at line {line_number}"
                    )

                yield text
        return

    raise ValueError(f"unsupported corpus file: {path}")


def load_corpus(
    files: Sequence[Path],
    *,
    text_field: str,
    max_documents: int | None,
    max_bytes: int | None,
    special_tokens: Sequence[str],
) -> tuple[list[str], CorpusStats]:
    """Load, normalize and measure the tokenizer-training corpus.

    The hash is computed over length-prefixed normalized UTF-8 documents, so
    document boundaries are part of the corpus identity and cannot collide
    through simple concatenation.
    """

    documents: list[str] = []
    documents_seen = 0
    skipped_empty = 0
    characters = 0
    utf8_bytes = 0
    line_count = 0
    special_collisions = 0
    digest = hashlib.sha256()

    stop = False

    for path in files:
        if stop:
            break

        for raw_text in iter_file_documents(path, text_field=text_field):
            documents_seen += 1
            text = normalize_text(raw_text)

            if text == "":
                skipped_empty += 1
                continue

            encoded = text.encode("utf-8", errors="strict")

            if max_documents is not None and len(documents) >= max_documents:
                stop = True
                break

            if max_bytes is not None and utf8_bytes + len(encoded) > max_bytes:
                stop = True
                break

            documents.append(text)
            characters += len(text)
            utf8_bytes += len(encoded)
            line_count += text.count("\n") + (1 if text else 0)

            collision_count = sum(text.count(token) for token in special_tokens)
            special_collisions += collision_count

            # Domain-separated, length-prefixed document hashing.
            digest.update(b"LLM_TOKENIZER_DOC_V1\x00")
            digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
            digest.update(encoded)

    stats = CorpusStats(
        files_discovered=len(files),
        documents_seen=documents_seen,
        documents_used=len(documents),
        documents_skipped_empty=skipped_empty,
        characters=characters,
        utf8_bytes=utf8_bytes,
        lines=line_count,
        special_token_collisions=special_collisions,
        corpus_sha256=digest.hexdigest(),
    )

    return documents, stats


def ensure_output_directory(path: Path, *, overwrite: bool) -> Path:
    output = path.expanduser().resolve()

    if output.exists() and not output.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output}")

    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output}\n"
            "Use --overwrite to replace tokenizer artifact files."
        )

    output.mkdir(parents=True, exist_ok=True)
    return output


def token_record(tokenizer: Tokenizer, token_id: int) -> dict:
    """Create a readable JSON vocabulary entry without assuming valid UTF-8."""

    special = tokenizer.id_to_special_token.get(token_id)
    if special is not None:
        return {
            "id": token_id,
            "type": "special",
            "text": special,
        }

    value = tokenizer.bpe.vocab.get(token_id)
    if value is None:
        return {
            "id": token_id,
            "type": "unassigned",
        }

    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        text = None

    return {
        "id": token_id,
        "type": "byte" if token_id < 256 else "bpe",
        "bytes_hex": value.hex(),
        "utf8": text,
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_artifacts(
    tokenizer: Tokenizer,
    output: Path,
    *,
    settings: TrainingSettings,
    corpus_stats: CorpusStats,
    input_files: Sequence[Path],
    elapsed_seconds: float,
) -> None:
    """Save the complete tokenizer plus human/audit-friendly sidecars."""

    tokenizer.save(output / "tokenizer.json")

    write_json(
        output / "config.json",
        {
            "tokenizer_version": "0.1",
            "algorithm": "byte_level_bpe",
            "encoding": "UTF-8",
            "unicode_normalization": "NFC",
            "preserve_case": True,
            "settings": asdict(settings),
        },
    )

    vocabulary = [token_record(tokenizer, token_id) for token_id in range(tokenizer.vocab_size)]
    write_json(output / "vocab.json", vocabulary)

    merges = [
        {
            "rank": rank,
            "left_id": left,
            "right_id": right,
            "new_token_id": new_token_id,
        }
        for rank, ((left, right), new_token_id) in enumerate(tokenizer.bpe.merges.items())
    ]
    write_json(output / "merges.json", merges)

    write_json(
        output / "training_report.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 6),
            "corpus": asdict(corpus_stats),
            "input_files": [str(path) for path in input_files],
            "result": {
                "requested_vocab_size": tokenizer.vocab_size,
                "learned_content_vocab_size": tokenizer.learned_content_vocab_size,
                "special_token_count": tokenizer.special_token_count,
                "learned_merge_count": len(tokenizer.bpe.merges),
                "unused_content_ids": (
                    tokenizer.content_vocab_limit - tokenizer.learned_content_vocab_size
                ),
            },
        },
    )


def validate_saved_tokenizer(
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    documents: Sequence[str],
) -> None:
    """Reload the artifact and prove deterministic encode/decode behavior."""

    loaded = Tokenizer.load(tokenizer_path)

    if loaded.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("saved tokenizer vocabulary size changed after reload")

    # Validate a bounded sample so artifact validation stays quick even when the
    # tokenizer-training corpus grows.
    sample = list(documents[: min(10, len(documents))])
    sample.extend(
        [
            "HTTP != http; MHz != mHz; R = 4.7 kΩ ± 5%.",
            "ΔV = I × R; T = 25 °C; C = 10 µF.",
            "class HTTPClient:\n    MAX_RETRIES = 5\n",
        ]
    )

    for text in sample:
        normalized = normalize_text(text)
        original_ids = tokenizer.encode(normalized, parse_special_tokens=False)
        loaded_ids = loaded.encode(normalized, parse_special_tokens=False)

        if original_ids != loaded_ids:
            raise RuntimeError("saved tokenizer produces different token IDs after reload")

        decoded = loaded.decode(loaded_ids)
        if decoded != normalized:
            raise RuntimeError(
                "saved tokenizer failed round-trip validation:\n"
                f"expected={normalized!r}\n"
                f"decoded={decoded!r}"
            )


def status(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(message, flush=True)


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        validate_args(args)
        output = ensure_output_directory(args.output, overwrite=args.overwrite)
        files = discover_input_files(args.input)

        status(f"Discovered {len(files)} corpus file(s).", quiet=args.quiet)
        status("Loading and normalizing corpus...", quiet=args.quiet)

        documents, corpus_stats = load_corpus(
            files,
            text_field=args.text_field,
            max_documents=args.max_documents,
            max_bytes=args.max_bytes,
            special_tokens=DEFAULT_SPECIAL_TOKENS,
        )

        if not documents:
            raise ValueError("the corpus contains no non-empty training documents")

        status(
            f"Using {corpus_stats.documents_used:,} document(s), "
            f"{human_bytes(corpus_stats.utf8_bytes)} normalized UTF-8, "
            f"{corpus_stats.characters:,} characters.",
            quiet=args.quiet,
        )

        if corpus_stats.special_token_collisions:
            status(
                "WARNING: corpus contains "
                f"{corpus_stats.special_token_collisions:,} literal reserved special-token "
                "occurrence(s). Base-corpus tokenization should later use "
                "parse_special_tokens=False and insert BOS/EOS explicitly.",
                quiet=args.quiet,
            )

        settings = TrainingSettings(
            vocab_size=args.vocab_size,
            min_pair_frequency=args.min_pair_frequency,
            text_field=args.text_field,
            max_documents=args.max_documents,
            max_bytes=args.max_bytes,
            special_tokens=list(DEFAULT_SPECIAL_TOKENS),
        )

        status(
            f"Training byte-level BPE: vocab_size={args.vocab_size:,}, "
            f"min_pair_frequency={args.min_pair_frequency}...",
            quiet=args.quiet,
        )

        started = time.perf_counter()
        tokenizer = Tokenizer.train(
            texts=documents,
            vocab_size=args.vocab_size,
            special_tokens=DEFAULT_SPECIAL_TOKENS,
            min_pair_frequency=args.min_pair_frequency,
        )
        elapsed = time.perf_counter() - started

        status(
            f"Training finished in {elapsed:.3f}s; learned "
            f"{len(tokenizer.bpe.merges):,} merges and "
            f"{tokenizer.learned_content_vocab_size:,} content tokens.",
            quiet=args.quiet,
        )

        save_artifacts(
            tokenizer,
            output,
            settings=settings,
            corpus_stats=corpus_stats,
            input_files=files,
            elapsed_seconds=elapsed,
        )

        status("Reloading and validating tokenizer artifact...", quiet=args.quiet)
        validate_saved_tokenizer(tokenizer, output / "tokenizer.json", documents)

        status("Tokenizer artifact validated successfully.", quiet=args.quiet)
        status(f"Output: {output}", quiet=args.quiet)

        if tokenizer.learned_content_vocab_size < tokenizer.content_vocab_limit:
            status(
                "NOTE: The corpus/minimum frequency did not produce enough merges "
                "to fill the requested content vocabulary. This is expected for small "
                "development corpora; use a larger representative corpus before "
                "freezing tokenizer-v0.1.",
                quiet=args.quiet,
            )

        return 0

    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())