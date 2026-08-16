r"""Build model-ready train/validation token shards from cleaned text documents.

Example (PowerShell)::

    python scripts\build_pretraining_dataset.py `
        --input data\cleaned\pretraining `
        --tokenizer artifacts\tokenizer-E011\tokenizer.json `
        --output data\tokenized\v0.1 `
        --validation-fraction 0.01 `
        --context-length 2048 `
        --tokens-per-shard 1000000

Supported inputs
----------------
* ``.txt`` / ``.text`` / ``.md``: one file is one document.
* ``.jsonl`` / ``.ndjson``: one object per line, with text in ``--text-field``.

Any of these may additionally be gzip compressed, for example
``corpus-00000.jsonl.gz``.  The cleaned pretraining corpus is written that way,
so this reader consumes it directly with no intermediate decompression step.
Compression is transparent: a plain file and its gzipped copy yield byte-identical
documents.

For JSONL records, ``document_id`` is used when present.  Otherwise a stable id
is derived from file path + line number.  If records contain ``source_file`` and
``source_record``, all chunks from that original source document are assigned to
the same train/validation split; ``source_chunk`` is intentionally ignored.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import IO, Iterator, Sequence

from llm.data.dataset_training import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    TrainingDocument,
    build_pretraining_dataset,
)
from llm.data.packing import DEFAULT_CONTEXT_LENGTH
from llm.data.shards import DEFAULT_TOKENS_PER_SHARD
from llm.tokenizer.normalizer import normalize_text
from llm.tokenizer.tokenizer import Tokenizer


SUPPORTED_TEXT_SUFFIXES = {".txt", ".text", ".md"}
SUPPORTED_JSONL_SUFFIXES = {".jsonl", ".ndjson"}
SUPPORTED_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | SUPPORTED_JSONL_SUFFIXES
COMPRESSED_SUFFIX = ".gz"
DEFAULT_TEXT_FIELD = "text"


def logical_suffix(path: Path) -> str:
    """Return the content suffix, ignoring a trailing ``.gz``.

    ``Path.suffix`` reports ``.gz`` for ``corpus-00000.jsonl.gz``, which would
    make the cleaned corpus look like an unsupported file type.  Compression is
    a transport detail; what matters is the format underneath it.
    """

    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] == COMPRESSED_SUFFIX:
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes else ""


def is_compressed(path: Path) -> bool:
    return path.suffix.lower() == COMPRESSED_SUFFIX


def open_text_file(path: Path) -> IO[str]:
    """Open a possibly gzipped text file for reading as UTF-8.

    ``errors="strict"`` is deliberate and matches the corpus builder: silently
    replacing undecodable bytes here would put mojibake into the token stream
    with nothing downstream able to detect it.
    """

    if is_compressed(path):
        return gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="")
    return path.open("r", encoding="utf-8", errors="strict", newline="")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Cleaned input file or directory; may be repeated.",
    )
    parser.add_argument("--tokenizer", required=True, help="Frozen tokenizer.json path.")
    parser.add_argument("--output", required=True, help="Output dataset directory.")
    parser.add_argument("--text-field", default=DEFAULT_TEXT_FIELD)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
    )
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=DEFAULT_TOKENS_PER_SHARD,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def discover_input_files(inputs: Sequence[str]) -> list[Path]:
    files: set[Path] = set()
    for raw in inputs:
        path = Path(raw).resolve()
        if path.is_file():
            if logical_suffix(path) not in SUPPORTED_SUFFIXES:
                raise ValueError(f"unsupported input file type: {path}")
            files.add(path)
        elif path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and logical_suffix(candidate) in SUPPORTED_SUFFIXES:
                    files.add(candidate.resolve())
        else:
            raise ValueError(f"input path is neither a file nor directory: {path}")

    ordered = sorted(files, key=lambda p: str(p).casefold())
    if not ordered:
        raise ValueError("no supported cleaned pretraining files were found")
    return ordered


def _record_split_group(record: dict, document_id: str) -> str:
    if "split_group" in record:
        value = record["split_group"]
        if not isinstance(value, str) or not value:
            raise ValueError("JSONL split_group must be a non-empty string")
        return value

    if "source_file" in record:
        source_file = record["source_file"]
        source_record = record.get("source_record")
        if not isinstance(source_file, str) or not source_file:
            raise ValueError("JSONL source_file must be a non-empty string")
        # source_chunk is intentionally excluded: chunks from the same original
        # source document must stay in one split.
        return json.dumps(
            [source_file, source_record],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    return document_id


def iter_training_documents(
    files: Sequence[Path],
    *,
    text_field: str,
) -> Iterator[TrainingDocument]:
    for path in files:
        suffix = logical_suffix(path)

        if suffix in SUPPORTED_TEXT_SUFFIXES:
            with open_text_file(path) as handle:
                text = handle.read()
            if normalize_text(text) == "":
                continue
            document_id = str(path)
            yield TrainingDocument(
                document_id=document_id,
                text=text,
                split_group=document_id,
            )
            continue

        if suffix in SUPPORTED_JSONL_SUFFIXES:
            with open_text_file(path) as handle:
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
                            f"missing text field {text_field!r} in {path} at line {line_number}"
                        )
                    text = record[text_field]
                    if not isinstance(text, str):
                        raise ValueError(
                            f"field {text_field!r} must be a string in {path} at line {line_number}"
                        )
                    if normalize_text(text) == "":
                        continue

                    raw_document_id = record.get("document_id")
                    if raw_document_id is None:
                        document_id = f"{path}#record={line_number}"
                    elif isinstance(raw_document_id, str) and raw_document_id:
                        document_id = raw_document_id
                    else:
                        raise ValueError(
                            f"document_id must be a non-empty string in {path} at line {line_number}"
                        )

                    yield TrainingDocument(
                        document_id=document_id,
                        text=text,
                        split_group=_record_split_group(record, document_id),
                    )
            continue

        raise ValueError(f"unsupported cleaned pretraining file: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    files = discover_input_files(args.input)
    tokenizer = Tokenizer.load(args.tokenizer)

    # A factory, not a list: the builder streams the corpus several times and
    # must be able to restart it.  Collecting it here would put the entire
    # cleaned corpus in memory before tokenization had even begun.
    def document_source() -> Iterator[TrainingDocument]:
        return iter_training_documents(files, text_field=args.text_field)

    if next(document_source(), None) is None:
        raise ValueError("cleaned pretraining input contains no non-empty documents")

    result = build_pretraining_dataset(
        document_source,
        tokenizer,
        args.output,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        context_length=args.context_length,
        tokens_per_shard=args.tokens_per_shard,
        overwrite=args.overwrite,
    )

    manifest = json.loads(result.dataset_manifest_path.read_text(encoding="utf-8"))
    train = manifest["splits"]["train"]
    validation = manifest["splits"]["validation"]

    def row(label: str, key: str) -> str:
        return f"{label:<26}{train[key]:>18,}{validation[key]:>18,}"

    print()
    print("Pretraining Dataset Build")
    print("=" * 80)
    print(f"Output:                   {result.output_dir}")
    print(f"Tokenizer vocabulary:     {tokenizer.vocab_size:,}")
    print(f"Context length:           {result.context_length:,}")
    print()
    print(f"{'':<26}{'train':>18}{'validation':>18}")
    print("-" * 80)
    print(row("documents", "document_count"))
    print(row("split groups", "split_group_count"))
    print(row("normalized characters", "normalized_characters"))
    print(row("normalized UTF-8 bytes", "normalized_utf8_bytes"))
    print(row("tokens", "total_tokens"))
    print(row("shards", "shard_count"))
    print(row(f"windows of {result.context_length}", "complete_examples"))
    print(row("tail tokens (unused)", "tail_tokens"))
    print("-" * 80)

    total_tokens = train["total_tokens"] + validation["total_tokens"]
    total_bytes = train["normalized_utf8_bytes"] + validation["normalized_utf8_bytes"]
    print(f"{'total tokens':<26}{total_tokens:>36,}")
    if total_tokens:
        print(f"{'bytes per token':<26}{total_bytes / total_tokens:>36.3f}")
    print()
    print(f"Dataset manifest:         {result.dataset_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
