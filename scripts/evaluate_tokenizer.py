"""

This script measures tokenizer *quality* after the tokenizer core has already
passed correctness/unit tests.  It is designed for the project's
case-preserving, Unicode-safe byte-level BPE tokenizer.

The evaluator reports:

* round-trip correctness;
* token counts and compression statistics;
* tokens per word-like unit;
* UTF-8 bytes / characters per token;
* byte-token versus learned-BPE-token usage;
* vocabulary utilization;
* token byte-length distribution;
* most frequently used tokens;
* longest learned tokens actually used;
* built-in technical probe results and token segmentations;
* case-sensitive encoding checks.

Important
---------
For base-corpus evaluation, literal strings such as ``<|system|>`` are treated
as ordinary text by using ``parse_special_tokens=False``.  Reserved control
symbols should only become special token IDs when the application explicitly
constructs assistant/chat sequences.

Examples
--------
Evaluate one held-out text file::

    python scripts/evaluate_tokenizer.py \
        --tokenizer artifacts/tokenizer-dev/tokenizer.json \
        --input data/evaluation/tokenizer_eval.txt

Evaluate a directory and write a report explicitly::

    python scripts/evaluate_tokenizer.py \
        --tokenizer artifacts/tokenizer-v0.1/tokenizer.json \
        --input data/evaluation \
        --output artifacts/tokenizer-v0.1/evaluation_report.json

Run only the built-in probes::

    python scripts/evaluate_tokenizer.py \
        --tokenizer artifacts/tokenizer-dev/tokenizer.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Iterator, Sequence

from llm.tokenizer.normalizer import normalize_text
from llm.tokenizer.tokenizer import Tokenizer


SUPPORTED_TEXT_SUFFIXES = {".txt", ".text", ".md"}
SUPPORTED_JSONL_SUFFIXES = {".jsonl", ".ndjson"}
SUPPORTED_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | SUPPORTED_JSONL_SUFFIXES
DEFAULT_TEXT_FIELD = "text"
DEFAULT_TOP_TOKENS = 25
DEFAULT_LONGEST_TOKENS = 20

# This is deliberately called a "word-like unit" rather than a word.  \w
# keeps identifiers such as calculate_voltage and STM32F411 together, which
# makes the metric more useful for a technical/code-oriented tokenizer.
WORD_LIKE_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class Probe:
    category: str
    name: str
    text: str


BUILTIN_PROBES: tuple[Probe, ...] = (
    Probe(
        "general_english",
        "plain_sentence",
        "The system processes information and returns a clear response to the user.",
    ),
    Probe(
        "electronics",
        "engineering_units",
        "A 10 µF decoupling capacitor is connected between VCC and GND near the MCU; R = 4.7 kΩ ± 5%.",
    ),
    Probe(
        "electronics",
        "embedded_system",
        "The STM32F411 communicates with the VL53L8CX over I²C at 400 kHz and operates from a 3.3 V rail.",
    ),
    Probe(
        "ai_ml",
        "transformer_terms",
        "A decoder-only Transformer uses causal self-attention, token embeddings, RMSNorm, RoPE, and SwiGLU feed-forward layers.",
    ),
    Probe(
        "ai_ml",
        "training_terms",
        "Backpropagation computes gradients of the cross-entropy loss so AdamW can update the model parameters.",
    ),
    Probe(
        "python",
        "python_function",
        "def calculate_voltage(current: float, resistance: float) -> float:\n    return current * resistance\n",
    ),
    Probe(
        "cpp",
        "cpp_identifiers",
        "std::vector<std::string> values;\nuint32_t tick = HAL_GetTick();\nGPIO_WritePin(GPIOA, GPIO_PIN_13, GPIO_PIN_SET);\n",
    ),
    Probe(
        "networking",
        "protocols_and_url",
        "HTTPClient sends a GET request to https://example.com/api/v1/devices?id=42 from 192.168.1.100.",
    ),
    Probe(
        "math_unicode",
        "technical_unicode",
        "ΔV = I × R; A = πr²; x ≤ y; √2 ≈ 1.41421356; T = 25 °C.",
    ),
    Probe(
        "case_sensitivity",
        "case_distinctions",
        "HTTP http GET get RAM ram Python python MHz mHz ClassName className MAX_CURRENT max_current",
    ),
    Probe(
        "numbers",
        "technical_numbers",
        "0xDEADBEEF 0b10101010 1.25e-6 6.022e23 2026-08-08 v1.2.3 STM32F411 VL53L8CX",
    ),
    Probe(
        "structured_text",
        "json_and_path",
        '{"device":"STM32F411","temperature":23.5,"enabled":true}\n/home/user/project/src/main.py\n',
    ),
)

CASE_CHECKS: tuple[tuple[str, str], ...] = (
    ("HTTP", "http"),
    ("RAM", "ram"),
    ("GET", "get"),
    ("Python", "python"),
    ("MHz", "mHz"),
    ("ClassName", "classname"),
    ("MAX_CURRENT", "max_current"),
)


@dataclass
class AggregateMetrics:
    documents: int = 0
    characters: int = 0
    utf8_bytes: int = 0
    lines: int = 0
    word_like_units: int = 0
    tokens: int = 0
    byte_tokens: int = 0
    bpe_tokens: int = 0
    special_tokens: int = 0
    roundtrip_failures: int = 0
    literal_special_token_occurrences: int = 0


# ---------------------------------------------------------------------------
# CLI / corpus loading
# ---------------------------------------------------------------------------


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate tokenizer quality and technical-text behavior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--tokenizer",
        "-t",
        required=True,
        type=Path,
        help="Path to tokenizer.json produced by train_tokenizer.py.",
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Optional held-out .txt/.md/.jsonl files or directories. "
            "If omitted, only built-in technical probes are evaluated."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "JSON report path. By default, evaluation_report.json is written "
            "beside tokenizer.json."
        ),
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
        help="Optional development limit on evaluated non-empty documents.",
    )
    parser.add_argument(
        "--max-bytes",
        type=positive_int,
        default=None,
        help="Optional limit on normalized UTF-8 bytes evaluated from --input.",
    )
    parser.add_argument(
        "--top-tokens",
        type=positive_int,
        default=DEFAULT_TOP_TOKENS,
        help="Number of most-frequent emitted tokens stored in the report.",
    )
    parser.add_argument(
        "--longest-tokens",
        type=positive_int,
        default=DEFAULT_LONGEST_TOKENS,
        help="Number of longest used learned-BPE tokens stored in the report.",
    )
    parser.add_argument(
        "--show-segmentation",
        action="store_true",
        help="Print each built-in probe's token segmentation to the console.",
    )
    parser.add_argument(
        "--no-probes",
        action="store_true",
        help="Skip the built-in technical probe suite.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable console summary; JSON is still written.",
    )

    return parser.parse_args(argv)


def discover_input_files(paths: Iterable[Path]) -> list[Path]:
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
        raise ValueError("no supported evaluation files were found")
    return ordered


def iter_file_documents(path: Path, *, text_field: str) -> Iterator[str]:
    suffix = path.suffix.lower()

    if suffix in SUPPORTED_TEXT_SUFFIXES:
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
                        f"missing text field {text_field!r} in {path} at line {line_number}"
                    )
                text = record[text_field]
                if not isinstance(text, str):
                    raise ValueError(
                        f"field {text_field!r} must be a string in {path} at line {line_number}"
                    )
                yield text
        return

    raise ValueError(f"unsupported evaluation file: {path}")


def load_documents(
    files: Sequence[Path],
    *,
    text_field: str,
    max_documents: int | None,
    max_bytes: int | None,
) -> list[str]:
    documents: list[str] = []
    bytes_used = 0

    for path in files:
        for raw_text in iter_file_documents(path, text_field=text_field):
            text = normalize_text(raw_text)
            if text == "":
                continue

            encoded = text.encode("utf-8", errors="strict")
            if max_documents is not None and len(documents) >= max_documents:
                return documents
            if max_bytes is not None and bytes_used + len(encoded) > max_bytes:
                return documents

            documents.append(text)
            bytes_used += len(encoded)

    return documents


# ---------------------------------------------------------------------------
# Token inspection helpers
# ---------------------------------------------------------------------------


def token_kind(tokenizer: Tokenizer, token_id: int) -> str:
    if token_id in tokenizer.id_to_special_token:
        return "special"
    if token_id < 256:
        return "byte"
    if token_id in tokenizer.bpe.vocab:
        return "bpe"
    return "unassigned"


def token_raw_bytes(tokenizer: Tokenizer, token_id: int) -> bytes | None:
    value = tokenizer.id_to_token(token_id)
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return None


def printable_text(value: str) -> str:
    """Return a terminal-friendly escaped representation without outer quotes."""

    return json.dumps(value, ensure_ascii=False)[1:-1]


def token_record(tokenizer: Tokenizer, token_id: int, *, count: int | None = None) -> dict:
    kind = token_kind(tokenizer, token_id)
    value = tokenizer.id_to_token(token_id)

    record: dict[str, object] = {
        "id": token_id,
        "type": kind,
    }
    if count is not None:
        record["count"] = count

    if isinstance(value, str):
        encoded = value.encode("utf-8")
        record.update(
            {
                "text": value,
                "bytes_hex": encoded.hex(),
                "byte_length": len(encoded),
                "character_length": len(value),
            }
        )
        return record

    if isinstance(value, bytes):
        record["bytes_hex"] = value.hex()
        record["byte_length"] = len(value)
        try:
            text = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            text = None
        record["utf8"] = text
        record["character_length"] = len(text) if text is not None else None
        return record

    return record


def segmentation(tokenizer: Tokenizer, text: str) -> list[dict]:
    normalized = normalize_text(text)
    token_ids = tokenizer.encode(normalized, parse_special_tokens=False)
    return [token_record(tokenizer, token_id) for token_id in token_ids]


def length_bucket(byte_length: int) -> str:
    if byte_length == 1:
        return "1"
    if byte_length == 2:
        return "2"
    if byte_length == 3:
        return "3"
    if byte_length == 4:
        return "4"
    if byte_length <= 8:
        return "5-8"
    if byte_length <= 16:
        return "9-16"
    return "17+"


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def rounded(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


# ---------------------------------------------------------------------------
# Corpus evaluation
# ---------------------------------------------------------------------------


def evaluate_documents(
    tokenizer: Tokenizer,
    documents: Sequence[str],
    *,
    top_n: int,
    longest_n: int,
) -> dict:
    aggregate = AggregateMetrics()
    usage: Counter[int] = Counter()
    token_length_distribution: Counter[str] = Counter()
    failure_examples: list[dict] = []

    for document_index, raw_text in enumerate(documents):
        text = normalize_text(raw_text)
        encoded_bytes = text.encode("utf-8", errors="strict")
        token_ids = tokenizer.encode(text, parse_special_tokens=False)

        aggregate.documents += 1
        aggregate.characters += len(text)
        aggregate.utf8_bytes += len(encoded_bytes)
        aggregate.lines += text.count("\n") + (1 if text else 0)
        aggregate.word_like_units += len(WORD_LIKE_RE.findall(text))
        aggregate.tokens += len(token_ids)
        aggregate.literal_special_token_occurrences += sum(
            text.count(special) for special in tokenizer.special_tokens
        )

        usage.update(token_ids)

        for token_id in token_ids:
            kind = token_kind(tokenizer, token_id)
            if kind == "byte":
                aggregate.byte_tokens += 1
            elif kind == "bpe":
                aggregate.bpe_tokens += 1
            elif kind == "special":
                aggregate.special_tokens += 1

            raw_bytes = token_raw_bytes(tokenizer, token_id)
            if raw_bytes is not None:
                token_length_distribution[length_bucket(len(raw_bytes))] += 1

        try:
            decoded = tokenizer.decode(token_ids)
        except (TypeError, ValueError) as exc:
            aggregate.roundtrip_failures += 1
            if len(failure_examples) < 10:
                failure_examples.append(
                    {
                        "document_index": document_index,
                        "error": str(exc),
                        "text_prefix": text[:300],
                    }
                )
            continue

        if decoded != text:
            aggregate.roundtrip_failures += 1
            if len(failure_examples) < 10:
                failure_examples.append(
                    {
                        "document_index": document_index,
                        "error": "decoded text differs from normalized input",
                        "expected_prefix": text[:300],
                        "decoded_prefix": decoded[:300],
                    }
                )

    content_usage_ids = {
        token_id
        for token_id in usage
        if token_id < tokenizer.content_vocab_limit and token_id in tokenizer.bpe.vocab
    }
    learned_bpe_usage_ids = {token_id for token_id in usage if token_kind(tokenizer, token_id) == "bpe"}

    top_tokens = [
        token_record(tokenizer, token_id, count=count)
        for token_id, count in sorted(
            usage.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_n]
    ]

    longest_used_bpe = []
    for token_id in learned_bpe_usage_ids:
        raw = tokenizer.bpe.vocab.get(token_id)
        if raw is not None:
            longest_used_bpe.append((len(raw), usage[token_id], token_id))
    longest_used_bpe.sort(key=lambda item: (-item[0], -item[1], item[2]))

    longest_tokens = [
        token_record(tokenizer, token_id, count=count)
        for _, count, token_id in longest_used_bpe[:longest_n]
    ]

    metrics = {
        **asdict(aggregate),
        "tokens_per_document": rounded(safe_ratio(aggregate.tokens, aggregate.documents)),
        "tokens_per_word_like_unit": rounded(
            safe_ratio(aggregate.tokens, aggregate.word_like_units)
        ),
        "characters_per_token": rounded(
            safe_ratio(aggregate.characters, aggregate.tokens)
        ),
        "utf8_bytes_per_token": rounded(
            safe_ratio(aggregate.utf8_bytes, aggregate.tokens)
        ),
        "byte_token_ratio": rounded(safe_ratio(aggregate.byte_tokens, aggregate.tokens)),
        "bpe_token_ratio": rounded(safe_ratio(aggregate.bpe_tokens, aggregate.tokens)),
        "distinct_token_ids_used": len(usage),
        "distinct_content_token_ids_used": len(content_usage_ids),
        "content_vocabulary_utilization": rounded(
            safe_ratio(len(content_usage_ids), tokenizer.learned_content_vocab_size)
        ),
        "distinct_learned_bpe_ids_used": len(learned_bpe_usage_ids),
        "token_byte_length_distribution": {
            bucket: token_length_distribution.get(bucket, 0)
            for bucket in ("1", "2", "3", "4", "5-8", "9-16", "17+")
        },
        "roundtrip_failure_examples": failure_examples,
        "top_tokens": top_tokens,
        "longest_used_bpe_tokens": longest_tokens,
    }
    return metrics


# ---------------------------------------------------------------------------
# Built-in technical probes
# ---------------------------------------------------------------------------


def evaluate_probe(tokenizer: Tokenizer, probe: Probe) -> dict:
    text = normalize_text(probe.text)
    token_ids = tokenizer.encode(text, parse_special_tokens=False)
    bytes_count = len(text.encode("utf-8", errors="strict"))
    words = len(WORD_LIKE_RE.findall(text))

    decoded = tokenizer.decode(token_ids)
    roundtrip_ok = decoded == text

    byte_count = sum(token_kind(tokenizer, token_id) == "byte" for token_id in token_ids)
    bpe_count = sum(token_kind(tokenizer, token_id) == "bpe" for token_id in token_ids)

    return {
        "category": probe.category,
        "name": probe.name,
        "text": text,
        "characters": len(text),
        "utf8_bytes": bytes_count,
        "word_like_units": words,
        "tokens": len(token_ids),
        "tokens_per_word_like_unit": rounded(safe_ratio(len(token_ids), words)),
        "characters_per_token": rounded(safe_ratio(len(text), len(token_ids))),
        "utf8_bytes_per_token": rounded(safe_ratio(bytes_count, len(token_ids))),
        "byte_tokens": byte_count,
        "bpe_tokens": bpe_count,
        "byte_token_ratio": rounded(safe_ratio(byte_count, len(token_ids))),
        "roundtrip_ok": roundtrip_ok,
        "segmentation": segmentation(tokenizer, text),
    }


def evaluate_case_checks(tokenizer: Tokenizer) -> list[dict]:
    results: list[dict] = []
    for left, right in CASE_CHECKS:
        left_ids = tokenizer.encode(left, parse_special_tokens=False)
        right_ids = tokenizer.encode(right, parse_special_tokens=False)
        results.append(
            {
                "left": left,
                "right": right,
                "left_ids": left_ids,
                "right_ids": right_ids,
                "distinct": left_ids != right_ids,
            }
        )
    return results


def summarize_probe_categories(probes: Sequence[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for probe in probes:
        grouped.setdefault(str(probe["category"]), []).append(probe)

    summaries: list[dict] = []
    for category in sorted(grouped):
        group = grouped[category]
        total_tokens = sum(int(item["tokens"]) for item in group)
        total_words = sum(int(item["word_like_units"]) for item in group)
        total_chars = sum(int(item["characters"]) for item in group)
        total_bytes = sum(int(item["utf8_bytes"]) for item in group)
        total_byte_tokens = sum(int(item["byte_tokens"]) for item in group)
        summaries.append(
            {
                "category": category,
                "probes": len(group),
                "tokens": total_tokens,
                "word_like_units": total_words,
                "tokens_per_word_like_unit": rounded(
                    safe_ratio(total_tokens, total_words)
                ),
                "characters_per_token": rounded(safe_ratio(total_chars, total_tokens)),
                "utf8_bytes_per_token": rounded(safe_ratio(total_bytes, total_tokens)),
                "byte_token_ratio": rounded(safe_ratio(total_byte_tokens, total_tokens)),
                "roundtrip_ok": all(bool(item["roundtrip_ok"]) for item in group),
            }
        )
    return summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def tokenizer_metadata(tokenizer: Tokenizer, tokenizer_path: Path) -> dict:
    return {
        "path": str(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "learned_content_vocab_size": tokenizer.learned_content_vocab_size,
        "content_vocab_limit": tokenizer.content_vocab_limit,
        "special_token_count": tokenizer.special_token_count,
        "learned_merge_count": len(tokenizer.bpe.merges),
        "unused_content_ids": (
            tokenizer.content_vocab_limit - tokenizer.learned_content_vocab_size
        ),
        "special_tokens": [
            {"token": token, "id": tokenizer.special_token_to_id[token]}
            for token in tokenizer.special_tokens
        ],
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def format_float(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def display_token_piece(record: dict) -> str:
    if record.get("type") == "special":
        return f"<{record.get('text')}>"

    utf8 = record.get("utf8")
    if isinstance(utf8, str):
        return printable_text(utf8)
    return f"<0x{record.get('bytes_hex', '')}>"


def print_report_summary(
    *,
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    corpus_metrics: dict | None,
    probe_results: Sequence[dict],
    category_summaries: Sequence[dict],
    case_checks: Sequence[dict],
    output_path: Path,
    show_segmentation: bool,
) -> None:
    print("\nTokenizer Evaluation")
    print("=" * 80)
    print(f"Tokenizer:              {tokenizer_path}")
    print(f"Vocabulary size:        {tokenizer.vocab_size:,}")
    print(f"Learned content tokens: {tokenizer.learned_content_vocab_size:,}")
    print(f"Learned BPE merges:     {len(tokenizer.bpe.merges):,}")
    print(f"Special tokens:         {tokenizer.special_token_count:,}")

    if corpus_metrics is not None:
        print("\nHeld-out corpus")
        print("-" * 80)
        print(f"Documents:              {corpus_metrics['documents']:,}")
        print(f"Characters:             {corpus_metrics['characters']:,}")
        print(f"UTF-8 bytes:            {corpus_metrics['utf8_bytes']:,}")
        print(f"Word-like units:        {corpus_metrics['word_like_units']:,}")
        print(f"Tokens:                 {corpus_metrics['tokens']:,}")
        print(
            "Tokens / word-like:     "
            f"{format_float(corpus_metrics['tokens_per_word_like_unit'])}"
        )
        print(
            "Characters / token:     "
            f"{format_float(corpus_metrics['characters_per_token'])}"
        )
        print(
            "UTF-8 bytes / token:    "
            f"{format_float(corpus_metrics['utf8_bytes_per_token'])}"
        )
        print(
            "Byte-token ratio:       "
            f"{format_float(corpus_metrics['byte_token_ratio'] * 100 if corpus_metrics['byte_token_ratio'] is not None else None, 2)}%"
        )
        print(
            "BPE-token ratio:        "
            f"{format_float(corpus_metrics['bpe_token_ratio'] * 100 if corpus_metrics['bpe_token_ratio'] is not None else None, 2)}%"
        )
        print(
            "Content vocab used:     "
            f"{format_float(corpus_metrics['content_vocabulary_utilization'] * 100 if corpus_metrics['content_vocabulary_utilization'] is not None else None, 2)}%"
        )
        print(f"Round-trip failures:    {corpus_metrics['roundtrip_failures']}")

    if category_summaries:
        print("\nBuilt-in technical probes")
        print("-" * 80)
        print(
            f"{'Category':<22} {'Tok/word':>10} {'Bytes/tok':>10} "
            f"{'Byte %':>9} {'RT':>5}"
        )
        for item in category_summaries:
            byte_ratio = item["byte_token_ratio"]
            byte_pct = None if byte_ratio is None else byte_ratio * 100
            print(
                f"{str(item['category']):<22} "
                f"{format_float(item['tokens_per_word_like_unit']):>10} "
                f"{format_float(item['utf8_bytes_per_token']):>10} "
                f"{format_float(byte_pct, 1):>8}% "
                f"{('OK' if item['roundtrip_ok'] else 'FAIL'):>5}"
            )

    if case_checks:
        passed = sum(bool(item["distinct"]) for item in case_checks)
        print(f"\nCase-distinction checks: {passed}/{len(case_checks)} passed")

    if show_segmentation and probe_results:
        print("\nProbe segmentations")
        print("-" * 80)
        for probe in probe_results:
            print(f"\n[{probe['category']}/{probe['name']}] {probe['text']}")
            pieces = [display_token_piece(item) for item in probe["segmentation"]]
            print(" | ".join(pieces))
            print(f"tokens={probe['tokens']}  tok/word={format_float(probe['tokens_per_word_like_unit'])}")

    print(f"\nReport: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        tokenizer_path = args.tokenizer.expanduser().resolve()
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer file does not exist: {tokenizer_path}")

        tokenizer = Tokenizer.load(tokenizer_path)

        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else tokenizer_path.parent / "evaluation_report.json"
        )

        input_files: list[Path] = []
        documents: list[str] = []
        corpus_metrics: dict | None = None

        if args.input:
            input_files = discover_input_files(args.input)
            documents = load_documents(
                input_files,
                text_field=args.text_field,
                max_documents=args.max_documents,
                max_bytes=args.max_bytes,
            )
            if not documents:
                raise ValueError("the evaluation corpus contains no non-empty documents")

            corpus_metrics = evaluate_documents(
                tokenizer,
                documents,
                top_n=args.top_tokens,
                longest_n=args.longest_tokens,
            )

        probe_results: list[dict] = []
        category_summaries: list[dict] = []
        case_checks: list[dict] = []

        if not args.no_probes:
            probe_results = [evaluate_probe(tokenizer, probe) for probe in BUILTIN_PROBES]
            category_summaries = summarize_probe_categories(probe_results)
            case_checks = evaluate_case_checks(tokenizer)

        correctness = {
            "held_out_roundtrip_ok": (
                None
                if corpus_metrics is None
                else corpus_metrics["roundtrip_failures"] == 0
            ),
            "builtin_probes_roundtrip_ok": (
                None
                if not probe_results
                else all(bool(item["roundtrip_ok"]) for item in probe_results)
            ),
            "case_distinctions_ok": (
                None
                if not case_checks
                else all(bool(item["distinct"]) for item in case_checks)
            ),
        }

        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluator_version": "0.1",
            "tokenizer": tokenizer_metadata(tokenizer, tokenizer_path),
            "evaluation_input": {
                "files": [str(path) for path in input_files],
                "documents_loaded": len(documents),
                "text_field": args.text_field,
                "max_documents": args.max_documents,
                "max_bytes": args.max_bytes,
            },
            "correctness": correctness,
            "held_out_corpus": corpus_metrics,
            "builtin_probe_categories": category_summaries,
            "builtin_probes": probe_results,
            "case_distinction_checks": case_checks,
            "metric_notes": {
                "word_like_unit": (
                    "Unicode regex \\w+; code identifiers such as calculate_voltage "
                    "and STM32F411 count as one unit."
                ),
                "byte_token_ratio": (
                    "Fraction of emitted content tokens that are raw IDs 0..255. "
                    "Lower is generally better on in-domain text, but there is no "
                    "universal pass/fail threshold."
                ),
                "utf8_bytes_per_token": (
                    "Average normalized UTF-8 bytes represented by one emitted token. "
                    "Higher means stronger compression, but quality must also be "
                    "checked across domains and technical notation."
                ),
                "content_vocabulary_utilization": (
                    "Distinct usable content token IDs observed in the held-out corpus "
                    "divided by learned_content_vocab_size. This depends strongly on "
                    "evaluation corpus size."
                ),
            },
        }

        write_json(output_path, report)

        if not args.quiet:
            print_report_summary(
                tokenizer=tokenizer,
                tokenizer_path=tokenizer_path,
                corpus_metrics=corpus_metrics,
                probe_results=probe_results,
                category_summaries=category_summaries,
                case_checks=case_checks,
                output_path=output_path,
                show_segmentation=args.show_segmentation,
            )

        # Correctness failures should fail CI.  Compression/fragmentation
        # quality is intentionally *not* given arbitrary hard thresholds here.
        correctness_values = [value for value in correctness.values() if value is not None]
        return 0 if all(correctness_values) else 2

    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
