r"""Audit near duplicates in an existing cleaned pretraining corpus.

This script is intentionally audit-only.  It NEVER deletes documents and does
not rewrite the cleaned corpus.

Typical use after the 100k quality/language audit:

    python scripts\audit_near_duplicates.py `
        --input data\cleaned\pretraining\trial-100k-script `
        --output data\audits\near-duplicates-100k.json `
        --index data\audits\near-duplicates-100k.sqlite `
        --overwrite

The report records the best verified previous near-duplicate match for each
document that has one.  The SQLite index persists all comparison features and
is useful for debugging/reproducibility, but this first audit intentionally
rebuilds from scratch unless the caller chooses a new path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys
from typing import Iterator, Sequence

from llm.data.near_dedup import (
    AMBIGUOUS_OVERLAP,
    SAFE_NEAR_DUPLICATE,
    NearDedupConfig,
    NearDuplicateIndex,
)


DEFAULT_PATTERN = "corpus-*.jsonl.gz"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be within [0, 1]")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a cleaned corpus for near duplicates without deleting data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Cleaned corpus directory containing corpus-*.jsonl.gz.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="JSON audit report path.",
    )
    parser.add_argument(
        "--index",
        required=True,
        type=Path,
        help="SQLite feature/candidate index path.",
    )
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--max-documents",
        type=positive_int,
        default=None,
        help="Optional audit limit for development runs.",
    )
    parser.add_argument(
        "--max-report-pairs",
        type=positive_int,
        default=500,
        help="Maximum verified pairs retained in the JSON report.",
    )
    parser.add_argument(
        "--shingle-size",
        type=positive_int,
        default=8,
    )
    parser.add_argument(
        "--min-tokens",
        type=positive_int,
        default=50,
    )
    parser.add_argument(
        "--similarity-threshold",
        type=unit_interval,
        default=0.90,
    )
    parser.add_argument(
        "--min-length-ratio",
        type=unit_interval,
        default=0.90,
    )
    parser.add_argument(
        "--max-unique-shingles",
        type=positive_int,
        default=16,
        help=(
            "Absolute unique-information guard. A verified match whose larger "
            "side contributes more unique shingles than this is classified "
            "ambiguous_overlap and kept, however high its similarity."
        ),
    )
    parser.add_argument(
        "--max-unique-share",
        type=unit_interval,
        default=0.02,
        help=(
            "Relative unique-information guard, as a fraction of the combined "
            "shingle set. Stops short technical records being deleted because "
            "their distinctive payload is only a few shingles long."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=positive_int,
        default=2000,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing report/index.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    return parser.parse_args(argv)


def discover_shards(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"input directory not found: {directory}")
    shards = sorted(directory.glob(pattern), key=lambda p: p.name)
    if not shards:
        raise FileNotFoundError(
            f"no files matching {pattern!r} under {directory}"
        )
    return shards


def iter_documents(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in cleaned corpus {path.name}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"cleaned record is not an object: {path.name}:{line_number}"
                )
            yield record


def prepare_outputs(
    report_path: Path,
    index_path: Path,
    *,
    overwrite: bool,
) -> None:
    for path in (report_path, index_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists; pass --overwrite or choose a new path"
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite:
        if report_path.exists():
            report_path.unlink()
        if index_path.exists():
            index_path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(index_path) + suffix)
            if sidecar.exists():
                sidecar.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)

    config = NearDedupConfig(
        shingle_size=args.shingle_size,
        similarity_threshold=args.similarity_threshold,
        min_length_ratio=args.min_length_ratio,
        max_unique_shingles=args.max_unique_shingles,
        max_unique_share=args.max_unique_share,
        min_tokens=args.min_tokens,
        max_candidates=args.max_candidates,
    )

    shards = discover_shards(args.input, args.pattern)
    prepare_outputs(args.output, args.index, overwrite=args.overwrite)

    documents_scanned = 0
    documents_indexed = 0
    documents_too_short = 0
    candidate_pairs_examined = 0
    near_duplicate_documents = 0
    safe_documents = 0
    ambiguous_documents = 0
    retained_pairs: list[dict] = []

    # Retain both classes independently. A single first-N quota would fill with
    # whichever class happens to appear early, and the ambiguous ones are
    # precisely the examples worth reading before freezing a deletion rule.
    per_class_quota = max(1, args.max_report_pairs // 2)
    retained_by_class: dict[str, int] = {
        SAFE_NEAR_DUPLICATE: 0,
        AMBIGUOUS_OVERLAP: 0,
    }

    with NearDuplicateIndex(args.index, config=config) as index:
        stop = False

        for shard in shards:
            if stop:
                break

            index.begin()
            try:
                for record in iter_documents(shard):
                    if (
                        args.max_documents is not None
                        and documents_scanned >= args.max_documents
                    ):
                        stop = True
                        break

                    documents_scanned += 1

                    document_id = record.get("document_id")
                    text = record.get("text")
                    url = record.get("url")

                    if not isinstance(document_id, str) or not document_id:
                        raise ValueError(
                            f"cleaned document missing document_id in {shard.name}"
                        )
                    if not isinstance(text, str):
                        raise ValueError(
                            f"cleaned document missing text in {shard.name}"
                        )

                    features = index.analyze_text(text)
                    if features is None:
                        documents_too_short += 1
                    else:
                        result = index.find_near_duplicates(features)
                        candidate_pairs_examined += result.candidate_count

                        if result.matches:
                            near_duplicate_documents += 1

                            # A document is only removable when at least one of
                            # its matches is safe. Merely being similar to
                            # something is not grounds for deletion.
                            safe = result.best_safe_match
                            if safe is not None:
                                best = safe
                                classification = SAFE_NEAR_DUPLICATE
                                safe_documents += 1
                            else:
                                best = result.matches[0]
                                classification = AMBIGUOUS_OVERLAP
                                ambiguous_documents += 1

                            if retained_by_class[classification] < per_class_quota:
                                retained_by_class[classification] += 1
                                retained_pairs.append(
                                    {
                                        "document_id": document_id,
                                        "matched_document_id": best.document_id,
                                        "classification": classification,
                                        "similarity": best.similarity,
                                        "length_ratio": best.length_ratio,
                                        "shared_shingles": best.shared_shingles,
                                        "unique_query_shingles": (
                                            best.unique_query_shingles
                                        ),
                                        "unique_candidate_shingles": (
                                            best.unique_candidate_shingles
                                        ),
                                        "max_unique_shingles": (
                                            best.max_unique_shingles
                                        ),
                                        "unique_share": best.unique_share,
                                        "total_matches": len(result.matches),
                                        "url": (
                                            url if isinstance(url, str) else None
                                        ),
                                        "matched_url": best.url,
                                        "excerpt": text[:500],
                                        "matched_excerpt": best.excerpt,
                                    }
                                )

                        index.add_document(
                            document_id,
                            features,
                            url=url if isinstance(url, str) else None,
                            excerpt=text[:500],
                        )
                        documents_indexed += 1

                    if (
                        not args.quiet
                        and documents_scanned % 10_000 == 0
                    ):
                        print(
                            f"  scanned {documents_scanned:>9,}  "
                            f"indexed {documents_indexed:>9,}  "
                            f"near-dup docs {near_duplicate_documents:>7,}",
                            flush=True,
                        )

                index.commit()
            except BaseException:
                index.rollback()
                raise

    finished = datetime.now(timezone.utc)

    report = {
        "format": "llm_near_duplicate_audit",
        "format_version": 2,
        "created_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "input": {
            "path": str(args.input),
            "pattern": args.pattern,
            "shards": [path.name for path in shards],
        },
        "config": asdict(config),
        "stats": {
            "documents_scanned": documents_scanned,
            "documents_indexed": documents_indexed,
            "documents_too_short": documents_too_short,
            "candidate_pairs_examined": candidate_pairs_examined,
            "near_duplicate_documents": near_duplicate_documents,
            "near_duplicate_rate": (
                near_duplicate_documents / documents_indexed
                if documents_indexed
                else 0.0
            ),
            "safe_near_duplicate_documents": safe_documents,
            "ambiguous_overlap_documents": ambiguous_documents,
            "would_remove": safe_documents,
            "would_keep_unique_content": ambiguous_documents,
            "would_remove_rate": (
                safe_documents / documents_indexed if documents_indexed else 0.0
            ),
            "retained_pair_examples": len(retained_pairs),
            "retained_by_classification": dict(retained_by_class),
        },
        "pairs": retained_pairs,
    }

    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    if not args.quiet:
        rate = (
            near_duplicate_documents / documents_indexed
            if documents_indexed
            else 0.0
        )
        remove_rate = (
            safe_documents / documents_indexed if documents_indexed else 0.0
        )

        print()
        print("NEAR-DUPLICATE AUDIT v0.2")
        print("=" * 64)
        print(f"Documents scanned:            {documents_scanned:,}")
        print(f"Documents indexed:            {documents_indexed:,}")
        print(f"Too short:                    {documents_too_short:,}")
        print()
        print(f"LSH candidates:               {candidate_pairs_examined:,}")
        print()
        print("Similarity matches:")
        print(
            f"  Jaccard >= {config.similarity_threshold:.2f}:"
            f"{near_duplicate_documents:>19,}"
        )
        print(f"  rate:                       {rate:>12.3%}")
        print()
        print(
            f"Classified (unique <= {config.max_unique_shingles} "
            f"AND <= {config.max_unique_share:.0%} of union):"
        )
        print(f"  safe_near_duplicate:        {safe_documents:>12,}")
        print(f"  ambiguous_overlap:          {ambiguous_documents:>12,}")
        print()
        print(f"Would remove:                 {safe_documents:>12,}"
              f"  ({remove_rate:.3%})")
        print(f"Would keep due unique content:{ambiguous_documents:>12,}")
        print()
        print(f"Report:                       {args.output}")
        print(f"Index:                        {args.index}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
