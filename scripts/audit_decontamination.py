r"""Audit a cleaned pretraining corpus for evaluation contamination.

Audit-only: this script never deletes documents and never rewrites the corpus.
It reports which training documents would be removed so the threshold can be
chosen from evidence.

    python scripts\audit_decontamination.py `
        --corpus data\cleaned\pretraining\trial-100k-script `
        --evaluation data\evaluation\model\v0.1\pretraining_eval.jsonl `
        --output data\audits\decontamination-100k.json

Why a clean report is not proof
-------------------------------
The v0.1 evaluation passages were written for the purpose and appear nowhere on
the web, so a zero-contamination result is the *expected* outcome and says
nothing about whether detection works.

``--planted-controls N`` therefore takes N real documents out of the corpus and
feeds them to the index as additional evaluation items.  Every one of them must
then be detected when the corpus is scanned.  A run that reports zero natural
contamination and N of N planted controls found is evidence; a run that reports
only zero is not.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import random
import sys
from typing import Iterator, Sequence

from llm.data.decontamination import (
    REASON_CONTAINMENT,
    REASON_EXACT_MATCH,
    DecontaminationConfig,
    EvaluationIndex,
    EvaluationItem,
    load_evaluation_items,
)


DEFAULT_PATTERN = "corpus-*.jsonl.gz"
PLANTED_PREFIX = "planted-control-"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be within (0, 1]")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report training documents that contain frozen evaluation material."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--min-containment",
        type=unit_interval,
        default=0.80,
        help="Fraction of an evaluation item that must appear to reject.",
    )
    parser.add_argument("--shingle-size", type=positive_int, default=8)
    parser.add_argument(
        "--min-evaluation-tokens",
        type=positive_int,
        default=50,
        help="Evaluation items shorter than this are too generic to match on.",
    )
    parser.add_argument(
        "--max-documents",
        type=positive_int,
        default=None,
        help="Limit for development runs.",
    )
    parser.add_argument(
        "--max-report-hits",
        type=positive_int,
        default=200,
        help="Contaminated documents retained as examples in the report.",
    )
    parser.add_argument(
        "--planted-controls",
        type=int,
        default=0,
        help=(
            "Take this many real corpus documents and add them as evaluation "
            "items. All of them must then be detected, which is what proves "
            "the detector works against real text."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args(argv)


def discover_shards(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")
    shards = sorted(directory.glob(pattern), key=lambda item: item.name)
    if not shards:
        raise FileNotFoundError(f"no files matching {pattern!r} under {directory}")
    return shards


def iter_documents(shards: Sequence[Path]) -> Iterator[dict]:
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def sample_planted_controls(
    shards: Sequence[Path],
    *,
    count: int,
    seed: int,
    min_characters: int = 2_000,
) -> list[EvaluationItem]:
    """Reservoir-sample real corpus documents to act as known positives."""

    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0

    for record in iter_documents(shards):
        text = record.get("text")
        if not isinstance(text, str) or len(text) < min_characters:
            continue
        seen += 1
        if len(reservoir) < count:
            reservoir.append(record)
        else:
            index = rng.randrange(seen)
            if index < count:
                reservoir[index] = record

    return [
        EvaluationItem(
            item_id=f"{PLANTED_PREFIX}{record['document_id']}",
            category="planted_control",
            text=record["text"],
        )
        for record in reservoir
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)

    try:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(
                f"{args.output} exists; pass --overwrite or choose a new path"
            )

        shards = discover_shards(args.corpus, args.pattern)
        config = DecontaminationConfig(
            shingle_size=args.shingle_size,
            min_containment=args.min_containment,
            min_evaluation_tokens=args.min_evaluation_tokens,
        )

        items = load_evaluation_items(args.evaluation)
        planted: list[EvaluationItem] = []

        if args.planted_controls > 0:
            if not args.quiet:
                print(
                    f"Sampling {args.planted_controls} planted controls "
                    "from the corpus ...",
                    flush=True,
                )
            planted = sample_planted_controls(
                shards,
                count=args.planted_controls,
                seed=args.seed,
            )
            items = items + planted

        index = EvaluationIndex(items, config=config)

        if not args.quiet:
            print(
                f"Indexed {len(index)} evaluation items "
                f"({index.indexed_shingles:,} distinct shingles); "
                f"skipped {len(index.skipped_short)} as too short."
            )
            print("Scanning corpus ...", flush=True)

        documents_scanned = 0
        contaminated_documents = 0
        by_reason: dict[str, int] = {
            REASON_EXACT_MATCH: 0,
            REASON_CONTAINMENT: 0,
        }
        hit_items: dict[str, int] = {}
        planted_found: set[str] = set()
        retained: list[dict] = []

        for record in iter_documents(shards):
            if (
                args.max_documents is not None
                and documents_scanned >= args.max_documents
            ):
                break

            text = record.get("text")
            if not isinstance(text, str):
                continue

            documents_scanned += 1
            verdict = index.check(text)

            if verdict.contaminated:
                contaminated_documents += 1
                best = verdict.best
                assert best is not None
                by_reason[best.reason] = by_reason.get(best.reason, 0) + 1

                for match in verdict.matches:
                    hit_items[match.item_id] = hit_items.get(match.item_id, 0) + 1
                    if match.item_id.startswith(PLANTED_PREFIX):
                        planted_found.add(match.item_id)

                if len(retained) < args.max_report_hits:
                    retained.append(
                        {
                            "document_id": record.get("document_id"),
                            "url": record.get("url"),
                            "source_shard": record.get("source_shard"),
                            "matched_item_id": best.item_id,
                            "matched_category": best.category,
                            "reason": best.reason,
                            "containment": best.containment,
                            "shared_shingles": best.shared_shingles,
                            "evaluation_shingles": best.evaluation_shingles,
                            "excerpt": text[:400],
                        }
                    )

            if not args.quiet and documents_scanned % 20_000 == 0:
                print(
                    f"  scanned {documents_scanned:>9,}  "
                    f"contaminated {contaminated_documents:>7,}",
                    flush=True,
                )

        finished = datetime.now(timezone.utc)

        planted_ids = {item.item_id for item in planted}
        planted_missing = sorted(planted_ids - planted_found)
        natural_hits = sum(
            count
            for item_id, count in hit_items.items()
            if not item_id.startswith(PLANTED_PREFIX)
        )

        report = {
            "format": "llm_decontamination_audit",
            "format_version": 1,
            "created_at_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "corpus": {
                "path": str(args.corpus),
                "shards": [shard.name for shard in shards],
            },
            "evaluation": {
                "path": str(args.evaluation),
                "items_loaded": len(items) - len(planted),
                "items_indexed": len(index),
                "skipped_too_short": index.skipped_short,
                "categories": index.category_counts(),
            },
            "config": asdict(config),
            "planted_controls": {
                "requested": args.planted_controls,
                "sampled": len(planted),
                "detected": len(planted_found),
                "missing": planted_missing,
                "all_detected": (
                    bool(planted) and not planted_missing
                ),
            },
            "stats": {
                "documents_scanned": documents_scanned,
                "contaminated_documents": contaminated_documents,
                "contamination_rate": (
                    contaminated_documents / documents_scanned
                    if documents_scanned
                    else 0.0
                ),
                "by_reason": by_reason,
                "natural_evaluation_hits": natural_hits,
                "hits_per_item": dict(sorted(hit_items.items())),
            },
            "examples": retained,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        if not args.quiet:
            print()
            print("DECONTAMINATION AUDIT")
            print("=" * 64)
            print(f"Documents scanned:        {documents_scanned:,}")
            print(f"Evaluation items indexed: {len(index):,}")
            print()
            print(f"Contaminated documents:   {contaminated_documents:,}")
            print(f"  exact match:            {by_reason[REASON_EXACT_MATCH]:,}")
            print(f"  containment:            {by_reason[REASON_CONTAINMENT]:,}")
            print(
                f"Contamination rate:       "
                f"{report['stats']['contamination_rate']:.4%}"
            )
            print()
            print(f"Natural evaluation hits:  {natural_hits:,}")

            if planted:
                status = "PASS" if not planted_missing else "FAIL"
                print()
                print("Planted-control validation")
                print("-" * 64)
                print(f"  planted:                {len(planted):,}")
                print(f"  detected:               {len(planted_found):,}")
                print(f"  result:                 {status}")
                for missing in planted_missing[:10]:
                    print(f"    MISSED {missing}")
            else:
                print()
                print(
                    "No planted controls. A zero result here is expected and "
                    "does not demonstrate detection works."
                )

            print()
            print(f"Report: {args.output}")

        if planted and planted_missing:
            return 2

        return 0

    except (OSError, ValueError, TypeError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
