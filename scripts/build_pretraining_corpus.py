r"""Build a cleaned pretraining corpus from raw Common Crawl extractor output.

This is the stage between the WET extractor and tokenization::

    D:\Common Crawl\Data\cc_text_*.jsonl.gz
            |
            v
    build_pretraining_corpus.py
            |
            v
    data\cleaned\pretraining\v0.1\corpus-*.jsonl.gz

Development sequence
--------------------
Never make the first run a full-corpus run.  Work up in stages::

    # 1. small trial, inspect the samples in build_report.json
    python scripts\build_pretraining_corpus.py `
        --source "D:\Common Crawl\Data" `
        --output data\cleaned\pretraining\trial-10k `
        --max-records 10000 --overwrite

    # 2. statistical audit
    python scripts\build_pretraining_corpus.py `
        --source "D:\Common Crawl\Data" `
        --output data\cleaned\pretraining\trial-100k `
        --max-records 100000 --overwrite

    # 3. full corpus, then freeze
    python scripts\build_pretraining_corpus.py `
        --source "D:\Common Crawl\Data" `
        --output data\cleaned\pretraining\v0.1

Interrupted full runs can be continued with ``--resume``, which skips input
shards already recorded as complete in ``progress.json``.

Outputs
-------
<output>\
    corpus-00000.jsonl.gz ...
    manifest.json
    build_report.json
    progress.json
    dedup.sqlite
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from llm.data.pretraining_corpus import (
    DEDUP_NAME,
    DEFAULT_COMPRESS_LEVEL,
    DEFAULT_LANGUAGE,
    DEFAULT_SHARD_BYTES,
    DEFAULT_SOURCE_GLOB,
    MANIFEST_NAME,
    PROGRESS_NAME,
    REPORT_NAME,
    CorpusStats,
    build_pretraining_corpus,
)
from llm.data.decontamination import DecontaminationConfig
from llm.data.language_sanity import ScriptSanityThresholds
from llm.data.quality import QualityThresholds


DEFAULT_SOURCE = Path(r"D:\Common Crawl\Data")


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def unit_interval(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be within [0, 1]")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream, filter, deduplicate, and shard Common Crawl text into a "
            "reproducible pretraining corpus."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Directory containing the extractor's .jsonl.gz shards.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output corpus directory, for example data/cleaned/pretraining/v0.1",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_SOURCE_GLOB,
        help="Glob for input shards. Excludes crawler bookkeeping files.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Required primary language code of the extractor's lang field.",
    )

    limits = parser.add_argument_group("limits (use these for trial runs)")
    limits.add_argument(
        "--max-records",
        type=positive_int,
        default=None,
        help="Stop after examining this many input records.",
    )
    limits.add_argument(
        "--max-documents",
        type=positive_int,
        default=None,
        help="Stop after accepting this many documents.",
    )

    quality = parser.add_argument_group("quality thresholds")
    quality.add_argument(
        "--min-characters",
        type=positive_int,
        default=QualityThresholds.min_characters,
        help="Reject documents shorter than this after normalization.",
    )
    quality.add_argument(
        "--min-words",
        type=positive_int,
        default=QualityThresholds.min_words,
        help="Reject documents with fewer word-like tokens than this.",
    )
    quality.add_argument(
        "--min-alphabetic-ratio",
        type=unit_interval,
        default=QualityThresholds.min_alphabetic_ratio,
        help=(
            "Reject documents below this alphabetic ratio. Keep low: code and "
            "formulas legitimately depress it."
        ),
    )
    quality.add_argument(
        "--max-duplicate-line-ratio",
        type=unit_interval,
        default=QualityThresholds.max_duplicate_line_ratio,
        help="Reject documents whose non-empty lines repeat above this ratio.",
    )
    quality.add_argument(
        "--no-secret-filter",
        action="store_true",
        help="Disable high-confidence secret detection (not recommended).",
    )

    language_sanity = parser.add_argument_group("script sanity")
    language_sanity.add_argument(
        "--no-script-sanity",
        action="store_true",
        help="Disable the conservative secondary Unicode-script sanity check.",
    )
    language_sanity.add_argument(
        "--script-min-alphabetic-characters",
        type=positive_int,
        default=ScriptSanityThresholds.min_alphabetic_characters,
        help="Documents with fewer alphabetic characters are never judged.",
    )
    language_sanity.add_argument(
        "--script-min-latin-share",
        type=unit_interval,
        default=ScriptSanityThresholds.min_latin_share,
        help="Latin share at or above this always passes the script gate.",
    )
    language_sanity.add_argument(
        "--script-min-dominant-non-latin-share",
        type=unit_interval,
        default=ScriptSanityThresholds.min_dominant_non_latin_share,
        help=(
            "Share of one non-Latin script family required to reject. "
            "Raise to be more permissive."
        ),
    )

    decontamination = parser.add_argument_group("evaluation decontamination")
    decontamination.add_argument(
        "--evaluation",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Frozen model-evaluation JSONL to protect, for example "
            "data/evaluation/model/v0.1/pretraining_eval.jsonl. Required "
            "unless --no-decontamination is given."
        ),
    )
    decontamination.add_argument(
        "--no-decontamination",
        action="store_true",
        help=(
            "Build without protecting any evaluation set. Only for "
            "experiments; never for a corpus a model will be evaluated after."
        ),
    )
    decontamination.add_argument(
        "--min-containment",
        type=unit_interval,
        default=DecontaminationConfig.min_containment,
        help=(
            "Fraction of an evaluation item that must appear in a document "
            "before it is rejected. Provisional v0.1 policy: 0.80."
        ),
    )
    decontamination.add_argument(
        "--decontamination-shingle-size",
        type=positive_int,
        default=DecontaminationConfig.shingle_size,
    )
    decontamination.add_argument(
        "--min-evaluation-tokens",
        type=positive_int,
        default=DecontaminationConfig.min_evaluation_tokens,
        help="Evaluation items shorter than this are too generic to match on.",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--shard-bytes",
        type=positive_int,
        default=DEFAULT_SHARD_BYTES,
        help="Rotate output shards after this many uncompressed JSONL bytes.",
    )
    output_group.add_argument(
        "--compress-level",
        type=int,
        choices=range(1, 10),
        default=DEFAULT_COMPRESS_LEVEL,
        help="gzip compression level for output shards.",
    )
    output_group.add_argument(
        "--keep-rejected-samples",
        type=positive_int,
        default=50,
        help="Rejected excerpts retained in build_report.json for auditing.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted build, skipping completed input shards.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an output directory that already has a build.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output except errors.",
    )

    return parser.parse_args(argv)


def ensure_output_directory(
    path: Path,
    *,
    overwrite: bool,
    resume: bool,
) -> Path:
    output = path.expanduser().resolve()

    if output.exists() and not output.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output}")

    output.mkdir(parents=True, exist_ok=True)

    existing = sorted(output.glob("corpus-*.jsonl.gz"))
    managed = [
        output / name
        for name in (MANIFEST_NAME, REPORT_NAME)
        if (output / name).exists()
    ]

    if (existing or managed) and not (overwrite or resume):
        raise FileExistsError(
            f"{output} already contains a corpus build; pass --resume to "
            "continue it or --overwrite to rebuild from scratch"
        )

    if overwrite and not resume:
        # --overwrite means a genuinely fresh build.  The deduplication
        # database and progress file must go too: leaving them would make the
        # rebuild reject every document as an exact duplicate of the previous
        # run, which looks like a filtering bug rather than stale state.
        for shard in existing:
            shard.unlink()
        for name in (MANIFEST_NAME, REPORT_NAME, PROGRESS_NAME, DEDUP_NAME):
            target = output / name
            if target.exists():
                target.unlink()
        # SQLite WAL sidecars.
        for suffix in ("-wal", "-shm"):
            sidecar = output / f"{DEDUP_NAME}{suffix}"
            if sidecar.exists():
                sidecar.unlink()

    return output


def print_summary(report: dict, *, output: Path) -> None:
    stats = report["stats"]
    rejected = stats["rejected_by_reason"]

    print()
    print("PRETRAINING CORPUS BUILD")
    print("=" * 64)
    print(f"Output:                {output}")
    print(f"Duration (s):          {report['duration_seconds']:.1f}")
    print()
    print("Input")
    print("-" * 64)
    print(f"Shards read:           {stats['shards_read']:,}")
    print(f"Records seen:          {stats['records_seen']:,}")
    print()
    print("Rejected")
    print("-" * 64)
    width = max((len(reason) for reason in rejected), default=5)
    width = max(width, len("total"))
    for reason, count in sorted(
        rejected.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"{reason:<{width}}   {count:>12,}")
    print(f"{'total':<{width}}   {stats['documents_rejected']:>12,}")
    print()
    print("Accepted")
    print("-" * 64)
    print(f"Documents:             {stats['documents_accepted']:,}")
    print(f"Characters:            {stats['accepted_characters']:,}")
    print(f"UTF-8 bytes:           {stats['accepted_utf8_bytes']:,}")
    print(f"Acceptance rate:       {stats['acceptance_rate']:.2%}")
    print()
    print("Review rejected_samples in build_report.json before scaling up.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        # Fail closed before any work happens, so the mistake is caught in the
        # first second rather than after a multi-hour corpus build.
        if not args.no_decontamination and args.evaluation is None:
            raise ValueError(
                "--evaluation is required so the frozen evaluation set is "
                "protected. Pass --no-decontamination only for experimental "
                "corpora that no model will be evaluated against."
            )
        if args.no_decontamination and args.evaluation is not None:
            raise ValueError(
                "--evaluation and --no-decontamination are contradictory"
            )

        output = ensure_output_directory(
            args.output,
            overwrite=args.overwrite,
            resume=args.resume,
        )

        thresholds = QualityThresholds(
            min_characters=args.min_characters,
            min_words=args.min_words,
            min_alphabetic_ratio=args.min_alphabetic_ratio,
            max_duplicate_line_ratio=args.max_duplicate_line_ratio,
        )

        decontamination_config = DecontaminationConfig(
            shingle_size=args.decontamination_shingle_size,
            min_containment=args.min_containment,
            min_evaluation_tokens=args.min_evaluation_tokens,
        )

        script_thresholds = ScriptSanityThresholds(
            min_alphabetic_characters=args.script_min_alphabetic_characters,
            min_latin_share=args.script_min_latin_share,
            min_dominant_non_latin_share=(
                args.script_min_dominant_non_latin_share
            ),
        )

        def on_progress(stats: CorpusStats) -> None:
            print(
                f"  seen {stats.records_seen:>10,}  "
                f"accepted {stats.documents_accepted:>10,}  "
                f"({stats.acceptance_rate:.1%})",
                flush=True,
            )

        if not args.quiet:
            print(f"Reading {args.source} ...")

        report = build_pretraining_corpus(
            source=args.source,
            output=output,
            thresholds=thresholds,
            script_thresholds=script_thresholds,
            check_script_sanity=not args.no_script_sanity,
            evaluation_path=args.evaluation,
            decontamination_config=decontamination_config,
            check_decontamination=not args.no_decontamination,
            language=args.language,
            source_pattern=args.pattern,
            shard_bytes=args.shard_bytes,
            compress_level=args.compress_level,
            max_documents=args.max_documents,
            max_records=args.max_records,
            resume=args.resume,
            keep_rejected_samples=args.keep_rejected_samples,
            check_secrets=not args.no_secret_filter,
            on_progress=None if args.quiet else on_progress,
        )

        if not args.quiet:
            print_summary(report, output=output)

        return 0

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Re-run with --resume to continue.",
            file=sys.stderr,
        )
        return 130
    except (
        OSError,
        ValueError,
        TypeError,
        UnicodeError,
        FileExistsError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
