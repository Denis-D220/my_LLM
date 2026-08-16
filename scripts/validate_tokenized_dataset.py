r"""Validate a tokenized pretraining dataset, then optionally freeze it.

This is the last gate before model code exists.  Everything downstream --
every loss curve, every evaluation number -- is computed against whatever this
script signs off on, so it checks the token stream itself rather than the
manifest's description of it.

    python scripts\validate_tokenized_dataset.py `
        --dataset data\tokenized\v0.1 `
        --tokenizer artifacts\tokenizer-E011\tokenizer.json `
        --corpus data\cleaned\pretraining\v0.1 `
        --report data\audits\dataset-v0.1-validation.json `
        --freeze

What gets proved, and how
-------------------------
*Storage integrity* comes free: :class:`llm.data.dataset.PretrainingDataset`
verifies every shard SHA-256 and the whole stream digest at construction.

*Document framing* is the check worth the most.  The entire stream is walked
with a two-state machine that requires BOS and EOS to strictly alternate,
beginning with BOS and ending with EOS.  If that holds and the BOS count equals
the manifest's document count, then every document was framed exactly once and
no framing token was lost at a shard boundary.  Sampling could not establish
this; a single dropped EOS 700 million tokens in would pass any sample and
quietly teach the model that documents do not end.

*No accidental special parsing* is the same walk asking a different question.
The seven non-boundary special tokens (pad, system, user, assistant, end_turn,
tool, tool_result) must appear exactly zero times.  A non-zero count means
source text was parsed as control tokens somewhere in 5 GB of web scrape, which
is precisely what ``parse_special_tokens=False`` is supposed to prevent and
precisely the kind of thing nobody notices until a chat model emits ``<|user|>``
mid-sentence.

*Shard-boundary transparency* is tested at every internal boundary rather than
one representative one.  The windows are cheap and the failure mode -- a
silently truncated or misaligned read -- is not.

*Split integrity* (``--corpus``) re-derives the train/validation split from the
frozen corpus and compares the resulting ordered document-identity digests
against the ones in the dataset manifest.  Matching digests prove the shipped
dataset really is the declared policy applied to the declared corpus, and
therefore that the two splits are document-disjoint.  This costs two extra
passes over the cleaned corpus.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterator, Sequence

from llm.data.dataset import PretrainingDataset
from llm.data.document import BOS_TOKEN, EOS_TOKEN
from llm.data.shards import TOKEN_ITEM_BYTES
from llm.tokenizer import Tokenizer

# scripts/ is on sys.path when this file is run directly, and pytest's conftest
# adds it too.  Sharing the check bookkeeping keeps the two validators
# reporting in one voice instead of drifting into two dialects.
from validate_pretraining_corpus import (
    CheckLog,
    MAX_REPORTED_PROBLEMS,
    STATUS_PASS,
    load_json,
    positive_int,
    sha256_file,
)


VALIDATOR_FORMAT = "llm_tokenized_dataset_validation"
VALIDATOR_FORMAT_VERSION = 1

FREEZE_FORMAT = "llm_tokenized_dataset_freeze"
FREEZE_FORMAT_VERSION = 1
FREEZE_FILENAME = "FROZEN.json"

DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
SPLITS = ("train", "validation")

EXPECTED_CONTEXT_LENGTH = 2048
EXPECTED_VOCAB_SIZE = 24_000

# One shard is 20 MB at 10M tokens/shard; reading a whole shard at a time keeps
# the framing state machine simple without a meaningful memory cost.
SCAN_CHUNK_TOKENS = 4_000_000


# --------------------------------------------------------------------------
# whole-stream scan
# --------------------------------------------------------------------------


@dataclass
class StreamScan:
    """Everything learned from one full walk of a split's token stream."""

    tokens: int = 0
    max_token_id: int = -1
    out_of_range: list[str] = field(default_factory=list)

    bos_count: int = 0
    eos_count: int = 0
    framing_problems: list[str] = field(default_factory=list)
    first_token: int | None = None
    last_token: int | None = None

    other_special_counts: dict[str, int] = field(default_factory=dict)

    def note(self, bucket: list[str], message: str) -> None:
        if len(bucket) < MAX_REPORTED_PROBLEMS:
            bucket.append(message)


def scan_stream(
    dataset: PretrainingDataset,
    *,
    bos_id: int,
    eos_id: int,
    other_specials: dict[int, str],
    quiet: bool,
) -> StreamScan:
    """Walk every token once, answering four questions at the same time.

    Reading 2.2 GB is the expensive part, so range checking, framing, and
    special-token counting all ride on the single pass.
    """

    scan = StreamScan()
    for name in other_specials.values():
        scan.other_special_counts[name] = 0

    vocab_size = dataset.vocab_size
    expecting_bos = True  # a stream must open a document before closing one
    position = 0

    for shard in dataset.manifest.shards:
        path = dataset.manifest_path.parent / shard.filename
        if not quiet:
            print(f"         scanning {shard.filename}", flush=True)

        with path.open("rb") as handle:
            while True:
                raw = handle.read(SCAN_CHUNK_TOKENS * TOKEN_ITEM_BYTES)
                if not raw:
                    break

                values = array("H")
                values.frombytes(raw)
                if sys.byteorder != "little":
                    values.byteswap()

                if scan.first_token is None and values:
                    scan.first_token = values[0]
                if values:
                    scan.last_token = values[-1]

                chunk_max = max(values)
                if chunk_max > scan.max_token_id:
                    scan.max_token_id = chunk_max
                if chunk_max >= vocab_size:
                    scan.note(
                        scan.out_of_range,
                        f"{shard.filename}: token id {chunk_max} >= vocab {vocab_size}",
                    )

                for special_id, name in other_specials.items():
                    found = values.count(special_id)
                    if found:
                        scan.other_special_counts[name] += found

                # Alternation walk.  Only the consumed side is re-searched, so
                # this costs one C-level index() call per framing token rather
                # than one per token in the stream.
                def find(value: int, start: int) -> int | None:
                    try:
                        return values.index(value, start)
                    except ValueError:
                        return None

                next_bos = find(bos_id, 0)
                next_eos = find(eos_id, 0)
                while next_bos is not None or next_eos is not None:
                    take_bos = next_eos is None or (
                        next_bos is not None and next_bos < next_eos
                    )
                    if take_bos:
                        assert next_bos is not None
                        if not expecting_bos:
                            scan.note(
                                scan.framing_problems,
                                f"BOS at token {position + next_bos} where EOS was "
                                "expected (previous document never closed)",
                            )
                        expecting_bos = False
                        scan.bos_count += 1
                        next_bos = find(bos_id, next_bos + 1)
                    else:
                        assert next_eos is not None
                        if expecting_bos:
                            scan.note(
                                scan.framing_problems,
                                f"EOS at token {position + next_eos} where BOS was "
                                "expected (document closed without opening)",
                            )
                        expecting_bos = True
                        scan.eos_count += 1
                        next_eos = find(eos_id, next_eos + 1)

                position += len(values)
                scan.tokens += len(values)

    if not expecting_bos:
        scan.note(
            scan.framing_problems,
            "stream ends inside an unclosed document (final EOS missing)",
        )

    return scan


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_geometry(manifest: dict[str, Any], log: CheckLog) -> None:
    geometry = manifest.get("training_geometry", {})
    expected = {
        "context_length": EXPECTED_CONTEXT_LENGTH,
        "window_tokens": EXPECTED_CONTEXT_LENGTH + 1,
        "window_stride": EXPECTED_CONTEXT_LENGTH,
        "incomplete_final_tail_padded": False,
    }
    drift = {
        key: (value, geometry.get(key))
        for key, value in expected.items()
        if geometry.get(key) != value
    }
    if drift:
        log.fail(
            "geometry.frozen",
            f"{len(drift)} geometry field(s) differ from the frozen contract",
            {"expected_vs_actual": {k: list(v) for k, v in drift.items()}},
        )
    else:
        log.ok(
            "geometry.frozen",
            f"context {EXPECTED_CONTEXT_LENGTH}, windows of "
            f"{EXPECTED_CONTEXT_LENGTH + 1}, stride {EXPECTED_CONTEXT_LENGTH}, no padding",
        )

    storage = manifest.get("storage", {})
    if storage.get("dtype") == "uint16" and storage.get("byte_order") == "little":
        log.ok("geometry.storage", "uint16 little-endian token storage")
    else:
        log.fail("geometry.storage", f"unexpected storage declaration: {storage}")


def check_tokenizer_identity(
    manifest: dict[str, Any],
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    log: CheckLog,
) -> None:
    declared = manifest.get("tokenizer", {})
    if declared.get("vocab_size") == EXPECTED_VOCAB_SIZE == tokenizer.vocab_size:
        log.ok(
            "tokenizer.vocab_size",
            f"{EXPECTED_VOCAB_SIZE:,}, agreed by the manifest and the tokenizer file",
        )
    else:
        log.fail(
            "tokenizer.vocab_size",
            f"manifest={declared.get('vocab_size')!r} "
            f"tokenizer={tokenizer.vocab_size!r} expected={EXPECTED_VOCAB_SIZE}",
        )

    payload = json.dumps(
        tokenizer.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual_state = hashlib.sha256(payload).hexdigest()

    if declared.get("state_sha256") == actual_state:
        log.ok(
            "tokenizer.state_sha256",
            f"{tokenizer_path.name} is the exact tokenizer the shards were built with",
        )
    else:
        log.fail(
            "tokenizer.state_sha256",
            "the supplied tokenizer is NOT the one used to build these shards: "
            f"manifest={declared.get('state_sha256')!r} actual={actual_state!r}",
        )


def check_dataset_contract(
    dataset: PretrainingDataset,
    split: str,
    declared: dict[str, Any],
    log: CheckLog,
    *,
    sample_count: int,
    seed: int,
) -> None:
    expected_windows = declared.get("complete_examples")
    if len(dataset) == expected_windows:
        log.ok(
            f"{split}.window_count",
            f"{len(dataset):,} windows, matching the manifest",
        )
    else:
        log.fail(
            f"{split}.window_count",
            f"dataset reports {len(dataset):,} windows, manifest says {expected_windows!r}",
        )

    recomputed = (
        (dataset.total_tokens - 1) // EXPECTED_CONTEXT_LENGTH
        if dataset.total_tokens > 1
        else 0
    )
    if recomputed == len(dataset):
        log.ok(
            f"{split}.window_count_derivation",
            f"({dataset.total_tokens:,} - 1) // {EXPECTED_CONTEXT_LENGTH} = {recomputed:,}",
        )
    else:
        log.fail(
            f"{split}.window_count_derivation",
            f"expected {recomputed:,} windows from the token count, got {len(dataset):,}",
        )

    if declared.get("total_tokens") == dataset.total_tokens:
        log.ok(f"{split}.token_count", f"{dataset.total_tokens:,} tokens")
    else:
        log.fail(
            f"{split}.token_count",
            f"dataset {dataset.total_tokens!r} != manifest {declared.get('total_tokens')!r}",
        )

    if len(dataset) == 0:
        log.skip(f"{split}.item_contract", "split exposes no complete windows")
        return

    rng = random.Random(seed)
    indexes = [0, len(dataset) // 2, len(dataset) - 1]
    indexes += [rng.randrange(len(dataset)) for _ in range(max(0, sample_count))]

    problems: list[str] = []
    for index in sorted(set(indexes)):
        inputs, targets = dataset[index]
        if tuple(inputs.shape) != (EXPECTED_CONTEXT_LENGTH,):
            problems.append(f"index {index}: input shape {tuple(inputs.shape)}")
        if tuple(targets.shape) != (EXPECTED_CONTEXT_LENGTH,):
            problems.append(f"index {index}: target shape {tuple(targets.shape)}")
        if inputs.dtype is not torch_long() or targets.dtype is not torch_long():
            problems.append(
                f"index {index}: dtypes {inputs.dtype}/{targets.dtype}, expected int64"
            )
        if not bool((inputs[1:] == targets[:-1]).all()):
            problems.append(f"index {index}: x[1:] != y[:-1]")
        if int(inputs.min()) < 0 or int(inputs.max()) >= EXPECTED_VOCAB_SIZE:
            problems.append(
                f"index {index}: ids outside [0, {EXPECTED_VOCAB_SIZE}): "
                f"[{int(inputs.min())}, {int(inputs.max())}]"
            )

    log.verdict(
        f"{split}.item_contract",
        problems,
        f"{len(set(indexes))} windows: shape ({EXPECTED_CONTEXT_LENGTH},), int64, "
        "x[1:] == y[:-1], ids in range",
    )


def torch_long():
    import torch

    return torch.long


def check_shard_boundaries(
    dataset: PretrainingDataset, split: str, log: CheckLog
) -> None:
    """Every internal shard boundary must be invisible to window reads."""

    shards = dataset.manifest.shards
    if len(shards) < 2:
        log.skip(f"{split}.shard_boundaries", "single-shard split has no internal boundary")
        return

    problems: list[str] = []
    tested = 0
    for shard in shards[:-1]:
        boundary = shard.token_end
        # The window that contains the boundary strictly inside it.
        index = (boundary - 1) // EXPECTED_CONTEXT_LENGTH
        for candidate in {index - 1, index, index + 1}:
            if candidate < 0 or candidate >= len(dataset):
                continue
            start = candidate * EXPECTED_CONTEXT_LENGTH
            end = start + EXPECTED_CONTEXT_LENGTH + 1
            if not (start < boundary < end):
                continue

            tested += 1
            window = dataset.read_token_range(start, EXPECTED_CONTEXT_LENGTH + 1)
            if len(window) != EXPECTED_CONTEXT_LENGTH + 1:
                problems.append(
                    f"boundary {boundary}: window {candidate} returned {len(window)} tokens"
                )
                continue
            inputs, targets = dataset[candidate]
            if not bool((inputs[1:] == targets[:-1]).all()):
                problems.append(f"boundary {boundary}: window {candidate} misaligned")
            if window[:-1] != inputs.tolist():
                problems.append(
                    f"boundary {boundary}: window {candidate} disagrees with __getitem__"
                )

    log.verdict(
        f"{split}.shard_boundaries",
        problems,
        f"{tested} spanning windows across {len(shards) - 1} internal boundaries "
        "read whole and stay aligned",
    )


def check_stream_scan(
    scan: StreamScan,
    split: str,
    declared: dict[str, Any],
    *,
    bos_id: int,
    eos_id: int,
    log: CheckLog,
) -> None:
    log.verdict(
        f"{split}.token_ids_in_range",
        scan.out_of_range,
        f"all {scan.tokens:,} ids < {EXPECTED_VOCAB_SIZE:,} "
        f"(max observed {scan.max_token_id:,}), so uint16 storage is exact",
    )

    documents = declared.get("document_count")
    if scan.bos_count == scan.eos_count == documents:
        log.ok(
            f"{split}.document_framing_counts",
            f"{scan.bos_count:,} BOS and {scan.eos_count:,} EOS, one pair per "
            f"declared document",
        )
    else:
        log.fail(
            f"{split}.document_framing_counts",
            f"BOS={scan.bos_count:,} EOS={scan.eos_count:,} documents={documents!r}",
        )

    log.verdict(
        f"{split}.document_framing_order",
        scan.framing_problems,
        "BOS and EOS strictly alternate across the whole stream",
    )

    endpoints: list[str] = []
    if scan.first_token != bos_id:
        endpoints.append(f"stream starts with {scan.first_token!r}, expected BOS {bos_id}")
    if scan.last_token != eos_id:
        endpoints.append(f"stream ends with {scan.last_token!r}, expected EOS {eos_id}")
    log.verdict(
        f"{split}.stream_endpoints", endpoints, "stream opens with BOS and closes with EOS"
    )

    leaked = {
        name: count for name, count in scan.other_special_counts.items() if count
    }
    if leaked:
        log.fail(
            f"{split}.no_accidental_special_parsing",
            f"{sum(leaked.values()):,} non-boundary special token(s) in the stream: {leaked}",
            {"counts": leaked},
        )
    else:
        log.ok(
            f"{split}.no_accidental_special_parsing",
            f"none of the {len(scan.other_special_counts)} non-boundary special "
            "tokens appear anywhere in the stream",
        )


def check_splits_are_distinct(
    manifest: dict[str, Any], datasets: dict[str, PretrainingDataset], log: CheckLog
) -> None:
    train = manifest["splits"]["train"]
    validation = manifest["splits"]["validation"]

    if train["stream_sha256"] != validation["stream_sha256"]:
        log.ok(
            "splits.streams_distinct",
            "train and validation stream digests differ",
        )
    else:
        log.fail(
            "splits.streams_distinct",
            "train and validation have identical stream digests",
        )

    if train["document_identity_sha256"] != validation["document_identity_sha256"]:
        log.ok(
            "splits.identities_distinct",
            "train and validation document-identity digests differ",
        )
    else:
        log.fail("splits.identities_distinct", "identical document-identity digests")

    total_groups = train["split_group_count"] + validation["split_group_count"]
    total_documents = train["document_count"] + validation["document_count"]
    log.ok(
        "splits.totals",
        f"{total_documents:,} documents across {total_groups:,} split groups",
        {
            "train_documents": train["document_count"],
            "validation_documents": validation["document_count"],
            "train_groups": train["split_group_count"],
            "validation_groups": validation["split_group_count"],
        },
    )


def check_split_reproduces_from_corpus(
    corpus: Path,
    manifest: dict[str, Any],
    log: CheckLog,
    *,
    quiet: bool,
) -> None:
    """Re-derive the split from the frozen corpus and compare identity digests.

    A match means the shipped dataset is exactly the declared policy applied to
    the declared corpus.  Because the policy assigns whole groups to one side,
    that also establishes the two splits are document-disjoint -- without
    holding either document list in memory.
    """

    from llm.data.dataset_training import plan_training_split
    from build_pretraining_dataset import discover_input_files, iter_training_documents

    policy = manifest.get("split_policy", {})
    fraction = policy.get("validation_fraction")
    seed = policy.get("seed")
    if not isinstance(fraction, (int, float)) or not isinstance(seed, int):
        log.fail(
            "splits.reproduce_from_corpus",
            f"manifest split_policy is unusable: {policy}",
        )
        return

    files = discover_input_files([str(corpus)])

    def source() -> Iterator[Any]:
        return iter_training_documents(files, text_field="text")

    if not quiet:
        print(f"         re-deriving the split from {len(files)} corpus shards", flush=True)

    plan = plan_training_split(source, validation_fraction=float(fraction), seed=seed)

    digests = {"train": hashlib.sha256(), "validation": hashlib.sha256()}
    counts = {"train": 0, "validation": 0}
    for document in source():
        side = "validation" if plan.is_validation(document) else "train"
        encoded = document.document_id.encode("utf-8", errors="strict")
        digests[side].update(len(encoded).to_bytes(8, "big", signed=False))
        digests[side].update(encoded)
        counts[side] += 1

    problems: list[str] = []
    for split in SPLITS:
        declared = manifest["splits"][split]
        if digests[split].hexdigest() != declared["document_identity_sha256"]:
            problems.append(
                f"{split}: identity digest {digests[split].hexdigest()} != "
                f"manifest {declared['document_identity_sha256']}"
            )
        if counts[split] != declared["document_count"]:
            problems.append(
                f"{split}: {counts[split]} documents != manifest "
                f"{declared['document_count']}"
            )

    log.verdict(
        "splits.reproduce_from_corpus",
        problems,
        f"the split re-derives exactly from the frozen corpus at seed {seed}, "
        f"fraction {fraction}; train and validation are document-disjoint",
    )


def decode_samples(
    dataset: PretrainingDataset,
    tokenizer: Tokenizer,
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(dataset) == 0 or count <= 0:
        return []

    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    for index in sorted({rng.randrange(len(dataset)) for _ in range(count)}):
        inputs, _ = dataset[index]
        text = tokenizer.decode(inputs.tolist(), skip_special_tokens=False)
        samples.append(
            {
                "window": index,
                "characters": len(text),
                "excerpt": text[:300],
            }
        )
    return samples


# --------------------------------------------------------------------------
# freeze
# --------------------------------------------------------------------------


def build_freeze_stamp(
    dataset_dir: Path,
    manifest: dict[str, Any],
    datasets: dict[str, PretrainingDataset],
    tokenizer_path: Path,
    report_digest: str,
) -> dict[str, Any]:
    train = manifest["splits"]["train"]
    validation = manifest["splits"]["validation"]

    return {
        "format": FREEZE_FORMAT,
        "format_version": FREEZE_FORMAT_VERSION,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_dir.name,
        "status": "FROZEN",
        "policy": (
            "This tokenized dataset is immutable. Retokenization, a different "
            "tokenizer, or a different split produces a new dataset version; it "
            "never modifies this one."
        ),
        "tokenizer": {
            "artifact": tokenizer_path.parent.name,
            "path": str(tokenizer_path),
            "vocab_size": manifest["tokenizer"]["vocab_size"],
            "state_sha256": manifest["tokenizer"]["state_sha256"],
            "file_sha256": sha256_file(tokenizer_path),
        },
        "context_length": manifest["training_geometry"]["context_length"],
        "train_documents": train["document_count"],
        "validation_documents": validation["document_count"],
        "train_tokens": train["total_tokens"],
        "validation_tokens": validation["total_tokens"],
        "total_tokens": train["total_tokens"] + validation["total_tokens"],
        "train_windows": train["complete_examples"],
        "validation_windows": validation["complete_examples"],
        "train_shards": train["shard_count"],
        "validation_shards": validation["shard_count"],
        "train_stream_sha256": train["stream_sha256"],
        "validation_stream_sha256": validation["stream_sha256"],
        "train_document_identity_sha256": train["document_identity_sha256"],
        "validation_document_identity_sha256": validation["document_identity_sha256"],
        "dataset_manifest_sha256": sha256_file(dataset_dir / DATASET_MANIFEST_FILENAME),
        "split_manifest_sha256": {
            split: sha256_file(datasets[split].manifest_path) for split in SPLITS
        },
        "validation_report": {"sha256": report_digest},
    }


def compare_freeze_stamp(
    existing: dict[str, Any], candidate: dict[str, Any], log: CheckLog
) -> None:
    drift: list[str] = []
    for key in (
        "train_tokens",
        "validation_tokens",
        "total_tokens",
        "train_windows",
        "validation_windows",
        "train_stream_sha256",
        "validation_stream_sha256",
        "dataset_manifest_sha256",
    ):
        if existing.get(key) != candidate.get(key):
            drift.append(f"{key}: frozen={existing.get(key)!r} now={candidate.get(key)!r}")

    if drift:
        log.fail(
            "freeze.dataset_unchanged",
            f"{len(drift)} difference(s) from the existing freeze stamp",
            {"drift": drift[:MAX_REPORTED_PROBLEMS]},
        )
    else:
        log.ok(
            "freeze.dataset_unchanged",
            f"still matches the freeze stamp from "
            f"{existing.get('frozen_at_utc', 'unknown time')}",
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a tokenized pretraining dataset against its manifest, its "
            "tokenizer, and optionally the frozen corpus it came from."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "Frozen cleaned corpus. Enables re-deriving the split and proving "
            "train/validation document disjointness. Costs two corpus passes."
        ),
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help=f"Write {FREEZE_FILENAME} on a fully passing run with nothing skipped.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip shard and stream SHA-256 verification at construction.",
    )
    parser.add_argument(
        "--skip-stream-scan",
        action="store_true",
        help="Skip the full-stream range, framing, and special-token walk.",
    )
    parser.add_argument("--sample-count", type=positive_int, default=8)
    parser.add_argument("--decode-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)

    try:
        dataset_dir: Path = args.dataset
        if not dataset_dir.is_dir():
            print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
            return 1

        manifest_path = dataset_dir / DATASET_MANIFEST_FILENAME
        if not manifest_path.is_file():
            print(f"ERROR: missing {manifest_path}", file=sys.stderr)
            return 1
        if args.report is not None and args.report.exists() and not args.overwrite:
            print(
                f"ERROR: report already exists (use --overwrite): {args.report}",
                file=sys.stderr,
            )
            return 1

        manifest = load_json(manifest_path)
        tokenizer = Tokenizer.load(args.tokenizer)

        specials = dict(tokenizer.id_to_special_token)
        by_name = {name: token_id for token_id, name in specials.items()}
        bos_id = by_name.get(BOS_TOKEN)
        eos_id = by_name.get(EOS_TOKEN)
        if bos_id is None or eos_id is None:
            print("ERROR: tokenizer has no BOS/EOS special tokens", file=sys.stderr)
            return 1
        other_specials = {
            token_id: name
            for token_id, name in specials.items()
            if token_id not in (bos_id, eos_id)
        }

        log = CheckLog(quiet=args.quiet)

        if not args.quiet:
            print()
            print("TOKENIZED DATASET VALIDATION")
            print("=" * 72)
            print(f"Dataset:   {dataset_dir}")
            print(f"Tokenizer: {args.tokenizer}")
            print()
            print("Contract")
            print("-" * 72)

        check_geometry(manifest, log)
        check_tokenizer_identity(manifest, tokenizer, args.tokenizer, log)

        if not args.quiet:
            print()
            print("Storage")
            print("-" * 72)

        datasets: dict[str, PretrainingDataset] = {}
        for split in SPLITS:
            split_manifest = dataset_dir / split / "manifest.json"
            if not split_manifest.is_file():
                log.fail(f"{split}.storage", f"missing {split_manifest}")
                return 2
            if not args.quiet:
                print(f"         opening {split} ({'verifying digests' if not args.skip_checksums else 'digests skipped'})", flush=True)
            datasets[split] = PretrainingDataset(
                split_manifest,
                context_length=EXPECTED_CONTEXT_LENGTH,
                expected_vocab_size=EXPECTED_VOCAB_SIZE,
                verify_checksums=not args.skip_checksums,
            )
            if args.skip_checksums:
                log.skip(f"{split}.storage", "--skip-checksums")
            else:
                log.ok(
                    f"{split}.storage",
                    f"{len(datasets[split].manifest.shards)} shards: every shard "
                    "SHA-256 and the full stream digest verified",
                )

        if not args.quiet:
            print()
            print("Dataset contract")
            print("-" * 72)

        for split in SPLITS:
            check_dataset_contract(
                datasets[split],
                split,
                manifest["splits"][split],
                log,
                sample_count=args.sample_count,
                seed=args.seed,
            )
            check_shard_boundaries(datasets[split], split, log)

        if not args.quiet:
            print()
            print("Token stream")
            print("-" * 72)

        scans: dict[str, StreamScan] = {}
        if args.skip_stream_scan:
            for split in SPLITS:
                for name in (
                    "token_ids_in_range",
                    "document_framing_counts",
                    "document_framing_order",
                    "stream_endpoints",
                    "no_accidental_special_parsing",
                ):
                    log.skip(f"{split}.{name}", "--skip-stream-scan")
        else:
            for split in SPLITS:
                scans[split] = scan_stream(
                    datasets[split],
                    bos_id=bos_id,
                    eos_id=eos_id,
                    other_specials=other_specials,
                    quiet=args.quiet,
                )
                check_stream_scan(
                    scans[split],
                    split,
                    manifest["splits"][split],
                    bos_id=bos_id,
                    eos_id=eos_id,
                    log=log,
                )

        if not args.quiet:
            print()
            print("Splits")
            print("-" * 72)

        check_splits_are_distinct(manifest, datasets, log)
        if args.corpus is None:
            log.skip(
                "splits.reproduce_from_corpus",
                "--corpus not supplied; disjointness is structural, not verified",
            )
        else:
            check_split_reproduces_from_corpus(
                args.corpus, manifest, log, quiet=args.quiet
            )

        samples = {
            split: decode_samples(
                datasets[split],
                tokenizer,
                count=args.decode_samples,
                seed=args.seed + 1,
            )
            for split in SPLITS
        }

        finished = datetime.now(timezone.utc)
        report = {
            "format": VALIDATOR_FORMAT,
            "format_version": VALIDATOR_FORMAT_VERSION,
            "created_at_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "dataset": {
                "path": str(dataset_dir),
                "manifest_sha256": sha256_file(manifest_path),
            },
            "tokenizer": {
                "path": str(args.tokenizer),
                "file_sha256": sha256_file(args.tokenizer),
            },
            "options": {
                "checksums": not args.skip_checksums,
                "stream_scan": not args.skip_stream_scan,
                "corpus_reproduction": args.corpus is not None,
            },
            "splits": {
                split: {
                    "tokens": datasets[split].total_tokens,
                    "windows": len(datasets[split]),
                    "shards": len(datasets[split].manifest.shards),
                    "documents": manifest["splits"][split]["document_count"],
                    "max_token_id": scans[split].max_token_id if split in scans else None,
                    "bos_count": scans[split].bos_count if split in scans else None,
                    "eos_count": scans[split].eos_count if split in scans else None,
                }
                for split in SPLITS
            },
            "decode_samples": samples,
            "checks": [check.as_dict() for check in log.checks],
            "summary": {
                "total": len(log.checks),
                "passed": sum(1 for c in log.checks if c.status == STATUS_PASS),
                "failed": len(log.failed),
                "skipped": len(log.skipped),
                "warnings": len(log.warned),
                "result": "PASS" if log.passed else "FAIL",
            },
        }

        report_bytes = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        report_digest = hashlib.sha256(report_bytes).hexdigest()
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_bytes(report_bytes)

        freeze_written = False
        freeze_path = dataset_dir / FREEZE_FILENAME
        if args.freeze:
            candidate = build_freeze_stamp(
                dataset_dir, manifest, datasets, args.tokenizer, report_digest
            )
            blockers: list[str] = []
            if not log.passed:
                blockers.append(f"{len(log.failed)} check(s) failed")
            if log.skipped:
                blockers.append(
                    f"{len(log.skipped)} check(s) skipped: "
                    + ", ".join(c.name for c in log.skipped)
                )
            if blockers:
                log.fail("freeze.written", "refused to freeze: " + "; ".join(blockers))
            elif freeze_path.is_file():
                compare_freeze_stamp(load_json(freeze_path), candidate, log)
                if log.passed:
                    log.ok(
                        "freeze.written",
                        f"{FREEZE_FILENAME} already present and still accurate",
                    )
            else:
                freeze_path.write_text(
                    json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                freeze_written = True
                log.ok("freeze.written", f"wrote {freeze_path}")

        if not args.quiet:
            train = manifest["splits"]["train"]
            validation = manifest["splits"]["validation"]
            print()
            print("=" * 72)
            print(
                f"{len(log.checks)} checks: "
                f"{sum(1 for c in log.checks if c.status == STATUS_PASS)} passed, "
                f"{len(log.failed)} failed, {len(log.skipped)} skipped"
            )
            print()
            print(f"{'':<22}{'train':>18}{'validation':>18}")
            print("-" * 72)
            for label, key in (
                ("documents", "document_count"),
                ("tokens", "total_tokens"),
                ("shards", "shard_count"),
                ("windows of 2048", "complete_examples"),
            ):
                print(f"{label:<22}{train[key]:>18,}{validation[key]:>18,}")
            print("-" * 72)
            print(
                f"{'total tokens':<22}"
                f"{train['total_tokens'] + validation['total_tokens']:>36,}"
            )
            print(f"Elapsed: {report['duration_seconds']:.1f}s")
            print()
            if log.failed:
                print("RESULT: FAIL")
                for check in log.failed:
                    print(f"  FAILED {check.name}: {check.detail}")
            else:
                print("RESULT: PASS")
                if freeze_written:
                    print()
                    print("TOKENIZED PRETRAINING DATASET v0.1")
                    print("STATUS: FROZEN")
                    print(f"Stamp:  {freeze_path}")
            if args.report is not None:
                print()
                print(f"Report: {args.report}")

        return 0 if log.passed else 2

    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
