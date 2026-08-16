"""Build deterministic train/validation pretraining token datasets.

This module owns the *training-dataset build* stage that sits above document
encoding and binary shard storage.  It deliberately does not contain the neural
network training loop.

Pipeline
--------

    cleaned documents
        -> deterministic document/group split
        -> encode each document as <BOS> content <EOS>
        -> concatenate documents within each split
        -> write flat uint16 token shards
        -> dataset_manifest.json

Train/validation separation happens before tokenization and before packing.
Documents that share a ``split_group`` are always assigned to the same split,
which lets upstream chunked records keep all chunks from one original source
document together.

The binary shards remain flat token streams.  :class:`llm.data.dataset.PretrainingDataset`
constructs ``context_length + 1`` causal windows dynamically at training time.

Documents are streamed, never collected
---------------------------------------
Nothing here holds the corpus in memory.  Callers supply a *document source*:
either a sequence, or a zero-argument callable returning a fresh iterator.  A
one-shot iterator is rejected, because the only way to honour it would be to
materialise it, and materialising a 5 GB corpus is precisely the failure this
module exists to avoid.

The build makes three passes over the source:

    pass 1  identity, split_group, normalized byte count   -> the split plan
    pass 2  train documents      -> tokenize -> train shards
    pass 3  validation documents -> tokenize -> validation shards

Passes 2 and 3 each walk the whole source but tokenize only their own side, so
every document is still tokenized exactly once; the extra cost is re-reading
and re-parsing the cleaned corpus, not re-encoding it.  Splitting the token
writes this way keeps :func:`llm.data.shards.write_token_shards` a single
sequential stream writer, which is what makes the shard bytes identical to
those the previous in-memory implementation produced.

Peak memory is therefore set by the number of distinct ``document_id`` and
``split_group`` values, not by the size of the text.  For a corpus of 1.2M
documents that is a few hundred megabytes of identifiers; the text itself is
never resident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
from typing import Callable, Iterable, Iterator, Sequence

from llm.data.document import encode_pretraining_document
from llm.data.packing import DEFAULT_CONTEXT_LENGTH
from llm.data.shards import DEFAULT_TOKENS_PER_SHARD, ShardManifest, write_token_shards
from llm.tokenizer.normalizer import normalize_text
from llm.tokenizer.tokenizer import Tokenizer


DATASET_FORMAT = "llm_pretraining_dataset"
DATASET_FORMAT_VERSION = 1
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
DEFAULT_VALIDATION_FRACTION = 0.01
DEFAULT_SPLIT_SEED = 2026

#: A zero-argument callable returning a fresh iterator over the corpus.  The
#: build calls it once per pass, so it must yield the same documents in the
#: same order every time.
DocumentSource = Callable[[], Iterator["TrainingDocument"]]


@dataclass(frozen=True)
class TrainingDocument:
    """One cleaned source document plus deterministic split identity."""

    document_id: str
    text: str
    split_group: str | None = None

    @property
    def effective_split_group(self) -> str:
        return self.split_group if self.split_group is not None else self.document_id


@dataclass(frozen=True)
class SplitPlan:
    """The complete train/validation decision, without any document text.

    This is what pass 1 produces and passes 2 and 3 consume.  It is small: one
    entry per validation group plus a handful of counters, so the plan for a
    1.2M-document corpus is a few megabytes even though the corpus is gigabytes.
    """

    validation_groups: frozenset[str]
    document_count: int
    group_count: int
    total_normalized_utf8_bytes: int
    validation_normalized_utf8_bytes: int
    target_validation_bytes: float
    identity_sha256: str

    @property
    def validation_group_count(self) -> int:
        return len(self.validation_groups)

    @property
    def train_group_count(self) -> int:
        return self.group_count - len(self.validation_groups)

    @property
    def achieved_validation_fraction(self) -> float:
        if self.total_normalized_utf8_bytes == 0:
            return 0.0
        return (
            self.validation_normalized_utf8_bytes / self.total_normalized_utf8_bytes
        )

    def is_validation(self, document: TrainingDocument) -> bool:
        return document.effective_split_group in self.validation_groups


@dataclass
class _SplitAccumulator:
    """Streaming replacement for the per-split summary of a document list.

    Every field is a running total updated as documents pass through, so the
    manifest can report exactly what the previous implementation reported
    without a list of documents to walk a second time.
    """

    document_count: int = 0
    normalized_characters: int = 0
    normalized_utf8_bytes: int = 0
    _identity: "hashlib._Hash" = field(default_factory=hashlib.sha256)

    def add(self, document_id: str, normalized_text: str) -> None:
        self.document_count += 1
        self.normalized_characters += len(normalized_text)
        self.normalized_utf8_bytes += len(normalized_text.encode("utf-8", errors="strict"))
        encoded = document_id.encode("utf-8", errors="strict")
        self._identity.update(len(encoded).to_bytes(8, "big", signed=False))
        self._identity.update(encoded)

    @property
    def identity_sha256(self) -> str:
        return self._identity.hexdigest()


@dataclass(frozen=True)
class DatasetBuildResult:
    """Summary returned after a successful dataset build."""

    output_dir: Path
    train_manifest: ShardManifest
    validation_manifest: ShardManifest
    train_document_count: int
    validation_document_count: int
    context_length: int
    dataset_manifest_path: Path


def _validate_tokenizer(tokenizer: Tokenizer) -> None:
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must be a Tokenizer, got {type(tokenizer).__name__}"
        )


def _validate_context_length(context_length: int) -> None:
    if not isinstance(context_length, int) or isinstance(context_length, bool):
        raise TypeError("context_length must be an integer")
    if context_length <= 0:
        raise ValueError("context_length must be > 0")


def _validate_validation_fraction(validation_fraction: float) -> None:
    if isinstance(validation_fraction, bool) or not isinstance(
        validation_fraction, (int, float)
    ):
        raise TypeError("validation_fraction must be a number")
    value = float(validation_fraction)
    if value < 0.0 or value >= 1.0:
        raise ValueError("validation_fraction must satisfy 0 <= value < 1")


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")


def _validate_document_fields(document: TrainingDocument) -> TrainingDocument:
    if not isinstance(document, TrainingDocument):
        raise TypeError(
            "documents must contain TrainingDocument objects, got "
            f"{type(document).__name__}"
        )
    if not isinstance(document.document_id, str) or not document.document_id:
        raise ValueError("document_id must be a non-empty string")
    if not isinstance(document.text, str):
        raise TypeError("document text must be a string")
    if document.split_group is not None and (
        not isinstance(document.split_group, str) or not document.split_group
    ):
        raise ValueError("split_group must be None or a non-empty string")
    return document


def _validated_normalized_text(document: TrainingDocument) -> str:
    """Validate a document and return its normalized text once.

    Normalization is the expensive part of validation, and every caller that
    validates also needs the normalized form -- for byte accounting, character
    counts, or the identity hash.  Returning it means the corpus is normalized
    once per pass instead of once per question asked about it.
    """

    _validate_document_fields(document)
    return normalize_text(document.text)


def _validate_document(document: TrainingDocument) -> TrainingDocument:
    # Validate the canonical text policy now so split byte accounting uses the
    # exact normalized representation the tokenizer will later encode.
    _validated_normalized_text(document)
    return document


def _as_document_source(
    documents: Sequence[TrainingDocument] | DocumentSource,
) -> DocumentSource:
    """Coerce a caller's argument into a re-iterable document source.

    A one-shot iterator is refused rather than quietly buffered.  Buffering is
    the exact behaviour this module was rewritten to remove, and doing it
    silently would reintroduce an out-of-memory failure that only appears at
    full corpus scale, long after the small-input tests have passed.
    """

    if callable(documents):
        return documents
    if isinstance(documents, Sequence) and not isinstance(documents, (str, bytes)):
        sequence = documents
        return lambda: iter(sequence)
    raise TypeError(
        "documents must be a sequence or a zero-argument callable returning a "
        f"fresh iterator, got {type(documents).__name__}. A one-shot iterator "
        "cannot be used: the build streams the corpus more than once, and "
        "buffering it would defeat the point."
    )


def _group_rank(group: str, seed: int) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"LLM_PRETRAIN_SPLIT_V1\x00")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\x00")
    digest.update(group.encode("utf-8", errors="strict"))
    return digest.digest()


def _normalized_utf8_bytes(document: TrainingDocument) -> int:
    return len(normalize_text(document.text).encode("utf-8", errors="strict"))


def _document_identity_sha256(documents: Iterable[TrainingDocument]) -> str:
    """Hash ordered document identities without storing a giant id list."""

    digest = hashlib.sha256()
    for document in documents:
        encoded = document.document_id.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _tokenizer_state_sha256(tokenizer: Tokenizer) -> str:
    payload = json.dumps(
        tokenizer.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def plan_training_split(
    documents: Sequence[TrainingDocument] | DocumentSource,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> SplitPlan:
    """Decide the train/validation split in one streaming pass (pass 1).

    Groups are sorted by a seeded SHA-256 rank.  Validation receives the prefix
    whose cumulative normalized UTF-8 bytes is closest to the requested byte
    fraction.  When ``validation_fraction > 0`` and at least two groups exist,
    both train and validation are guaranteed to contain at least one group.

    Only three things are read from each document -- identity, split group, and
    normalized byte count -- so the text is released as soon as it has been
    measured.  What survives the pass is one integer per distinct group.

    The ranking, the byte target, and the tie-breaking rule are unchanged from
    the original list-based implementation, and the group-rank sort is total
    (rank, then group name), so the resulting split does not depend on the
    order groups were first encountered.
    """

    _validate_validation_fraction(validation_fraction)
    _validate_seed(seed)
    source = _as_document_source(documents)

    seen_ids: set[str] = set()
    group_bytes: dict[str, int] = {}
    identity = hashlib.sha256()
    document_count = 0

    for document in source():
        normalized = _validated_normalized_text(document)
        if document.document_id in seen_ids:
            raise ValueError(f"duplicate document_id: {document.document_id!r}")
        seen_ids.add(document.document_id)

        encoded_id = document.document_id.encode("utf-8", errors="strict")
        identity.update(len(encoded_id).to_bytes(8, "big", signed=False))
        identity.update(encoded_id)

        group = document.effective_split_group
        group_bytes[group] = group_bytes.get(group, 0) + len(
            normalized.encode("utf-8", errors="strict")
        )
        document_count += 1

    del seen_ids

    total_bytes = sum(group_bytes.values())
    identity_sha256 = identity.hexdigest()

    if document_count == 0 or float(validation_fraction) == 0.0:
        return SplitPlan(
            validation_groups=frozenset(),
            document_count=document_count,
            group_count=len(group_bytes),
            total_normalized_utf8_bytes=total_bytes,
            validation_normalized_utf8_bytes=0,
            target_validation_bytes=0.0,
            identity_sha256=identity_sha256,
        )

    if len(group_bytes) < 2:
        raise ValueError(
            "validation_fraction > 0 requires at least two distinct split groups"
        )

    ranked_groups = sorted(
        ((_group_rank(group, seed), group, size) for group, size in group_bytes.items()),
        key=lambda item: (item[0], item[1]),
    )
    target_validation_bytes = total_bytes * float(validation_fraction)

    # Choose a non-empty, non-complete prefix whose cumulative byte total is
    # closest to the requested validation fraction.
    cumulative = 0
    candidates: list[tuple[float, int]] = []
    for prefix_count, (_, _, size) in enumerate(ranked_groups[:-1], start=1):
        cumulative += size
        candidates.append((abs(cumulative - target_validation_bytes), prefix_count))

    _, chosen_prefix_count = min(candidates, key=lambda item: (item[0], item[1]))
    chosen = ranked_groups[:chosen_prefix_count]

    return SplitPlan(
        validation_groups=frozenset(group for _, group, _ in chosen),
        document_count=document_count,
        group_count=len(group_bytes),
        total_normalized_utf8_bytes=total_bytes,
        validation_normalized_utf8_bytes=sum(size for _, _, size in chosen),
        target_validation_bytes=target_validation_bytes,
        identity_sha256=identity_sha256,
    )


def split_training_documents(
    documents: Sequence[TrainingDocument] | DocumentSource,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[TrainingDocument], list[TrainingDocument]]:
    """Return the split as two document lists.

    This is the eager form of :func:`plan_training_split`, kept for callers
    working with corpora small enough to hold.  It returns documents in their
    original input order within each split.

    The dataset build does **not** use this function: it holds every document
    twice over, which is exactly what fails at full corpus scale.
    """

    source = _as_document_source(documents)
    plan = plan_training_split(
        source, validation_fraction=validation_fraction, seed=seed
    )

    train: list[TrainingDocument] = []
    validation: list[TrainingDocument] = []
    for document in source():
        if plan.is_validation(document):
            validation.append(document)
        else:
            train.append(document)

    if plan.validation_groups and (not train or not validation):
        raise RuntimeError("internal split invariant produced an empty required split")

    return train, validation


def _encoded_token_stream(
    documents: Iterable[TrainingDocument],
    tokenizer: Tokenizer,
):
    for document in documents:
        for token_id in encode_pretraining_document(document.text, tokenizer):
            yield token_id


def _encoded_split_stream(
    source: DocumentSource,
    plan: SplitPlan,
    tokenizer: Tokenizer,
    accumulator: _SplitAccumulator,
    *,
    want_validation: bool,
    full_identity: "hashlib._Hash | None" = None,
):
    """Yield one split's token stream while measuring it in flight.

    The shard writer pulls tokens from this generator, so the split's counters
    are filled in as a side effect of writing.  Nothing is retained: each
    document is normalized, measured, encoded, and dropped.

    ``full_identity`` is updated for *every* document seen, not just this
    split's.  Running it during the train pass gives a digest directly
    comparable with pass 1's, which is how a source that changed between
    passes is caught before it can produce a silently wrong dataset.
    """

    for document in source():
        normalized = _validated_normalized_text(document)

        if full_identity is not None:
            encoded_id = document.document_id.encode("utf-8", errors="strict")
            full_identity.update(len(encoded_id).to_bytes(8, "big", signed=False))
            full_identity.update(encoded_id)

        if plan.is_validation(document) is not want_validation:
            continue

        accumulator.add(document.document_id, normalized)
        yield from encode_pretraining_document(document.text, tokenizer)


def _split_summary(
    accumulator: _SplitAccumulator,
    manifest: ShardManifest,
    *,
    context_length: int,
    split_group_count: int,
) -> dict[str, object]:
    characters = accumulator.normalized_characters
    utf8_bytes = accumulator.normalized_utf8_bytes

    examples = (
        (manifest.total_tokens - 1) // context_length
        if manifest.total_tokens > 1
        else 0
    )

    return {
        "document_count": accumulator.document_count,
        "split_group_count": split_group_count,
        "normalized_characters": characters,
        "normalized_utf8_bytes": utf8_bytes,
        "document_identity_sha256": accumulator.identity_sha256,
        "total_tokens": manifest.total_tokens,
        "shard_count": manifest.shard_count,
        "complete_examples": examples,
        "prediction_tokens": examples * context_length,
        "tail_tokens": (
            manifest.total_tokens - examples * context_length
            if manifest.total_tokens > 0
            else 0
        ),
        "manifest": f"{manifest.split}/manifest.json",
        "stream_sha256": manifest.stream_sha256,
    }


def _verify_source_was_stable(
    plan: SplitPlan,
    train: _SplitAccumulator,
    validation: _SplitAccumulator,
    observed_identity: "hashlib._Hash",
) -> None:
    """Confirm passes 2 and 3 saw the same corpus pass 1 planned against.

    Streaming trades a private in-memory copy for repeated reads of something
    the caller controls.  If that something changes between passes -- a file
    rewritten, a generator that filters differently on a second call -- the
    split plan silently stops matching the data, and the resulting dataset is
    wrong in ways no downstream check would attribute to this stage.

    The ordered document-identity digest catches any such change.
    """

    total = train.document_count + validation.document_count
    if total != plan.document_count:
        raise RuntimeError(
            "document source is not stable across passes: the split plan saw "
            f"{plan.document_count} documents, tokenization saw {total}"
        )

    if observed_identity.hexdigest() != plan.identity_sha256:
        raise RuntimeError(
            "document source is not stable across passes: document identities "
            "differ between the planning pass and the tokenization pass"
        )

    if plan.validation_groups and (
        train.document_count == 0 or validation.document_count == 0
    ):
        raise RuntimeError("internal split invariant produced an empty required split")


def build_pretraining_dataset(
    documents: Sequence[TrainingDocument] | DocumentSource,
    tokenizer: Tokenizer,
    output_dir: str | Path,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    tokens_per_shard: int = DEFAULT_TOKENS_PER_SHARD,
    overwrite: bool = False,
) -> DatasetBuildResult:
    """Build train/validation token shards and a deterministic top manifest.

    ``documents`` is a sequence or a zero-argument callable returning a fresh
    iterator; see :data:`DocumentSource`.  The corpus is streamed three times
    and never held, so peak memory is set by the number of distinct identifiers
    rather than by the volume of text.
    """

    _validate_tokenizer(tokenizer)
    _validate_validation_fraction(validation_fraction)
    _validate_seed(seed)
    _validate_context_length(context_length)
    if not isinstance(tokens_per_shard, int) or isinstance(tokens_per_shard, bool):
        raise TypeError("tokens_per_shard must be an integer")
    if tokens_per_shard <= 0:
        raise ValueError("tokens_per_shard must be > 0")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")

    source = _as_document_source(documents)

    output_path = Path(output_dir)
    parent = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Checked before the planning pass: on a full corpus that pass costs
    # minutes, and there is no reason to spend them only to refuse to write.
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"pretraining dataset output already exists: {output_path}")

    stage_path = parent / f".{output_path.name}.buildtmp"
    backup_path = parent / f".{output_path.name}.backup"
    if stage_path.exists() or backup_path.exists():
        raise FileExistsError(
            "stale dataset build/backup directory exists; inspect and remove it first: "
            f"{stage_path} or {backup_path}"
        )

    plan = plan_training_split(
        source,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    try:
        stage_path.mkdir(parents=True)

        train_accumulator = _SplitAccumulator()
        validation_accumulator = _SplitAccumulator()
        observed_identity = hashlib.sha256()

        train_manifest = write_token_shards(
            _encoded_split_stream(
                source,
                plan,
                tokenizer,
                train_accumulator,
                want_validation=False,
                full_identity=observed_identity,
            ),
            stage_path / "train",
            tokenizer,
            split="train",
            tokens_per_shard=tokens_per_shard,
        )
        validation_manifest = write_token_shards(
            _encoded_split_stream(
                source,
                plan,
                tokenizer,
                validation_accumulator,
                want_validation=True,
            ),
            stage_path / "validation",
            tokenizer,
            split="validation",
            tokens_per_shard=tokens_per_shard,
        )

        _verify_source_was_stable(
            plan,
            train_accumulator,
            validation_accumulator,
            observed_identity,
        )

        manifest_payload = {
            "format": DATASET_FORMAT,
            "format_version": DATASET_FORMAT_VERSION,
            "tokenizer": {
                "vocab_size": tokenizer.vocab_size,
                "state_sha256": _tokenizer_state_sha256(tokenizer),
            },
            "split_policy": {
                "method": "seeded_sha256_group_prefix_by_normalized_utf8_bytes",
                "validation_fraction": float(validation_fraction),
                "seed": seed,
                "split_before_tokenization": True,
                "group_safe": True,
            },
            "training_geometry": {
                "context_length": context_length,
                "window_tokens": context_length + 1,
                "window_stride": context_length,
                "incomplete_final_tail_padded": False,
            },
            "storage": {
                "dtype": "uint16",
                "byte_order": "little",
                "tokens_per_shard": tokens_per_shard,
            },
            "splits": {
                "train": _split_summary(
                    train_accumulator,
                    train_manifest,
                    context_length=context_length,
                    split_group_count=plan.train_group_count,
                ),
                "validation": _split_summary(
                    validation_accumulator,
                    validation_manifest,
                    context_length=context_length,
                    split_group_count=plan.validation_group_count,
                ),
            },
        }

        dataset_manifest_path = stage_path / DATASET_MANIFEST_FILENAME
        dataset_manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        # Commit the complete staged dataset only after both splits and the top
        # manifest have been written successfully.  With overwrite=True, keep
        # the old directory as a same-filesystem backup until the new stage is
        # in place.
        if output_path.exists():
            output_path.rename(backup_path)
        try:
            stage_path.rename(output_path)
        except Exception:
            if backup_path.exists() and not output_path.exists():
                backup_path.rename(output_path)
            raise
        else:
            if backup_path.exists():
                shutil.rmtree(backup_path)

        return DatasetBuildResult(
            output_dir=output_path,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
            train_document_count=train_accumulator.document_count,
            validation_document_count=validation_accumulator.document_count,
            context_length=context_length,
            dataset_manifest_path=output_path / DATASET_MANIFEST_FILENAME,
        )

    except Exception:
        if stage_path.exists():
            shutil.rmtree(stage_path)
        if backup_path.exists() and not output_path.exists():
            backup_path.rename(output_path)
        raise
