"""Evaluation decontamination for the pretraining corpus.

Purpose
-------
Guarantee that text used to evaluate the trained model does not appear in the
text used to train it.  Without this, a benchmark measures memorisation rather
than capability, and the number it produces is worthless in a way that is very
hard to notice after the fact.

    training corpus  ∩  evaluation corpus  ≈  ∅

Why this is not near-deduplication
----------------------------------
``near_dedup`` asks whether two documents are substantially the same document,
and answers with symmetric Jaccard.  That is the wrong question here, because
contamination is **asymmetric**::

    training page      5,000 words
    evaluation passage   300 words

If the entire evaluation passage appears verbatim inside the training page, the
evaluation item is completely compromised - yet symmetric Jaccard is roughly
300/5000, far below any sane duplicate threshold.  Jaccard would report these
documents as unrelated while the model has already seen every token it will be
tested on.

The correct measure is **containment of the evaluation item**::

    containment = |eval_shingles ∩ document_shingles| / |eval_shingles|

This asks "how much of the evaluation item does this document contain", which
is exactly the contamination question, and is insensitive to how much
unrelated material surrounds it.

Two levels
----------
1. Exact match on the SHA-256 of the normalized text.  Cheap, and catches
   verbatim republication.
2. Shingle containment above a threshold.  Catches an evaluation passage
   embedded in a larger page, and survives formatting, punctuation, and
   casing differences because comparison tokenization discards them.

Asymmetry of cost
-----------------
Near-dedup is deliberately conservative because a false positive destroys
information.  Decontamination is deliberately *aggressive* in the opposite
direction: discarding a training document costs almost nothing, since the
corpus has millions more, while admitting a contaminated one silently
invalidates a benchmark.  When in doubt, drop the training document.

Thresholds here are audit parameters, not frozen constants.  Run the audit and
read the matches before deciding a production value.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from llm.data.near_dedup import (
    DEFAULT_SHINGLE_SIZE,
    comparison_tokens,
    hashed_word_shingles,
)
from llm.tokenizer.normalizer import normalize_text


DEFAULT_MIN_CONTAINMENT = 0.80
DEFAULT_MIN_EVALUATION_TOKENS = 50

REASON_EXACT_MATCH = "evaluation_exact_match"
REASON_CONTAINMENT = "evaluation_containment"

EVALUATION_FORMAT = "llm_pretraining_evaluation"
EVALUATION_FORMAT_VERSION = 1


@dataclass(frozen=True)
class DecontaminationConfig:
    """Detection policy.

    ``min_containment`` is the fraction of an evaluation item's shingles that
    must appear in a training document before that document is rejected.
    """

    shingle_size: int = DEFAULT_SHINGLE_SIZE
    min_containment: float = DEFAULT_MIN_CONTAINMENT
    min_evaluation_tokens: int = DEFAULT_MIN_EVALUATION_TOKENS

    def __post_init__(self) -> None:
        if self.shingle_size < 1:
            raise ValueError("shingle_size must be >= 1")
        if not 0.0 < self.min_containment <= 1.0:
            raise ValueError("min_containment must be within (0, 1]")
        if self.min_evaluation_tokens < 1:
            raise ValueError("min_evaluation_tokens must be >= 1")


@dataclass(frozen=True)
class EvaluationItem:
    """One frozen held-out evaluation passage."""

    item_id: str
    category: str
    text: str


@dataclass(frozen=True)
class ContaminationMatch:
    """Evidence that a training document overlaps an evaluation item."""

    item_id: str
    category: str
    reason: str
    containment: float
    shared_shingles: int
    evaluation_shingles: int


@dataclass(frozen=True)
class ContaminationVerdict:
    """Outcome of checking one training document."""

    contaminated: bool
    matches: tuple[ContaminationMatch, ...]

    @property
    def best(self) -> ContaminationMatch | None:
        """Strongest evidence, or ``None`` when the document is clean."""

        return self.matches[0] if self.matches else None

    @property
    def reason(self) -> str | None:
        best = self.best
        return best.reason if best else None


def normalized_fingerprint(text: str) -> str:
    """SHA-256 hex digest of normalized text, matching the corpus contract."""

    return hashlib.sha256(
        normalize_text(text).encode("utf-8")
    ).hexdigest()


def containment(
    evaluation_shingles: Sequence[int],
    document_shingles: Sequence[int],
) -> float:
    """Fraction of the evaluation item present in the document.

    Asymmetric by design: a short evaluation passage fully embedded in a long
    document yields 1.0, whereas symmetric Jaccard would be near zero.
    """

    if not evaluation_shingles:
        return 0.0

    document_set = set(document_shingles)
    shared = sum(1 for value in evaluation_shingles if value in document_set)
    return shared / len(evaluation_shingles)


def load_evaluation_items(path: Path | str) -> list[EvaluationItem]:
    """Read ``pretraining_eval.jsonl``.

    Each line is an object with ``id``, ``category``, and ``text``.  Duplicate
    identifiers are rejected: the evaluation set is a frozen artifact and an
    accidental duplicate would silently double an item's weight.
    """

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"evaluation file does not exist: {resolved}")

    items: list[EvaluationItem] = []
    seen: set[str] = set()

    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {resolved.name}:{line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"evaluation record must be an object: "
                    f"{resolved.name}:{line_number}"
                )

            item_id = record.get("id")
            text = record.get("text")
            category = record.get("category", "uncategorized")

            if not isinstance(item_id, str) or not item_id:
                raise ValueError(
                    f"evaluation record missing id: {resolved.name}:{line_number}"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"evaluation record {item_id!r} has no text"
                )
            if item_id in seen:
                raise ValueError(f"duplicate evaluation id: {item_id!r}")

            seen.add(item_id)
            items.append(
                EvaluationItem(
                    item_id=item_id,
                    category=str(category),
                    text=text,
                )
            )

    if not items:
        raise ValueError(f"evaluation file contains no items: {resolved}")

    return items


class EvaluationIndex:
    """Index of frozen evaluation material, queried per training document.

    Built once and reused for every candidate.  An inverted index from shingle
    hash to evaluation items keeps the per-document cost proportional to the
    document's own size rather than to the size of the evaluation set.
    """

    def __init__(
        self,
        items: Iterable[EvaluationItem],
        *,
        config: DecontaminationConfig | None = None,
    ) -> None:
        self.config = config or DecontaminationConfig()

        self.items: list[EvaluationItem] = []
        self.skipped_short: list[str] = []

        self._exact: dict[str, int] = {}
        self._postings: dict[int, list[int]] = {}
        self._shingle_counts: list[int] = []

        for item in items:
            tokens = comparison_tokens(item.text)
            if len(tokens) < self.config.min_evaluation_tokens:
                # Too short to identify reliably. A handful of tokens would
                # match innocent documents constantly.
                self.skipped_short.append(item.item_id)
                continue

            index = len(self.items)
            self.items.append(item)

            self._exact.setdefault(normalized_fingerprint(item.text), index)

            shingles = hashed_word_shingles(
                item.text,
                shingle_size=self.config.shingle_size,
            )
            self._shingle_counts.append(len(shingles))

            for value in shingles:
                self._postings.setdefault(value, []).append(index)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def indexed_shingles(self) -> int:
        return len(self._postings)

    def check(self, text: str) -> ContaminationVerdict:
        """Decide whether a training document is contaminated."""

        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        if not self.items:
            return ContaminationVerdict(False, ())

        matches: list[ContaminationMatch] = []

        exact_index = self._exact.get(normalized_fingerprint(text))
        if exact_index is not None:
            item = self.items[exact_index]
            matches.append(
                ContaminationMatch(
                    item_id=item.item_id,
                    category=item.category,
                    reason=REASON_EXACT_MATCH,
                    containment=1.0,
                    shared_shingles=self._shingle_counts[exact_index],
                    evaluation_shingles=self._shingle_counts[exact_index],
                )
            )

        document_shingles = hashed_word_shingles(
            text,
            shingle_size=self.config.shingle_size,
        )

        # Walk the document's shingles once, tallying hits per evaluation item.
        # Most shingles have no posting at all, so this is cheap.
        shared_counts: Counter[int] = Counter()
        for value in document_shingles:
            postings = self._postings.get(value)
            if postings:
                shared_counts.update(postings)

        for index, shared in shared_counts.items():
            total = self._shingle_counts[index]
            if total == 0:
                continue
            score = shared / total
            if score < self.config.min_containment:
                continue
            if index == exact_index:
                # Already reported with the stronger exact-match reason.
                continue

            item = self.items[index]
            matches.append(
                ContaminationMatch(
                    item_id=item.item_id,
                    category=item.category,
                    reason=REASON_CONTAINMENT,
                    containment=score,
                    shared_shingles=shared,
                    evaluation_shingles=total,
                )
            )

        # Exact matches first, then by how much of the item was found.
        matches.sort(
            key=lambda match: (
                match.reason != REASON_EXACT_MATCH,
                -match.containment,
                match.item_id,
            )
        )

        return ContaminationVerdict(bool(matches), tuple(matches))

    def category_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter(item.category for item in self.items)
        return dict(sorted(counts.items()))


def iter_evaluation_categories(
    items: Sequence[EvaluationItem],
) -> Iterator[tuple[str, int]]:
    """Yield ``(category, count)`` pairs in deterministic order."""

    counts: Counter[str] = Counter(item.category for item in items)
    yield from sorted(counts.items())
