"""Near-duplicate detection for cleaned pretraining documents.

This module is deliberately separate from exact deduplication.

Exact dedup answers:
    "Is this normalized document byte-for-byte identical to one already seen?"

Near dedup answers:
    "Is this document *substantially the same content* as one already seen,
     despite small edits such as a changed footer, date, title casing, or
     punctuation?"

Design goals
------------
* First principles: Python standard library only.
* Deterministic across processes and machines.
* Streaming-friendly and SQLite-backed.
* Transaction-aware so it can later share the corpus builder's
  "one input shard = one transaction" crash-safety rule.
* Conservative: candidate generation may over-propose, but the final decision
  uses exact Jaccard similarity over hashed word shingles plus a length-ratio
  guard.
* Audit-first: this module does not delete or rewrite corpus text.

Algorithm
---------
1. Comparison-only tokenization:
   - Unicode casefold
   - keep Unicode ``\\w+`` tokens
   - punctuation/whitespace differences do not matter
   The original training text is NEVER modified.

2. Build 8-token shingles.

3. Hash shingles deterministically with a 64-bit rolling hash.  Individual
   token hashes use BLAKE2b and an LRU cache; shingle hashes then slide in O(1).

4. Build an 8-component partitioned MinHash-style signature.  Each 64-bit
   shingle hash selects one partition by its high bits; the minimum remaining
   value becomes that component.

5. Split the signature into 4 bands of 2 components and use SQLite as an LSH
   candidate index.

6. Candidate pairs are verified with exact Jaccard similarity over the stored
   sorted unique 64-bit shingle hashes.

The default acceptance rule for an actual near duplicate is:

    exact shingle Jaccard >= 0.90
    AND
    token-count length ratio >= 0.90

This is intentionally strict.  A false negative leaves some duplicated
training text; a false positive silently deletes potentially useful knowledge.

Why similarity alone is not sufficient
--------------------------------------
A 100k audit showed that a percentage threshold answers only half the question.
Jaccard says *how much is common*; it says nothing about *how much would be
lost*.  Two pages carrying a 2,000-shingle site template can exceed 0.90
similarity while differing in genuinely distinct technical payloads - two NIST
chemical species, two Arctic ice observations from different years, two
machined parts with different tolerances.  Deleting one of those pairs silently
discards real information.

Every verified match therefore also reports absolute unique-shingle counts and
is classified:

    max(unique_a, unique_b) <= max_unique_shingles  -> safe_near_duplicate
    otherwise                                       -> ambiguous_overlap

A changed copyright year disturbs only the handful of shingles that span the
changed token, so it lands in ``safe_near_duplicate``.  A different
specification block contributes dozens of unique shingles even when the
surrounding template is identical, so it lands in ``ambiguous_overlap`` and is
kept.

This module classifies; it does not delete.  Removal policy belongs to the
caller.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re
import sqlite3
import struct
import sys
from types import TracebackType
from typing import Iterable, Sequence
import zlib


UINT64_MASK = (1 << 64) - 1
ROLLING_BASE = 1_000_003

DEFAULT_SHINGLE_SIZE = 8
DEFAULT_SIGNATURE_COMPONENTS = 8
DEFAULT_BANDS = 4
DEFAULT_ROWS_PER_BAND = 2
DEFAULT_SIMILARITY_THRESHOLD = 0.90
DEFAULT_MIN_LENGTH_RATIO = 0.90
DEFAULT_MIN_TOKENS = 50
DEFAULT_MAX_CANDIDATES = 2_000

# Absolute unique-information guard. An audit parameter, not a frozen constant:
# it should be tuned against real classification counts before any deletion
# policy depends on it.
DEFAULT_MAX_UNIQUE_SHINGLES = 16

# Relative companion to the absolute guard. An absolute count alone is
# scale-blind: 15 unique shingles is 9% of a 161-shingle NIST species record but
# 0.3% of a 4,557-shingle forum template. The first is a distinct technical
# record, the second is boilerplate. Ordering by absolute count alone places the
# record above the boilerplate, so no single absolute threshold separates them.
DEFAULT_MAX_UNIQUE_SHARE = 0.02

# Classifications for a verified match.
SAFE_NEAR_DUPLICATE = "safe_near_duplicate"
AMBIGUOUS_OVERLAP = "ambiguous_overlap"

_WORD_RE = re.compile(r"\w+", re.UNICODE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    token_count   INTEGER NOT NULL,
    shingle_count INTEGER NOT NULL,
    signature     BLOB NOT NULL,
    shingles      BLOB NOT NULL,
    url           TEXT,
    excerpt       TEXT
);

CREATE TABLE IF NOT EXISTS bands (
    band_index INTEGER NOT NULL,
    band_key   BLOB NOT NULL,
    document_id TEXT NOT NULL,
    PRIMARY KEY (band_index, band_key, document_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_bands_lookup
    ON bands (band_index, band_key);
"""


@dataclass(frozen=True)
class NearDedupConfig:
    """Configuration for candidate generation and final verification."""

    shingle_size: int = DEFAULT_SHINGLE_SIZE
    signature_components: int = DEFAULT_SIGNATURE_COMPONENTS
    bands: int = DEFAULT_BANDS
    rows_per_band: int = DEFAULT_ROWS_PER_BAND
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    min_length_ratio: float = DEFAULT_MIN_LENGTH_RATIO
    max_unique_shingles: int = DEFAULT_MAX_UNIQUE_SHINGLES
    max_unique_share: float = DEFAULT_MAX_UNIQUE_SHARE
    min_tokens: int = DEFAULT_MIN_TOKENS
    max_candidates: int = DEFAULT_MAX_CANDIDATES

    def __post_init__(self) -> None:
        if self.shingle_size < 1:
            raise ValueError("shingle_size must be >= 1")
        if self.signature_components < 2:
            raise ValueError("signature_components must be >= 2")
        if self.signature_components & (self.signature_components - 1):
            raise ValueError("signature_components must be a power of two")
        if self.bands < 1 or self.rows_per_band < 1:
            raise ValueError("bands and rows_per_band must be >= 1")
        if self.bands * self.rows_per_band != self.signature_components:
            raise ValueError(
                "bands * rows_per_band must equal signature_components"
            )
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be within [0, 1]")
        if not 0.0 <= self.min_length_ratio <= 1.0:
            raise ValueError("min_length_ratio must be within [0, 1]")
        if self.max_unique_shingles < 0:
            raise ValueError("max_unique_shingles must be >= 0")
        if not 0.0 <= self.max_unique_share <= 1.0:
            raise ValueError("max_unique_share must be within [0, 1]")
        if self.min_tokens < 1:
            raise ValueError("min_tokens must be >= 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")


@dataclass(frozen=True)
class DocumentFeatures:
    """Comparison features derived from one document."""

    token_count: int
    shingle_hashes: tuple[int, ...]
    signature: tuple[int, ...]

    @property
    def shingle_count(self) -> int:
        return len(self.shingle_hashes)


@dataclass(frozen=True)
class ShingleOverlap:
    """Set arithmetic between two documents' shingle sets.

    ``jaccard`` answers "how much is common".  ``max_unique`` answers the
    question a ratio cannot: "how much distinct content would be destroyed by
    treating these as the same document".
    """

    shared: int
    unique_first: int
    unique_second: int

    @property
    def union(self) -> int:
        return self.shared + self.unique_first + self.unique_second

    @property
    def jaccard(self) -> float:
        union = self.union
        # Two empty documents are vacuously identical.
        return self.shared / union if union else 1.0

    @property
    def max_unique(self) -> int:
        return max(self.unique_first, self.unique_second)

    @property
    def unique_share(self) -> float:
        """Unique content as a fraction of the combined shingle set.

        This is what makes the guard scale-aware. The same absolute count means
        something very different in a 161-shingle data record than in a
        4,557-shingle site template.
        """

        union = self.union
        return self.max_unique / union if union else 0.0


@dataclass(frozen=True)
class NearDuplicateMatch:
    """A verified near-duplicate relation against an indexed document."""

    document_id: str
    similarity: float
    length_ratio: float
    token_count: int
    shingle_count: int
    shared_shingles: int
    unique_query_shingles: int
    unique_candidate_shingles: int
    max_unique_shingles: int
    unique_share: float
    classification: str
    url: str | None
    excerpt: str | None

    @property
    def is_safe_near_duplicate(self) -> bool:
        return self.classification == SAFE_NEAR_DUPLICATE


@dataclass(frozen=True)
class MatchSearchResult:
    """Result of an LSH candidate lookup plus exact verification."""

    candidate_count: int
    matches: tuple[NearDuplicateMatch, ...]

    @property
    def safe_matches(self) -> tuple[NearDuplicateMatch, ...]:
        return tuple(
            match for match in self.matches if match.is_safe_near_duplicate
        )

    @property
    def ambiguous_matches(self) -> tuple[NearDuplicateMatch, ...]:
        return tuple(
            match for match in self.matches if not match.is_safe_near_duplicate
        )

    @property
    def best_safe_match(self) -> NearDuplicateMatch | None:
        """Highest-similarity match that is safe to treat as a duplicate.

        A document should only be considered removable when such a match
        exists.  Matches that are merely similar are not grounds for deletion.
        """

        safe = self.safe_matches
        return safe[0] if safe else None


def comparison_tokens(text: str) -> list[str]:
    """Tokenize text for duplicate comparison only.

    Case and punctuation are intentionally ignored here.  This function never
    changes the text stored in the training corpus.
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return _WORD_RE.findall(text.casefold())


@lru_cache(maxsize=200_000)
def _stable_token_hash(token: str) -> int:
    digest = hashlib.blake2b(
        token.encode("utf-8"),
        digest_size=8,
        person=b"LLMtok01",
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def _rolling_shingle_hashes(
    token_hashes: Sequence[int],
    shingle_size: int,
) -> tuple[int, ...]:
    """Return sorted unique deterministic rolling hashes."""

    count = len(token_hashes)
    if count < shingle_size:
        return ()

    highest_power = pow(ROLLING_BASE, shingle_size - 1, 1 << 64)

    value = 0
    for token_hash in token_hashes[:shingle_size]:
        value = ((value * ROLLING_BASE) + token_hash) & UINT64_MASK

    unique: set[int] = {value}

    for index in range(shingle_size, count):
        outgoing = token_hashes[index - shingle_size]
        incoming = token_hashes[index]

        value = (
            value - ((outgoing * highest_power) & UINT64_MASK)
        ) & UINT64_MASK
        value = ((value * ROLLING_BASE) + incoming) & UINT64_MASK
        unique.add(value)

    return tuple(sorted(unique))


def hashed_word_shingles(
    text: str,
    *,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> tuple[int, ...]:
    """Return sorted unique 64-bit hashes of comparison-token shingles."""

    if shingle_size < 1:
        raise ValueError("shingle_size must be >= 1")
    tokens = comparison_tokens(text)
    token_hashes = [_stable_token_hash(token) for token in tokens]
    return _rolling_shingle_hashes(token_hashes, shingle_size)


def partitioned_signature(
    shingle_hashes: Sequence[int],
    *,
    components: int = DEFAULT_SIGNATURE_COMPONENTS,
) -> tuple[int, ...]:
    """Build a deterministic one-permutation partitioned MinHash-style signature.

    ``components`` must be a power of two.  The high bits select a component;
    the minimum remaining low-bit value is retained in that component.

    Empty components receive the maximum representable remainder.  With the
    default 8 components and documents above the 50-token audit threshold,
    empties are uncommon.  They can create extra LSH candidates but cannot
    create false near-duplicate decisions because candidates are always
    verified with exact shingle Jaccard.
    """

    if components < 2 or components & (components - 1):
        raise ValueError("components must be a power of two >= 2")

    bucket_bits = int(math.log2(components))
    remainder_bits = 64 - bucket_bits
    remainder_mask = (1 << remainder_bits) - 1
    empty = remainder_mask

    mins = [empty] * components

    for raw_value in shingle_hashes:
        value = int(raw_value) & UINT64_MASK
        bucket = value >> remainder_bits
        remainder = value & remainder_mask
        if remainder < mins[bucket]:
            mins[bucket] = remainder

    return tuple(mins)


def build_features(
    text: str,
    *,
    config: NearDedupConfig | None = None,
) -> DocumentFeatures:
    """Derive all near-dedup features for one document."""

    cfg = config or NearDedupConfig()
    tokens = comparison_tokens(text)
    token_hashes = [_stable_token_hash(token) for token in tokens]
    shingles = _rolling_shingle_hashes(token_hashes, cfg.shingle_size)
    signature = partitioned_signature(
        shingles,
        components=cfg.signature_components,
    )
    return DocumentFeatures(
        token_count=len(tokens),
        shingle_hashes=shingles,
        signature=signature,
    )


def signature_similarity(
    first: Sequence[int],
    second: Sequence[int],
) -> float:
    """Fraction of equal signature components."""

    if len(first) != len(second):
        raise ValueError("signatures must have equal length")
    if not first:
        return 1.0
    equal = sum(a == b for a, b in zip(first, second))
    return equal / len(first)


def length_ratio(first_tokens: int, second_tokens: int) -> float:
    """Shorter/longer token-count ratio."""

    if first_tokens < 0 or second_tokens < 0:
        raise ValueError("token counts must be >= 0")
    larger = max(first_tokens, second_tokens)
    if larger == 0:
        return 1.0
    return min(first_tokens, second_tokens) / larger


def shingle_overlap(
    first: Sequence[int],
    second: Sequence[int],
) -> ShingleOverlap:
    """Compare two sorted unique integer sequences with one merge walk.

    Returns shared and per-side unique counts, which together give both the
    Jaccard ratio and the absolute amount of distinct content.
    """

    i = 0
    j = 0
    intersection = 0

    while i < len(first) and j < len(second):
        left = first[i]
        right = second[j]
        if left == right:
            intersection += 1
            i += 1
            j += 1
        elif left < right:
            i += 1
        else:
            j += 1

    return ShingleOverlap(
        shared=intersection,
        unique_first=len(first) - intersection,
        unique_second=len(second) - intersection,
    )


def exact_jaccard(
    first: Sequence[int],
    second: Sequence[int],
) -> float:
    """Exact Jaccard similarity of two sorted unique integer sequences."""

    return shingle_overlap(first, second).jaccard


def classify_overlap(
    overlap: ShingleOverlap,
    *,
    max_unique_shingles: int = DEFAULT_MAX_UNIQUE_SHINGLES,
    max_unique_share: float = DEFAULT_MAX_UNIQUE_SHARE,
) -> str:
    """Label a verified overlap by how much unique content it would destroy.

    Both guards must pass.  The absolute count catches "a copyright year
    changed"; the relative share stops a short technical record from being
    deleted because its distinctive payload happens to be only a dozen
    shingles long.
    """

    if max_unique_shingles < 0:
        raise ValueError("max_unique_shingles must be >= 0")
    if not 0.0 <= max_unique_share <= 1.0:
        raise ValueError("max_unique_share must be within [0, 1]")

    if (
        overlap.max_unique <= max_unique_shingles
        and overlap.unique_share <= max_unique_share
    ):
        return SAFE_NEAR_DUPLICATE
    return AMBIGUOUS_OVERLAP


def _pack_uint64(values: Sequence[int]) -> bytes:
    output = array("Q", (int(value) & UINT64_MASK for value in values))
    if sys.byteorder != "little":
        output.byteswap()
    return output.tobytes()


def _unpack_uint64(payload: bytes) -> tuple[int, ...]:
    values = array("Q")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(int(value) for value in values)


def _pack_signature(signature: Sequence[int]) -> bytes:
    return _pack_uint64(signature)


def _unpack_signature(payload: bytes) -> tuple[int, ...]:
    return _unpack_uint64(payload)


def _pack_shingles(shingles: Sequence[int]) -> bytes:
    return zlib.compress(_pack_uint64(shingles), level=1)


def _unpack_shingles(payload: bytes) -> tuple[int, ...]:
    return _unpack_uint64(zlib.decompress(payload))


def _band_keys(
    signature: Sequence[int],
    config: NearDedupConfig,
) -> tuple[bytes, ...]:
    if len(signature) != config.signature_components:
        raise ValueError("signature length does not match config")

    keys: list[bytes] = []
    rows = config.rows_per_band
    for band in range(config.bands):
        start = band * rows
        stop = start + rows
        # Domain-separate the band index so identical row bytes in different
        # bands never share a lookup key.
        payload = bytes([band]) + _pack_uint64(signature[start:stop])
        key = hashlib.blake2b(
            payload,
            digest_size=16,
            person=b"LLMband1",
        ).digest()
        keys.append(key)
    return tuple(keys)


class NearDuplicateIndex:
    """SQLite-backed LSH candidate index with exact Jaccard verification.

    Writes are explicit transactions.  ``close()`` rolls back an active
    transaction rather than committing it, matching the crash-safe corpus
    builder rule that only a completed input shard may become durable.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        config: NearDedupConfig | None = None,
    ) -> None:
        self.path = Path(path)
        self.config = config or NearDedupConfig()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(self.path, isolation_level=None)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.executescript(_SCHEMA)

        self._in_transaction = False
        self._closed = False

    def __enter__(self) -> "NearDuplicateIndex":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.close()
        return False

    @property
    def in_transaction(self) -> bool:
        return self._in_transaction

    def begin(self) -> None:
        self._require_open()
        if self._in_transaction:
            raise RuntimeError("transaction already active")
        self._connection.execute("BEGIN IMMEDIATE")
        self._in_transaction = True

    def commit(self) -> None:
        self._require_open()
        if not self._in_transaction:
            return
        self._connection.execute("COMMIT")
        self._in_transaction = False

    def rollback(self) -> None:
        self._require_open()
        if not self._in_transaction:
            return
        self._connection.execute("ROLLBACK")
        self._in_transaction = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._in_transaction:
                self._connection.execute("ROLLBACK")
                self._in_transaction = False
        finally:
            self._connection.close()
            self._closed = True

    def count(self) -> int:
        self._require_open()
        row = self._connection.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()
        return int(row[0]) if row else 0

    def contains(self, document_id: str) -> bool:
        self._require_open()
        row = self._connection.execute(
            "SELECT 1 FROM documents WHERE document_id = ? LIMIT 1",
            (document_id,),
        ).fetchone()
        return row is not None

    def add_document(
        self,
        document_id: str,
        features: DocumentFeatures,
        *,
        url: str | None = None,
        excerpt: str | None = None,
    ) -> None:
        """Index one document inside an explicit transaction."""

        self._require_transaction()

        if not isinstance(document_id, str) or not document_id:
            raise ValueError("document_id must be a non-empty string")
        if len(features.signature) != self.config.signature_components:
            raise ValueError("feature signature does not match index config")

        signature_blob = _pack_signature(features.signature)
        shingle_blob = _pack_shingles(features.shingle_hashes)

        self._connection.execute(
            """
            INSERT INTO documents (
                document_id, token_count, shingle_count, signature,
                shingles, url, excerpt
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                features.token_count,
                features.shingle_count,
                signature_blob,
                shingle_blob,
                url,
                excerpt,
            ),
        )

        for band_index, key in enumerate(
            _band_keys(features.signature, self.config)
        ):
            self._connection.execute(
                """
                INSERT INTO bands (band_index, band_key, document_id)
                VALUES (?, ?, ?)
                """,
                (band_index, key, document_id),
            )

    def find_near_duplicates(
        self,
        features: DocumentFeatures,
        *,
        exclude_document_id: str | None = None,
    ) -> MatchSearchResult:
        """Find and exactly verify near-duplicate candidates."""

        self._require_open()

        candidate_ids: set[str] = set()
        for band_index, key in enumerate(
            _band_keys(features.signature, self.config)
        ):
            remaining = self.config.max_candidates - len(candidate_ids)
            if remaining <= 0:
                break

            rows = self._connection.execute(
                """
                SELECT document_id
                FROM bands
                WHERE band_index = ? AND band_key = ?
                ORDER BY document_id
                LIMIT ?
                """,
                (band_index, key, remaining),
            ).fetchall()

            for row in rows:
                candidate_id = str(row[0])
                if candidate_id != exclude_document_id:
                    candidate_ids.add(candidate_id)

        verified: list[NearDuplicateMatch] = []

        for candidate_id in sorted(candidate_ids):
            row = self._connection.execute(
                """
                SELECT token_count, shingle_count, shingles, url, excerpt
                FROM documents
                WHERE document_id = ?
                """,
                (candidate_id,),
            ).fetchone()
            if row is None:
                continue

            candidate_token_count = int(row[0])
            candidate_shingle_count = int(row[1])
            ratio = length_ratio(
                features.token_count,
                candidate_token_count,
            )
            if ratio < self.config.min_length_ratio:
                continue

            candidate_shingles = _unpack_shingles(bytes(row[2]))
            overlap = shingle_overlap(
                features.shingle_hashes,
                candidate_shingles,
            )
            similarity = overlap.jaccard
            if similarity < self.config.similarity_threshold:
                continue

            verified.append(
                NearDuplicateMatch(
                    document_id=candidate_id,
                    similarity=similarity,
                    length_ratio=ratio,
                    token_count=candidate_token_count,
                    shingle_count=candidate_shingle_count,
                    shared_shingles=overlap.shared,
                    unique_query_shingles=overlap.unique_first,
                    unique_candidate_shingles=overlap.unique_second,
                    max_unique_shingles=overlap.max_unique,
                    unique_share=overlap.unique_share,
                    classification=classify_overlap(
                        overlap,
                        max_unique_shingles=self.config.max_unique_shingles,
                        max_unique_share=self.config.max_unique_share,
                    ),
                    url=str(row[3]) if row[3] is not None else None,
                    excerpt=str(row[4]) if row[4] is not None else None,
                )
            )

        verified.sort(
            key=lambda match: (
                -match.similarity,
                -match.length_ratio,
                match.document_id,
            )
        )

        return MatchSearchResult(
            candidate_count=len(candidate_ids),
            matches=tuple(verified),
        )

    def analyze_text(
        self,
        text: str,
    ) -> DocumentFeatures | None:
        """Build features, returning ``None`` below the configured token floor."""

        features = build_features(text, config=self.config)
        if features.token_count < self.config.min_tokens:
            return None
        if features.shingle_count == 0:
            return None
        return features

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("near-duplicate index is closed")

    def _require_transaction(self) -> None:
        self._require_open()
        if not self._in_transaction:
            raise RuntimeError(
                "add_document requires an explicit begin()/commit() transaction"
            )
