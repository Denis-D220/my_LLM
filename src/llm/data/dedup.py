"""Persistent exact deduplication for the pretraining corpus.

Why this exists
---------------
The Common Crawl extractor deduplicates within a single process run.  Its
in-memory set is discarded when the process exits, so documents duplicated
*across* extraction sessions survive into the shards on disk.  This module
provides deduplication that is global across:

* all input shards,
* all extractor runs,
* all corpus-build runs, including interrupted ones.

Storage
-------
Fingerprints live in SQLite rather than a Python ``set`` because a 5 GB corpus
can contain millions of documents, and because the set must survive process
restarts.  The schema is deliberately minimal::

    CREATE TABLE exact_hashes (
        sha256 BLOB PRIMARY KEY
    ) WITHOUT ROWID;

``WITHOUT ROWID`` stores the key directly in the B-tree, which for a
pure-key table avoids a second index and roughly halves the storage.

Identity
--------
A document's fingerprint is the SHA-256 of its **normalized** UTF-8 text::

    sha256 = hashlib.sha256(normalized_text.encode("utf-8")).digest()

Normalization must happen before fingerprinting, otherwise two documents
differing only in line endings would be treated as distinct.

Restart safety
--------------
There are two modes, and the difference matters.

**Batched mode** (``transactional=False``) commits every ``commit_interval``
inserts.  It is fine for standalone use, but it is *unsafe* for a corpus build:
a crash can leave fingerprints committed for documents whose output shard was
never finished.  On the next run those documents are rejected as duplicates and
are silently lost.

**Transactional mode** (``transactional=True``) never commits on its own.  The
caller wraps a unit of work - for the corpus builder, one Common Crawl input
shard - and either commits it whole or rolls it back whole::

    dedup.begin()
    for document in shard:
        writer.write(document)
        dedup.add_fingerprint(digest)
    writer.rotate()
    dedup.commit()      # fingerprints and output shards advance together

On failure the caller calls :meth:`rollback`, and the fingerprints for that
shard vanish so the shard can be reprocessed cleanly.

:meth:`close` deliberately behaves differently in the two modes.  In batched
mode it commits pending work.  In transactional mode it **rolls back**, because
an uncommitted transaction at close time means the unit of work did not finish,
and committing it would reintroduce exactly the data-loss bug this mode exists
to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3
from types import TracebackType


DEFAULT_DATABASE_NAME = "dedup.sqlite"
DEFAULT_COMMIT_INTERVAL = 10_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS exact_hashes (
    sha256 BLOB PRIMARY KEY
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def text_fingerprint(normalized_text: str) -> bytes:
    """Return the 32-byte SHA-256 digest of normalized document text."""

    if not isinstance(normalized_text, str):
        raise TypeError(
            f"normalized_text must be str, got {type(normalized_text).__name__}"
        )
    return hashlib.sha256(normalized_text.encode("utf-8")).digest()


def fingerprint_hex(normalized_text: str) -> str:
    """Return the SHA-256 digest as lowercase hex, for manifests and reports."""

    return text_fingerprint(normalized_text).hex()


@dataclass
class DedupStats:
    """Counters describing one deduplication session."""

    checked: int = 0
    unique: int = 0
    duplicates: int = 0

    @property
    def duplicate_ratio(self) -> float:
        return self.duplicates / self.checked if self.checked else 0.0


class ExactDeduplicator:
    """SQLite-backed exact-duplicate filter.

    Typical use::

        with ExactDeduplicator(path) as dedup:
            for text in documents:
                if dedup.seen(text):
                    continue
                write(text)
                dedup.add(text)
            dedup.commit()

    The class is not thread-safe; SQLite connections are bound to the creating
    thread by default and no locking is layered on top.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        commit_interval: int = DEFAULT_COMMIT_INTERVAL,
        read_only: bool = False,
        transactional: bool = False,
    ) -> None:
        if commit_interval < 1:
            raise ValueError("commit_interval must be >= 1")

        self.path = Path(path)
        self.commit_interval = commit_interval
        self.read_only = read_only
        self.transactional = transactional
        self.stats = DedupStats()

        self._pending = 0
        self._closed = False

        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        # NORMAL is durable across process crashes (only a power loss can lose
        # the tail), and is far faster than FULL for millions of inserts.
        self._connection.execute("PRAGMA synchronous=NORMAL")

        if not read_only:
            self._connection.executescript(_SCHEMA)
            self._connection.commit()

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "ExactDeduplicator":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.close()
        return False

    # -- core API ----------------------------------------------------------

    def seen(self, normalized_text: str) -> bool:
        """Return whether this exact normalized text was already recorded."""

        return self.seen_fingerprint(text_fingerprint(normalized_text))

    def seen_fingerprint(self, digest: bytes) -> bool:
        """Return whether ``digest`` is already recorded."""

        self._require_open()
        row = self._connection.execute(
            "SELECT 1 FROM exact_hashes WHERE sha256 = ? LIMIT 1",
            (digest,),
        ).fetchone()
        return row is not None

    def add(self, normalized_text: str) -> bytes:
        """Record text as seen and return its fingerprint."""

        digest = text_fingerprint(normalized_text)
        self.add_fingerprint(digest)
        return digest

    def add_fingerprint(self, digest: bytes) -> None:
        """Record ``digest`` as seen."""

        self._require_writable()
        self._connection.execute(
            "INSERT OR IGNORE INTO exact_hashes (sha256) VALUES (?)",
            (digest,),
        )
        self._pending += 1

        # In transactional mode the caller owns the commit boundary. Committing
        # here would break the "one input shard, one checkpoint" guarantee.
        if not self.transactional and self._pending >= self.commit_interval:
            self.commit()

    def check_and_add(self, normalized_text: str) -> bool:
        """Return ``True`` if the document is new, recording it in that case.

        This is the common path and performs a single lookup followed by an
        insert only when needed.  Statistics are updated for every call, so the
        caller does not have to maintain its own counters.
        """

        self._require_writable()
        digest = text_fingerprint(normalized_text)
        self.stats.checked += 1

        if self.seen_fingerprint(digest):
            self.stats.duplicates += 1
            return False

        self.add_fingerprint(digest)
        self.stats.unique += 1
        return True

    # -- transactions ------------------------------------------------------

    @property
    def pending(self) -> int:
        """Number of inserts not yet committed."""

        return self._pending

    def begin(self) -> None:
        """Mark the start of a unit of work.

        ``sqlite3`` opens transactions implicitly on the first write, so this
        exists to make the caller's intent explicit and to assert that no
        uncommitted work is being silently absorbed into a new unit.
        """

        self._require_writable()
        if self._pending:
            raise RuntimeError(
                "cannot begin a new transaction with "
                f"{self._pending} uncommitted inserts; commit or roll back first"
            )

    def commit(self) -> None:
        """Flush pending inserts to disk."""

        if self._closed or self.read_only:
            return
        self._connection.commit()
        self._pending = 0

    def rollback(self) -> None:
        """Discard every insert since the last commit."""

        if self._closed or self.read_only:
            return
        self._connection.rollback()
        self._pending = 0

    # -- maintenance -------------------------------------------------------

    def count(self) -> int:
        """Return the number of stored fingerprints."""

        self._require_open()
        row = self._connection.execute(
            "SELECT COUNT(*) FROM exact_hashes"
        ).fetchone()
        return int(row[0]) if row else 0

    def set_metadata(self, key: str, value: str) -> None:
        """Store a small provenance value alongside the fingerprints."""

        self._require_writable()
        self._connection.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._pending += 1

    def get_metadata(self, key: str) -> str | None:
        self._require_open()
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return str(row[0]) if row else None

    def close(self) -> None:
        """Close the database.

        In transactional mode any uncommitted work is **rolled back**, not
        committed.  Reaching close with pending inserts means the caller's unit
        of work did not finish, and committing it would record fingerprints for
        documents whose output was never completed - the exact failure this
        mode exists to prevent.
        """

        if self._closed:
            return
        try:
            if self.transactional and self._pending:
                self.rollback()
            else:
                self.commit()
        finally:
            self._connection.close()
            self._closed = True

    # -- internals ---------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("deduplicator is closed")

    def _require_writable(self) -> None:
        self._require_open()
        if self.read_only:
            raise RuntimeError("deduplicator was opened read-only")
