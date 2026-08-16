"""Deterministic, dependency-free byte-level Byte Pair Encoding (BPE).

This module keeps the original tokenizer semantics while using an incremental
training kernel and a heap-based encoder.

Training compatibility
----------------------
The learned merge sequence is intentionally identical to the original
reference implementation:

* every document is an independent token sequence;
* pair frequency counts every adjacent pair, including overlapping pairs;
* the most frequent pair is selected;
* ties choose the lexicographically smallest ``(left_id, right_id)`` pair;
* all non-overlapping occurrences are merged left-to-right;
* learned token ids are contiguous starting at 256.

The original implementation recomputed all pair counts and rebuilt all
documents after every merge.  That is easy to understand but costs roughly
O(number_of_merges × corpus_tokens).  The optimized kernel maintains:

* a flattened doubly-linked token stream with document boundaries;
* exact incremental pair counts;
* lazy occurrence indexes for each pair;
* a lazy max-priority heap for deterministic pair selection.

Encoding is also incremental: a heap applies the lowest-ranked currently
available merge without repeatedly rescanning the complete input sequence.

The public JSON format and BPE semantics remain unchanged.
"""

from __future__ import annotations

from array import array
from collections import Counter
import heapq
import json
from pathlib import Path
from typing import Iterable, Sequence


BYTE_VOCAB_SIZE = 256
Pair = tuple[int, int]
TokenSequence = list[int]

# Linked-list sentinels.  Node ids themselves are always >= 0.
_END = -1
_REMOVED = -2


class BPETrainer:
    """Train and apply deterministic byte-level BPE merges.

    Parameters
    ----------
    vocab_size:
        Maximum content vocabulary size, including the 256 base-byte tokens.
        Special tokens are managed by :class:`Tokenizer` and are therefore not
        counted here.
    min_pair_frequency:
        Minimum pair count required to learn a merge.
    """

    def __init__(self, vocab_size: int, min_pair_frequency: int = 1):
        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer")
        if vocab_size < BYTE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be at least {BYTE_VOCAB_SIZE} for byte-level BPE"
            )
        if not isinstance(min_pair_frequency, int):
            raise TypeError("min_pair_frequency must be an integer")
        if min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be >= 1")

        self.vocab_size = vocab_size
        self.min_pair_frequency = min_pair_frequency

        # Mapping: (left_token_id, right_token_id) -> merged_token_id.
        # Dict insertion order is the BPE rank.
        self.merges: dict[Pair, int] = {}

        # Mapping used for decoding.
        self.vocab: dict[int, bytes] = {
            token_id: bytes([token_id])
            for token_id in range(BYTE_VOCAB_SIZE)
        }

        # Cached pair -> rank used by the heap-based encoder.
        self._merge_ranks: dict[Pair, int] = {}
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self.merges.clear()
        self.vocab = {
            token_id: bytes([token_id])
            for token_id in range(BYTE_VOCAB_SIZE)
        }
        self._merge_ranks.clear()
        self._trained = False

    def train(self, texts: Iterable[str | bytes]) -> "BPETrainer":
        """Learn BPE merge rules using an incremental exact-count kernel.

        Each item in *texts* remains an independent sequence, so no merge can
        cross a document boundary.  Calling :meth:`train` again resets the
        trainer before learning a new vocabulary.
        """

        self._reset()

        # Flatten all documents into compact arrays while retaining independent
        # boundaries in prev/next.  array('I') / array('i') avoids the very high
        # object overhead of millions of Python integer nodes.
        tokens = array("I")
        prev_nodes = array("i")
        next_nodes = array("i")

        # Current exact pair frequencies.
        pair_counts: dict[Pair, int] = {}

        # Lazy inverted index: each pair stores every left-node id at which the
        # pair has existed.  Stale entries are cheap to validate later and
        # avoid expensive set deletion/update operations.
        occurrences: dict[Pair, array] = {}

        def record_occurrence(pair: Pair, left_node: int) -> None:
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            bucket = occurrences.get(pair)
            if bucket is None:
                bucket = array("I")
                occurrences[pair] = bucket
            bucket.append(left_node)

        for item in texts:
            if isinstance(item, str):
                raw = item.encode("utf-8", errors="strict")
            elif isinstance(item, (bytes, bytearray, memoryview)):
                raw = bytes(item)
            else:
                raise TypeError(
                    "training texts must contain only str or bytes values; "
                    f"got {type(item).__name__}"
                )

            if not raw:
                continue

            # ``prev_nodes`` / ``next_nodes`` use signed 32-bit indexes for
            # memory efficiency.  In-memory corpora larger than ~2.1 billion
            # byte nodes require a streaming/distributed trainer instead.
            if len(tokens) + len(raw) > 2_147_483_647:
                raise ValueError(
                    "in-memory BPE training corpus exceeds the 32-bit node "
                    "index limit; use a sharded/streaming training kernel"
                )

            start = len(tokens)
            length = len(raw)
            end = start + length

            tokens.extend(raw)

            # Preserve document boundaries explicitly.
            for node in range(start, end):
                prev_nodes.append(node - 1 if node > start else _END)
                next_nodes.append(node + 1 if node + 1 < end else _END)

            # Initial adjacent pairs for this document.
            for left_node in range(start, end - 1):
                pair = (tokens[left_node], tokens[left_node + 1])
                record_occurrence(pair, left_node)

        if not tokens:
            self._trained = True
            return self

        # Max-count + lexicographically-smallest tie breaker:
        # (-count, left_id, right_id).
        pair_heap: list[tuple[int, int, int]] = [
            (-count, pair[0], pair[1])
            for pair, count in pair_counts.items()
        ]
        heapq.heapify(pair_heap)

        def pop_best_pair() -> tuple[Pair, int] | None:
            """Pop the best pair, lazily discarding outdated heap records."""

            while pair_heap:
                neg_count, left_id, right_id = heapq.heappop(pair_heap)
                pair = (left_id, right_id)
                current = pair_counts.get(pair, 0)
                if current <= 0:
                    continue
                if current != -neg_count:
                    continue
                return pair, current
            return None

        def remove_edge(left_node: int, dirty: set[Pair]) -> None:
            """Remove the currently-live adjacency starting at *left_node*."""

            if left_node < 0:
                return
            right_node = next_nodes[left_node]
            if right_node < 0:
                return

            pair = (tokens[left_node], tokens[right_node])
            count = pair_counts.get(pair)
            if count is None or count <= 0:
                raise RuntimeError(
                    f"internal BPE pair-count underflow for pair {pair!r}"
                )

            if count == 1:
                del pair_counts[pair]
            else:
                pair_counts[pair] = count - 1
            dirty.add(pair)

        def add_edge(left_node: int, dirty: set[Pair]) -> None:
            """Add the currently-live adjacency starting at *left_node*."""

            if left_node < 0:
                return
            right_node = next_nodes[left_node]
            if right_node < 0:
                return

            pair = (tokens[left_node], tokens[right_node])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

            bucket = occurrences.get(pair)
            if bucket is None:
                bucket = array("I")
                occurrences[pair] = bucket
            bucket.append(left_node)
            dirty.add(pair)

        next_token_id = BYTE_VOCAB_SIZE

        while next_token_id < self.vocab_size:
            best = pop_best_pair()
            if best is None:
                break

            best_pair, best_count = best
            if best_count < self.min_pair_frequency:
                break

            left_token, right_token = best_pair
            candidate_nodes = occurrences.get(best_pair)
            if candidate_nodes is None:
                raise RuntimeError(
                    f"internal BPE occurrence index missing pair {best_pair!r}"
                )

            # Same-token pairs can overlap: [a, a, a] has two counted (a, a)
            # edges, but reference BPE merges only the leftmost non-overlapping
            # occurrence.  Original node ids preserve document/position order,
            # so sorting candidates exactly reproduces left-to-right behavior.
            #
            # Different-token instances of the same pair cannot overlap, so
            # their processing order has no effect and we avoid the sort.
            if left_token == right_token:
                candidates: Iterable[int] = sorted(candidate_nodes)
            else:
                candidates = candidate_nodes

            dirty_pairs: set[Pair] = set()
            merged_occurrences = 0

            for left_node in candidates:
                # Lazy occurrence validation.
                if next_nodes[left_node] < 0:
                    continue

                right_node = next_nodes[left_node]
                if (
                    tokens[left_node] != left_token
                    or tokens[right_node] != right_token
                ):
                    continue

                prev_node = prev_nodes[left_node]
                after_node = next_nodes[right_node]

                # Remove every old edge affected by replacing L,R with NEW.
                if prev_node >= 0:
                    remove_edge(prev_node, dirty_pairs)
                remove_edge(left_node, dirty_pairs)
                if after_node >= 0:
                    remove_edge(right_node, dirty_pairs)

                # Reuse the left node for the merged token and unlink right.
                tokens[left_node] = next_token_id
                next_nodes[left_node] = after_node

                if after_node >= 0:
                    prev_nodes[after_node] = left_node

                next_nodes[right_node] = _REMOVED
                prev_nodes[right_node] = _REMOVED

                # Add the two possible new boundary edges.
                if prev_node >= 0:
                    add_edge(prev_node, dirty_pairs)
                if after_node >= 0:
                    add_edge(left_node, dirty_pairs)

                merged_occurrences += 1

            if merged_occurrences == 0:
                raise RuntimeError(
                    f"internal BPE count/index mismatch for pair {best_pair!r}"
                )

            # The selected old pair cannot survive once all of its
            # non-overlapping instances are replaced.
            remaining_best = pair_counts.get(best_pair, 0)
            if remaining_best:
                raise RuntimeError(
                    "internal BPE merge left selected pair alive: "
                    f"{best_pair!r} count={remaining_best}"
                )

            self.merges[best_pair] = next_token_id
            self.vocab[next_token_id] = (
                self.vocab[left_token] + self.vocab[right_token]
            )

            # We will never need the selected pair's historical occurrence
            # bucket again.  Releasing it reduces peak memory.
            occurrences.pop(best_pair, None)

            # Push one final heap record per changed pair.  Older records remain
            # in the heap and are discarded lazily by pop_best_pair().
            for pair in dirty_pairs:
                count = pair_counts.get(pair, 0)
                if count > 0:
                    heapq.heappush(pair_heap, (-count, pair[0], pair[1]))

            # Lazy invalidation is fast, but long runs can accumulate stale
            # heap records.  Periodically rebuild from exact current counts.
            # Rebuilding does not change selection semantics because heap
            # ordering depends only on (-count, left_id, right_id).
            live_pair_count = len(pair_counts)
            if len(pair_heap) > max(10_000, 4 * max(live_pair_count, 1)):
                pair_heap = [
                    (-count, pair[0], pair[1])
                    for pair, count in pair_counts.items()
                    if count > 0
                ]
                heapq.heapify(pair_heap)

            next_token_id += 1

        self._merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges.keys())
        }
        self._trained = True
        return self

    # ------------------------------------------------------------------
    # Reference helpers kept for tests, teaching, and regression checks
    # ------------------------------------------------------------------

    @staticmethod
    def get_pair_counts(sequences: Sequence[Sequence[int]]) -> Counter[Pair]:
        """Count adjacent token pairs across independent sequences."""

        counts: Counter[Pair] = Counter()
        for sequence in sequences:
            if len(sequence) >= 2:
                counts.update(zip(sequence, sequence[1:]))
        return counts

    @staticmethod
    def merge_pair(
        sequences: Sequence[Sequence[int]],
        pair: Pair,
        new_token_id: int,
    ) -> list[TokenSequence]:
        """Replace every non-overlapping occurrence of *pair* left-to-right."""

        return [
            BPETrainer._merge_one_sequence(sequence, pair, new_token_id)
            for sequence in sequences
        ]

    # ------------------------------------------------------------------
    # Applying learned merges
    # ------------------------------------------------------------------

    def encode_bytes(self, data: bytes) -> list[int]:
        """Encode bytes using learned BPE ranks in O(n log n)-style updates.

        This produces the same token sequence as the reference implementation
        that repeatedly scanned the whole input for the lowest-ranked pair.
        """

        if not data:
            return []

        if not self.merges or len(data) < 2:
            return list(data)

        ranks = self._merge_ranks
        if len(ranks) != len(self.merges):
            # Defensive recovery for callers that construct/modify ``merges``
            # directly instead of using train()/from_dict().
            ranks = {
                pair: rank
                for rank, pair in enumerate(self.merges.keys())
            }
            self._merge_ranks = ranks

        tokens = list(data)
        n = len(tokens)
        prev_nodes = [i - 1 if i > 0 else _END for i in range(n)]
        next_nodes = [i + 1 if i + 1 < n else _END for i in range(n)]

        # Heap order exactly represents "lowest merge rank first"; position is
        # the deterministic left-to-right tie-break for repeated occurrences.
        merge_heap: list[tuple[int, int, int, int]] = []

        def push_edge(left_node: int) -> None:
            if left_node < 0:
                return
            right_node = next_nodes[left_node]
            if right_node < 0:
                return
            pair = (tokens[left_node], tokens[right_node])
            rank = ranks.get(pair)
            if rank is not None:
                heapq.heappush(
                    merge_heap,
                    (rank, left_node, tokens[left_node], tokens[right_node]),
                )

        for left_node in range(n - 1):
            push_edge(left_node)

        while merge_heap:
            rank, left_node, expected_left, expected_right = heapq.heappop(
                merge_heap
            )

            right_node = next_nodes[left_node]
            if right_node < 0:
                continue

            if (
                tokens[left_node] != expected_left
                or tokens[right_node] != expected_right
            ):
                continue

            pair = (expected_left, expected_right)
            current_rank = ranks.get(pair)
            if current_rank != rank:
                continue

            merged_id = self.merges[pair]
            prev_node = prev_nodes[left_node]
            after_node = next_nodes[right_node]

            tokens[left_node] = merged_id
            next_nodes[left_node] = after_node
            if after_node >= 0:
                prev_nodes[after_node] = left_node

            next_nodes[right_node] = _REMOVED
            prev_nodes[right_node] = _REMOVED

            # Only the two boundary pairs can have changed.
            if prev_node >= 0:
                push_edge(prev_node)
            if after_node >= 0:
                push_edge(left_node)

        output: TokenSequence = []
        node = 0
        while node >= 0:
            output.append(tokens[node])
            node = next_nodes[node]

        return output

    @staticmethod
    def _merge_one_sequence(
        sequence: Sequence[int],
        pair: Pair,
        new_token_id: int,
    ) -> TokenSequence:
        """Reference left-to-right non-overlapping merge for one sequence."""

        left, right = pair
        output: TokenSequence = []
        i = 0
        n = len(sequence)

        while i < n:
            if (
                i + 1 < n
                and sequence[i] == left
                and sequence[i + 1] == right
            ):
                output.append(new_token_id)
                i += 2
            else:
                output.append(sequence[i])
                i += 1

        return output

    def decode_ids(self, token_ids: Iterable[int]) -> bytes:
        """Decode non-special BPE token ids back to raw bytes."""

        output = bytearray()

        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("token ids must be integers")
            try:
                output.extend(self.vocab[token_id])
            except KeyError as exc:
                raise ValueError(f"unknown BPE token id: {token_id}") from exc

        return bytes(output)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the trained BPE."""

        return {
            "format": "byte_level_bpe",
            "format_version": 1,
            "vocab_size": self.vocab_size,
            "min_pair_frequency": self.min_pair_frequency,
            "merges": [
                [left, right, new_token_id]
                for (left, right), new_token_id in self.merges.items()
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BPETrainer":
        """Reconstruct a BPE trainer/codec from serialized data."""

        if not isinstance(payload, dict):
            raise TypeError("BPE payload must be a dictionary")

        if payload.get("format") != "byte_level_bpe":
            raise ValueError("unsupported or missing BPE format")

        trainer = cls(
            vocab_size=int(payload["vocab_size"]),
            min_pair_frequency=int(payload.get("min_pair_frequency", 1)),
        )

        merges = payload.get("merges", [])
        next_expected_id = BYTE_VOCAB_SIZE

        for item in merges:
            if not (isinstance(item, list) and len(item) == 3):
                raise ValueError(f"invalid BPE merge record: {item!r}")

            left, right, new_token_id = map(int, item)
            if new_token_id != next_expected_id:
                raise ValueError(
                    "BPE merge ids must be contiguous starting at 256; "
                    f"expected {next_expected_id}, got {new_token_id}"
                )
            if left not in trainer.vocab or right not in trainer.vocab:
                raise ValueError(
                    f"merge references unknown parent token(s): {(left, right)!r}"
                )
            if new_token_id >= trainer.vocab_size:
                raise ValueError(
                    f"merge token id {new_token_id} exceeds BPE vocab size "
                    f"{trainer.vocab_size}"
                )

            pair = (left, right)
            if pair in trainer.merges:
                raise ValueError(f"duplicate BPE merge pair: {pair!r}")

            trainer.merges[pair] = new_token_id
            trainer.vocab[new_token_id] = (
                trainer.vocab[left] + trainer.vocab[right]
            )
            next_expected_id += 1

        trainer._merge_ranks = {
            pair: rank
            for rank, pair in enumerate(trainer.merges.keys())
        }
        trainer._trained = True
        return trainer

    def save(self, path: str | Path) -> None:
        """Save BPE vocabulary metadata and merge rules as UTF-8 JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETrainer":
        """Load BPE merge rules previously written by :meth:`save`."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def learned_vocab_size(self) -> int:
        """Number of currently usable byte/BPE token ids."""

        return BYTE_VOCAB_SIZE + len(self.merges)

    @property
    def trained(self) -> bool:
        return self._trained