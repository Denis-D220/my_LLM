"""Conservative secondary language check based on Unicode script.

Why this exists
---------------
The Common Crawl extractor's ``lang`` field is metadata, and it is wrong often
enough to matter.  A 10k audit of accepted documents surfaced a Russian-language
page labelled ``eng`` that passed the primary language filter; the 100k audit
surfaced Cyrillic and Korean pages in the accepted corpus.  Those documents are
not English by any reading, and they teach a byte-level BPE tokenizer script
statistics it should not be spending vocabulary on.

This module is the second gate.  It ignores the metadata entirely and looks at
what script the text is actually written in.

Deliberately narrow scope
-------------------------
This is **not** language identification.  It answers one question: is this
document overwhelmingly written in a non-Latin script?  It therefore says
nothing about Spanish, French, Vietnamese, or Turkish, which use Latin script
and are the primary language filter's responsibility.

The default rule rejects only when all three hold:

* at least 100 alphabetic characters (short quotations are never judged), and
* Latin share below 0.50 (the document is not mostly Latin), and
* a single non-Latin family accounts for at least 0.70 of alphabetic characters

Requiring all three keeps the filter conservative.  A page of English prose
containing a Greek formula block, a Cyrillic pull-quote, or a CJK example stays
accepted, because Latin still dominates.

Only alphabetic characters count
--------------------------------
Digits, punctuation, mathematical operators, currency symbols, whitespace,
emoji, and programming syntax are ignored entirely.  A C source file is almost
entirely non-alphabetic once braces and operators are removed, and what remains
is Latin identifiers, so code is unaffected.

East Asian scripts are grouped
------------------------------
Han, Hiragana, Katakana, and Hangul are counted as one ``east_asian`` family.
Japanese mixes Han and Hiragana freely, so scoring them separately would let a
Japanese page fall below the dominance threshold on both and evade the gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


# Sentinel distinguishing "not cached" from a cached ``None`` classification.
_MISSING = object()


REASON_LANGUAGE_SCRIPT_MISMATCH = "language_script_mismatch"

LATIN = "latin"
CYRILLIC = "cyrillic"
GREEK = "greek"
EAST_ASIAN = "east_asian"
ARABIC = "arabic"
HEBREW = "hebrew"
DEVANAGARI = "devanagari"
THAI = "thai"
OTHER = "other"


# Code-point ranges per script family, inclusive.  Ranges are checked only for
# characters that are already known to be alphabetic, so symbols that share a
# block with letters (for example U+00D7 MULTIPLICATION SIGN inside Latin-1
# Supplement) never reach this table.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0041, 0x005A, LATIN),
    (0x0061, 0x007A, LATIN),
    (0x00C0, 0x024F, LATIN),
    (0x1E00, 0x1EFF, LATIN),
    (0x2C60, 0x2C7F, LATIN),
    (0xA720, 0xA7FF, LATIN),
    (0x0370, 0x03FF, GREEK),
    (0x1F00, 0x1FFF, GREEK),
    (0x0400, 0x052F, CYRILLIC),
    (0x2DE0, 0x2DFF, CYRILLIC),
    (0xA640, 0xA69F, CYRILLIC),
    (0x0590, 0x05FF, HEBREW),
    (0x0600, 0x06FF, ARABIC),
    (0x0750, 0x077F, ARABIC),
    (0x08A0, 0x08FF, ARABIC),
    (0xFB50, 0xFDFF, ARABIC),
    (0xFE70, 0xFEFF, ARABIC),
    (0x0900, 0x097F, DEVANAGARI),
    (0x0E00, 0x0E7F, THAI),
    (0x1100, 0x11FF, EAST_ASIAN),   # Hangul Jamo
    (0x3040, 0x309F, EAST_ASIAN),   # Hiragana
    (0x30A0, 0x30FF, EAST_ASIAN),   # Katakana
    (0x3130, 0x318F, EAST_ASIAN),   # Hangul Compatibility Jamo
    (0x3400, 0x4DBF, EAST_ASIAN),   # CJK Extension A
    (0x4E00, 0x9FFF, EAST_ASIAN),   # CJK Unified Ideographs
    (0xA960, 0xA97F, EAST_ASIAN),   # Hangul Jamo Extended-A
    (0xAC00, 0xD7AF, EAST_ASIAN),   # Hangul Syllables
    (0xD7B0, 0xD7FF, EAST_ASIAN),   # Hangul Jamo Extended-B
    (0xF900, 0xFAFF, EAST_ASIAN),   # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF, EAST_ASIAN),  # CJK Extension B
    (0x2A700, 0x2EBEF, EAST_ASIAN),  # CJK Extensions C-F
)


# Bytes that are not ASCII letters, for the pure-ASCII fast path.  Deleting
# them with bytes.translate is a single pass in C with no per-character Python
# objects allocated.
_DELETE_ASCII_NON_LETTERS = bytes(
    code
    for code in range(256)
    if not (0x41 <= code <= 0x5A or 0x61 <= code <= 0x7A)
)

# Cache of code point -> script family.  Documents reuse the same few hundred
# characters, so classification cost is paid once per distinct character per
# process rather than once per occurrence.
_SCRIPT_CACHE: dict[int, str | None] = {}


@dataclass(frozen=True)
class ScriptSanityThresholds:
    """Tunable policy for the script sanity gate."""

    min_alphabetic_characters: int = 100
    min_latin_share: float = 0.50
    min_dominant_non_latin_share: float = 0.70

    def __post_init__(self) -> None:
        if self.min_alphabetic_characters < 0:
            raise ValueError("min_alphabetic_characters must be >= 0")
        if not 0.0 <= self.min_latin_share <= 1.0:
            raise ValueError("min_latin_share must be within [0, 1]")
        if not 0.0 <= self.min_dominant_non_latin_share <= 1.0:
            raise ValueError(
                "min_dominant_non_latin_share must be within [0, 1]"
            )


@dataclass(frozen=True)
class ScriptMetrics:
    """Script composition of one document, counting alphabetic characters only."""

    alphabetic_characters: int
    latin_characters: int
    script_counts: dict[str, int] = field(default_factory=dict)
    latin_share: float = 0.0
    dominant_non_latin_script: str | None = None
    dominant_non_latin_share: float = 0.0


@dataclass(frozen=True)
class ScriptVerdict:
    """Outcome of the script sanity check."""

    accepted: bool
    reason: str | None
    metrics: ScriptMetrics


def classify_alphabetic_script(character: str) -> str:
    """Return the script family of a single alphabetic character.

    Raises
    ------
    ValueError
        If *character* is not exactly one alphabetic character.  Digits,
        punctuation, and symbols have no script family in this model.
    """

    if not isinstance(character, str):
        raise TypeError(
            f"character must be str, got {type(character).__name__}"
        )
    if len(character) != 1:
        raise ValueError("expected exactly one character")
    if not character.isalpha():
        raise ValueError(f"character is not alphabetic: {character!r}")

    script = _script_for_code_point(ord(character))
    # isalpha() was already checked above, so this cannot be None.
    return script or OTHER


def _script_for_code_point(code_point: int) -> str | None:
    """Script family for a code point, or ``None`` if it is not alphabetic.

    Results are memoised because a corpus reuses the same characters endlessly.
    """

    cached = _SCRIPT_CACHE.get(code_point, _MISSING)
    if cached is not _MISSING:
        return cached  # type: ignore[return-value]

    if not chr(code_point).isalpha():
        # Combining marks, digits, punctuation, and symbols have no script
        # family here. Checking isalpha() first is what keeps script counts
        # consistent with the alphabetic denominator: Arabic and Devanagari
        # diacritics sit inside their script's code-point range but are not
        # alphabetic, and counting them would push a share above 1.0.
        _SCRIPT_CACHE[code_point] = None
        return None

    script = OTHER
    for start, end, name in _SCRIPT_RANGES:
        if start <= code_point <= end:
            script = name
            break

    _SCRIPT_CACHE[code_point] = script
    return script


def compute_script_metrics(text: str) -> ScriptMetrics:
    """Measure the script composition of *text* without judging it."""

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    # Overwhelmingly the common case: pure ASCII English. Counting letters with
    # bytes.translate is a single C pass, and every letter is Latin by
    # construction, so no classification is needed at all.
    if text.isascii():
        alphabetic = len(
            text.encode("ascii").translate(None, _DELETE_ASCII_NON_LETTERS)
        )
        if alphabetic == 0:
            return ScriptMetrics(alphabetic_characters=0, latin_characters=0)
        return ScriptMetrics(
            alphabetic_characters=alphabetic,
            latin_characters=alphabetic,
            script_counts={LATIN: alphabetic},
            latin_share=1.0,
            dominant_non_latin_script=None,
            dominant_non_latin_share=0.0,
        )

    # Non-ASCII: tally distinct characters once, then classify each distinct
    # character rather than each occurrence. A document contains thousands of
    # characters but only a few hundred distinct ones.
    counts: dict[str, int] = {}
    alphabetic = 0

    for character, occurrences in Counter(text).items():
        script = _script_for_code_point(ord(character))
        if script is None:
            continue
        alphabetic += occurrences
        counts[script] = counts.get(script, 0) + occurrences

    if alphabetic == 0:
        return ScriptMetrics(alphabetic_characters=0, latin_characters=0)

    latin = counts.get(LATIN, 0)

    non_latin = {
        script: count
        for script, count in counts.items()
        if script not in (LATIN, OTHER)
    }
    if non_latin:
        dominant_script = max(non_latin, key=lambda name: non_latin[name])
        dominant_share = non_latin[dominant_script] / alphabetic
    else:
        dominant_script = None
        dominant_share = 0.0

    return ScriptMetrics(
        alphabetic_characters=alphabetic,
        latin_characters=latin,
        script_counts=counts,
        latin_share=latin / alphabetic,
        dominant_non_latin_script=dominant_script,
        dominant_non_latin_share=dominant_share,
    )


def assess_english_script(
    text: str,
    *,
    thresholds: ScriptSanityThresholds | None = None,
) -> ScriptVerdict:
    """Reject documents overwhelmingly written in a non-Latin script.

    All three conditions must hold for a rejection.  Any one of them alone is
    not evidence enough, which is what keeps English pages containing foreign
    quotations, formula symbols, or CJK examples in the corpus.
    """

    policy = thresholds or ScriptSanityThresholds()
    metrics = compute_script_metrics(text)

    long_enough = (
        metrics.alphabetic_characters >= policy.min_alphabetic_characters
    )
    not_mostly_latin = metrics.latin_share < policy.min_latin_share
    dominated = (
        metrics.dominant_non_latin_share
        >= policy.min_dominant_non_latin_share
    )

    if long_enough and not_mostly_latin and dominated:
        return ScriptVerdict(False, REASON_LANGUAGE_SCRIPT_MISMATCH, metrics)

    return ScriptVerdict(True, None, metrics)
