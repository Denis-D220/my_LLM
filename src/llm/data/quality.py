"""Document-quality measurement and filtering for web pretraining text.

This module decides whether a normalized Common Crawl document deserves to
become training data.  It is deliberately **conservative**: the cost of
discarding a good technical document is higher than the cost of admitting a
mediocre one, because the former silently removes capability the model could
have learned.

Design policy
-------------
* Measurement and judgement are separated.  :func:`compute_metrics` never
  rejects anything; :func:`assess_document` applies thresholds to metrics.
  This keeps thresholds tunable without re-deriving statistics, and makes the
  build report able to explain *why* a document was rejected.
* Documents are accepted or rejected whole.  Text is never rewritten here.
  Repeatedly editing web text destroys provenance: you end up unable to say
  what the training document actually was.
* High punctuation or high digit density is **not** evidence of low quality.
  Source code, formulas, datasheets, and tables are exactly the technical
  material this corpus exists to capture::

      if (status != HAL_OK) {
          return ERROR_I2C_TIMEOUT;
      }

      Vout = Vin × R2 / (R1 + R2)
      R = 4.7 kΩ ± 5%

  Rejection therefore keys on structural noise signals - navigation dumps,
  massive line repetition, keyword walls, decoding damage - rather than on
  "this does not look like prose".

Secret filtering
----------------
:func:`find_secret` detects **high-confidence** credential formats only.  It
must not fire on ordinary technical documentation such as::

    user@example.com
    192.168.1.100
    http://localhost:8080
    API_KEY="example"

Broad regex deletion of anything resembling PII would remove a large fraction
of legitimate technical writing, so this stage targets structured secrets with
distinctive prefixes and lengths.

Inputs to this module are expected to be **already normalized** by
``llm.tokenizer.normalizer.normalize_text`` (NFC, LF line endings).  Metrics
assume LF-only text.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


# --------------------------------------------------------------------------
# Rejection reasons
# --------------------------------------------------------------------------
# Stable identifiers.  These strings appear in build reports and tests, so
# treat them as part of the module's public contract.

REASON_EMPTY = "empty_text"
REASON_TOO_SHORT = "too_short"
REASON_TOO_FEW_WORDS = "too_few_words"
REASON_LOW_ALPHA = "low_alphabetic_ratio"
REASON_CONTROL_CHARS = "control_characters"
REASON_REPLACEMENT_CHARS = "replacement_characters"
REASON_DUPLICATE_LINES = "duplicate_lines"
REASON_NAVIGATION = "navigation_boilerplate"
REASON_URL_DIRECTORY = "url_directory"
REASON_KEYWORD_WALL = "keyword_wall"
REASON_SECRET = "secret_material"


# A "short line" is a structural property of the text, not a tunable policy,
# so it lives with the measurement code rather than in QualityThresholds.
SHORT_LINE_CHARS = 30

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_URL_LINE_RE = re.compile(
    r"^\s*(?:https?://|www\.)\S+\s*$",
    re.IGNORECASE,
)

# Words typical of site chrome.  Used only in combination with structural
# signals (many very short lines), never on their own.
_NAV_WORDS = frozenset(
    {
        "home", "login", "register", "sign in", "sign up", "contact",
        "about", "about us", "privacy", "privacy policy", "terms",
        "terms of use", "cookie", "cookies", "accept cookies", "next",
        "previous", "prev", "back", "menu", "search", "subscribe",
        "share", "tweet", "read more", "continue reading", "click here",
        "categories", "archives", "tags", "comments", "reply", "newsletter",
        "follow us", "sitemap", "faq", "help", "support", "download",
        "add to cart", "checkout", "my account", "skip to content",
    }
)


@dataclass(frozen=True)
class DocumentMetrics:
    """Structural measurements of one normalized document."""

    characters: int
    utf8_bytes: int
    lines: int
    non_empty_lines: int
    words: int
    unique_words: int
    alphabetic_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    control_ratio: float
    replacement_chars: int
    mean_line_length: float
    duplicate_line_ratio: float
    short_line_ratio: float
    url_line_ratio: float
    type_token_ratio: float
    nav_line_ratio: float


@dataclass(frozen=True)
class QualityThresholds:
    """Tunable acceptance policy.

    Defaults are intentionally permissive.  Every threshold below was chosen so
    that a legitimate technical page passes; tighten them only with evidence
    from a real sample audit.
    """

    min_characters: int = 300
    min_words: int = 50
    min_alphabetic_ratio: float = 0.40
    max_control_ratio: float = 0.005
    max_replacement_chars: int = 5

    # Line repetition. Requires enough lines for the ratio to mean anything.
    #
    # The minimum was lowered from 10 to 4 after a 10k-document audit: HTTP
    # error pages such as "The server is temporarily unable to service your
    # request..." repeat 9 times and were being accepted because they sat just
    # under the old gate.  Short repeated pages are exactly the case worth
    # catching, and legitimate documents that repeat a heading twice stay well
    # below max_duplicate_line_ratio.
    max_duplicate_line_ratio: float = 0.50
    duplicate_line_min_lines: int = 4

    # Navigation dumps: many very short lines, mostly site chrome.
    nav_min_lines: int = 15
    nav_max_mean_line_length: float = 30.0
    nav_min_short_line_ratio: float = 0.80
    nav_min_nav_line_ratio: float = 0.25

    max_url_line_ratio: float = 0.50

    # Keyword walls: many words, almost no vocabulary.
    #
    # Lowered from 0.12 to 0.10 after the same audit.  Legitimate patent and
    # stock-footage catalogue prose measured 0.122 and 0.123, leaving almost no
    # margin above the old value.  The junk this rule was catching at 0.106 is
    # repeated-line boilerplate, which duplicate_line_ratio now rejects first,
    # so the safer threshold costs nothing.
    keyword_wall_min_words: int = 200
    keyword_wall_max_type_token_ratio: float = 0.10

    def __post_init__(self) -> None:
        if self.min_characters < 0:
            raise ValueError("min_characters must be >= 0")
        if self.min_words < 0:
            raise ValueError("min_words must be >= 0")
        if not 0.0 <= self.min_alphabetic_ratio <= 1.0:
            raise ValueError("min_alphabetic_ratio must be within [0, 1]")
        if not 0.0 <= self.max_control_ratio <= 1.0:
            raise ValueError("max_control_ratio must be within [0, 1]")
        if not 0.0 <= self.max_duplicate_line_ratio <= 1.0:
            raise ValueError("max_duplicate_line_ratio must be within [0, 1]")
        if not 0.0 <= self.max_url_line_ratio <= 1.0:
            raise ValueError("max_url_line_ratio must be within [0, 1]")
        if not 0.0 <= self.keyword_wall_max_type_token_ratio <= 1.0:
            raise ValueError(
                "keyword_wall_max_type_token_ratio must be within [0, 1]"
            )


@dataclass(frozen=True)
class QualityVerdict:
    """Outcome of assessing one document."""

    accepted: bool
    reason: str | None
    metrics: DocumentMetrics


# --------------------------------------------------------------------------
# High-confidence secret patterns
# --------------------------------------------------------------------------
# Each entry is (name, compiled pattern).  Patterns must be structurally
# distinctive: a fixed prefix plus a length constraint, or an explicit armored
# key header.  Anything looser produces false positives on documentation.

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"
        ),
    ),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe_secret_key", re.compile(r"\b[sr]k_live_[0-9A-Za-z]{24,}\b")),
    ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    (
        "json_web_token",
        re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\."),
    ),
    (
        "bearer_high_entropy",
        re.compile(
            r"(?i:authorization)\s*:\s*(?i:bearer)\s+[A-Za-z0-9_\-\.=+/]{40,}"
        ),
    ),
)


def find_secret(text: str) -> str | None:
    """Return the name of a high-confidence secret pattern, or ``None``.

    Only structurally distinctive credentials are reported.  Ordinary
    documentation values such as ``API_KEY="example"``, private IP addresses,
    ``localhost`` URLs, and example email addresses are not secrets and must
    not be reported here.
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    for name, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            return name

    return None


def compute_metrics(text: str) -> DocumentMetrics:
    """Measure one normalized document without judging it.

    ``text`` is expected to use LF line endings.  No rejection happens here;
    this function is pure measurement so that thresholds can change without
    altering how documents are described.
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    characters = len(text)
    utf8_bytes = len(text.encode("utf-8"))

    raw_lines = text.split("\n")
    lines = len(raw_lines)

    stripped_lines = [line.strip() for line in raw_lines]
    non_empty = [line for line in stripped_lines if line]
    non_empty_count = len(non_empty)

    alphabetic = 0
    digits = 0
    punctuation = 0
    control = 0
    replacement = 0
    non_space = 0

    for char in text:
        if char == "�":
            replacement += 1

        if char.isspace():
            continue

        non_space += 1

        if char.isalpha():
            alphabetic += 1
        elif char.isdigit():
            digits += 1
        else:
            category = unicodedata.category(char)
            if category.startswith("P") or category.startswith("S"):
                punctuation += 1
            elif category in ("Cc", "Cf"):
                control += 1

    words = _WORD_RE.findall(text)
    word_count = len(words)
    unique_words = len({word.casefold() for word in words})

    if non_empty_count:
        seen: set[str] = set()
        duplicates = 0
        short_lines = 0
        url_lines = 0
        nav_lines = 0
        total_length = 0

        for line in non_empty:
            total_length += len(line)

            if line in seen:
                duplicates += 1
            else:
                seen.add(line)

            if len(line) < SHORT_LINE_CHARS:
                short_lines += 1

            if _URL_LINE_RE.match(line):
                url_lines += 1

            if line.casefold().strip(" \t:;.|-–—>»") in _NAV_WORDS:
                nav_lines += 1

        duplicate_line_ratio = duplicates / non_empty_count
        short_line_ratio = short_lines / non_empty_count
        url_line_ratio = url_lines / non_empty_count
        nav_line_ratio = nav_lines / non_empty_count
        mean_line_length = total_length / non_empty_count
    else:
        duplicate_line_ratio = 0.0
        short_line_ratio = 0.0
        url_line_ratio = 0.0
        nav_line_ratio = 0.0
        mean_line_length = 0.0

    denominator = non_space or 1

    return DocumentMetrics(
        characters=characters,
        utf8_bytes=utf8_bytes,
        lines=lines,
        non_empty_lines=non_empty_count,
        words=word_count,
        unique_words=unique_words,
        alphabetic_ratio=alphabetic / denominator,
        digit_ratio=digits / denominator,
        punctuation_ratio=punctuation / denominator,
        control_ratio=control / denominator,
        replacement_chars=replacement,
        mean_line_length=mean_line_length,
        duplicate_line_ratio=duplicate_line_ratio,
        short_line_ratio=short_line_ratio,
        url_line_ratio=url_line_ratio,
        type_token_ratio=(unique_words / word_count) if word_count else 0.0,
        nav_line_ratio=nav_line_ratio,
    )


def assess_document(
    text: str,
    *,
    thresholds: QualityThresholds | None = None,
    check_secrets: bool = True,
) -> QualityVerdict:
    """Decide whether one normalized document should enter the corpus.

    Checks run cheapest-first and stop at the first failure, so ``reason``
    identifies the first violated rule rather than every violated rule.
    """

    policy = thresholds or QualityThresholds()
    metrics = compute_metrics(text)

    if not text.strip():
        return QualityVerdict(False, REASON_EMPTY, metrics)

    if metrics.characters < policy.min_characters:
        return QualityVerdict(False, REASON_TOO_SHORT, metrics)

    if metrics.words < policy.min_words:
        return QualityVerdict(False, REASON_TOO_FEW_WORDS, metrics)

    # Decoding damage.  Genuine technical text does not carry U+FFFD.
    if metrics.replacement_chars > policy.max_replacement_chars:
        return QualityVerdict(False, REASON_REPLACEMENT_CHARS, metrics)

    if metrics.control_ratio > policy.max_control_ratio:
        return QualityVerdict(False, REASON_CONTROL_CHARS, metrics)

    # Deliberately low: code and formulas legitimately depress this ratio.
    if metrics.alphabetic_ratio < policy.min_alphabetic_ratio:
        return QualityVerdict(False, REASON_LOW_ALPHA, metrics)

    if (
        metrics.non_empty_lines >= policy.duplicate_line_min_lines
        and metrics.duplicate_line_ratio > policy.max_duplicate_line_ratio
    ):
        return QualityVerdict(False, REASON_DUPLICATE_LINES, metrics)

    if (
        metrics.non_empty_lines >= policy.nav_min_lines
        and metrics.mean_line_length < policy.nav_max_mean_line_length
        and metrics.short_line_ratio >= policy.nav_min_short_line_ratio
        and metrics.nav_line_ratio >= policy.nav_min_nav_line_ratio
    ):
        return QualityVerdict(False, REASON_NAVIGATION, metrics)

    if metrics.url_line_ratio > policy.max_url_line_ratio:
        return QualityVerdict(False, REASON_URL_DIRECTORY, metrics)

    if (
        metrics.words >= policy.keyword_wall_min_words
        and metrics.type_token_ratio < policy.keyword_wall_max_type_token_ratio
    ):
        return QualityVerdict(False, REASON_KEYWORD_WALL, metrics)

    if check_secrets and find_secret(text) is not None:
        return QualityVerdict(False, REASON_SECRET, metrics)

    return QualityVerdict(True, None, metrics)
