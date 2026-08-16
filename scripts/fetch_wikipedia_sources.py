"""
Fetch Wikipedia article text into the tokenizer source pool.

This script populates ``data/tokenizer_sources/<category>/wiki/`` with plain
text articles so that ``build_tokenizer_corpus.py`` has enough material to
reach its target corpus size.

    fetch_wikipedia_sources.py
        -> data/tokenizer_sources/<category>/wiki/*.txt
        -> build_tokenizer_corpus.py
        -> tokenizer_corpus.jsonl
        -> train_tokenizer.py

Scope
-----
This is an acquisition step only.  It does NOT normalize, chunk, sample, or
weight anything; the corpus builder owns all of that.  The only guarantees
made here are:

* text is plain prose (MediaWiki ``extracts``, not wikitext or HTML);
* citation-only trailing sections are removed;
* each article is written exactly once, to exactly one category;
* each category stops at its configured byte budget;
* re-runs top up what is short instead of refetching.

Category budgets
----------------
Budgets mirror the corpus weights in ``build_tokenizer_corpus.py``:

    budget(category) = total_bytes * weight(category) * pool_factor

``--pool-factor`` exists because the builder samples from the pool.  A pool
exactly equal to the corpus target forces the builder to take everything,
which removes its ability to balance.  1.5x is a reasonable default.

Bytes already present in a category directory count toward its budget, so
hand-written sources are credited and only the shortfall is fetched.

Deduplication
-------------
An article may legitimately sit under seed categories of several buckets
(``Thermodynamics`` is both science and engineering).  Page IDs are claimed
globally, first come first served, and categories are processed in ascending
budget order so that narrow buckets claim shared articles before broad ones.
Without this the builder's SHA-256 dedup would still drop the duplicate, but
it would do so in arbitrary category order and quietly distort the mixture.

Politeness
----------
The Wikimedia API requires a descriptive User-Agent with contact details.
Set ``--user-agent`` accordingly.  Requests use ``maxlag`` and exponential
backoff, and honor ``Retry-After``.

Note on the API
---------------
``prop=extracts`` caps ``exlimit`` at 1 whenever full page text is requested
(``exlimit`` above 1 is only honored together with ``exintro``).  Title
enumeration is therefore batched 500 at a time, but text extraction costs one
request per article.

Licensing
---------
Wikipedia text is CC BY-SA 4.0.  Attribution metadata (title, page id,
revision id, URL) is recorded in the fetch manifest rather than embedded in
the ``.txt`` files, which keeps the training text clean.

Examples
--------
Plan only, no text fetched::

    python scripts/fetch_wikipedia_sources.py --dry-run

Fill the pool for a 30 MB corpus::

    python scripts/fetch_wikipedia_sources.py --total-bytes 30000000

Top up two categories only::

    python scripts/fetch_wikipedia_sources.py ^
        --category networking --category humanities
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys
import threading
import time
from typing import Any, Sequence
import unicodedata
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_ENDPOINT = "https://en.wikipedia.org/w/api.php"
DEFAULT_SOURCES_ROOT = Path("data/tokenizer_sources")
DEFAULT_MANIFEST_NAME = "wikipedia_fetch_manifest.json"
DEFAULT_SUBDIR = "wiki"

DEFAULT_TOTAL_BYTES = 30_000_000
DEFAULT_POOL_FACTOR = 1.5
DEFAULT_MIN_CHARS = 1_500
DEFAULT_MAX_CHARS = 200_000
DEFAULT_DEPTH = 2
DEFAULT_WORKERS = 4
DEFAULT_REQUEST_DELAY = 0.10
DEFAULT_MAX_RETRIES = 5
DEFAULT_SEED = 42
DEFAULT_TIMEOUT = 30.0

DEFAULT_USER_AGENT = (
    "my_LLM-tokenizer-corpus/0.3 "
    "(https://example.invalid/contact; set --user-agent with real contact info)"
)

# Mirrors DEFAULT_CATEGORY_WEIGHTS in build_tokenizer_corpus.py.
# Keep the two in sync when the mixture changes.
DEFAULT_CATEGORY_WEIGHTS = {
    "general": 0.30,
    "logical": 0.12,
    "mathematics": 0.10,
    "science": 0.12,
    "engineering": 0.10,
    "software": 0.08,
    "code": 0.05,
    "networking": 0.04,
    "business_finance": 0.04,
    "humanities": 0.05,
}

# Trailing sections that are citation apparatus rather than prose.  Everything
# from the first match onward is discarded.
STOP_SECTIONS = {
    "references",
    "external links",
    "see also",
    "further reading",
    "bibliography",
    "notes",
    "footnotes",
    "citations",
    "sources",
    "works cited",
    "explanatory notes",
    "notes and references",
    "general and cited references",
}

# Titles that are navigation or tabular rather than explanatory prose.
TITLE_REJECT_PATTERNS = (
    re.compile(r"^List of ", re.IGNORECASE),
    re.compile(r"^Lists of ", re.IGNORECASE),
    re.compile(r"^Index of ", re.IGNORECASE),
    re.compile(r"^Outline of ", re.IGNORECASE),
    re.compile(r"^Comparison of ", re.IGNORECASE),
    re.compile(r"\(disambiguation\)$", re.IGNORECASE),
)

# Subcategories that lead away from explanatory content.
SUBCATEGORY_REJECT_PATTERNS = (
    re.compile(r"\bstubs?\b", re.IGNORECASE),
    re.compile(r"\bby (year|country|century|decade|nationality)\b", re.IGNORECASE),
    re.compile(r"\b(births|deaths)\b", re.IGNORECASE),
    re.compile(r"\b(templates?|categories|redirects|articles|pages)\b", re.IGNORECASE),
    re.compile(r"\bWikipedia\b", re.IGNORECASE),
    re.compile(r"\blists?\b", re.IGNORECASE),
    re.compile(r"\bimages?\b|\bmedia\b", re.IGNORECASE),
)

# Literal reserved strings; build_tokenizer_corpus.py rejects whole documents
# containing these, so filter them here rather than fetching wasted bytes.
SPECIAL_TOKEN_MARKERS = ("<|", "|>")


@dataclass(frozen=True)
class CategorySeeds:
    """Seed material for one tokenizer category.

    ``articles`` is a reliable backbone that does not depend on the shape of
    the category graph.  ``categories`` is breadth for filling the budget.
    """

    articles: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()


# Derived from the project's per-category subject lists.  Seed article titles
# are exact; category names omit the "Category:" prefix.  Missing or renamed
# categories are reported and skipped rather than treated as fatal.
CATEGORY_SEEDS: dict[str, CategorySeeds] = {
    "general": CategorySeeds(
        articles=(
            "Geography", "Transport", "Rail transport", "Aviation",
            "Electric power transmission", "Electricity generation",
            "Water supply network", "Sanitation", "Infrastructure",
            "Agriculture", "Food industry", "Manufacturing", "Education",
            "Telecommunication", "Environmental protection", "Weather",
            "Climate", "History of technology", "History of transport",
            "Construction", "Architecture", "Urban planning",
            "Health care system", "Emergency service", "Navigation",
            "Measurement", "International System of Units", "Cartography",
            "Map projection", "Geographic coordinate system", "Logistics",
            "Mail", "Demography", "City", "Bridge", "Tunnel", "Dam",
            "Public transport", "Water treatment", "Waste management",
        ),
        categories=(
            "Geography", "Transport", "Energy", "Infrastructure",
            "Agriculture", "Manufacturing", "Education", "Construction",
            "Architecture", "Urban planning", "Health care", "Navigation",
            "Units of measurement", "Cartography", "Logistics", "Demography",
            "Water supply and sanitation", "History of technology",
            "Emergency services", "Environmental engineering",
        ),
    ),
    "logical": CategorySeeds(
        articles=(
            "Logic", "Propositional calculus", "First-order logic",
            "Boolean algebra", "Truth table", "Logical conjunction",
            "Logical disjunction", "Exclusive or", "Negation",
            "Material conditional", "Logical equivalence", "Syllogism",
            "Deductive reasoning", "Inductive reasoning", "Abductive reasoning",
            "Proof by contradiction", "Mathematical induction",
            "Set theory", "Venn diagram", "Constraint satisfaction problem",
            "Scheduling (computing)", "Causality", "Decision tree",
            "Finite-state machine", "Graph theory", "Mathematical optimization",
            "Modus ponens", "Modus tollens", "Fallacy", "Necessity and sufficiency",
            "Argument", "Validity (logic)", "Soundness", "Boolean satisfiability problem",
        ),
        categories=(
            "Logic", "Mathematical logic", "Propositional calculus",
            "Rules of inference", "Logical fallacies", "Boolean algebra",
            "Mathematical proofs", "Constraint programming",
            "Automata theory", "Decision theory", "Reasoning",
        ),
    ),
    "mathematics": CategorySeeds(
        articles=(
            "Integer", "Fraction", "Ratio", "Percentage", "Exponentiation",
            "Square root", "Scientific notation", "Algebraic expression",
            "Linear equation", "Quadratic equation", "System of linear equations",
            "Inequality (mathematics)", "Polynomial", "Function (mathematics)",
            "Logarithm", "Exponential function", "Analytic geometry",
            "Euclidean geometry", "Trigonometry", "Euclidean vector", "Matrix (mathematics)",
            "Determinant", "Eigenvalues and eigenvectors", "Limit (mathematics)",
            "Derivative", "Integral", "Differential equation", "Series (mathematics)",
            "Probability", "Probability distribution", "Descriptive statistics",
            "Statistical hypothesis testing", "Regression analysis", "Combinatorics",
            "Number theory", "Modular arithmetic", "Complex number",
            "Numerical analysis", "Mathematical proof",
        ),
        categories=(
            "Arithmetic", "Elementary algebra", "Linear algebra",
            "Euclidean geometry", "Trigonometry", "Calculus",
            "Differential equations", "Sequences and series",
            "Probability theory", "Probability distributions", "Statistics",
            "Regression analysis", "Combinatorics", "Number theory",
            "Complex analysis", "Numerical analysis",
        ),
    ),
    "science": CategorySeeds(
        articles=(
            "Classical mechanics", "Force", "Motion", "Energy", "Momentum",
            "Thermodynamics", "Heat transfer", "Fluid mechanics", "Wave",
            "Acoustics", "Optics", "Electromagnetism", "Atomic physics",
            "Nuclear physics", "Chemistry", "Chemical bond", "Acid",
            "Stoichiometry", "Electrochemistry", "Organic chemistry",
            "Biochemistry", "Cell (biology)", "Genetics", "Evolution",
            "Physiology", "Microbiology", "Ecology", "Geology",
            "Plate tectonics", "Mineralogy", "Meteorology", "Oceanography",
            "Astronomy", "Planetary science", "Materials science",
            "Measurement uncertainty", "Laboratory", "Photosynthesis",
            "Cellular respiration", "DNA",
        ),
        categories=(
            "Classical mechanics", "Thermodynamics", "Fluid mechanics",
            "Optics", "Electromagnetism", "Atomic physics", "Nuclear physics",
            "Chemistry", "Organic chemistry", "Biochemistry", "Cell biology",
            "Genetics", "Evolutionary biology", "Physiology", "Microbiology",
            "Ecology", "Geology", "Mineralogy", "Meteorology", "Oceanography",
            "Astronomy", "Planetary science", "Materials science",
        ),
    ),
    "engineering": CategorySeeds(
        articles=(
            "Electronic circuit", "Analogue electronics", "Digital electronics",
            "Ohm's law", "Kirchhoff's circuit laws", "Resistor", "Capacitor",
            "Inductor", "Diode", "Bipolar junction transistor", "MOSFET",
            "Operational amplifier", "Analog-to-digital converter",
            "Digital-to-analog converter", "Power supply", "DC-to-DC converter",
            "Electric battery", "Battery management system", "Embedded system",
            "Microcontroller", "General-purpose input/output", "Pulse-width modulation",
            "Interrupt", "Direct memory access", "Universal asynchronous receiver-transmitter",
            "Serial Peripheral Interface", "I²C", "CAN bus", "Sensor", "Actuator",
            "Electric motor", "Rotary encoder", "Robotics", "PID controller",
            "Control theory", "Digital signal processing", "Printed circuit board",
            "Ground (electricity)", "Electromagnetic compatibility",
            "Electric power system", "Transformer", "Modulation",
            "Radio frequency", "Antenna (radio)", "Statics", "Dynamics (mechanics)",
            "Gear", "Bearing (mechanical)", "Machining",
            "Computer-aided design", "Engineering tolerance", "Heat sink",
            "Reliability engineering", "Datasheet",
        ),
        categories=(
            "Electronic circuits", "Digital electronics", "Semiconductor devices",
            "Power electronics", "Embedded systems", "Microcontrollers",
            "Sensors", "Electric motors", "Robotics", "Control theory",
            "Digital signal processing", "Electromagnetic compatibility",
            "Electric power", "Antennas (radio)", "Mechanical engineering",
            "Manufacturing", "Heat transfer", "Reliability engineering",
        ),
    ),
    "software": CategorySeeds(
        articles=(
            "Software architecture", "Modular programming", "Interface (computing)",
            "Object-oriented programming", "Functional programming",
            "Operating system", "Process (computing)", "Thread (computing)",
            "Memory management", "Virtual memory", "File system",
            "Synchronization (computer science)", "Concurrency (computer science)",
            "Asynchronous I/O", "Data structure", "Algorithm",
            "Computational complexity theory", "Database",
            "Relational model", "SQL", "Database index", "Database transaction",
            "API", "REST", "Authentication", "Authorization", "Cryptography",
            "Software testing", "Debugging", "Logging (computing)",
            "CI/CD", "Version control", "Git", "Compiler", "Interpreter (computing)",
            "Parsing", "Distributed computing", "Cache (computing)",
            "Message queue", "Containerization (computing)", "Virtualization",
            "Cloud computing", "Observability (software)", "Software design pattern",
            "Code refactoring", "Software requirements specification",
        ),
        categories=(
            "Software architecture", "Object-oriented programming",
            "Functional programming", "Operating system technology",
            "Concurrent computing", "Data structures", "Algorithms",
            "Computational complexity theory", "Database management systems",
            "Cryptography", "Software testing", "Version control",
            "Compiler construction", "Distributed computing", "Virtualization",
            "Cloud computing", "Software design patterns",
        ),
    ),
    "networking": CategorySeeds(
        articles=(
            "OSI model", "Internet protocol suite", "Ethernet", "MAC address",
            "Address Resolution Protocol", "IPv4", "IPv6", "Subnetwork",
            "Classless Inter-Domain Routing", "Routing", "Routing table",
            "Network address translation", "VLAN", "Network switch",
            "Spanning Tree Protocol", "Domain Name System",
            "Dynamic Host Configuration Protocol", "Transmission Control Protocol",
            "User Datagram Protocol", "Port (computer networking)",
            "Network socket", "HTTP", "HTTP/2", "HTTP/3", "HTTPS",
            "Transport Layer Security", "Public key certificate",
            "Secure Shell", "File Transfer Protocol", "SFTP",
            "Simple Mail Transfer Protocol", "Internet Message Access Protocol",
            "WebSocket", "Wi-Fi", "Virtual private network", "Firewall (computing)",
            "Proxy server", "Load balancing (computing)", "Network security",
            "Packet analyzer", "Latency (engineering)", "Bandwidth (computing)",
            "Quality of service", "Border Gateway Protocol",
            "Open Shortest Path First", "Network packet",
        ),
        categories=(
            "Network protocols", "Internet protocols", "Internet Standards",
            "Routing protocols", "Transport layer protocols",
            "Application layer protocols", "Link protocols", "Ethernet",
            "Wi-Fi", "Network architecture", "Computer network security",
        ),
    ),
    "business_finance": CategorySeeds(
        articles=(
            "Accounting", "Asset", "Liability (financial accounting)", "Equity (finance)",
            "Balance sheet", "Income statement", "Cash flow statement", "Revenue",
            "Expense", "Gross income", "Net income", "EBITDA", "Profit margin",
            "Accounts receivable", "Accounts payable", "Inventory", "Depreciation",
            "Budget", "Forecasting", "Break-even (economics)",
            "Return on investment", "Return on equity", "Net present value",
            "Internal rate of return", "Compound interest", "Interest",
            "Loan", "Amortization (business)", "Bond (finance)", "Stock",
            "Market capitalization", "Exchange rate", "Inflation",
            "Monetary policy", "Interest rate", "Supply and demand",
            "Elasticity (economics)", "Microeconomics", "Macroeconomics",
            "Productivity", "Supply chain", "Procurement", "Pricing",
        ),
        categories=(
            "Accounting", "Financial statements", "Corporate finance",
            "Financial ratios", "Interest", "Bonds (finance)",
            "Stock market", "Foreign exchange market", "Inflation",
            "Monetary policy", "Microeconomics", "Macroeconomics",
            "Supply chain management", "Pricing",
        ),
    ),
    "humanities": CategorySeeds(
        articles=(
            "Ancient history", "Middle Ages", "Modern history",
            "History of science", "History of engineering", "Industrial Revolution",
            "History of computing", "Epistemology", "Logic in philosophy",
            "Ethics", "Political philosophy", "Linguistics", "Grammar",
            "Semantics", "Rhetoric", "Historical linguistics",
            "Literary criticism", "History of architecture", "Art history",
            "History of music", "Archaeology", "Anthropology",
            "Cultural history", "History of mathematics", "History of medicine",
            "Primary source", "Secondary source", "Historiography",
            "Phonology", "Morphology (linguistics)", "Syntax", "Pragmatics",
        ),
        categories=(
            "Ancient history", "Medieval history", "Modern history",
            "History of science", "History of mathematics", "History of medicine",
            "Epistemology", "Ethics", "Political philosophy", "Linguistics",
            "Rhetoric", "Literary criticism", "Art history", "Archaeology",
            "Anthropology", "Historiography",
        ),
    ),
    # 'code' is intentionally absent.  Wikipedia has prose *about* programming
    # languages, not realistic source files.  That category should be filled
    # from permissively licensed repositories instead.
}


@dataclass
class FetchStats:
    categories_planned: int = 0
    titles_enumerated: int = 0
    api_requests: int = 0
    api_retries: int = 0
    articles_fetched: int = 0
    articles_written: int = 0
    bytes_written: int = 0
    skipped_already_have: int = 0
    skipped_claimed: int = 0
    skipped_title_pattern: int = 0
    skipped_too_short: int = 0
    skipped_too_long: int = 0
    skipped_disambiguation: int = 0
    skipped_special_token: int = 0
    skipped_empty_extract: int = 0
    missing_categories: list[str] = field(default_factory=list)


@dataclass
class CategoryPlan:
    category: str
    weight: float
    budget_bytes: int
    existing_bytes: int
    remaining_bytes: int
    candidate_titles: list[str] = field(default_factory=list)
    candidate_pageids: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Article:
    pageid: int
    revid: int | None
    title: str
    text: str
    utf8_bytes: int


class ApiError(RuntimeError):
    """Unrecoverable MediaWiki API failure."""


class RateLimiter:
    """Serialize a minimum delay between outbound requests."""

    def __init__(self, delay: float) -> None:
        self._delay = max(0.0, delay)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._delay <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                sleep_for = self._next_allowed - now
            else:
                sleep_for = 0.0
            self._next_allowed = max(now, self._next_allowed) + self._delay
        if sleep_for > 0:
            time.sleep(sleep_for)


class WikiClient:
    """Minimal MediaWiki API client with retry and backoff."""

    def __init__(
        self,
        *,
        endpoint: str,
        user_agent: str,
        limiter: RateLimiter,
        stats: FetchStats,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._endpoint = endpoint
        self._user_agent = user_agent
        self._limiter = limiter
        self._stats = stats
        self._timeout = timeout
        self._max_retries = max_retries
        self._stats_lock = threading.Lock()

    def _bump(self, attribute: str) -> None:
        with self._stats_lock:
            setattr(self._stats, attribute, getattr(self._stats, attribute) + 1)

    def query(self, params: dict[str, Any]) -> dict[str, Any]:
        merged = {
            "format": "json",
            "formatversion": "2",
            "maxlag": "5",
            **params,
        }
        url = self._endpoint + "?" + urllib.parse.urlencode(merged)

        backoff = 1.0
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            if attempt:
                self._bump("api_retries")

            self._limiter.wait()
            self._bump("api_requests")

            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self._user_agent,
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
            )

            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self._timeout,
                ) as response:
                    payload = json.loads(
                        response.read().decode("utf-8", errors="strict")
                    )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in (429, 503) or 500 <= exc.code < 600:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    delay = backoff
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    time.sleep(delay)
                    backoff = min(backoff * 2, 60.0)
                    continue
                raise ApiError(f"HTTP {exc.code} for {params}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue

            # maxlag and other soft failures arrive as HTTP 200 with an error.
            if "error" in payload:
                code = payload["error"].get("code", "")
                if code in ("maxlag", "readonly", "internal_api_error"):
                    last_error = ApiError(str(payload["error"]))
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue
                raise ApiError(f"API error {payload['error']}")

            return payload

        raise ApiError(
            f"exhausted {self._max_retries} attempts for {params}: {last_error}"
        )


def slugify(title: str, *, max_length: int = 60) -> str:
    """Build a filesystem-safe, deterministic slug from an article title."""

    decomposed = unicodedata.normalize("NFKD", title)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if not cleaned:
        cleaned = "article"
    return cleaned[:max_length].rstrip("_")


def clean_extract(raw: str) -> str:
    """Trim citation sections and convert wiki headings to plain lines.

    ``exsectionformat=wiki`` renders headings as ``== Title ==`` which makes
    stop-section detection reliable.  Headings are kept as plain text lines so
    the document retains structure without markup noise.
    """

    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept: list[str] = []

    heading = re.compile(r"^\s*(={2,6})\s*(.+?)\s*\1\s*$")

    for line in lines:
        match = heading.match(line)
        if match:
            title = match.group(2).strip()
            if title.casefold() in STOP_SECTIONS:
                break
            kept.append(title)
            continue
        kept.append(line)

    text = "\n".join(kept)
    # Collapse runs of blank lines left behind by removed elements.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def title_is_rejected(title: str) -> bool:
    return any(pattern.search(title) for pattern in TITLE_REJECT_PATTERNS)


def subcategory_is_rejected(title: str) -> bool:
    return any(pattern.search(title) for pattern in SUBCATEGORY_REJECT_PATTERNS)


def contains_special_token(text: str) -> bool:
    return all(marker in text for marker in SPECIAL_TOKEN_MARKERS)


def enumerate_category(
    client: WikiClient,
    *,
    category: str,
    depth: int,
    per_category_limit: int,
    stats: FetchStats,
) -> dict[str, int]:
    """Breadth-first walk of a category, returning ``{title: pageid}``."""

    found: dict[str, int] = {}
    visited: set[str] = set()
    frontier: list[tuple[str, int]] = [(category, 0)]

    while frontier and len(found) < per_category_limit:
        name, level = frontier.pop(0)
        key = name.casefold()
        if key in visited:
            continue
        visited.add(key)

        cmtitle = name if name.startswith("Category:") else f"Category:{name}"
        cmcontinue: str | None = None
        saw_any = False

        while len(found) < per_category_limit:
            params: dict[str, Any] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": cmtitle,
                "cmlimit": "500",
                "cmtype": "page|subcat",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            try:
                payload = client.query(params)
            except ApiError:
                break

            members = payload.get("query", {}).get("categorymembers", [])
            if members:
                saw_any = True

            for member in members:
                title = member.get("title", "")
                pageid = member.get("pageid")
                ns = member.get("ns")

                if ns == 14:  # subcategory
                    if level >= depth:
                        continue
                    bare = title.split(":", 1)[-1]
                    if subcategory_is_rejected(bare):
                        continue
                    frontier.append((title, level + 1))
                    continue

                if ns != 0 or not isinstance(pageid, int):
                    continue
                if title_is_rejected(title):
                    stats.skipped_title_pattern += 1
                    continue

                found.setdefault(title, pageid)
                if len(found) >= per_category_limit:
                    break

            cont = payload.get("continue", {}).get("cmcontinue")
            if not cont:
                break
            cmcontinue = cont

        if level == 0 and not saw_any:
            stats.missing_categories.append(category)

    return found


def fetch_article(
    client: WikiClient,
    *,
    title: str,
) -> Article | None:
    """Fetch one article's plain-text extract."""

    payload = client.query(
        {
            "action": "query",
            "prop": "extracts|revisions|pageprops",
            "explaintext": "1",
            "exsectionformat": "wiki",
            "redirects": "1",
            "rvprop": "ids",
            "ppprop": "disambiguation",
            "titles": title,
        }
    )

    pages = payload.get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    if page.get("missing") or page.get("invalid"):
        return None
    if "disambiguation" in (page.get("pageprops") or {}):
        return None

    pageid = page.get("pageid")
    if not isinstance(pageid, int):
        return None

    raw = page.get("extract") or ""
    text = clean_extract(raw)
    if not text:
        return None

    revisions = page.get("revisions") or []
    revid = revisions[0].get("revid") if revisions else None

    return Article(
        pageid=pageid,
        revid=revid,
        title=page.get("title", title),
        text=text,
        utf8_bytes=len(text.encode("utf-8")),
    )


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and item.suffix.lower() in {".txt", ".text", ".md"}:
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "format": "wikipedia_fetch_manifest",
            "format_version": 1,
            "endpoint": DEFAULT_ENDPOINT,
            "license": "CC BY-SA 4.0",
            "runs": [],
            "articles": {},
        }

    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("articles", {})
    data.setdefault("runs", [])
    return data


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_plans(
    *,
    categories: Sequence[str],
    weights: dict[str, float],
    total_bytes: int,
    pool_factor: float,
    sources_root: Path,
) -> list[CategoryPlan]:
    """Compute per-category budgets, crediting bytes already on disk."""

    plans: list[CategoryPlan] = []

    for category in categories:
        weight = weights.get(category, 0.0)
        budget = int(total_bytes * weight * pool_factor)
        existing = directory_bytes(sources_root / category)
        plans.append(
            CategoryPlan(
                category=category,
                weight=weight,
                budget_bytes=budget,
                existing_bytes=existing,
                remaining_bytes=max(0, budget - existing),
            )
        )

    # Ascending remaining budget: narrow buckets claim shared articles first.
    plans.sort(key=lambda plan: (plan.remaining_bytes, plan.category))
    return plans


def plan_candidates(
    client: WikiClient,
    plan: CategoryPlan,
    *,
    seeds: CategorySeeds,
    depth: int,
    per_category_limit: int,
    seed: int,
    stats: FetchStats,
) -> None:
    """Populate ``plan`` with candidate titles in deterministic order."""

    discovered: dict[str, int] = {}

    for name in seeds.categories:
        if len(discovered) >= per_category_limit:
            break
        discovered.update(
            enumerate_category(
                client,
                category=name,
                depth=depth,
                per_category_limit=per_category_limit - len(discovered),
                stats=stats,
            )
        )

    rng = random.Random(f"{seed}:{plan.category}")
    shuffled = sorted(discovered)
    rng.shuffle(shuffled)

    # Seed articles first: a reliable backbone independent of the category graph.
    ordered: list[str] = []
    seen: set[str] = set()

    for title in seeds.articles:
        key = title.casefold()
        if key not in seen and not title_is_rejected(title):
            seen.add(key)
            ordered.append(title)

    for title in shuffled:
        key = title.casefold()
        if key not in seen:
            seen.add(key)
            ordered.append(title)

    plan.candidate_titles = ordered
    plan.candidate_pageids = discovered
    stats.titles_enumerated += len(ordered)


def fetch_for_category(
    client: WikiClient,
    plan: CategoryPlan,
    *,
    sources_root: Path,
    subdir: str,
    claimed_pageids: set[int],
    claim_lock: threading.Lock,
    manifest_articles: dict[str, Any],
    min_chars: int,
    max_chars: int,
    workers: int,
    stats: FetchStats,
    quiet: bool,
) -> int:
    """Fetch and write until the category budget is met. Returns bytes written."""

    if plan.remaining_bytes <= 0:
        return 0

    target_dir = sources_root / plan.category / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    written_bytes = 0
    write_lock = threading.Lock()
    stop = threading.Event()

    pending = [
        title
        for title in plan.candidate_titles
        # Cheap pre-filter for titles whose page id we already know.
        if plan.candidate_pageids.get(title) not in claimed_pageids
    ]

    def handle(title: str) -> None:
        nonlocal written_bytes

        if stop.is_set():
            return

        known_id = plan.candidate_pageids.get(title)
        if known_id is not None:
            with claim_lock:
                if known_id in claimed_pageids:
                    stats.skipped_claimed += 1
                    return

        try:
            article = fetch_article(client, title=title)
        except ApiError:
            return

        if article is None:
            stats.skipped_empty_extract += 1
            return

        stats.articles_fetched += 1

        with claim_lock:
            if article.pageid in claimed_pageids:
                stats.skipped_claimed += 1
                return
            if str(article.pageid) in manifest_articles:
                stats.skipped_already_have += 1
                claimed_pageids.add(article.pageid)
                return
            claimed_pageids.add(article.pageid)

        if len(article.text) < min_chars:
            stats.skipped_too_short += 1
            return
        if len(article.text) > max_chars:
            stats.skipped_too_long += 1
            return
        if contains_special_token(article.text):
            stats.skipped_special_token += 1
            return

        filename = f"{article.pageid:08d}_{slugify(article.title)}.txt"
        destination = target_dir / filename

        with write_lock:
            if stop.is_set():
                return

            destination.write_text(
                article.text + "\n",
                encoding="utf-8",
                newline="\n",
            )

            written_bytes += article.utf8_bytes
            stats.articles_written += 1
            stats.bytes_written += article.utf8_bytes

            manifest_articles[str(article.pageid)] = {
                "title": article.title,
                "category": plan.category,
                "revid": article.revid,
                "utf8_bytes": article.utf8_bytes,
                "file": str(destination.as_posix()),
                "url": "https://en.wikipedia.org/?curid=" + str(article.pageid),
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            }

            if written_bytes >= plan.remaining_bytes:
                stop.set()

            if not quiet and stats.articles_written % 25 == 0:
                pct = 100.0 * written_bytes / max(1, plan.remaining_bytes)
                print(
                    f"  {plan.category:<18} "
                    f"{written_bytes:>10,} / {plan.remaining_bytes:,} bytes "
                    f"({pct:5.1f}%)",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(handle, pending):
            if stop.is_set():
                break

    return written_bytes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Wikipedia plain-text articles into the tokenizer source "
            "pool, budgeted per category."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        default=None,
        help=(
            "Limit the run to one category. Repeat as needed. "
            "Default: every category with seed definitions."
        ),
    )
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=DEFAULT_SOURCES_ROOT,
        help="Root directory containing per-category source folders.",
    )
    parser.add_argument(
        "--subdir",
        default=DEFAULT_SUBDIR,
        help="Subdirectory inside each category for fetched articles.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Fetch manifest path. Default: <sources-root>/"
        + DEFAULT_MANIFEST_NAME,
    )
    parser.add_argument(
        "--total-bytes",
        type=int,
        default=DEFAULT_TOTAL_BYTES,
        help="Target corpus size the pool must be able to support.",
    )
    parser.add_argument(
        "--pool-factor",
        type=float,
        default=DEFAULT_POOL_FACTOR,
        help=(
            "Multiplier applied to each category target so the corpus builder "
            "has room to sample rather than take everything."
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help="Maximum subcategory recursion depth.",
    )
    parser.add_argument(
        "--per-category-limit",
        type=int,
        default=4000,
        help="Maximum candidate titles enumerated per tokenizer category.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Discard articles shorter than this after cleaning.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Discard articles longer than this after cleaning.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent extract requests.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help="Minimum seconds between outbound requests, across all workers.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="MediaWiki API endpoint.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help=(
            "User-Agent header. Wikimedia policy requires contact details; "
            "replace the default before any substantial run."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for deterministic candidate ordering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Enumerate candidates and print the plan without fetching or "
            "writing any article text."
        ),
    )
    parser.add_argument(
        "--show-titles",
        type=int,
        default=15,
        help="Sample titles to print per category in --dry-run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output except errors.",
    )

    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.total_bytes <= 0:
        raise ValueError("--total-bytes must be > 0")
    if args.pool_factor <= 0:
        raise ValueError("--pool-factor must be > 0")
    if args.depth < 0:
        raise ValueError("--depth must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.max_chars < args.min_chars:
        raise ValueError("--max-chars must be >= --min-chars")

    if args.categories:
        unknown = sorted(set(args.categories) - set(CATEGORY_SEEDS))
        if unknown:
            raise ValueError(
                "no seed definitions for category: "
                + ", ".join(unknown)
                + " (known: "
                + ", ".join(sorted(CATEGORY_SEEDS))
                + ")"
            )


def print_plan(plans: Sequence[CategoryPlan], *, show_titles: int) -> None:
    print()
    print("Fetch plan")
    print("=" * 80)
    print(
        f"{'Category':<18} {'Weight':>7} {'Budget':>12} "
        f"{'On disk':>12} {'To fetch':>12} {'Cands':>7}"
    )
    for plan in plans:
        print(
            f"{plan.category:<18} "
            f"{plan.weight:>6.1%} "
            f"{plan.budget_bytes:>12,} "
            f"{plan.existing_bytes:>12,} "
            f"{plan.remaining_bytes:>12,} "
            f"{len(plan.candidate_titles):>7,}"
        )

    if show_titles > 0:
        for plan in plans:
            if not plan.candidate_titles:
                continue
            print()
            print(f"{plan.category} sample:")
            for title in plan.candidate_titles[:show_titles]:
                print(f"  - {title}")


def print_summary(stats: FetchStats, *, manifest_path: Path) -> None:
    print()
    print("Wikipedia Source Fetch")
    print("=" * 80)
    print(f"Categories planned:      {stats.categories_planned:,}")
    print(f"Titles enumerated:       {stats.titles_enumerated:,}")
    print(f"API requests:            {stats.api_requests:,}")
    print(f"API retries:             {stats.api_retries:,}")
    print(f"Articles fetched:        {stats.articles_fetched:,}")
    print(f"Articles written:        {stats.articles_written:,}")
    print(f"Bytes written:           {stats.bytes_written:,}")
    print()
    print("Filtered")
    print("-" * 80)
    print(f"Already in manifest:     {stats.skipped_already_have:,}")
    print(f"Claimed by other cat.:   {stats.skipped_claimed:,}")
    print(f"Title pattern:           {stats.skipped_title_pattern:,}")
    print(f"Disambiguation/missing:  {stats.skipped_empty_extract:,}")
    print(f"Too short:               {stats.skipped_too_short:,}")
    print(f"Too long:                {stats.skipped_too_long:,}")
    print(f"Special-token collision: {stats.skipped_special_token:,}")

    if stats.missing_categories:
        print()
        print("Seed categories that returned nothing (renamed or nonexistent):")
        for name in sorted(set(stats.missing_categories)):
            print(f"  - {name}")

    print()
    print(f"Manifest: {manifest_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        validate_args(args)

        sources_root = args.sources_root.expanduser().resolve()
        manifest_path = (
            args.manifest.expanduser().resolve()
            if args.manifest
            else sources_root / DEFAULT_MANIFEST_NAME
        )

        categories = args.categories or sorted(CATEGORY_SEEDS)

        stats = FetchStats()
        limiter = RateLimiter(args.request_delay)
        client = WikiClient(
            endpoint=args.endpoint,
            user_agent=args.user_agent,
            limiter=limiter,
            stats=stats,
        )

        plans = build_plans(
            categories=categories,
            weights=DEFAULT_CATEGORY_WEIGHTS,
            total_bytes=args.total_bytes,
            pool_factor=args.pool_factor,
            sources_root=sources_root,
        )
        stats.categories_planned = len(plans)

        if not args.quiet:
            print("Enumerating candidate articles...")

        for plan in plans:
            plan_candidates(
                client,
                plan,
                seeds=CATEGORY_SEEDS[plan.category],
                depth=args.depth,
                per_category_limit=args.per_category_limit,
                seed=args.seed,
                stats=stats,
            )

        if args.dry_run:
            print_plan(plans, show_titles=args.show_titles)
            print()
            print("Dry run: no article text fetched, no files written.")
            return 0

        manifest = load_manifest(manifest_path)
        manifest_articles: dict[str, Any] = manifest["articles"]
        claimed_pageids: set[int] = {
            int(pageid) for pageid in manifest_articles if pageid.isdigit()
        }
        claim_lock = threading.Lock()

        if not args.quiet:
            print_plan(plans, show_titles=0)
            print()
            print("Fetching...")

        for plan in plans:
            fetch_for_category(
                client,
                plan,
                sources_root=sources_root,
                subdir=args.subdir,
                claimed_pageids=claimed_pageids,
                claim_lock=claim_lock,
                manifest_articles=manifest_articles,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                workers=args.workers,
                stats=stats,
                quiet=args.quiet,
            )

        manifest["endpoint"] = args.endpoint
        manifest["runs"].append(
            {
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "total_bytes_target": args.total_bytes,
                "pool_factor": args.pool_factor,
                "categories": categories,
                "stats": {
                    key: value
                    for key, value in asdict(stats).items()
                    if key != "missing_categories"
                },
            }
        )
        write_json(manifest_path, manifest)

        if not args.quiet:
            print_summary(stats, manifest_path=manifest_path)

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted; partial results are on disk.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, UnicodeError, ApiError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
