"""Shared pytest configuration.

``scripts/`` is a directory of command-line entry points rather than an
installed package, so it is not importable by default.  Adding it to
``sys.path`` lets tests exercise script-level behaviour directly instead of
shelling out to a subprocess, which keeps failures readable and fast.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
