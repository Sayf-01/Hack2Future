"""Shared test fixtures and import path setup."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def root():
    """The project root directory."""
    return ROOT


@pytest.fixture(scope="session")
def peptides():
    """A few real-looking peptide sequences of varying length and charge."""
    return [
        "KWKLFKKIGAVLKVL",      # cationic, amphipathic, the classic shape
        "SSGNEENKKEEDSDEI",     # mixed charge, net negative
        "YSL",                  # very short
        "GLPVCGETCFTGTCYTNGCTCDPWPVCTRN",  # cysteine-rich
    ]
