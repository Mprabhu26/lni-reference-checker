"""
conftest.py — shared pytest fixtures available to all test files.
"""

import sys, os
from pathlib import Path
import pytest

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Env guard: redirect disk caches to temp dirs ──────────────────────────────
@pytest.fixture(autouse=True, scope="session")
def redirect_disk_cache(tmp_path_factory):
    """All tests use a fresh temp cache dir — never pollute the project cache."""
    tmp = str(tmp_path_factory.mktemp("lni_cache"))
    os.environ.setdefault("LNI_CACHE_DIR", tmp)
    os.environ.setdefault("LNI_DB_DIR", tmp)
    yield
    # No cleanup needed — pytest handles tmp_path


# ── BibEntry factory ──────────────────────────────────────────────────────────
@pytest.fixture
def make_bib_entry():
    """Factory fixture: make_bib_entry(key, title, authors, year, ...) → BibEntry."""
    from parser import BibEntry

    def _factory(key="AB20", raw_text="", title=None, authors=None,
                 year=None, entry_type=None, doi=None, url=None,
                 urldate=None, pages=None, journal=None, publisher=None):
        e = BibEntry(key=key, raw_text=raw_text)
        e.title = title
        e.authors = authors
        e.year = year
        e.entry_type = entry_type
        e.doi = doi
        e.url = url
        e.urldate = urldate
        e.pages = pages
        e.journal = journal
        e.publisher = publisher
        return e

    return _factory


# ── Real bib text fixture ─────────────────────────────────────────────────────
@pytest.fixture
def perfect_bib_text():
    return (
        "[LBH15] LeCun, Yann; Bengio, Yoshua; Hinton, Geoffrey: Deep Learning. "
        "In: Nature, Vol. 521, 2015; S. 436--444.\n"
        "[VSP17] Vaswani, Ashish; Shazeer, Noam; Parmar, Niki: "
        "Attention Is All You Need. "
        "In: Advances in Neural Information Processing Systems, 2017; S. 5998--6008."
    )


@pytest.fixture
def perfect_body_text():
    return (
        "We follow the approach of LeCun et al. [LBH15] and extend it using "
        "the transformer architecture proposed by [VSP17]."
    )
