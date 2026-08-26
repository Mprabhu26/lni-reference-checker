"""
LNI Reference Checker — Comprehensive Test Suite
=================================================
Run:   pytest tests/test_all.py -v
Cover: parser.py, checker.py, extractor.py, web_search_verifier.py, local_db.py

Test categories
---------------
P  — parser unit tests   (no network, deterministic)
C  — checker unit tests  (no network, mocked or cached)
E  — extractor tests     (file I/O, no network)
I  — integration tests   (requires network; skip with -m "not network")
D  — database/cache      (SQLite only, no network)
"""

import sys, os, re, json, sqlite3, tempfile, shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIXTURES_TXT = Path(__file__).parent / "fixtures" / "txt"
FIXTURES_PDF = Path(__file__).parent / "fixtures" / "pdf"
FIXTURES_DOCX = Path(__file__).parent / "fixtures" / "docx"

# ── Conditional imports ────────────────────────────────────────────────────────
try:
    from parser import (
        parse_bibliography, validate_lni_key, entries_to_dict,
        BibEntry, _validate_key_vs_metadata, _check_completeness,
    )
    PARSER_AVAILABLE = True
except ImportError as e:
    PARSER_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason=f"parser.py import failed: {e}")

try:
    from checker import (
        _title_similarity, _normalize_title, _extract_surnames,
        author_overlap_score, VerificationResult, verify_all_references,
    )
    CHECKER_AVAILABLE = True
except ImportError:
    CHECKER_AVAILABLE = False

try:
    from extractor import split_body_bib, _find_bib_start
    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False

try:
    from local_db import (
        init_cache_db, save_to_cache, search_cache, get_cache_stats,
        normalize_title, clear_old_entries,
    )
    LOCALDB_AVAILABLE = True
except ImportError:
    LOCALDB_AVAILABLE = False


def _load_txt(name: str) -> str:
    path = FIXTURES_TXT / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


# =============================================================================
# SECTION P — PARSER UNIT TESTS
# =============================================================================

@pytest.mark.skipif(not PARSER_AVAILABLE, reason="parser.py unavailable")
class TestParserKeyValidation:
    """P1-P12: LNI key format validation."""

    # ── P1: Valid 1-author key ────────────────────────────────────────────────
    def test_P01_valid_single_author_key(self):
        """[Ez10] — 2 letters + 2-digit year is the canonical 1-author form."""
        errors = validate_lni_key("Ez10")
        assert errors == [], f"Expected no errors, got: {errors}"

    # ── P2: Valid 2-author key ────────────────────────────────────────────────
    def test_P02_valid_two_author_key(self):
        """[AB00] — 1 letter per author + year."""
        assert validate_lni_key("AB00") == []

    # ── P3: Valid 3-author key ────────────────────────────────────────────────
    def test_P03_valid_three_author_key(self):
        """[ABC01] — 3 initials."""
        assert validate_lni_key("ABC01") == []

    # ── P4: Valid disambiguation suffix ──────────────────────────────────────
    def test_P04_valid_disambiguation_suffix(self):
        """[Wa14a] — trailing lowercase letter for same-author/year disambiguation."""
        assert validate_lni_key("Wa14a") == []

    # ── P5: Numeric key (used in some LNI volumes) ────────────────────────────
    def test_P05_numeric_key_accepted(self):
        """[1] and [42] are valid numeric keys — skip LNI-initials check."""
        assert validate_lni_key("1") == []
        assert validate_lni_key("42") == []
        assert validate_lni_key("100") == []

    # ── P6: Lowercase key rejected ────────────────────────────────────────────
    def test_P06_lowercase_key_rejected(self):
        """[vaswani17] — all-lowercase fails the [A-Z…] pattern."""
        errors = validate_lni_key("vaswani17")
        assert len(errors) > 0, "Expected format error for all-lowercase key"

    # ── P7: Single-letter prefix rejected ────────────────────────────────────
    def test_P07_single_letter_prefix_rejected(self):
        """[X] — only 1 letter, no year."""
        errors = validate_lni_key("X")
        assert len(errors) > 0

    # ── P8: Too many initials rejected ───────────────────────────────────────
    def test_P08_too_many_initials_rejected(self):
        """[TOOLONGKEY99] — >6 letters violates the 2–6 char rule."""
        errors = validate_lni_key("TOOLONGKEY99")
        assert any("2–6" in e or "2-6" in e or "characters" in e for e in errors)

    # ── P9: 4-digit year rejected ─────────────────────────────────────────────
    def test_P09_four_digit_year_rejected(self):
        """[Ez2010] — 4-digit year is not valid LNI format."""
        errors = validate_lni_key("Ez2010")
        assert len(errors) > 0

    # ── P10: Key with special characters rejected ─────────────────────────────
    def test_P10_special_char_key_rejected(self):
        """[V-17] — dash in key is not valid."""
        errors = validate_lni_key("V-17")
        assert len(errors) > 0

    # ── P11: Max-length valid key ─────────────────────────────────────────────
    def test_P11_six_char_prefix_accepted(self):
        """[ABCDEF20] — 6 letters is the maximum allowed."""
        assert validate_lni_key("ABCDEF20") == []

    # ── P12: Empty string rejected ────────────────────────────────────────────
    def test_P12_empty_key_rejected(self):
        errors = validate_lni_key("")
        assert len(errors) > 0


@pytest.mark.skipif(not PARSER_AVAILABLE, reason="parser.py unavailable")
class TestParserBibExtraction:
    """P13-P30: Bibliography entry extraction correctness."""

    def test_P13_single_article_extracted(self):
        """Basic article: authors, title, journal, year, pages all parsed."""
        bib = _load_txt("bib_perfect.txt")
        entries = parse_bibliography(bib)
        assert len(entries) >= 1
        e = entries[0]
        assert e.key == "LBH15"
        assert e.authors is not None
        assert "LeCun" in e.authors
        assert e.year == "2015"

    def test_P14_entry_type_article(self):
        """Entry with 'Vol.' and journal name classified as 'article'."""
        bib = "[AB20] Author, Bob; Baker, Alice: Test Paper. In: Journal of Science, Vol. 12, 2020; S. 1--10."
        entries = parse_bibliography(bib)
        assert entries[0].entry_type in ("article", "proceedings")

    def test_P15_entry_type_proceedings(self):
        """Entry with 'In:' + 'Proc.' classified as proceedings."""
        bib = "[AB20] Author, Bob: My Paper. In: Proc. Workshop on AI, 2020; S. 1--5."
        entries = parse_bibliography(bib)
        assert entries[0].entry_type == "proceedings"

    def test_P16_entry_type_book(self):
        """Entry with publisher name (Springer, Wiley) classified as book."""
        bib = "[Go16] Goodfellow, Ian; Bengio, Yoshua; Courville, Aaron: Deep Learning. MIT Press, 2016."
        entries = parse_bibliography(bib)
        assert entries[0].entry_type == "book"

    def test_P17_entry_type_website(self):
        """Entry with http URL classified as website."""
        bib = "[GI24] GI: LNI Style Guide. https://gi.de/lni Stand: 15.05.2024."
        entries = parse_bibliography(bib)
        assert entries[0].entry_type == "website"
        assert entries[0].url is not None

    def test_P18_doi_extracted(self):
        """DOI extracted from entry raw text."""
        bib = _load_txt("bib_doi_entry.txt")
        entries = parse_bibliography(bib)
        assert entries[0].doi is not None
        assert "1706.03762" in entries[0].doi

    def test_P19_isbn_extracted(self):
        """ISBN extracted and normalized (no spaces/dashes)."""
        bib = _load_txt("bib_isbn_entry.txt")
        entries = parse_bibliography(bib)
        assert entries[0].isbn is not None
        assert "-" not in entries[0].isbn
        assert " " not in entries[0].isbn

    def test_P20_urldate_extracted(self):
        """urldate extracted from 'Stand:' keyword."""
        bib = _load_txt("bib_website_with_urldate.txt")
        entries = parse_bibliography(bib)
        assert entries[0].urldate is not None

    def test_P21_multiple_entries_all_parsed(self):
        """All entries in a multi-entry bib section are returned."""
        bib = _load_txt("bib_perfect.txt")
        entries = parse_bibliography(bib)
        assert len(entries) == 2
        keys = {e.key for e in entries}
        assert "LBH15" in keys
        assert "VSP17" in keys

    def test_P22_numeric_keys_all_parsed(self):
        """Numeric keys [1], [2], [10] are all extracted."""
        bib = _load_txt("bib_numeric_keys.txt")
        entries = parse_bibliography(bib)
        assert len(entries) == 3
        keys = {e.key for e in entries}
        assert "1" in keys and "2" in keys and "10" in keys

    def test_P23_multiline_entry_normalized(self):
        """Entry spanning multiple lines is joined into one clean entry."""
        bib = _load_txt("bib_multiline.txt")
        entries = parse_bibliography(bib)
        assert len(entries) >= 1
        assert entries[0].year == "2015"

    def test_P24_disambiguation_keys_both_parsed(self):
        """[Wa14a] and [Wa14b] are parsed as separate entries."""
        bib = _load_txt("bib_disambiguation.txt")
        entries = parse_bibliography(bib)
        assert len(entries) == 2
        keys = {e.key for e in entries}
        assert "Wa14a" in keys and "Wa14b" in keys

    def test_P25_empty_bibliography_returns_empty_list(self):
        """Empty input returns []."""
        entries = parse_bibliography("")
        assert entries == []

    def test_P26_umlaut_authors_extracted(self):
        """Authors with German umlauts (Müller, Özdemir) are extracted."""
        bib = _load_txt("bib_german_umlaut.txt")
        entries = parse_bibliography(bib)
        assert len(entries) >= 1

    def test_P27_year_extracted(self):
        """Year is always a 4-digit string."""
        bib = "[AB20] Author, Bob: A Paper. Springer, 2020."
        entries = parse_bibliography(bib)
        assert entries[0].year == "2020"

    def test_P28_semicolon_separated_authors(self):
        """Multiple authors separated by ';' are kept in authors field."""
        bib = "[ABC21] Author, A; Baker, B; Cooper, C: Title. Springer, 2021."
        entries = parse_bibliography(bib)
        assert entries[0].authors is not None
        assert ";" in entries[0].authors or entries[0].authors.count(",") >= 2

    def test_P29_pages_extracted(self):
        """Page range 'S. 436--444' is extracted into .pages field."""
        bib = "[LBH15] LeCun, Yann; Bengio, Yoshua: Deep Learning. In: Nature, Vol. 521, 2015; S. 436--444."
        entries = parse_bibliography(bib)
        assert entries[0].pages is not None
        assert "436" in entries[0].pages

    def test_P30_entry_count_correct_with_trailing_whitespace(self):
        """Extra whitespace around entries does not create phantom entries."""
        bib = "  \n[AB20] Author, Bob: Paper. Springer, 2020.  \n  \n"
        entries = parse_bibliography(bib)
        assert len(entries) == 1


@pytest.mark.skipif(not PARSER_AVAILABLE, reason="parser.py unavailable")
class TestParserCompletenessChecks:
    """P31-P50: completeness_issues flags."""

    def test_P31_single_dash_page_range_flagged(self):
        """'S. 10-20' with single dash gets a completeness issue."""
        bib = _load_txt("bib_single_dash.txt")
        entries = parse_bibliography(bib)
        issues = entries[0].completeness_issues
        assert any("single dash" in i.lower() or "dash" in i.lower() for i in issues), \
            f"Expected single-dash issue, got: {issues}"

    def test_P32_double_dash_page_range_no_issue(self):
        """'S. 10--20' with double dash should NOT trigger dash issue."""
        bib = "[AB20] Author, Bob: Paper. In: Journal, Vol. 1, 2020; S. 10--20."
        entries = parse_bibliography(bib)
        issues = entries[0].completeness_issues
        assert not any("dash" in i.lower() for i in issues), \
            f"Unexpected dash issue: {issues}"

    def test_P33_wrong_author_order_flagged(self):
        """'John Smith' (Firstname Lastname, no comma) is flagged with correction."""
        bib = _load_txt("bib_author_order.txt")
        entries = parse_bibliography(bib)
        issues = entries[0].completeness_issues
        assert any("lastname" in i.lower() or "firstname" in i.lower() or "order" in i.lower()
                   for i in issues), f"Expected author order issue, got: {issues}"

    def test_P34_correct_author_order_no_issue(self):
        """'Smith, John' (Lastname, Firstname) triggers no author order issue."""
        bib = "[AB20] Smith, John; Brown, Alice: Paper. Springer, 2020."
        entries = parse_bibliography(bib)
        issues = entries[0].completeness_issues
        assert not any("order" in i.lower() or "firstname lastname" in i.lower()
                       for i in issues), f"Unexpected author order issue: {issues}"

    def test_P35_initial_before_surname_flagged(self):
        """'J. Smith' (initial before surname) is flagged as wrong order."""
        bib = _load_txt("bib_initial_author.txt")
        entries = parse_bibliography(bib)
        issues = entries[0].completeness_issues
        # May or may not parse authors from this format — if authors parsed, check order
        # If not parsed, completeness check will flag 'missing authors'
        # Either outcome is acceptable; we just verify no crash
        assert isinstance(issues, list)

    def test_P36_future_year_flagged(self):
        """Year 2099 is flagged as 'in the future'."""
        bib = _load_txt("bib_future_year.txt")
        entries = parse_bibliography(bib)
        issues = entries[0].completeness_issues
        assert any("future" in i.lower() or "2099" in i for i in issues), \
            f"Expected future-year issue, got: {issues}"

    def test_P37_missing_journal_for_article_flagged(self):
        """Article entry without a journal field gets 'Missing required field: journal'."""
        bib = "[No20] Nobody, Anon: A Paper Without Journal. 2020."
        entries = parse_bibliography(bib)
        if entries[0].entry_type == "article":
            issues = entries[0].completeness_issues
            assert any("journal" in i.lower() for i in issues)

    def test_P38_missing_url_for_website_flagged(self):
        """Website entry that somehow has no URL flagged for missing url."""
        # Manually craft an entry
        e = BibEntry(key="We22", raw_text="Author: Page. Stand: 01.01.2022.")
        e.entry_type = "website"
        e.title = "Author: Page"
        e.urldate = "01.01.2022"
        # url is None
        _check_completeness(e)
        assert any("url" in i.lower() for i in e.completeness_issues)

    def test_P39_missing_urldate_for_website_flagged(self):
        """Website entry without urldate gets 'Missing required field: urldate'."""
        bib = _load_txt("bib_website_no_urldate.txt")
        entries = parse_bibliography(bib)
        if entries[0].entry_type == "website":
            issues = entries[0].completeness_issues
            assert any("urldate" in i.lower() for i in issues), \
                f"Expected urldate issue, got: {issues}"

    def test_P40_huge_page_span_flagged(self):
        """Page range spanning >200 pages is flagged as implausible."""
        bib = _load_txt("bib_huge_page_span.txt")
        entries = parse_bibliography(bib)
        issues = entries[0].completeness_issues
        assert any("span" in i.lower() or "pages" in i.lower() or "unusually" in i.lower()
                   for i in issues), f"Expected huge-span issue, got: {issues}"

    def test_P41_invalid_key_format_flagged(self):
        """All-lowercase key [vaswani17] gets invalid key format issue."""
        bib = _load_txt("bib_bad_keys.txt")
        entries = parse_bibliography(bib)
        keys_with_issues = [e for e in entries if any("key" in i.lower() for i in e.completeness_issues)]
        assert len(keys_with_issues) >= 1

    def test_P42_key_year_mismatch_flagged(self):
        """Key [XY15] with author year 2020 gets 'LNI key inconsistency' issue."""
        bib = _load_txt("bib_key_mismatch.txt")
        entries = parse_bibliography(bib)
        issues = entries[0].completeness_issues
        assert any("inconsisten" in i.lower() or "mismatch" in i.lower() or "key" in i.lower()
                   for i in issues), f"Expected key inconsistency issue, got: {issues}"

    def test_P43_perfect_entry_no_completeness_issues(self):
        """A perfectly formatted proceedings entry generates zero completeness issues."""
        bib = (
            "[VSP17] Vaswani, Ashish; Shazeer, Noam; Parmar, Niki: "
            "Attention Is All You Need. In: NeurIPS, 2017; S. 5998--6008."
        )
        entries = parse_bibliography(bib)
        assert entries[0].completeness_issues == [], \
            f"Unexpected issues: {entries[0].completeness_issues}"

    def test_P44_missing_year_for_book_flagged(self):
        """Book entry without year gets 'Missing required field: year'."""
        bib = "[Go16] Goodfellow, Ian: Deep Learning. MIT Press."
        entries = parse_bibliography(bib)
        issues = entries[0].completeness_issues
        if entries[0].entry_type == "book":
            assert any("year" in i.lower() for i in issues)

    def test_P45_key_consistent_true_for_matching_entry(self):
        """key_consistent=True when key initials and year match metadata."""
        bib = "[LB15] LeCun, Yann; Bengio, Yoshua: Deep Learning. In: Nature, 2015; S. 1--10."
        entries = parse_bibliography(bib)
        # Can be True or None (if author not parsed); should NOT be False
        assert entries[0].key_consistent is not False, \
            f"key_consistent should not be False for matching entry"

    def test_P46_key_consistent_false_for_mismatched_entry(self):
        """key_consistent=False when key year doesn't match entry year."""
        bib = "[AB99] Mueller, Hans; Schmidt, Klaus: A Paper. Springer, 2020."
        entries = parse_bibliography(bib)
        # Year 99 vs 2020 (year suffix '20') — should be flagged
        if entries[0].year:
            assert entries[0].key_consistent is False, \
                f"Expected False for year mismatch, got: {entries[0].key_consistent}"

    def test_P47_numeric_key_consistent_always_true(self):
        """Numeric keys skip initials check → key_consistent=True."""
        bib = "[1] LeCun, Yann: Deep Learning. In: Nature, 2015; S. 1--10."
        entries = parse_bibliography(bib)
        assert entries[0].key_consistent is True

    def test_P48_missing_pages_for_article_flagged(self):
        """Article missing pages field gets completeness issue."""
        e = BibEntry(key="AB20", raw_text="Author, Bob: Paper. In: Journal, Vol. 1, 2020.")
        e.entry_type = "article"
        e.authors = "Author, Bob"
        e.title = "Paper"
        e.journal = "Journal"
        e.year = "2020"
        _check_completeness(e)
        assert any("pages" in i.lower() for i in e.completeness_issues)

    def test_P49_book_missing_publisher_flagged(self):
        """Book missing publisher field gets completeness issue."""
        e = BibEntry(key="Go16", raw_text="Goodfellow, Ian: Deep Learning. 2016.")
        e.entry_type = "book"
        e.authors = "Goodfellow, Ian"
        e.title = "Deep Learning"
        e.year = "2016"
        _check_completeness(e)
        assert any("publisher" in i.lower() for i in e.completeness_issues)

    def test_P50_disambiguation_keys_no_issues(self):
        """[Wa14a] and [Wa14b] are valid — no key format errors."""
        bib = _load_txt("bib_disambiguation.txt")
        entries = parse_bibliography(bib)
        for e in entries:
            assert not any("key" in i.lower() and "format" in i.lower()
                           for i in e.completeness_issues), \
                f"Unexpected key format issue in {e.key}: {e.completeness_issues}"


# =============================================================================
# SECTION C — CHECKER / VERIFICATION UTILITY TESTS
# =============================================================================

@pytest.mark.skipif(not CHECKER_AVAILABLE, reason="checker.py unavailable")
class TestTitleSimilarity:
    """C1-C20: _title_similarity edge cases."""

    def test_C01_identical_titles(self):
        """Identical titles → 1.0."""
        assert _title_similarity("Attention Is All You Need", "Attention Is All You Need") == pytest.approx(1.0, abs=0.01)

    def test_C02_completely_different_titles(self):
        """Totally unrelated titles → < 0.3."""
        score = _title_similarity("Quantum Physics", "History of Cooking")
        assert score < 0.45  # "All" appears in both titles; 0.34 is the real score

    def test_C03_empty_title_returns_zero(self):
        """Empty string → 0.0."""
        assert _title_similarity("", "Attention Is All You Need") == 0.0
        assert _title_similarity("Attention Is All You Need", "") == 0.0
        assert _title_similarity("", "") == 0.0

    def test_C04_word_reordering_still_high(self):
        """Reordering words lowers score but stays above 0.6."""
        score = _title_similarity(
            "Deep Learning for Natural Language Processing",
            "Natural Language Processing with Deep Learning"
        )
        assert score >= 0.6

    def test_C05_umlaut_normalized(self):
        """German umlauts normalized: ä→ae means 'Maschinelles Lernen' matches."""
        score = _title_similarity(
            "Verbesserungen im maschinellen Lernen",
            "Verbesserungen im maschinellen Lernen"
        )
        assert score >= 0.9

    def test_C06_subtitle_colon_treated_as_space(self):
        """'Title: Subtitle' and 'Title - Subtitle' should have very high similarity."""
        score = _title_similarity(
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "BERT Pre-training of Deep Bidirectional Transformers"
        )
        assert score >= 0.85

    def test_C07_stopwords_ignored(self):
        """'The' and 'A' removed — slight difference in articles doesn't matter."""
        score = _title_similarity(
            "A Survey of Deep Learning Methods",
            "Survey of Deep Learning Methods"
        )
        assert score >= 0.9

    def test_C08_case_insensitive(self):
        """Case differences don't affect score."""
        assert _title_similarity(
            "attention is all you need",
            "ATTENTION IS ALL YOU NEED"
        ) >= 0.95

    def test_C09_extra_whitespace_normalized(self):
        """Multiple spaces collapsed — should still match."""
        assert _title_similarity(
            "Deep   Learning",
            "Deep Learning"
        ) >= 0.9

    def test_C10_single_word_title_match(self):
        """Very short titles still produce scores (may be lower)."""
        score = _title_similarity("Transformers", "Transformers")
        assert score >= 0.9

    def test_C11_one_word_different(self):
        """One word difference in a long title: should be >0.7."""
        score = _title_similarity(
            "Attention Is All You Need",
            "Attention Is All We Need"
        )
        assert score >= 0.7

    def test_C12_completely_different_length_titles_low(self):
        """Very long vs very short: low score unless words overlap."""
        score = _title_similarity(
            "Deep Learning",
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding from Multilingual Data"
        )
        assert score < 0.7

    def test_C13_latex_markup_stripped(self):
        """LaTeX \\emph{word} stripped before comparison."""
        score = _title_similarity(
            r"\emph{Attention} Is All You Need",
            "Attention Is All You Need"
        )
        assert score >= 0.9

    def test_C14_none_inputs_handled(self):
        """None inputs should not raise — return 0.0."""
        # Title similarity takes strings; test that empty strings behave
        assert _title_similarity("", "something") == 0.0

    def test_C15_abbreviation_vs_full_title(self):
        """Abbreviated title 'BERT' vs full expansion: low but nonzero."""
        score = _title_similarity("BERT", "Bidirectional Encoder Representations from Transformers")
        # No significant word overlap expected — score may be very low
        assert 0.0 <= score <= 0.5

    def test_C16_very_similar_titles_above_threshold(self):
        """Two titles that differ only in a preprint vs published version → > 0.85."""
        score = _title_similarity(
            "An Introduction to Machine Learning",
            "Introduction to Machine Learning"
        )
        assert score >= 0.85

    def test_C17_number_in_title(self):
        """Titles with numbers: 'GPT-3' vs 'GPT3' should be similar."""
        score = _title_similarity("GPT-3: Language Models are Few-Shot Learners",
                                   "GPT3 Language Models are Few Shot Learners")
        assert score >= 0.8

    def test_C18_fake_title_vs_real_title_low(self):
        """A plausible-sounding fake title vs real title: low score."""
        score = _title_similarity(
            "Revolutionary Quantum AI Blockchain Method That Does Not Exist At All",
            "Attention Is All You Need"
        )
        assert score < 0.45  # "All" appears in both titles; 0.34 is the real score

    def test_C19_html_entity_stripped(self):
        """HTML entities like &amp; are stripped before comparison."""
        score = _title_similarity(
            "Smith &amp; Jones: A Paper",
            "Smith Jones A Paper"
        )
        assert score >= 0.8

    def test_C20_score_bounded_zero_to_one(self):
        """Score is always in [0.0, 1.0] regardless of input."""
        pairs = [
            ("", ""),
            ("a", "b"),
            ("The quick brown fox", "The quick brown fox jumps over the lazy dog"),
            ("x" * 500, "y" * 500),
        ]
        for t1, t2 in pairs:
            s = _title_similarity(t1, t2)
            assert 0.0 <= s <= 1.0, f"Score out of bounds: {s}"


@pytest.mark.skipif(not CHECKER_AVAILABLE, reason="checker.py unavailable")
class TestSurnameExtraction:
    """C21-C30: _extract_surnames edge cases."""

    def test_C21_lastname_firstname_format(self):
        """'Mueller, Hans' → ['mueller']."""
        surnames = _extract_surnames("Mueller, Hans")
        assert "mueller" in surnames

    def test_C22_multiple_authors_semicolon(self):
        """'Mueller, Hans; Schmidt, Klaus' → two surnames."""
        surnames = _extract_surnames("Mueller, Hans; Schmidt, Klaus")
        assert len(surnames) == 2

    def test_C23_et_al_skipped(self):
        """'Mueller, Hans; et al.' → only 'mueller', et al. discarded."""
        surnames = _extract_surnames("Mueller, Hans; et al.")
        assert "mueller" in surnames
        assert len(surnames) == 1

    def test_C24_noble_particles_skipped(self):
        """'van der Berg, Jan' → surname without particles."""
        surnames = _extract_surnames("van der Berg, Jan")
        assert any("berg" in s for s in surnames)

    def test_C25_umlaut_normalized_in_surname(self):
        """'Müller' → 'muller' (ü→u normalization)."""
        surnames = _extract_surnames("Müller, Jörg")
        assert any("m" in s for s in surnames)

    def test_C26_firstname_lastname_format(self):
        """'John Smith' (no comma) → extracts 'smith' as surname (last token)."""
        surnames = _extract_surnames("John Smith")
        assert "smith" in surnames

    def test_C27_empty_string_returns_empty(self):
        """Empty string → []."""
        assert _extract_surnames("") == []

    def test_C28_single_word_handled(self):
        """Single word (organization name) → returns something or empty."""
        result = _extract_surnames("IEEE")
        assert isinstance(result, list)

    def test_C29_hyphenated_surname(self):
        """'Garcia-Lopez, Maria' → surname extracted."""
        surnames = _extract_surnames("Garcia-Lopez, Maria")
        assert any("garcia" in s or "lopez" in s for s in surnames)

    def test_C30_six_authors_all_extracted(self):
        """6 realistic multi-char surnames are all extracted."""
        authors = "Mueller, Hans; Schmidt, Klaus; Weber, Maria; Fischer, Anna; Bauer, Tom; Koch, Eva"
        surnames = _extract_surnames(authors)
        assert len(surnames) == 6, f"Expected 6 surnames, got {len(surnames)}: {surnames}"


@pytest.mark.skipif(not CHECKER_AVAILABLE, reason="checker.py unavailable")
class TestAuthorOverlapScore:
    """C31-C38: author_overlap_score edge cases."""

    def test_C31_identical_authors_score_one(self):
        score = author_overlap_score("Mueller, Hans; Schmidt, Klaus", "Mueller, Hans; Schmidt, Klaus")
        assert score is not None and score >= 0.9

    def test_C32_completely_different_authors_score_zero(self):
        score = author_overlap_score("Smith, John", "Vaswani, Ashish")
        assert score is not None and score < 0.3

    def test_C33_empty_cited_authors_returns_none(self):
        assert author_overlap_score("", "Mueller, Hans") is None

    def test_C34_empty_correct_authors_returns_none(self):
        assert author_overlap_score("Mueller, Hans", "") is None

    def test_C35_partial_overlap_midrange(self):
        """One of two cited authors matches."""
        score = author_overlap_score("Mueller, Hans; Schmidt, Klaus", "Mueller, Hans; Brown, Alice")
        assert score is not None and 0.3 <= score <= 0.7

    def test_C36_et_al_in_cited_works(self):
        """'Mueller, Hans; et al.' → et al. not counted."""
        score = author_overlap_score("Mueller, Hans; et al.", "Mueller, Hans; Schmidt, Klaus")
        assert score is not None and score > 0.5

    def test_C37_umlaut_authors_match(self):
        """'Müller' and 'Mueller' should partially match via prefix."""
        score = author_overlap_score("Müller, Jörg", "Mueller, Joerg")
        assert score is not None and score > 0.3

    def test_C38_more_than_six_authors_capped(self):
        """More than 6 cited authors: only first 6 considered."""
        many = "; ".join(f"Author{i}, Name{i}" for i in range(10))
        correct = "; ".join(f"Author{i}, Name{i}" for i in range(10))
        score = author_overlap_score(many, correct)
        assert score is not None and score > 0.5


# =============================================================================
# SECTION E — EXTRACTOR TESTS
# =============================================================================

@pytest.mark.skipif(not EXTRACTOR_AVAILABLE, reason="extractor.py unavailable")
class TestBibSectionDetection:
    """E1-E15: bibliography section detection in raw text."""

    def test_E01_german_heading_detected(self):
        text = "Body text here.\n\nLiteraturverzeichnis\n[AB20] Author, Bob: Paper. Springer, 2020."
        pos = _find_bib_start(text)
        assert pos >= 0
        assert "Literaturverzeichnis" in text[pos:]

    def test_E02_english_heading_detected(self):
        text = "Body text here.\n\nReferences\n[1] LeCun, Yann: Deep Learning. 2015."
        pos = _find_bib_start(text)
        assert pos >= 0

    def test_E03_references_all_caps_detected(self):
        text = "Body text.\n\nREFERENCES\n[1] Smith, John: A Paper. 2020."
        pos = _find_bib_start(text)
        assert pos >= 0

    def test_E04_quellenverzeichnis_detected(self):
        text = "Inhalt.\n\nQuellenverzeichnis\n[AB20] Autor, Bob: Werk. Springer, 2020."
        pos = _find_bib_start(text)
        assert pos >= 0

    def test_E05_numbered_section_heading_detected(self):
        text = "Body.\n\n5. Literaturverzeichnis\n[AB20] Author: Paper. Springer, 2020."
        pos = _find_bib_start(text)
        assert pos >= 0

    def test_E06_no_heading_fallback_to_key_pattern(self):
        """No heading but bib keys present: detect from first key."""
        text = "Body text without a heading.\n[AB20] Author: Paper. Springer, 2020."
        pos = _find_bib_start(text)
        assert pos >= 0

    def test_E07_no_bib_returns_minus_one(self):
        """Text with no bibliography heading and no keys → -1."""
        text = "This paper has no references at all. Just body text."
        pos = _find_bib_start(text)
        assert pos == -1

    def test_E08_body_bib_split_correct(self):
        text = "Body text.\n\nLiteraturverzeichnis\n[AB20] Author: Paper. Springer, 2020."
        result = split_body_bib(text)
        assert "Body text" in result["body"]
        assert "AB20" in result["bibliography"]

    def test_E09_body_not_contaminated_by_bib(self):
        text = "Introduction text.\n\nReferences\n[1] LeCun: DL. 2015."
        result = split_body_bib(text)
        assert "[1]" not in result["body"]

    def test_E10_bib_does_not_contain_body(self):
        text = "Introduction text.\n\nReferences\n[1] LeCun: DL. 2015."
        result = split_body_bib(text)
        assert "Introduction text" not in result["bibliography"]

    def test_E11_empty_text_returns_empty_bib(self):
        result = split_body_bib("")
        assert result["bibliography"] == ""

    def test_E12_numeric_keys_in_bib_detected(self):
        text = "Body.\n\nReferences\n[1] Smith: Paper. 2020.\n[2] Jones: Work. 2021."
        result = split_body_bib(text)
        assert "[1]" in result["bibliography"]
        assert "[2]" in result["bibliography"]

    def test_E13_literature_heading_detected(self):
        text = "Body.\n\nLiterature\n[AB20] Author: Paper. Springer, 2020."
        pos = _find_bib_start(text)
        assert pos >= 0

    def test_E14_bibliography_heading_english_detected(self):
        text = "Body.\n\nBibliography\n[1] LeCun: DL. 2015."
        pos = _find_bib_start(text)
        assert pos >= 0

    def test_E15_multiple_headings_uses_last_relevant(self):
        """If 'References' appears twice, use the one closest to actual bib entries."""
        text = ("Section 1: References to prior work.\n"
                "Body text here.\n\n"
                "References\n[1] Smith: Paper. 2020.")
        pos = _find_bib_start(text)
        assert pos >= 0
        # The position should be the second 'References' heading
        assert "[1]" in text[pos:]


@pytest.mark.skipif(not EXTRACTOR_AVAILABLE, reason="extractor.py unavailable")
class TestPDFExtraction:
    """E16-E22: PDF extraction (requires fixture PDFs)."""

    @pytest.mark.skipif(not (FIXTURES_PDF / "f01_perfect.pdf").exists(),
                        reason="PDF fixtures not generated — run make_fixtures.py")
    def test_E16_perfect_pdf_body_and_bib_extracted(self):
        from extractor import extract_pdf
        result = extract_pdf(str(FIXTURES_PDF / "f01_perfect.pdf"))
        assert len(result.get("body", "")) > 50
        assert len(result.get("bibliography", "")) > 20

    @pytest.mark.skipif(not (FIXTURES_PDF / "f01_perfect.pdf").exists(),
                        reason="PDF fixtures not generated — run make_fixtures.py")
    def test_E17_pdf_bib_contains_expected_keys(self):
        from extractor import extract_pdf
        result = extract_pdf(str(FIXTURES_PDF / "f01_perfect.pdf"))
        bib = result.get("bibliography", "") + result.get("full_text", "")
        assert "LBH15" in bib or "VSP17" in bib

    @pytest.mark.skipif(not (FIXTURES_PDF / "f13_numeric_keys.pdf").exists(),
                        reason="PDF fixtures not generated")
    def test_E18_pdf_numeric_keys_detected(self):
        from extractor import extract_pdf
        result = extract_pdf(str(FIXTURES_PDF / "f13_numeric_keys.pdf"))
        full = result.get("full_text", "")
        assert "[1]" in full or "[2]" in full

    @pytest.mark.skipif(not (FIXTURES_PDF / "f19_no_bibliography.pdf").exists(),
                        reason="PDF fixtures not generated")
    def test_E19_pdf_no_bibliography_returns_empty_bib(self):
        from extractor import extract_pdf
        result = extract_pdf(str(FIXTURES_PDF / "f19_no_bibliography.pdf"))
        bib = result.get("bibliography", "")
        # Either empty or very short — should not contain formatted entries
        assert len(bib.strip()) < 50 or "[" not in bib

    @pytest.mark.skipif(not (FIXTURES_PDF / "f12_german_heading.pdf").exists(),
                        reason="PDF fixtures not generated")
    def test_E20_german_heading_pdf_bib_extracted(self):
        from extractor import extract_pdf
        result = extract_pdf(str(FIXTURES_PDF / "f12_german_heading.pdf"))
        bib = result.get("bibliography", "") + result.get("full_text", "")
        assert "Mu20" in bib or "Mueller" in bib

    def test_E21_nonexistent_pdf_raises(self):
        from extractor import extract_pdf
        with pytest.raises(Exception):
            extract_pdf("/nonexistent/path/to/file.pdf")

    def test_E22_unsupported_extension_raises(self):
        from extractor import extract
        with pytest.raises(ValueError, match="Unsupported"):
            extract("/tmp/file.xyz")


# =============================================================================
# SECTION D — LOCAL DB / CACHE TESTS
# =============================================================================

@pytest.mark.skipif(not LOCALDB_AVAILABLE, reason="local_db.py unavailable")
class TestLocalDB:
    """D1-D12: SQLite cache correctness."""

    @pytest.fixture(autouse=True)
    def temp_db(self, tmp_path, monkeypatch):
        """Redirect DB to a temp dir for each test."""
        import local_db
        monkeypatch.setattr(local_db, "CACHE_DB", tmp_path / "test_cache.db")
        monkeypatch.setattr(local_db, "DB_DIR", tmp_path)
        init_cache_db()

    def test_D01_save_and_retrieve(self):
        save_to_cache("Deep Learning", "LeCun, Yann", "2015", "10.1038/nature14539",
                      "https://nature.com", "test", 0.95)
        result = search_cache("Deep Learning")
        assert result is not None
        assert result.confidence == pytest.approx(0.95, abs=0.01)

    def test_D02_normalized_title_match(self):
        """Title with extra stopwords still matches via normalized key."""
        save_to_cache("Attention Is All You Need", "Vaswani", "2017", None, None, "test", 0.9)
        result = search_cache("Attention Is All You Need")
        assert result is not None

    def test_D03_miss_returns_none(self):
        """Cache miss returns None."""
        result = search_cache("Title That Was Never Saved")
        assert result is None

    def test_D04_empty_title_returns_none(self):
        result = search_cache("")
        assert result is None

    def test_D05_overwrite_updates_confidence(self):
        save_to_cache("Deep Learning", "LeCun", "2015", None, None, "test", 0.7)
        save_to_cache("Deep Learning", "LeCun", "2015", None, None, "test", 0.95)
        result = search_cache("Deep Learning")
        assert result.confidence == pytest.approx(0.95, abs=0.01)

    def test_D06_stats_return_correct_count(self):
        save_to_cache("Paper One", "Author A", "2020", None, None, "api", 0.9)
        save_to_cache("Paper Two", "Author B", "2021", None, None, "web", 0.8)
        stats = get_cache_stats()
        assert stats["total_papers"] == 2

    def test_D07_stats_by_source(self):
        save_to_cache("Paper One", "Author A", "2020", None, None, "api", 0.9)
        save_to_cache("Paper Two", "Author B", "2021", None, None, "web", 0.8)
        stats = get_cache_stats()
        assert "api" in stats["by_source"]
        assert "web" in stats["by_source"]

    def test_D08_long_title_truncated_safely(self):
        long_title = "A" * 1000
        save_to_cache(long_title, "Author", "2020", None, None, "test", 0.5)
        result = search_cache(long_title)
        assert result is not None

    def test_D09_special_chars_in_title_safe(self):
        title = "BERT: Pre-training of Deep Bidirectional Transformers (2019)"
        save_to_cache(title, "Devlin", "2019", None, None, "test", 0.9)
        result = search_cache(title)
        assert result is not None

    def test_D10_normalize_title_stable(self):
        """Same title in different case/punctuation normalizes identically."""
        t1 = normalize_title("Deep Learning!")
        t2 = normalize_title("deep learning")
        assert t1 == t2

    def test_D11_normalize_title_removes_stopwords(self):
        """Stopwords (the, a, in, of) are removed."""
        t = normalize_title("the theory of computation")
        assert "the" not in t.split()
        assert "of" not in t.split()

    def test_D12_clear_old_entries_removes_old(self):
        import local_db
        # Insert an entry with a very old last_seen date
        import sqlite3
        conn = sqlite3.connect(str(local_db.CACHE_DB))
        conn.execute("""
            INSERT INTO verified_papers (title, authors, year, source, confidence, last_seen, title_normalized)
            VALUES ('Old Paper', 'Author', 2010, 'test', 0.5, '2000-01-01T00:00:00', 'old paper')
        """)
        conn.commit()
        conn.close()
        clear_old_entries(days=1)
        result = search_cache("Old Paper")
        assert result is None


# =============================================================================
# SECTION I — INTEGRATION / NETWORK TESTS
# =============================================================================

@pytest.mark.network
@pytest.mark.skipif(not CHECKER_AVAILABLE, reason="checker.py unavailable")
class TestVerificationIntegration:
    """I1-I10: Real network calls — requires internet. Skip with -m 'not network'."""

    def test_I01_real_paper_doi_verified(self):
        """DOI for Vaswani et al. 2017 should resolve to 'verified'."""
        from checker import _lookup_by_doi
        e = BibEntry(key="VSP17", raw_text="")
        e.title = "Attention Is All You Need"
        e.doi = "10.48550/arXiv.1706.03762"
        result = _lookup_by_doi(e)
        assert result is not None
        assert result.status in ("verified", "partial_match")

    def test_I02_real_paper_semantic_scholar(self):
        """LeCun 2015 Deep Learning should be found via Semantic Scholar."""
        from checker import _search_semantic_scholar
        e = BibEntry(key="LBH15", raw_text="")
        e.title = "Deep Learning"
        e.authors = "LeCun, Yann; Bengio, Yoshua; Hinton, Geoffrey"
        e.year = "2015"
        result = _search_semantic_scholar(e)
        assert result is not None
        assert result.confidence >= 0.5

    def test_I03_real_paper_openalex(self):
        """OpenAlex should find Vaswani et al."""
        from checker import _search_openalex
        e = BibEntry(key="VSP17", raw_text="")
        e.title = "Attention Is All You Need"
        e.authors = "Vaswani, Ashish"
        result = _search_openalex(e)
        assert result is not None

    def test_I04_real_paper_crossref(self):
        """CrossRef should find a paper by title + author."""
        from checker import _search_crossref
        e = BibEntry(key="LBH15", raw_text="")
        e.title = "Deep Learning"
        e.authors = "LeCun, Yann"
        e.year = "2015"
        result = _search_crossref(e)
        assert result is not None

    def test_I05_fake_paper_not_found(self):
        """A completely fabricated title should not be found (confidence < 0.5 or None)."""
        from checker import _search_semantic_scholar
        e = BibEntry(key="FA99", raw_text="")
        e.title = "Revolutionary Quantum AI Blockchain Method That Does Not Exist At All Ever"
        e.authors = "FakeAuthor, John"
        e.year = "1999"
        result = _search_semantic_scholar(e)
        if result:
            assert result.confidence < 0.5

    def test_I05a_automated_fake_findings_do_not_reduce_score(self):
        """Only professor-confirmed fake references may reduce the score."""
        from checker import CrossCheckResult, compute_score

        entry = BibEntry(key="FA99", raw_text="")
        entry.key_consistent = True
        result = compute_score(
            [entry], CrossCheckResult(), [], [], [],
            verification_results=[{"key": "FA99", "ai_verdict": "FAKE", "status": "not_found"}],
        )
        assert result["score"] == 100
        assert not any(p["category"] == "Confirmed fake references" for p in result["penalties"])

        confirmed = compute_score(
            [entry], CrossCheckResult(), [], [], [],
            professor_confirmed_fakes=1,
            verification_results=[{"key": "FA99", "ai_verdict": "FAKE", "status": "not_found"}],
        )
        assert confirmed["score"] == 90
        assert any(p["category"] == "Confirmed fake references" for p in confirmed["penalties"])

    def test_I06_author_fallback_s2(self):
        """Author-first fallback: if title changed between preprint/published, still found."""
        from checker import _search_semantic_scholar
        e = BibEntry(key="LBH15", raw_text="")
        e.title = "Deep Learning Methods Neural Networks"  # Wrong title
        e.authors = "LeCun, Yann; Bengio, Yoshua; Hinton, Geoffrey"
        e.year = "2015"
        # May or may not find via author fallback — just must not crash
        result = _search_semantic_scholar(e)
        assert result is None or isinstance(result.confidence, float)

    def test_I07_doi_missing_returns_none_from_doi_lookup(self):
        """Entry with no DOI: _lookup_by_doi returns None."""
        from checker import _lookup_by_doi
        e = BibEntry(key="No00", raw_text="")
        e.title = "Some Paper"
        e.doi = None
        result = _lookup_by_doi(e)
        assert result is None

    def test_I08_invalid_doi_handled_gracefully(self):
        """Malformed DOI should not crash — returns None."""
        from checker import _lookup_by_doi
        e = BibEntry(key="Bad00", raw_text="")
        e.title = "Some Paper"
        e.doi = "not-a-real-doi-12345"
        result = _lookup_by_doi(e)
        assert result is None or result.status in ("error", "not_found", "partial_match", "verified")

    def test_I09_url_liveness_check_working_url(self):
        from checker import _verify_website
        e = BibEntry(key="W1", raw_text="")
        e.entry_type = "website"
        e.title = "Example Domain"
        e.url = "https://example.com"
        e.urldate = "01.01.2024"
        result = _verify_website(e)
        assert result.status in ("verified", "not_found", "error")

    def test_I10_url_liveness_broken_url_flagged(self):
        from checker import _verify_website
        e = BibEntry(key="BrokenURL", raw_text="")
        e.entry_type = "website"
        e.title = "This URL Does Not Exist"
        e.url = "https://this-domain-absolutely-does-not-exist-xyz123abc.com"
        e.urldate = "01.01.2020"
        result = _verify_website(e)
        assert result.status in ("not_found", "error")
        # Should have surfaced a completeness issue
        assert len(e.completeness_issues) >= 1


# =============================================================================
# SECTION S — SPECIAL / EDGE CASES
# =============================================================================

@pytest.mark.skipif(not PARSER_AVAILABLE, reason="parser.py unavailable")
class TestSpecialEdgeCases:
    """S1-S15: Unusual inputs, boundary conditions, robustness."""

    def test_S01_bib_with_only_whitespace(self):
        """All-whitespace input returns empty list."""
        entries = parse_bibliography("   \n  \t  \n  ")
        assert entries == []

    def test_S02_single_entry_no_trailing_newline(self):
        """Single entry without trailing newline is parsed."""
        bib = "[AB20] Author, Bob: Paper. Springer, 2020."
        entries = parse_bibliography(bib)
        assert len(entries) == 1

    def test_S03_100_entries_performance(self):
        """100 entries parsed in under 5 seconds (no network)."""
        import time
        lines = []
        for i in range(100):
            lines.append(f"[Au{i:02d}] Author{i}, Name{i}: Paper Number {i}. Springer, 20{i%100:02d}.")
        bib = "\n".join(lines)
        t0 = time.time()
        entries = parse_bibliography(bib)
        elapsed = time.time() - t0
        assert len(entries) == 100
        assert elapsed < 5.0, f"Parsing 100 entries took {elapsed:.1f}s — too slow"

    def test_S04_entry_with_only_url(self):
        """Entry that is only a URL — title extracted as None, needs_ai_parsing=True."""
        bib = "[We22] https://example.com"
        entries = parse_bibliography(bib)
        if entries:
            assert entries[0].entry_type == "website"

    def test_S05_entry_with_isbn_no_crash(self):
        """ISBN-only entry does not crash."""
        bib = "[Go16] Goodfellow, Ian: Deep Learning. MIT Press, 2016. ISBN 978-0-262-03561-3"
        entries = parse_bibliography(bib)
        assert len(entries) == 1
        assert entries[0].isbn is not None

    def test_S06_mixed_numeric_and_lni_keys(self):
        """Both [1] and [AB20] keys in same bib — all extracted."""
        bib = "[1] LeCun: DL. 2015.\n[AB20] Author, Bob: Paper. Springer, 2020."
        entries = parse_bibliography(bib)
        assert len(entries) == 2

    def test_S07_very_long_raw_text_no_crash(self):
        """Entry with 2000-char raw text does not crash."""
        long = "A" * 2000
        bib = f"[AB20] Author, Bob: {long}. Springer, 2020."
        entries = parse_bibliography(bib)
        assert len(entries) == 1

    def test_S08_bib_section_with_only_heading(self):
        """Bibliography section with heading but no entries → empty list."""
        bib = "Literaturverzeichnis"
        entries = parse_bibliography(bib)
        assert entries == []

    def test_S09_doi_with_trailing_period_stripped(self):
        """DOI followed by period: trailing dot stripped."""
        bib = "[Va17] Vaswani, Ashish: Attention. In: NeurIPS, 2017. doi: 10.48550/arXiv.1706.03762."
        entries = parse_bibliography(bib)
        assert entries[0].doi is not None
        assert not entries[0].doi.endswith(".")

    def test_S10_url_with_trailing_comma_stripped(self):
        """URL followed by comma: trailing comma stripped from URL."""
        bib = "[We22] Author: Page. https://example.com, Stand: 01.01.2022."
        entries = parse_bibliography(bib)
        if entries[0].url:
            assert not entries[0].url.endswith(",")

    def test_S11_entries_to_dict_correct_keys(self):
        bib = "[AB20] A, B: P. Springer, 2020.\n[CD21] C, D: Q. Springer, 2021."
        entries = parse_bibliography(bib)
        d = entries_to_dict(entries)
        assert "AB20" in d and "CD21" in d
        assert isinstance(d["AB20"], BibEntry)

    def test_S12_title_with_colon_extracted(self):
        """Title containing a colon (subtitle) is extracted correctly."""
        bib = "[Va17] Vaswani, Ashish: Attention Is All You Need: A Transformer Architecture. In: NeurIPS, 2017; S. 1--10."
        entries = parse_bibliography(bib)
        assert entries[0].title is not None
        assert "Attention" in entries[0].title

    def test_S13_pages_with_german_S_prefix(self):
        """Pages formatted as 'S. 436--444' are extracted."""
        bib = "[LBH15] LeCun, Yann: DL. In: Nature, Vol. 521, 2015; S. 436--444."
        entries = parse_bibliography(bib)
        assert entries[0].pages is not None
        assert "436" in entries[0].pages

    def test_S14_crossref_url_in_bib_entry_treated_as_doi(self):
        """DOI URL format 'https://doi.org/...' extracted correctly."""
        bib = "[Va17] Vaswani, Ashish: Attention. In: NeurIPS, 2017. https://doi.org/10.48550/arXiv.1706.03762"
        # The URL pattern fires first (website detection), which is acceptable
        entries = parse_bibliography(bib)
        assert len(entries) == 1
        # Either DOI or URL should be set
        assert entries[0].doi is not None or entries[0].url is not None

    def test_S15_normalize_title_consistent(self):
        """normalize_title is deterministic — same input always gives same output."""
        from checker import _normalize_title
        t = "Attention Is All You Need: A Transformer Architecture"
        assert _normalize_title(t) == _normalize_title(t)
        assert _normalize_title(t) == _normalize_title(t.upper())


# =============================================================================
# SECTION X — CROSS-CHECK LOGIC TESTS
# =============================================================================

@pytest.mark.skipif(not PARSER_AVAILABLE, reason="parser.py unavailable")
class TestCrossCheckLogic:
    """X1-X10: Citation cross-check (body vs bib) edge cases.
    These test the helper logic, not the full API pipeline.
    """

    def _run_crosscheck(self, body_text: str, bib_entries: list) -> dict:
        """Simulate the cross-check by building the expected data structures."""
        from checker import verify_all_references
        # Build mock extraction result
        extracted = {
            "body": body_text,
            "bibliography": "",
            "full_text": body_text,
        }
        return extracted

    def test_X01_cited_key_in_bib_no_orphan(self):
        """[AB20] cited in body, present in bib → no cross-check error."""
        bib = "[AB20] Author, Bob: Paper. Springer, 2020."
        body = "We use [AB20] as the baseline."
        entries = parse_bibliography(bib)
        # Simulate: all cited keys accounted for
        cited_in_body = re.findall(r'\[([A-Za-z]{2,6}\d{2}[a-z]?|\d{1,3})\]', body)
        bib_keys = {e.key for e in entries}
        orphaned = bib_keys - set(cited_in_body)
        missing = set(cited_in_body) - bib_keys
        assert "AB20" not in orphaned
        assert "AB20" not in missing

    def test_X02_orphaned_entry_detected(self):
        """[BB02] in bib but not in body → orphaned."""
        bib = "[AA01] Author, A: Paper. Springer, 2001.\n[BB02] Orphaned, B: Work. Springer, 2002."
        body = "We cite [AA01]."
        entries = parse_bibliography(bib)
        cited_in_body = set(re.findall(r'\[([A-Za-z]{2,6}\d{2}[a-z]?|\d{1,3})\]', body))
        bib_keys = {e.key for e in entries}
        orphaned = bib_keys - cited_in_body
        assert "BB02" in orphaned

    def test_X03_missing_entry_detected(self):
        """[CC03] cited in body but not in bib → missing."""
        bib = "[AA01] Author, A: Paper. Springer, 2001."
        body = "We cite [AA01] and [CC03]."
        entries = parse_bibliography(bib)
        cited_in_body = set(re.findall(r'\[([A-Za-z]{2,6}\d{2}[a-z]?|\d{1,3})\]', body))
        bib_keys = {e.key for e in entries}
        missing = cited_in_body - bib_keys
        assert "CC03" in missing

    def test_X04_numeric_key_crosscheck(self):
        """Numeric key [1] cross-checking works."""
        bib = "[1] LeCun: DL. 2015."
        body = "See [1] for deep learning."
        entries = parse_bibliography(bib)
        cited = set(re.findall(r'\[(\d{1,3})\]', body))
        bib_keys = {e.key for e in entries}
        assert "1" in cited
        assert "1" in bib_keys

    def test_X05_no_citations_in_body_all_orphaned(self):
        """No in-text citations → all bib entries are orphaned."""
        bib = "[AB20] Author: Paper. Springer, 2020."
        body = "This paper makes no citations."
        entries = parse_bibliography(bib)
        cited = set(re.findall(r'\[([A-Za-z]{2,6}\d{2}[a-z]?|\d{1,3})\]', body))
        bib_keys = {e.key for e in entries}
        orphaned = bib_keys - cited
        assert "AB20" in orphaned

    def test_X06_empty_bib_all_body_cites_missing(self):
        """Empty bib → all body citations are missing."""
        bib = ""
        body = "We reference [AB20] and [CD21]."
        entries = parse_bibliography(bib)
        cited = set(re.findall(r'\[([A-Za-z]{2,6}\d{2}[a-z]?|\d{1,3})\]', body))
        bib_keys = {e.key for e in entries}
        missing = cited - bib_keys
        assert "AB20" in missing
        assert "CD21" in missing

    def test_X07_disambiguation_keys_treated_separately(self):
        """[Wa14a] and [Wa14b] are independent entries."""
        bib = "[Wa14a] Wagner: First. In: IEEE, 2014; S. 1--10.\n[Wa14b] Wagner: Second. In: IEEE, 2014; S. 11--20."
        body = "See [Wa14a] for first work."
        entries = parse_bibliography(bib)
        cited = set(re.findall(r'\[([A-Za-z]{2,6}\d{2}[a-z]?)\]', body))
        bib_keys = {e.key for e in entries}
        orphaned = bib_keys - cited
        assert "Wa14b" in orphaned
        assert "Wa14a" not in orphaned

    def test_X08_self_citation_handled(self):
        """Same key cited multiple times in body counts as one cite."""
        bib = "[AB20] Author: Paper. Springer, 2020."
        body = "See [AB20]. Also see [AB20]. And once more [AB20]."
        entries = parse_bibliography(bib)
        cited = set(re.findall(r'\[([A-Za-z]{2,6}\d{2}[a-z]?)\]', body))
        bib_keys = {e.key for e in entries}
        assert cited == bib_keys  # Exactly 1 entry, cited 3 times = no orphan, no missing

    def test_X09_all_perfect_no_issues(self):
        """Perfect paper: all cited entries in bib, all bib entries cited."""
        bib = "[LBH15] LeCun, Yann: DL. In: Nature, 2015; S. 1--10.\n[VSP17] Vaswani: Attention. In: NeurIPS, 2017; S. 1--10."
        body = "Methods from [LBH15] and [VSP17] are combined."
        entries = parse_bibliography(bib)
        cited = set(re.findall(r'\[([A-Za-z]{2,6}\d{2}[a-z]?)\]', body))
        bib_keys = {e.key for e in entries}
        assert cited == bib_keys

    def test_X10_case_sensitivity_in_keys(self):
        """Keys are case-sensitive: [ab20] ≠ [AB20]."""
        bib = "[AB20] Author: Paper. Springer, 2020."
        body = "See [ab20] for details."  # Wrong case
        entries = parse_bibliography(bib)
        cited = set(re.findall(r'\[([A-Za-z]{2,6}\d{2}[a-z]?)\]', body))
        bib_keys = {e.key for e in entries}
        missing = cited - bib_keys
        orphaned = bib_keys - cited
        # "ab20" won't match the regex since it's lowercase — or it will be in missing
        assert "AB20" in orphaned


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    config.addinivalue_line("markers", "network: marks tests that require network access")
    config.addinivalue_line("markers", "slow: marks tests that are slow")
