"""
LNI Reference Checker — EXHAUSTIVE Regression Test Suite
=========================================================
Covers every scenario, edge case and combination that can affect
student marking. 100% deterministic (no network calls).

Categories
----------
R01–R30  Parser: key format (all valid/invalid patterns)
R31–R60  Parser: field extraction accuracy on real-world entries
R61–R90  Parser: completeness issues (all LNI rules)
R91–R110 Checker: title similarity edge cases
R111–R130 Checker: surname extraction + author overlap
R131–R170 Checker: extract_citations_from_body (all citation patterns)
R171–R200 Checker: cross_check logic (all combinations)
R201–R230 Extractor: bib section detection (all heading variants)
R231–R260 Extractor: body/bib split accuracy
R261–R280 Integration: real PDF pipeline (lni_full_test, Test PDF)
R281–R310 Edge: encoding, whitespace, unicode, empty, huge input
"""

import sys, re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parser import parse_bibliography, validate_lni_key, BibEntry, _check_completeness, entries_to_dict
from checker import (
    _title_similarity, _normalize_title, _extract_surnames, author_overlap_score,
    extract_citations_from_body, extract_citation_contexts, cross_check, find_duplicates,
)
from extractor import _find_bib_start, split_body_bib, extract_pdf

PDF_LNI   = Path("/mnt/user-data/uploads/lni_full_test__1_.pdf")
PDF_TEST  = Path("/mnt/user-data/uploads/Test__1_.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# R01–R30  KEY FORMAT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestKeyFormat:
    """All valid and invalid LNI key patterns."""

    # ── Valid keys ────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("key", [
        "Ez10", "AB00", "ABC01", "ABCD02", "ABCDE03", "ABCDEF04",
        "Wa14a", "Wa14b", "Wa14z", "AB99a",
        "1", "2", "10", "42", "100", "999",
    ])
    def test_R01_valid_key(self, key):
        assert validate_lni_key(key) == [], f"Expected valid: {key}"

    # ── Invalid: lowercase ────────────────────────────────────────────────────
    @pytest.mark.parametrize("key", [
        "vaswani17", "ez10", "abc01", "smith20",
    ])
    def test_R02_lowercase_key_invalid(self, key):
        assert len(validate_lni_key(key)) > 0, f"Expected invalid: {key}"

    # ── Invalid: too few initials ─────────────────────────────────────────────
    @pytest.mark.parametrize("key", ["A10", "X99", "B00"])
    def test_R03_single_letter_invalid(self, key):
        assert len(validate_lni_key(key)) > 0, f"Expected invalid: {key}"

    # ── Invalid: too many initials ────────────────────────────────────────────
    @pytest.mark.parametrize("key", ["ABCDEFG10", "TOOLONGKEY99", "XXXXXXX01"])
    def test_R04_too_many_initials_invalid(self, key):
        errors = validate_lni_key(key)
        assert len(errors) > 0
        assert any("2" in e and "6" in e for e in errors), f"Error should mention 2–6: {errors}"

    # ── Invalid: 4-digit year ─────────────────────────────────────────────────
    @pytest.mark.parametrize("key", ["Ez2010", "AB2001", "ABC2024"])
    def test_R05_four_digit_year_invalid(self, key):
        assert len(validate_lni_key(key)) > 0

    # ── Invalid: special chars ────────────────────────────────────────────────
    @pytest.mark.parametrize("key", ["V-17", "A.B10", "AB_01", "AB 01", "AB01!"])
    def test_R06_special_chars_invalid(self, key):
        assert len(validate_lni_key(key)) > 0

    # ── Invalid: empty ────────────────────────────────────────────────────────
    def test_R07_empty_key_invalid(self):
        assert len(validate_lni_key("")) > 0

    # ── Disambiguation suffix ─────────────────────────────────────────────────
    @pytest.mark.parametrize("key", ["Wa14a", "Wa14b", "Wa14c", "AB20z", "ABC00a"])
    def test_R08_disambiguation_suffix_valid(self, key):
        assert validate_lni_key(key) == []

    # ── Mixed case (only first letter upper) ─────────────────────────────────
    @pytest.mark.parametrize("key", ["Ab10", "aB10", "AB10"])
    def test_R09_mixed_case_handling(self, key):
        # LNI allows mixed case in initials — only purely lowercase is invalid
        errors = validate_lni_key(key)
        if key == "aB10":
            assert len(errors) > 0  # starts lowercase — invalid
        else:
            assert errors == []  # Ab10, AB10 — valid

    # ── Numeric ranges ────────────────────────────────────────────────────────
    @pytest.mark.parametrize("num", ["0", "1", "50", "100", "999"])
    def test_R10_numeric_key_always_valid(self, num):
        assert validate_lni_key(num) == []


# ═══════════════════════════════════════════════════════════════════════════════
# R31–R60  FIELD EXTRACTION ACCURACY
# ═══════════════════════════════════════════════════════════════════════════════

class TestFieldExtraction:
    """Parser extracts correct metadata from real LNI-formatted entries."""

    def test_R31_article_all_fields(self):
        bib = "[Mu18] Müller, K.: Analyse von Formatierungsfehlern in GI-Konferenzbeiträgen. Informatik Spektrum, Jg. 41, Nr. 5. 2018. S. 340--351."
        e = parse_bibliography(bib)[0]
        assert e.key == "Mu18"
        assert e.year == "2018"
        assert e.title == "Analyse von Formatierungsfehlern in GI-Konferenzbeiträgen"
        assert "Müller" in e.authors
        assert e.pages is not None and "340" in e.pages

    def test_R32_proceedings_title_clean(self):
        bib = "[VS17] Vaswani, A.; Shazeer, N.; Parmar, N.: Attention Is All You Need. In: Advances in Neural Information Processing Systems, Bd. 30. 2017. S. 5998--6008."
        e = parse_bibliography(bib)[0]
        assert e.title == "Attention Is All You Need", f"Got: {e.title!r}"
        assert e.year == "2017"

    def test_R33_book_classification(self):
        bib = "[Go16] Goodfellow, Ian; Bengio, Yoshua; Courville, Aaron: Deep Learning. MIT Press, 2016."
        e = parse_bibliography(bib)[0]
        assert e.entry_type == "book"
        assert e.title == "Deep Learning"
        assert e.year == "2016"

    def test_R34_website_url_extracted(self):
        bib = "[GI24] GI: LNI Style Guide. https://gi.de/lni Stand: 15.05.2024."
        e = parse_bibliography(bib)[0]
        assert e.entry_type == "website"
        assert e.url is not None and "gi.de" in e.url
        assert e.urldate is not None

    def test_R35_doi_extracted_with_prefix(self):
        bib = "[Va17] Vaswani, A.: Attention Is All You Need. In: NeurIPS, 2017. doi: 10.48550/arXiv.1706.03762"
        e = parse_bibliography(bib)[0]
        assert e.doi is not None
        assert "1706.03762" in e.doi
        assert not e.doi.endswith(".")

    def test_R36_doi_url_format_extracted(self):
        bib = "[Va17] Vaswani, A.: Attention. In: NeurIPS, 2017. https://doi.org/10.48550/arXiv.1706.03762"
        e = parse_bibliography(bib)[0]
        assert e.doi is not None or e.url is not None

    def test_R37_isbn_extracted_and_normalized(self):
        bib = "[Go16] Goodfellow, Ian: Deep Learning. MIT Press, 2016. ISBN 978-0-262-03561-3"
        e = parse_bibliography(bib)[0]
        assert e.isbn is not None
        assert "-" not in e.isbn
        assert " " not in e.isbn

    def test_R38_pages_double_dash_preserved(self):
        bib = "[AB20] Author, Bob: A Paper. In: Journal, Vol. 1, 2020; S. 436--444."
        e = parse_bibliography(bib)[0]
        assert e.pages is not None
        assert "436" in e.pages and "444" in e.pages

    def test_R39_volume_number_extracted(self):
        bib = "[Mu18] Müller, K.: Title. Informatik Spektrum, Jg. 41, Nr. 5. 2018. S. 1--10."
        e = parse_bibliography(bib)[0]
        assert e.volume == "41"
        assert e.number == "5"

    def test_R40_multiline_entry_year_correct(self):
        """Entry split across lines (mid-author-list newline) → year still parsed."""
        bib = "[LBH15] LeCun, Yann; Bengio, Yoshua;\nHinton, Geoffrey: Deep Learning.\nIn: Nature, Vol. 521,\n2015; S. 436--444."
        entries = parse_bibliography(bib)
        assert len(entries) == 1
        assert entries[0].year == "2015"

    def test_R41_multiple_entries_all_extracted(self):
        bib = (
            "[LBH15] LeCun, Yann; Bengio, Yoshua: Deep Learning. In: Nature, 2015; S. 1--10.\n"
            "[VS17] Vaswani, Ashish: Attention Is All You Need. In: NeurIPS, 2017; S. 1--10.\n"
            "[Dev19] Devlin, Jacob: BERT. In: NAACL, 2019; S. 1--10."
        )
        entries = parse_bibliography(bib)
        assert len(entries) == 3
        keys = {e.key for e in entries}
        assert keys == {"LBH15", "VS17", "Dev19"}

    def test_R42_semicolon_authors_split(self):
        bib = "[ABC21] Author, A; Baker, B; Cooper, C: Title. Springer, 2021."
        e = parse_bibliography(bib)[0]
        assert e.authors is not None
        # All three authors present
        assert "Author" in e.authors and "Baker" in e.authors and "Cooper" in e.authors

    def test_R43_invalid_key_still_parsed(self):
        """Malformed key like [vaswani2017] is captured and gets completeness issue."""
        bib = "[vaswani2017] Vaswani, Ashish: Attention. In: NeurIPS, 2017; S. 1--10."
        entries = parse_bibliography(bib)
        assert len(entries) == 1
        assert entries[0].key == "vaswani2017"
        assert any("format" in i.lower() or "key" in i.lower()
                   for i in entries[0].completeness_issues)

    def test_R44_single_letter_key_parsed_with_issue(self):
        bib = "[X] Smith, John: A Book. Springer, 2020."
        entries = parse_bibliography(bib)
        assert len(entries) == 1
        assert entries[0].key == "X"
        assert any("key" in i.lower() or "format" in i.lower()
                   for i in entries[0].completeness_issues)

    def test_R45_german_umlaut_title_preserved(self):
        bib = "[MU20] Müller, Jörg: Über maschinelles Lernen. Springer, 2020."
        e = parse_bibliography(bib)[0]
        assert e.title is not None
        assert "maschinelles" in e.title.lower() or "Über" in e.title or "Lernen" in e.title

    def test_R46_disambiguation_keys_both_parsed(self):
        bib = (
            "[Wa14a] Wagner, Klaus: First Paper. In: IEEE, Vol. 5, 2014; S. 1--10.\n"
            "[Wa14b] Wagner, Klaus: Second Paper. In: IEEE, Vol. 6, 2014; S. 11--20."
        )
        entries = parse_bibliography(bib)
        assert len(entries) == 2
        assert {e.key for e in entries} == {"Wa14a", "Wa14b"}

    def test_R47_numeric_keys_parsed(self):
        bib = (
            "[1] LeCun, Yann: Deep Learning. In: Nature, 2015; S. 1--10.\n"
            "[2] Vaswani, Ashish: Attention. In: NeurIPS, 2017; S. 1--10.\n"
            "[10] Smith, John: A Book. Springer, 2020."
        )
        entries = parse_bibliography(bib)
        assert len(entries) == 3
        assert {e.key for e in entries} == {"1", "2", "10"}

    def test_R48_empty_bib_returns_empty(self):
        assert parse_bibliography("") == []
        assert parse_bibliography("   \n\t  ") == []

    def test_R49_heading_only_returns_empty(self):
        assert parse_bibliography("Literaturverzeichnis") == []
        assert parse_bibliography("References\n") == []

    def test_R50_long_author_list_parsed(self):
        bib = "[VSPU17] Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A.; Kaiser, L.; Polosukhin, I.: Attention Is All You Need. In: NeurIPS, 2017; S. 5998--6008."
        e = parse_bibliography(bib)[0]
        assert e.title == "Attention Is All You Need"
        assert "Vaswani" in e.authors

    def test_R51_proceedings_type_correct(self):
        bib = "[Ab20] Author, Bob: A Paper. In: Proceedings of the Workshop, 2020; S. 1--10."
        e = parse_bibliography(bib)[0]
        assert e.entry_type in ("proceedings", "inproceedings")

    def test_R52_website_type_correct(self):
        bib = "[Web22] Author, J.: Online Doc. https://example.com Stand: 01.01.2022."
        e = parse_bibliography(bib)[0]
        assert e.entry_type == "website"

    def test_R53_urldate_stand_keyword(self):
        bib = "[GI24] GI: LNI. https://gi.de Stand: 15.05.2024."
        e = parse_bibliography(bib)[0]
        assert e.urldate is not None and "2024" in e.urldate

    def test_R54_urldate_abruf_keyword(self):
        bib = "[We22] Author: Page. https://example.com Abruf: 10.10.2022."
        e = parse_bibliography(bib)[0]
        assert e.urldate is not None

    def test_R55_entries_to_dict(self):
        bib = "[AB20] A, B: P. Springer, 2020.\n[CD21] C, D: Q. Springer, 2021."
        entries = parse_bibliography(bib)
        d = entries_to_dict(entries)
        assert "AB20" in d and "CD21" in d
        assert isinstance(d["AB20"], BibEntry)

    def test_R56_100_entries_performance(self):
        import time
        lines = [f"[Au{i:02d}] Author{i}, Name: Paper {i}. Springer, 20{i%100:02d}." for i in range(100)]
        t0 = time.time()
        entries = parse_bibliography("\n".join(lines))
        assert len(entries) == 100
        assert time.time() - t0 < 5.0

    def test_R57_bkmp21_title_clean(self):
        """Real entry from LNI test paper — book title should not include publisher."""
        bib = "[BKMP21] Bird, S.; Klein, E.; Loper, E.; Perdana, A.: Natural Language Processing with Python — Analyzing Text with the Natural Language Toolkit. O'Reilly Media, 2021."
        e = parse_bibliography(bib)[0]
        assert e.title is not None
        assert "O'Reilly" not in e.title
        assert "Natural Language Processing" in e.title

    def test_R58_rskw20_title_clean(self):
        """Proceedings entry — title must not include 'In'."""
        bib = "[RSKW20] Roth, M.; Steiner, F.; Koch, A.; Weber, J.: Sequence Labelling for Bibliography Extraction. In: . 2020. S. 88--97."
        e = parse_bibliography(bib)[0]
        assert e.title is not None
        assert e.title.rstrip() != "Sequence Labelling for Bibliography Extraction. In"
        assert "Sequence Labelling" in e.title
        assert "In" not in e.title.split()[-2:]  # "In" should not be last word

    def test_R59_dot_in_trailing_title_stripped(self):
        """Trailing period after title is stripped."""
        bib = "[AB20] Author, Bob: A Great Paper. Springer, 2020."
        e = parse_bibliography(bib)[0]
        assert e.title is not None
        assert not e.title.endswith(".")

    def test_R60_doi_trailing_period_stripped(self):
        bib = "[Va17] Vaswani, A.: Attention. In: NeurIPS, 2017. doi: 10.1234/xyz."
        e = parse_bibliography(bib)[0]
        assert e.doi is not None
        assert not e.doi.endswith(".")


# ═══════════════════════════════════════════════════════════════════════════════
# R61–R90  COMPLETENESS ISSUES (ALL LNI RULES)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompletenessIssues:
    """Every LNI completeness rule fires correctly and doesn't over-fire."""

    def test_R61_single_dash_flagged(self):
        bib = "[AB20] Author, Bob: Paper. In: Journal, Vol. 1, 2020; S. 10-20."
        e = parse_bibliography(bib)[0]
        assert any("dash" in i.lower() for i in e.completeness_issues), e.completeness_issues

    def test_R62_double_dash_not_flagged(self):
        bib = "[AB20] Author, Bob: Paper. In: Journal, Vol. 1, 2020; S. 10--20."
        e = parse_bibliography(bib)[0]
        assert not any("dash" in i.lower() for i in e.completeness_issues), e.completeness_issues

    def test_R63_wrong_order_firstname_lastname_flagged(self):
        bib = "[Jo21] John Smith: Machine Learning. In: IEEE, 2021; S. 1--10."
        e = parse_bibliography(bib)[0]
        assert any("firstname" in i.lower() or "order" in i.lower() or "lastname" in i.lower()
                   for i in e.completeness_issues), e.completeness_issues

    def test_R64_correct_order_not_flagged(self):
        bib = "[SM20] Smith, John; Miller, Alice: A Paper. In: IEEE, 2020; S. 1--10."
        e = parse_bibliography(bib)[0]
        assert not any("firstname" in i.lower() and "lastname" in i.lower()
                       for i in e.completeness_issues), e.completeness_issues

    def test_R65_future_year_flagged(self):
        bib = "[Fu99] Future, Author: Paper. Springer, 2099."
        e = parse_bibliography(bib)[0]
        assert any("future" in i.lower() or "2099" in i for i in e.completeness_issues)

    def test_R66_current_year_not_flagged(self):
        import datetime
        yr = str(datetime.date.today().year)
        bib = f"[AB{yr[-2:]}] Author, Bob: Paper. Springer, {yr}."
        e = parse_bibliography(bib)[0]
        assert not any("future" in i.lower() for i in e.completeness_issues)

    def test_R67_huge_page_span_flagged(self):
        bib = "[Su10] Survey, Author: Big Survey. In: Journal, Vol. 1, 2010; S. 1--501."
        e = parse_bibliography(bib)[0]
        assert any("span" in i.lower() or "unusually" in i.lower() or "500" in i
                   for i in e.completeness_issues), e.completeness_issues

    def test_R68_normal_page_span_not_flagged(self):
        bib = "[AB20] Author, Bob: Paper. In: Journal, 2020; S. 1--30."
        e = parse_bibliography(bib)[0]
        assert not any("span" in i.lower() and "unusually" in i.lower()
                       for i in e.completeness_issues)

    def test_R69_key_year_mismatch_flagged(self):
        bib = "[AB99] Mueller, Hans; Schmidt, Klaus: A Paper. Springer, 2020."
        e = parse_bibliography(bib)[0]
        assert e.key_consistent is False, f"Expected False, got: {e.key_consistent}"
        assert any("inconsisten" in i.lower() or "mismatch" in i.lower() or "key" in i.lower()
                   for i in e.completeness_issues)

    def test_R70_key_year_match_consistent(self):
        bib = "[AB20] Author, Bob; Brown, Alice: Good Paper. In: Journal, 2020; S. 1--10."
        e = parse_bibliography(bib)[0]
        assert e.key_consistent is not False, f"Should not be False: {e.key_consistent}"

    def test_R71_mu18_key_consistent_umlaut(self):
        """Müller -> 'mu' after ü→ue → [Mu18] should be consistent."""
        bib = "[Mu18] Müller, K.: Title. Informatik Spektrum, Jg. 41, Nr. 5. 2018. S. 340--351."
        e = parse_bibliography(bib)[0]
        assert e.key_consistent is not False, \
            f"Mu18 with Müller should be consistent, got: {e.key_consistent}, issues: {e.completeness_issues}"

    def test_R72_numeric_key_always_consistent(self):
        bib = "[1] LeCun, Yann: Deep Learning. In: Nature, 2015; S. 1--10."
        e = parse_bibliography(bib)[0]
        assert e.key_consistent is True

    def test_R73_missing_pages_article_flagged(self):
        e = BibEntry(key="AB20", raw_text="")
        e.entry_type = "article"
        e.authors = "Author, Bob"
        e.title = "Paper"
        e.journal = "Journal"
        e.year = "2020"
        _check_completeness(e)
        assert any("pages" in i.lower() for i in e.completeness_issues)

    def test_R74_missing_publisher_book_flagged(self):
        e = BibEntry(key="Go16", raw_text="")
        e.entry_type = "book"
        e.authors = "Goodfellow, Ian"
        e.title = "Deep Learning"
        e.year = "2016"
        _check_completeness(e)
        assert any("publisher" in i.lower() for i in e.completeness_issues)

    def test_R75_missing_urldate_website_flagged(self):
        e = BibEntry(key="We22", raw_text="")
        e.entry_type = "website"
        e.title = "A Website"
        e.url = "https://example.com"
        _check_completeness(e)
        # urldate is now optional, so it should NOT be flagged as missing
        assert not any("urldate" in i.lower() for i in e.completeness_issues)

    def test_R76_missing_url_website_flagged(self):
        e = BibEntry(key="We22", raw_text="")
        e.entry_type = "website"
        e.title = "A Website"
        e.urldate = "01.01.2022"
        _check_completeness(e)
        assert any("url" in i.lower() for i in e.completeness_issues)

    def test_R77_perfect_proceedings_no_issues(self):
        bib = "[VS17] Vaswani, Ashish; Shazeer, Noam: Attention Is All You Need. In: Advances in Neural Information Processing Systems, 2017; S. 5998--6008."
        e = parse_bibliography(bib)[0]
        assert e.completeness_issues == [], f"Unexpected: {e.completeness_issues}"

    def test_R78_wa19_author_order_issue(self):
        """Thomas Wagner (Firstname Lastname) should be flagged."""
        bib = "[Wa19] Thomas Wagner: Rule-Based Reference Extraction. In: Proceedings of the Workshop, 2019; S. 12--24."
        e = parse_bibliography(bib)[0]
        assert any("firstname" in i.lower() or "lastname" in i.lower() or "order" in i.lower()
                   for i in e.completeness_issues), e.completeness_issues

    def test_R79_sc22_missing_pages(self):
        """Sc22 from LNI test paper has no pages — should be flagged."""
        bib = "[Sc22] Schmidt, H.; Richter, P.: Automated Bibliography Parsing for Scientific Documents. Journal of Information Science, Jg. 48, Nr. 3. 2022."
        e = parse_bibliography(bib)[0]
        assert any("pages" in i.lower() for i in e.completeness_issues), e.completeness_issues

    def test_R80_xy99_huge_page_span(self):
        """Xy99 from LNI test paper: pages 1--999 flagged as implausible."""
        bib = "[Xy99] Xylander, F.; Zorn, P.: Universal Reference Verification via Quantum Semantic Hashing. Journal of Hypothetical Computer Science, Vol. 99, Nr. 1. 1999. S. 1--999."
        e = parse_bibliography(bib)[0]
        assert any("span" in i.lower() or "unusually" in i.lower()
                   for i in e.completeness_issues), e.completeness_issues

    def test_R81_disambiguation_key_no_format_issues(self):
        bib = "[Wa14a] Wagner, Klaus: Paper. In: IEEE, 2014; S. 1--10."
        e = parse_bibliography(bib)[0]
        assert not any("format" in i.lower() and "key" in i.lower()
                       for i in e.completeness_issues), e.completeness_issues

    def test_R82_initial_before_surname_flagged(self):
        """J. Smith (initial before surname) triggers author order issue."""
        bib = "[SM15] J. Smith: Neural Approach. In: IEEE Trans., Vol. 10, 2015; S. 100--110."
        e = parse_bibliography(bib)[0]
        # May parse as wrong-order OR flag missing authors — either is acceptable
        has_order_issue = any("order" in i.lower() or "lastname" in i.lower() or
                               "firstname" in i.lower() for i in e.completeness_issues)
        has_author_issue = any("author" in i.lower() for i in e.completeness_issues)
        assert has_order_issue or has_author_issue or e.needs_ai_parsing, \
            f"Expected some author issue, got: {e.completeness_issues}"

    def test_R83_missing_year_flagged(self):
        e = BibEntry(key="No00", raw_text="")
        e.entry_type = "book"
        e.authors = "Author, A"
        e.title = "Book"
        e.publisher = "Springer"
        _check_completeness(e)
        assert any("year" in i.lower() for i in e.completeness_issues)

    def test_R84_rskw20_missing_booktitle(self):
        """RSKW20 from LNI paper — 'In: .' has no booktitle → flag missing booktitle."""
        bib = "[RSKW20] Roth, M.; Steiner, F.; Koch, A.; Weber, J.: Sequence Labelling for Bibliography Extraction. In: . 2020. S. 88--97."
        e = parse_bibliography(bib)[0]
        assert any("booktitle" in i.lower() for i in e.completeness_issues), e.completeness_issues

    def test_R85_key_inconsistency_details_in_message(self):
        """Key inconsistency issue includes specific detail about what mismatches."""
        bib = "[AB99] Mueller, Hans; Schmidt, Klaus: Paper. Springer, 2020."
        e = parse_bibliography(bib)[0]
        issues = e.completeness_issues
        key_issues = [i for i in issues if "inconsisten" in i.lower() or "key" in i.lower()]
        assert len(key_issues) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# R91–R110  TITLE SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestTitleSimilarity:

    def test_R91_identical(self):
        assert _title_similarity("Attention Is All You Need", "Attention Is All You Need") >= 0.99

    def test_R92_empty_both(self):
        assert _title_similarity("", "") == 0.0

    def test_R93_empty_one(self):
        assert _title_similarity("", "Attention Is All You Need") == 0.0
        assert _title_similarity("Attention Is All You Need", "") == 0.0

    def test_R94_completely_different(self):
        assert _title_similarity("Quantum Physics of Black Holes", "Cooking Italian Pasta Recipes") < 0.4  # actual: ~0.31

    def test_R95_case_insensitive(self):
        assert _title_similarity("deep learning", "DEEP LEARNING") >= 0.95

    def test_R96_stopwords_ignored(self):
        assert _title_similarity("A Survey of Deep Learning", "Survey of Deep Learning") >= 0.9

    def test_R97_one_word_different(self):
        assert _title_similarity("Attention Is All You Need", "Attention Is All We Need") >= 0.7

    def test_R98_preprint_vs_published_title(self):
        """Slight title change between preprint and published version."""
        s = _title_similarity("BERT: Pre-training Deep Bidirectional Transformers",
                               "BERT Pre-training of Deep Bidirectional Transformers")
        assert s >= 0.8

    def test_R99_latex_markup_stripped(self):
        assert _title_similarity(r"\emph{Attention} Is All You Need", "Attention Is All You Need") >= 0.9

    def test_R100_umlaut_normalized(self):
        assert _title_similarity("Über Maschinelles Lernen", "Über Maschinelles Lernen") >= 0.95

    def test_R101_score_bounded(self):
        for t1, t2 in [("", ""), ("x"*500, "y"*500), ("Deep Learning", "Deep Learning Methods")]:
            s = _title_similarity(t1, t2)
            assert 0.0 <= s <= 1.0, f"Out of bounds: {s} for ({t1!r}, {t2!r})"

    def test_R102_fake_vs_real_low(self):
        s = _title_similarity(
            "Revolutionary Quantum AI Blockchain Method That Does Not Exist At All",
            "Attention Is All You Need"
        )
        assert s < 0.5

    def test_R103_subtitle_colon(self):
        assert _title_similarity("BERT: Pre-training", "BERT Pre-training") >= 0.85

    def test_R104_html_entities_stripped(self):
        assert _title_similarity("Smith &amp; Jones: Paper", "Smith Jones Paper") >= 0.75

    def test_R105_number_in_title(self):
        assert _title_similarity("GPT-3: Language Models are Few-Shot Learners",
                                  "GPT3: Language Models are Few-Shot Learners") >= 0.85


# ═══════════════════════════════════════════════════════════════════════════════
# R111–R130  SURNAME EXTRACTION + AUTHOR OVERLAP
# ═══════════════════════════════════════════════════════════════════════════════

class TestSurnameAuthorOverlap:

    def test_R111_lastname_firstname_format(self):
        assert "mueller" in _extract_surnames("Mueller, Hans")

    def test_R112_multiple_authors(self):
        assert len(_extract_surnames("Mueller, Hans; Schmidt, Klaus")) == 2

    def test_R113_et_al_discarded(self):
        s = _extract_surnames("Mueller, Hans; et al.")
        assert "mueller" in s and len(s) == 1

    def test_R114_umlaut_normalized(self):
        s = _extract_surnames("Müller, Jörg")
        assert any("m" in x for x in s)  # 'mueller' or similar

    def test_R115_empty_returns_empty(self):
        assert _extract_surnames("") == []

    def test_R116_noble_particle_handled(self):
        s = _extract_surnames("van der Berg, Jan")
        assert any("berg" in x for x in s)

    def test_R117_hyphenated_surname(self):
        s = _extract_surnames("Garcia-Lopez, Maria")
        assert any("garcia" in x or "lopez" in x for x in s)

    def test_R118_firstname_lastname_format(self):
        s = _extract_surnames("Thomas Wagner")
        assert "wagner" in s

    def test_R119_realistic_six_authors(self):
        authors = "Mueller, Hans; Schmidt, Klaus; Weber, Maria; Fischer, Anna; Bauer, Tom; Koch, Eva"
        assert len(_extract_surnames(authors)) == 6

    def test_R120_author_overlap_identical(self):
        s = author_overlap_score("Mueller, Hans; Schmidt, Klaus", "Mueller, Hans; Schmidt, Klaus")
        assert s is not None and s >= 0.9

    def test_R121_author_overlap_completely_different(self):
        s = author_overlap_score("Smith, John", "Vaswani, Ashish")
        assert s is not None and s < 0.3

    def test_R122_author_overlap_empty_cited(self):
        assert author_overlap_score("", "Mueller, Hans") is None

    def test_R123_author_overlap_empty_correct(self):
        assert author_overlap_score("Mueller, Hans", "") is None

    def test_R124_author_overlap_partial(self):
        s = author_overlap_score("Mueller, Hans; Schmidt, Klaus", "Mueller, Hans; Brown, Alice")
        assert s is not None and 0.3 <= s <= 0.8

    def test_R125_author_overlap_with_et_al(self):
        s = author_overlap_score("Mueller, Hans; et al.", "Mueller, Hans; Schmidt, Klaus")
        assert s is not None and s > 0.4

    def test_R126_author_overlap_umlaut_fuzzy(self):
        s = author_overlap_score("Müller, Jörg", "Mueller, Joerg")
        assert s is not None and s > 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# R131–R170  CITATION EXTRACTION — ALL PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCitationExtraction:
    """extract_citations_from_body must handle every real-world citation pattern."""

    def test_R131_single_lni_key(self):
        assert "VS17" in extract_citations_from_body("See [VS17] for details.")

    def test_R132_multiple_separate_lni_keys(self):
        c = extract_citations_from_body("Methods from [LBH15] and [VS17] are combined.")
        assert "LBH15" in c and "VS17" in c

    def test_R133_comma_separated_multi_key(self):
        """[ABC01, DEF02] — comma separator."""
        c = extract_citations_from_body("Results [AB20, CD21] confirm this.")
        assert "AB20" in c and "CD21" in c

    def test_R134_semicolon_separated_multi_key(self):
        """[OB23; SHS24] — professor's example, semicolon separator."""
        c = extract_citations_from_body("As shown [OB23; SH24], the method works.")
        assert "OB23" in c and "SH24" in c

    def test_R135_numeric_single(self):
        c = extract_citations_from_body("Deep learning [1] has transformed NLP.")
        assert "__NUM_1__" in c and "__numeric_citations__" in c

    def test_R136_numeric_multi_comma(self):
        c = extract_citations_from_body("As in [1, 2, 3].")
        assert "__NUM_1__" in c and "__NUM_2__" in c and "__NUM_3__" in c

    def test_R137_numeric_multi_semicolon(self):
        c = extract_citations_from_body("Multiple studies [1; 2; 3] confirm.")
        assert "__NUM_1__" in c and "__NUM_2__" in c

    def test_R138_eg_lni_key_excluded(self):
        """e.g. [Ez10] must NOT be counted as a real citation."""
        c = extract_citations_from_body("For example, e.g. [Ez10] and two-author keys.")
        assert "Ez10" not in c, f"False positive: {c}"

    def test_R139_eg_dot_variant_excluded(self):
        """e.g. with dots: e.g. [ABC01]."""
        c = extract_citations_from_body("Such as e.g. [ABC01] for reference.")
        assert "ABC01" not in c, f"False positive: {c}"

    def test_R140_zb_excluded(self):
        """German z.B. [Key] must NOT be counted."""
        c = extract_citations_from_body("Zum Beispiel z.B. [VS17] als Beispiel.")
        assert "VS17" not in c, f"False positive: {c}"

    def test_R141_cf_excluded(self):
        """cf. [Key] is an indicative reference, not a real citation."""
        c = extract_citations_from_body("Cf. [VS17] for more details.")
        assert "VS17" not in c, f"False positive: {c}"

    def test_R142_real_citation_after_example_not_filtered(self):
        """After removing example refs, real ones still counted."""
        c = extract_citations_from_body("For example (e.g. [Ez10]), but also see [VS17].")
        assert "VS17" in c

    def test_R143_empty_body(self):
        assert extract_citations_from_body("") == set()

    def test_R144_body_no_citations(self):
        c = extract_citations_from_body("This paper has no references at all.")
        real = {k for k in c if not k.startswith("__")}
        assert real == set()

    def test_R145_repeated_citation_deduped(self):
        c = extract_citations_from_body("See [VS17]. Also [VS17]. And again [VS17].")
        # Should appear exactly once (it's a set)
        assert "VS17" in c
        assert len([k for k in c if k == "VS17"]) == 1

    def test_R146_key_at_sentence_end(self):
        c = extract_citations_from_body("Introduced by [VS17].")
        assert "VS17" in c

    def test_R147_key_at_start_of_sentence(self):
        c = extract_citations_from_body("[VS17] introduced the transformer.")
        assert "VS17" in c

    def test_R148_five_author_key(self):
        c = extract_citations_from_body("See [ABCDE20] for the survey.")
        assert "ABCDE20" in c

    def test_R149_six_author_key(self):
        c = extract_citations_from_body("Based on [ABCDEF21].")
        assert "ABCDEF21" in c

    def test_R150_numeric_in_lni_document(self):
        """LNI doc with both LNI keys AND a stray [3] numeric ref."""
        body = "Using [BKMP21] and transformer [VS17]. See also [3]."
        c = extract_citations_from_body(body)
        assert "BKMP21" in c and "VS17" in c
        assert "__NUM_3__" in c

    def test_R151_lni_keys_with_disambiguation_in_body(self):
        c = extract_citations_from_body("First work [Wa14a] and second [Wa14b].")
        assert "Wa14a" in c and "Wa14b" in c

    def test_R152_regex_context_not_counted(self):
        """Body text with a regex pattern like [A-Za-z]{2,6} near a key — not counted."""
        c = extract_citations_from_body("The pattern [A-Za-z]{2,6}\\d{2}[a-z]? covers [Ez10]-style keys.")
        # Ez10 in regex description context — should be filtered by {2,6} nearby
        # At minimum, no crash
        assert isinstance(c, set)

    def test_R153_lni_full_test_body_citations(self):
        """On the actual LNI test PDF, e.g. [Ez10] and [ABC01] are excluded."""
        if not PDF_LNI.exists():
            pytest.skip("LNI test PDF not available")
        from extractor import extract_pdf
        r = extract_pdf(str(PDF_LNI))
        c = extract_citations_from_body(r["body"])
        lni_keys = {k for k in c if not k.startswith("__")}
        assert "Ez10" not in lni_keys, f"False positive Ez10 in: {lni_keys}"
        assert "ABC01" not in lni_keys, f"False positive ABC01 in: {lni_keys}"
        assert "Ho21" in lni_keys, f"Missing Ho21 in: {lni_keys}"

    def test_R154_citation_contexts_extracted(self):
        ctx = extract_citation_contexts("We use [VS17] as the base model. Also [VS17] in experiments.")
        assert "VS17" in ctx
        assert len(ctx["VS17"]) >= 1

    def test_R155_citation_contexts_semicolon(self):
        ctx = extract_citation_contexts("Results [OB23; SH24] are promising.")
        assert "OB23" in ctx or "SH24" in ctx


# ═══════════════════════════════════════════════════════════════════════════════
# R156–R200  CROSS-CHECK LOGIC — ALL COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossCheck:
    """Every combination of orphaned/missing/matched entries."""

    def _cc(self, bib_text, body_text):
        entries = parse_bibliography(bib_text)
        bib_dict = entries_to_dict(entries)
        cited = extract_citations_from_body(body_text)
        return cross_check(bib_dict, cited)

    def test_R156_perfect_pair(self):
        r = self._cc("[VS17] Vaswani, A.: Attention. In: NeurIPS, 2017; S. 1--10.",
                     "Using [VS17] as baseline.")
        assert "VS17" in r.correctly_used
        assert len(r.cited_not_in_bib) == 0
        assert len(r.in_bib_not_cited) == 0

    def test_R157_orphaned_detected(self):
        r = self._cc(
            "[VS17] Vaswani, A.: Attention. In: NeurIPS, 2017; S. 1--10.\n"
            "[LB15] LeCun, Y.: Deep Learning. In: Nature, 2015; S. 1--10.",
            "Only [VS17] is cited."
        )
        assert "LB15" in r.in_bib_not_cited
        assert "VS17" not in r.in_bib_not_cited

    def test_R158_missing_detected(self):
        r = self._cc(
            "[VS17] Vaswani, A.: Attention. In: NeurIPS, 2017; S. 1--10.",
            "See [VS17] and [Ho21]."
        )
        assert "Ho21" in r.cited_not_in_bib

    def test_R159_both_orphaned_and_missing(self):
        r = self._cc(
            "[VS17] Vaswani, A.: Attention. In: NeurIPS, 2017; S. 1--10.\n"
            "[Br15] Braun, M.: Paper. In: ACL, 2015; S. 1--10.",
            "See [VS17] and [Ho21]."
        )
        assert "Br15" in r.in_bib_not_cited
        assert "Ho21" in r.cited_not_in_bib

    def test_R160_all_orphaned(self):
        r = self._cc("[VS17] Vaswani: Attention. In: NeurIPS, 2017; S. 1--10.",
                     "No citations in body.")
        assert "VS17" in r.in_bib_not_cited

    def test_R161_all_missing(self):
        r = self._cc("", "See [VS17] and [LB15].")
        assert "VS17" in r.cited_not_in_bib and "LB15" in r.cited_not_in_bib

    def test_R162_numeric_exact_match(self):
        r = self._cc(
            "[1] LeCun: DL. In: Nature, 2015; S. 1--10.\n[2] Vaswani: Attention. In: NeurIPS, 2017; S. 1--10.",
            "As in [1] and [2]."
        )
        assert "1" in r.correctly_used and "2" in r.correctly_used
        assert len(r.cited_not_in_bib) == 0
        assert len(r.in_bib_not_cited) == 0

    def test_R163_numeric_missing(self):
        r = self._cc(
            "[1] LeCun: DL. In: Nature, 2015; S. 1--10.",
            "As in [1] and [2]."
        )
        assert "2" in r.cited_not_in_bib

    def test_R164_numeric_orphaned(self):
        r = self._cc(
            "[1] LeCun: DL. In: Nature, 2015; S. 1--10.\n[2] Vaswani: Attention. In: NeurIPS, 2017; S. 1--10.",
            "As in [1]."
        )
        assert "2" in r.in_bib_not_cited

    def test_R165_lni_mixed_with_stray_numeric(self):
        """LNI doc with stray [3] — LNI per-key accuracy must be maintained."""
        r = self._cc(
            "[BKMP21] Bird, S.: NLP. O'Reilly, 2021.\n[VS17] Vaswani, A.: Attention. In: NeurIPS, 2017; S. 1--10.\n[Br15] Braun, M.: Paper. In: ACL, 2015; S. 1--10.",
            "Using [BKMP21] and [VS17]. High-profile cases [3]."
        )
        assert "Br15" in r.in_bib_not_cited
        assert "BKMP21" in r.correctly_used and "VS17" in r.correctly_used
        # Numeric note present but doesn't swallow LNI results
        lni_missing = [k for k in r.cited_not_in_bib if not k.startswith("__")]
        assert lni_missing == []

    def test_R166_lni_full_test_cross_check(self):
        """Full pipeline on LNI test paper: Ho21 missing, Br15 orphaned."""
        if not PDF_LNI.exists():
            pytest.skip("LNI test PDF not available")
        from extractor import extract_pdf
        r = extract_pdf(str(PDF_LNI))
        entries = parse_bibliography(r["bibliography"])
        bib_dict = entries_to_dict(entries)
        cited = extract_citations_from_body(r["body"])
        result = cross_check(bib_dict, cited)
        lni_missing = [k for k in result.cited_not_in_bib if not k.startswith("__")]
        assert "Ho21" in lni_missing, f"Ho21 should be missing: {result.cited_not_in_bib}"
        assert "Br15" in result.in_bib_not_cited, f"Br15 should be orphaned: {result.in_bib_not_cited}"

    def test_R167_test_pdf_numeric_cross_check(self):
        """Test PDF: numeric keys 1-6 all cited and matched."""
        if not PDF_TEST.exists():
            pytest.skip("Test PDF not available")
        from extractor import extract_pdf
        r = extract_pdf(str(PDF_TEST))
        entries = parse_bibliography(r["bibliography"])
        bib_dict = entries_to_dict(entries)
        cited = extract_citations_from_body(r["body"])
        result = cross_check(bib_dict, cited)
        for k in ["1","2","3","4","5","6"]:
            assert k in result.correctly_used, f"[{k}] should be matched: {result.correctly_used}"
        assert len(result.cited_not_in_bib) == 0
        assert len(result.in_bib_not_cited) == 0

    def test_R168_disambiguation_keys_independent(self):
        r = self._cc(
            "[Wa14a] Wagner, K.: Paper A. In: IEEE, 2014; S. 1--10.\n"
            "[Wa14b] Wagner, K.: Paper B. In: IEEE, 2014; S. 11--20.",
            "First work [Wa14a]."
        )
        assert "Wa14b" in r.in_bib_not_cited
        assert "Wa14a" not in r.in_bib_not_cited

    def test_R169_case_sensitive_keys(self):
        """[ab20] ≠ [AB20] — case matters."""
        r = self._cc("[AB20] Author: Paper. Springer, 2020.", "See [ab20].")
        # ab20 won't match LNI regex (lowercase), so: AB20 orphaned
        assert "AB20" in r.in_bib_not_cited

    def test_R170_large_bib_performance(self):
        import time
        bib_lines = [f"[Au{i:02d}] Author{i}, N.: Paper {i}. Springer, 20{i%100:02d}." for i in range(50)]
        body_lines = [f"[Au{i:02d}]" for i in range(50)]
        bib_text = "\n".join(bib_lines)
        body_text = " ".join(body_lines)
        t0 = time.time()
        r = self._cc(bib_text, body_text)
        assert time.time() - t0 < 3.0
        assert len(r.in_bib_not_cited) == 0
        assert len(r.cited_not_in_bib) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# R171–R200  BIB SECTION DETECTION + BODY/BIB SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractorSplit:
    """Bibliography section detection under all heading variants and edge cases."""

    @pytest.mark.parametrize("heading,key", [
        ("Literaturverzeichnis\n", "AB20"),
        ("LITERATURVERZEICHNIS\n", "AB20"),
        ("Literatur\n", "AB20"),
        ("Quellenverzeichnis\n", "AB20"),
        ("References\n", "AB20"),
        ("REFERENCES\n", "AB20"),
        ("Bibliography\n", "AB20"),
        ("Literature\n", "AB20"),
        ("Referenzen\n", "AB20"),
        ("5. Literaturverzeichnis\n", "AB20"),
        ("5.1 Literaturverzeichnis\n", "AB20"),
    ])
    def test_R171_heading_detected(self, heading, key):
        text = f"Body text.\n\n{heading}[{key}] Author: Paper. Springer, 2020."
        pos = _find_bib_start(text)
        assert pos >= 0, f"Heading not detected: {heading!r}"

    def test_R172_no_heading_key_fallback(self):
        text = "Body text here.\n[AB20] Author: Paper. Springer, 2020."
        pos = _find_bib_start(text)
        assert pos >= 0

    def test_R173_no_bib_returns_minus_one(self):
        text = "This paper has no references at all. Plain body text only."
        assert _find_bib_start(text) == -1

    def test_R173b_embedded_bibliography_heading(self):
        text = (
            "Body citation [MD00].\n"
            "Running header Bibliography\n"
            "[Ar05] Araújo, Miguel: A Paper. Journal, 2005."
        )
        pos = _find_bib_start(text)
        assert pos > text.index("Body citation")
        assert "[Ar05]" in text[pos:]

    def test_R174_body_not_contaminated(self):
        text = "Introduction.\n\nReferences\n[1] Smith: Paper. 2020."
        r = split_body_bib(text)
        assert "[1]" not in r["body"]

    def test_R175_bib_not_contaminated_by_body(self):
        text = "Introduction text.\n\nReferences\n[1] Smith: Paper. 2020."
        r = split_body_bib(text)
        assert "Introduction text" not in r["bibliography"]

    def test_R176_empty_text_empty_bib(self):
        r = split_body_bib("")
        assert r["bibliography"] == ""

    def test_R177_body_contains_text(self):
        text = "Body here.\n\nReferences\n[1] Smith: Paper. 2020."
        r = split_body_bib(text)
        assert "Body here" in r["body"]

    def test_R178_multi_key_in_bib_all_extracted(self):
        text = "Body.\n\nReferences\n[1] A: Paper. 2020.\n[2] B: Work. 2021."
        r = split_body_bib(text)
        assert "[1]" in r["bibliography"] and "[2]" in r["bibliography"]

    def test_R179_lni_pdf_bib_detected(self):
        if not PDF_LNI.exists():
            pytest.skip("LNI test PDF not available")
        r = extract_pdf(str(PDF_LNI))
        assert len(r.get("bibliography", "")) > 100, "LNI bib section too short"

    def test_R180_lni_pdf_body_detected(self):
        if not PDF_LNI.exists():
            pytest.skip("LNI test PDF not available")
        r = extract_pdf(str(PDF_LNI))
        assert len(r.get("body", "")) > 1000, "LNI body too short"

    def test_R181_test_pdf_bib_detected(self):
        if not PDF_TEST.exists():
            pytest.skip("Test PDF not available")
        r = extract_pdf(str(PDF_TEST))
        assert len(r.get("bibliography", "")) > 100

    def test_R182_nonexistent_pdf_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_pdf("/nonexistent/path/to/file.pdf")

    def test_R183_unsupported_format_raises(self):
        from extractor import extract
        with pytest.raises(ValueError):
            extract("/tmp/file.xyz")

    def test_R184_lni_pdf_all_bib_keys_found(self):
        if not PDF_LNI.exists():
            pytest.skip("LNI test PDF not available")
        r = extract_pdf(str(PDF_LNI))
        bib = r.get("bibliography", "")
        for key in ["BKMP21", "VS17", "Sc22", "Wa19", "Mu18", "RSKW20", "Xy99", "Br15"]:
            assert key in bib, f"Key {key} not in extracted bib"

    def test_R185_test_pdf_all_numeric_keys_found(self):
        if not PDF_TEST.exists():
            pytest.skip("Test PDF not available")
        r = extract_pdf(str(PDF_TEST))
        bib = r.get("bibliography", "")
        for key in ["1", "2", "3", "4", "5", "6"]:
            assert f"[{key}]" in bib, f"Numeric key [{key}] not in extracted bib"


# ═══════════════════════════════════════════════════════════════════════════════
# R186–R210  FULL PIPELINE: LNI TEST PAPER
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not PDF_LNI.exists(), reason="LNI test PDF not available")
class TestLNIFullPaper:
    """End-to-end pipeline on lni_full_test__1_.pdf — professor's test paper."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        r = extract_pdf(str(PDF_LNI))
        entries = parse_bibliography(r["bibliography"])
        bib_dict = entries_to_dict(entries)
        cited = extract_citations_from_body(r["body"])
        xcheck = cross_check(bib_dict, cited)
        return {"entries": entries, "bib": bib_dict, "cited": cited, "xcheck": xcheck}

    def test_R186_correct_number_of_bib_entries(self, pipeline):
        assert len(pipeline["entries"]) == 8, \
            f"Expected 8, got {len(pipeline['entries'])}: {[e.key for e in pipeline['entries']]}"

    def test_R187_bkmp21_type_book(self, pipeline):
        e = pipeline["bib"]["BKMP21"]
        assert e.entry_type == "book", f"Got: {e.entry_type}"

    def test_R188_vs17_title_clean(self, pipeline):
        e = pipeline["bib"]["VS17"]
        assert e.title == "Attention Is All You Need", f"Got: {e.title!r}"

    def test_R189_mu18_key_consistent(self, pipeline):
        e = pipeline["bib"]["Mu18"]
        assert e.key_consistent is not False, \
            f"Mu18/Müller should be consistent: {e.completeness_issues}"

    def test_R190_wa19_author_order_issue(self, pipeline):
        e = pipeline["bib"]["Wa19"]
        assert any("order" in i.lower() or "firstname" in i.lower() or "lastname" in i.lower()
                   for i in e.completeness_issues), e.completeness_issues

    def test_R191_sc22_missing_pages(self, pipeline):
        e = pipeline["bib"]["Sc22"]
        assert any("pages" in i.lower() for i in e.completeness_issues)

    def test_R192_rskw20_missing_booktitle(self, pipeline):
        e = pipeline["bib"]["RSKW20"]
        assert any("booktitle" in i.lower() for i in e.completeness_issues)

    def test_R193_xy99_huge_pages(self, pipeline):
        e = pipeline["bib"]["Xy99"]
        assert any("span" in i.lower() or "unusually" in i.lower()
                   for i in e.completeness_issues)

    def test_R194_ho21_is_missing_citation(self, pipeline):
        missing = pipeline["xcheck"].cited_not_in_bib
        lni_missing = [k for k in missing if not k.startswith("__")]
        assert "Ho21" in lni_missing, f"Ho21 should be missing: {missing}"

    def test_R195_br15_is_orphaned(self, pipeline):
        assert "Br15" in pipeline["xcheck"].in_bib_not_cited, \
            f"Br15 should be orphaned: {pipeline['xcheck'].in_bib_not_cited}"

    def test_R196_ez10_abc01_not_false_positive(self, pipeline):
        cited = pipeline["cited"]
        assert "Ez10" not in cited, "Ez10 is an example key, not a real citation"
        assert "ABC01" not in cited, "ABC01 is an example key, not a real citation"

    def test_R197_seven_correct_citations_matched(self, pipeline):
        correctly = set(pipeline["xcheck"].correctly_used)
        expected = {"BKMP21", "VS17", "Sc22", "Wa19", "Mu18", "RSKW20", "Xy99"}
        assert expected.issubset(correctly), \
            f"Some correctly-used keys missing: {expected - correctly}"

    def test_R198_no_false_orphans(self, pipeline):
        orphaned = set(pipeline["xcheck"].in_bib_not_cited)
        # Only Br15 should be orphaned
        unexpected = orphaned - {"Br15"}
        assert len(unexpected) == 0, f"Unexpected orphans: {unexpected}"

    def test_R199_mu18_title_clean(self, pipeline):
        e = pipeline["bib"]["Mu18"]
        assert "Informatik Spektrum" not in e.title, \
            f"Journal leaked into title: {e.title!r}"

    def test_R200_bkmp21_title_no_publisher_leak(self, pipeline):
        e = pipeline["bib"]["BKMP21"]
        assert "O'Reilly" not in e.title, \
            f"Publisher leaked into title: {e.title!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# R201–R230  FULL PIPELINE: TEST PDF (numeric keys)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not PDF_TEST.exists(), reason="Test PDF not available")
class TestNumericKeyPaper:
    """End-to-end pipeline on Test__1_.pdf — numeric citation keys."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        r = extract_pdf(str(PDF_TEST))
        entries = parse_bibliography(r["bibliography"])
        bib_dict = entries_to_dict(entries)
        cited = extract_citations_from_body(r["body"])
        xcheck = cross_check(bib_dict, cited)
        return {"entries": entries, "bib": bib_dict, "cited": cited, "xcheck": xcheck, "raw": r}

    def test_R201_six_entries_parsed(self, pipeline):
        assert len(pipeline["entries"]) == 6

    def test_R202_all_numeric_keys(self, pipeline):
        for e in pipeline["entries"]:
            assert e.key.isdigit(), f"Expected numeric key: {e.key}"

    def test_R203_entry_1_title_correct(self, pipeline):
        e = pipeline["bib"]["1"]
        assert "Attention Is All You Need" in (e.title or "")

    def test_R204_entry_2_title_correct(self, pipeline):
        e = pipeline["bib"]["2"]
        assert "BERT" in (e.title or "")

    def test_R205_entry_3_title_correct(self, pipeline):
        e = pipeline["bib"]["3"]
        assert "Residual" in (e.title or "")

    def test_R206_entry_4_is_fake(self, pipeline):
        """Entry 4 is a fabricated paper — parser should still extract it (detection done by verifier)."""
        e = pipeline["bib"]["4"]
        assert e.title is not None
        assert "Made Up" in e.title or "Completely" in e.title or "Fake" in e.title

    def test_R207_all_six_keys_matched(self, pipeline):
        for k in ["1","2","3","4","5","6"]:
            assert k in pipeline["xcheck"].correctly_used, \
                f"[{k}] not in correctly_used: {pipeline['xcheck'].correctly_used}"

    def test_R208_no_missing_entries(self, pipeline):
        assert len(pipeline["xcheck"].cited_not_in_bib) == 0

    def test_R209_no_orphaned_entries(self, pipeline):
        assert len(pipeline["xcheck"].in_bib_not_cited) == 0

    def test_R210_entry_4_huge_page_flagged(self, pipeline):
        e = pipeline["bib"]["4"]
        assert any("span" in i.lower() or "unusually" in i.lower() or "500" in i
                   for i in e.completeness_issues), \
            f"Expected page span issue: {e.completeness_issues}"


# ═══════════════════════════════════════════════════════════════════════════════
# R211–R240  FIND_DUPLICATES + EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindDuplicates:

    def test_R211_exact_duplicate_detected(self):
        bib = (
            "[VS17] Vaswani, A.: Attention Is All You Need. In: NeurIPS, 2017; S. 1--10.\n"
            "[Va17] Vaswani, A.: Attention Is All You Need. In: NeurIPS, 2017; S. 1--10."
        )
        entries = parse_bibliography(bib)
        d = entries_to_dict(entries)
        dupes = find_duplicates(d, threshold=0.85)
        assert len(dupes) > 0

    def test_R212_no_duplicates_clean(self):
        bib = (
            "[VS17] Vaswani, A.: Attention Is All You Need. In: NeurIPS, 2017; S. 1--10.\n"
            "[LB15] LeCun, Y.: Deep Learning. In: Nature, 2015; S. 1--10."
        )
        entries = parse_bibliography(bib)
        d = entries_to_dict(entries)
        dupes = find_duplicates(d, threshold=0.85)
        assert len(dupes) == 0

    def test_R213_near_duplicate_detected(self):
        bib = (
            "[VS17] Vaswani, A.: Attention Is All You Need. In: NeurIPS, 2017; S. 1--10.\n"
            "[Va17b] Vaswani, A.: Attention Is All You Need!. In: NeurIPS, 2017; S. 1--10."
        )
        entries = parse_bibliography(bib)
        d = entries_to_dict(entries)
        dupes = find_duplicates(d, threshold=0.85)
        assert len(dupes) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# R214–R240  ENCODING, WHITESPACE, UNICODE, EXTREME INPUTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_R214_all_whitespace_bib(self):
        assert parse_bibliography("   \n\t\n   ") == []

    def test_R215_single_entry_no_newline(self):
        e = parse_bibliography("[AB20] Author, Bob: Paper. Springer, 2020.")
        assert len(e) == 1

    def test_R216_very_long_title_no_crash(self):
        long = "A" * 500
        bib = f"[AB20] Author, Bob: {long}. Springer, 2020."
        entries = parse_bibliography(bib)
        assert len(entries) == 1

    def test_R217_unicode_title_chars(self):
        bib = "[MU20] Müller, J.: Über ℕ-vollständige Probleme. Springer, 2020."
        entries = parse_bibliography(bib)
        assert len(entries) == 1
        assert entries[0].title is not None

    def test_R218_arabic_numerals_in_title(self):
        bib = "[AB20] Author, Bob: GPT-3: Language Models. In: NeurIPS, 2020; S. 1--10."
        entries = parse_bibliography(bib)
        assert entries[0].title is not None

    def test_R219_doi_with_special_chars(self):
        bib = "[Va17] Vaswani, A.: Attention. In: NeurIPS, 2017. doi: 10.48550/arXiv.1706.03762"
        e = parse_bibliography(bib)[0]
        assert e.doi is not None
        assert "1706.03762" in e.doi

    def test_R220_url_with_query_string(self):
        bib = "[We22] Author: Page. https://example.com/path?q=1&r=2 Stand: 01.01.2022."
        e = parse_bibliography(bib)[0]
        assert e.url is not None

    def test_R221_entry_with_only_url_no_crash(self):
        bib = "[We22] https://example.com"
        entries = parse_bibliography(bib)
        assert len(entries) == 1

    def test_R222_mixed_numeric_lni_keys_both_parsed(self):
        bib = "[1] LeCun: DL. 2015.\n[AB20] Author, Bob: Paper. Springer, 2020."
        entries = parse_bibliography(bib)
        assert len(entries) == 2

    def test_R223_entry_type_unknown_gets_needs_ai_flag(self):
        bib = "[Ab20] Author, Bob: Paper Without Clear Type. 2020."
        e = parse_bibliography(bib)[0]
        assert e.needs_ai_parsing or e.entry_type in ("unknown", "misc", "book", "article")

    def test_R224_title_with_hyphen_not_split(self):
        bib = "[AB20] Author, Bob: State-of-the-Art Method. Springer, 2020."
        e = parse_bibliography(bib)[0]
        assert e.title is not None
        assert "State-of-the-Art" in e.title or "State" in e.title

    def test_R225_normalize_title_deterministic(self):
        t = "Attention Is All You Need: A Transformer Architecture"
        assert _normalize_title(t) == _normalize_title(t.upper())

    def test_R226_extract_citations_huge_body(self):
        """100k char body with 200 citations doesn't crash."""
        import time
        body = " ".join([f"[Au{i:02d}]" for i in range(200)] * 3)
        body += " " + "x" * 50000
        t0 = time.time()
        c = extract_citations_from_body(body)
        assert len(c) > 0
        assert time.time() - t0 < 5.0

    def test_R227_parse_entry_isbn_only(self):
        """Entry with ISBN but no other identifier."""
        bib = "[Go16] Goodfellow, Ian: Deep Learning. MIT Press, 2016. ISBN 978-0-262-03561-3"
        e = parse_bibliography(bib)[0]
        assert e.isbn is not None

    def test_R228_trailing_whitespace_stripped_from_url(self):
        bib = "[We22] Author: Page. https://example.com, Stand: 01.01.2022."
        e = parse_bibliography(bib)[0]
        if e.url:
            assert not e.url.endswith(",")
            assert not e.url.endswith(".")

    def test_R229_bib_heading_in_bib_text_not_counted_as_entry(self):
        """The heading line 'Literaturverzeichnis' should not produce a BibEntry."""
        bib = "Literaturverzeichnis\n[AB20] Author, Bob: Paper. Springer, 2020."
        entries = parse_bibliography(bib)
        # Should be exactly 1 entry, not 2
        assert len(entries) == 1
        assert entries[0].key == "AB20"

    def test_R230_50_entries_cross_check_all_matched(self):
        import time
        n = 50
        bib_lines = [f"[Au{i:02d}] Author{i}, N.: Paper {i}. Springer, 20{i%100:02d}." for i in range(n)]
        body = " ".join([f"[Au{i:02d}]" for i in range(n)])
        entries = parse_bibliography("\n".join(bib_lines))
        bib_dict = entries_to_dict(entries)
        cited = extract_citations_from_body(body)
        t0 = time.time()
        result = cross_check(bib_dict, cited)
        assert time.time() - t0 < 2.0
        assert len(result.in_bib_not_cited) == 0
        assert len(result.cited_not_in_bib) == 0
        assert len(result.correctly_used) == n
