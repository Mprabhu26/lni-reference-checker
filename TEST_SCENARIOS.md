# LNI Reference Checker — Complete Test Scenarios & Expected Outputs
## 110 test cases across 7 categories

---

## HOW TO RUN

```bash
# Install test dependencies
pip install pytest reportlab python-docx --break-system-packages

# Generate test PDF/DOCX fixtures (run once)
cd /path/to/project
python tests/make_fixtures.py

# Run all non-network tests (fast, ~5s)
pytest tests/test_all.py -m "not network" -v

# Run all tests including network calls (slow, ~60s, requires internet)
pytest tests/test_all.py -v

# Run just one category
pytest tests/test_all.py -k "TestParserKeyValidation" -v

# Run with coverage
pytest tests/test_all.py -m "not network" --cov=. --cov-report=html
```

---

## CATEGORY P — PARSER UNIT TESTS (50 tests)

### P1–P12: LNI Key Format Validation

| ID | Input | Expected Output | What is tested |
|----|-------|-----------------|----------------|
| P01 | `validate_lni_key("Ez10")` | `[]` (no errors) | 1-author canonical key |
| P02 | `validate_lni_key("AB00")` | `[]` | 2-author key |
| P03 | `validate_lni_key("ABC01")` | `[]` | 3-author key |
| P04 | `validate_lni_key("Wa14a")` | `[]` | Disambiguation suffix accepted |
| P05 | `validate_lni_key("1")`, `"42"`, `"100"` | `[]` each | Numeric keys always valid |
| P06 | `validate_lni_key("vaswani17")` | `[error]` | All-lowercase rejected |
| P07 | `validate_lni_key("X")` | `[error]` | 1-char prefix + no year rejected |
| P08 | `validate_lni_key("TOOLONGKEY99")` | `[error mentioning 2–6]` | >6 initials rejected |
| P09 | `validate_lni_key("Ez2010")` | `[error]` | 4-digit year rejected |
| P10 | `validate_lni_key("V-17")` | `[error]` | Special chars in key rejected |
| P11 | `validate_lni_key("ABCDEF20")` | `[]` | 6-char prefix is maximum valid |
| P12 | `validate_lni_key("")` | `[error]` | Empty key rejected |

### P13–P30: Bibliography Entry Extraction

| ID | Fixture / Input | Expected Parsed Fields | What is tested |
|----|-----------------|------------------------|----------------|
| P13 | `bib_perfect.txt` | key=LBH15, authors contains "LeCun", year="2015" | Basic article parse |
| P14 | Entry with "Vol." | entry_type in ("article","proceedings") | Article type detection |
| P15 | Entry with "In: Proc." | entry_type="proceedings" | Proceedings type detection |
| P16 | Entry with "MIT Press" | entry_type="book" | Book type detection |
| P17 | Entry with "https://" | entry_type="website", url≠None | Website detection |
| P18 | `bib_doi_entry.txt` | doi contains "1706.03762" | DOI extraction |
| P19 | `bib_isbn_entry.txt` | isbn≠None, no dashes/spaces | ISBN extraction+normalization |
| P20 | `bib_website_with_urldate.txt` | urldate≠None | urldate from "Stand:" |
| P21 | `bib_perfect.txt` | len=2, keys={LBH15, VSP17} | All entries found |
| P22 | `bib_numeric_keys.txt` | len=3, keys={1,2,10} | Numeric keys extracted |
| P23 | `bib_multiline.txt` | year="2015" | Multiline entry normalized |
| P24 | `bib_disambiguation.txt` | len=2, keys={Wa14a,Wa14b} | Disambiguation keys both parsed |
| P25 | `""` | `[]` | Empty input → empty list |
| P26 | `bib_german_umlaut.txt` | len≥1 (no crash) | Umlaut authors handled |
| P27 | Single entry | year="2020" (4-digit string) | Year format |
| P28 | 3-author entry | ";" in authors or count≥2 | Semicolon-separated authors |
| P29 | Entry with "S. 436--444" | pages contains "436" | German page prefix |
| P30 | Entry with extra whitespace | len=1 (no phantom entries) | Whitespace robustness |

### P31–P50: Completeness Issues

| ID | Input | Expected completeness_issues | What is tested |
|----|-------|------------------------------|----------------|
| P31 | `bib_single_dash.txt` | Issue mentioning "dash" | Single-dash page range |
| P32 | Double-dash page range | No dash issue | Double dash OK |
| P33 | `bib_author_order.txt` | Issue mentioning "Lastname" or "order" | Wrong author order flagged |
| P34 | "Smith, John; Brown, Alice" | No author order issue | Correct order: no flag |
| P35 | `bib_initial_author.txt` | Issue OR no crash | Initial-before-surname |
| P36 | `bib_future_year.txt` | Issue mentioning "future" or "2099" | Future year flagged |
| P37 | Article without journal | Issue mentioning "journal" | Missing field: journal |
| P38 | Website with no URL | Issue mentioning "url" | Missing field: url |
| P39 | `bib_website_no_urldate.txt` | Issue mentioning "urldate" | Missing field: urldate |
| P40 | `bib_huge_page_span.txt` | Issue mentioning "span" or "unusually" | Implausible page range |
| P41 | `bib_bad_keys.txt` | ≥1 entry with key format issue | Invalid key detected |
| P42 | `bib_key_mismatch.txt` | Issue mentioning "inconsisten" | Key year mismatch |
| P43 | Perfect proceedings entry | completeness_issues=[] | Perfect entry: no issues |
| P44 | Book without year | Issue mentioning "year" | Missing field: year |
| P45 | Matching key+entry | key_consistent≠False | Consistent key not flagged |
| P46 | [AB99] with year 2020 | key_consistent=False | Year mismatch sets False |
| P47 | Numeric key [1] | key_consistent=True | Numeric keys always consistent |
| P48 | Article without pages | Issue mentioning "pages" | Missing field: pages |
| P49 | Book without publisher | Issue mentioning "publisher" | Missing field: publisher |
| P50 | [Wa14a], [Wa14b] | No key format errors | Disambiguation: valid keys |

---

## CATEGORY C — CHECKER UTILITY TESTS (38 tests)

### C1–C20: Title Similarity

| ID | Input | Expected Score | What is tested |
|----|-------|----------------|----------------|
| C01 | identical titles | ≈1.0 | Perfect match |
| C02 | "Quantum Physics" vs "History of Cooking" | <0.3 | Unrelated titles |
| C03 | ("", "Attention...") | 0.0 | Empty string |
| C04 | Word-reordered title | ≥0.6 | Token sort ratio |
| C05 | "Lernen" vs "Lernen" (umlauts) | ≥0.9 | Umlaut normalization |
| C06 | "BERT: Pre-training..." vs "BERT Pre-training..." | ≥0.85 | Colon as space |
| C07 | "A Survey..." vs "Survey..." | ≥0.9 | Stopword removal |
| C08 | lowercase vs UPPERCASE | ≥0.95 | Case insensitive |
| C09 | "Deep   Learning" vs "Deep Learning" | ≥0.9 | Whitespace normalized |
| C10 | "Transformers" vs "Transformers" | ≥0.9 | Single-word match |
| C11 | 1-word difference in 5-word title | ≥0.7 | Near match |
| C12 | Short vs very long title | <0.7 | Length mismatch |
| C13 | LaTeX \emph{word} vs plain | ≥0.9 | LaTeX stripped |
| C14 | ("", "something") | 0.0 | Empty input |
| C15 | "BERT" vs "Bidirectional Encoder..." | 0.0–0.5 | Abbreviation vs full |
| C16 | "An Introduction to ML" vs "Introduction to ML" | ≥0.85 | Article difference |
| C17 | "GPT-3..." vs "GPT3..." | ≥0.8 | Numbers in title |
| C18 | Fake title vs real title | <0.3 | Fake detection |
| C19 | "&amp;" in title | ≥0.8 | HTML entity stripped |
| C20 | Various pairs | 0.0≤score≤1.0 always | Score bounded |

### C21–C30: Surname Extraction

| ID | Input | Expected Output | What is tested |
|----|-------|-----------------|----------------|
| C21 | "Mueller, Hans" | ["mueller"] | Lastname, Firstname |
| C22 | "Mueller, Hans; Schmidt, Klaus" | 2 surnames | Semicolon split |
| C23 | "Mueller, Hans; et al." | only "mueller" | et al. skipped |
| C24 | "van der Berg, Jan" | includes "berg" | Noble particles skipped |
| C25 | "Müller, Jörg" | any "m..." surname | Umlaut normalized |
| C26 | "John Smith" | "smith" | Firstname Lastname format |
| C27 | `""` | `[]` | Empty string |
| C28 | "IEEE" | list (no crash) | Organization name |
| C29 | "Garcia-Lopez, Maria" | includes "garcia" or "lopez" | Hyphenated name |
| C30 | 6 authors | 6 surnames | Many authors |

### C31–C38: Author Overlap Score

| ID | Input | Expected | What is tested |
|----|-------|----------|----------------|
| C31 | Identical author strings | ≥0.9 | Perfect match |
| C32 | Completely different | <0.3 | No overlap |
| C33 | cited="" | None | Empty cited |
| C34 | correct="" | None | Empty correct |
| C35 | 1 of 2 matches | 0.3–0.7 | Partial overlap |
| C36 | "Mueller; et al." vs "Mueller; Schmidt" | >0.5 | et al. in cited |
| C37 | "Müller" vs "Mueller" | >0.3 | Umlaut fuzzy match |
| C38 | 10 authors each side | >0.5 | Capped at 6 |

---

## CATEGORY E — EXTRACTOR TESTS (22 tests)

### E1–E15: Bibliography Section Detection

| ID | Input Text | Expected | What is tested |
|----|-----------|----------|----------------|
| E01 | "Literaturverzeichnis\n[AB20]..." | pos≥0, heading in bib | German heading |
| E02 | "References\n[1]..." | pos≥0 | English heading |
| E03 | "REFERENCES\n[1]..." | pos≥0 | All-caps heading |
| E04 | "Quellenverzeichnis\n[AB20]..." | pos≥0 | Alt German heading |
| E05 | "5. Literaturverzeichnis\n..." | pos≥0 | Numbered section |
| E06 | No heading + bib keys | pos≥0 | Fallback to key pattern |
| E07 | No heading + no keys | pos=-1 | No bib detected |
| E08 | Body+heading+entries | body contains "Body text" | Correct split |
| E09 | Body+bib | "[1]" not in body | Body not contaminated |
| E10 | Body+bib | "Introduction" not in bib | Bib not contaminated |
| E11 | `""` | bibliography="" | Empty text |
| E12 | Numeric keys [1],[2] | "[1]" and "[2]" in bib | Numeric keys detected |
| E13 | "Literature\n[AB20]..." | pos≥0 | "Literature" heading |
| E14 | "Bibliography\n[1]..." | pos≥0 | English "Bibliography" |
| E15 | "References" appears twice | second occurrence selected | Last relevant heading |

### E16–E22: PDF Extraction

| ID | Fixture PDF | Expected | What is tested |
|----|-------------|----------|----------------|
| E16 | f01_perfect.pdf | body>50 chars, bib>20 chars | Basic extraction |
| E17 | f01_perfect.pdf | "LBH15" or "VSP17" in extracted text | Key preservation |
| E18 | f13_numeric_keys.pdf | "[1]" or "[2]" in full_text | Numeric keys in PDF |
| E19 | f19_no_bibliography.pdf | bib<50 chars or no "[" | Empty bib detected |
| E20 | f12_german_heading.pdf | "Mu20" or "Mueller" in text | German heading PDF |
| E21 | `/nonexistent/path.pdf` | raises Exception | Error handling |
| E22 | `file.xyz` | raises ValueError with "Unsupported" | Unsupported format |

---

## CATEGORY D — DATABASE / CACHE TESTS (12 tests)

| ID | Action | Expected | What is tested |
|----|--------|----------|----------------|
| D01 | save + search same title | result≠None, confidence≈0.95 | Basic save/retrieve |
| D02 | save "Attention Is All You Need", search same | result≠None | Normalized title match |
| D03 | search unsaved title | None | Cache miss |
| D04 | search_cache("") | None | Empty title |
| D05 | save twice with different confidence | result.confidence=second value | Overwrite updates |
| D06 | save 2 papers, get_cache_stats() | total_papers=2 | Stats count |
| D07 | save with source="api" and "web" | both appear in by_source | Stats by source |
| D08 | 1000-char title | result≠None | Long title safe |
| D09 | Title with ":" and "()" | result≠None | Special chars safe |
| D10 | normalize_title("Deep Learning!") vs ("deep learning") | equal | Normalization stable |
| D11 | normalize_title("the theory of computation") | "the" not in result | Stopwords removed |
| D12 | Insert old entry, clear_old_entries(1) | search returns None | Old entry purged |

---

## CATEGORY I — INTEGRATION / NETWORK TESTS (10 tests)

*These require internet. Run with: `pytest -m network`*

| ID | What is tested | Input | Expected (with internet) |
|----|---------------|-------|--------------------------|
| I01 | DOI lookup | doi=10.48550/arXiv.1706.03762 | status in (verified, partial_match) |
| I02 | Semantic Scholar real paper | "Deep Learning" by LeCun | confidence≥0.5 |
| I03 | OpenAlex real paper | "Attention Is All You Need" | result≠None |
| I04 | CrossRef real paper | "Deep Learning" + LeCun | result≠None |
| I05 | Fake paper S2 | "Revolutionary Quantum AI Blockchain..." | result is None OR confidence<0.5 |
| I06 | Author-first fallback S2 | Wrong title but right authors | No crash, returns None or float |
| I07 | DOI=None → _lookup_by_doi | doi=None | result=None |
| I08 | Invalid DOI | doi="not-a-real-doi" | None or any valid status (no crash) |
| I09 | URL liveness: example.com | https://example.com | status in (verified, not_found, error) |
| I10 | Broken URL flagged | nonexistent domain | status in (not_found, error), completeness_issues≥1 |

---

## CATEGORY S — SPECIAL / EDGE CASES (15 tests)

| ID | Input | Expected | What is tested |
|----|-------|----------|----------------|
| S01 | All-whitespace bib | `[]` | Whitespace-only input |
| S02 | Single entry, no trailing newline | len=1 | Edge of regex matching |
| S03 | 100 entries | len=100, elapsed<5s | Performance |
| S04 | Entry that is only a URL | entry_type="website" | URL-only entry |
| S05 | Entry with ISBN | isbn≠None | ISBN-only identifier |
| S06 | Mixed numeric+LNI keys | len=2 | Mixed key styles |
| S07 | 2000-char raw entry text | len=1 (no crash) | Very long entry |
| S08 | Bib with only heading | `[]` | No entries after heading |
| S09 | DOI with trailing period | doi doesn't end with "." | DOI stripping |
| S10 | URL with trailing comma | url doesn't end with "," | URL stripping |
| S11 | entries_to_dict | keys in dict, values are BibEntry | Utility function |
| S12 | Title with colon | title contains "Attention" | Subtitle colon kept |
| S13 | "S. 436--444" pages | pages contains "436" | German page prefix |
| S14 | "https://doi.org/..." in entry | doi≠None or url≠None | DOI URL format |
| S15 | normalize_title called twice | same result both times | Deterministic |

---

## CATEGORY X — CROSS-CHECK LOGIC TESTS (10 tests)

| ID | Body | Bib | Expected Orphaned | Expected Missing | What is tested |
|----|------|-----|-------------------|------------------|----------------|
| X01 | "[AB20]" | [AB20] | {} | {} | Perfect pair |
| X02 | "[AA01]" | [AA01],[BB02] | {BB02} | {} | Orphaned detected |
| X03 | "[AA01],[CC03]" | [AA01] | {} | {CC03} | Missing detected |
| X04 | "[1]" (numeric) | [1] | {} | {} | Numeric crosscheck |
| X05 | No citations | [AB20] | {AB20} | {} | All orphaned |
| X06 | "[AB20],[CD21]" | (empty) | {} | {AB20,CD21} | All missing |
| X07 | "[Wa14a]" | [Wa14a],[Wa14b] | {Wa14b} | {} | Disambiguation |
| X08 | "[AB20]" x3 | [AB20] | {} | {} | Repeated cite |
| X09 | "[LBH15],[VSP17]" | [LBH15],[VSP17] | {} | {} | All perfect |
| X10 | "[ab20]" (lowercase) | [AB20] | {AB20} | {ab20 or nothing} | Case sensitivity |

---

## FIXTURE FILES GENERATED

### PDF Fixtures (25 files in `tests/fixtures/pdf/`)

| File | Purpose | Key scenarios tested |
|------|---------|---------------------|
| f01_perfect.pdf | Golden path | All fields correct, all citations matched |
| f02_orphaned_entry.pdf | Cross-check | Orphaned bib entry |
| f03_missing_entry.pdf | Cross-check | Missing bib entry |
| f04_bad_key_format.pdf | Parser | Invalid LNI keys |
| f05_single_dash_pages.pdf | Completeness | Single-dash page range |
| f06_author_order.pdf | Completeness | Wrong author name order |
| f07_future_year.pdf | Completeness | Future year |
| f08_key_mismatch.pdf | Completeness | Key initials don't match authors |
| f09_missing_fields.pdf | Completeness | Missing required fields |
| f10_doi_verification.pdf | Verification | DOI for real paper |
| f11_fake_reference.pdf | Verification | Hallucinated reference |
| f12_german_heading.pdf | Extractor | Quellenverzeichnis heading |
| f13_numeric_keys.pdf | Parser | [1],[2] numeric keys |
| f14_website_citation.pdf | Completeness | Website with urldate |
| f15_website_no_urldate.pdf | Completeness | Website missing urldate |
| f16_disambiguation.pdf | Parser | [Wa14a],[Wa14b] |
| f17_long_author_list.pdf | Parser | 4+ authors, et al. key |
| f18_crossref_bib.pdf | Extractor | crossref field in BibTeX |
| f19_no_bibliography.pdf | Extractor | No bib section |
| f20_mixed_batch.pdf | Integration | Real+fake+real entries |
| f21_huge_page_span.pdf | Completeness | 500-page article |
| f22_initial_author.pdf | Completeness | F. Lastname format |
| f23_empty_bibliography.pdf | Extractor | Empty bib section |
| f24_mixed_citation_styles.pdf | Cross-check | LNI + other styles mixed |
| f25_unicode_authors.pdf | Parser | Umlaut/Unicode authors |

### DOCX Fixtures (2 files in `tests/fixtures/docx/`)

| File | Purpose |
|------|---------|
| d01_perfect.docx | Golden path DOCX |
| d02_cross_check.docx | Orphaned + missing in DOCX |

### Plain Text Fixtures (15 files in `tests/fixtures/txt/`)

Used by parser unit tests (no PDF dependency, always available):

`bib_perfect.txt`, `bib_bad_keys.txt`, `bib_single_dash.txt`,
`bib_author_order.txt`, `bib_future_year.txt`, `bib_key_mismatch.txt`,
`bib_website_no_urldate.txt`, `bib_website_with_urldate.txt`,
`bib_disambiguation.txt`, `bib_huge_page_span.txt`, `bib_doi_entry.txt`,
`bib_isbn_entry.txt`, `bib_numeric_keys.txt`, `bib_german_umlaut.txt`,
`bib_initial_author.txt`, `bib_empty.txt`, `bib_multiline.txt`,
`bib_initial_author.txt`

---

## ACCURACY EXPECTATIONS BY SCENARIO

| Scenario | Expected Detection Rate | Notes |
|----------|------------------------|-------|
| Valid LNI key format | 100% | Deterministic regex |
| Invalid key (wrong format) | 100% | Deterministic regex |
| Single-dash page range | 100% | Deterministic regex |
| Wrong author order (Firstname Lastname) | ~90% | May miss edge cases with particles |
| Wrong author order (F. Lastname) | ~85% | Pattern-dependent |
| Future year | 100% | Deterministic date comparison |
| Key year mismatch | ~95% | Depends on year extraction quality |
| Missing required fields | ~90% | Depends on entry type classification |
| Real paper found (with DOI) | ~99% | Near-certain via CrossRef |
| Real paper found (title only) | ~85–90% | Parallel 8-API search + fallbacks |
| Fake/hallucinated paper flagged | ~80–85% | LLM+web search required for certainty |
| Orphaned bib entry detected | ~99% | Deterministic regex on body text |
| Missing citation detected | ~95% | Depends on citation key pattern matching |
| URL liveness (broken URL) | ~90% | Network-dependent, some false negatives |
| urldate staleness | 100% | Deterministic date arithmetic |

---

## KNOWN LIMITATIONS

1. **Author order detection**: Multi-word surnames with particles (von, van, de) may
   occasionally be misclassified. The validator checks only the most common wrong patterns.

2. **Entry type classification**: "unknown" entries occur when no publisher, journal, or
   proceedings keyword is found. These trigger `needs_ai_parsing=True`.

3. **Title extraction from complex entries**: Very long or unusually formatted entries may
   have truncated titles. This is flagged via `needs_ai_parsing=True`.

4. **Fake paper detection without LLM keys**: Without GROQ_API_KEY or GEMINI_API_KEY,
   the web-search LLM fallback is unavailable. Detection relies on API database lookups only.

5. **PDF scanned documents**: Scanned PDFs (image-only) cannot be text-extracted. The
   tool detects and warns about these but cannot check them.

6. **Rate limiting**: Running all 100+ entries in batch mode may be throttled by external
   APIs. The tool includes rate limiting but a burst of papers may still cause temporary
   failures that retry once.
