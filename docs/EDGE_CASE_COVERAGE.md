# Test PDF Coverage Analysis

## Edge Cases COVERED ✓

### Bibliography Parsing
- ✓ PDF extraction (01-07, 12-15)
- ✓ DOCX extraction (implied framework)
- ✓ LaTeX extraction (implied framework)
- ✓ German heading "Literaturverzeichnis" (all)
- ✓ Empty bibliography (15)
- ✓ No bibliography section (15)
- ✓ Multiline entries (not explicit - MISSING)
- ✓ Very long author lists (not explicit - MISSING)

### LNI Key Validation
- ✓ Valid 2-4 letter keys [LBH15], [VSP17] (01, 03-07, 13)
- ✓ Invalid lowercase keys [vaswani2017] (08)
- ✓ Single char keys [X] (08)
- ✓ Numeric keys [123] (08)
- ✓ Too long keys [LongKeyNameTooManyChars99] (08)
- ✓ Mixed format [a1b2c3] (08)
- ✓ Key ambiguity with letters (04, 14)
- ✓ Initials mismatch (09)
- ✗ Key with special chars [A-B20] - NOT COVERED
- ✗ Key with spaces [A B 20] - NOT COVERED
- ✗ Key with unicode [Ä15] - NOT COVERED

### Author Handling
- ✓ Multiple authors (01, 03-07, 13-14)
- ✓ Author initials before surname [J. Smith] (not explicit - MISSING)
- ✓ Unicode authors [Müller, Jörg] (13)
- ✓ Mixed author formats (09, 14)
- ✓ Ambiguous authors (same surname) (14)
- ✓ Very long author lists (not explicit - MISSING)
- ✓ Author with suffixes Jr./Sr. - NOT COVERED
- ✓ Author missing - partial (10)
- ✗ Chinese author names - WEAK (13 has Chinese paper but not tested for parsing)
- ✗ Arabic/Hebrew authors - NOT COVERED

### Title Handling
- ✓ Normal titles (01-07, 09-15)
- ✓ Very long titles (02, 07)
- ✓ Unicode in titles (13)
- ✓ Titles with colons (many)
- ✓ Titles with quotes (not explicit - MISSING)
- ✓ Titles with special chars (not explicit - MISSING)
- ✗ Titles in other languages fully tested - WEAK
- ✗ Very short titles (1-2 words) - NOT COVERED
- ✗ Identical titles (not covered)

### Year Handling
- ✓ Valid years 2014-2026 (01, 04-07, 11, 13-14)
- ✓ Future years 2030, 2050 (12)
- ✓ Very old years 1850 (12)
- ✓ Year 2099 (02, 12)
- ✓ Year mismatches in keys (09)
- ✗ Invalid month/day - NOT COVERED
- ✗ BC dates - NOT COVERED
- ✗ Ambiguous year formats (1900 vs 1900s) - NOT COVERED

### Journal/Conference/Book Details
- ✓ Journal articles [In: Journal, Vol. X] (01, 03, 06)
- ✓ Conference proceedings [In: Proc./Proceedings] (01, 04, 05, 13-14)
- ✓ Books [MIT Press, Springer] (01, 03)
- ✓ Missing journal (10)
- ✓ Abbreviated journal names (not explicit - MISSING)
- ✗ Journal with special names [IEEE Transactions on...] - WEAK
- ✗ Workshop papers - NOT COVERED (explicit)
- ✗ ArXiv preprints - NOT COVERED
- ✗ Technical reports - NOT COVERED

### Page Numbers
- ✓ Double dash pages [S. 436--444] (01, 03-07, 13-14)
- ✓ Single dash pages [S. 436-444] (not explicit - MISSING)
- ✓ Missing pages (10)
- ✓ Huge page spans >500 (not explicit - MISSING)
- ✗ Roman numeral pages [S. i--xii] - NOT COVERED
- ✗ Non-standard notation [pp. 10-20] - NOT COVERED
- ✗ Single page [S. 100] - NOT COVERED

### DOI/URL/ISBN
- ✓ DOI field (not explicit - MISSING from PDFs)
- ✓ URLs with urldate (not explicit - MISSING)
- ✓ URLs without urldate (not explicit - MISSING)
- ✓ ISBN (not explicit - MISSING)
- ✗ ArXiv ID - NOT COVERED
- ✗ ResearchGate links - NOT COVERED
- ✗ GitHub repos as citations - NOT COVERED
- ✗ DOI with multiple formats (10.xxxx vs http://doi.org) - NOT COVERED

### Cross-Checking
- ✓ Missing bibliography entries (05)
- ✓ Orphaned bibliography entries (06)
- ✓ Both missing and orphaned (05, 06)
- ✓ All citations in body found (01, 03)
- ✓ No citations at all (15)
- ✓ Duplicate references (04)
- ✗ Self-citations (11) - COVERED BUT NOT WEIGHTED
- ✗ Circular references (A cites B, B cites A) - NOT COVERED
- ✗ Chain citations (A->B->C->D) - NOT COVERED

### Formatting Violations
- ✓ Wrong author order (03)
- ✓ Lowercase keys (03, 08)
- ✓ Missing brackets (03)
- ✓ Parenthetical format (03)
- ✓ Single dashes instead of double (not explicit - MISSING)
- ✓ Key initials mismatch (09)
- ✗ Inconsistent spacing - NOT COVERED
- ✗ Inconsistent punctuation - NOT COVERED
- ✗ Mixed citation styles (APA, MLA, IEEE mixed) - NOT COVERED
- ✗ Trailing commas/semicolons - NOT COVERED

### Reference Verification
- ✓ Real verifiable papers (01, 03, 05-06, 13-14)
- ✓ Fake hallucinated papers (02, 07)
- ✓ Real papers with format issues (03)
- ✓ Papers with missing fields (10)
- ✓ Future year papers (12)
- ✗ Papers with wrong metadata (title changed) - NOT COVERED
- ✗ Papers with DOI mismatch - NOT COVERED
- ✗ Papers from predatory journals - NOT COVERED
- ✗ Retracted papers - NOT COVERED

### Language & Encoding
- ✓ German (all, especially 13)
- ✓ Polish (13)
- ✓ Chinese (13)
- ✓ Japanese (13)
- ✓ Turkish (13)
- ✗ Russian/Cyrillic - NOT COVERED
- ✗ Arabic - NOT COVERED
- ✗ Hebrew - NOT COVERED
- ✗ Korean - NOT COVERED
- ✗ Hindi - NOT COVERED
- ✗ Greek letters in titles - NOT COVERED

### Special Scenarios
- ✓ No bibliography (15)
- ✓ Empty bibliography (15)
- ✓ Heavy self-citations (11)
- ✓ Duplicate references (04)
- ✓ Mixed quality (07)
- ✗ Whitepaper/technical reports - NOT COVERED
- ✗ PhD thesis citations - NOT COVERED
- ✗ Dataset citations - NOT COVERED
- ✗ Software package citations - NOT COVERED
- ✗ Standards/RFCs - NOT COVERED

---

## MISSING Edge Cases (Critical & Non-Critical)

### CRITICAL (Should Add)
1. **Multiline entries** - entries spanning multiple lines in PDF
2. **ArXiv/preprints** - papers with arXiv IDs instead of DOI
3. **URLs without urldate** - website citations missing date
4. **Single dashes** - pages with single dash instead of double
5. **Workshop papers** - underindexed in databases
6. **Dataset citations** - Zenodo, Figshare, etc.
7. **Mixed citation styles** - multiple formats in one paper
8. **Page number edge cases** - single page, Roman numerals

### MEDIUM (Nice to Have)
1. **Very long author lists** - 10+ authors
2. **Author with suffixes** - Jr., Sr., III
3. **Technical reports** - institutional reports
4. **Retracted papers** - flagged as problematic
5. **Predatory journals** - flagged as suspicious
6. **Greek letters in titles** - special character handling
7. **Russian/Cyrillic** - non-Latin script
8. **Non-standard page notation** - pp., pgs., etc.

### LOW (Nice to Have)
1. **Arabic/Hebrew** - RTL languages
2. **BC/ancient dates** - historical papers
3. **Circular citations** - A->B->A
4. **Chain of citations** - deep reference trails
5. **GitHub repos** - code citations
6. **ResearchGate links** - preprint servers

---

## Coverage Score

**Current Coverage: ~65-70%**

| Category | Coverage | Status |
|----------|----------|--------|
| Basic parsing | 90% | ✓ Good |
| LNI validation | 70% | ⚠ Medium (missing special chars) |
| Authors | 75% | ⚠ Medium (weak on international names) |
| Years | 85% | ✓ Good |
| Formatting | 65% | ⚠ Medium (missing styles, spacing) |
| Cross-checking | 85% | ✓ Good |
| References | 70% | ⚠ Medium (missing niche types) |
| Languages | 60% | ⚠ Medium (5 languages, missing 10+) |
| Special cases | 60% | ⚠ Medium (missing datasets, theses, standards) |

---

## Recommendations

### Add These 5 PDFs to Reach 85%+ Coverage:

1. **16_multiline_entries.pdf** - Entries spanning multiple lines
   - Tests parser robustness on wrapped bibliography entries
   
2. **17_special_types.pdf** - ArXiv, datasets, theses, reports
   - [AX20] arXiv preprint
   - [ZE21] Zenodo dataset
   - [TH19] PhD thesis
   - [TR22] Technical report

3. **18_url_citations.pdf** - Website citations with/without urldate
   - [GI24] Website with urldate
   - [WE22] Website without urldate
   - [MD23] Medium article

4. **19_edge_formatting.pdf** - Mixed styles, special chars, spacing
   - [Mix1] APA format
   - [Mix2] IEEE format  
   - [Mix3] MLA format
   - Greek/special chars in titles

5. **20_international_edge.pdf** - Russian, Arabic, Korean
   - [RU19] Russian paper
   - [AR21] Arabic paper
   - [KO22] Korean paper
   - With proper author/title encoding

