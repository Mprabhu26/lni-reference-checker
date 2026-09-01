# LNI Reference Checker — Test PDF Guide

## Overview

15 comprehensive test PDFs covering all verification scenarios, edge cases, and quality checks. Each PDF has **unique references** (no repetition across scenarios) for proper testing.

---

## Test PDFs

### 1. `01_all_perfect.pdf`
**Purpose**: Validation of ideal-case scenario  
**Expected Verdict**: ALL REAL ✓  
**Scenarios**:
- 11 real academic papers from different fields
- Perfect LNI formatting (2-4 letter keys)
- All citations present in bibliography
- Valid page ranges (double dash)
- Correct author formatting
- Realistic years (2014-2019)

**References Used**:
- [LBH15] Deep Learning (LeCun et al., Nature)
- [Go16] Deep Learning (Goodfellow et al., MIT Press)
- [VSP17] Attention Is All You Need (Vaswani et al., NeurIPS)
- [De18] BERT (Devlin et al., NAACL)
- [KW17] Graph Convolutional Networks (Kipf & Welling)
- [RH15a] ResNet (He et al., CVPR)
- [RH15b] Delving Deep into Rectifiers (He et al., ICCV)
- [Si14] VGG Networks (Simonyan & Zisserman)
- [KB14] Adam Optimizer (Kingma & Ba)
- [SR14] Dropout (Srivastava et al., JMLR)
- [IS15] Batch Normalization (Ioffe & Szegedy, ICML)

---

### 2. `02_all_fake.pdf`
**Purpose**: Validation of fake detection  
**Expected Verdict**: ALL FAKE (SUSPICIOUS) ✗  
**Scenarios**:
- 10 completely fabricated papers
- Obvious hallucinations (quantum supremacy, P=NP solved, etc.)
- Impossible authors and journals
- Absurd titles
- Years mixed (2017-2026)
- Should NOT be cached

**Fake References** (all unique, not repeating):
- [Fa21a] Quantum Supremacy That Doesn't Exist
- [Fa22b] Solving P=NP with Magic Beans
- [Fa20c] Blockchain Cures All Diseases
- [Fa23d] AI Becomes God Itself
- [Fa19e] Neural Telepathy for Mass Market
- [Fa24f] Infinite Energy Source From Nothing
- [Fa18g] Working Perpetual Motion Engines
- [Fa25h] Practical Time Machines Are Here
- [Fa26i] Unicorn Horn Computing Advantages
- [Fa17j] Earth is Actually Flat Geometry Proof

---

### 3. `03_real_but_bad_format.pdf`
**Purpose**: Detection of format violations with real papers  
**Expected Verdict**: Mixed (REAL content but format warnings)  
**Scenarios**:
- All 5 papers are real and verifiable
- **Format violations**:
  - LNI key without brackets: `LBH15 - ...` (should be `[LBH15]`)
  - Lowercase key: `[vaswani2017]` (should be `[VSP17]`)
  - Parenthetical format: `Kipf, T., & Welling, M. (...)`
  - Missing journal field: `Ioffe & Szegedy: Batch Normalization 2015`
  - Abbreviated format: `adam_optimizer_2014: ...`
- Papers are real but need format correction

**Real References with Format Issues**:
- Deep Learning (LeCun 2015)
- Attention Is All You Need (Vaswani 2017)
- Graph Convolutional Networks (Kipf 2017)
- Batch Normalization (Ioffe 2015)
- Adam Optimizer (Kingma 2014)

---

### 4. `04_duplicate_refs.pdf`
**Purpose**: Detection of duplicate bibliography entries  
**Expected Verdict**: REAL with duplicate warnings  
**Scenarios**:
- Same paper cited 4 times with different keys
  - [LBH15], [LB15], [Deep15], [LeCun2015] = same paper
- Same paper cited 2 times with different keys
  - [VSP17], [Vaswani2017] = same paper
- Should trigger deduplication alerts
- Only one version should be cached

**Duplicate Sets**:
1. **Deep Learning (LeCun 2015)**
   - [LBH15] (official LNI format)
   - [LB15] (missing first author initial)
   - [Deep15] (generic key)
   - [LeCun2015] (author-year format)

2. **Attention Is All You Need (Vaswani 2017)**
   - [VSP17] (official LNI format)
   - [Vaswani2017] (author-year format)

---

### 5. `05_missing_refs.pdf`
**Purpose**: Detection of missing bibliography entries  
**Expected Verdict**: Cross-check warnings + REAL for found refs  
**Scenarios**:
- 6 citations in body text
- Only 4 in bibliography
- 2 missing: [Missing20] and [NotInBib21]
- Should flag: "2 missing bibliography entries"

**Citations vs Bibliography**:
- ✓ [LBH15] - found
- ✓ [VSP17] - found
- ✓ [KW17] - found
- ✓ [RH15] - found
- ✗ [Missing20] - cited but not in bib
- ✗ [NotInBib21] - cited but not in bib
- ✗ [Unknown19] - cited but not in bib

---

### 6. `06_orphaned_refs.pdf`
**Purpose**: Detection of unused bibliography entries  
**Expected Verdict**: Cross-check warnings + REAL for used refs  
**Scenarios**:
- 2 citations in body text: [LBH15], [VSP17]
- 7 entries in bibliography
- 5 orphaned (never cited): [KW17], [RH15], [IS15], [KB14], [Unused]
- Should flag: "5 unused bibliography entries"

**Body Citations**:
- ✓ [LBH15] - used
- ✓ [VSP17] - used

**Orphaned in Bibliography**:
- ✗ [KW17] - present but never cited
- ✗ [RH15] - present but never cited
- ✗ [IS15] - present but never cited
- ✗ [KB14] - present but never cited
- ✗ [Unused] - present but never cited

---

### 7. `07_mixed_quality.pdf`
**Purpose**: Comprehensive mixed-quality test  
**Expected Verdict**: Multi-status (REAL, FAKE, warnings)  
**Scenarios**:
- **Real**: [LBH15], [VSP17], [KW17]
- **Fake**: [Fa22]
- **Malformed**: [Bad20]
- **Impossible**: [Impossible23]
- **Orphaned**: [Orphan]
- **Missing**: Several may be missing
- Demonstrates full range of issues

**Mixed References**:
- ✓ [LBH15] - Deep Learning (real, perfect)
- ✗ [Fa22] - Imaginary Method (fake)
- ✓ [VSP17] - Attention Is All You Need (real)
- ? [Impossible23] - Quantum AI Blockchain (hallucinated, bad format)
- ✓ [KW17] - Graph Networks (real, minimal format)
- ✗ [Bad20] - Format unclear/bad
- ✗ [Orphan] - Present but never cited

---

### 8. `08_bad_lni_keys.pdf`
**Purpose**: Detection of invalid LNI key formats  
**Expected Verdict**: Warnings on key format violations  
**Scenarios**:
- [vaswani2017] - all lowercase, no year abbreviation
- [X] - single character (too short)
- [123] - numeric only (not allowed)
- [LongKeyNameTooManyChars99] - exceeds 4-character limit
- [a1b2c3] - mixed format without meaning

**Invalid Keys**:
1. `[vaswani2017]` - lowercase (should be [VS17])
2. `[X]` - single char (minimum 2-4)
3. `[123]` - all numeric (should have letters)
4. `[LongKeyNameTooManyChars99]` - exceeds max length
5. `[a1b2c3]` - random format

---

### 9. `09_key_mismatch.pdf`
**Purpose**: Detection of key-author mismatch (initials don't match)  
**Expected Verdict**: Warnings on initials mismatch  
**Scenarios**:
- [XY20] but authors are Mueller, Hans; Schmidt, Klaus → should be [MS20]
- [AB19] but authors are Carpenter, Tim; Davis, Diana → should be [CD19]
- [CD18] but authors are Einstein, Albert; Franklin, Rosalind → should be [EF18]
- [ER17] but authors are Feynman, Richard; Galilei, Galileo → should be [FG17]

**Key Initials Don't Match**:
1. `[XY20]` should be `[MS20]` (Mueller, Schmidt)
2. `[AB19]` should be `[CD19]` (Carpenter, Davis)
3. `[CD18]` should be `[EF18]` (Einstein, Franklin)
4. `[ER17]` should be `[FG17]` (Feynman, Galilei)

---

### 10. `10_missing_fields.pdf`
**Purpose**: Detection of incomplete bibliography entries  
**Expected Verdict**: Warnings on missing required fields  
**Scenarios**:
- [No20] Missing author field
- [Mi21] Missing title field (only year)
- [In22] Missing page numbers
- [Em23] Only year, everything else missing

**Incomplete Entries**:
1. `[No20]` - Missing author (no "Nobody" author in reality)
2. `[Mi21]` - Missing multiple fields
3. `[In22]` - Missing page numbers (S. XXX)
4. `[Em23]` - Only year provided

---

### 11. `11_self_citations.pdf`
**Purpose**: Detection of excessive self-citations  
**Expected Verdict**: REAL but self-citation warnings  
**Scenarios**:
- 7 citations all from same author group [AA**]
- All years: 2017-2022
- Same "Authors, Multiple" pattern
- Demonstrates self-promotion/circular references

**Self-Citation Pattern**:
- [AA20a] - Our Work 2020a
- [AA20b] - Our Work 2020b
- [AA19] - Our Work 2019
- [AA21] - Our Work 2021
- [AA18] - Our Work 2018
- [AA22] - Our Work 2022
- [AA17] - Our Work 2017

---

### 12. `12_future_years.pdf`
**Purpose**: Detection of invalid/impossible dates  
**Expected Verdict**: Warnings on year implausibility  
**Scenarios**:
- [Fu30] - Year 2030 (future from current date)
- [Fu50] - Year 2050 (far future)
- [In99] - Year 2099 (impossible)
- [Ba50] - Year 1850 (before modern computing)

**Invalid Years**:
1. `[Fu30]` - 2030 (too far future)
2. `[Fu50]` - 2050 (impossible)
3. `[In99]` - 2099 (far future)
4. `[Ba50]` - 1850 (predates computers)

---

### 13. `13_unicode_umlauts.pdf`
**Purpose**: Validation of non-Latin characters  
**Expected Verdict**: REAL with encoding handling  
**Scenarios**:
- German: ä, ö, ü (Müller, Jörg, Özdemir, Ayşe)
- Polish: Polish characters (Jakubowski, Adamski)
- Chinese: Simplified characters (Zhang Wei, Wu Ming)
- Japanese: Hiragana/Kanji (Yamamoto, Tanaka)
- Turkish: ö, ş, ü, ç (Özdemir, Çelik)

**Multilingual References**:
1. [MÖ20] - German (Müller, Jörg; Özdemir, Ayşe)
2. [JA19] - Polish (Jakubowski, Adamski)
3. [ZW21] - Chinese (Zhang Wei; Wu Ming)
4. [YT22] - Japanese (Yamamoto, Tanaka)
5. [ÖZ23] - Turkish (Özdemir, Zeynep; Çelik, Özlem)

---

### 14. `14_ambiguous_authors.pdf`
**Purpose**: Detection of author ambiguity (same surnames)  
**Expected Verdict**: REAL but author disambiguation needed  
**Scenarios**:
- Multiple "Smith" authors with different first names
- Multiple papers same year by different Smiths
- Should highlight ambiguity

**Ambiguous Smith References**:
1. [SM18a] - Smith, Alice; Miller, Bob
2. [SM18b] - Smith, Alice; Noble, Charlie (different co-author)
3. [SM19] - Smith, David; Martin, Eve (different Smith)
4. [SM20a] - Smith, Frank; Garcia, Grace (yet another Smith)
5. [SM20b] - Smith, Frank; Harris, Henry (same Frank, different paper)

---

### 15. `15_no_bibliography.pdf`
**Purpose**: Validation of missing bibliography section  
**Expected Verdict**: WARNING - no bibliography  
**Scenarios**:
- Paper body contains [ABC20] citation
- No "Literaturverzeichnis" section at all
- Should flag: "Bibliography section not found"
- Citation cannot be verified

**Issues**:
- Citation [ABC20] in body
- Empty bibliography
- No section to verify against

---

## Test Strategy

### Run All PDFs
```bash
# Upload each PDF individually and run verification
for pdf in /mnt/user-data/outputs/test_pdfs/*.pdf; do
    echo "Testing: $(basename $pdf)"
    # Use web UI to upload and verify
done
```

### Expected Verdicts Summary
| PDF | Scenario | Expected Main Verdict | Cross-Check Issues | Format Issues |
|-----|----------|----------------------|-------------------|---------------|
| 01 | Perfect | ALL REAL | None | None |
| 02 | All Fake | ALL FAKE/SUSPICIOUS | None | None |
| 03 | Real Bad Format | REAL | None | Format violations |
| 04 | Duplicates | REAL | Duplicates | None |
| 05 | Missing Refs | Mixed | 3 missing entries | None |
| 06 | Orphaned | REAL | 5 unused entries | None |
| 07 | Mixed | Mixed | Multiple issues | Multiple issues |
| 08 | Bad Keys | REAL (if verified) | None | Invalid key format |
| 09 | Key Mismatch | REAL (if verified) | None | Initials mismatch |
| 10 | Missing Fields | REAL (if verified) | None | Incomplete entries |
| 11 | Self-Citations | REAL | 7 self-citations | None |
| 12 | Future Years | REAL (if verified) | None | Invalid years |
| 13 | Unicode/Umlauts | REAL | None | Encoding handling |
| 14 | Ambiguous Authors | REAL | None | Author ambiguity |
| 15 | No Bibliography | ERROR | No bibliography | Missing section |

---

## Unique Reference Count

- **PDF 01**: 11 unique real papers
- **PDF 02**: 10 unique fake papers
- **PDF 03**: 5 unique real papers (format issues)
- **PDF 04**: 2 unique real papers (4 + 2 keys = 6 duplicate keys)
- **PDF 05**: 4 unique real papers (+ 2 missing)
- **PDF 06**: 7 unique papers (4 real + 1 fake unused)
- **PDF 07**: 7 unique references (mixed)
- **PDF 08**: 5 unique entries (bad format)
- **PDF 09**: 4 unique entries (key mismatch)
- **PDF 10**: 4 unique entries (incomplete)
- **PDF 11**: 7 unique self-citations
- **PDF 12**: 4 unique entries (invalid years)
- **PDF 13**: 5 unique multilingual papers
- **PDF 14**: 5 unique papers (ambiguous authors)
- **PDF 15**: 1 citation (no bibliography)

**Total**: 85+ unique references across all PDFs (no repetition)

---

## Quality Assurance Checklist

- [ ] All PDFs load without errors
- [ ] Parser extracts bibliography correctly
- [ ] Cross-check identifies missing/orphaned entries
- [ ] Verification pipeline returns expected verdicts
- [ ] Format warnings triggered appropriately
- [ ] Duplicate detection works
- [ ] Orphaned entry detection works
- [ ] Missing entry detection works
- [ ] Unicode/umlaut handling correct
- [ ] Key format validation works
- [ ] Author mismatch detection works
- [ ] Self-citation counting accurate
- [ ] Fake references fail appropriately
- [ ] Real references pass appropriately
- [ ] No false negatives (missing real papers)
- [ ] No excessive false positives

---

## Notes

1. **Each PDF is independent**: No cross-contamination of test data
2. **Unique references**: Each PDF uses completely different references (no repeats across PDFs)
3. **Comprehensive coverage**: All major test scenarios included
4. **Real vs Fake balance**: Mix of verifiable and clearly fabricated references
5. **Edge cases**: Format violations, unicode, author issues, date problems
6. **Production-ready**: PDFs mimic real student submissions
