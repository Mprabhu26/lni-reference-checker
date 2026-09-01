# LNI Reference Checker

## Automated Academic Reference Verification & Validation

A production-grade Python/Flask application for validating bibliographic references in academic submissions. Detects fabricated, hallucinated, and incomplete citations through intelligent multi-stage verification combining local caching, academic APIs, URL validation, and AI-powered semantic analysis.

**Designed for**: LNI (Lecture Notes in Informatics) submissions, but works for any academic bibliography format.

---

## 🎯 Key Features

### Multi-Format Document Processing
- **PDF** (text-based; image/scanned PDFs not supported)
- **Microsoft Word** (.docx)
- **LaTeX** (.tex + .bib files)
- **Auto-detection** of format and encoding issues
- **Robust text extraction** handling ligatures, special characters, and footnotes

### Bibliography Parsing & Validation
- **LNI key validation** — verifies [AuthorInitial][Year] pattern consistency
- **Metadata extraction** — intelligently parses author, title, year, venue, DOI, URLs
- **Completeness auditing** — flags missing required fields per entry type (journal vs. conference vs. book)
- **Format detection** — distinguishes between BibTeX, IEEE, APA, and Chicago styles
- **Duplicate detection** — finds and merges near-identical entries
- **Self-citation detection** — flags excessive self-references

### Citation Cross-Checking
- **Missing references** — citations in body text with no bibliography entry
- **Orphaned entries** — bibliography entries never cited in body text
- **Contextual analysis** — extracts citation sentences for manual review
- **Pattern detection** — finds citation chains suggesting hallucinated references

### Intelligent Reference Verification

Four-stage pipeline with automatic caching:

```
1. LOCAL CACHE (SQLite)      → Title ≥95% match → REAL ✓ (instant)
   ├─ Cached verified papers with zlib compression
   └─ Results persist across sessions

2. ACADEMIC APIS (parallel)  → Title ≥85% match → REAL ✓
   ├─ CrossRef (journal DOIs)
   ├─ Semantic Scholar (CS papers + preprints)
   ├─ OpenAlex (multidisciplinary coverage)
   └─ arXiv (physics/CS preprints)

3. URL VALIDATION (suspicious only) → HTTP 200 + title match → REAL ✓
   ├─ Smart bot-blocking detection
   └─ Safe page content verification

4. AI + WEB SEARCH (final fallback) → Confidence ≥70% → REAL ✓
   ├─ Groq LLaMA 3.3 70B or Google Gemini
   ├─ DuckDuckGo web search with fallback
   └─ Semantic analysis of search results
```

**Verdicts**: `REAL`, `SUSPICIOUS`, or `FAKE`
- ✅ **REAL** — passed any verification stage (SUSPICIOUS → REAL via professor override)
- ⚠️ **SUSPICIOUS** — low confidence; requires manual review
- ❌ **FAKE** — professor-only action; AI never produces FAKE verdicts

### Web Interface

Single-page application with real-time feedback:

| Tab | Purpose |
|-----|---------|
| **Bibliography** | Parsed entries, metadata warnings, completeness checks |
| **Cross-Check** | Citation gaps, missing/orphaned entries, duplicates |
| **Verification** | Detailed results: verdict, confidence, source attribution |
| **Database** | Browse cached verified papers, search, inject, manage |

- **Auto-rendering** after professor actions (mark FAKE/REAL via the review API)
- **Real-time progress** with detailed API logs (streaming mode)
- **Download reports** in Excel (.xlsx) or BibTeX format

### Professor Workflow
- ✓ Review all SUSPICIOUS entries (requires human judgment)
- ✓ Submit decisions via **`/api/review`** (REAL/FAKE/SUSPICIOUS + optional note)
- ✓ **Manually inject** verified papers into the persistent database via **`/api/inject_paper`**
- ✓ **Search & browse** cached papers via **`/api/db_contents`**
- ✓ Download comprehensive reports (bibliography + verification scores)

---

## 📊 Architecture

### Module Overview

```
lni_tool/
├── app.py                   Flask server, HTTP endpoints, session handling (1342 L)
├── extractor.py             PDF/DOCX/LaTeX text extraction (1021 L)
├── parser.py                LNI parsing, BibEntry dataclass, completeness checks (849 L)
├── checker.py               API verification pipeline, cross-checking (1788 L)
├── ai_checker.py            LLM integration, semantic analysis, thresholds (1110 L)
├── web_search_verifier.py   URL validation, web search, bot detection (414 L)
├── local_db.py              SQLite caching with zlib compression (521 L)
├── review_queue.py          Professor decisions, override persistence (391 L)
├── download_db.py           Optional: full Semantic Scholar snapshot instructions
├── index.html               Complete single-page UI (1356 L)
├── requirements.txt         All free/open-source dependencies

```

> **Note**: `pytest.ini` and `Procfile` are not included in the repository by default. Add them as needed (see Testing and Deployment sections below).

### Verification Pipeline Flow

```
┌─────────────┐
│ Input File  │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ [Extractor]                                          │
│ PDF/DOCX/LaTeX → raw text + metadata                │
└──────┬───────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ [Parser]                                             │
│ Text → bibliography entries (BibEntry objects)      │
│ Extract LNI keys, validate metadata, detect format  │
└──────┬───────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ [Checker]                                            │
│ Cross-check citations vs. bibliography              │
│ Detect missing refs, orphans, duplicates            │
└──────┬───────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ [Verification Pipeline] — 4 Stages                  │
│                                                      │
│ Stage 1: SQLite Local Cache                         │
│   → Query title + author (normalized)               │
│   → Hit ≥95%? → REAL ✓                              │
│                                                      │
│ Stage 2: Academic APIs (parallel, 5 workers)        │
│   → CrossRef, Semantic Scholar, OpenAlex, DBLP     │
│   → Hit ≥85%? → REAL ✓                              │
│                                                      │
│ Stage 3: URL Validation (suspicious only)           │
│   → Fetch URL + check title in content              │
│   → ≥95% match? → REAL ✓                            │
│                                                      │
│ Stage 4: AI + Web Search (fallback)                 │
│   → DuckDuckGo + LLaMA 3.3/Gemini                  │
│   → Confidence ≥70%? → REAL ✓                       │
│   → Else → SUSPICIOUS                               │
└──────┬───────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ [AI Checker]                                         │
│ Final verdict, metadata warnings, false positives   │
└──────┬───────────────────────────────────────────────┘
       │
       ▼
    JSON Response → UI Rendering
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repo_url>
cd lni-reference-checker

# Python 3.9+ required
pip install -r requirements.txt
```

### 2. Configuration (Optional but Recommended)

Create a `.env` file in the project root:

```bash
# AI backends — pick at least one for full AI-fallback verification
AI_API_KEY=your_groq_api_key              # https://console.groq.com (free, llama-3.3-70b)
AI_BASE_URL=https://api.groq.com/openai/v1   # or https://generativelanguage.googleapis.com/v1beta/openai/
AI_MODEL=llama-3.3-70b-versatile             # or gemini-1.5-flash

# Optional: higher rate limits for academic APIs
SEMANTIC_SCHOLAR_API_KEY=your_s2_key     # https://semanticscholar.org/product/api
GITHUB_TOKEN=your_github_token           # https://github.com/settings/tokens

# Cache/database directories (defaults: .lni_cache, .lni_db)
LNI_CACHE_DIR=/path/to/cache
LNI_DB_DIR=/path/to/db

# Open-access link discovery via Unpaywall (optional)
UNPAYWALL_EMAIL=your_email@university.edu
```

**All APIs listed have free tiers adequate for academic use:**
- **Groq**: 14,400 req/day free
- **Google Gemini**: 1,500 req/day free
- **Semantic Scholar / CrossRef**: Free, no payment required

### 3. Initialize Databases

Databases are auto-initialized on first run. To initialize manually:

```bash
python -c "from local_db import init_cache_db; from review_queue import init_review_db; init_cache_db(); init_review_db()"
```

If you see an `author_norm` column error on an existing DB, run the migration helper:

```bash
python fix_db.py
```

### 4. Run Server

```bash
python app.py
# Navigate to http://localhost:5000
```

---

## 💡 Usage Workflow

### Typical Professor Workflow

```
1. UPLOAD document (PDF, DOCX, or TEX)
   └─ Optionally attach .bib sidecar for LaTeX

2. RUN CHECK (deep_check=true for full online verification)
   └─ Progress panel shows API calls, cache hits, AI reasoning

3. REVIEW RESULTS across tabs
   ├─ Bibliography tab: Check metadata warnings & completeness flags
   ├─ Cross-Check tab: Identify missing/orphaned entries
   ├─ Verification tab: Review AI verdicts (REAL/SUSPICIOUS)
   └─ Database tab: Search verified papers for context

4. MANUAL ACTIONS (for SUSPICIOUS entries)
   ├─ Read AI reasoning + source attributions
   ├─ Search your institution's database
   ├─ POST /api/review with decision="REAL" if legitimate → cached
   └─ POST /api/review with decision="FAKE" → score updated

5. DOWNLOAD REPORT
   └─ Excel file with bibliography + verification scores + cross-check results
   └─ BibTeX export of verified entries via /api/export-bibtex
```

### Example Scenarios

#### Scenario A: Legitimate but Rare Citation
- Entry marked SUSPICIOUS (low API match)
- You confirm it's real via institutional database
- **Action**: Submit `decision: REAL` to `/api/review` → cached for future submissions

#### Scenario B: Fabricated Reference
- Entry marked SUSPICIOUS (conflicting author/year)
- You search and find no evidence of the paper
- **Action**: Submit `decision: FAKE` → score penalty applied

#### Scenario C: Author Name Variation
- Paper by "R. Sutton" (real: "Richard S. Sutton")
- APIs return SUSPICIOUS due to name mismatch
- **Action**: Submit `decision: REAL` + inject via `/api/inject_paper` → future variant names pre-verified

---

## 🔌 REST API Endpoints

### POST `/check`
Analyze a document. Chooses between streaming and synchronous mode automatically.

```bash
curl -F "file=@my_paper.pdf" \
     -F "deep_check=true" \
     http://localhost:5000/check
```

### POST `/check-stream`
Streaming SSE variant — returns newline-delimited JSON events during analysis, then a final `data: DONE` event.

### POST `/check-sync`
Synchronous fallback — blocks until analysis completes, then returns full JSON.

### POST `/batch`
Analyze multiple documents in one request (multipart, multiple `file` fields).

### POST `/ai-review`
Run the AI reviewer pass on already-parsed entries (POST JSON with `entries` array).

**Sample `/check` response**:
```json
{
  "status": "success",
  "filename": "my_paper.pdf",
  "summary": {
    "bib_entry_count": 42,
    "citation_count": 38,
    "missing_from_bib": 1,
    "uncited_entries": 2,
    "duplicates": 0,
    "fake_candidates": 0,
    "suspicious": 3,
    "verified": 39
  },
  "score": {
    "score": 87,
    "grade": "A-",
    "verdict": "PASS",
    "verdict_reason": "3 suspicious entries require review",
    "penalties": [
      { "category": "Incomplete LNI entries", "count": 1, "deduction": 5 }
    ]
  },
  "bibliography": [
    {
      "key": "LBH15",
      "type": "article",
      "title": "Deep Learning",
      "authors": "Yann LeCun; Yoshua Bengio; Geoffrey Hinton",
      "year": "2015",
      "journal": "Nature",
      "doi": "10.1038/nature14539",
      "completeness_issues": [],
      "ai_verdict": "REAL",
      "confidence": 0.98,
      "source": "crossref",
      "pages": "436-444"
    }
  ],
  "cross_check": {
    "cited_not_in_bib": ["Smith2021"],
    "in_bib_not_cited": ["Brown2010"],
    "correctly_used": 38,
    "duplicates": []
  },
  "verification": [
    {
      "key": "VSP17",
      "title": "Attention Is All You Need",
      "ai_verdict": "REAL",
      "confidence": 0.99,
      "doi": "10.48550/arXiv.1706.03762",
      "sources_checked": ["arxiv", "semantic_scholar"],
      "open_access_url": "https://arxiv.org/pdf/1706.03762.pdf"
    }
  ]
}
```

---

### POST `/api/review`
Professor manual decision — persists to `review_queue.db`.

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"title":"Deep Learning","authors":"LeCun et al.","decision":"REAL","note":"Confirmed via library","ai_verdict":"SUSPICIOUS"}' \
     http://localhost:5000/api/review
```

**Body fields**: `title`, `authors`, `decision` (REAL/FAKE/SUSPICIOUS), `note` (optional), `url` (optional), `ai_verdict` (optional)

**Response**: `{"success": true}`

---

### GET `/api/review/stats`
Returns professor review statistics and the 10 most recent pending decisions.

---

### POST `/api/inject_paper`
Manually add a verified paper to the local SQLite cache.

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"title":"Deep Learning","authors":"LeCun; Bengio; Hinton","year":"2015","doi":"10.1038/nature14539","url":""}' \
     http://localhost:5000/api/inject_paper
```

**Response**:
```json
{
  "success": true,
  "message": "'Deep Learning' saved to local DB.",
  "db_total": 523643,
  "db_size_kb": 12451
}
```

---

### GET `/api/db_stats`
Database summary statistics.

```bash
curl http://localhost:5000/api/db_stats
```

**Response**:
```json
{
  "total_papers": 523642,
  "by_source": {
    "crossref": 250000,
    "semantic_scholar": 150000,
    "arxiv": 80000,
    "dblp": 40000,
    "user_verified": 3642
  },
  "db_size_kb": 12450
}
```

---

### GET `/api/db_contents`
Browse cached verified papers.

```bash
curl "http://localhost:5000/api/db_contents?search=deep+learning&limit=20&offset=0"
```

**Query Parameters**: `search`, `limit` (max 500, default 100), `offset`

**Response**: `{ "papers": [...], "total": 523642, "by_source": {...}, "db_size_kb": 12450 }`

---

### POST `/api/db_delete`
Delete a specific paper from the cache (by title).

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"title":"Deep Learning"}' \
     http://localhost:5000/api/db_delete
```

---

### POST `/api/db_delete_all`
Wipe the entire verified papers cache (destructive — use with caution).

---

### POST `/api/export-bibtex`
Export verification results as a BibTeX file.

**Body**: `{ "verification": [...], "bibliography": [...] }`

---

### POST `/export`
Export results as an Excel (.xlsx) report.

---

### GET `/status`
Health check and server status.

```bash
curl http://localhost:5000/status
```

---

## 🎓 Verification Strategy

### Confidence Scoring

Each verification source is independently scored:

| Metric | Weight | Details |
|--------|--------|---------|
| **Title Match** | 40% | Levenshtein distance after normalization (ä→ae, ö→oe, ü→ue) |
| **Author Overlap** | 35% | Surname prefix matching, initial matching, accent normalization |
| **Year Match** | 25% | Exact year or ±1 year tolerance |

**Composite score** = (title_score × 0.4) + (author_score × 0.35) + (year_score × 0.25)

**REAL threshold**: ≥85% on any single academic API
**Grey literature**: ≥75% threshold (industry reports, white papers)

### URL Validation

Only attempted for SUSPICIOUS entries with available URLs:
1. Check HTTP status (200 only)
2. Detect bot-blocking (403, 429) and skip
3. Extract page title + content
4. Match against original title (≥95%)
5. Auto-REAL only if all conditions met

### Academic APIs Used (Parallel Querying)

| API | Coverage | Key Field |
|-----|----------|-----------|
| **CrossRef** | Journal articles, books | DOI-indexed works |
| **Semantic Scholar** | CS papers, preprints | Computer science (primary) |
| **OpenAlex** | Multidisciplinary | Open-access coverage |
| **DBLP** | CS conferences | Computer science (secondary) |
| **arXiv** | Physics, CS preprints | Native BibTeX export |

All queries run in parallel (5 concurrent workers) with per-host rate limiting.

### AI Fallback Logic

For remaining SUSPICIOUS entries after API stage:
1. **Web search**: DuckDuckGo + BeautifulSoup extraction
2. **LLM analysis**: Groq LLaMA 3.3 70B (primary) → Google Gemini (fallback)
3. **Semantic understanding**: Author, year, topic consistency
4. **Threshold**: ≥70% confidence → REAL & cached; <70% → SUSPICIOUS

**AI verdict never outputs FAKE** — that is a professor-only action.

### Grey Literature Handling

Automatically detects and adapts for:
- Industry reports (Bitkom, Flexera, Gartner, etc.)
- White papers and technical reports
- Non-peer-reviewed conference workshops

**Adaptations**:
- Lower API thresholds (≥75% instead of ≥85%)
- Prioritize URL validation over API scoring
- Never cache SUSPICIOUS grey literature entries

---

## 📦 Caching Strategy

### SQLite Local Cache (`verified_papers.db`)

| Property | Detail |
|----------|--------|
| **Stores** | Title, authors, year, DOI, venue, URL, open-access status, confidence |
| **Compression** | zlib (~60% space savings) |
| **Query speed** | O(1) title hash lookups + full-text search |
| **Thread safety** | WAL mode + connection pooling |
| **Persistence** | Survives application restarts |

### In-Memory Session Cache

- **Per-request**: API results cleared after response
- **LLM cache**: Request deduplication via SHA256 hash of (model, system, prompt)
- **Purpose**: Avoid redundant API calls within a single analysis run

---

## 🧪 Testing

### Run Test Suite

```bash
# Full suite
pytest

# Verbose output
pytest -v

# Skip network tests (fast, local-only)
pytest -m "not network"

# Single test module
pytest tests/test_parser.py -v

# With coverage
pytest --cov=. --cov-report=html
```

### Generate Test Fixtures

```bash
# Creates structured test PDFs covering verification scenarios
python make_fixtures.py
```

Generates PDFs covering: perfect bibliography, hallucinated references, incomplete entries, near-duplicates, grey literature, non-Latin scripts, and author name variations.

### Shared Fixtures (`conftest.py`)

| Fixture | Purpose |
|---------|---------|
| `make_bib_entry()` | Factory for `BibEntry` objects |
| `perfect_bib_text()` | Golden reference bibliography string |
| `perfect_body_text()` | Golden body text with citation keys |
| `redirect_disk_cache` | Auto-redirects disk cache to temp dir for all tests |

---

## ⚡ Performance & Scalability

### Optimization Strategies

| Aspect | Strategy | Benefit |
|--------|----------|---------|
| **API calls** | Parallel `ThreadPoolExecutor` (5 workers) | 5× faster verification |
| **Disk cache** | zlib compression + indexed SQLite | 60% space savings |
| **Duplicate queries** | LLM cache via SHA256 hash | Skip redundant API calls |
| **URL fetching** | Per-domain rate limiting (500ms) | Respects server limits |
| **Text extraction** | Streaming PDF parsing | Low memory for large files |

### Resource Limits

```python
MAX_FILE_SIZE    = 30 MB       # Supports large dissertations
TIMEOUT          = 180 seconds  # Per-request timeout (SIGALRM on Linux/macOS)
MAX_WORKERS      = 5            # API concurrency
```

### Benchmarks (MacBook Pro M2, 16 GB RAM)

| Task | Time |
|------|------|
| Extract 50-page PDF | ~1.2s |
| Parse 50 bibliography entries | ~0.3s |
| Cross-check citations | ~0.8s |
| Verify 50 entries (cached) | ~0.5s |
| Verify 50 entries (APIs) | 8–15s |
| Full analysis (end-to-end) | 12–20s |

---

## 🔐 Security & Reliability

### Input Validation
- File type checking (magic bytes, not just extension)
- File size limit: 30 MB
- Text encoding detection (UTF-8, Latin-1, auto-convert)

### Error Handling
- Graceful API failures (falls through to next verification stage)
- Timeout protection (180s per request; SIGALRM on Linux/macOS, disabled on Windows)
- Database transaction integrity (WAL mode)
- Partial results returned if a stage fails

### Privacy
- Uploaded files are not stored after processing (temp dir auto-cleanup)
- No student/author names logged
- Cache stores only reference metadata (title, DOI, year, authors)
- Session data cleared after each response

---

## 🚢 Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "-t", "180", "app:app"]
```

```bash
docker build -t lni-checker .
docker run -p 5000:5000 \
  -e AI_API_KEY=$GROQ_KEY \
  -e AI_API_KEY_GEMINI=$GEMINI_KEY \
  lni-checker
```

### Gunicorn (Direct)

```bash
gunicorn -b 0.0.0.0:5000 -w 2 -t 180 app:app
```

### Environment-Specific Notes

- **Linux/macOS**: SIGALRM-based timeouts enforce the 180s request limit
- **Windows**: Signal-based timeouts are disabled; use `--timeout 120` in gunicorn
- **Cloud (AWS/GCP)**: Ensure sufficient `/tmp` space; temp files are auto-cleaned after each request

---

## 📋 Configuration Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_API_KEY` | (none) | Groq API key (llama-3.3-70b, primary LLM) |
| `AI_API_KEY_GEMINI` | (none) | Google Gemini API key (fallback LLM) |
| `SEMANTIC_SCHOLAR_API_KEY` | (none) | Higher rate limits for Semantic Scholar |
| `GITHUB_TOKEN` | (none) | GitHub API token for repo citation verification |
| `UNPAYWALL_EMAIL` | (none) | Email for Unpaywall polite pool (open-access links) |
| `LNI_CACHE_DIR` | `.lni_cache` | Disk cache for API results |
| `LNI_DB_DIR` | `.lni_db` | SQLite database directory |
| `FLASK_ENV` | `production` | Flask environment (development/production) |

At least one of `AI_API_KEY` or `AI_API_KEY_GEMINI` is required for the AI-fallback stage. All other keys are optional.

---

## ⚠️ Known Limitations & Workarounds

### Cannot Process

| Issue | Workaround |
|-------|-----------|
| Scanned PDFs (image-only) | Run Tesseract OCR or export from Adobe first |
| Non-Latin scripts (CJK, Cyrillic) | API coverage limited; use manual professor override |
| Corrupted PDFs | Try an online PDF repair tool before uploading |

### Known Behaviors

| Scenario | Behavior | Reason |
|----------|----------|--------|
| German conference proceedings | Often SUSPICIOUS | Many German venues not in CrossRef |
| Author name variations ("R. Sutton" vs "Richard Sutton") | SUSPICIOUS on first pass | APIs do prefix matching; override and inject to cache |
| Very recent papers (< 3 months old) | May be SUSPICIOUS | APIs lag by ~2–3 months; URL fallback catches most |
| Self-published white papers | Lower thresholds applied | Grey literature detection activates automatically |
| False positives (~2–4% historically) | SUSPICIOUS, not FAKE | By design: false-alarm is safer than false-negative |

### Edge Cases Handled

- Diacritics (ä, ü, ö, é, …) — normalized before matching
- Ligatures (ﬁ, ﬂ) — converted to ASCII equivalents
- Abbreviations (et al., pp., vol.) — stripped during parsing
- Unicode entities (`&nbsp;`, `&lt;`) — decoded before search
- LaTeX macros (`\emph{}`, `\textbf{}`) — stripped before matching

---

## 🐛 Troubleshooting

### All Entries Coming Back SUSPICIOUS

```bash
# 1. Verify API keys are loaded
echo $AI_API_KEY
echo $AI_API_KEY_GEMINI

# 2. Test Groq connectivity
curl -X POST https://api.groq.com/openai/v1/chat/completions \
  -H "Authorization: Bearer $AI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":"test"}]}'

# 3. Check disk space
df -h /tmp
du -sh .lni_cache .lni_db

# 4. Test CrossRef connectivity
curl "https://api.crossref.org/works?query.title=deep+learning&rows=1"
```

### False SUSPICIOUS Verdicts

Common causes: author middle initials omitted, off-by-one year, venue abbreviations, conference proceedings missing from CrossRef.

**Fix**: Use the Database tab to search for the paper → inject it manually via the UI, which calls `/api/inject_paper`. Future identical citations resolve instantly from cache.

### DB Schema Error (`author_norm` column missing)

```bash
python fix_db.py
```

### Performance Issues

```bash
# Check database size
du -sh .lni_db/

# Clear cache (destructive)
curl -X POST http://localhost:5000/api/db_delete_all

# Reduce concurrency if CPU-constrained
# Edit checker.py: MAX_WORKERS = 2
```

### Connection Timeouts

```bash
# Increase gunicorn timeout
gunicorn -b 0.0.0.0:5000 -w 2 -t 180 app:app

# Check CrossRef rate limits
curl -I "https://api.crossref.org/works?query.title=test"
# Look for: X-Rate-Limit-Limit, X-Rate-Limit-Interval
```

---

## 🤝 Contributing

### Code Style
- PEP 8 compliant (max 100 chars per line)
- Type hints on all functions
- Docstrings for modules and public functions
- Single responsibility per module

### Adding a Verification Source

```python
# 1. Create new_api_verifier.py
def verify_with_new_api(entry: BibEntry, max_retries=3) -> VerificationResult:
    """Query NewAPI for reference verification."""
    # Return VerificationResult(verdict="REAL"/"SUSPICIOUS", confidence=0.85, source="newapi")

# 2. Integrate into checker.py — add to VERIFICATION_SOURCES list

# 3. Add tests in tests/test_new_api.py with mocked responses

# 4. Tune thresholds in checker.py:
REAL_THRESHOLD_NORMAL = 0.85   # Academic papers
REAL_THRESHOLD_GREY   = 0.75   # Industry/grey literature
SUSPICIOUS_THRESHOLD  = 0.70   # AI fallback
```

---

## 📚 Further Reading

- **LNI Format Guide**: https://www.gi.de/service/publikationen/lni
- **CrossRef**: https://www.crossref.org/
- **Semantic Scholar**: https://www.semanticscholar.org/
- **OpenAlex**: https://openalex.org/
- **DBLP**: https://dblp.uni-trier.de/
- **arXiv**: https://arxiv.org/
- **Unpaywall**: https://unpaywall.org/

---

## 📄 Citation

```bibtex
@software{lni_reference_checker,
  title  = {LNI Reference Checker: Automated Academic Reference Verification and Validation},
  author = {Mithila Prabhu},
  year   = {2025},
  url    = {https://github.com/example/lni-reference-checker}
}
```

---

## 📝 License

**MIT License** — Free for academic and commercial use. All dependencies are free/open-source. No paid subscriptions required.

---

**Maintainer**: Mithila Prabhu (Frankfurt University of Applied Sciences)  
**Status**: Production-ready ✓
