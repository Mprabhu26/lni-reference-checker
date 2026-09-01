# LNI Reference Checker v7.0
### Academic Reference Verification & Validation Tool

A production-grade Flask web application for validating bibliographic references in LNI-formatted academic submissions. Detects fabricated, hallucinated, and incomplete citations through multi-stage verification: local caching, academic API queries, URL validation, and AI-powered semantic analysis.

---

## Features

### Reference Extraction & Parsing
- **Multi-format support**: PDF, Word (.docx), LaTeX (.tex + .bib)
- **LNI key validation**: Verifies author initials and year consistency against parsed metadata
- **Intelligent metadata extraction**: Distinguishes between journal articles, conference papers, books, and miscellaneous works
- **Completeness auditing**: Flags missing required fields per entry type

### Citation Cross-Checking
- **Missing references**: Identifies citations in body text with no bibliography entry
- **Unused entries**: Detects bibliography entries never cited
- **Self-citation detection**: Flags repetitive self-references
- **Duplicate detection**: Merges identical or near-identical entries

### Reference Verification Pipeline
Four-stage verification process with automatic caching:

1. **SQLite Local Cache** → ≥95% title match → REAL (instant)
2. **Academic APIs** (CrossRef, Semantic Scholar, OpenAlex, arXiv, DBLP) → ≥85% match → REAL
3. **URL Validation** → HTTP 200 + ≥95% title → REAL (only for flagged entries)
4. **AI Fallback** (Groq LLaMA 3.3 70B or Google Gemini) → ≥70% confidence → REAL

**Verdicts**: REAL, SUSPICIOUS, or FAKE (manual professor override only)

### Web Interface
Single-page React application with four analysis tabs:
- **Bibliography**: Parsed entries with metadata warnings and completeness flags
- **Cross-Check**: Citation gaps, unused entries, duplicates
- **Verification**: Detailed verification results with confidence scores and source attribution
- **Database**: Browser for cached verified papers with search and bulk management

Real-time progress feedback, automatic re-rendering after professor actions.

### Professor Workflow
- Mark FAKE entries (manual override for obvious hallucinations)
- Mark REAL entries (confirm suspicious results as legitimate)
- Inject manually-verified papers into persistent SQLite cache
- Download verification reports

---

## Architecture

```
lni_tool/
├── app.py                    Flask server, HTTP endpoints, session management
├── extractor.py              PDF/DOCX/LaTeX text extraction
├── parser.py                 LNI bibliography parsing, BibEntry dataclass
├── checker.py                Cross-checking, API verification (4-stage pipeline)
├── ai_checker.py             LLM integration, semantic analysis, grey literature handling
├── web_search_verifier.py    URL fetching, bot detection, fallback strategies
├── local_db.py               SQLite caching with zlib compression (verified papers only)
├── review_queue.py           Whitelist persistence, professor manual actions
├── make_fixtures.py          Test fixture generator (ReportLab PDFs)
├── static/
│   └── index.html            Complete single-page web UI (React, no external CDN)
├── requirements.txt          Python dependencies (all free/open-source)
├── conftest.py               pytest fixtures (BibEntry factory, test data)
├── pytest.ini                Test configuration
└── Procfile                  Heroku/container deployment config
```

### Verification Pipeline Flow
```
Input File
    ↓
[Extractor] → text extraction from PDF/DOCX/LaTeX
    ↓
[Parser] → LNI key & metadata extraction
    ↓
[Checker] → Cross-checking + citation extraction
    ↓
[Verification Pipeline]
    ├→ Local DB (SQLite) → hit → REAL ✓
    ├→ Academic APIs (parallel) → hit → REAL ✓
    ├→ URL fetch (suspicious only) → hit → REAL ✓
    └→ AI + web search → confidence score → REAL/SUSPICIOUS
    ↓
[AI Checker] → final verdict, metadata warnings
    ↓
JSON response → UI rendering
```

---

## Installation

### 1. Clone & Install
```bash
git clone <repo_url>
cd lni_tool
pip install -r requirements.txt
```

### 2. Configure Environment
Create `.env` file with API keys (all optional for basic use):
```bash
# LLM backends (at least one recommended)
AI_API_KEY=your_groq_api_key                    # console.groq.com (free)
AI_API_KEY_GEMINI=your_gemini_api_key           # aistudio.google.com (free)

# Optional: higher rate limits
SEMANTIC_SCHOLAR_API_KEY=your_key               # semanticscholar.org/product/api
GITHUB_TOKEN=your_github_token                  # github.com/settings/tokens

# Cache directories (defaults: .lni_cache, .lni_db)
LNI_CACHE_DIR=/path/to/cache
LNI_DB_DIR=/path/to/db

# Unpaywall open-access API
UNPAYWALL_EMAIL=your_email@university.edu
```

### 3. Initialize Databases
```bash
python -c "from local_db import init_cache_db; from review_queue import init_review_db; init_cache_db(); init_review_db()"
```

### 4. Run Server
```bash
python app.py
# Navigate to http://localhost:5000
```

---

## Usage

### Basic Workflow
1. **Upload document** (PDF, DOCX, or TEX+BIB)
2. **Optionally enable "Verify references online"** (queries academic APIs)
3. Click **"Run Check"**
4. Review results across tabs
5. Approve/reject entries; click **"Mark FAKE"** or **"Mark REAL"** for overrides
6. Download report

### API Endpoints

#### POST `/check`
Verify a single document. Multipart form with file upload.

**Response**:
```json
{
  "bibentries": [
    {
      "key": "AB20",
      "title": "Deep Learning",
      "authors": "Bengio, Lecun",
      "year": "2020",
      "verdict": "REAL",
      "confidence": 0.98,
      "source": "crossref",
      "metadata_warnings": []
    }
  ],
  "cross_check": {
    "missing_bib_entries": ["Smith2021"],
    "unused_entries": [],
    "self_citations": 0,
    "duplicates": []
  },
  "stats": {
    "total_entries": 42,
    "real_count": 39,
    "suspicious_count": 2,
    "fake_count": 1
  }
}
```

#### POST `/mark_fake`, `/mark_real`
Manual professor override. Persists to SQLite.

#### GET `/database/papers`
Browse cached papers with pagination and search.

#### GET `/database/stats`
Cache statistics (total papers, breakdown by source, disk size).

---

## Verification Strategy

### Academic APIs Used (Parallel Queries)
- **CrossRef**: Journal articles, DOI-indexed works
- **Semantic Scholar**: Computer science papers, preprints
- **OpenAlex**: Multidisciplinary coverage, open-access links
- **DBLP**: Computer science conferences and workshops
- **arXiv**: Preprints with native BibTeX export

### Confidence Scoring
Each API result scored on:
- **Title match** (40%): Levenshtein distance + umlaut tolerance (ä→ae, ö→oe, ü→ue)
- **Author overlap** (35%): Surname prefix matching, initials, accent normalization
- **Year exact** (25%): Must be within ±1 year

Threshold for REAL: ≥85% composite score (any single API)

### URL Validation
- Only attempted for SUSPICIOUS entries with URLs
- Detects bot-blocking (403, 429) and non-200 responses
- Requires ≥95% title match in page content (bot-safe)
- Auto-REAL only if all conditions met

### AI Fallback
Triggered for remaining SUSPICIOUS entries:

1. **Web search**: DuckDuckGo + BeautifulSoup extraction (fallback to Groq web search)
2. **LLM analysis**: Semantic Scholar + web search results → confidence score
3. **Threshold**: ≥70% confidence → REAL & cache; <70% → stays SUSPICIOUS
4. **Never FAKE**: AI verdict cannot produce FAKE (professor-only action)

### Grey Literature Handling
Detects and adapts thresholds for industry reports (Bitkom, Flexera), white papers, and conference workshops:
- Lower API thresholds (≥75% instead of ≥85%)
- Prioritizes URL validation over API scoring
- Never caches SUSPICIOUS grey literature entries

---

## Testing

### Run Test Suite
```bash
pytest                          # Full suite
pytest -v                       # Verbose output
pytest -m "not network"         # Skip network tests
pytest tests/test_parser.py     # Single module
```

### Key Test Suites
- **test_parser.py**: LNI key validation, metadata extraction, edge cases
- **test_checker.py**: Verification pipeline, scoring, API mocking
- **test_ai_checker.py**: Semantic analysis, false positive detection
- **test_extractor.py**: PDF/DOCX/LaTeX extraction

### Test Fixtures
Provided by `conftest.py`:
- `make_bib_entry()`: Factory for BibEntry objects
- `perfect_bib_text()`, `perfect_body_text()`: Golden reference data

### Generate Test PDFs
```bash
python make_fixtures.py
# Creates 20 structured test PDFs covering all verification scenarios
```

---

## Performance & Scalability

### Caching Strategy
- **Local DB**: SQLite with WAL mode for safe concurrent reads; zlib compression saves ~60% space
- **In-memory**: Per-session API results (cleared after response)
- **LLM cache**: Request deduplication across sessions (SHA256 hash)

### Rate Limiting
- **CrossRef**: 1 req/sec (50 req/sec limit respected)
- **Semantic Scholar**: 100 req/sec (backoff on 429)
- **Google APIs**: Exponential backoff
- **URL fetch**: 500ms minimum interval per domain

### Parallel Processing
- **ThreadPoolExecutor** for API calls (5 concurrent requests default)
- **Non-blocking uploads**: File processing in temporary directory
- **Streaming**: Progress feedback via chunked JSON responses

### Resource Limits
- **Max file size**: 30 MB
- **Request timeout**: 180 seconds
- **Temp storage**: OS default (auto-cleanup after verification)

---

## Configuration Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_API_KEY` | (none) | Groq API key (llama-3.3-70b-versatile) |
| `AI_API_KEY_GEMINI` | (none) | Google Gemini API key (fallback) |
| `LNI_CACHE_DIR` | `.lni_cache` | Disk cache for API results |
| `LNI_DB_DIR` | `.lni_db` | SQLite database directory |
| `SEMANTIC_SCHOLAR_API_KEY` | (none) | Higher rate limits for Semantic Scholar |
| `GITHUB_TOKEN` | (none) | GitHub API token (repo citation verification) |
| `UNPAYWALL_EMAIL` | (none) | Email for Unpaywall polite pool |
| `FLASK_ENV` | `production` | Flask environment mode |

---

## Limitations & Known Issues

### Cannot Process
- **Scanned PDFs** (image-only): Text extraction requires a text-based PDF
- **Non-Latin scripts**: Limited support for CJK, Cyrillic; umlauts handled
- **Handwritten citations**: No OCR support

### Known Behaviors
- **Not found ≠ fake**: German-language proceedings often absent from CrossRef; results flagged as SUSPICIOUS for manual review
- **Author name variation**: Middle initials, suffixes, and variant spellings may cause false SUSPICIOUS verdicts
- **Grey literature**: Industry reports, white papers have lower confidence thresholds (explicit design choice)
- **False positives**: Rare (~4%, identified in v6 audit); marked as SUSPICIOUS, not FAKE

---

## Deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "-t", "120", "app:app"]
```

### Heroku
```bash
git push heroku main
# Uses Procfile: web: gunicorn -b 0.0.0.0:$PORT -w 2 -t 120 app:app
```

### Environment-Specific Notes
- **Linux/macOS**: Signal-based timeouts work (SIGALRM)
- **Windows**: Timeouts disabled; use `--timeout 120` in gunicorn

---

## Contributing

### Code Structure
- Each module has a single responsibility (extractor, parser, checker, etc.)
- All fixtures are generated (no hardcoded test data)
- Tests use temporary directories for disk I/O

### Adding New Verification Sources
1. Implement API wrapper in a new module (e.g., `new_api_verifier.py`)
2. Return a scored `VerificationResult` object
3. Integrate into the 4-stage pipeline in `checker.py`
4. Add confidence score tuning + tests

### Modifying Scoring Logic
- Update thresholds in `checker.py` + `ai_checker.py`
- Re-run audit suite (`python make_fixtures.py && pytest`)
- Document changes in module docstring

---

## Support & Troubleshooting

### No API Results (All Entries SUSPICIOUS)
1. Check `GROQ_API_KEY` or `AI_API_KEY_GEMINI` env vars
2. Verify network access to `api.groq.com` or `generativelanguage.googleapis.com`
3. Check rate limits: `GET /stats` endpoint

### False SUSPICIOUS Verdicts
- Review metadata_warnings: author/year/publisher mismatches are common false positives
- Use `/mark_real` to override and cache the entry
- Report patterns to improve grey literature detection

### Performance Issues
1. Check temp directory disk space
2. Clear old cache: `python -c "from local_db import vacuum_db; vacuum_db()"`
3. Reduce `MAX_WORKERS` in `checker.py` if CPU-constrained
4. Use `--processes 1` with gunicorn on limited hardware

---

## Citation

If you use this tool in academic work, cite as:

```bibtex
@software{lni_reference_checker_2024,
  title={LNI Reference Checker: Automated Academic Reference Verification},
  author={Author Name},
  year={2024},
  url={https://github.com/example/lni-reference-checker}
}
```

---

## License

MIT License — Free for academic and commercial use.

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v7.0 | 2025 | Strict 4-stage verification pipeline; removed AI full-text pass; professor-only FAKE verdict |
| v6.0 | 2024 | Web UI with React frontend; SQLite persistent caching; 96% accuracy audit |
| v5.0 | 2024 | Multi-format extraction (PDF/DOCX/LaTeX); API parallelization |

---

## Contact

**Project**: LNI Reference Checker  
**Maintainer**: M. I. (Frankfurt University of Applied Sciences)  
**Issue Tracker**: GitHub Issues  
**Email**: [contact email]
