# LNI Reference Checker

A Flask-based Python tool for validating reference lists in academic submissions, especially LNI-style bibliography formatting.

The application parses uploaded documents, extracts bibliography entries, checks them against local and external metadata sources, and flags ambiguous or conflicting results for professor review. It is designed to be conservative: missing metadata or mismatches are routed to manual review rather than auto-accepted.

## What the app does

- Extracts text from supported PDF, DOCX, and LaTeX inputs
- Parses bibliography entries into structured metadata
- Validates citation keys, authors, titles, years, venues, DOIs, and URLs
- Cross-checks in-text citations against bibliography entries
- Detects duplicates, missing references, orphaned entries, and malformed records
- Uses a local SQLite cache for previously verified papers
- Queries academic metadata sources when available
- Uses URL/web validation as supporting evidence for grey literature
- Uses AI only as an advisory layer and never as an automatic FAKE authority

## Project status

This project is a working research and academic utility rather than a universal guarantee engine. It is best suited to structured text-based inputs and LNI-style bibliographies, but ambiguous or incomplete entries still require manual review.

## Installation

Requirements:
- Python 3.9+
- pip

```bash
git clone <repo-url>
cd lni-reference-checker
pip install -r requirements.txt
```

## Environment setup

The app reads environment variables from a `.env` file when present. For AI fallback support, configure:

```env
AI_BASE_URL=https://api.openai.com/v1
AI_MODEL=gpt-4o-mini
AI_API_KEY=your_key_here
```

Optional variables used by the project:

```env
LNI_DB_DIR=.lni_db
LNI_CACHE_DIR=.lni_cache
SEMANTIC_SCHOLAR_API_KEY=your_key_here
UNPAYWALL_EMAIL=you@example.com
```

## Run the app

```bash
python app.py
```

Open the app in a browser at:

```text
http://localhost:5000
```

## Verification logic

The code follows this precedence in practice:

1. Professor review decisions
2. Duplicate detection
3. Structural validation
4. Local SQLite cache
5. Academic database queries
6. URL/web evidence
7. AI assessment

This design is intentionally conservative: a substantive mismatch is not silently converted into a positive match. If the metadata is inconsistent or incomplete, the result is kept for manual review.

## Main project files

```text
.
├── app.py
├── ai_checker.py
├── api_config.py
├── author_journal_verifier.py
├── author_validator.py
├── checker.py
├── citation_analysis.py
├── debug_matching.py
├── download_db.py
├── extractor.py
├── field_validators.py
├── local_db.py
├── parser.py
├── professor_workflow.py
├── review_queue.py
├── web_search_verifier.py
├── static/
│   └── index.html
├── tests/
├── docs/
├── requirements.txt
├── README.md
├── Procfile
└── .env.example (if present)
```

## Core endpoints

- `GET /` — serves the web interface
- `GET /status` — health and cache status
- `GET /status/apis` — API reachability summary
- `POST /check` — run a document verification job
- `POST /check-stream` — stream progress updates
- `POST /check-sync` — synchronous verification response
- `POST /ai-review` — AI verification pass on parsed entries
- `POST /api/review` — submit professor review decision
- `GET /api/review/stats` — review summary stats
- `POST /api/inject_paper` — manually add a verified paper to the cache
- `POST /api/confirm_paper` — confirm a paper in the review workflow
- `GET /api/db_stats` — cache database stats
- `GET /api/db_contents` — browse cached papers
- `POST /api/db_delete` — remove a cached paper
- `POST /api/db_delete_all` — clear the cache
- `POST /api/export-bibtex` — export BibTeX output
- `POST /export` — export a report

## Typical workflow

1. Upload a document file
2. Extract bibliography entries
3. Validate metadata and keys
4. Cross-check in-text citations with bibliography entries
5. Query local cache and academic sources
6. Review ambiguous items using URL or AI evidence
7. Submit professor decisions for final confirmation
8. Export or save verified results

## Important implementation notes

- The UI is a static HTML/JavaScript interface served by Flask from `static/`; it is not a React application.
- The local validation cache is stored in SQLite and uses WAL mode.
- AI output is advisory and explanatory; it does not override deterministic mismatch rules.
- URL and web verification are supporting evidence, not proof by themselves.
- Scanned or image-only PDFs are detected and usually require OCR or external preprocessing.

## Limitations

- Best suited for structured text-based academic PDFs and LNI-like bibliography formats
- Heavily malformed inputs may require manual review
- Academic APIs may be unavailable, rate-limited, or blocked in some environments
- Grey literature can be verified only as supporting evidence unless confirmed elsewhere

## Key files to inspect when modifying logic

- `checker.py` — core verification and mismatch routing
- `local_db.py` — SQLite cache and title/author normalization
- `ai_checker.py` — AI fallback behavior and advisory verdicts
- `parser.py` — bibliography parsing and validation
- `app.py` — Flask routes and orchestration

## Troubleshooting

### All entries come back as manual review

Check:
- whether the environment has access to configured APIs
- whether `.env` is loaded correctly
- whether your network allows outbound calls to academic metadata sources

### DB schema issues

If the cache database is missing expected columns, rebuild or migrate it according to the project scripts and database initialization flow.

### AI fallback is unavailable

Set the required AI environment variables and confirm the configured provider is reachable.

## Notes

This README reflects the behavior present in the current codebase and deliberately avoids claiming fixed benchmark numbers or universal guarantees that the implementation does not guarantee.

  title  = {LNI Reference Checker: Automated Academic Reference Verification and Validation},
  author = {Mithila Prabhu},
  year   = {2025},
  url    = {https://github.com/example/lni-reference-checker}
}
```

---


**Maintainer**: Mithila Prabhu (Frankfurt University of Applied Sciences)  
**Status**: Production-ready ✓
