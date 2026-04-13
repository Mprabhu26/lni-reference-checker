# LNI Reference Checker — Web UI
### Module 12 Individual Project · Information Technology

A browser-based tool to validate references in LNI-format academic submissions.
Supports **PDF**, **Word (.docx)**, and **LaTeX (.tex + .bib)** files.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

---

## Project Structure

```
lni_tool/
├── app.py              ← Flask server (upload endpoint + routing)
├── extractor.py        ← Extracts text from PDF / DOCX / LaTeX
├── parser.py           ← Parses [Key] bibliography entries
├── checker.py          ← Cross-checks citations + verifies via APIs
├── requirements.txt    ← Python dependencies
├── static/
│   └── index.html      ← Complete single-page web UI
└── README.md
```

---

## How to Use

1. **Upload** a PDF, DOCX, or TEX file using the left panel
2. For **LaTeX**: optionally attach the `.bib` sidecar file
3. Toggle **"Verify references online"** (queries CrossRef + Semantic Scholar)
4. Click **Run Check**
5. View results across 3 tabs:
   - **Bibliography** — all parsed entries with completeness warnings
   - **Cross-Check** — which citations are missing or uncited
   - **Verification** — which references are real, fake, or open access

---

## What Gets Checked

| Check | Description |
|-------|-------------|
| Extract entries | Parses all `[Key]` entries and metadata from the bibliography |
| Completeness | Flags missing required fields per entry type (author, title, year, etc.) |
| Cross-check | Detects citations in text with no bib entry, and bib entries never cited |
| Fake detection | Queries CrossRef + Semantic Scholar; scores title match confidence |
| Full text | Checks Unpaywall for open-access PDF links |

---

## Supported Formats

| Format | How it's handled |
|--------|-----------------|
| `.pdf` | Text extracted with `pdfplumber`; split at "Literaturverzeichnis" heading |
| `.docx` | Paragraphs extracted with `python-docx`; same split logic |
| `.tex` | LaTeX markup stripped; `.bib` file parsed directly if provided |

---

## Notes
- **Scanned PDFs** (image-only) won't work — text extraction requires a text-based PDF
- **Not found ≠ fake**: German conference proceedings are often not in CrossRef. Treat "not found" as a flag for manual review
- CrossRef and Semantic Scholar are free APIs — no keys needed
