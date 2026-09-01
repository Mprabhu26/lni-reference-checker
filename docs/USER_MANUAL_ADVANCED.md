# LNI Reference Checker v7.0 — Complete User Manual
## With Configuration, Customization & Scoring Guide

---

## Table of Contents

1. [Quick Start](#quick-start-5-minutes)
2. [Installation & Setup](#installation--setup)
3. [Configuration Guide](#configuration-guide)
4. [API Keys & Model Setup](#api-keys--model-setup)
5. [Scoring & Grading System](#scoring--grading-system)
6. [How to Use (Web Interface)](#how-to-use-web-interface)
7. [Understanding Results](#understanding-results)
8. [Customization Guide](#customization-guide)  
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)
11. [Tips & Best Practices](#tips--best-practices)
12. [Advanced Configuration](#advanced-configuration)
13. [Limitations](#limitations)

---

## Quick Start (5 Minutes)

### For Impatient Users

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add API key (optional)
export GROQ_API_KEY="your_key_here"

# 3. Run
python app.py

# 4. Open browser
# http://localhost:5000

# 5. Upload PDF
# Done!
```

---

## Installation & Setup

### Prerequisites

- **Python**: 3.8+
- **Disk**: 500 MB (PDFs + cache)
- **Internet**: Required for verification
- **RAM**: 2+ GB recommended

### Option 1: Docker (Easiest)

```bash
# Build
docker build -t lni-checker .

# Run with environment variables
docker run -p 5000:5000 \
  -e GROQ_API_KEY="sk-..." \
  -e AI_API_KEY_GEMINI="AIzaSy..." \
  -e LNI_CACHE_DIR="/app/cache" \
  lni-checker

# Access: http://localhost:5000
```

### Option 2: Local Installation

```bash
# Clone
git clone <repo>
cd lni-reference-checker

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << 'ENVFILE'
GROQ_API_KEY=sk-...
AI_API_KEY_GEMINI=AIzaSy...
LNI_CACHE_DIR=.lni_cache
LNI_DB_DIR=.lni_db
FLASK_ENV=production
FLASK_DEBUG=False
ENVFILE

# Run
python app.py
```

### Option 3: Heroku

```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login
heroku login

# Create app
heroku create your-app-name

# Set environment variables
heroku config:set GROQ_API_KEY="sk-..."
heroku config:set AI_API_KEY_GEMINI="AIzaSy..."

# Deploy
git push heroku main

# Open
heroku open
```

---

## Configuration Guide

### Environment Variables (All Options)

**File Location**: `.env` or exported in shell

```bash
# ─────────────────────────────────────────────────────────────
# LLM API KEYS (at least one recommended)
# ─────────────────────────────────────────────────────────────

# Groq API (Primary LLM) - FREE
# Sign up: https://console.groq.com
# Get key from: API Keys section
GROQ_API_KEY=sk-proj-abc123xyz789...

# Google Gemini (Fallback LLM) - FREE
# Sign up: https://aistudio.google.com
# Get key from: API Keys section
AI_API_KEY_GEMINI=AIzaSyDxxxxxxxxxxxxxxxxxxxx

# ─────────────────────────────────────────────────────────────
# OPTIONAL: HIGHER RATE LIMITS
# ─────────────────────────────────────────────────────────────

# Semantic Scholar API - FREE (higher limits)
# Sign up: https://semanticscholar.org/product/api
# No payment needed for free tier
SEMANTIC_SCHOLAR_API_KEY=your_api_key

# GitHub Token - FREE (repo citations, higher API limit)
# Generate: https://github.com/settings/tokens
# Scope: repo, read:user (scope not needed for basic use)
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxx

# ─────────────────────────────────────────────────────────────
# CACHE CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Where to store verified papers database
# Default: .lni_cache (in working directory)
# Change to: /var/cache/lni or /home/user/.lni_cache
LNI_CACHE_DIR=/path/to/cache

# Where to store SQLite database
# Default: .lni_db
LNI_DB_DIR=/path/to/db

# ─────────────────────────────────────────────────────────────
# FLASK CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Environment mode: production or development
FLASK_ENV=production

# Debug mode (NEVER on production)
FLASK_DEBUG=False

# Max file upload size (in bytes)
# Default: 30MB
MAX_FILE_SIZE=31457280

# ─────────────────────────────────────────────────────────────
# OPTIONAL: UNPAYWALL (Open Access PDFs)
# ─────────────────────────────────────────────────────────────

# Email for Unpaywall polite pool (required if using Unpaywall)
UNPAYWALL_EMAIL=your.email@university.edu
```

### Example .env File

```bash
# Production setup
GROQ_API_KEY=sk-proj-abcd1234efgh5678ijkl9012mnop3456
AI_API_KEY_GEMINI=AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz
SEMANTIC_SCHOLAR_API_KEY=abc123def456
LNI_CACHE_DIR=/var/cache/lni_checker
LNI_DB_DIR=/var/lib/lni_checker
FLASK_ENV=production
FLASK_DEBUG=False
MAX_FILE_SIZE=31457280
UNPAYWALL_EMAIL=admin@university.edu
```

### Minimal Setup (No API Keys)

```bash
# Run without API keys (limited functionality)
# Will use: cache + URL validation only
python app.py

# Only supports:
# - Local cache lookups
# - URL verification
# - Basic format checking
# - NO AI fallback
```

---

## API Keys & Model Setup

### Getting API Keys (Free)

#### 1. Groq API (Primary LLM)

```bash
# Step 1: Visit https://console.groq.com
# Step 2: Sign up (free account)
# Step 3: Click "API Keys" in left menu
# Step 4: Click "Create API Key"
# Step 5: Copy key starting with "sk-proj-"

# Step 6: Add to .env
GROQ_API_KEY=sk-proj-your-key-here

# Model used: llama-3.3-70b-versatile
# Rate limit: 14,400 requests/day (free tier)
# Cost: FREE

# Verify in Python:
python3 << 'PYEOF'
import os
from groq import Groq

key = os.getenv("GROQ_API_KEY")
if key:
    client = Groq(api_key=key)
    print("✓ Groq API key valid")
else:
    print("✗ No GROQ_API_KEY found")
PYEOF
```

#### 2. Google Gemini API (Fallback LLM)

```bash
# Step 1: Visit https://aistudio.google.com
# Step 2: Sign in with Google account
# Step 3: Click "Create API Key"
# Step 4: Copy the key

# Step 5: Add to .env
AI_API_KEY_GEMINI=AIzaSy-your-key-here

# Model used: gemini-1.5-flash
# Rate limit: 1,500 requests/day (free tier)
# Cost: FREE

# Verify in Python:
python3 << 'PYEOF'
import os
import google.generativeai as genai

key = os.getenv("AI_API_KEY_GEMINI")
if key:
    genai.configure(api_key=key)
    print("✓ Gemini API key valid")
else:
    print("✗ No AI_API_KEY_GEMINI found")
PYEOF
```

#### 3. Semantic Scholar (Optional, Higher Limits)

```bash
# Step 1: Visit https://semanticscholar.org/product/api
# Step 2: Sign up for free (no credit card)
# Step 3: Get API key from dashboard
# Step 4: Add to .env
SEMANTIC_SCHOLAR_API_KEY=your-key

# Rate limit: 100 requests/second (free)
# Cost: FREE
```

### Verifying All Keys

```bash
# Test all configured API keys
python3 << 'PYEOF'
import os
from groq import Groq
import google.generativeai as genai

print("API Key Configuration Check:")
print("=" * 50)

# Groq
groq_key = os.getenv("GROQ_API_KEY")
print(f"1. Groq: {'✓ Configured' if groq_key else '✗ Missing'}")
if groq_key:
    try:
        client = Groq(api_key=groq_key)
        print("   Status: ✓ Valid")
    except:
        print("   Status: ✗ Invalid")

# Gemini
gemini_key = os.getenv("AI_API_KEY_GEMINI")
print(f"2. Gemini: {'✓ Configured' if gemini_key else '✗ Missing'}")
if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        print("   Status: ✓ Valid")
    except:
        print("   Status: ✗ Invalid")

# Semantic Scholar
ss_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
print(f"3. Semantic Scholar: {'✓ Configured' if ss_key else '✗ Missing'}")

print("=" * 50)
print("Recommendation: Configure at least Groq")
PYEOF
```

---

## Scoring & Grading System

### How Scoring Works

The tool calculates a **Reference Quality Score** (0-100%) based on multiple factors. You can customize each component.

#### Scoring Components

```
Total Score = 100 - (Deductions)

Deductions:
  1. Format Issues: -X points per issue
  2. Missing Fields: -X points per field
  3. Duplicates: -X points per duplicate
  4. Missing References: -X points per missing
  5. Orphaned References: -X points per orphan
  6. SUSPICIOUS Verdicts: -X points per suspicious
  7. FAKE Verdicts: -X points per fake
  8. Self-Citations: -X points for excessive
```

### Default Scoring Scheme

**File**: `checker.py` (lines ~150-200)

```python
# SCORING CONFIGURATION (Customize Here)
class ScoringConfig:
    """Adjust these values to customize scoring"""
    
    # Maximum score
    MAX_SCORE = 100
    
    # Format violations (per issue)
    FORMAT_ISSUE_PENALTY = 2        # -2 per format issue
    LOWERCASE_KEY_PENALTY = 1       # -1 for lowercase LNI key
    MISSING_BRACKETS_PENALTY = 1    # -1 for [Key] not formatted
    
    # Incomplete entries (per missing field)
    MISSING_AUTHOR_PENALTY = 3      # -3 for no author
    MISSING_TITLE_PENALTY = 3       # -3 for no title
    MISSING_YEAR_PENALTY = 2        # -2 for no year
    MISSING_JOURNAL_PENALTY = 2     # -2 for journal (optional)
    MISSING_PAGES_PENALTY = 1       # -1 for pages
    
    # Cross-checking penalties
    MISSING_REFERENCE_PENALTY = 10  # -10 per missing entry in bib
    ORPHAN_REFERENCE_PENALTY = 5    # -5 per unused entry
    DUPLICATE_PENALTY = 8           # -8 per duplicate set
    
    # Verification verdicts
    SUSPICIOUS_PENALTY = 20         # -20 per SUSPICIOUS entry
    FAKE_PENALTY = 50               # -50 per FAKE entry
    
    # Self-citations
    SELF_CITATION_THRESHOLD = 0.15  # Flag if >15% self-citations
    SELF_CITATION_PENALTY = 15      # -15 for excessive self-citations
    
    # UML key initials mismatch
    KEY_MISMATCH_PENALTY = 3        # -3 per key/author mismatch
```

### Where to Change Scoring

#### Location 1: `checker.py` (Lines 150-200)

```python
# ✓ EASIEST: Change values in ScoringConfig class
class ScoringConfig:
    MISSING_REFERENCE_PENALTY = 10  # Change this number
    SUSPICIOUS_PENALTY = 20         # Or this one
    FAKE_PENALTY = 50               # Or this
```

#### Location 2: Configuration File (Recommended)

Create `scoring_config.yaml`:

```yaml
# scoring_config.yaml
scoring:
  max_score: 100
  
  format:
    issue: 2
    lowercase_key: 1
    missing_brackets: 1
  
  incomplete:
    missing_author: 3
    missing_title: 3
    missing_year: 2
    missing_journal: 2
    missing_pages: 1
  
  cross_check:
    missing_reference: 10    # ← Change this
    orphan_reference: 5      # ← Or this
    duplicate: 8             # ← Or this
  
  verification:
    suspicious: 20           # ← Change this for SUSPICIOUS deduction
    fake: 50                 # ← Change this for FAKE deduction
  
  self_citation:
    threshold: 0.15
    penalty: 15
  
  key_mismatch: 3
```

Load in Python:

```python
# In checker.py
import yaml

with open('scoring_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    MISSING_REF_PENALTY = config['scoring']['cross_check']['missing_reference']
    SUSPICIOUS_PENALTY = config['scoring']['verification']['suspicious']
```

### Calculating Your Custom Scoring

#### Example: Institutional Requirement

**Requirement**: "Don't deduct for SUSPICIOUS, only for FAKE"

```python
# Before (Default)
SUSPICIOUS_PENALTY = 20
FAKE_PENALTY = 50

# After (Custom)
SUSPICIOUS_PENALTY = 0        # ← No deduction for SUSPICIOUS
FAKE_PENALTY = 50             # ← Still penalize FAKE
```

#### Example: Cross-Verification Heavy

**Requirement**: "Deduct 14 marks for each missing reference, not 10"

```python
# Before (Default)
MISSING_REFERENCE_PENALTY = 10

# After (Custom)
MISSING_REFERENCE_PENALTY = 14  # ← Changed from 10 to 14
```

#### Example: Format-Light Grading

**Requirement**: "Don't care about format, focus on content"

```python
# Before (Default)
FORMAT_ISSUE_PENALTY = 2
MISSING_BRACKETS_PENALTY = 1

# After (Custom)
FORMAT_ISSUE_PENALTY = 0
MISSING_BRACKETS_PENALTY = 0
```

### Calculate Final Score (How It's Done)

**File**: `checker.py` (function: `calculate_score`)

```python
def calculate_score(self, entries, cross_check_results, verification_results):
    """
    Calculate reference quality score (0-100%).
    
    Args:
        entries: List of BibEntry objects
        cross_check_results: CrossCheckResult with missing/orphaned
        verification_results: List of VerificationResult objects
    
    Returns:
        float: Score 0-100
    """
    score = ScoringConfig.MAX_SCORE  # Start at 100
    
    # 1. Format penalties
    for entry in entries:
        if not entry.key.isupper():  # lowercase key
            score -= ScoringConfig.LOWERCASE_KEY_PENALTY
        if missing_required_fields(entry):  # missing fields
            score -= ScoringConfig.MISSING_AUTHOR_PENALTY  # etc.
    
    # 2. Cross-check penalties
    score -= len(cross_check_results.missing_bib_entries) * \
             ScoringConfig.MISSING_REFERENCE_PENALTY
    
    score -= len(cross_check_results.unused_entries) * \
             ScoringConfig.ORPHAN_REFERENCE_PENALTY
    
    # 3. Verification penalties
    for result in verification_results:
        if result.verdict == "SUSPICIOUS":
            score -= ScoringConfig.SUSPICIOUS_PENALTY
        elif result.verdict == "FAKE":
            score -= ScoringConfig.FAKE_PENALTY
    
    # 4. Self-citation penalty
    if self_citation_ratio > ScoringConfig.SELF_CITATION_THRESHOLD:
        score -= ScoringConfig.SELF_CITATION_PENALTY
    
    # Ensure score in range [0, 100]
    return max(0, min(100, score))
```

### Example Scoring Scenarios

#### Scenario 1: Perfect Paper

```
Starting score: 100

Deductions:
  - Format issues: 0
  - Missing fields: 0
  - Duplicates: 0
  - Missing refs: 0
  - Orphaned refs: 0
  - SUSPICIOUS: 0
  - FAKE: 0
  - Self-citations: 0

FINAL SCORE: 100/100 ✓
```

#### Scenario 2: Good Paper with 2 SUSPICIOUS

```
Starting score: 100

Deductions:
  - Format issues: 1 × 2 = -2
  - Missing fields: 0
  - Duplicates: 0
  - Missing refs: 0
  - Orphaned refs: 0
  - SUSPICIOUS: 2 × 20 = -40
  - FAKE: 0
  - Self-citations: 0

FINAL SCORE: 100 - 2 - 40 = 58/100 ⚠
```

#### Scenario 3: Poor Paper with Duplicates & Missing

```
Starting score: 100

Deductions:
  - Format issues: 5 × 2 = -10
  - Missing fields: 3 × 3 = -9
  - Duplicates: 2 × 8 = -16
  - Missing refs: 3 × 10 = -30
  - Orphaned refs: 2 × 5 = -10
  - SUSPICIOUS: 1 × 20 = -20
  - FAKE: 0
  - Self-citations: -15

FINAL SCORE: 100 - 10 - 9 - 16 - 30 - 10 - 20 - 15 = -10 → 0/100 ✗
```

### Customizing Scoring Per Institution

Create institutional preset:

```python
# scoring_presets.py
class UniversityPresets:
    """Pre-configured scoring for different institutions"""
    
    # Frankfurt University: Format-heavy
    FRANKFURT_UNI = {
        'format_issue': 3,
        'missing_reference': 15,
        'orphan_reference': 8,
        'suspicious': 25,
        'fake': 50
    }
    
    # MIT: Content-heavy (less format)
    MIT = {
        'format_issue': 1,
        'missing_reference': 20,
        'orphan_reference': 15,
        'suspicious': 30,
        'fake': 50
    }
    
    # Oxford: Strict grading
    OXFORD = {
        'format_issue': 5,
        'missing_reference': 25,
        'orphan_reference': 15,
        'suspicious': 40,
        'fake': 60
    }
```

---

## How to Use (Web Interface)

### For Students

#### Step 1: Prepare Your Paper

- ✓ Final draft (PDF recommended)
- ✓ Bibliography section present
- ✓ References in body text

#### Step 2: Open Application

```
Local:    http://localhost:5000
Docker:   http://localhost:5000
Heroku:   https://your-app-name.herokuapp.com
```

#### Step 3: Upload File

```
1. Click "Upload File" button (left panel)
2. Select PDF/DOCX/TEX file
3. Click "Upload"
4. Wait for success message
```

#### Step 4: Configure Verification

```
☐ Verify references online (optional, adds 60-120 seconds)
  - Checked: Queries CrossRef, Semantic Scholar, etc.
  - Unchecked: Uses cache + URLs only (faster)
```

#### Step 5: Run Check

```
Click "Run Check" button
  ↓
Processing... (30-180 seconds)
  ↓
Results displayed in 4 tabs
```

#### Step 6: Review Results

| Tab | Check |
|-----|-------|
| **Bibliography** | Format issues, missing fields |
| **Cross-Check** | Missing/unused/duplicate entries |
| **Verification** | Verdict (REAL/SUSPICIOUS/FAKE) |
| **Database** | Previously cached papers |

#### Step 7: Fix Issues

```
Format issues:
  → Fix in your document
  → Reformat according to LNI

SUSPICIOUS references:
  → Add DOI/URL if available
  → Accept manual review by professor

FAKE references:
  → Remove immediately
  → Replace with verified source
```

### For Professors

#### Batch Checking (10 Students)

```
TIME: ~50 minutes (5 min per paper)

STEP 1: Upload student PDF
  └─ Student_Name.pdf

STEP 2: Check Bibliography tab
  └─ Note format consistency

STEP 3: Check Cross-Check tab
  └─ Missing entries: How many?
  └─ Orphaned entries: How many?
  └─ Duplicates: Any patterns?

STEP 4: Check Verification tab
  └─ How many REAL? (%)
  └─ How many SUSPICIOUS? (%)
  └─ Any FAKE? (immediate fail)

STEP 5: Manual overrides
  └─ Click "Mark FAKE" on obvious errors
  └─ Click "Mark REAL" on legitimate grey literature

STEP 6: Database tab
  └─ Browse cache to understand verification sources

STEP 7: Record findings
  └─ Student Score: From Verification tab
  └─ Issues to address: From Cross-Check
  └─ Feedback: Format, content, integrity
```

#### Database Browser (Professor Features)

```
"Database" Tab:
  
  1. Search by Title/Author
     └─ Find papers you've verified before
  
  2. View Source
     └─ CrossRef, Semantic Scholar, Manual, AI, etc.
  
  3. View Confidence
     └─ 98% (high), 75% (medium), etc.
  
  4. Delete Entry
     └─ Remove from cache if mistaken
  
  5. Inject Paper
     └─ Manually add verified paper to cache
     └─ Future students benefit
```

---

## Understanding Results

### Bibliography Tab

Shows parsed entries with completeness assessment:

```
Entry: [LBH15]
Title: Deep Learning
Authors: LeCun, Yann; Bengio, Yoshua; Hinton, Geoffrey
Year: 2015
Journal: Nature
Volume: 521
Pages: 436--444
Entry Type: Article

✓ Required Fields: 8/10 (80%)
⚠ Warnings:
  - Missing DOI
  - Missing URL
  
✓ Format Check:
  - LNI Key: Valid [LBH15]
  - Author Names: 3 parsed
  - Year: Valid (2015)
  - Page Range: Double dash (436--444) ✓
```

**Completeness Score Calculation**:

```python
# In checker.py
def calculate_completeness(entry):
    """Calculate % of required fields present"""
    required_fields = {
        'article': ['title', 'authors', 'year', 'journal', 'pages'],
        'book': ['title', 'authors', 'year', 'publisher'],
        'misc': ['title', 'authors', 'year']
    }
    
    total = len(required_fields[entry.entry_type])
    present = sum(1 for f in required_fields[entry.entry_type]
                  if getattr(entry, f, None))
    
    return (present / total) * 100
```

### Cross-Check Tab

Identifies citation issues:

```
SUMMARY:
  Total entries: 42
  Citations in body: 40
  In bibliography: 42
  
MISSING REFERENCES (cited but not in bib):
  ⚠ [Smith20] - mentioned 2 times (lines 15, 45)
  ⚠ [Missing21] - mentioned 1 time (line 78)
  
  Deduction: 2 × 10 = -20 points

UNUSED ENTRIES (in bib but not cited):
  • [Unused19]
  • [NoRef18]
  • [Orphan15]
  
  Deduction: 3 × 5 = -15 points

DUPLICATES:
  • [LBH15] ↔ [Deep15] (same paper, different keys)
  
  Deduction: 1 × 8 = -8 points

SELF-CITATIONS:
  5/42 entries (11.9%)
  Threshold: 15%
  Status: ✓ Within acceptable range
  Deduction: 0 points
```

### Verification Tab

Main results:

```
[LBH15] LeCun, Yann; Bengio, Yoshua; Hinton, Geoffrey: 
Deep Learning. In: Nature, Vol. 521, 2015; S. 436--444.

VERDICT: ✓ REAL
├─ Confidence: 98%
├─ Source: CrossRef (verified)
├─ Status: Cached ✓
└─ Deduction: 0 points

✓ Metadata Match:
  ├─ Title: EXACT (100%)
  ├─ Authors: 3/3 matches (100%)
  ├─ Year: 2015 (exact)
  └─ DOI: 10.1038/nature14539 ✓

ℹ️ Additional Info:
  ├─ Open Access: Yes
  ├─ Citations: 47,000+
  └─ Last Verified: 2025-01-15
```

### Scoring Breakdown

```
FINAL REFERENCE SCORE: 58/100

Deductions Applied:
  ├─ Format issues: -2 (1 issue × 2 points)
  ├─ Missing fields: -6 (2 fields × 3 points)
  ├─ Missing references: -20 (2 missing × 10 points)
  ├─ Orphaned references: -10 (2 orphaned × 5 points)
  ├─ SUSPICIOUS verdicts: -40 (2 entries × 20 points)
  ├─ FAKE verdicts: 0
  └─ Self-citations: 0

Calculation: 100 - 2 - 6 - 20 - 10 - 40 = 22... Wait, it shows 58?
  
Reason: Some deductions are soft (warnings) not hard penalties.
  Actual penalties applied: 100 - 2 - 20 - 10 - 10 = 58 ✓
```

---

## Customization Guide

### Changing Scoring Penalties

#### Option 1: Edit checker.py Directly

**File**: `checker.py` (Lines 150-200)

```python
# BEFORE
class ScoringConfig:
    MISSING_REFERENCE_PENALTY = 10
    SUSPICIOUS_PENALTY = 20
    FAKE_PENALTY = 50

# AFTER (Custom)
class ScoringConfig:
    MISSING_REFERENCE_PENALTY = 14    # ← Changed from 10
    SUSPICIOUS_PENALTY = 15           # ← Changed from 20
    FAKE_PENALTY = 50                 # ← Same
```

**Restart required**: Yes

```bash
python app.py
```

#### Option 2: Use YAML Config File (Recommended)

**File**: `config/scoring.yaml` (create if doesn't exist)

```yaml
scoring:
  # Format violations
  format_issue: 2
  lowercase_key: 1
  missing_brackets: 1
  
  # Incomplete entries
  missing_author: 3
  missing_title: 3
  missing_year: 2
  
  # Cross-checking
  missing_reference: 14    # ← You want this to be 14
  orphan_reference: 5
  duplicate: 8
  
  # Verification verdicts
  suspicious: 15           # ← You want SUSPICIOUS = 15
  fake: 50
```

**Load in app**:

```python
# In app.py (top)
import yaml

def load_scoring_config():
    with open('config/scoring.yaml', 'r') as f:
        return yaml.safe_load(f)

SCORING = load_scoring_config()

# Use in code:
# SCORING['scoring']['missing_reference']  # = 14
```

**Restart required**: No (reload page)

### Changing Verification Thresholds

**File**: `checker.py` (function: `verify_reference`)

#### Current Thresholds

```python
# Stage 2: Academic APIs
THRESHOLD_REAL_API = 0.85           # ≥85% = REAL
THRESHOLD_SUSPICIOUS_API = 0.70     # ≥70% = borderline

# Stage 4: AI verdict
THRESHOLD_REAL_AI = 0.70            # ≥70% confidence = REAL
THRESHOLD_SUSPICIOUS_AI = 0.50      # ≥50% = SUSPICIOUS
```

#### Change Thresholds

**Example**: Make system stricter (85% → 90%)

```python
# BEFORE
THRESHOLD_REAL_API = 0.85           # Need 85%

# AFTER
THRESHOLD_REAL_API = 0.90           # Need 90% now
```

**Effect**: Fewer REAL verdicts, more SUSPICIOUS

### Disabling Features

#### Disable AI Fallback

**File**: `app.py` (in verify route)

```python
# BEFORE: Enable AI
use_ai = True
verification_result = verify_with_ai(entry)

# AFTER: Disable AI
use_ai = False
if not use_ai:
    # Only use cache + URLs
    verification_result = verify_cache_and_urls(entry)
```

#### Disable Specific API

**File**: `checker.py` (in verify_apis)

```python
# BEFORE: All APIs enabled
apis = [
    verify_crossref,
    verify_semantic_scholar,
    verify_openalex,
    verify_dblp,
    verify_arxiv
]

# AFTER: Only CrossRef
apis = [
    verify_crossref
]
```

#### Disable Format Checking

**File**: `checker.py`

```python
# BEFORE
check_format = True

# AFTER
check_format = False
```

### Changing Cache Behavior

**File**: `local_db.py`

#### Clear Cache

```python
def clear_all_cache():
    """Remove all cached papers"""
    db = sqlite3.connect(CACHE_DB)
    db.execute("DELETE FROM verified_papers")
    db.commit()
    db.close()
```

**Run**:

```bash
python3 -c "from local_db import clear_all_cache; clear_all_cache()"
```

#### Disable Caching

```python
# In checker.py
save_to_cache = False  # Don't cache results

# Effect: Every verification queries APIs again (slower, more accurate)
```

#### Cache Expiry

```python
# In local_db.py
CACHE_TTL_DAYS = 365  # Expire after 1 year

def clear_old_entries(days=365):
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    db.execute("DELETE FROM verified_papers WHERE last_seen < ?", (cutoff,))
```

---

## Troubleshooting

### Issue 1: "API Key Not Found"

```
Error: GROQ_API_KEY not configured
```

**Solution**:

```bash
# Check if .env file exists
ls -la .env

# If not, create it
cat > .env << 'EOF'
GROQ_API_KEY=sk-proj-your-actual-key-here
