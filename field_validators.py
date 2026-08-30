"""
Field validators and LNI standard compliance checks.

These validators produce WARNINGS, not verdicts. They do NOT change the 
REAL/SUSPICIOUS/FAKE decision made by the main pipeline. They are 
informational layers for the UI and professor review.
"""

import re
from typing import List, Dict, Optional, Tuple
from parser import BibEntry


# ============================================================================
# ISBN / ISSN Validation
# ============================================================================

def validate_isbn(isbn: str) -> Tuple[bool, str]:
    """
    Validate ISBN-10 or ISBN-13 checksum.
    Returns: (is_valid, message)
    """
    if not isbn:
        return True, ""
    
    isbn_clean = re.sub(r'[\s\-]', '', isbn)
    
    # ISBN-13 validation
    if len(isbn_clean) == 13 and isbn_clean.isdigit():
        try:
            total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(isbn_clean[:12]))
            check_digit = (10 - (total % 10)) % 10
            if int(isbn_clean[-1]) == check_digit:
                return True, ""
            else:
                return False, f"ISBN-13 checksum invalid (got {isbn_clean[-1]}, expected {check_digit})"
        except ValueError:
            return False, "ISBN-13 contains non-digit characters"
    
    # ISBN-10 validation
    if len(isbn_clean) == 10:
        if isbn_clean[:-1].isdigit() and (isbn_clean[-1].isdigit() or isbn_clean[-1].upper() == 'X'):
            try:
                total = sum(int(d) * (10 - i) for i, d in enumerate(isbn_clean[:9]))
                check = isbn_clean[-1]
                expected = (10 - (total % 10)) % 10
                expected_str = 'X' if expected == 10 else str(expected)
                if check.upper() == expected_str:
                    return True, ""
                else:
                    return False, f"ISBN-10 checksum invalid (got {check}, expected {expected_str})"
            except ValueError:
                return False, "ISBN-10 contains invalid characters"
    
    if len(isbn_clean) in [10, 13]:
        return False, f"Invalid ISBN format: {isbn}"
    
    return False, f"ISBN length invalid: {len(isbn_clean)} (expected 10 or 13)"


def validate_issn(issn: str) -> Tuple[bool, str]:
    """
    Validate ISSN format and checksum.
    Returns: (is_valid, message)
    """
    if not issn:
        return True, ""
    
    issn_clean = re.sub(r'[\s\-]', '', issn)
    
    # ISSN should be 8 characters: 7 digits + check digit (or X)
    if len(issn_clean) != 8:
        return False, f"ISSN should be 8 characters, got {len(issn_clean)}"
    
    if not issn_clean[:-1].isdigit():
        return False, "ISSN first 7 characters must be digits"
    
    if not (issn_clean[-1].isdigit() or issn_clean[-1].upper() == 'X'):
        return False, "ISSN check digit must be digit or X"
    
    # ISSN checksum validation (using modulo 11)
    try:
        total = sum(int(d) * (8 - i) for i, d in enumerate(issn_clean[:7]))
        check = issn_clean[-1]
        expected = (11 - (total % 11)) % 11
        expected_str = 'X' if expected == 10 else str(expected)
        if check.upper() == expected_str:
            return True, ""
        else:
            return False, f"ISSN checksum invalid (got {check}, expected {expected_str})"
    except ValueError:
        return False, "ISSN validation error"


def validate_doi_format(doi: str) -> Tuple[bool, str]:
    """
    Validate DOI format (basic structure).
    Returns: (is_valid, message)
    
    DOI format: 10.xxxx/yyyy where xxxx is registrant code, yyyy is suffix.
    """
    if not doi:
        return True, ""
    
    # Remove common DOI prefixes
    doi_clean = doi
    doi_clean = re.sub(r'^(?:https?://)?(?:dx\.)?doi\.org/', '', doi_clean, flags=re.IGNORECASE)
    doi_clean = re.sub(r'^(?:DOI:\s*|doi:\s*)', '', doi_clean, flags=re.IGNORECASE)
    doi_clean = doi_clean.strip()
    
    # DOI must start with 10.
    if not doi_clean.startswith('10.'):
        return False, f"DOI must start with '10.', got: {doi_clean[:20]}"
    
    # Must have at least one slash after 10.
    parts = doi_clean.split('/', 1)
    if len(parts) < 2:
        return False, f"DOI missing slash separator: {doi_clean}"
    
    registrant = parts[0][3:]  # Everything after '10.'
    if not registrant or not registrant.isdigit():
        return False, f"DOI registrant code must be numeric: {registrant}"
    
    suffix = parts[1]
    if not suffix or len(suffix) < 2:
        return False, f"DOI suffix too short: {suffix}"
    
    return True, ""


def validate_url_accessible(url: str) -> Tuple[bool, str]:
    """
    Quick check if URL has valid format (not actual HTTP check).
    Returns: (is_valid, message)
    """
    if not url:
        return True, ""
    
    # Check basic URL format
    if not re.match(r'^https?://', url, re.IGNORECASE):
        return False, "URL must start with http:// or https://"
    
    # Must have a domain
    if not re.search(r'(?:[\w-]+\.)+(?:com|org|net|edu|gov|de|eu|io|co\.uk|ac\.uk)\b', url, re.IGNORECASE):
        return False, "URL lacks recognizable domain"
    
    # Check for obviously broken URLs (common PDF extraction issues)
    if re.search(r'\s', url):
        return False, "URL contains whitespace (likely broken by PDF extraction)"
    
    if len(url) > 2000:
        return False, "URL suspiciously long (likely corrupted)"
    
    return True, ""


# ============================================================================
# Entry Type Consistency Checks
# ============================================================================

def check_entry_type_consistency(entry: BibEntry) -> List[Dict]:
    """
    Check if entry type matches its metadata.
    Returns list of warnings: {type, severity, message}
    """
    warnings = []
    entry_type = (entry.entry_type or "unknown").lower()
    
    # Journal article should have journal OR volume
    if entry_type == "article":
        if not entry.journal and not entry.volume:
            warnings.append({
                "type": "missing_journal_info",
                "severity": "warn",
                "message": "Journal article missing journal name or volume number"
            })
        if not entry.pages:
            warnings.append({
                "type": "missing_pages",
                "severity": "warn",
                "message": "Journal article should include page numbers"
            })
    
    # Conference paper should have booktitle or conference indicator
    elif entry_type in ("proceedings", "inproceedings"):
        if not entry.booktitle and "conference" not in (entry.raw_text or "").lower():
            warnings.append({
                "type": "missing_booktitle",
                "severity": "warn",
                "message": "Conference paper missing booktitle (venue name)"
            })
        if not entry.pages:
            warnings.append({
                "type": "missing_pages",
                "severity": "warn",
                "message": "Conference paper should include page numbers"
            })
    
    # Book must have publisher
    elif entry_type == "book":
        if not entry.publisher:
            warnings.append({
                "type": "missing_publisher",
                "severity": "error",
                "message": "Book entry missing publisher (required for LNI)"
            })
    
    # Website should have URL
    elif entry_type == "website":
        if not entry.url:
            warnings.append({
                "type": "missing_url",
                "severity": "error",
                "message": "Website entry missing URL"
            })
    
    # Entry type unknown — AI parsing likely needed
    if entry_type == "unknown":
        warnings.append({
            "type": "unknown_entry_type",
            "severity": "warn",
            "message": "Entry type could not be determined; manual review recommended"
        })
    
    return warnings


# ============================================================================
# Conference Abbreviation Validation
# ============================================================================

KNOWN_CONFERENCES = {
    "NeurIPS": {"abbr": ["NeurIPS", "NIPS"], "domain": "machine-learning", "since": 1987},
    "ICML": {"abbr": ["ICML"], "domain": "machine-learning", "since": 1988},
    "ICLR": {"abbr": ["ICLR"], "domain": "machine-learning", "since": 2013},
    "CVPR": {"abbr": ["CVPR"], "domain": "computer-vision", "since": 1983},
    "ICCV": {"abbr": ["ICCV"], "domain": "computer-vision", "since": 1987},
    "ECCV": {"abbr": ["ECCV"], "domain": "computer-vision", "since": 1990},
    "ACL": {"abbr": ["ACL"], "domain": "nlp", "since": 1961},
    "EMNLP": {"abbr": ["EMNLP"], "domain": "nlp", "since": 1996},
    "NAACL": {"abbr": ["NAACL"], "domain": "nlp", "since": 2000},
    "AAAI": {"abbr": ["AAAI"], "domain": "ai", "since": 1980},
    "IJCAI": {"abbr": ["IJCAI"], "domain": "ai", "since": 1969},
    "SIGMOD": {"abbr": ["SIGMOD"], "domain": "databases", "since": 1975},
    "VLDB": {"abbr": ["VLDB"], "domain": "databases", "since": 1975},
    "SOSP": {"abbr": ["SOSP"], "domain": "systems", "since": 1967},
    "OSDI": {"abbr": ["OSDI"], "domain": "systems", "since": 1994},
    "USENIX": {"abbr": ["USENIX", "LISA"], "domain": "systems", "since": 1984},
    "CHI": {"abbr": ["CHI"], "domain": "hci", "since": 1982},
    "SIGGRAPH": {"abbr": ["SIGGRAPH"], "domain": "graphics", "since": 1974},
    "KDD": {"abbr": ["KDD"], "domain": "data-mining", "since": 1995},
    "WWW": {"abbr": ["WWW", "IW3C2"], "domain": "web", "since": 1994},
}


def validate_conference_abbreviation(venue: str, year: str) -> List[Dict]:
    """
    Check if conference name is a known venue and if year is plausible.
    Returns list of warnings.
    """
    warnings = []
    
    if not venue:
        return warnings
    
    # Search for known conference abbreviations
    venue_upper = venue.upper()
    
    found_conf = None
    for conf_name, conf_info in KNOWN_CONFERENCES.items():
        for abbr in conf_info["abbr"]:
            if abbr.upper() in venue_upper or venue_upper in abbr.upper():
                found_conf = conf_info
                break
        if found_conf:
            break
    
    if found_conf:
        # Check year validity
        if year:
            try:
                year_int = int(year)
                since = found_conf["since"]
                if year_int < since:
                    warnings.append({
                        "type": "venue_year_mismatch",
                        "severity": "error",
                        "message": f"Conference '{venue}' did not exist before {since}, but paper is from {year}"
                    })
                elif year_int > 2035:
                    warnings.append({
                        "type": "future_year",
                        "severity": "error",
                        "message": f"Paper year {year} is implausibly far in future"
                    })
            except ValueError:
                pass
    else:
        # Unknown conference — could be legitimate or fabricated.
        # Long, legitimate journal/venue names are common (IEEE, ACM, and many
        # domain-specific journals routinely exceed 80 chars), so length alone
        # is not a useful signal and is dropped as a standalone warning.

        # Check for common fake conference/predatory-journal patterns. These
        # patterns are aimed at recognizable *fabrication* templates (vague
        # hype terms combined with grandiose framing), not at any journal
        # whose title happens to contain words like "International" or
        # "Digital" — real venues (e.g. "International Journal of Digital
        # Earth") should not match. Patterns are anchored to whole venue
        # names and require multiple red-flag terms together, not a single
        # generic word.
        fake_patterns = [
            r"^(?:The\s+)?(?:Great|Premier|Elite|Prestigious|Top-Tier)\s+(?:International\s+)?(?:Conference|Workshop|Summit)\s+on\b",
            r"^World\s+(?:Congress|Conference|Summit)\s+on\s+(?:Advanced\s+)?(?:AI|Artificial Intelligence|Blockchain|Metaverse)\b",
            r"^International\s+Conference\s+on\s+(?:Advanced\s+)?(?:AI|Artificial Intelligence|Machine Learning|Deep Learning|Quantum|Blockchain|Metaverse)\s+(?:Innovations|Breakthroughs|Excellence)\b",
        ]

        for pattern in fake_patterns:
            if re.search(pattern, venue.strip(), re.IGNORECASE):
                warnings.append({
                    "type": "suspicious_venue_name",
                    "severity": "warn",
                    "message": f"Venue name matches common fabrication patterns: {venue}"
                })
                break
    
    return warnings


# ============================================================================
# Volume/Issue Completeness
# ============================================================================

def check_volume_issue_consistency(entry: BibEntry) -> List[Dict]:
    """
    Check volume/issue/pages consistency for journal articles.
    Returns list of warnings.
    """
    warnings = []
    
    if (entry.entry_type or "").lower() != "article":
        return warnings
    
    if not entry.journal:
        return warnings
    
    # Journal article should have volume
    if not entry.volume:
        warnings.append({
            "type": "missing_volume",
            "severity": "warn",
            "message": "Journal article missing volume number"
        })
    
    # If volume exists, check for suspicious values
    if entry.volume:
        try:
            vol_int = int(entry.volume)
            if vol_int < 1 or vol_int > 1000:
                warnings.append({
                    "type": "suspicious_volume",
                    "severity": "warn",
                    "message": f"Volume number {vol_int} is outside typical range (1-1000)"
                })
            
            # Check for obviously fake volumes (repeating digits, 3+ digits only).
            # Real journals routinely have volume 1, 11, 22, etc. (e.g. vol. 11 of a
            # journal founded ~11 years ago) — only long, uniform-digit strings like
            # "111", "222", "999" are actually suspicious.
            if len(entry.volume) >= 3 and len(set(entry.volume)) == 1:
                warnings.append({
                    "type": "suspicious_repeating_volume",
                    "severity": "warn",
                    "message": f"Volume '{entry.volume}' has suspiciously repeating digits"
                })
        except ValueError:
            warnings.append({
                "type": "non_numeric_volume",
                "severity": "warn",
                "message": f"Volume should be numeric, got: {entry.volume}"
            })
    
    # Issue/number should also be checked if present
    if entry.number:
        try:
            num_int = int(entry.number)
            if num_int < 1 or num_int > 52:  # Max ~52 issues per year
                warnings.append({
                    "type": "suspicious_issue",
                    "severity": "warn",
                    "message": f"Issue number {num_int} is implausibly high (max ~52 per year)"
                })
        except ValueError:
            pass
    
    # Pages checks
    if entry.pages:
        m = re.search(r'(\d+)\s*[-–—]+\s*(\d+)', entry.pages)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            
            if start > end:
                warnings.append({
                    "type": "invalid_page_range",
                    "severity": "error",
                    "message": f"Page range invalid: start {start} > end {end}"
                })
            
            span = end - start
            if span > 50:
                warnings.append({
                    "type": "unusual_page_span",
                    "severity": "warn",
                    "message": f"Page span {span} is unusually large (likely wrong entry type)"
                })
    
    return warnings


# ============================================================================
# Overall LNI Compliance Summary
# ============================================================================

def get_lni_compliance_warnings(entry: BibEntry) -> Dict[str, List[Dict]]:
    """
    Gather all LNI compliance warnings for a single entry.
    Returns: {
        'field_validation': [...],
        'entry_type': [...],
        'venue': [...],
        'volume_issue': [...],
        'all_warnings': [...]
    }
    """
    results = {
        'field_validation': [],
        'entry_type': [],
        'venue': [],
        'volume_issue': [],
    }
    
    # ISBN validation
    if entry.isbn:
        is_valid, msg = validate_isbn(entry.isbn)
        if not is_valid:
            results['field_validation'].append({
                "type": "invalid_isbn",
                "severity": "warn",
                "message": msg
            })
    
    # ISSN validation
    if entry.journal and entry.volume:
        # Only validate if looks like ISSN (e.g., in journal field)
        issn_match = re.search(r'\b\d{4}-\d{3}[0-9X]\b', entry.journal)
        if issn_match:
            is_valid, msg = validate_issn(issn_match.group(0))
            if not is_valid:
                results['field_validation'].append({
                    "type": "invalid_issn",
                    "severity": "warn",
                    "message": msg
                })
    
    # DOI validation
    if entry.doi:
        is_valid, msg = validate_doi_format(entry.doi)
        if not is_valid:
            results['field_validation'].append({
                "type": "invalid_doi",
                "severity": "warn",
                "message": msg
            })
    
    # URL validation
    if entry.url:
        is_valid, msg = validate_url_accessible(entry.url)
        if not is_valid:
            results['field_validation'].append({
                "type": "invalid_url",
                "severity": "warn",
                "message": msg
            })
    
    # Entry type consistency
    results['entry_type'] = check_entry_type_consistency(entry)
    
    # Conference/venue checks
    venue = entry.booktitle or entry.journal or entry.publisher or ""
    results['venue'] = validate_conference_abbreviation(venue, entry.year)
    
    # Volume/Issue completeness
    results['volume_issue'] = check_volume_issue_consistency(entry)
    
    # Flatten all warnings
    results['all_warnings'] = (
        results['field_validation'] +
        results['entry_type'] +
        results['venue'] +
        results['volume_issue']
    )
    
    return results