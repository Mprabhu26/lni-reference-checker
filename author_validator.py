"""
TIER 2: Enhanced Author Validation
Purpose: Detect suspicious author patterns that indicate fake references
Version: 1.0
"""

import re
from typing import Tuple, List, Dict, Optional


# Known fake/placeholder author patterns
SUSPICIOUS_AUTHOR_PATTERNS = {
    # Direct fakes
    "ghost", "fake", "test", "example", "placeholder", "anonymous",
    "unknown", "nobody", "someone", "author", "dummy", "sample", 
    "demo", "null", "none", "invalid",
    
    # Research team patterns
    "research team", "group members", "contributors", "authors",
    "team", "lab", "authors list", "et al",
    
    # Obviously made up
    "aaa", "bbb", "ccc", "john doe", "jane smith", "first author",
    "second author", "test user", "admin", "student", "researcher",
    
    # Bot/system patterns
    "bot", "script", "system", "machine", "computer", "ai",
    "algorithm", "model", "code", "github contributors",
}

# Real academic surnames (samples - much more comprehensive list could be used)
COMMON_ACADEMIC_SURNAMES = {
    "smith", "johnson", "brown", "miller", "davis", "wilson",
    "moore", "taylor", "anderson", "thomas", "jackson", "white",
    "harris", "martin", "thompson", "garcia", "martinez", "robinson",
    "clark", "rodriguez", "lewis", "lee", "walker", "hall",
    "allen", "young", "king", "wright", "lopez", "hill",
    "scott", "green", "adams", "nelson", "carter", "roberts",
    "phillips", "campbell", "parker", "evans", "edwards", "collins",
    # Academic-heavy surnames
    "einstein", "curie", "darwin", "newton", "hawking", "feynman",
    "tutte", "dijkstra", "knuth", "turing", "church", "godel",
    "papadimitriou", "sipser", "arora", "barak", "spielman",
    # European academics
    "mueller", "schmidt", "weber", "fischer", "schwartz", "kahn",
    "jensen", "larsen", "petersen", "anderson", "johnsson",
    "rossi", "russo", "bianchi", "colombo", "ferrari", "bruno",
    # Asian academics
    "wang", "zhang", "liu", "chen", "yang", "wu", "zhou", "xu",
    "sun", "tang", "shi", "li", "zhu", "ma", "huang",
    # Verify by presence of real academic patterns
}


def score_author_plausibility(authors_str: str) -> Tuple[float, str]:
    """
    Score author string plausibility (0.0-1.0).
    
    Returns: (score, reason)
    - 0.0: Definitely suspicious (fake patterns detected)
    - 0.3-0.7: Questionable (some red flags)
    - 0.85+: Plausible (looks like real author)
    
    High score (0.8+) = confidence that author is real
    Low score (0.2-) = high likelihood of fabrication
    """
    if not authors_str or not authors_str.strip():
        return 0.0, "Empty author field"
    
    authors_lower = authors_str.lower().strip()
    
    # Check for direct fake patterns
    for pattern in SUSPICIOUS_AUTHOR_PATTERNS:
        if pattern in authors_lower or authors_lower == pattern:
            return 0.0, f"Matches fake author pattern: '{pattern}'"
    
    # Parse first author (most important for LNI-style keys)
    parts = authors_str.split(";")
    if not parts:
        return 0.0, "Cannot parse author field"
    
    first_author = parts[0].strip()
    if not first_author:
        return 0.2, "Empty first author"
    
    # Split by comma (Last, First format) or space (First Last format)
    if "," in first_author:
        # Assumed format: "Lastname, Firstname"
        surname_part = first_author.split(",")[0].strip().lower()
    else:
        # Assumed format: "Firstname Lastname"
        author_words = first_author.split()
        surname_part = author_words[-1].lower() if author_words else ""
    
    if not surname_part:
        return 0.3, "Cannot extract surname"
    
    # Length check - real surnames are usually 2-15 chars
    if len(surname_part) < 2:
        return 0.1, f"Surname too short: '{surname_part}'"
    if len(surname_part) > 20:
        return 0.4, f"Surname unusually long: '{surname_part}' ({len(surname_part)} chars)"
    
    # Check for repeated characters (suspicious)
    if re.search(r'(.)\1{2,}', surname_part):  # aaa, bbbb, etc.
        return 0.0, f"Repeated characters in surname: '{surname_part}'"
    
    # Check for numeric surname (rare but possible)
    if surname_part.isdigit():
        return 0.2, f"Surname is all numeric: '{surname_part}'"
    
    # Check for mixed case or ALL CAPS (suspicious for surname)
    first_char_upper = first_author[0].isupper()
    all_caps = first_author.isupper()
    all_lower = first_author.islower()
    
    if all_lower:
        return 0.5, f"Author entirely lowercase: '{first_author}'"
    if all_caps and len(first_author) > 3:
        return 0.6, f"Author entirely uppercase: '{first_author}'"
    
    # Check for common academic surnames
    if surname_part in COMMON_ACADEMIC_SURNAMES:
        return 0.92, f"Surname matches common academic names: '{surname_part}'"
    
    # Check for vowel-consonant pattern (real surnames have these)
    vowels = sum(1 for c in surname_part if c in 'aeiou')
    if vowels == 0:
        return 0.3, f"No vowels in surname: '{surname_part}' (unusual)"
    if vowels == len(surname_part):
        return 0.3, f"All vowels in surname: '{surname_part}' (unusual)"
    
    # Default: seems plausible
    base_score = 0.80
    
    # Bonus for multiple authors (more credible)
    author_count = len(parts)
    if author_count >= 2:
        base_score += 0.08
    if author_count >= 3:
        base_score += 0.05
    
    # Cap at 0.99
    final_score = min(0.99, base_score)
    return final_score, f"Plausible author(s): {author_count} author(s), surname '{surname_part}'"


def validate_author_consistency(
    entry_authors: str, 
    api_authors: str
) -> Tuple[float, str]:
    """
    Check if entry authors match API-returned authors (metadata consistency).
    
    Returns: (consistency_score, reason)
    - 0.95+: Perfect match
    - 0.7-0.95: Minor differences (abbreviations, ordering)
    - 0.5-0.7: Moderate differences (missing some authors)
    - <0.5: Major discrepancy (might be wrong paper)
    """
    if not entry_authors or not api_authors:
        return 0.5, "Missing author data for comparison"
    
    entry_lower = entry_authors.lower().strip()
    api_lower = api_authors.lower().strip()
    
    # Exact match
    if entry_lower == api_lower:
        return 0.99, "Authors match exactly"
    
    # Extract surnames for fuzzy matching
    def extract_surnames(auth_str: str) -> set:
        """Extract all surnames from author string"""
        surnames = set()
        for author in auth_str.split(";"):
            author = author.strip()
            if not author:
                continue
            if "," in author:
                surname = author.split(",")[0].strip().lower()
            else:
                parts = author.split()
                surname = parts[-1].lower() if parts else ""
            if surname:
                surnames.add(surname)
        return surnames
    
    entry_surnames = extract_surnames(entry_authors)
    api_surnames = extract_surnames(api_authors)
    
    if not entry_surnames or not api_surnames:
        return 0.3, "Cannot extract surnames for comparison"
    
    # Calculate overlap
    overlap = entry_surnames & api_surnames
    all_surnames = entry_surnames | api_surnames
    
    if len(all_surnames) == 0:
        return 0.3, "No surnames found"
    
    overlap_ratio = len(overlap) / len(all_surnames)
    
    if overlap_ratio >= 0.95:
        return 0.95, f"Authors match well ({len(overlap)}/{len(all_surnames)} surnames)"
    elif overlap_ratio >= 0.80:
        return 0.85, f"Most authors match ({len(overlap)}/{len(all_surnames)} surnames)"
    elif overlap_ratio >= 0.60:
        return 0.70, f"Some authors match ({len(overlap)}/{len(all_surnames)} surnames)"
    elif overlap_ratio >= 0.40:
        return 0.50, f"Few authors match ({len(overlap)}/{len(all_surnames)} surnames)"
    else:
        return 0.30, f"Authors significantly different ({len(overlap)}/{len(all_surnames)} surnames)"


def get_author_validation_report(
    entry_authors: str,
    api_authors: Optional[str] = None
) -> Dict:
    """
    Generate comprehensive author validation report.
    
    Returns dict with:
    - entry_plausibility: (score, reason)
    - consistency: (score, reason) if api_authors provided
    - overall_score: weighted average
    - warnings: list of warnings
    - confidence_adjustment: -X to +Y percent adjustment to reference confidence
    """
    plausibility, plausibility_reason = score_author_plausibility(entry_authors)
    
    warnings = []
    if plausibility < 0.5:
        warnings.append(f"Author plausibility LOW: {plausibility_reason}")
    
    consistency_score = 1.0
    consistency_reason = "No API comparison"
    if api_authors:
        consistency_score, consistency_reason = validate_author_consistency(
            entry_authors, api_authors
        )
        if consistency_score < 0.7:
            warnings.append(f"Author mismatch with API: {consistency_reason}")
    
    # Overall score is weighted average
    if api_authors:
        overall_score = plausibility * 0.6 + consistency_score * 0.4
    else:
        overall_score = plausibility
    
    # Confidence adjustment
    confidence_delta = (overall_score - 0.75) * 0.15  # -15% to +15% adjustment
    
    return {
        "entry_plausibility": (plausibility, plausibility_reason),
        "consistency_with_api": (consistency_score, consistency_reason),
        "overall_score": overall_score,
        "warnings": warnings,
        "confidence_adjustment": confidence_delta,
    }
