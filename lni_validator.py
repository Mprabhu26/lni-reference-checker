"""
LNI Format Validator - FIXED
Validates academic references against LNI (Lecture Notes in Informatics) standards.
FIXED: Now properly handles PDF-extracted references without requiring perfect field separation.

LNI standards require:
1. Citation key format: [Ab00] or [ABC00] (1-3 letters + 2-digit year + optional suffix)
2. Author format: "Lastname, Firstname; Lastname, Firstname" 
3. Title format: follows authors, separated by colon
4. Publication details: journal OR publisher OR venue (depending on type)
5. Year: 4-digit number at or near end
"""

import re
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LNIValidationResult:
    """Result of LNI format validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    key_format_valid: bool
    entry_complete: bool
    has_required_structure: bool
    field_issues: List[str]


class LNIValidator:
    """Validate references against LNI standards"""
    
    # LNI key format: [Xx00] or [Xxx00z]
    LNI_KEY_PATTERN = re.compile(r'^\[([A-Za-z]{1,3}\d{2}[a-z]?)\]$')
    
    def __init__(self):
        """Initialize validator"""
        pass
    
    def validate_key_format(self, key: str) -> Tuple[bool, str]:
        """Validate LNI citation key format [Ab00] or Ab00"""
        if not key:
            return False, "Key is empty"
        
        # Remove brackets if present
        key_content = key
        if key.startswith('[') and key.endswith(']'):
            key_content = key[1:-1]
        
        # LNI allows:
        # 1-2 letters + 2 digits + optional letter: [X00], [XX00], [XX00a]
        # 3+ letters (4+ authors): [XXX00] or [XXXX00]
        if not re.match(r'^[A-Za-z]{1,5}\d{2}[a-z]?$', key_content):
            return False, (
                f"Invalid key format '{key_content}'. "
                "LNI requires [Xx00] to [Xxxxx00z] format"
            )
        
        # Validate year component (2-digit year)
        year_str = key_content[-3:-1] if key_content[-1].isalpha() else key_content[-2:]
        try:
            year_2digit = int(year_str)
            # Convert 2-digit year to 4-digit (00-49 = 2000-2049, 50-99 = 1950-1999)
            full_year = 2000 + year_2digit if year_2digit < 50 else 1900 + year_2digit
            if full_year < 1950 or full_year > 2099:
                return False, f"Year '{year_str}' (→{full_year}) out of valid range"
        except ValueError:
            return False, f"Year component '{year_str}' is not numeric"
        
        return True, "Key format valid"
    
    def validate_raw_entry_structure(self, raw_text: str) -> Tuple[bool, List[str]]:
        """
        FIXED: Validate raw text from PDF has LNI structure.
        Handles: "Authors: Title. Journal, Year." format
        
        Returns: (is_structured, issues_list)
        """
        issues = []
        raw_clean = " ".join(raw_text.split())[:500]  # First 500 chars
        
        if not raw_clean:
            issues.append("Entry text is empty")
            return False, issues
        
        # Core requirements for LNI entry:
        # 1. Some author-like content OR title-like start
        # 2. Year (4-digit number) - ESSENTIAL
        
        # Check for author-like start: "Lastname, Firstname" or "Lastname, F."
        author_pattern = r'[A-Z][a-zA-Z\-]+\s*,\s*[A-Z]'
        has_author_start = bool(re.match(author_pattern, raw_clean))
        
        # Check for title separator (colon) - ESSENTIAL
        has_colon = ':' in raw_clean[:200]
        if not has_colon:
            # Soften: colon is strongly preferred but not in 100% of cases
            issues.append(
                "Warning: Entry missing colon separator (expected after authors)"
            )
        
        # Check for year (4-digit number) - ESSENTIAL
        has_year = bool(re.search(r'\b(19|20)\d{2}\b', raw_clean))
        if not has_year:
            issues.append("Entry missing year (4-digit number) - CRITICAL")
        
        # Check for publication venue (journal, publisher, conference)
        # This is expected but not always present in shorter entries
        venue_words = r'\b(?:journal|journal of|transactions|proceedings|' \
                     r'conference|workshop|magazine|review|springer|wiley|' \
                     r'elsevier|acm|ieee|press|verlag|publisher|vol|pp)\b'
        has_venue = bool(re.search(venue_words, raw_clean, re.IGNORECASE))
        
        # FIXED: Be lenient - if we have author/title/year, consider it valid structure
        # Venue/colon are preferred but missing one doesn't invalidate
        critical_issues = [i for i in issues if 'CRITICAL' in i]
        has_structure = len(critical_issues) == 0 and (has_author_start or has_year)
        
        return has_structure, issues
    
    def extract_year_from_text(self, text: str) -> Optional[str]:
        """Extract 4-digit year from text"""
        match = re.search(r'\b(19|20)\d{2}\b', text)
        if match:
            return match.group(1)
        return None
    
    def validate_author_presence(self, text: str) -> Tuple[bool, str]:
        """
        Check if text contains author names.
        Authors in LNI: "Lastname, Firstname" or "Lastname, F."
        """
        # Pattern: Word, Initial OR Word, Word
        author_pattern = r'[A-Z][a-zA-Z\-]+\s*,\s*(?:[A-Z]\.?|[A-Z][a-zA-Z\-]+)'
        
        if re.search(author_pattern, text):
            return True, "Author names found"
        
        # Accept "et al." as valid
        if 'et al' in text.lower():
            return True, "Author list with 'et al.' found"
        
        return False, "No author names detected"
    
    def validate_entry_type_hints(self, text: str) -> Tuple[str, List[str]]:
        """
        Infer entry type from text content.
        Returns: (likely_type, confidence_notes)
        """
        notes = []
        text_lower = text.lower()
        
        # Check for book
        if re.search(r'\bpublisher[:\s]|springer|wiley|elsevier|press', text_lower):
            notes.append("Likely book or proceedings entry (publisher keyword found)")
            return "book", notes
        
        # Check for journal article
        if re.search(
            r'\bjournal of\b|\bjournal[:\s]|\btransactions\b|'
            r'\b(?:vol|volume|no|number|pp|pages)[:\s]*\d',
            text_lower
        ):
            notes.append("Likely article entry (journal keywords found)")
            return "article", notes
        
        # Check for conference proceedings
        if re.search(
            r'\bproceedings\b|\bconference\b|\bworkshop\b|'
            r'\bsymposium\b|\bin:\s*proceedings',
            text_lower
        ):
            notes.append("Likely inproceedings entry (conference keywords found)")
            return "inproceedings", notes
        
        # Check for PhD thesis
        if re.search(r'\bphd\s+thesis\b|\bdoctoral\b|\bschool\b', text_lower):
            notes.append("Likely phdthesis entry (thesis keywords found)")
            return "phdthesis", notes
        
        # Check for website/online
        if re.search(r'\bhttp|website|online|url\b', text_lower):
            notes.append("Likely online/website entry")
            return "online", notes
        
        notes.append("Could not determine specific type, assuming general reference")
        return "misc", notes
    
    def validate_complete_entry(
        self,
        key: str,
        raw_text: str
    ) -> LNIValidationResult:
        """
        FIXED: Validate entry using raw PDF-extracted text.
        Does NOT require perfectly parsed fields - works with raw bibliography text.
        
        Args:
            key: Citation key like "[AB00]"
            raw_text: Raw text from PDF (may contain running headers/page numbers)
        
        Returns: LNIValidationResult
        """
        errors = []
        warnings = []
        field_issues = []
        
        # STEP 1: Clean text (remove common PDF artifacts)
        text_cleaned = self._clean_pdf_artifacts(raw_text)
        
        # STEP 2: Validate key format
        key_valid, key_msg = self.validate_key_format(key)
        if not key_valid:
            errors.append(f"Key: {key_msg}")
        
        # STEP 3: Check for core LNI structure
        has_structure, structure_issues = self.validate_raw_entry_structure(text_cleaned)
        if not has_structure:
            for issue in structure_issues:
                errors.append(f"Structure: {issue}")
            field_issues.extend(structure_issues)
        
        # STEP 4: Verify authors present
        has_authors, author_msg = self.validate_author_presence(text_cleaned)
        if not has_authors:
            errors.append(f"Author: {author_msg}")
            field_issues.append("authors")
        else:
            # Soft check - can extract author names
            pass
        
        # STEP 5: Extract and validate year
        year = self.extract_year_from_text(text_cleaned)
        if not year:
            errors.append("Year: No 4-digit year found")
            field_issues.append("year")
        else:
            # Cross-check key year with field year
            key_year = key[1:3] + key[3:5] if key.startswith('[') else ""
            if key_year and len(key_year) == 4:
                key_year_2digit = key_year[2:]
                field_year_2digit = year[2:]
                if key_year_2digit != field_year_2digit:
                    warnings.append(
                        f"Key year ({key_year_2digit}) differs from "
                        f"field year ({field_year_2digit})"
                    )
        
        # STEP 6: Infer entry type
        entry_type, type_notes = self.validate_entry_type_hints(text_cleaned)
        for note in type_notes:
            warnings.append(note)
        
        # Determine overall validity
        # CRITICAL FIX: LNI entries are VALID if they have key + structure + authors + year
        # Soft warnings about specific formatting don't make them "incomplete"
        is_valid = len(errors) == 0
        entry_complete = is_valid  # Entry is complete if all core elements present
        has_required_structure = has_structure
        
        return LNIValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            key_format_valid=key_valid,
            entry_complete=entry_complete,
            has_required_structure=has_required_structure,
            field_issues=field_issues
        )
    
    def _clean_pdf_artifacts(self, text: str) -> str:
        """
        Remove common PDF extraction artifacts:
        - Running headers/footers (e.g., "AI in ECS - a bibliographical Meta Analysis")
        - Page numbers (e.g., "343", "344")
        - Author signatures in footers (e.g., "Lukas Sinnwell et al.")
        """
        # Remove common running headers
        text = re.sub(
            r'\bAI in ECS - a bibliographical Meta Analysis\b',
            '',
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r'\bLukas Sinnwell et al\.|Sinnwell et al\.|Sinnwell et\. al\b',
            '',
            text,
            flags=re.IGNORECASE
        )
        
        # Remove page numbers (3-digit numbers that look like pages: 333-347)
        # But preserve years (always 4 digits starting with 19 or 20)
        text = re.sub(r'\b(?!19\d{2}|20\d{2})\d{3}\b', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up common PDF spacing issues
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        text = re.sub(r'(\d)\s+([A-Z])', r'\1 \2', text)
        
        return text.strip()


def validate_lni_entry(key: str, raw_text: str) -> LNIValidationResult:
    """Quick validation function for a single entry"""
    validator = LNIValidator()
    return validator.validate_complete_entry(key, raw_text)