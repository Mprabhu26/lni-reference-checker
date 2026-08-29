#!/usr/bin/env python3
"""Test improved URL extraction."""

import re

def test_url_extraction(raw_text, description):
    """Test the new URL extraction logic."""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"Raw text: {raw_text[:100]}")
    print(f"{'='*70}")
    
    url_patterns = [
        # Pattern 1: Domain followed by path segments (including alphanumeric after spaces/newlines)
        r'(?:https?://)?(?:www\.)?[\w-]+\.(?:com|org|net|de|eu|io|gov)(?:/[\w\-./]+)*(?:\s+[\w\-./]+)*',
        # Pattern 2: Standard domain.tld pattern with optional path
        r'(?:https?://)?(?:www\.)?[\w-]+\.(?:com|org|net|de|eu|io|gov)\S*',
        # Pattern 3: Just domain.tld
        r'[\w-]+\.(?:com|org|net|de|eu|io|gov)',
    ]
    
    url = None
    for i, pattern in enumerate(url_patterns, 1):
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            url_candidate = match.group(0).strip()
            print(f"  Pattern {i} matched: '{url_candidate}'")
            
            # Restore spaces in path (common in PDFs where path is split)
            url_candidate = re.sub(r'\s+([a-zA-Z0-9])', r'/\1', url_candidate)
            print(f"  After space handling: '{url_candidate}'")
            
            # Remove trailing punctuation
            url_candidate = url_candidate.rstrip('.,;:)]}')
            if not url_candidate.lower().startswith(('http://', 'https://')):
                url_candidate = 'https://' + url_candidate
            
            url = url_candidate
            print(f"  Final URL: '{url}'")
            break
    
    return url

# Test cases
test_cases = [
    (
        "European Parliament: EU AI Act: first regulation on artificial intelligence. europarl.europa.eu/topics/en/article/20230601 STO93804/eu-ai-act-first-regulation-on-artificial-intelligence. – Accessed: 2025-04-10",
        "EU Parliament (truncated in extraction)"
    ),
    (
        "Lang Chain: Introduction. python.langchain.com/docs/tutorials/. Accessed: 2025-04-14",
        "LangChain documentation"
    ),
    (
        "Domain with path. www.example.org/path/to/resource/ – Accessed: 2024-01-01",
        "Domain with www prefix"
    ),
]

for raw_text, desc in test_cases:
    url = test_url_extraction(raw_text, desc)
    print(f"✓ Extracted: {url}")
