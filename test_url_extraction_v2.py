#!/usr/bin/env python3
"""Test improved URL extraction v2."""

import re

def test_url_extraction(raw_text, description):
    """Test the new URL extraction logic."""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"Raw text: {raw_text}")
    print(f"{'='*70}")
    
    # Look for: [scheme://][www.]domain.tld[/path...][space/punct keywords]
    # Stop at known keywords that signal end of URL
    domain_match = re.search(
        r'(?:https?://)?(?:www\.)?[\w-]+\.(?:com|org|net|de|eu|io|gov)',
        raw_text, re.IGNORECASE)
    
    entry_url = None
    if domain_match:
        domain = domain_match.group(0)
        domain_start = domain_match.start()
        domain_end = domain_match.end()
        
        print(f"  Domain found: '{domain}'")
        
        # Extract path after the domain, stopping at keywords or punctuation
        after_domain = raw_text[domain_end:]
        print(f"  After domain: '{after_domain[:60]}'")
        
        # Match path segments until we hit "Accessed", "–", period-space, etc.
        path_match = re.match(
            r'((?:/[\w\-./]+)*)\s*(?:–|Accessed|Abruf|Stand|accessed)\b',
            after_domain, re.IGNORECASE)
        
        if path_match:
            path = path_match.group(1)
            print(f"  Path (from keyword stop): '{path}'")
            entry_url = domain + path
        else:
            # No path match, just use domain or look for minimal path
            path_match = re.match(r'(/[^\s\.]*)', after_domain)
            if path_match:
                path = path_match.group(1).rstrip('.,;:)]}')
                print(f"  Path (minimal): '{path}'")
                entry_url = domain + path
            else:
                print(f"  No path found, using domain only")
                entry_url = domain
        
        # Add scheme if missing
        if not entry_url.lower().startswith(('http://', 'https://')):
            entry_url = 'https://' + entry_url
    
    return entry_url

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
    print(f"✓ Final URL: {url}")
