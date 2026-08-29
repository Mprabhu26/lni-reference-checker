#!/usr/bin/env python3
"""Test improved URL extraction v4 - extract from before "Accessed" keyword."""

import re

def test_url_extraction(raw_text, description):
    """Test the new URL extraction logic."""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"Raw text: {raw_text}")
    print(f"{'='*70}")
    
    entry_url = None
    
    # Find the "Accessed" or similar keyword position
    access_pattern = r'(?:Stand:|Abruf:|abgerufen am|accessed|besucht am|Accessed:)\s*'
    access_match = re.search(access_pattern, raw_text, re.IGNORECASE)
    
    if access_match:
        # Text BEFORE the access keyword likely contains the URL
        before_access = raw_text[:access_match.start()].strip().rstrip(',.')
        print(f"  Text before 'Accessed': '{before_access}'")
        
        # Extract domain.tld or scheme://domain.tld from end of this text
        # Look for the last occurrence of a URL-like pattern
        url_pattern = r'(?:https?://)?(?:[\w-]+\.)*[\w-]+\.(?:com|org|net|de|eu|io|gov|co\.uk|ac\.uk|europa\.eu|co\.in)(?:/[\w\-./]*)?'
        urls_found = re.findall(url_pattern, before_access, re.IGNORECASE)
        
        print(f"  URLs found: {urls_found}")
        
        if urls_found:
            # Use the LAST (rightmost) URL found, as URLs typically come at the end
            entry_url = urls_found[-1].rstrip('.,;:)]}')
            print(f"  Selected (last): '{entry_url}'")
            if not entry_url.lower().startswith(('http://', 'https://')):
                entry_url = 'https://' + entry_url
    else:
        print(f"  No 'Accessed' keyword found")
    
    # Fallback: if no URL before access keyword, do a simple search
    if not entry_url:
        print(f"  Falling back to simple search...")
        domain_match = re.search(
            r'(?:https?://)?(?:www\.)?[\w-]+\.(?:com|org|net|de|eu|io|gov)\S*',
            raw_text, re.IGNORECASE)
        if domain_match:
            entry_url = domain_match.group(0).rstrip('.,;:)]}')
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
