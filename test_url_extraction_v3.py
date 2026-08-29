#!/usr/bin/env python3
"""Test improved URL extraction v3 - with better TLD handling."""

import re

def test_url_extraction(raw_text, description):
    """Test the new URL extraction logic."""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"Raw text: {raw_text}")
    print(f"{'='*70}")
    
    # More comprehensive pattern to handle:
    # - Multiple subdomains (www.python.langchain.com)
    # - Multi-level TLDs (example.europa.eu)
    # Matches: [scheme://] [www.] [subdomain.] ... domain.tld
    domain_pattern = r'(?:https?://)?(?:[\w-]+\.)+(?:com|org|net|de|eu|io|gov|co\.uk|ac\.uk|europa\.eu|co\.in|etc\.)?(?:\.\w{2})?'
    domain_match = re.search(domain_pattern, raw_text, re.IGNORECASE)
    
    entry_url = None
    if domain_match:
        # Start with the matched domain
        domain_with_prefix = domain_match.group(0).rstrip('.')
        domain_end = domain_match.end()
        
        print(f"  Domain found: '{domain_with_prefix}'")
        
        # Now look for path segments after the domain
        after_domain = raw_text[domain_end:]
        print(f"  After domain: '{after_domain[:60]}'")
        
        # Capture path: /segment/segment/... until we hit a keyword or punctuation
        # Allow path segments with hyphens, underscores, alphanumerics
        path_pattern = r'(/[\w\-./]*)'
        path_match = re.match(path_pattern, after_domain)
        
        if path_match:
            path = path_match.group(1)
            print(f"  Raw path: '{path}'")
            # Remove trailing periods or slashes followed by keywords
            path = re.sub(r'/?\s*(?:–|Accessed|Abruf|Stand|accessed|Accessed:)\b.*$', '', path, flags=re.IGNORECASE)
            path = path.rstrip('.,;:)]}')
            print(f"  Clean path: '{path}'")
            entry_url = domain_with_prefix + path
        else:
            print(f"  No path found")
            entry_url = domain_with_prefix
        
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
