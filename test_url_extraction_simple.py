#!/usr/bin/env python3
"""Test better URL extraction - match until clear delimiters."""

import re

test_cases = [
    ('European Parliament: EU AI Act: first regulation on artificial intelligence. europarl.europa.eu/topics/en/article/20230601 STO93804/eu-ai-act-first-regulation-on-artificial-intelligence. – Accessed: 2025-04-10', 'EU Parliament'),
    ('Lang Chain: Introduction. python.langchain.com/docs/tutorials/. Accessed: 2025-04-14', 'LangChain'),
]

for test_text, desc in test_cases:
    print(f"\n{desc}:")
    print(f"  Text: {test_text}")
    
    # Strategy: Find domain, then capture everything after it until we hit delimiters
    # Delimiters: ". " (period-space), " – " (space-dash), "Accessed", or end of text
    
    # First find any domain-like pattern
    domain_match = re.search(
        r'(?:https?://)?(?:[\w-]+\.)+(?:com|org|net|de|eu|io|gov|co\.uk|ac\.uk|europa\.eu)',
        test_text, re.IGNORECASE)
    
    if domain_match:
        start = domain_match.start()
        # Find clear delimiters after the domain
        delimiter_match = re.search(
            r'\.\s*$|(?:\s*–\s)|(?:\s*Accessed)',
            test_text[domain_match.end():], re.IGNORECASE)
        
        if delimiter_match:
            end = domain_match.end() + delimiter_match.start()
        else:
            end = len(test_text)
        
        url_text = test_text[start:end].strip()
        print(f"  Extracted: '{url_text}'")
    else:
        print(f"  No domain found")
