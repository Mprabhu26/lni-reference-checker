#!/usr/bin/env python3
"""Test improved URL extraction with space handling."""

import re

url_pattern = r'(?:https?://)?(?:[\w-]+\.)*[\w-]+\.(?:com|org|net|de|eu|io|gov|co\.uk|ac\.uk|europa\.eu|co\.in)(?:(?:/[\w\-.]+ )*(?:/[\w\-.]+)?)*'

test_cases = [
    ('European Parliament: EU AI Act: first regulation on artificial intelligence. europarl.europa.eu/topics/en/article/20230601 STO93804/eu-ai-act-first-regulation-on-artificial-intelligence. –', 'EU Parliament'),
    ('Lang Chain: Introduction. python.langchain.com/docs/tutorials/. Accessed: 2025-04-14', 'LangChain'),
]

for test_text, desc in test_cases:
    print(f"\n{desc}:")
    print(f"  Text: {test_text}")
    urls = re.findall(url_pattern, test_text, re.IGNORECASE)
    print(f"  URLs found: {urls}")
    print(f"  Last URL: {urls[-1] if urls else 'None'}")
