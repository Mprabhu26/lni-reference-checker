#!/usr/bin/env python3
"""Test URL verification for Eu23 and La25."""

import sys
sys.path.insert(0, '.')

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
from urllib.parse import urljoin

# Test the URLs
urls = {
    'Eu23_truncated': 'https://europarl.europa.eu/topics/en/article/20230601',
    'Eu23_full': 'https://www.europarl.europa.eu/topics/en/article/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence',
    'La25_extracted': 'https://python.langchain.com/docs/tutorials/',
    'La25_from_result': 'https://langchain.com/docs/tutorials/',
}

print("Testing URL verification:\n")

for name, url in urls.items():
    print(f"{name}:")
    print(f"  URL: {url}")
    
    # Try HEAD
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True)
        print(f"  HEAD: {resp.status_code}")
    except Exception as e:
        print(f"  HEAD: Error - {type(e).__name__}: {str(e)[:100]}")
    
    # Try GET
    try:
        resp = requests.get(url, timeout=5, allow_redirects=True)
        print(f"  GET: {resp.status_code}")
        if resp.status_code == 200:
            # Check if we can extract title
            if 'langchain' in url.lower():
                print(f"  Content length: {len(resp.text)}")
            else:
                # For europarl
                print(f"  Content length: {len(resp.text)}")
    except Exception as e:
        print(f"  GET: Error - {type(e).__name__}: {str(e)[:100]}")
    
    print()
