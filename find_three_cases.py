#!/usr/bin/env python3
"""Find entries matching the three cases mentioned by user."""

import sys
sys.path.insert(0, '.')

# Fix encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app import _run_full_check

result = _run_full_check('tests/files (3)/Kott_et_al.pdf', verify=False, filename='Kott_et_al.pdf')

refs = result.get('verification', [])
print(f"\nTotal entries: {len(refs)}\n")

# Search patterns
search_patterns = [
    ('omada', "Journal article with abbreviated author"),
    ('langchain', "LangChain documentation"),
    ('europarl', "EU Parliament AI Act"),
    ('python.langchain', "LangChain Python docs"),
    ('accessed', "Any entry with access date"),
]

found_count = 0
for pattern, desc in search_patterns:
    print(f"\n{'='*70}")
    print(f"Searching for: {desc} (pattern: '{pattern}')")
    print(f"{'='*70}")
    
    for ref in refs:
        title = str(ref.get('title', '')).lower()
        authors = str(ref.get('authors', '')).lower()
        url = str(ref.get('url', '')).lower() if 'url' in ref else ''
        note = str(ref.get('note', '')).lower()
        
        if pattern.lower() in title or pattern.lower() in authors or pattern.lower() in url or pattern.lower() in note:
            found_count += 1
            print(f"\nKey: {ref.get('key')}")
            print(f"Title: {ref.get('title')}")
            print(f"Authors: {ref.get('authors')}")
            print(f"Year: {ref.get('year')}")
            if 'url' in ref:
                print(f"URL: {ref.get('url')}")
            print(f"AI Verdict: {ref.get('ai_verdict')}")
            print(f"Status: {ref.get('status')}")

if found_count == 0:
    print("\nNo matches found for any pattern. Showing all entries:")
    for i, ref in enumerate(refs, 1):
        print(f"{i}. {ref.get('key')}: {ref.get('title', 'N/A')[:80]}")
