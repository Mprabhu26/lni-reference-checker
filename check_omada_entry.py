#!/usr/bin/env python3
"""Check how the journal article entry is parsed."""

import sys
sys.path.insert(0, '.')
from app import _run_full_check

# Run a minimal check to see what references are extracted
result = _run_full_check('tests/files (3)/Kott_et_al.pdf', verify=False, filename='Kott_et_al.pdf')

# Find entries with "omada" in any field
refs = result.get('verification', [])
found_omada = False
for ref in refs:
    if 'omada' in str(ref).lower():
        found_omada = True
        print("=" * 70)
        print(f"Key: {ref.get('key')}")
        print(f"Title: {ref.get('title')}")
        print(f"Authors: {ref.get('authors')}")
        print(f"Year: {ref.get('year')}")
        print(f"Note: {ref.get('note')}")
        print("=" * 70)

if not found_omada:
    print("No entry with 'omada' found. Showing all entries with short author names:")
    for ref in refs:
        authors = ref.get('authors', '')
        if authors and len(authors) < 30:
            print(f"{ref.get('key')}: {ref.get('title')} | Authors: {authors}")
