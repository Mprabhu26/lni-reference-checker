#!/usr/bin/env python3
"""Show detailed info about La25 and Eu23 entries."""

import sys
import json
sys.path.insert(0, '.')

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app import _run_full_check

result = _run_full_check('tests/files (3)/Kott_et_al.pdf', verify=False, filename='Kott_et_al.pdf')

refs = result.get('verification', [])

# Find specific entries
for ref in refs:
    key = ref.get('key', '')
    if key in ['La25', 'Eu23']:
        print(f"\n{'='*80}")
        print(f"Entry Key: {key}")
        print(f"{'='*80}")
        
        # Print all fields
        for field, value in ref.items():
            if value:  # Only print non-empty fields
                value_str = str(value)
                if len(value_str) > 200:
                    print(f"{field}: {value_str[:200]}...")
                else:
                    print(f"{field}: {value_str}")

# Also show raw extracted text for these keys
print(f"\n\n{'='*80}")
print("Looking for raw BibEntry objects...")
print(f"{'='*80}")

# Try to access the parser directly
from parser import parse_bibliography

# Extract text from PDF
from extractor import extract

text = extract('tests/files (3)/Kott_et_al.pdf')
bib_entries = parse_bibliography(text, use_ai=False)

print(f"Total parsed entries: {len(bib_entries)}")

for entry in bib_entries:
    if entry.key in ['La25', 'Eu23']:
        print(f"\n{'='*60}")
        print(f"Raw BibEntry: {entry.key}")
        print(f"{'='*60}")
        print(f"Title: {entry.title}")
        print(f"Authors: {entry.authors}")
        print(f"Year: {entry.year}")
        print(f"URL: {entry.url if hasattr(entry, 'url') else 'N/A'}")
        print(f"Publisher: {entry.publisher if hasattr(entry, 'publisher') else 'N/A'}")
        print(f"Journal: {entry.journal if hasattr(entry, 'journal') else 'N/A'}")
        print(f"Raw text: {entry.raw_text[:200] if hasattr(entry, 'raw_text') and entry.raw_text else 'N/A'}")
