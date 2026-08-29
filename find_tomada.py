#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import pdfplumber
from parser import parse_bibliography

# Extract bibliography
with pdfplumber.open('tests/files (3)/Kott_et_al.pdf') as pdf:
    text = ''.join([p.extract_text() for p in pdf.pages])

refs = parse_bibliography(text)

# Search for Tomada (could be key like To22, To23, etc.)
print("Searching for Tomada reference...")
for r in refs:
    if r.authors and 'tomada' in r.authors.lower():
        print(f"\nKEY: {r.key}")
        print(f"Authors: {r.authors}")
        print(f"Title: {r.title}")
        print(f"Journal: {r.journal}")
        print(f"Year: {r.year}")
        print(f"Volume: {r.volume}")
        print(f"Pages: {r.pages}")
        print(f"Entry type: {r.entry_type}")
        break
else:
    # If not found, show all references with 2022 year
    print("\nNot found by author. Checking 2022 references...")
    for r in refs:
        if r.year == '2022':
            print(f"{r.key}: {r.authors[:30] if r.authors else 'N/A'} - {r.title[:50] if r.title else 'N/A'}")
