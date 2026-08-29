#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from extractor import extract_pdf
from parser import parse_bibliography

# Extract text from PDF
result = extract_pdf('tests/files (3)/Kott_et_al.pdf')
bib_text = result.get('bibliography', '')

# Parse bibliography
entries = parse_bibliography(bib_text)

# Find La25
la25 = [ref for ref in entries if ref.key == 'La25']
if la25:
    r = la25[0]
    print(f"La25 entry: {r}")
    if r.url:
        print(f"\nExtracted URL: {r.url}")
        print(f"Contains /docs/: {'/docs/' in r.url.lower()}")
        print(f"Contains /tutorials/: {'/tutorials/' in r.url.lower()}")
        print(f"Contains /guide/: {'/guide/' in r.url.lower()}")
