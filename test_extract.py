#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import pdfplumber
from parser import parse_bibliography

# Just extract bibliography
with pdfplumber.open('tests/files (3)/Kott_et_al.pdf') as pdf:
    text = ''.join([p.extract_text() for p in pdf.pages])

refs = parse_bibliography(text)
la25 = [r for r in refs if r.key == 'La25']
eu23 = [r for r in refs if r.key == 'Eu23']

if la25:
    r = la25[0]
    print(f"La25:")
    print(f"  Title: {r.title}")
    print(f"  URL: {r.url}")
    print(f"  Type: {r.entry_type}")

if eu23:
    r = eu23[0]
    print(f"\nEu23:")
    print(f"  Title: {r.title}")
    print(f"  URL: {r.url}")
    print(f"  Type: {r.entry_type}")
