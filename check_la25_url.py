#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from extractor import extract_pdf

bibdata = extract_pdf('tests/files (3)/Kott_et_al.pdf')
la25 = [ref for ref in bibdata if ref.get('key') == 'La25']
if la25:
    r = la25[0]
    print(f"La25 entry: {r}")
    if 'url' in r:
        print(f"\nExtracted URL: {r['url']}")
        print(f"Contains /docs/: {'/docs/' in r['url'].lower()}")
        print(f"Contains /tutorials/: {'/tutorials/' in r['url'].lower()}")
        print(f"Contains /guide/: {'/guide/' in r['url'].lower()}")
