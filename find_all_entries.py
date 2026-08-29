#!/usr/bin/env python3
"""Find all entries to locate the omada reference."""

import sys
import json
sys.path.insert(0, '.')
from app import _run_full_check

result = _run_full_check('tests/files (3)/Kott_et_al.pdf', verify=False, filename='Kott_et_al.pdf')

refs = result.get('verification', [])
print(f"Total entries: {len(refs)}")
print("\n" + "=" * 80)
print("All entries in bibliography:")
print("=" * 80)

for i, ref in enumerate(refs, 1):
    print(f"\n{i}. Key: {ref.get('key')}")
    print(f"   Title: {ref.get('title')[:100] if ref.get('title') else 'None'}")
    print(f"   Authors: {ref.get('authors')[:100] if ref.get('authors') else 'None'}")
    print(f"   Year: {ref.get('year')}")
    
    # Look for the omada entry
    title_lower = str(ref.get('title', '')).lower()
    authors_lower = str(ref.get('authors', '')).lower()
    
    if 'intellectual property' in title_lower or 'omada' in authors_lower:
        print(f"   *** MATCHED SEARCH ***")
        print(f"   Full Title: {ref.get('title')}")
        print(f"   Full Authors: {ref.get('authors')}")
