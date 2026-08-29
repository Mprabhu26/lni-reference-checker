#!/usr/bin/env python3
"""Test verification with improved URL extraction."""

import sys
sys.path.insert(0, '.')

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app import _run_full_check

# Run full check with verification
result = _run_full_check('tests/files (3)/Kott_et_al.pdf', verify=True, filename='Kott_et_al.pdf')

# Focus on Eu23 and La25
refs = result.get('verification', [])
for ref in refs:
    key = ref.get('key', '')
    if key in ['La25', 'Eu23']:
        print(f'\n{key}:')
        print(f'  Title: {ref.get("title")[:60] if ref.get("title") else "N/A"}')
        print(f'  Status: {ref.get("status")}')
        print(f'  AI Verdict: {ref.get("ai_verdict")}')
        print(f'  Confidence: {ref.get("confidence")}')
        print(f'  URL: {ref.get("open_access_url", "N/A")}')
        note = ref.get("note", "")
        print(f'  Note: {note[:100] if note else "N/A"}')
