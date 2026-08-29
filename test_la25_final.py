#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from app import _run_full_check

# Run full check
result = _run_full_check('tests/files (3)/Kott_et_al.pdf', verify=True, filename='Kott_et_al.pdf')

# Extract La25 and Eu23
refs = result.get('verification', [])
for ref in refs:
    if ref.get('key') in ['La25', 'Eu23']:
        print(f"{ref.get('key')}: {ref.get('ai_verdict')} (status: {ref.get('status')})")
