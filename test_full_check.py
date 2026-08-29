#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import app

print("Testing full verification pipeline...")
result = app._run_full_check('tests/files (3)/Kott_et_al.pdf', verify=True, filename='Kott_et_al.pdf')
print(f"✓ Completed successfully!")
print(f"Score: {result.get('score')}")
print(f"Verdict: {result.get('overall', {}).get('verdict')}")
print(f"Summary: {result.get('summary')}")
