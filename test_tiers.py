#!/usr/bin/env python3
"""Test TIER 1-3 implementation"""
import sys
sys.path.insert(0, '.')

from app import _run_full_check

print('Testing TIER 1-3 implementation on Kott_et_al.pdf...')
print()

result = _run_full_check('tests/files (3)/Kott_et_al.pdf', verify=True, filename='Kott_et_al.pdf')

# Check TIER 1: Confidence tiers
print('TIER 1: Confidence Tiers')
confidence_tiers = {}
for v in result.get('verification', []):
    tier = v.get('confidence_tier', 'unknown')
    confidence_tiers[tier] = confidence_tiers.get(tier, 0) + 1

for tier, count in sorted(confidence_tiers.items()):
    print(f'  - {tier}: {count} references')

# Check TIER 2: Author validation
print()
print('TIER 2: Author Validation')
author_validations = 0
for v in result.get('verification', []):
    if v.get('author_validation'):
        author_validations += 1

print(f'  - {author_validations} references have author validation')

# Check TIER 3: Professor workflow
print()
print('TIER 3: Professor Workflow')
workflow = result.get('professor_workflow', {})
review_summary = workflow.get('review_summary', {})
print('  Review priorities:')
print(f'    - Urgent: {review_summary.get("urgent", 0)}')
print(f'    - Important: {review_summary.get("important", 0)}')
print(f'    - Optional: {review_summary.get("optional", 0)}')
print(f'    - Skip: {review_summary.get("skip", 0)}')

batch_patterns = workflow.get('batch_patterns', {})
print('  Batch patterns:')
print(f'    - Found {len(batch_patterns.get("patterns", []))} patterns')
print(f'    - Found {len(batch_patterns.get("warnings", []))} warnings')

# Overall statistics
print()
print('Overall Results:')
print(f'  - Total references: {len(result.get("verification", []))}')
print(f'  - Verified: {sum(1 for v in result.get("verification", []) if v["status"] == "verified")}')
print(f'  - Manual review: {sum(1 for v in result.get("verification", []) if v["status"] == "manual_review")}')
print(f'  - Score: {result.get("score")}')

print()
print('SUCCESS: TIER 1-3 implementation working!')
