#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from checker import verify_reference
from parser import BibEntry

# Create To22 entry
entry = BibEntry(
    key='To22',
    raw_text='Tomada, L.: Start-ups and the Proposed EU AI Act...'
)
entry.authors = 'Tomada, L.'
entry.title = 'Start-ups and the Proposed EU AI Act: Bridges or Barriers in the Path from Invention to Innovation'
entry.journal = 'Journal of Intellectual Property, Information Technology and Electronic Commerce Law'
entry.year = '2022'
entry.volume = '13'
entry.pages = '53'
entry.entry_type = 'article'

print(f"Verifying: {entry.title}")
print(f"Author: {entry.authors}, Year: {entry.year}")
print()

result = verify_reference(entry, allow_ai_fallback=False)  # Skip AI to see API results

if result:
    print(f"Status: {result.status}")
    print(f"Confidence: {result.confidence}")
    print(f"Note: {result.note}")
else:
    print("No result returned")
