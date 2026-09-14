from extractor import extract_pdf, extract_references_from_bibliography
from parser import _extract_venue_structural, _classify_and_parse, BibEntry

# Get the bibliography
result = extract_pdf("Seminar Arbeit DT.pdf")
bib = result.get("bibliography", "")
refs = extract_references_from_bibliography(bib)

# Find Te24
for r in refs:
    if r['key'] == 'Te24':
        raw = r['raw_text']
        print("1. raw text:")
        print("   " + repr(raw))
        print()

        print("2. What _extract_venue_structural returns on this raw text:")
        print("   " + repr(_extract_venue_structural(raw)))
        print()

        # Run the full pipeline
        e = BibEntry(key='Te24', raw_text=raw)
        _classify_and_parse(e, raw)

        print("3. Final journal after full pipeline:")
        print("   " + repr(e.journal))
        print()
        print("4. Final title after full pipeline:")
        print("   " + repr(e.title))
        break