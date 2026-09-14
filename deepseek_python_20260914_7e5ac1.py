from extractor import extract_pdf
from parser import parse_bibliography

result = extract_pdf("Seminar Arbeit DT.pdf")
bib = result.get("bibliography", "")

entries = parse_bibliography(bib)
print(f"parsed {len(entries)} entries\n")

# The entries that were wrong in the earlier Excel report
PROBLEMATIC = {"CDE25", "HC24", "SUS25", "DS21", "SIB25", "BT23", "WW02", "Vo19", "PFN23"}

for e in entries:
    marker = "*** " if e.key in PROBLEMATIC else "    "
    print(f"{marker}[{e.key}]")
    print(f"      type:      {e.entry_type!r}")
    print(f"      authors:   {e.authors!r}")
    print(f"      title:     {e.title!r}")
    print(f"      journal:   {e.journal!r}")
    print(f"      booktitle: {e.booktitle!r}")
    print(f"      publisher: {e.publisher!r}")
    print(f"      year:      {e.year!r}")
    print(f"      pages:     {e.pages!r}")
    print(f"      doi:       {e.doi!r}")
    print(f"      issues:    {e.completeness_issues}")
    print()