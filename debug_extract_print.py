from extractor import extract_pdf
from parser import parse_bibliography, entries_to_dict
from checker import extract_citations_from_body, cross_check

p = 'C:/Users/mithi/OneDrive/Desktop/MyProject/lni-reference-checker/tests/files (3)/proceedings-example.pdf'
r = extract_pdf(p)
print('BODY_LEN', len(r.get('body', '')))
print('BIB_LEN', len(r.get('bibliography', '')))
print('BODY_HEAD')
print((r.get('body','')[:1500]))
print('\nBIB_HEAD')
print((r.get('bibliography','')[:2000]))
entries = parse_bibliography(r.get('bibliography',''))
print('\nENTRY_COUNT', len(entries))
print('KEYS', [e.key for e in entries[:25]])
cited = extract_citations_from_body(r.get('body',''))
print('CITED_COUNT', len(cited))
print('CITED', sorted(list(cited))[:50])
if entries:
    d = entries_to_dict(entries)
    print('CROSS', cross_check(d, cited))
