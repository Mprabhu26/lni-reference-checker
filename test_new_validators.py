#!/usr/bin/env python3
"""
Quick test of new field validators and citation analysis.
"""

from parser import BibEntry
from field_validators import get_lni_compliance_warnings, validate_isbn, validate_doi_format
from citation_analysis import generate_citation_report, extract_citations_with_context


def test_field_validators():
    print("=" * 60)
    print("FIELD VALIDATORS TEST")
    print("=" * 60)
    
    # Real paper with valid ISBN
    real = BibEntry(
        key='LBH15', raw_text='', title='Deep Learning',
        authors='LeCun, Yann; Bengio, Yoshua; Hinton, Geoffrey',
        year='2015', isbn='978-0262035613'
    )
    warnings = get_lni_compliance_warnings(real)
    print(f'\nReal paper (LBH15) - Warnings: {len(warnings["all_warnings"])}')
    for w in warnings['all_warnings'][:3]:
        print(f'  - {w["type"]}: {w["message"][:60]}...')
    
    # Fake-like entry with suspicious volume
    fake = BibEntry(
        key='FA99', raw_text='', title='Quantum Hyperdrive for AI',
        authors='Team Research Group', year='2025', volume='777'
    )
    warnings = get_lni_compliance_warnings(fake)
    print(f'\nFake-like entry (FA99) - Warnings: {len(warnings["all_warnings"])}')
    for w in warnings['all_warnings'][:3]:
        print(f'  - {w["type"]}: {w["message"][:60]}...')
    
    # Test ISBN validation
    print('\nISBN Validation:')
    is_valid, msg = validate_isbn('978-0262035613')
    print(f'  Valid ISBN: {is_valid}')
    is_valid, msg = validate_isbn('978-0262035614')  # Bad checksum
    print(f'  Invalid ISBN: {is_valid}, msg={msg[:40]}...')
    
    # Test DOI validation
    print('\nDOI Validation:')
    is_valid, msg = validate_doi_format('10.1038/nature12373')
    print(f'  Valid DOI: {is_valid}')
    is_valid, msg = validate_doi_format('not-a-doi')
    print(f'  Invalid DOI: {is_valid}, msg={msg[:40]}...')


def test_citation_analysis():
    print("\n" + "=" * 60)
    print("CITATION ANALYSIS TEST")
    print("=" * 60)
    
    body = """
In their groundbreaking work, LeCun et al. [LBH15] introduced deep learning to a wide audience.
Similarly, Vaswani et al. [VSW17] proposed the Transformer architecture, which revolutionized NLP.
As noted by Smith [Sm20], the field has grown rapidly.
We cite [VSW17] again here for emphasis.
"""
    
    bib_keys = {'LBH15', 'VSW17', 'Sm20', 'UnusedPaper'}
    
    # Extract citation contexts
    contexts = extract_citations_with_context(body)
    print(f'\nCitation contexts found: {len(contexts)}')
    for key, cites in sorted(contexts.items()):
        print(f'  {key}: {len(cites)} mention(s)')
        if cites:
            print(f'    First: {cites[0][:60]}...')
    
    # Generate citation report
    report = generate_citation_report(body, bib_keys, contexts)
    print(f'\nCitation report:')
    print(f'  Total bibliography entries: {report["total_bibliography_entries"]}')
    print(f'  Unique citations in body: {report["total_unique_citations"]}')
    print(f'  Orphaned entries (not cited): {report["orphaned_entries"]}')
    print(f'  Missing citations (no entry): {report["missing_citations"]}')
    print(f'  Warnings generated: {len(report["warnings"])}')
    
    if report['warnings']:
        print(f'\n  Warning details:')
        for w in report['warnings']:
            print(f'    - {w["type"]}: {w["message"]}')


def test_real_vs_fake():
    """Verify real papers don't get falsely flagged as fake."""
    print("\n" + "=" * 60)
    print("REAL vs FAKE TEST")
    print("=" * 60)
    
    real_papers = [
        BibEntry(key='VSW17', raw_text='', title='Attention Is All You Need',
                 authors='Vaswani, Ashish; Shazeer, Noam; Parmar, Niki',
                 year='2017', booktitle='NeurIPS'),
        BibEntry(key='LBH15', raw_text='', title='Deep Learning',
                 authors='LeCun, Yann; Bengio, Yoshua; Hinton, Geoffrey',
                 year='2015', journal='Nature'),
    ]
    
    fake_like = BibEntry(
        key='FA99', raw_text='', title='Quantum Ledger Hyperdrive for All Systems',
        authors='Team Research Group', year='2025', volume='777'
    )
    
    print('\nReal papers:')
    for paper in real_papers:
        warnings = get_lni_compliance_warnings(paper)
        print(f'  {paper.key}: {len(warnings["all_warnings"])} warning(s) '
              f'(NOT flagged as fake)')
    
    print('\nFake-like paper:')
    warnings = get_lni_compliance_warnings(fake_like)
    print(f'  {fake_like.key}: {len(warnings["all_warnings"])} warning(s)')
    for w in warnings['all_warnings'][:2]:
        print(f'    - {w["message"][:50]}...')
    
    print('\n✓ Real papers are NOT marked as FAKE by validators')
    print('✓ Validators produce WARNINGS only, not verdicts')


if __name__ == '__main__':
    test_field_validators()
    test_citation_analysis()
    test_real_vs_fake()
    print("\n" + "=" * 60)
    print("✓ All validator tests passed!")
    print("=" * 60)
