#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_matching.py
=================
Directly exercises the checker's matching and database lookup on KNOWN-GOOD
references, printing exactly which field passes or fails. Run this from your
project folder:

    python debug_matching.py

Databases are reachable, so if genuine references still land in MANUAL_REVIEW,
this tells us precisely which line of matching rejects them.
"""

import sys
import traceback

# Ensure we import THIS project's checker
sys.path.insert(0, ".")

from parser import BibEntry
import checker


def make_entry(authors, title, journal, year, doi=None, key="K1", entry_type="article"):
    return BibEntry(
        key=key, raw_text="", entry_type=entry_type,
        authors=authors, title=title, journal=journal,
        year=year, pages="", volume="", number="", doi=doi,
    )


KNOWN_GOOD = [
    # Vaswani et al. — "Attention is all you need", NeurIPS 2017
    make_entry(
        "Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L.; Polosukhin, I.",
        "Attention is all you need",
        "Advances in neural information processing systems",
        "2017",
    ),
    # A real DOI-based paper
    make_entry(
        "LeCun, Y.; Bengio, Y.; Hinton, G.",
        "Deep learning",
        "Nature",
        "2015",
        doi="10.1038/nature14539",
    ),
]


def show_match(entry, rec, label):
    print(f"\n--- {label} ---")
    print(f"  entry.title : {entry.title!r}")
    print(f"  cited authors: {entry.authors!r}")
    print(f"  entry.year   : {entry.year!r}")
    print(f"  source.title : {rec.get('title')!r}")
    print(f"  source.auth  : {rec.get('authors')!r}")
    print(f"  source.year  : {rec.get('year')!r}")
    try:
        ok, checks = checker._full_combination_match(entry, rec)
        print(f"  MATCHED: {ok}")
        for k, v in checks.items():
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"  EXCEPTION in _full_combination_match: {e}")
        traceback.print_exc()


def main():
    print("=" * 66)
    print("  Checker matching + database lookup — live diagnostic")
    print("=" * 66)

    # ---- 1. Pure matching test against an EXACT copy (should be True) ----
    print("\n[1] SYNTHETIC exact-match test (a perfect DB record vs the citation)")
    e1 = KNOWN_GOOD[0]
    exact_rec = {
        "title": "Attention Is All You Need",
        "authors": "Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; Uszkoreit, Jakob; Jones, Llion; Gomez, Aidan N; Kaiser, Lukasz; Polosukhin, Illia",
        "year": "2017",
        "venue": "Advances in neural information processing systems",
        "doi": None,
    }
    ok, checks = checker._full_combination_match(e1, exact_rec)
    print(f"  exact match -> {ok}   (expect True)")
    for k, v in checks.items():
        print(f"    {k}: {v}")

    # Test title similarity values directly
    print("\n[2] title-similarity sanity check:")
    sim = checker._title_similarity(
        "Attention is all you need",
        "Attention Is All You Need")
    print(f"  'Attention is all you need' vs 'Attention Is All You Need' = {sim}")
    sim2 = checker._title_similarity(
        "Deep learning", "Deep Learning")
    print(f"  'Deep learning' vs 'Deep Learning' = {sim2}")

    # ---- 2. Live CrossRef + OpenAlex lookup on a known-good reference ----
    print("\n[3] LIVE database lookup:")
    for i, e in enumerate(KNOWN_GOOD):
        print(f"\n--- Known-good ref #{i+1}: {e.title} ({e.year}) ---")
        try:
            r = checker._search_crossref(e)
            if r:
                print(f"  CrossRef -> status={r.status} conf={r.confidence}")
                print(f"    sources_checked={r.sources_checked}")
                print(f"    note={r.note}")
                print(f"    field_checks={r.field_checks}")
            else:
                print("  CrossRef -> None (no verified match)")
        except Exception as ex:
            print(f"  CrossRef EXCEPTION: {ex}")
        try:
            r2 = checker._search_openalex(e)
            if r2:
                print(f"  OpenAlex -> status={r2.status} conf={r2.confidence}")
                print(f"    sources_checked={r2.sources_checked}")
                print(f"    note={r2.note}")
                print(f"    field_checks={r2.field_checks}")
            else:
                print("  OpenAlex -> None (no verified match)")
        except Exception as ex:
            print(f"  OpenAlex EXCEPTION: {ex}")

    # ---- 3. Full verify_reference pipeline on the DOI-based one ----
    print("\n[4] Full verify_reference pipeline (Deep learning / Nature / LeCun):")
    e2 = KNOWN_GOOD[1]
    try:
        vr = checker.verify_reference(e2, dup_map=None)
        print(f"  verify_reference -> status={vr.status}")
        print(f"    sources_checked={vr.sources_checked}")
        print(f"    confidence={vr.confidence}")
        print(f"    note={vr.note}")
    except Exception as ex:
        print(f"  verify_reference EXCEPTION: {ex}")
        traceback.print_exc()

    print("\n" + "=" * 66)
    print("Done. Paste this output to Claude.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())