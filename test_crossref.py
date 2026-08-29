#!/usr/bin/env python3
"""Test CrossRef API for the journal article with abbreviated author."""

import requests

# Journal article: omada, L. in 'Journal of Intellectual Property, Information Technology and Electronic Commerce Law' vol. 13, p. 53, 2022
title = 'Journal of Intellectual Property, Information Technology and Electronic Commerce Law'
authors = 'omada, L.'

print(f"Testing CrossRef API:")
print(f"  Title: {title}")
print(f"  Authors: {authors}")
print()

# Test 1: Query by title only
params1 = {"query.title": title, "rows": 5}
first_author = authors.split(';')[0].split(',')[0].strip()
print(f"First author extracted: '{first_author}'")

if first_author and len(first_author) > 2:
    params1["query.author"] = first_author
    print(f"Adding author filter: '{first_author}'")

ua = "LNI-Checker/8.1"
print("\nQuery 1: By title and author")
print(f"Params: {params1}")

resp = requests.get("https://api.crossref.org/works", params=params1, timeout=12, headers={"User-Agent": ua})
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    items = resp.json().get("message", {}).get("items", [])
    print(f"Found {len(items)} results")
    for i, item in enumerate(items[:3], 1):
        print(f"  {i}. {item.get('title', ['N/A'])[0]}")
        authors_list = item.get('author', [])
        if authors_list:
            print(f"     Authors: {', '.join(a.get('family', '') for a in authors_list[:2])}")
else:
    print(f"Error: {resp.text[:200]}")

# Test 2: Search for articles by "Omada" author
print("\n" + "="*60)
print("Query 2: Search for 'Omada' as author")
params2 = {"query.author": "Omada", "rows": 5}
print(f"Params: {params2}")

resp2 = requests.get("https://api.crossref.org/works", params=params2, timeout=12, headers={"User-Agent": ua})
print(f"Status: {resp2.status_code}")

if resp2.status_code == 200:
    items2 = resp2.json().get("message", {}).get("items", [])
    print(f"Found {len(items2)} results")
    for i, item in enumerate(items2[:5], 1):
        print(f"  {i}. {item.get('title', ['N/A'])[0]}")
else:
    print(f"Error: {resp2.text[:200]}")

# Test 3: Search for article title (not journal)
print("\n" + "="*60)
print("Query 3: Search for article title keyword")
params3 = {"query": "Intellectual Property Information Technology", "rows": 5}
print(f"Params: {params3}")

resp3 = requests.get("https://api.crossref.org/works", params=params3, timeout=12, headers={"User-Agent": ua})
print(f"Status: {resp3.status_code}")

if resp3.status_code == 200:
    items3 = resp3.json().get("message", {}).get("items", [])
    print(f"Found {len(items3)} results")
    for i, item in enumerate(items3[:5], 1):
        print(f"  {i}. {item.get('title', ['N/A'])[0]}")
else:
    print(f"Error: {resp3.text[:200]}")
