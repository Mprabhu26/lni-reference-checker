from local_db import search_cache, get_cache_stats, save_to_cache
from checker import verify_reference
from parser import BibEntry

print("=" * 50)
print("TEST 1: Check if cache reading works")
print("=" * 50)
result = search_cache('Test Paper')
print(f'Found: {result.title if result else "Not found"}')
stats = get_cache_stats()
print(f'Stats: {stats}')

print("\n" + "=" * 50)
print("TEST 2: Manually save Attention paper")
print("=" * 50)
save_to_cache(
    title='Attention Is All You Need',
    authors='Vaswani',
    year='2017',
    doi='10.48550/arXiv.1706.03762',
    url='https://arxiv.org/abs/1706.03762',
    source='manual_test',
    confidence=0.95
)
result = search_cache('Attention Is All You Need')
print(f'Found: {result.title if result else "Not found"}')
stats = get_cache_stats()
print(f'Cache stats: {stats}')

print("\n" + "=" * 50)
print("TEST 3: Check verify_reference status")
print("=" * 50)
test_entry = BibEntry(key='Tes17', title='Attention Is All You Need', authors='Vaswani', raw_text='')
result = verify_reference(test_entry)
print(f'Status: {result.status}')
print(f'Status repr: {repr(result.status)}')
print(f'Is verified: {result.status == "verified"}')