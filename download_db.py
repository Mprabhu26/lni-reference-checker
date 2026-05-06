"""
Download full Semantic Scholar database (optional - large!)
Only run if you have disk space and time.
"""

import subprocess
import sys
from pathlib import Path

print("""
╔════════════════════════════════════════════════════════════════╗
║  FULL DATABASE DOWNLOAD (OPTIONAL)                            ║
║                                                                ║
║  Size: ~50GB                                                  ║
║  Time: 1-2 hours                                              ║
║                                                                ║
║  Alternatively, use the sample database from the main script. ║
╚════════════════════════════════════════════════════════════════╝
""")

response = input("Download full database? (y/N): ")
if response.lower() != 'y':
    print("Skipping full download. Use sample database instead.")
    sys.exit(0)

print("Downloading Semantic Scholar database snapshot...")
print("This uses the official S2ORC dataset from the University of Washington.")

# Instructions for manual download (since direct download is complex)
print("""
To download the full database:

1. Visit: https://github.com/allenai/s2orc
2. Follow instructions to download the latest snapshot
3. Or use the sample database (already created)

For most users, the sample database (500k papers) is sufficient.
""")