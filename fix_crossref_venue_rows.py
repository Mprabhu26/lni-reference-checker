#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_indentation.py
==================
Fix indentation error in checker.py line 735
"""
import pathlib, sys

TARGET = pathlib.Path("checker.py")
if not TARGET.exists():
    print("ERROR: checker.py not found"); sys.exit(1)

lines = TARGET.read_text(encoding="utf-8").splitlines()

# Find the problematic line
for i, line in enumerate(lines):
    if 'if not title:' in line and line.startswith('        '):  # 8 spaces
        # Should be 8 spaces, but check if it's actually 12
        if line.startswith('            '):  # 12 spaces
            print(f"Fixing line {i+1}: removing 4 spaces")
            lines[i] = line[4:]  # Remove 4 spaces
            break

TARGET.write_text('\n'.join(lines), encoding="utf-8")
print("Fixed indentation error in checker.py")