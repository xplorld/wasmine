#!/usr/bin/env python3
import sys

for line in sys.stdin:
    op, *words = line.strip().split(' ')
    variant = ''.join(w.capitalize() for w in words)
    print(f'| {op} => value!(Instr::{variant})')
