#!/usr/bin/env python3
import sys

template = [('I32', 'u32'), ('I64', 'u64'), ('F32', 'f32'), ('F64', 'f64')]


for line in sys.stdin:
    line = line.strip()
    for b, s in template:
        print(line.replace("{big}", b).replace("{small}", s))
