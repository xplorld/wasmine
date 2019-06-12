#!/usr/bin/env python3
import sys
import unicodedata

# for whatever reason, WASM Spec is displayed via MathJax as Unicode math alphabets.
# Have to convert to ascii
for line in sys.stdin:
    line = unicodedata.normalize('NFKD', line.strip()).encode(
        'ascii', 'ignore').decode('ascii')
    words = line.split('_')
    camel = ''.join(word.capitalize() for word in words)
    print(camel)
