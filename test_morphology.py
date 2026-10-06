#!/usr/bin/env python3
"""
Test BPE splitting for suffixes in Python.
"""
import re

# Test strings
test_cases = [
    ("colon suffix", "x:m w:b"),
    ("dot suffix", "x.m w.b"),
    ("bare suffix", "xm wb"),
    ("dollar prefix", "$m x $b w"),
    ("sigil prefix", "%m x %b w"),
    ("direct operator", "x @ w"),
    ("pure positional", "dot(x, w)"),
    ("latin root fused", "tensm tensb"),
]

print("Analisi strutturale della tokenizzazione morfologica:")
for label, s in test_cases:
    tokens = re.findall(r"\w+|[^\w\s]|\s+", s)
    toks = [t for t in tokens if t.strip()]
    print(f"{label:<20}: '{s}' -> {len(toks)} tokens: {toks}")
