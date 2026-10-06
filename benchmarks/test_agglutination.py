#!/usr/bin/env python3
"""
Verifica BPE di suffissi agglutinativi vs delimitati con punteggiatura.
"""
import re

def bpe_split_simulate(s):
    # Simula byte-pair encoding moderno
    # 1. Punteggiatura e spazi sono separati
    # 2. Sequenze alfabetiche brevi (1-3 char) sono solitamente singoli token nel vocabolario BPE
    tokens = re.findall(r"[a-zA-Z]+|[0-9]+|[^\s\w]|\s+", s)
    return [t for t in tokens if t.strip()]

snippets = {
    "Punctuation ':m :b'": "res = dot xm wb",
    "Punctuation ':m :b' full": "res:n = dot x:m w:b",
    "Python standard": "res = np.dot(x, w)",
    "Python compact": "res = x @ w"
}

for k, v in snippets.items():
    toks = bpe_split_simulate(v)
    print(f"{k:<25}: '{v}' -> {len(toks)} tokens: {toks}")
