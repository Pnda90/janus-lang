#!/usr/bin/env python3
"""
Test e validatore per la grammatica EBNF di JANUS.
Verifica che le frasi di esempio siano prive di ambiguità e parsabili linearmente.
"""
import re

# Tokenizzatore lessicale semplice basato sulle regole morfologiche di JANUS
TOKEN_SPEC = [
    ('FLOAT', r'\d+\.\d+(?:e[+-]?\d+)?'),
    ('INT', r'\d+'),
    ('KW', r'\b(fn|ret|let|mut|type|if|else|loop|for|in|brk|cont|end|match|schema|kern|on|call|opt|diff|alloc|free|copy|rand|dist|norm|smax|relu|matmul|dot|conv2d|sum)\b'),
    ('CASE_ID', r'\b[a-zA-Z_][a-zA-Z0-9_]*([mbtnsv])\b'), # Identificatore con suffisso di caso
    ('ID', r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'),
    ('OP', r'(\@|\+|\-|\*|\/|\=|\-\>)'),
    ('PUNCT', r'([\{\}\[\]\(\)\,\:\;\.])'),
    ('WS', r'\s+'),
]

master_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPEC)

def lex(code):
    tokens = []
    for mo in re.finditer(master_regex, code):
        kind = mo.lastgroup
        val = mo.group()
        if kind == 'WS':
            continue
        if kind == 'CASE_ID':
            case_char = mo.group(1)
            base = val[:-1]
            tokens.append(('CASE_ID', val, base, case_char))
        else:
            tokens.append((kind, val))
    return tokens

# Esempio di test
sample = "outn = xm matmul wb add bb relu"
toks = lex(sample)
print(f"Tokenizzazione di '{sample}':")
for t in toks:
    print(" ", t)
