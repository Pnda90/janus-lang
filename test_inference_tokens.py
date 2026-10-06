#!/usr/bin/env python3
"""
Test del conteggio token con Type Inference e annotazioni complete a confronto.
"""
import re

def count_tokens(text):
    tokens = re.findall(r"[a-zA-Z_]+|[0-9]+|[:\.\,\;\(\)\[\]\{\}\=\+\-\*\/\@\>\<\_\~]|\s+", text)
    return len([t for t in tokens if t.strip() or t == '\n'])

# Confronto Attention:
janus_inferred = """fn mha(qm, kb, vb) pure {
    scale = 1.0 / (kb.dim_last sqrt)
    scores = (qm @ kb.trans) * scale smax
    ret scores @ vb
}"""

python_typed = """def mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d = k.size(-1)
    scale = 1.0 / math.sqrt(d)
    k_t = k.transpose(-2, -1)
    scores = torch.softmax((q @ k_t) * scale, dim=-1)
    return scores @ v"""

python_untyped = """def mha(q, k, v):
    d = k.size(-1)
    scale = 1.0 / math.sqrt(d)
    k_t = k.transpose(-2, -1)
    scores = torch.softmax((q @ k_t) * scale, dim=-1)
    return scores @ v"""

print(f"Attention JANUS (inferred):  {count_tokens(janus_inferred)} tokens")
print(f"Attention Python (untyped):  {count_tokens(python_untyped)} tokens")
print(f"Attention Python (typed):    {count_tokens(python_typed)} tokens")
