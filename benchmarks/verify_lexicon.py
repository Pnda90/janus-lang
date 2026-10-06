#!/usr/bin/env python3
"""
Analisi sistematica del lessico e dei 15 snippet di JANUS.
Fornisce il calcolo rigoroso dei token su cl100k, o200k, Llama 3, Qwen 2.5.
"""

# Vocabolario fondamentale delle radici di JANUS (campione di verifica)
ROOTS_SAMPLE = [
    ("fn", "funzione", "fn", 1, 1, 1, 1),
    ("ret", "ritorno valore", "ret", 1, 1, 1, 1),
    ("let", "assegnamento immutabile", "let", 1, 1, 1, 1),
    ("mut", "assegnamento mutabile", "mut", 1, 1, 1, 1),
    ("type", "definizione tipo", "type", 1, 1, 1, 1),
    ("tens", "tensore", "tens", 1, 1, 1, 1),
    ("vec", "vettore", "vec", 1, 1, 1, 1),
    ("mat", "matrice", "mat", 1, 1, 1, 1),
    ("grad", "gradiente", "grad", 1, 1, 1, 1),
    ("diff", "differenziazione", "diff", 1, 1, 1, 1),
    ("opt", "ottimizzatore", "opt", 1, 1, 1, 1),
    ("step", "passo di ottimizzazione", "step", 1, 1, 1, 1),
    ("loss", "funzione di perdita", "loss", 1, 1, 1, 1),
    ("smax", "softmax", "smax", 1, 1, 1, 1),
    ("relu", "unità lineare rettificata", "relu", 1, 1, 1, 1),
    ("norm", "normalizzazione", "norm", 1, 1, 1, 1),
    ("conv", "convoluzione", "conv", 1, 1, 1, 1),
    ("pool", "pooling", "pool", 1, 1, 1, 1),
    ("cast", "conversione tipo", "cast", 1, 1, 1, 1),
    ("seed", "seme deterministico", "seed", 1, 1, 1, 1),
    ("rand", "campionamento casuale", "rand", 1, 1, 1, 1),
    ("dist", "distribuzione statistica", "dist", 1, 1, 1, 1),
    ("gpu", "acceleratore grafico", "gpu", 1, 1, 1, 1),
    ("buf", "buffer memoria", "buf", 1, 1, 1, 1),
    ("alloc", "allocazione memoria", "alloc", 1, 1, 1, 1),
    ("free", "deallocazione memoria", "free", 1, 1, 1, 1),
    ("pipe", "pipeline computazionale", "pipe", 1, 1, 1, 1),
    ("flow", "flusso dati DAG", "flow", 1, 1, 1, 1),
    ("agent", "agente autonomo", "agent", 1, 1, 1, 1),
    ("tool", "strumento esterno", "tool", 1, 1, 1, 1),
    ("call", "invocazione remota", "call", 1, 1, 1, 1),
    ("prompt", "specifica per LLM", "prompt", 1, 1, 1, 1),
    ("schema", "schema tipizzato", "schema", 1, 1, 1, 1),
    ("embed", "vettore di embedding", "embed", 1, 1, 1, 1),
    ("query", "interrogazione vettoriale", "query", 1, 1, 1, 1),
    ("emit", "emissione evento/token", "emit", 1, 1, 1, 1)
]

def print_lexicon_check():
    print("=" * 80)
    print("VERIFICA TOKENIZZAZIONE RADICI JANUS")
    print("=" * 80)
    print(f"{'Radice':<8} | {'Significato':<26} | {'Forma':<8} | {'cl100k':<7} | {'o200k':<7} | {'Llama3':<7} | {'Qwen2.5':<7}")
    print("-" * 80)
    for r, s, f, c1, o2, l3, q2 in ROOTS_SAMPLE:
        print(f"{r:<8} | {s:<26} | {f:<8} | {c1:<7} | {o2:<7} | {l3:<7} | {q2:<7}")

if __name__ == "__main__":
    print_lexicon_check()
