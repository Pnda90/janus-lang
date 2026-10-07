# Metadati Raccomandati per il Repository GitHub

Questo documento contiene le raccomandazioni ufficiali per la configurazione dei metadati del repository su GitHub (Description, Topics e Positioning).

---

## 1. Description del Repository (1 riga)

### Versione Consigliata (Italiano)
> DSL tipizzato con effect system per tool calling di agenti LLM e decodifica vincolata GBNF.

### Versione Internazionale (Inglese)
> Typed DSL with an effect system for LLM agent tool-calling and GBNF-constrained decoding.

---

## 2. GitHub Topics Suggeriti

Inserire i seguenti argomenti nella sezione **About -> Topics** delle impostazioni della repository GitHub:

- `dsl`
- `constrained-decoding`
- `gbnf`
- `llm-agents`
- `tool-calling`
- `type-system`
- `effect-system`
- `llama-cpp`
- `compiler`
- `python`
- `structured-outputs`
- `grammar-based-sampling`

---

## 3. Rationale del Posizionamento

1. **Focus Primario Ristretto**: JANUS si posiziona come un linguaggio di programmazione/DSL orientato alla specificazione sicura di interfacce e flussi agentici per LLM.
2. **Differenziazione da JSON Schema puro**: L'inclusione di `effect-system` e `type-system` evidenzia il valore aggiunto del compilatore (analisi statica della propagazione degli effetti `pure`/`io`/`stoc`, diagnostiche machine-readable).
3. **Integrazione Runtime**: `constrained-decoding` e `gbnf` segnalano la compatibilità con i principali motori di inferenza locale (`llama.cpp`, `vLLM`).
