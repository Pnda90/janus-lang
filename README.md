# JANUS: Deterministic Agentic Execution Graph DSL & Constrained Decoding

[![CI](https://github.com/Pnda90/janus-lang/actions/workflows/ci.yml/badge.svg)](https://github.com/Pnda90/janus-lang/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](pyproject.toml)

> **Un Domain-Specific Language LL(1) per l'orchestrazione sicura di tool, sintesi di grammatiche GBNF per decodifica vincolata (zero allucinazioni di schema) e transpilazione deterministica verso Python 3.13 / PyTorch.**

---

## 🎯 Panoramica e Visione Architetturale

Nei sistemi agenziali contemporanei (LangChain, CrewAI, AutoGen, OpenAI tool-calling), l'invocazione di tool tramite JSON non vincolato soffre di vulnerabilità intrinseche:
1. **Allucinazioni di schema:** I modelli inventano parametri arbitrari o omettono argomenti obbligatori (tasso di violazione: ~20%).
2. **Type mismatch a runtime:** Passaggio di stringhe al posto di interi o float, causando crash dell'interprete.
3. **Violazioni dei side-effect:** Mancanza di confini formali tra calcoli verificabili e operazioni con effetti collaterali (I/O, cancellazione file, mutazione di stato).

**JANUS** risolve questi limiti attraverso un approccio basato sulla teoria dei compilatori:
- **Schemi di Tool Tipizzati di Prima Classe (`schema Name [effect]`):** Contratti formali con tipi statici, parametri opzionali e tracking dell'effetto monadico (`pure`, `io`, `stoc`).
- **Sintesi Multi-Tool GBNF:** Compilazione automatica degli schemi in grammatiche formali GBNF compatibili con *llama.cpp*, *Ollama* e *vLLM*, garantendo **100% di conformità sintattica e di tipo per costruzione matematica (token masking)**.
- **Tool Sandbox Runtime (`janus.agent_runtime`):** Esecuzione isolata con audit trail granulare (`ToolTrace`), misurazione dei tempi, intercettazione degli errori e supporto nativo a modalità dry-run.
- **Backend di Calcolo Tensoriale e Autodiff:** Supporto integrato per calcoli neurali con differenziazione automatica (`diff`), compilati in Python 3.13 / PyTorch.

---

## 🔬 Risultati Empirici e Validazione Scientifica

Il progetto è guidato da una metodologia di misurazione trasparente: **nessun dato hardcoded; ogni cifra deriva da script eseguibili salvati in `benchmarks/results/`**.

### 1. Benchmark Agentic Tool Calling (20 Task Realistici)
Script: `benchmarks/agent_eval.py` | Risultati: [`benchmarks/results/agent_eval_dry_run_*.json`](benchmarks/results/)

Confronto tra decodifica non vincolata (Unconstrained JSON Tool Calling) e decodifica vincolata JANUS GBNF:

| Metrica di Affidabilità | Unconstrained JSON | JANUS GBNF Constrained | Guadagno / Garanzia |
| :--- | :---: | :---: | :--- |
| **Sintassi JSON Valida** | 95.0% | **100.0%** | Zero JSON troncati o malformati |
| **Conformità dello Schema** | 80.0% | **100.0%** | Zero parametri allucinati o omessi |
| **Accuratezza dei Tipi** | 75.0% | **100.0%** | Parametri forzati dal token masking |
| **Invocazioni Perfette al 1° Tentativo** | 75.0% | **100.0%** | Determinismo matematico |
| **Token Consumati (cl100k_base)** | 484 token | **411 token** | **-15.1%** (eliminazione di rambling e metadata non richiesti) |

### 2. Valutazione della Tesi sui Token nel Calcolo Tensoriale (10 Programmi)
Script: `benchmarks/tokens.py` | Risultati: [`benchmarks/results/tokens_benchmark.json`](benchmarks/results/tokens_benchmark.json)

- **Verdetto:** Sui tokenizer BPE generici (`cl100k_base` GPT-4, `o200k_base` GPT-4o), JANUS consuma il **+52.2% di token in più rispetto a Python idiomatico**.
- **Causa:** Le keyword Python (`def `, `return `, `import torch`) sono singoli token atomici nei vocabolari commerciali, mentre la sintassi proprietaria subisce frammentazione subword.
- **Conclusione Tecnica:** Il valore di JANUS non risiede nella compressione dei token per calcolo matematico puro, ma nella **garanzia formale di correttezza e zero allucinazioni** per l'orchestrazione agenziale.

---

## 🏛️ Costrutti del Linguaggio

### 1. Dichiarazione di Schemi di Tool ed Effetti
```janus
schema WebSearch io {
    query: str,
    top_k: i32 = 10
} -> {
    raw_results: str
}

schema TextClassifier pure {
    content: str,
    threshold: f32 = 0.5
} -> {
    is_relevant: bool,
    confidence: f32
}
```

### 2. Pipeline Agenziale e Invocazione Protetta
```janus
# examples/11_safe_tool_pipeline.jn
fn orchestrate_search_and_cache(query: str, min_confidence: f32) io {
    search_res = call tool WebSearch(query = query, top_k = 3)
    filter_res = call tool TextClassifier(content = search_res.raw_results, threshold = min_confidence)
    if filter_res.is_relevant {
        cache_res = call tool CacheStorage(key = query, value = search_res.raw_results)
        ret 1
    }
    ret 0
}
```

### 3. Guardrail di Effetto (`pure` vs `io`)
Il compilatore applica rigorosamente le regole di purezza monadica:
- Una funzione dichiarata `pure` non può chiamare tool con effetto `io`.
- Il type checker segnala tempestivamente `ERR_EFFECT_PURITY_VIOLATION` con coordinate esatte e diagnostica machine-readable per autoriparazione.

---

## 📌 Stato dei Moduli: Implementato vs Pianificato

| Componente | Stato | Dettagli |
| :--- | :---: | :--- |
| **Lexer & Parser LL(1)** | ✅ Implementato | Sintassi non ambigua, morfologia a casi (`:m`, `:b`), `schema`, `call tool` |
| **Type & Effect Checker** | ✅ Implementato | Scope rigoroso, diagnostica JSON con codici stabili, guardrail monadici |
| **Tool Sandbox Runtime** | ✅ Implementato | `ToolSandbox`, tracciamento `ToolTrace`, audit trail, modalità dry-run |
| **Sintetizzatore GBNF** | ✅ Implementato | Generatore multi-tool, validatore sintattico formale, test accept/reject |
| **Transpiler Python/PyTorch**| ✅ Implementato | Generazione codice per 12 esempi verificati con autodiff reale |
| **Suite di Test (84 test)** | ✅ Implementato | Regressione B1-B5, test Hypothesis, esecuzione PyTorch CPU, benchmark audit |
| **Backend Nativo MLIR / LLVM** | 📋 Pianificato | Compilazione nativa C/WASM senza interprete Python |

---

## 🚀 Guida Rapida: Installazione ed Utilizzo

### 1. Setup dell'Ambiente
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest hypothesis tiktoken requests
pip install -e .
```

### 2. Esecuzione dei Test (84 test verdi)
```bash
pytest -v
```

### 3. Utilizzo della CLI (`janusc`)
```bash
# Validazione sintattica, tipi ed effetti
./janusc check examples/11_safe_tool_pipeline.jn

# Diagnostica JSON strutturata (per flussi agenziali con autoriparazione)
./janusc check examples/11_safe_tool_pipeline.jn --json

# Compilazione verso Python/PyTorch
./janusc compile examples/11_safe_tool_pipeline.jn -o pipeline.py

# Generazione della grammatica GBNF per llama.cpp
./janusc gbnf examples/11_safe_tool_pipeline.jn
```

### 4. Utilizzo di `AgentRuntime` in Python
```python
from janus.agent_runtime import AgentRuntime

runtime = AgentRuntime()

@runtime.sandbox.tool("WebSearch", effect="io")
def search(query: str, top_k: int = 10):
    return {"raw_results": f"Risultati per: {query}"}

code = open("examples/11_safe_tool_pipeline.jn").read()
result, traces = runtime.execute(code, entrypoint="orchestrate_search_and_cache", args=["machine learning", 0.7])
print(f"Esito: {result}, Traces: {len(traces)}")
```

### 5. Esecuzione dei Benchmark
```bash
# Benchmark Agentic Tool Calling (confronto unconstrained vs GBNF)
python benchmarks/agent_eval.py --dry-run

# Benchmark Token (misurazione LOC e BPE sui 10 programmi)
python benchmarks/tokens.py

# Harness di valutazione LLM tensoriale
python benchmarks/llm_eval.py --dry-run
```

---

## 📄 Licenza

Rilasciato sotto licenza [Apache 2.0](LICENSE).
Autore: **Pnda90** ([GitHub](https://github.com/Pnda90/janus-lang)).
