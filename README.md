# JANUS: Agentic Execution Graph DSL with Effect System & Constrained Decoding

[![CI](https://github.com/Pnda90/janus-lang/actions/workflows/ci.yml/badge.svg)](https://github.com/Pnda90/janus-lang/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](pyproject.toml)

> **Un Domain-Specific Language LL(1) per l'orchestrazione sicura di tool agentici, con effect system statico, sintesi di grammatiche GBNF per decodifica vincolata a livello di inferenza e transpilazione verso Python (3.10+) / PyTorch.**

---

## 🎯 Panoramica e Visione Architetturale

Nei sistemi agenziali contemporanei (LangChain, CrewAI, AutoGen, OpenAI tool-calling), l'invocazione di tool tramite prompt in linguaggio naturale o JSON non vincolato soffre di criticità note:
1. **Errori di schema e sintassi:** Il modello genera JSON malformato, omette parametri obbligatori o inventa argomenti spuri.
2. **Type mismatch a runtime:** Valori stringa passati al posto di interi o float provocano fallimenti a runtime.
3. **Assenza di confini per gli effetti collaterali:** Mancanza di distinzione formale tra calcoli puri e operazioni con mutazioni esterne (I/O, cancellazione file, chiamate di rete).

**JANUS** affronta queste problematiche attraverso la teoria dei compilatori:
- **Linguaggio con Effect System per Tool Agentici:** Contratti espliciti di tool (`schema Name [effect]`) con tracciamento formale della purezza monadica (`pure`, `io`, `stoc`) e propagazione transitiva tra funzioni.
- **Sintesi di Grammatiche GBNF:** Compilazione diretta degli schemi in grammatiche formali GBNF per motori di decodifica vincolata (*llama.cpp*, *Ollama*, *vLLM*). Il token masking a livello di logit garantisce **conformità sintattica e rispetto dei tipi dello schema per costruzione**.
- **Campionamento Stocastico e Semantica:** La decodifica vincolata assicura che ogni token emesso sia formalmente valido rispetto alla grammatica; la selezione del tool e il valore semantico degli argomenti restano tuttavia soggetti alla natura probabilistica del campionamento dell'LLM.
- **Tool Sandbox Runtime (`janus.agent_runtime`):** Ambiente di esecuzione applicativo con audit trail granulare (`ToolTrace`), profiling temporale, intercettazione dry-run e verifica opzionale `strict_effects`.

---

## ⚖️ Perché un DSL e non JSON Schema + Pydantic?

Un confronto onesto tra JANUS e le alternative industriali contemporanee:

| Dimensione | JSON Schema Nativo (llama.cpp / OpenAI) | Pydantic (Python) | Outlines / XGrammar | JANUS DSL |
| :--- | :---: | :---: | :---: | :---: |
| **Validazione Tipi a Runtime** | No (runtime delegato) | ✅ Eccellente | No (runtime delegato) | ✅ Integrata |
| **Token Masking a Livello di Logit** | ✅ Tramite runtime C++ | ❌ No | ✅ Ottimizzato a basso livello | ✅ Tramite GBNF generata |
| **Effect System Statico (`pure` vs `io`)** | ❌ Assente | ❌ Assente | ❌ Assente | ✅ **Statico & Transitivo** |
| **Pipeline Logica Multi-Tool Compilata** | ❌ Solo schemi singoli | ❌ Solo classi Python | ❌ Solo regex/CFG | ✅ **AST con flussi e composizione** |
| **Diagnostiche Machine-Readable per Self-Repair** | ❌ Errori generici | ❌ Traceback Python | ❌ Eccezioni motore | ✅ **JSON con spans e patch** |

### Cosa JANUS aggiunge davvero:
1. **Effect System Statico:** Il compilatore verifica a tempo di analisi statica che una funzione dichiarata `pure` non possa invocare tool con effetti `io` o `stoc`, né direttamente né transitivamente attraverso chiamate intermedie.
2. **Diagnostica Machine-Readable per Autoriparazione:** Il comando `./janusc check --json` emette diagnostiche standardizzate con codici stabili (`ERR_EFFECT_PURITY_VIOLATION`, `ERR_UNDEFINED_TOOL`), coordinate di riga/colonna e patch correttive automatiche utilizzabili direttamente in loop di self-repair dell'agente.
3. **Unica Fonte di Verità:** Da una singola specifica JANUS vengono derivati sia il codice Python/PyTorch eseguibile, sia la grammatica GBNF per il vincolo dell'inferenza, garantendo sincronizzazione intrinseca tra specifica e runtime.

### Cosa JANUS NON aggiunge:
- JANUS non è un nuovo motore di campionamento logit C/C++: delega il token masking effettivo a runtime consolidati come *llama.cpp* o *vLLM* tramite la grammatica GBNF generata.
- Per chiamate puntuali a un singolo tool senza pipeline composite e senza necessità di tracciamento degli effetti collaterali, la combinazione standard di JSON Schema con Pydantic è più immediata e ampiamente supportata dai provider commerciali.

---

## ⚠️ Limiti Noti

In ottica di trasparenza metodologica, l'architettura di JANUS presenta i seguenti limiti intrinseci:

1. **La grammatica non garantisce la correttezza semantica:** Il token masking assicura che il payload sia un JSON valido conforme allo schema e ai tipi; non impedisce tuttavia che il modello scelga il tool errato rispetto al contesto o inserisca argomenti semanticamente incoerenti (es. coordinate geografiche inventate, query di ricerca non attinenti).
2. **Il vincolo può influire sul ragionamento (Reasoning Degradation):** Obbligare il modello a generare token rigidamente conformi a una grammatica fin dal primo carattere rimuove lo spazio per catene di pensiero preliminari (chain-of-thought o scratchpad). Questo compromesso è documentato nella letteratura scientifica e richiede una misurazione attenta nei benchmark.
3. **L'effetto di un tool è dichiarato dall'utente e non verificato dal compilatore nel codice host:** Il compilatore verifica i contratti dichiarati nel sorgente JANUS; tuttavia, se una funzione Python esterna registrata con effetto `pure` esegue internamente chiamate di rete o I/O su file, il compilatore non può analizzare staticamente il codice Python arbitrario sottostante.
4. **ToolSandbox non è un isolamento di sicurezza a livello di processo/OS:** La classe `ToolSandbox` fornisce un confine applicativo utile per tracciamento, audit trail, profiling dei tempi e modalità dry-run. Non costituisce una sandbox di sicurezza forte contro codice non fidato (non impiega container Linux, cgroups, seccomp o microVM).

---

## 🔬 Risultati Empirici e Validazione

Nessun dato è inventato o hardcoded: ogni cifra deriva da script eseguibili tracciati in `benchmarks/results/`.

### 1. Benchmark Agentic Tool Calling (20 Task Realistici)
Script: `benchmarks/agent_eval.py` | Risultati simulati: [`benchmarks/results/simulated_agent_eval_*.json`](benchmarks/results/)

Confronto preliminare (baseline simulata con failure mode realistici documentati in letteratura su 20 scenari di tool calling):

| Metrica | Unconstrained JSON (Simulato) | JSON Schema Nativo | JANUS GBNF Constrained | Garanzia Fornita |
| :--- | :---: | :---: | :---: | :--- |
| **Sintassi JSON Valida** | 90.0% ± 0.0% | **100.0% ± 0.0%** | **100.0% ± 0.0%** | Token masking (zero JSON troncati o invalidi) |
| **Conformità dello Schema** | 76.7% ± 2.4% | **100.0% ± 0.0%** | **100.0% ± 0.0%** | Zero parametri allucinati o obbligatori omessi |
| **Accuratezza dei Tipi** | 73.3% ± 2.4% | **100.0% ± 0.0%** | **100.0% ± 0.0%** | Parametri forzati dai vincoli grammaticali |
| **Scelta Tool e Semantica Argomenti** | Variabile da prompt | Variabile da prompt | Variabile da prompt | Il campionamento resta stocastico |
| **Token Consumati (`cl100k_base`)** | 679 ± 9 | 615 ± 0 | **615 ± 0** | Eliminazione di chiavi e metadati spuri |

> [!NOTE]
> **Stato delle Misure su LLM Reale:** *Da eseguire su endpoint locale.*
> Il codice per la misurazione live su endpoint HTTP è implementato in `benchmarks/agent_eval.py`. Per eseguire la misurazione su un modello locale tramite *llama.cpp*, *Ollama* o *vLLM*:
> ```bash
> # Con llama.cpp server locale:
> python benchmarks/agent_eval.py --backend llama-cpp --url http://localhost:8080 --model meta-llama/Llama-3.2-3B-Instruct --seeds 42,43,44
>
> # Con Ollama locale:
> python benchmarks/agent_eval.py --backend ollama --url http://localhost:11434 --model llama3.2 --seeds 42,43,44
> ```

---

## 🧪 Esperimenti: Backend Tensoriale ed Efficienza dei Token

Oltre all'orchestrazione agenziale, JANUS include un backend di calcolo differenziabile sperimentale con transpilazione verso PyTorch.

### 1. Risultato Negativo sulla Compressione Lessicale dei Token
Script: `benchmarks/tokens.py` | Risultati: [`benchmarks/results/tokens_benchmark.json`](benchmarks/results/tokens_benchmark.json)  
Valutazione condotta sui tokenizer commerciali Foundation Model (`cl100k_base` GPT-4 e `o200k_base` GPT-4o) su 10 programmi completi:

- **Esito Sperimentale:** Su tokenizer BPE generici, JANUS consuma il **+52.2% di token in più rispetto a Python idiomatico**.
- **Motivazione Tecnica:** I vocabolari BPE sono fortemente ottimizzati sui costrutti standard di Python (`def `, `return `, `import torch` occupano singoli token atomici). La sintassi proprietaria di un DSL subisce frammentazione subword (es. `:m`, `stoc`).
- **Conclusioni:** Senza un tokenizer specificamente ri-addestrato sul vocabolario del DSL, un nuovo linguaggio non riduce i token per puro calcolo matematico. L'utilità di JANUS risiede nella **separazione degli effetti e nella decodifica vincolata a livello grammaticale**.

### 2. Autodiff e Strutture Vettorializzate
Il compilatore include il costrutto `diff target wrt param` e strutture dati `JanusStruct` con sovraccarico vettoriale out-of-place compatibili con `torch.autograd.grad`. I 10 programmi di riferimento (MLP, Attention, Conv2D, RMSNorm, Linear Regression, Training Loop) convergono numericamente con scarto nullo rispetto ai corrispettivi riferimenti PyTorch nativi (verificato in `tests/test_execution_reference.py`).

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

### 3. Guardrail di Effetto e Propagazione Transitiva
Il compilatore applica rigorosamente le regole di purezza monadica:
- Una funzione dichiarata `pure` non può invocare né direttamente né transitivamente tool o funzioni con effetto `io` o `stoc`.
- Il type checker emette diagnostiche strutturate con codice stabile `ERR_EFFECT_PURITY_VIOLATION` per l'autoriparazione automatica.

---

## 📌 Stato dei Moduli: Implementato vs Pianificato

| Componente | Stato | Dettagli |
| :--- | :---: | :--- |
| **Lexer & Parser LL(1)** | ✅ Implementato | Sintassi non ambigua, annotazioni di caso (`:m`, `:b`), `schema`, `call tool` |
| **Type & Effect Checker** | ✅ Implementato | Scope rigoroso, diagnostica JSON strutturata, controllo di purezza transitivo |
| **Tool Sandbox Runtime** | ✅ Implementato | `ToolSandbox`, tracciamento `ToolTrace`, audit trail, modalità dry-run e `strict_effects` |
| **Sintetizzatore GBNF** | ✅ Implementato | Generazione multi-tool per script e JSON tool calling, validatore formale |
| **Transpiler Python/PyTorch**| ✅ Implementato | Compatibile con Python 3.10+, supporto autodiff verificato |
| **Suite di Test (87+ test)** | ✅ Implementato | Regressione B1-B5, test Hypothesis, conformità GBNF, benchmark audit |
| **Backend Nativo MLIR / LLVM** | 📋 Pianificato | Compilazione nativa bare-metal senza interprete host |

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

### 2. Esecuzione dei Test
```bash
pytest -v
```

### 3. Utilizzo della CLI (`janusc`)
```bash
# Validazione sintattica, tipi ed effetti
./janusc check examples/11_safe_tool_pipeline.jn

# Diagnostica JSON strutturata (per autoriparazione agentica)
./janusc check examples/11_safe_tool_pipeline.jn --json

# Compilazione verso codice Python eseguibile
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
print(f"Esito: {result}, Traces registrate: {len(traces)}")
```

---

## 📄 Licenza

Rilasciato sotto licenza [Apache 2.0](LICENSE).
Autore: **Pnda90** ([GitHub](https://github.com/Pnda90/janus-lang)).
