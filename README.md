# JANUS: A Domain-Specific Language for Neural Computing and Constrained Decoding

[![CI](https://github.com/Pnda90/janus-lang/actions/workflows/ci.yml/badge.svg)](https://github.com/Pnda90/janus-lang/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](pyproject.toml)

> **Transpiler da DSL `.jn` a Python/PyTorch con grammatica deterministica LL(1), diagnostica JSON-first e decodifica vincolata (GBNF).**

---

## 🔬 Tesi da Verificare

Il progetto nasce per verificare empiricamente l'ipotesi:

> *"Un DSL con grammatica LL(1) e diagnostica leggibile dalle macchine permette a un LLM di scrivere codice di calcolo tensoriale corretto con meno token totali e più alta correttezza rispetto a Python."*

### Risultato Empirico della Tesi (Settembre/Ottobre 2026):
- **Efficienza dei Token (CONFUTATA sui tokenizer correnti):**  
  Sui tokenizer BPE standard pre-addestrati (`cl100k_base` di GPT-4 e `o200k_base` di GPT-4o), JANUS consuma il **+51.5% ~ +52.2% di token in più** rispetto al codice Python/PyTorch idiomatico equivalente, e quasi il doppio (+96% ~ +97%) rispetto a Python compatto.  
  *Causa:* I modelli linguistici correnti posseggono keyword Python (`def `, `return `, `import torch`) codificate come singoli token nel vocabolario BPE, mentre i costrutti del nuovo DSL e i tag di caso morfologico (`:m`, `:b`, `:t`, `stoc`) subiscono una severa frammentazione subword multi-token. Il punto di pareggio computazionale rispetto al costo del system prompt non viene mai raggiunto sui tokenizer generici.
- **Correttezza e Parsing LL(1) (CONFERMATA):**  
  La grammatica LL(1) garantisce parsing deterministico a tempo lineare e permette la generazione automatica di grammatiche **GBNF** valide per decodifica vincolata su motori di inferenza (*llama.cpp*, *vLLM*).
- **Diagnostica Machine-Readable (CONFERMATA):**  
  La diagnostica strutturata JSON (`./janusc check --json`) fornisce codice stabile, posizione esatta (riga/colonna) e suggerimenti correttivi che consentono un ciclo di auto-riparazione multi-turn automatizzato nei flussi agenziali.

---

## 📌 Stato del Progetto: Implementato vs Sperimentale vs Pianificato

Per garantire trasparenza scientifica, lo stato delle funzionalità è categorizzato in modo rigoroso:

### ✅ Implementato ed Eseguibile (Verificato con Test)
- **Lexer & Parser LL(1) Deterministico:** Sintassi non ambigua con morfologia a casi espliciti (`ident:case`, es. `x:m`, `w:b`).
- **Semantic Type Checker:** Risoluzione dello scope, rilevamento di variabili non definite, prevenzione di collisioni di parametri e tracciamento degli effetti monadici (`pure`, `io`, `stoc`).
- **Diagnostica LLM-Friendly:** Output JSON canonico (`Diagnostic.to_dict()`) contenente codice errore, coordinate span, token offending e patch correttiva.
- **Code Generator Python/PyTorch:** Transpilazione da `.jn` a Python 3.13 / PyTorch. Esecuzione reale con differenziazione automatica (`torch.autograd.grad`), struct con aritmetica tensoriale vettorializzata (`JanusStruct`), e convergenza numerica verificata (linreg, MLP, multi-head attention, RMSNorm, Conv2D).
- **Generatore & Validatore GBNF:** Compilazione di `schema` JANUS in regole di grammatica GBNF prive di doppi apici non bilanciati (`["]`), con validatore sintattico formale e test accept/reject.
- **Suite di Benchmark e Valutazione:** `benchmarks/tokens.py` (con `tiktoken` reale) e `benchmarks/llm_eval.py` (harness su 30 compiti tensoriali con pass@k e ciclo di autoriparazione).

### 🧪 Sperimentale (Prototipale / Non Ottimizzato)
- **Invocazione Agenti (`call agent:v`):** Compila in stub runtime di simulazione; non integrato con socket o broker RPC remoti.
- **Kernel GPU SAXPY (`kern`):** Validato a livello sintattico e transpilato in loop Python sequenziale; nessuna generazione di codice PTX/CUDA o kernel Triton nativi.
- **Inferenza Forme Simboliche:** Validazione sui tipi tensoriali a tempo di compilazione ma senza risolutore di vincoli SMT/Z3 completo per broadcasting multidimensionale dinamico.

### 📋 Pianificato (Non Esistente nel Codice Attuale)
- **Backend Nativo MLIR / LLVM:** Attualmente il runtime transila esclusivamente in Python; affermazioni di compilazione C/LLVM nativa appartengono alla roadmap futura.
- **Runtime Senza GC:** Il codice transpilato viene eseguito da CPython e risiede nell'allocatore con Garbage Collector di Python e PyTorch.
- **Tokenizer BPE Specialistico:** Addestramento di un vocabolario BPE proprietario con keyword JANUS fuse per verificare la tesi dei token a parità di rappresentazione lessicale.

---

## 📊 Risultati Misurati e Benchmark Riproducibili

Tutti i dati provengono direttamente da misurazioni eseguite salvate in `benchmarks/results/`. Nessun dato è hardcoded o stimato manualmente.

### 1. Benchmark Token (10 Programmi di Riferimento Completi)
Dati generati da `benchmarks/tokens.py` e registrati in [`benchmarks/results/tokens_benchmark.json`](benchmarks/results/tokens_benchmark.json):

| Metrica / Categoria | JANUS (`.jn`) | Python Idiomatico | Python Compatto | Differenza JANUS vs Idiomatico |
| :--- | :---: | :---: | :---: | :---: |
| **Righe di Codice (LOC)** | 99 | 86 | 40 | **+15.1%** |
| **Token `cl100k_base` (GPT-4)** | **1.414** | **929** | **718** | **+52.21%** |
| **Token `o200k_base` (GPT-4o)** | **1.413** | **933** | **720** | **+51.45%** |
| **Costo System Prompt In-Context** | 229 token | 0 token | 0 token | +229 token |
| **Punto di Pareggio (N. Programmi)** | *Mai raggiunto* | - | - | *JANUS costa costantemente di più* |

*Dettaglio per Programma (Tokenizer `cl100k_base`):*
- `01_linreg.jn`: JANUS 168 token vs Python Idiomatico 125 token (+34.4%)
- `02_mlp.jn`: JANUS 191 token vs Python Idiomatico 149 token (+28.2%)
- `03_attention.jn`: JANUS 71 token vs Python Idiomatico 69 token (+2.9%)
- `04_training_loop.jn`: JANUS 168 token vs Python Idiomatico 103 token (+63.1%)
- `05_conv2d.jn`: JANUS 103 token vs Python Idiomatico 63 token (+63.5%)
- `06_rmsnorm.jn`: JANUS 68 token vs Python Idiomatico 58 token (+17.2%)
- `07_diffusion.jn`: JANUS 111 token vs Python Idiomatico 78 token (+42.3%)
- `08_rag_pipeline.jn`: JANUS 178 token vs Python Idiomatico 88 token (+102.3%)
- `09_react_agent.jn`: JANUS 243 token vs Python Idiomatico 118 token (+105.9%)
- `10_gpu_saxpy.jn`: JANUS 113 token vs Python Idiomatico 82 token (+37.8%)

### 2. Valutazione LLM (30 Task Tensoriali)
Harness implementato in `benchmarks/llm_eval.py` ed eseguibile con qualsiasi provider LLM standard (OpenAI, Anthropic, Ollama):
- **Misura con API Pubbliche:** *Non misurata* su API esterne per assenza di `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` preimpostata nell'ambiente locale di esecuzione.
- **Validazione Architetturale (`--dry-run`):** Eseguita e salvata in [`benchmarks/results/llm_eval_dry_run.json`](benchmarks/results/llm_eval_dry_run.json), confermando il funzionamento del ciclo di auto-riparazione su fallimenti sintattici e il calcolo esatto di pass@1 e pass@5.

---

## ⚖️ Confronto Sintetico con Altri Linguaggi e Framework

| Caratteristica | JANUS (`.jn`) | Triton (OpenAI) | JAX (Google) | Mojo (Modular) |
| :--- | :--- | :--- | :--- | :--- |
| **Obiettivo Primario** | DSL per LLM generation, constrained decoding e autodiff | Compilatore per kernel GPU ad alte prestazioni (blocco tensoriale) | Calcolo differenziabile funzionale puro per CPU/GPU/TPU | Linguaggio di sistemi compilato nativo compatibile Python |
| **Target Compilazione** | Python 3.13 / PyTorch (MLIR pianificato) | LLVM / PTX / AMDGPU | XLA / HLO IR | LLVM Machine Code nativo |
| **Grammatica** | LL(1) deterministica, export diretto GBNF | Sottoinsieme Python (AST Python) | Python nativo (tracciamento funzionale) | Sintassi Python-like con tipi statici |
| **Constrained Decoding** | Diretto via automi GBNF per schemi | Non supportato | Non supportato | Non supportato |
| **Autodiff** | Primitiva `diff loss wrt wb` via PyTorch | Non integrata (kernel forward) | Primitiva funzionale di prima classe (`jax.grad`) | Non integrata nativamente nel frontend |
| **Overhead Token LLM** | Elevato sui tokenizer attuali (+52% vs Python) | Elevato (codice Python dettagliato di basso livello) | Standard Python | Standard Python/Rust-like |
| **Runtime & Memoria** | Runtime CPython + PyTorch GC | Driver GPU C++ / PyTorch wrapper | C++ runtime XLA / Zero-overhead | Runtime proprio compilato nativo senza GC obbligatorio |

---

## 🏛️ Sintassi e Morfologia a Casi

La morfologia di JANUS associa esplicitamente a ogni variabile o parametro il proprio ruolo tramite la notazione `ident:case`:

```
CASO         SEPARATORE   RUOLO NEL GRAFO COMPUTAZIONALE
----------------------------------------------------------------------------------------
Accusativo   :m           Operando / Input primario di calcolo
Ablativo     :b           Parametro peso / Iperparametro ausiliario
Dativo       :t           Buffer di memoria / Target di mutazione in-place
Nominativo   :n           Definizione valore SSA / Assegnazione sinistra
Genitivo     :s           Specifica di tipo / Forma tensoriale simbolica
Vocativo     :v           Invocazione di Agente Cognitivo o Tool Esterno
----------------------------------------------------------------------------------------
```

### Esempio: Regressione Lineare con SGD Nativo
```janus
# examples/01_linreg.jn
fn linreg(x:m, y:m, epochs: i32, lr: f32) pure {
    mut w:b = 0.0
    mut b:b = 0.0
    for ep in 0..epochs {
        pred:n = x:m * w:b + b:b
        loss:n = ((pred:n - y:m) pow 2) sum / x:m.len
        gw, gb = diff loss:n wrt (w:b, b:b)
        w:b = w:b - lr * gw
        b:b = b:b - lr * gb
    }
    ret (w:b, b:b)
}
```

---

## 🚀 Istruzioni di Installazione ed Esecuzione

### 1. Configurazione Ambiente
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest hypothesis tiktoken requests
pip install -e .
```

### 2. Esecuzione della Test Suite (66 test)
```bash
pytest -v
```

### 3. Riproduzione del Benchmark dei Token
Esegue il conteggio esatto sui 10 programmi con `tiktoken` reale:
```bash
python benchmarks/tokens.py
```

### 4. Esecuzione dell'Harness di Valutazione LLM
Modalità simulata (dry-run):
```bash
python benchmarks/llm_eval.py --dry-run
```

Modalità reale con modello API (richiede variabile d'ambiente):
```bash
export OPENAI_API_KEY="sk-..."
python benchmarks/llm_eval.py --provider openai --model gpt-4o-mini --samples 5
```

### 5. Utilizzo della CLI (`janusc`)
```bash
# Controllo sintassi, tipi ed effetti
./janusc check examples/01_linreg.jn

# Diagnostica strutturata per LLM
./janusc check examples/01_linreg.jn --json

# Compilazione verso Python/PyTorch
./janusc compile examples/03_attention.jn -o attention.py

# Esecuzione immediata
./janusc run examples/01_linreg.jn

# Generazione grammatica GBNF per llama.cpp
./janusc gbnf examples/08_rag_pipeline.jn
```

---

## ⚠️ Limiti Noti

1. **Target Python:** Poiché JANUS compila attualmente a codice sorgente Python che viene interpretato dal runtime CPython, le prestazioni temporali e di memoria sono vincolate a PyTorch e all'interprete Python sottostante.
2. **Subword Fragmentation:** L'adozione di costrutti non compresi nei vocabolari dei Foundation Model odierni genera una frammentazione dei token sfavorevole.
3. **Autodiff e Parametri Mutabili:** La differenziazione automatica richiede che le variabili nel grafo siano gestite tramite foglie tensoriali PyTorch con `requires_grad=True` e `detach()`.
