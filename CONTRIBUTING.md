# Guida per i Contributori (CONTRIBUTING.md)

Grazie per l'interesse a contribuire a **JANUS**! Questo repository implementa un DSL tipizzato con effect system per la specifica di interfacce e flussi agentici di LLM, con sintesi di grammatiche GBNF per la decodifica vincolata.

---

## 1. Principi Fondamentali del Progetto

1. **Integrità dei Dati e Nessun Numero Inventato**:
   - Qualsiasi metrica, percentuale o tempo riportato nella documentazione (`README.md`, `REPORT.md`, `ABSTRACT.md`) **deve provenire direttamente** da un file di risultato presente in `benchmarks/results/`, generato da uno script eseguibile e riproducibile presente in `benchmarks/`.
   - Se una misura non può essere eseguita (es. assenza di modelli locali o GPU nell'ambiente), contrassegnala esplicitamente come "da eseguire" o "simulata", specificando lo script esatto per raccogliere i dati reali.

2. **Test Baseline Sempre Verde**:
   - Ogni Pull Request o commit deve mantenere `pytest -v` verde al 100%.
   - Ogni nuova funzionalità richiede unit test che coprano sia i casi positivi che i casi negativi (errori attesi, diagnostiche del compilatore, violazioni di tipo/effetto).

3. **Rigore del Compilatore e dell'Effect System**:
   - JANUS include un type checker formale con propagazione transitiva degli effetti (`pure`, `io`, `stoc`).
   - I tool marcati come `pure` non devono poter invocare operazioni o tool `io`/`stoc` né direttamente né indirettamente.
   - Le grammatiche GBNF generate devono essere verificate tramite `GBNFValidator` per garantire la compatibilità con `llama.cpp`.

---

## 2. Configurazione dell'Ambiente Locale

```bash
# Clona il repository
git clone https://github.com/Pnda90/janus-lang.git
cd janus-lang

# Crea ed attiva il virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# Installa in modalità editable con le dipendenze di sviluppo
pip install -e ".[dev]"
```

---

## 3. Esecuzione dei Test

Prima di effettuare un commit, assicurati di eseguire:

```bash
# Esegui tutti i test
pytest -v

# Esegui i controlli di integrità dei benchmark
pytest tests/test_no_hardcoded_benchmarks.py -v
```

> **Nota per i test dei benchmark**: Il test `test_token_benchmark_artifact_integrity` riesegue il benchmark dei token. Se non hai modificato il tokenizer o la sintassi dei token, ripristina il file di output per non creare diff superflui:
> ```bash
> git checkout benchmarks/results/tokens_benchmark.json
> ```

---

## 4. Convenzione sui Commit

Utilizziamo lo standard **Conventional Commits**:
- `feat(...)`: nuova funzionalità (es. `feat(effects): implement transitive effect propagation`)
- `fix(...)`: correzione di bug (es. `fix(gbnf): prevent adjacent double quotes in rules`)
- `docs(...)`: modifiche alla documentazione (es. `docs: update quickstart guide`)
- `test(...)`: aggiunta o aggiornamento di test (es. `test(runtime): add strict effects checks`)
- `refactor(...)`: refactoring del codice che non modifica il comportamento esterno

---

## 5. Struttura del Codice

- `janus/lexer.py`, `janus/parser.py`: Lexer e parser ricorsivo discendente LL(1).
- `janus/type_checker.py`: Sistema dei tipi, controllo ruoli morfologici ed effect system (chiusura transitiva a punto fisso).
- `janus/gbnf_gen.py`, `janus/gbnf_validator.py`: Generazione e validazione sintattica di grammatiche GBNF.
- `janus/agent_runtime.py`: Runtime per l'orchestrazione degli agenti, ToolSandbox e audit trail.
- `janus/codegen.py`: Generazione di codice target (Python, PyTorch).
- `benchmarks/`: Script di benchmark per token e task agentici (`agent_eval.py`).
- `tests/`: Suite completa di test unitari e di regressione.
