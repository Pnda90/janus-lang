# Changelog

Tutte le modifiche rilevanti a questo progetto sono documentate in questo file.
Il formato è ispirato a [Keep a Changelog](https://keepachangelog.com/it/1.0.0/) e aderisce al [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-07

### Breaking Changes
- **Fail-Closed sui Tool Non Registrati**: La chiamata a tool non registrati nel runtime Python compilato solleva ora l'eccezione esplicita `JanusToolNotRegistered` (o `ToolNotFoundError` nella sandbox), invece di ritornare silenziosamente un dizionario mock con `{'status': 'simulated'}`.
- **Rigorosità dell'Effect System**:
  - Le funzioni dichiarate come `pure` che invocano tool con effetto `io` o `stoc` causano ora un errore semantico fatale a tempo di compilazione (`ERR_EFFECT_PURITY_VIOLATION`).
  - Le funzioni `stoc` non possono invocare tool `io` (`ERR_EFFECT_ESCALATION`).
  - L'invocazione di funzioni o tool inesistenti emette `ERR_UNDEFINED_FUNCTION` o `ERR_UNDEFINED_SCHEMA`.
- **Exit Code CLI POSIX-compliant**: Il compilatore `janusc` adotta codici di uscita standard:
  - `0`: Successo
  - `1`: Errore di compilazione, lexing, parsing o type checking
  - `2`: Errore di I/O, file non trovato o argomenti CLI errati.

### Added (Features)
- **Sintesi GBNF Conforme RFC 8259 (`janus/gbnf_gen.py`)**:
  - Regola `string` con escape completi JSON (`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, `\uXXXX`) ed esclusione dei caratteri di controllo ASCII `U+0000–U+001F`.
  - Distinzione numerica tra `integer` (`i8`, `i16`, `i32`, `i64`) senza zero iniziale (`007` vietato) e `number` floating-point (`f16`, `f32`, `f64`, `bf16`).
  - Supporto a tipi composti `ListType` e `CustomType`, con eccezione esplicita `GBNFGenerationError` per tipi non gestiti (nessun fallback silenzioso a stringa).
  - Ordinamento canonico dei parametri: obbligatori per primi in ordine di dichiarazione, seguiti da parametri opzionali.
  - Grammatica agentica completa con supporto a `if`/`else`, accesso a campi record (`ident.field`), e spazio obbligatorio dopo `ret`.
  - Nuove opzioni CLI `janusc gbnf --mode call|output|agent` e `--with-status`.
- **Esportazione JSON Schema (`janus/jsonschema_export.py`)**:
  - Conversione da dichiarazioni `schema` JANUS verso JSON Schema Draft-07 tramite `schema_to_json_schema()`.
  - Nuovo comando CLI `janusc export-jsonschema <file.jn> [--mode call|output]`.
- **Motore di Riconoscimento e Campionamento GBNF (`tests/support/gbnf_engine.py`)**:
  - Parser formale GBNF, riconoscitore memoizzato (Packrat) e campionatore stocastico seedable (`GBNFSampler`) per test differenziali e fuzzing.
- **Type Checker Avanzato (`janus/type_checker.py`)**:
  - Controllo di tipo per argomenti tool (widening consentito solo da `int` a `float`; errore `ERR_TOOL_ARG_TYPE_MISMATCH` per `float` verso `int` o tipi incompatibili).
  - Validazione arità e argomenti duplicati (`ERR_TOOL_ARITY`, `ERR_DUPLICATE_TOOL_ARGUMENT`).
  - Tipizzazione formale output tool tramite `ToolOutputType` e validazione accesso a campi (`ERR_UNKNOWN_OUTPUT_FIELD`).
  - Nuovo flag CLI `janusc check --effects` per ispezione della tabella degli effetti sia in formato tabellare che in JSON machine-readable.
- **Runtime Agentico e Sandbox (`janus/agent_runtime.py`)**:
  - True dry-run: esecuzione effettiva dei soli tool `pure` registrati e mock tipizzato per tool `io`/`stoc` (marcati con `mocked=True` nei trace).
  - Validazione runtime bidirezionale di argomenti e output contro lo schema (`ToolValidationError`).
  - Verifica di coerenza effetto a runtime tra schema dichiarato e funzione registrata (`ToolEffectMismatchError`).
  - Difesa cooperativa `strict_effects` thread-safe tramite flag `threading.local()` e monkey-patching ripristinato in `try/finally`.
- **Benchmark Multi-Backend Onesto (`benchmarks/agent_eval.py`)**:
  - Supporto per server LLM reali via HTTP (`llama-cpp`, `ollama`, `vllm`).
  - Condizione di baseline nativa JSON Schema derivata da `schema_to_json_schema`.
  - Modalità mock basata su campionamento reale da GBNF e generatore sintetico di errori.
  - Prefisso esplicito `simulated_` per tutti i file di risultati non derivati da inferenza reale, con metadati `is_simulated`, `commit`, e `janus_version`.

### Fixed (Bugs)
- **Lexer Robusto (`janus/lexer.py`)**:
  - Classe di eccezione dedicata `LexError` con coordinate esatte (linea, colonna, carattere).
  - Intercettazione di caratteri illegali, numeri malformati (`1e`, `1.2.3`), e stringhe non terminate a fine riga o EOF.
  - Supporto per letterali stringa con apici singoli `'...'` e operatori `%`, `^`, `!`.
  - Fuzz testing con Hypothesis per garantire l'assenza di crash non gestiti.
- **Benchmark Hardcoded Eliminati**: Rimosso qualsiasi risultato prefabbricato (100% o 75% fittizi); i risultati riflettono esclusivamente misure su modelli reali o simulazioni chiaramente etichettate.

### Changed (Refactoring & Docs)
- **Unificazione Versione**: Allineata la versione a `0.2.0` in `pyproject.toml`, `janus/__init__.py`, codegen e documentazione.
- **Pulizia del Testo**: Rimosse affermazioni iperboliche e promozionali dai messaggi CLI `--help` in favore di descrizioni tecniche sobrie.
- **Conteggio Token Reale (`janusc tokens`)**: Integrazione con `tiktoken` (cl100k_base) con fallback etichettato esplicitamente come stima euristica regex.
- **Integrazione CI**: Aggiornato il workflow GitHub Actions per testare su Python 3.10, 3.11, 3.12, 3.13 con extra `pip install -e ".[test,benchmarks]"`.

## [0.1.0] - Iniziale
- Rilascio iniziale prototipo con parser LL(1), grammatiche GBNF sperimentali e transpiler PyTorch.
