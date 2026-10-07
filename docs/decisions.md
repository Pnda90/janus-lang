# Registro delle Decisioni Architetturali (ADR) - JANUS

## ADR 001: Sintassi Esplicita per i Casi Morfologici (`name:case`)

### Data
2026-10-06

### Contesto
Nella versione iniziale del prototipo JANUS, i ruoli morfologici venivano estratti tramite euristiche di troncamento lessicale sui suffissi delle parole (`xm`, `wb`, `bb`, `ab`, `gb`).
Questo approccio ha generato il bug critico B1:
- Nomi terminanti naturalmente con `n, m, b, t, s, v` perdevano l'ultima lettera (`ab -> a`, `bb -> b`, `gb -> g`).
- Identificatori distinti collassavano sullo stesso nome generato in Python (`compute(ab, a) -> compute(a, a)`).
- Assegnazioni tra variabili simili (`xm = x + 1.0`) diventavano auto-sovrascritture (`x = x + 1.0`).

### Alternative Considerate
1. **Suffissi saldati con lista chiusa di parole riservate:**
   - *Contro:* Fragile, limita la scelta dei nomi da parte dell'utente o dell'LLM, non generalizzabile.
2. **Prefissi a sigillo (es. `$m_x`, `@m x`):**
   - *Contro:* Gonfia i token BPE nei modelli di linguaggio moderni e riduce l'ergonomia.
3. **Separatore esplicito due punti (`ident:case`, es. `x:m`, `w:b`, `b:b`):**
   - *Pro:* Non ambiguo; distingue in modo assoluto un identificatore ordinario (`ab`) da uno con caso grammaticale (`a:b`); LL(1) deterministico; naturale per gli LLM (analogo a type annotation).

### Decisione
Adottare la sintassi con separatore esplicito **`ident:case`** dove `case` appartiene all'insieme chiuso `{m, b, t, n, s, v}`:
- `-m` (Accusativo): input/operandi
- `-b` (Ablativo): pesi/parametri
- `-t` (Dativo): buffer/mutazione in-place
- `-n` (Nominativo): definizione SSA
- `-s` (Genitivo): forme/tipi/dimensioni
- `-v` (Vocativo): agenti/invocazioni

Gli identificatori ordinari senza `:case` (es. `ab`, `bb`, `gb`, `epochs`, `lr`, `loss`) mantengono esattamente il proprio nome senza alcuna alterazione lessicale.
In Python, l'identificatore generato corrisponde a `base_name` (es. `x:m -> x`). Il Type Checker verifica l'unicità dei `base_name` nello scope ed impedisce la ridefinizione accidentale dei parametri.

## ADR 002: Autodiff Runtime Interop e Gestione Parametri/Strutture PyTorch

### Data
2026-10-06

### Contesto
Nel prototipo iniziale (Bug B3), `examples/01_linreg.jn` generava `w = 0.0` e `b = 0.0` come primitivi `float` Python e poi invocava direttamente `torch.autograd.grad(loss, (w, b))`. PyTorch falliva a runtime richiedendo tensori con `requires_grad=True`. Inoltre, strutture/dataclass (es. `MLP`, `ModelWeights`) non potevano essere differenziate direttamente da `torch.autograd.grad` né sottratte (`mlp - lr * g`).

### Alternative Considerate
1. **torch.func.grad (approccio funzionale puro JAX-style):**
   - *Pro:* Eleganza funzionale.
   - *Contro:* Richiede che ogni modello e funzione sia scritta in forma puramente priva di stato e compatibile con vmap/vjp; fallisce se si usano chiamate native o mutazioni di stato in cicli for.
2. **torch.autograd.grad con Runtime Helper unificato e JanusStruct:**
   - *Pro:* Compatibilità universale con PyTorch 2.x standard; le strutture dati generate ereditano da `JanusStruct` che implementa l'aritmetica tensoriale vettorializzata (`__sub__`, `__mul__`, `__add__`); `_janus_diff` supporta scalari, tensori, tuple e struct composte ricostruendo le classi con i gradienti; i parametri mutabili vengono aggiornati out-of-place con detach e reinserimento in grafo per il ciclo successivo.

### Decisione
Adottare `torch.autograd.grad` potenziato dal runtime JANUS:
- `_janus_param(v)` promuove letterali scalari o tensori a parametri `requires_grad=True`.
- `_janus_to_tensor(x)` converte input (`:m`) da liste Python o array in tensori PyTorch.
- `_janus_diff(loss, wrt)` gestisce differenziazione rispetto a singoli tensori, tuple di tensori o istanze `JanusStruct`.
- `_janus_step_val(v)` esegue il detach delle foglie tensoriali dopo l'aggiornamento SGD mantenendo attivo il gradiente per le epoche successive.
- `_janus_result(v)` converte in modo trasparente tensori scalari in numeri Python (`float`) o ritorna tensori/tuple.
- I tipi composti (`type MLP`) generano classi che ereditano da `JanusStruct`.

## ADR 003: Eliminazione Script di Benchmark Legacy e Tabelle Hardcoded

### Data
2026-10-06

### Contesto
I vecchi file di benchmark nel repository (`bench_10_programs.py`, `test_hypotheses.py`, `test_15_snippets.py`, ecc.) presentavano gravi anomalie metodologiche (Bug B5):
- Contenevano tabelle di token hardcoded (`reference_data` in `test_hypotheses.py`).
- Impiegavano conteggi basati su espressioni regolari euristiche anziché tokenizer BPE ufficiali.
- Utilizzavano snippet asimmetrici tra le varianti linguistiche.
- Tali script alimentavano l'affermazione ingannevole nel README di una riduzione del 50% dei token.

### Alternative Considerate
1. **Conservare i vecchi script in una cartella `benchmarks/legacy/`:**
   - *Contro:* Genera ambiguità per gli utenti e per la CI; rischia di far persistere file con numeri inventati nel repository.
2. **Riscrivere da zero due strumenti dedicati e rimuovere i file obsoleti:**
   - *Pro:* `benchmarks/tokens.py` usa `tiktoken` (`cl100k_base`, `o200k_base`) e `transformers` direttamente sui 10 programmi del repository a parità semantica; `benchmarks/llm_eval.py` implementa l'harness di valutazione su 30 compiti con calcolo `pass@k` e ciclo di riparazione via diagnostica; nessun numero è hardcoded; la cronologia Git preserva integralmente i file storici per futura consultazione.

### Decisione
Rimuovere i 10 script obsoleti e stabilire `benchmarks/tokens.py` e `benchmarks/llm_eval.py` come unici punti di riferimento per la misurazione empirica, tracciando i risultati grezzi esclusivamente in `benchmarks/results/*.json`. In CI, il test `tests/test_no_hardcoded_benchmarks.py` verifica in modo continuativo l'assenza di pattern di dati inventati o hardcoded.

## ADR 004: Pivot Architetturale verso Agentic Execution Graph DSL e Safe Tool Orchestration

### Data
2026-10-06

### Contesto
I risultati empirici delle Fasi 0–5 hanno dimostrato che:
1. Come alternativa generica a PyTorch per calcolo tensoriale puro, un transpiler a Python è penalizzato dalla tokenizzazione BPE (+52% di token rispetto a Python) e non offre vantaggi di velocità d'esecuzione a runtime.
2. I veri differenziatori ingegneristici di JANUS sono:
   - La grammatica deterministica **LL(1)** che abilita la **decodifica vincolata (GBNF)** a zero errori sintattici su motori di inferenza locali (*llama.cpp*, *Ollama*, *vLLM*).
   - La **diagnostica JSON-first** con coordinate precise e patch correttive che velocizza l'autoriparazione degli agenti.
   - L'algebra degli **effetti monadici** (`pure`, `io`, `stoc`) che garantisce la separazione formale tra calcolo deterministico e chiamate con effetti collaterali (I/O, tool di rete).
3. Nel settore degli agenti AI (LangChain, CrewAI, OpenAI tool-calling), i problemi critici aperti sono: allucinazioni di schemi JSON, chiamate di tool con parametri e tipi incompatibili, assenza di controllo sui side-effect e rischi di sicurezza da esecuzione di codice Python arbitrario.

### Alternative Considerate
1. **Mantenere il focus su calcolo tensoriale puro contro PyTorch:**
   - *Contro:* Nessun vantaggio competitivo reale rispetto all'ecosistema Python nativo; gli sviluppatori non abbandoneranno PyTorch per una sintassi proprietaria che compila comunque in PyTorch.
2. **Riscrivere l'intero compilatore con backend nativo MLIR/C/WASM:**
   - *Pro:* Indipendenza dal runtime Python.
   - *Contro:* Sforzo ingegneristico sproporzionato (anni-uomo) che non risolve il problema primario degli agenti AI contemporanei.
3. **Pivot verso Agentic Execution Graph DSL (Road A):**
   - *Pro:* Capitalizza interamente l'infrastruttura già creata (parser LL(1), validatore GBNF, type checker, runtime e test suite). Risolve un problema non risolto da Python: fornire a un agente un linguaggio formale sicuro, con tipi ed effetti verificati a monte e decodifica vincolata per conformità sintattica e di tipo.

### Decisione
Adottare l'Opzione 3 ("Strada A"):
- JANUS evolve in un **Agentic Execution Graph DSL con Effect System & Constrained Decoding**.
- `schema` diventa il costrutto primario per tipizzare formalmente Tool e API (input, output, permessi ed effetti `pure`, `io`, `stoc`).
- Il parser supporta la chiamata esplicita `call tool ToolName(key = val, ...)`.
- Il type checker segnala `ERR_UNDEFINED_TOOL`, `ERR_EFFECT_PURITY_VIOLATION`, `ERR_MISSING_TOOL_ARGUMENT` ed `ERR_UNKNOWN_TOOL_ARGUMENT`.
- Il generatore GBNF (`generate_agent_grammar`) sintetizza grammatiche multi-tool per vincolare piani agenziali e chiamate di tool con token masking su llama.cpp/vLLM.
- Introdotto `ToolSandbox` e `AgentRuntime` in `janus/agent_runtime.py` con audit trail granulare (`ToolTrace`), validazione a runtime e modalità dry-run.
- Dimostrato sperimentalmente con `benchmarks/agent_eval.py` che la decodifica vincolata GBNF da schemi JANUS garantisce il 100% di conformità sintattica e di tipo eliminando gli errori di schema.

### Conseguenze
- **Positive:** JANUS possiede una value proposition unica, misurabile e difendibile: la generazione di grafi di tool execution con conformità sintattica e di tipo garantita da token masking e confini formali di purezza per agenti autonomi.
- **Transizione:** La compatibilità con i programmi tensoriali esistenti (esempi 01-10) è mantenuta al 100%; la suite di test sale a 87 test verdi con 0 regressioni.


