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
