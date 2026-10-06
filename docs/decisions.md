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
