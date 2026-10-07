# Quickstart: Chiamata di Tool con llama.cpp e JANUS

Questa guida illustra un flusso di lavoro end-to-end da zero:
dalla definizione di uno schema con effect system in JANUS, alla generazione della grammatica GBNF, fino all'invocazione di un modello locale con `llama-server` e all'esecuzione sicura del tool.

---

## 1. Prerequisiti

### 1.1 Ambiente JANUS
Assicurati di trovarti nel virtual environment con JANUS installato:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verifica l'installazione:
```bash
janusc --help
```

### 1.2 llama.cpp Server
Avvia un'istanza di `llama-server` con un modello quantizzato (ad es. Llama 3 o Qwen 2.5):
```bash
llama-server \
  -m ./models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
  --port 8080 \
  --ctx-size 4096
```

---

## 2. Passo 1: Definire lo Schema e gli Effetti in JANUS

Crea un file chiamato `agent_tools.jn`:

```janus
# agent_tools.jn - Definizione dei tool e vincoli di purezza

schema WeatherLookup [io] {
    city: str,
    days: i32 = 1
} -> {
    forecast: str,
    temp_c: f32
}

schema UnitConverter [pure] {
    value: f32,
    from_unit: str,
    to_unit: str
} -> {
    converted_value: f32
}

fn query_weather_flow(target_city: str) -> str io {
    weather = call tool WeatherLookup(city = target_city, days = 3)
    ret weather
}
```

### 2.1 Verifica Statica di Tipi ed Effetti
Verifica che non vi siano violazioni di affinità o di purezza:
```bash
janusc check agent_tools.jn
```

Output atteso:
```text
Controllo semantico superato con successo: 0 errori (agent_tools.jn)
```

---

## 3. Passo 2: Generare la Grammatica GBNF

Puoi generare la grammatica GBNF per ciascuno schema usando la CLI:
```bash
janusc gbnf agent_tools.jn > weather.gbnf
```

Oppure generare la grammatica multi-tool con dispatch JSON direttamente in Python:
```python
from janus.lexer import Lexer
from janus.parser import Parser
from janus.gbnf_gen import GBNFGenerator
from janus.ast_nodes import SchemaDecl

with open("agent_tools.jn") as f:
    ast = Parser(Lexer(f.read()).tokenize()).parse()

schemas = [d for d in ast.declarations if isinstance(d, SchemaDecl)]
generator = GBNFGenerator()
multi_tool_gbnf = generator.generate_tool_json_call_grammar(schemas)

with open("tools_dispatch.gbnf", "w") as f:
    f.write(multi_tool_gbnf)
print("Grammatica GBNF generata con successo in tools_dispatch.gbnf")
```

---

## 4. Passo 3: Invocazione del Modello Locale con Decodifica Vincolata

Invia una richiesta HTTP POST all'endpoint `/completion` di `llama-server`, passando la grammatica GBNF nel parametro `grammar`.

```bash
# Leggi la grammatica generata
GBNF_CONTENT=$(cat tools_dispatch.gbnf)

# Invia la richiesta con decodifica vincolata a llama-server
curl -s -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"User: Che tempo fa a Milano per i prossimi 3 giorni?\\nAssistant: Invocherò il tool appropriato per questa richiesta:\\n\",
    \"temperature\": 0.2,
    \"n_predict\": 128,
    \"grammar\": $(jq -Rs . < tools_dispatch.gbnf)
  }" | jq .content
```

### Output Atteso dal Modello
Grazie alla decodifica vincolata da GBNF, l'output soddisfa al 100% la sintassi JSON e lo schema tipizzato:
```json
"{\"tool\": \"WeatherLookup\", \"args\": {\"city\": \"Milano\", \"days\": 3}}"
```

---

## 5. Passo 4: Esecuzione Sicura con `ToolSandbox` e `strict_effects`

Ora ricevi la stringa generata dal modello ed eseguila in modo sicuro tramite la sandbox di JANUS:

```python
import json
from janus.agent_runtime import ToolSandbox, StrictPurityViolationError

# Inizializza la sandbox con la difesa runtime attiva
sandbox = ToolSandbox(strict_effects=True)

# Registra le implementazioni reali dei tool associando il rispettivo effetto
@sandbox.tool("WeatherLookup", effect="io")
def weather_impl(city: str, days: int = 1):
    return {"forecast": f"Sereno a {city}", "temp_c": 18.5, "days_requested": days}

@sandbox.tool("UnitConverter", effect="pure")
def converter_impl(value: float, from_unit: str, to_unit: str):
    if from_unit == "C" and to_unit == "F":
        return {"converted_value": (value * 9/5) + 32}
    return {"converted_value": value}

# Risposta ricevuta da llama.cpp
model_output = '{"tool": "WeatherLookup", "args": {"city": "Milano", "days": 3}}'
call_spec = json.loads(model_output)

# Esegui il tool
result = sandbox.call_tool(call_spec["tool"], **call_spec["args"])
print("Risultato esecuzione:", result)

# Ispeziona l'audit trace
trace = sandbox.get_traces()[-1]
print(f"Audit Trace: Tool={trace.tool_name}, Durata={trace.duration_ms:.2f}ms, Successo={trace.success}")
```

### Output a Schermo
```text
Risultato esecuzione: {'forecast': 'Sereno a Milano', 'temp_c': 18.5, 'days_requested': 3}
Audit Trace: Tool=WeatherLookup, Durata=0.08ms, Successo=True
```

---

## 6. Esecuzione del Benchmark Completo con Modello Reale

Per valutare il modello locale su tutti i 20 task agentici del benchmark (confrontando Unconstrained JSON vs Native JSON Schema vs JANUS GBNF):

```bash
python benchmarks/agent_eval.py \
  --backend llama-cpp \
  --endpoint-url http://localhost:8080 \
  --model-name "Meta-Llama-3-8B-Instruct" \
  --seeds 42 123 999 \
  --temperature 0.2
```

I risultati statistici aggregati (validità sintattica, aderenza allo schema, accuratezza dei tipi, scelta del tool e argomenti corretti) verranno salvati in `benchmarks/results/real_agent_eval_<timestamp>.json`.
