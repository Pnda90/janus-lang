#!/usr/bin/env python3
"""
benchmarks/agent_eval.py

Harness di valutazione per Agentic Tool Calling:
Confronta la decodifica non vincolata (Unconstrained JSON Tool Calling),
il JSON Schema nativo con decodifica vincolata (structured outputs / json_schema),
e le grammatiche formali GBNF sintetizzate da JANUS.

Supporta:
- Backend reale con endpoint HTTP: --backend llama-cpp|ollama|vllm
- Backend simulato deterministico: --backend mock (o --dry-run)
- Configurazione dinamica URL ed endpoint via CLI o variabili d'ambiente (JANUS_BENCH_URL, JANUS_BENCH_MODEL)
- N seed e temperature configurabili per calcolo di media e deviazione standard
- Metriche: Validità Sintattica, Conformità Schema, Accuratezza Tipi, Scelta Tool,
  Correttezza Semantica Argomenti (rispetto a ground truth), Token Consumati.
"""

import os
import sys
import json
import time
import math
import argparse
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(ENC.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return len(text.split())

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from janus.lexer import Lexer
from janus.parser import Parser
from janus.type_checker import TypeChecker
from janus.gbnf_gen import GBNFGenerator
from janus.gbnf_validator import GBNFValidator
from janus.jsonschema_export import schema_to_json_schema
from janus.agent_runtime import AgentRuntime, ToolSandbox
from tests.support.gbnf_engine import GBNFSampler


# ---------------------------------------------------------------------------
# 1. 20 TASK REALISTICI DI TOOL CALLING PER AGENTI CON GROUND TRUTH
# ---------------------------------------------------------------------------

AGENT_TASKS: List[Dict[str, Any]] = [
    {
        "id": "tool_01",
        "name": "Web Search Query",
        "schema_code": "schema WebSearch io { query: str, max_results: i32 = 10 } -> { results: str }",
        "tool_name": "WebSearch",
        "expected_tool": "WebSearch",
        "required_args": {"query": str},
        "optional_args": {"max_results": int},
        "prompt": "Search the web for the latest PyTorch release notes, returning up to 5 results.",
        "sample_valid_input": {"query": "PyTorch release notes", "max_results": 5},
        "ground_truth_args": {"query": "PyTorch release notes", "max_results": 5},
    },
    {
        "id": "tool_02",
        "name": "Database Record Fetch",
        "schema_code": "schema DbFetch io { table: str, record_id: i32 } -> { data: str }",
        "tool_name": "DbFetch",
        "expected_tool": "DbFetch",
        "required_args": {"table": str, "record_id": int},
        "optional_args": {},
        "prompt": "Fetch user record with ID 1042 from the 'users' table.",
        "sample_valid_input": {"table": "users", "record_id": 1042},
        "ground_truth_args": {"table": "users", "record_id": 1042},
    },
    {
        "id": "tool_03",
        "name": "Safe Math Calculation",
        "schema_code": "schema SafeCalc pure { expr: str, precision: i32 = 4 } -> { result: f32 }",
        "tool_name": "SafeCalc",
        "expected_tool": "SafeCalc",
        "required_args": {"expr": str},
        "optional_args": {"precision": int},
        "prompt": "Evaluate the arithmetic expression '3.14159 * 2.0^2' with precision 4.",
        "sample_valid_input": {"expr": "3.14159 * 2.0^2", "precision": 4},
        "ground_truth_args": {"expr": "3.14159 * 2.0^2", "precision": 4},
    },
    {
        "id": "tool_04",
        "name": "Send Email Notification",
        "schema_code": "schema SendEmail io { recipient: str, subject: str, body: str } -> { sent: bool }",
        "tool_name": "SendEmail",
        "expected_tool": "SendEmail",
        "required_args": {"recipient": str, "subject": str, "body": str},
        "optional_args": {},
        "prompt": "Send an alert email to 'ops@company.org' with subject 'High Memory' and body 'RAM > 90%'.",
        "sample_valid_input": {"recipient": "ops@company.org", "subject": "High Memory", "body": "RAM > 90%"},
        "ground_truth_args": {"recipient": "ops@company.org", "subject": "High Memory", "body": "RAM > 90%"},
    },
    {
        "id": "tool_05",
        "name": "File Content Reader",
        "schema_code": "schema ReadFile io { filepath: str, max_bytes: i32 = 4096 } -> { content: str }",
        "tool_name": "ReadFile",
        "expected_tool": "ReadFile",
        "required_args": {"filepath": str},
        "optional_args": {"max_bytes": int},
        "prompt": "Read the configuration file from '/etc/hosts' up to 1024 bytes.",
        "sample_valid_input": {"filepath": "/etc/hosts", "max_bytes": 1024},
        "ground_truth_args": {"filepath": "/etc/hosts", "max_bytes": 1024},
    },
    {
        "id": "tool_06",
        "name": "File Content Writer",
        "schema_code": "schema WriteFile io { filepath: str, content: str, append: bool = false } -> { bytes_written: i32 }",
        "tool_name": "WriteFile",
        "expected_tool": "WriteFile",
        "required_args": {"filepath": str, "content": str},
        "optional_args": {"append": bool},
        "prompt": "Write 'BUILD SUCCESS' to file '/var/log/build.log' in append mode.",
        "sample_valid_input": {"filepath": "/var/log/build.log", "content": "BUILD SUCCESS", "append": True},
        "ground_truth_args": {"filepath": "/var/log/build.log", "content": "BUILD SUCCESS", "append": True},
    },
    {
        "id": "tool_07",
        "name": "Stock Price Ticker",
        "schema_code": "schema StockTicker io { symbol: str, currency: str = 'USD' } -> { price: f32 }",
        "tool_name": "StockTicker",
        "expected_tool": "StockTicker",
        "required_args": {"symbol": str},
        "optional_args": {"currency": str},
        "prompt": "Get current price for ticker 'NVDA' in USD.",
        "sample_valid_input": {"symbol": "NVDA", "currency": "USD"},
        "ground_truth_args": {"symbol": "NVDA", "currency": "USD"},
    },
    {
        "id": "tool_08",
        "name": "Text Summarizer",
        "schema_code": "schema SummarizeText pure { text: str, max_words: i32 = 100 } -> { summary: str }",
        "tool_name": "SummarizeText",
        "expected_tool": "SummarizeText",
        "required_args": {"text": str},
        "optional_args": {"max_words": int},
        "prompt": "Summarize the quarterly financial earnings report in under 50 words.",
        "sample_valid_input": {"text": "Q3 earnings rose 15% due to robust AI chip demand...", "max_words": 50},
        "ground_truth_args": {"text": "Q3 earnings rose 15% due to robust AI chip demand...", "max_words": 50},
    },
    {
        "id": "tool_09",
        "name": "HTTP Webhook Post",
        "schema_code": "schema HttpPost io { url: str, payload_json: str, timeout_sec: i32 = 30 } -> { status_code: i32 }",
        "tool_name": "HttpPost",
        "expected_tool": "HttpPost",
        "required_args": {"url": str, "payload_json": str},
        "optional_args": {"timeout_sec": int},
        "prompt": "Send POST request to 'https://api.gateway.io/v1/notify' with payload '{\"event\": \"deploy\"}'.",
        "sample_valid_input": {"url": "https://api.gateway.io/v1/notify", "payload_json": "{\"event\": \"deploy\"}", "timeout_sec": 10},
        "ground_truth_args": {"url": "https://api.gateway.io/v1/notify", "payload_json": "{\"event\": \"deploy\"}", "timeout_sec": 10},
    },
    {
        "id": "tool_10",
        "name": "Vector Embedding Search",
        "schema_code": "schema VectorSearch io { collection: str, vector_dim: i32, top_k: i32 = 10 } -> { doc_ids: str }",
        "tool_name": "VectorSearch",
        "expected_tool": "VectorSearch",
        "required_args": {"collection": str, "vector_dim": int},
        "optional_args": {"top_k": int},
        "prompt": "Search vectors in collection 'kb_articles' with dimension 1536 returning top 3 matches.",
        "sample_valid_input": {"collection": "kb_articles", "vector_dim": 1536, "top_k": 3},
        "ground_truth_args": {"collection": "kb_articles", "vector_dim": 1536, "top_k": 3},
    },
    {
        "id": "tool_11",
        "name": "Git Branch Checkout",
        "schema_code": "schema GitCheckout io { branch: str, create_new: bool = false } -> { success: bool }",
        "tool_name": "GitCheckout",
        "expected_tool": "GitCheckout",
        "required_args": {"branch": str},
        "optional_args": {"create_new": bool},
        "prompt": "Create and check out a new git branch named 'feature/agent-dsl'.",
        "sample_valid_input": {"branch": "feature/agent-dsl", "create_new": True},
        "ground_truth_args": {"branch": "feature/agent-dsl", "create_new": True},
    },
    {
        "id": "tool_12",
        "name": "Docker Container Run",
        "schema_code": "schema ContainerRun io { image: str, port_map: i32, detached: bool = true } -> { container_id: str }",
        "tool_name": "ContainerRun",
        "expected_tool": "ContainerRun",
        "required_args": {"image": str, "port_map": int},
        "optional_args": {"detached": bool},
        "prompt": "Run docker image 'redis:7-alpine' mapping port 6379 in detached mode.",
        "sample_valid_input": {"image": "redis:7-alpine", "port_map": 6379, "detached": True},
        "ground_truth_args": {"image": "redis:7-alpine", "port_map": 6379, "detached": True},
    },
    {
        "id": "tool_13",
        "name": "Regex Pattern Match",
        "schema_code": "schema RegexMatch pure { pattern: str, text: str } -> { matched: bool }",
        "tool_name": "RegexMatch",
        "expected_tool": "RegexMatch",
        "required_args": {"pattern": str, "text": str},
        "optional_args": {},
        "prompt": "Test if string 'user@example.com' matches email pattern '^[\\w.-]+@[\\w.-]+$'.",
        "sample_valid_input": {"pattern": r"^[\w.-]+@[\w.-]+$", "text": "user@example.com"},
        "ground_truth_args": {"pattern": r"^[\w.-]+@[\w.-]+$", "text": "user@example.com"},
    },
    {
        "id": "tool_14",
        "name": "JSON Parser Validator",
        "schema_code": "schema ParseJson pure { json_str: str, strict: bool = true } -> { is_valid: bool }",
        "tool_name": "ParseJson",
        "expected_tool": "ParseJson",
        "required_args": {"json_str": str},
        "optional_args": {"strict": bool},
        "prompt": "Parse and validate the JSON string '{\"id\": 1, \"active\": true}'.",
        "sample_valid_input": {"json_str": '{"id": 1, "active": true}', "strict": True},
        "ground_truth_args": {"json_str": '{"id": 1, "active": true}', "strict": True},
    },
    {
        "id": "tool_15",
        "name": "Slack Message Post",
        "schema_code": "schema SlackNotify io { channel: str, message: str } -> { timestamp: str }",
        "tool_name": "SlackNotify",
        "expected_tool": "SlackNotify",
        "required_args": {"channel": str, "message": str},
        "optional_args": {},
        "prompt": "Post message 'Model checkpoint saved: ep=100' to Slack channel '#ml-training'.",
        "sample_valid_input": {"channel": "#ml-training", "message": "Model checkpoint saved: ep=100"},
        "ground_truth_args": {"channel": "#ml-training", "message": "Model checkpoint saved: ep=100"},
    },
    {
        "id": "tool_16",
        "name": "Hash Generator",
        "schema_code": "schema GenerateHash pure { data: str, algorithm: str = 'sha256' } -> { digest: str }",
        "tool_name": "GenerateHash",
        "expected_tool": "GenerateHash",
        "required_args": {"data": str},
        "optional_args": {"algorithm": str},
        "prompt": "Compute SHA256 digest of input payload string 'janus-runtime-2026'.",
        "sample_valid_input": {"data": "janus-runtime-2026", "algorithm": "sha256"},
        "ground_truth_args": {"data": "janus-runtime-2026", "algorithm": "sha256"},
    },
    {
        "id": "tool_17",
        "name": "Audio Transcription",
        "schema_code": "schema TranscribeAudio io { file_uri: str, language: str = 'en' } -> { transcript: str }",
        "tool_name": "TranscribeAudio",
        "expected_tool": "TranscribeAudio",
        "required_args": {"file_uri": str},
        "optional_args": {"language": str},
        "prompt": "Transcribe the audio recording at 's3://media/meeting_01.mp3' in English.",
        "sample_valid_input": {"file_uri": "s3://media/meeting_01.mp3", "language": "en"},
        "ground_truth_args": {"file_uri": "s3://media/meeting_01.mp3", "language": "en"},
    },
    {
        "id": "tool_18",
        "name": "Currency Exchange Converter",
        "schema_code": "schema ConvertCurrency pure { amount: f32, from_curr: str, to_curr: str } -> { converted: f32 }",
        "tool_name": "ConvertCurrency",
        "expected_tool": "ConvertCurrency",
        "required_args": {"amount": float, "from_curr": str, "to_curr": str},
        "optional_args": {},
        "prompt": "Convert 150.50 EUR to USD currency.",
        "sample_valid_input": {"amount": 150.5, "from_curr": "EUR", "to_curr": "USD"},
        "ground_truth_args": {"amount": 150.5, "from_curr": "EUR", "to_curr": "USD"},
    },
    {
        "id": "tool_19",
        "name": "S3 Bucket Object Upload",
        "schema_code": "schema S3Upload io { bucket: str, key: str, data: str } -> { etag: str }",
        "tool_name": "S3Upload",
        "expected_tool": "S3Upload",
        "required_args": {"bucket": str, "key": str, "data": str},
        "optional_args": {},
        "prompt": "Upload report data 'Sales: $100k' to bucket 'corporate-data' at key 'reports/q3.txt'.",
        "sample_valid_input": {"bucket": "corporate-data", "key": "reports/q3.txt", "data": "Sales: $100k"},
        "ground_truth_args": {"bucket": "corporate-data", "key": "reports/q3.txt", "data": "Sales: $100k"},
    },
    {
        "id": "tool_20",
        "name": "Process Terminator",
        "schema_code": "schema KillProcess io { pid: i32, force: bool = false } -> { terminated: bool }",
        "tool_name": "KillProcess",
        "expected_tool": "KillProcess",
        "required_args": {"pid": int},
        "optional_args": {"force": bool},
        "prompt": "Terminate hung background process PID 8421 with force=true.",
        "sample_valid_input": {"pid": 8421, "force": True},
        "ground_truth_args": {"pid": 8421, "force": True},
    },
]


# ---------------------------------------------------------------------------
# 2. GENERATORI DI SCHEMI E PROMPT
# ---------------------------------------------------------------------------

def task_to_json_schema(task: Dict[str, Any]) -> Dict[str, Any]:
    """Genera JSON Schema standard per decodifica vincolata nativa."""
    properties = {}
    required = []
    for k, t in task["required_args"].items():
        if t is int:
            properties[k] = {"type": "integer"}
        elif t is float:
            properties[k] = {"type": "number"}
        elif t is bool:
            properties[k] = {"type": "boolean"}
        else:
            properties[k] = {"type": "string"}
        required.append(k)

    for k, t in task["optional_args"].items():
        if t is int:
            properties[k] = {"type": "integer"}
        elif t is float:
            properties[k] = {"type": "number"}
        elif t is bool:
            properties[k] = {"type": "boolean"}
        else:
            properties[k] = {"type": "string"}

    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": [task["tool_name"]]},
            "args": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def build_task_prompt(task: Dict[str, Any]) -> str:
    """Costruisce il prompt uniforme per il modello."""
    return (
        "You are an AI assistant orchestrating external tools. Call the appropriate tool with exact arguments.\n"
        "Output ONLY a valid JSON object with the format: {\"tool\": \"<tool_name>\", \"args\": {<arguments>}}\n\n"
        f"Available Tool Schema:\n{task['schema_code']}\n\n"
        f"User Instruction:\n{task['prompt']}\n"
    )


# ---------------------------------------------------------------------------
# 3. VALIDAZIONE MULTI-CRITERIO DELLE RISPOSTE
# ---------------------------------------------------------------------------

def check_semantic_args(
    extracted_args: Dict[str, Any],
    ground_truth: Dict[str, Any],
    required_args: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Confronta gli argomenti estratti con la ground truth del task."""
    mismatches = []
    for k, expected_v in ground_truth.items():
        if k in required_args and k not in extracted_args:
            mismatches.append(f"Missing required arg '{k}'")
            continue
        if k not in extracted_args:
            continue
        actual_v = extracted_args[k]
        if isinstance(expected_v, str):
            if isinstance(actual_v, str):
                e_clean = expected_v.strip().lower()
                a_clean = actual_v.strip().lower()
                if e_clean != a_clean and e_clean not in a_clean and a_clean not in e_clean:
                    mismatches.append(f"Arg '{k}': expected '{expected_v}', got '{actual_v}'")
            else:
                mismatches.append(f"Arg '{k}': expected str, got {type(actual_v).__name__}")
        elif isinstance(expected_v, float):
            try:
                if abs(float(actual_v) - expected_v) > 1e-3:
                    mismatches.append(f"Arg '{k}': expected {expected_v}, got {actual_v}")
            except Exception:
                mismatches.append(f"Arg '{k}': numeric conversion failed for {actual_v}")
        elif isinstance(expected_v, int) and not isinstance(expected_v, bool):
            try:
                if int(actual_v) != expected_v:
                    mismatches.append(f"Arg '{k}': expected {expected_v}, got {actual_v}")
            except Exception:
                mismatches.append(f"Arg '{k}': int conversion failed for {actual_v}")
        elif isinstance(expected_v, bool):
            if bool(actual_v) != expected_v:
                mismatches.append(f"Arg '{k}': expected {expected_v}, got {actual_v}")
        else:
            if actual_v != expected_v:
                mismatches.append(f"Arg '{k}': expected {expected_v}, got {actual_v}")
    return (len(mismatches) == 0, mismatches)


def validate_tool_payload(payload_str: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analizza e verifica un payload di tool call.
    Controlla:
    1. Sintassi JSON valida
    2. Scelta del tool corretto
    3. Conformità dello schema (campi obbligatori presenti, no campi allucinati)
    4. Tipizzazione rigorosa
    5. Correttezza semantica degli argomenti vs ground truth
    """
    res = {
        "syntax_valid": False,
        "schema_conformance": False,
        "type_conformance": False,
        "tool_choice_correct": False,
        "semantic_args_correct": False,
        "perfect_call": False,
        "errors": []
    }

    try:
        data = json.loads(payload_str.strip())
        res["syntax_valid"] = True
    except Exception as e:
        res["errors"].append(f"JSON Syntax Error: {e}")
        return res

    if not isinstance(data, dict):
        res["errors"].append(f"Payload is not a JSON object: got {type(data).__name__}")
        return res

    # Estrazione tool_name
    expected_tool = task.get("expected_tool", task.get("tool_name"))
    has_tool_field = any(k in data for k in ("tool", "name", "tool_name"))
    if has_tool_field:
        chosen_tool = data.get("tool") or data.get("name") or data.get("tool_name")
        res["tool_choice_correct"] = (chosen_tool == expected_tool)
        if not res["tool_choice_correct"]:
            res["errors"].append(f"Wrong tool chosen: expected '{expected_tool}', got '{chosen_tool}'")
    else:
        # Se payload è piatto, si assume implicitamente il tool atteso
        res["tool_choice_correct"] = True

    # Estrazione argomenti
    if "args" in data and isinstance(data["args"], dict):
        args = data["args"]
    elif "arguments" in data:
        raw_args = data["arguments"]
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
    else:
        args = {k: v for k, v in data.items() if k not in ("tool", "name", "tool_name")}

    # 1. Campi obbligatori
    req_missing = [k for k in task["required_args"] if k not in args]
    if req_missing:
        res["errors"].append(f"Missing required fields: {req_missing}")

    # 2. Campi allucinati
    allowed_fields = set(task["required_args"].keys()).union(set(task["optional_args"].keys()))
    hallucinated = [k for k in args if k not in allowed_fields]
    if hallucinated:
        res["errors"].append(f"Hallucinated unknown fields: {hallucinated}")

    if not req_missing and not hallucinated:
        res["schema_conformance"] = True

    # 3. Validazione tipi
    all_specs = {**task["required_args"], **task["optional_args"]}
    type_errors = []
    for k, v in args.items():
        if k in all_specs:
            expected_type = all_specs[k]
            if expected_type is float and isinstance(v, (int, float)):
                continue
            if not isinstance(v, expected_type):
                type_errors.append(f"Field '{k}': expected {expected_type.__name__}, got {type(v).__name__}")

    if not type_errors and res["schema_conformance"]:
        res["type_conformance"] = True
    else:
        res["errors"].extend(type_errors)

    # 4. Validazione semantica vs Ground Truth
    gt = task.get("ground_truth_args", task.get("sample_valid_input", {}))
    sem_ok, sem_errs = check_semantic_args(args, gt, task["required_args"])
    res["semantic_args_correct"] = sem_ok
    if not sem_ok:
        res["errors"].extend(sem_errs)

    # 5. Perfect Call
    res["perfect_call"] = (
        res["syntax_valid"] and
        res["schema_conformance"] and
        res["type_conformance"] and
        res["tool_choice_correct"] and
        res["semantic_args_correct"]
    )

    return res


# ---------------------------------------------------------------------------
# 4. BACKEND RUNTIME (MOCK / LLAMA-CPP / OLLAMA / VLLM)
# ---------------------------------------------------------------------------

def simulate_mock_response(
    condition: str,
    task: Dict[str, Any],
    task_idx: int,
    seed: int,
    gbnf_grammar: Optional[str] = None,
    json_schema_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int]:
    """
    Simulatore deterministico per test e sviluppo locale senza modello reale.
    Dichiarato esplicitamente come MOCK.
    - Per 'janus_gbnf': campiona stringhe da GBNFSampler e verifica la validità con jsonschema.
    - Per 'json_schema': garantisce conformità strutturale identica.
    - Per 'unconstrained': genera errori sintetici realistici (sintassi JSON invalida,
      campi extra/allucinati, type mismatch) usando un generatore pseudo-stocastico.
    """
    base = dict(task["sample_valid_input"])
    tool_name = task["tool_name"]

    if condition == "janus_gbnf":
        if gbnf_grammar:
            try:
                sampler = GBNFSampler(gbnf_grammar)
                sampled_str = sampler.sample(seed=seed * 1000 + task_idx * 17)
                parsed = json.loads(sampled_str)
                if json_schema_dict:
                    import jsonschema
                    jsonschema.validate(parsed, json_schema_dict)
            except Exception:
                # Riporta la stringa non conforme se la grammatica dovesse fallire
                return sampled_str, count_tokens(sampled_str)

        payload = {"tool": tool_name, "args": base}
        txt = json.dumps(payload)
        return txt, count_tokens(txt)

    if condition == "json_schema":
        payload = {"tool": tool_name, "args": base}
        txt = json.dumps(payload)
        return txt, count_tokens(txt)

    # Condition: unconstrained
    # Riproduce i tipici failure mode dei modelli senza vincoli tramite simulatore sintetico
    pseudo_rand = (task_idx * 17 + seed * 31) % 100

    if pseudo_rand < 5:
        # Errore sintassi JSON
        txt = json.dumps({"tool": tool_name, "args": base})[:-1] + ', "trailing": true, }'
        return txt, count_tokens(txt)
    elif pseudo_rand < 20:
        # Parametri allucinati
        corrupted = dict(base)
        corrupted["reasoning_step"] = "Calling default index"
        corrupted["tool_version"] = "2.1"
        payload = {"tool": tool_name, "args": corrupted}
        txt = json.dumps(payload)
        return txt, count_tokens(txt)
    elif pseudo_rand < 30:
        # Type mismatch
        corrupted = dict(base)
        for k, t in task["required_args"].items():
            if t is int:
                corrupted[k] = str(corrupted[k])
                break
        payload = {"tool": tool_name, "args": corrupted}
        txt = json.dumps(payload)
        return txt, count_tokens(txt)

    payload = {"tool": tool_name, "args": base}
    txt = json.dumps(payload)
    return txt, count_tokens(txt)


def call_http_backend(
    backend: str,
    url: str,
    model: str,
    prompt: str,
    condition: str,
    task: Dict[str, Any],
    gbnf_grammar: str,
    json_schema_dict: Dict[str, Any],
    temperature: float = 0.0,
    seed: int = 42,
    timeout: float = 30.0
) -> Tuple[str, int]:
    """
    Invia la richiesta al server LLM locale o remoto via HTTP.
    Supporta: llama-cpp (/completion), ollama (/api/generate), vllm (/v1/chat/completions).
    Nessuna credenziale hardcoded: parametri passati via CLI o variabili d'ambiente.
    """
    if requests is None:
        raise RuntimeError("Modulo 'requests' non disponibile nel runtime Python.")

    backend_clean = backend.lower().strip()

    if backend_clean == "llama-cpp":
        endpoint = url.rstrip('/') + ("/completion" if not url.endswith("/completion") else "")
        body: Dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
            "seed": seed,
            "n_predict": 512,
        }
        if condition == "json_schema":
            body["json_schema"] = json_schema_dict
        elif condition == "janus_gbnf":
            body["grammar"] = gbnf_grammar

        resp = requests.post(endpoint, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", "")
        tok_eval = data.get("tokens_evaluated", 0)
        tok_pred = data.get("tokens_predicted", 0)
        tokens = (tok_eval + tok_pred) if (tok_eval or tok_pred) else count_tokens(prompt + content)
        return content, tokens

    elif backend_clean == "ollama":
        endpoint = url.rstrip('/') + ("/api/generate" if not url.endswith("/api/generate") else "")
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "seed": seed},
        }
        if condition == "json_schema":
            body["format"] = json_schema_dict
        elif condition == "janus_gbnf":
            body["format"] = "json"

        resp = requests.post(endpoint, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("response", "")
        tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        if not tokens:
            tokens = count_tokens(prompt + content)
        return content, tokens

    elif backend_clean == "vllm":
        endpoint = url.rstrip('/') + ("/v1/chat/completions" if not url.endswith("/v1/chat/completions") else "")
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "seed": seed,
        }
        if condition == "json_schema":
            body["guided_json"] = json_schema_dict
        elif condition == "janus_gbnf":
            body["guided_grammar"] = gbnf_grammar

        resp = requests.post(endpoint, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens = usage.get("total_tokens") or count_tokens(prompt + content)
        return content, tokens

    else:
        raise ValueError(f"Backend HTTP non supportato: '{backend}'. Usa 'llama-cpp', 'ollama', o 'vllm'.")


# ---------------------------------------------------------------------------
# 5. ESECUZIONE BENCHMARK SU TASK E SEED MULTIPLI
# ---------------------------------------------------------------------------

def calc_stats(values: List[float]) -> Dict[str, float]:
    """Calcola media e deviazione standard campionaria/popolazione."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return {"mean": round(mean, 4), "std": round(math.sqrt(variance), 4)}


def run_benchmark(
    dry_run: bool = True,
    backend: str = "mock",
    url: Optional[str] = None,
    model: Optional[str] = None,
    conditions: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Esegue la valutazione multi-condizione sui 20 task agentici.
    Calcola metriche aggregate (media ± deviazione standard su seed multipli).
    """
    # Risoluzione backend e parametri
    is_mock = dry_run or (backend.lower() == "mock")
    active_backend = "mock" if is_mock else backend.lower().strip()

    if conditions is None:
        conditions = ["unconstrained", "json_schema", "janus_gbnf"]

    if seeds is None:
        seeds = [42]

    # Defaults per endpoint HTTP
    if not url:
        if active_backend == "llama-cpp":
            url = os.environ.get("LLAMA_CPP_URL", os.environ.get("JANUS_BENCH_URL", "http://localhost:8080"))
        elif active_backend == "ollama":
            url = os.environ.get("OLLAMA_URL", os.environ.get("JANUS_BENCH_URL", "http://localhost:11434"))
        elif active_backend == "vllm":
            url = os.environ.get("VLLM_URL", os.environ.get("JANUS_BENCH_URL", "http://localhost:8000"))
        else:
            url = "mock://internal"

    if not model:
        model = os.environ.get("JANUS_BENCH_MODEL", "mock-model" if is_mock else "default-model")

    gbnf_validator = GBNFValidator()
    gen = GBNFGenerator()

    # Pre-compilazione schemi, grammatiche GBNF e JSON Schema
    compiled_tasks = []
    for task in AGENT_TASKS:
        tokens = Lexer(task["schema_code"]).tokenize()
        ast = Parser(tokens).parse()
        schema_decl = ast.declarations[0]

        gbnf_grammar = gen.generate_tool_json_call_grammar(schema_decl)
        assert gbnf_validator.validate_grammar_syntax(gbnf_grammar), f"Grammatica non valida per {task['name']}"
        json_schema_dict = schema_to_json_schema(schema_decl, mode="call")
        prompt_text = build_task_prompt(task)

        compiled_tasks.append({
            "task": task,
            "schema_decl": schema_decl,
            "gbnf_grammar": gbnf_grammar,
            "json_schema_dict": json_schema_dict,
            "prompt": prompt_text,
        })

    # Struttura risultati per seed
    results_by_seed: Dict[int, Dict[str, Any]] = {}

    for seed in seeds:
        seed_results: Dict[str, Any] = {}
        for cond in conditions:
            seed_results[cond] = {
                "syntax_valid": 0,
                "schema_conformance": 0,
                "type_conformance": 0,
                "tool_choice": 0,
                "semantic_args": 0,
                "perfect_calls": 0,
                "total_tokens": 0,
                "details": []
            }

        for idx, item in enumerate(compiled_tasks):
            task = item["task"]
            prompt = item["prompt"]
            gbnf_grammar = item["gbnf_grammar"]
            json_schema_dict = item["json_schema_dict"]

            for cond in conditions:
                if is_mock:
                    out_text, tokens = simulate_mock_response(
                        cond, task, idx, seed, gbnf_grammar=gbnf_grammar, json_schema_dict=json_schema_dict
                    )
                else:
                    out_text, tokens = call_http_backend(
                        backend=active_backend,
                        url=url,
                        model=model,
                        prompt=prompt,
                        condition=cond,
                        task=task,
                        gbnf_grammar=gbnf_grammar,
                        json_schema_dict=json_schema_dict,
                        temperature=temperature,
                        seed=seed,
                        timeout=timeout
                    )

                v = validate_tool_payload(out_text, task)
                c_res = seed_results[cond]
                c_res["syntax_valid"] += int(v["syntax_valid"])
                c_res["schema_conformance"] += int(v["schema_conformance"])
                c_res["type_conformance"] += int(v["type_conformance"])
                c_res["tool_choice"] += int(v["tool_choice_correct"])
                c_res["semantic_args"] += int(v["semantic_args_correct"])
                c_res["perfect_calls"] += int(v["perfect_call"])
                c_res["total_tokens"] += tokens

                c_res["details"].append({
                    "task_id": task["id"],
                    "output": out_text,
                    "validation": v,
                    "perfect": v["perfect_call"],
                    "tokens": tokens,
                })

        results_by_seed[seed] = seed_results

    # Aggregazione statistica su tutti i seed (mean, std)
    n_tasks = len(compiled_tasks)
    summary_stats: Dict[str, Any] = {}

    for cond in conditions:
        syntax_rates = [results_by_seed[s][cond]["syntax_valid"] / n_tasks for s in seeds]
        schema_rates = [results_by_seed[s][cond]["schema_conformance"] / n_tasks for s in seeds]
        type_rates = [results_by_seed[s][cond]["type_conformance"] / n_tasks for s in seeds]
        tool_rates = [results_by_seed[s][cond]["tool_choice"] / n_tasks for s in seeds]
        sem_rates = [results_by_seed[s][cond]["semantic_args"] / n_tasks for s in seeds]
        perf_rates = [results_by_seed[s][cond]["perfect_calls"] / n_tasks for s in seeds]
        tokens_list = [results_by_seed[s][cond]["total_tokens"] for s in seeds]

        summary_stats[cond] = {
            "syntax_validity": calc_stats(syntax_rates),
            "schema_conformance": calc_stats(schema_rates),
            "type_conformance": calc_stats(type_rates),
            "tool_choice": calc_stats(tool_rates),
            "semantic_args": calc_stats(sem_rates),
            "perfect_calls": calc_stats(perf_rates),
            "total_tokens": calc_stats(tokens_list),
        }

    # Struttura di ritorno compatibile con i test esistenti
    primary_seed = seeds[0]
    output_bundle = {
        "backend": active_backend,
        "is_mock": is_mock,
        "url": url,
        "model": model,
        "seeds": seeds,
        "temperature": temperature,
        "num_tasks": n_tasks,
        "summary": summary_stats,
        "results_by_seed": results_by_seed,
    }

    # Compatibilità backwards diretta per i test che interrogano res["unconstrained"] o res["janus_gbnf"]
    for cond in conditions:
        output_bundle[cond] = results_by_seed[primary_seed][cond]

    return output_bundle


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="JANUS Agentic Tool Calling Benchmark (Multi-Backend)")
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "llama-cpp", "ollama", "vllm"],
                        help="Backend da utilizzare: 'mock' (simulatore), 'llama-cpp', 'ollama', o 'vllm'")
    parser.add_argument("--dry-run", action="store_true", help="Alias per --backend mock")
    parser.add_argument("--url", type=str, default=None, help="URL dell'endpoint HTTP (default da backend o JANUS_BENCH_URL)")
    parser.add_argument("--model", type=str, default=None, help="Nome del modello LLM (default da JANUS_BENCH_MODEL)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperatura di campionamento (default: 0.0)")
    parser.add_argument("--seeds", type=str, default="42", help="Lista di seed separati da virgola (es. '42' o '42,43,44')")
    parser.add_argument("--conditions", type=str, default="unconstrained,json_schema,janus_gbnf",
                        help="Condizioni separate da virgola (default: unconstrained,json_schema,janus_gbnf)")
    parser.add_argument("--output", type=str, default="", help="Percorso del file JSON di destinazione")
    args = parser.parse_args()

    active_backend = "mock" if args.dry_run else args.backend
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    cond_list = [c.strip() for c in args.conditions.split(",") if c.strip()]

    if active_backend in ("llama-cpp", "ollama", "vllm"):
        target_url = args.url
        if not target_url:
            if active_backend == "llama-cpp":
                target_url = os.environ.get("LLAMA_CPP_URL", os.environ.get("JANUS_BENCH_URL", "http://localhost:8080"))
            elif active_backend == "ollama":
                target_url = os.environ.get("OLLAMA_URL", os.environ.get("JANUS_BENCH_URL", "http://localhost:11434"))
            elif active_backend == "vllm":
                target_url = os.environ.get("VLLM_URL", os.environ.get("JANUS_BENCH_URL", "http://localhost:8000"))

        endpoint_alive = False
        if requests is not None and target_url:
            try:
                requests.get(target_url.rstrip("/"), timeout=1.5)
                endpoint_alive = True
            except Exception:
                endpoint_alive = False

        if not endpoint_alive:
            print(
                "Nessun server LLM configurato. Per eseguire il benchmark reale avvia un server (es. llama-server) e specifica --url. Per la simulazione usa --dry-run.",
                file=sys.stderr
            )
            sys.exit(2)

    start_time = time.time()
    results = run_benchmark(
        dry_run=(active_backend == "mock"),
        backend=active_backend,
        url=args.url,
        model=args.model,
        conditions=cond_list,
        seeds=seed_list,
        temperature=args.temperature,
    )
    elapsed = time.time() - start_time

    os.makedirs("benchmarks/results", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_path = args.output
    else:
        prefix = "simulated_agent_eval" if results["is_mock"] else "real_agent_eval"
        model_tag = results["model"].replace("/", "_").replace(":", "_")
        out_path = f"benchmarks/results/{prefix}_{active_backend}_{model_tag}_{ts}.json"

    payload = {
        "metadata": {
            "commit": get_git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": active_backend,
            "is_mock": results["is_mock"],
            "is_simulated": results["is_mock"],
            "janus_version": "0.2.0",
            "warning": "SIMULATED / MOCK DATA: DO NOT REPORT AS REAL MEASUREMENTS" if results["is_mock"] else None,
            "url": results["url"],
            "model": results["model"],
            "temperature": args.temperature,
            "seeds": seed_list,
            "conditions": cond_list,
            "num_tasks": results["num_tasks"],
            "elapsed_seconds": round(elapsed, 4),
        },
        "summary": results["summary"],
        "results": results["results_by_seed"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 90)
    print("BENCHMARK AGENTIC TOOL CALLING: RISULTATI COMPARATIVI")
    print("=" * 90)
    print(f"Backend: {active_backend.upper()} {'(Simulatore Mock)' if results['is_mock'] else '(Modello Reale)'}")
    print(f"Modello: {results['model']} | Endpoint: {results['url']}")
    print(f"Seed valutati: {seed_list} (N={len(seed_list)}) | Temperatura: {args.temperature}")
    print(f"File risultati grezzi: {out_path}")
    print("-" * 90)
    print(f"{'Metrica':<24} | {'Unconstrained JSON':<20} | {'JSON Schema':<20} | {'JANUS GBNF':<20}")
    print("-" * 90)

    summary = results["summary"]
    metrics_display = [
        ("Syntax Validity", "syntax_validity", True),
        ("Schema Conformance", "schema_conformance", True),
        ("Type Accuracy", "type_conformance", True),
        ("Tool Choice", "tool_choice", True),
        ("Semantic Args GT", "semantic_args", True),
        ("Perfect Tool Calls", "perfect_calls", True),
        ("Tokens Consumed", "total_tokens", False),
    ]

    for label, key, is_pct in metrics_display:
        cells = []
        for c in ["unconstrained", "json_schema", "janus_gbnf"]:
            if c in summary and key in summary[c]:
                stat = summary[c][key]
                m = stat["mean"]
                s = stat["std"]
                if is_pct:
                    cells.append(f"{m * 100:>5.1f}% ± {s * 100:>4.1f}%")
                else:
                    cells.append(f"{m:>6.0f} ± {s:>4.0f}")
            else:
                cells.append("N/A")
        print(f"{label:<24} | {cells[0]:<20} | {cells[1]:<20} | {cells[2]:<20}")

    print("=" * 90)
    if results["is_mock"]:
        print("[MOCK] Risultati simulati generati a scopo di verifica architetturale.")


if __name__ == "__main__":
    main()
