"""
Test di Validazione GBNF e Vincoli di Decodifica per Schemi JANUS (Fase 3):
Verifica la correttezza formale della grammatica GBNF generata per tutti gli schema
e testa i criteri di accept / reject su istanze JSON conformi e non conformi.
"""

import json
import pytest
from janus.lexer import Lexer
from janus.parser import Parser
from janus.gbnf_gen import GBNFGenerator
from janus.gbnf_validator import GBNFValidator, GBNFValidationError
from janus.ast_nodes import SchemaDecl


def load_schemas_from_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    return [d for d in ast.declarations if isinstance(d, SchemaDecl)]


def test_gbnf_syntax_all_schemas():
    """Tutti gli schemi del repository devono produrre grammatiche GBNF sintatticamente valide."""
    generator = GBNFGenerator()
    files = ["examples/08_rag_pipeline.jn", "examples/09_react_agent.jn", "stdlib/agent.jn"]

    total_schemas = 0
    for f in files:
        schemas = load_schemas_from_file(f)
        for s in schemas:
            total_schemas += 1
            gbnf = generator.generate_for_schema(s)
            assert GBNFValidator.validate_grammar_syntax(gbnf) is True, f"Grammatica non valida per {s.name}"
            # Nessun doppio apice vuoto residuo
            assert '""' not in gbnf

    assert total_schemas >= 4, f"Attesi almeno 4 schemi, trovati {total_schemas}"


def test_schema_search_query_accept_reject():
    """Accept/Reject per SearchQuery (08_rag_pipeline.jn): docs (vec[str]), scores (vec[f32])."""
    schemas = load_schemas_from_file("examples/08_rag_pipeline.jn")
    schema = next(s for s in schemas if s.name == "SearchQuery")

    # CASO VALIDO (senza status fittizio per default)
    valid_json = json.dumps({
        "docs": ["Documento 1", "Documento 2"],
        "scores": [0.95, 0.82]
    })
    ok, msg = GBNFValidator.validate_json_against_schema(schema, valid_json)
    assert ok is True, f"JSON valido rifiutato: {msg}"

    # CASO CON with_status=True
    valid_with_status = json.dumps({
        "status": "ok",
        "docs": ["Documento 1"],
        "scores": [0.5]
    })
    ok, _ = GBNFValidator.validate_json_against_schema(schema, valid_with_status, with_status=True)
    assert ok is True

    invalid_status = json.dumps({"status": "error", "docs": ["D1"], "scores": [0.5]})
    ok, _ = GBNFValidator.validate_json_against_schema(schema, invalid_status, with_status=True)
    assert ok is False

    # CASO NON VALIDO: campo scores mancante
    missing_field = json.dumps({"docs": ["D1"]})
    ok, _ = GBNFValidator.validate_json_against_schema(schema, missing_field)
    assert ok is False

    # CASO NON VALIDO: tipo scores errato (stringa invece di lista di numeri)
    wrong_type = json.dumps({"docs": ["D1"], "scores": "0.95"})
    ok, _ = GBNFValidator.validate_json_against_schema(schema, wrong_type)
    assert ok is False

    # CASO NON VALIDO: tipo docs errato (lista di numeri invece di stringhe)
    wrong_elem_type = json.dumps({"docs": [1, 2, 3], "scores": [0.1, 0.2, 0.3]})
    ok, _ = GBNFValidator.validate_json_against_schema(schema, wrong_elem_type)
    assert ok is False

    # CASO NON VALIDO: campo extra non previsto
    extra_field = json.dumps({"docs": ["D1"], "scores": [0.5], "extra": 123})
    ok, _ = GBNFValidator.validate_json_against_schema(schema, extra_field)
    assert ok is False


def test_schema_action_accept_reject():
    """Accept/Reject per Action (09_react_agent.jn): result (str), ok (bool)."""
    schemas = load_schemas_from_file("examples/09_react_agent.jn")
    schema = next(s for s in schemas if s.name == "Action")

    # CASO VALIDO
    valid_json = json.dumps({
        "result": "Operazione completata con successo",
        "ok": True
    })
    ok, msg = GBNFValidator.validate_json_against_schema(schema, valid_json)
    assert ok is True, f"JSON valido rifiutato: {msg}"

    # CASO NON VALIDO: ok è stringa anziché boolean
    wrong_bool = json.dumps({
        "result": "test",
        "ok": "true"
    })
    ok, _ = GBNFValidator.validate_json_against_schema(schema, wrong_bool)
    assert ok is False

    # CASO NON VALIDO: result è un numero anziché str
    wrong_result = json.dumps({
        "result": 404,
        "ok": False
    })
    ok, _ = GBNFValidator.validate_json_against_schema(schema, wrong_result)
    assert ok is False


def test_schema_code_exec_tool_accept_reject():
    """Accept/Reject per CodeExecTool (stdlib/agent.jn): stdout (str), stderr (str), exit_code (i32)."""
    schemas = load_schemas_from_file("stdlib/agent.jn")
    schema = next(s for s in schemas if s.name == "CodeExecTool")

    # CASO VALIDO
    valid_json = json.dumps({
        "stdout": "Hello world\n",
        "stderr": "",
        "exit_code": 0
    })
    ok, msg = GBNFValidator.validate_json_against_schema(schema, valid_json)
    assert ok is True, f"JSON valido rifiutato: {msg}"

    # CASO NON VALIDO: exit_code è un float invece di un intero
    wrong_code = json.dumps({
        "stdout": "",
        "stderr": "error",
        "exit_code": 1.5
    })
    ok, _ = GBNFValidator.validate_json_against_schema(schema, wrong_code)
    assert ok is False


def test_schema_search_tool_accept_reject():
    """Accept/Reject per SearchTool (stdlib/agent.jn): docs (vec[str]), scores (vec[f32])."""
    schemas = load_schemas_from_file("stdlib/agent.jn")
    schema = next(s for s in schemas if s.name == "SearchTool")

    valid_json = json.dumps({
        "docs": ["Knowledge article A"],
        "scores": [0.99]
    })
    ok, msg = GBNFValidator.validate_json_against_schema(schema, valid_json)
    assert ok is True, f"JSON valido rifiutato: {msg}"

    # JSON sintatticamente rotto
    ok, _ = GBNFValidator.validate_json_against_schema(schema, "{ broken json")
    assert ok is False
