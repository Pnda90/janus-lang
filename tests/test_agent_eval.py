"""
Test per il benchmark di Tool Calling e valutazione vincoli agentici (Fase A.4).
Verifica l'integrità dei 20 task agentici, la logica di validazione e l'esecuzione del benchmark.
"""

import json
from benchmarks.agent_eval import AGENT_TASKS, validate_tool_payload, run_benchmark
from janus.lexer import Lexer
from janus.parser import Parser
from janus.ast_nodes import SchemaDecl
from janus.gbnf_gen import GBNFGenerator
from janus.gbnf_validator import GBNFValidator


def test_agent_eval_tasks_integrity():
    assert len(AGENT_TASKS) == 20
    for task in AGENT_TASKS:
        assert "id" in task
        assert "schema_code" in task
        assert "tool_name" in task
        assert "required_args" in task
        assert "sample_valid_input" in task

        # Verifica che lo schema compili
        tokens = Lexer(task["schema_code"]).tokenize()
        ast = Parser(tokens).parse()
        assert len(ast.declarations) == 1
        decl = ast.declarations[0]
        assert isinstance(decl, SchemaDecl)
        assert decl.name == task["tool_name"]


def test_agent_eval_validation_logic():
    task = AGENT_TASKS[0]  # WebSearch { query: str, max_results: i32 = 10 }

    # 1. Payload valido
    v1 = validate_tool_payload('{"query": "pytorch 2.0", "max_results": 5}', task)
    assert v1["syntax_valid"] is True
    assert v1["schema_conformance"] is True
    assert v1["type_conformance"] is True

    # 2. JSON malformato
    v2 = validate_tool_payload('{"query": "pytorch 2.0", ', task)
    assert v2["syntax_valid"] is False

    # 3. Campo obbligatorio mancante
    v3 = validate_tool_payload('{"max_results": 5}', task)
    assert v3["syntax_valid"] is True
    assert v3["schema_conformance"] is False

    # 4. Campo allucinato
    v4 = validate_tool_payload('{"query": "pytorch", "fake_param": 123}', task)
    assert v4["syntax_valid"] is True
    assert v4["schema_conformance"] is False

    # 5. Type mismatch
    v5 = validate_tool_payload('{"query": "pytorch", "max_results": "NOT_AN_INT"}', task)
    assert v5["syntax_valid"] is True
    assert v5["schema_conformance"] is True
    assert v5["type_conformance"] is False


def test_agent_eval_benchmark_execution():
    res = run_benchmark(dry_run=True)
    assert "unconstrained" in res
    assert "janus_gbnf" in res

    # GBNF deve garantire 100% di conformità sintattica e di schema per costruzione
    assert res["janus_gbnf"]["syntax_valid"] == 20
    assert res["janus_gbnf"]["schema_conformance"] == 20
    assert res["janus_gbnf"]["type_conformance"] == 20
    assert res["janus_gbnf"]["perfect_calls"] == 20

    # Unconstrained deve riflettere i tassi realistici di fallimento
    assert res["unconstrained"]["perfect_calls"] < 20


def test_agent_eval_multi_seed_and_conditions():
    res = run_benchmark(backend="mock", seeds=[42, 43], conditions=["unconstrained", "json_schema", "janus_gbnf"])
    assert "summary" in res
    assert "json_schema" in res["summary"]
    assert "janus_gbnf" in res["summary"]
    assert "unconstrained" in res["summary"]

    # Verifica presenza statistiche mean e std
    for cond in ["unconstrained", "json_schema", "janus_gbnf"]:
        assert "mean" in res["summary"][cond]["syntax_validity"]
        assert "std" in res["summary"][cond]["syntax_validity"]
        assert "mean" in res["summary"][cond]["semantic_args"]

    # Decodifica vincolata (JSON Schema e GBNF) deve garantire 100% sintassi e schema
    assert res["summary"]["json_schema"]["syntax_validity"]["mean"] == 1.0
    assert res["summary"]["janus_gbnf"]["syntax_validity"]["mean"] == 1.0
    assert res["summary"]["json_schema"]["schema_conformance"]["mean"] == 1.0
    assert res["summary"]["janus_gbnf"]["schema_conformance"]["mean"] == 1.0


def test_agent_eval_tool_choice_and_semantic_validation():
    task = AGENT_TASKS[0]  # WebSearch { query: str, max_results: i32 = 10 }

    # Scelta del tool errata
    v_wrong_tool = validate_tool_payload('{"tool": "DatabaseFetch", "args": {"query": "PyTorch"}}', task)
    assert v_wrong_tool["syntax_valid"] is True
    assert v_wrong_tool["tool_choice_correct"] is False
    assert v_wrong_tool["perfect_call"] is False

    # Scelta corretta con argomenti semantici corretti
    v_correct = validate_tool_payload('{"tool": "WebSearch", "args": {"query": "PyTorch release notes", "max_results": 5}}', task)
    assert v_correct["syntax_valid"] is True
    assert v_correct["tool_choice_correct"] is True
    assert v_correct["semantic_args_correct"] is True
    assert v_correct["perfect_call"] is True

    # Argomento semantico scorretto (query non attinente)
    v_sem_wrong = validate_tool_payload('{"tool": "WebSearch", "args": {"query": "unrelated topic recipes", "max_results": 5}}', task)
    assert v_sem_wrong["tool_choice_correct"] is True
    assert v_sem_wrong["semantic_args_correct"] is False
    assert v_sem_wrong["perfect_call"] is False


def test_agent_eval_gbnf_json_tool_grammar():
    gen = GBNFGenerator()
    val = GBNFValidator()
    for task in AGENT_TASKS:
        tokens = Lexer(task["schema_code"]).tokenize()
        ast = Parser(tokens).parse()
        schema_decl = ast.declarations[0]
        grammar = gen.generate_tool_json_call_grammar(schema_decl)
        assert val.validate_grammar_syntax(grammar)
        assert f"{schema_decl.name}Call" in grammar
        assert f"{schema_decl.name}Args" in grammar
