"""
Test per l'architettura Agentic Execution Graph (Fase A.1):
Verifica la dichiarazione formale di schemi di tool, la tipizzazione degli effetti
e l'invocazione sicura di tool con diagnostica strutturata.
"""

import pytest
from janus.lexer import Lexer
from janus.parser import Parser
from janus.type_checker import TypeChecker
from janus.codegen import CodeGenerator
from janus.ast_nodes import SchemaDecl, ToolCallExpr

def test_schema_with_explicit_effect():
    code = """
    schema SafeCalculator pure {
        expr: str
    } -> {
        result: f32
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    assert len(ast.declarations) == 1
    schema = ast.declarations[0]
    assert isinstance(schema, SchemaDecl)
    assert schema.name == "SafeCalculator"
    assert schema.effect == "pure"

def test_tool_call_syntax_and_codegen():
    code = """
    schema WebSearch io {
        query: str,
        limit: i32 = 10
    } -> {
        results: vec[str]
    }

    fn search_workflow(q:m: str) io {
        res = call tool WebSearch(query = q:m, limit = 5)
        ret res
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    diags = TypeChecker().check(ast)
    errs = [d for d in diags if d.code.startswith("ERR")]
    assert not errs, f"Errori inattesi: {[e.message for e in errs]}"

    py_code = CodeGenerator().generate(ast)
    assert "_janus_call_tool" in py_code
    assert "'WebSearch'" in py_code

def test_tool_call_undefined_schema_diagnostic():
    code = """
    fn run_task() io {
        res = call tool InexistentTool(query = "hello")
        ret res
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    diags = TypeChecker().check(ast)
    err_codes = [d.code for d in diags]
    assert "ERR_UNDEFINED_TOOL" in err_codes

def test_tool_call_purity_violation():
    code = """
    schema WebSearch io {
        query: str
    } -> {
        results: vec[str]
    }

    fn pure_function() pure {
        res = call tool WebSearch(query = "hello")
        ret res
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    diags = TypeChecker().check(ast)
    err_codes = [d.code for d in diags]
    assert "ERR_EFFECT_PURITY_VIOLATION" in err_codes

def test_tool_call_missing_required_argument():
    code = """
    schema WebSearch io {
        query: str,
        limit: i32 = 10
    } -> {
        results: vec[str]
    }

    fn search_workflow() io {
        res = call tool WebSearch(limit = 5)
        ret res
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    diags = TypeChecker().check(ast)
    err_codes = [d.code for d in diags]
    assert "ERR_MISSING_TOOL_ARGUMENT" in err_codes
