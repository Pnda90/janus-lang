"""
Test di verifica per gli esempi 11 e 12 (Agent Tool Pipeline e Guardrails).
Verifica che i programmi .jn compilino senza errori ed eseguano nella AgentRuntime.
"""

from pathlib import Path
from janus.lexer import Lexer
from janus.parser import Parser
from janus.type_checker import TypeChecker
from janus.codegen import CodeGenerator
from janus.agent_runtime import AgentRuntime


def test_example_11_safe_tool_pipeline_compilation_and_execution():
    ex_path = Path("examples/11_safe_tool_pipeline.jn")
    assert ex_path.exists()
    code = ex_path.read_text(encoding="utf-8")

    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    diags = TypeChecker().check(ast)
    errs = [d for d in diags if d.code.startswith("ERR")]
    assert not errs, f"Errori inattesi: {[e.message for e in errs]}"

    runtime = AgentRuntime()
    # Registra i tool con mock reali
    @runtime.sandbox.tool("SearchEngine", effect="io")
    def search_mock(query: str, top_k: int = 5):
        class SearchOutput:
            raw_results = f"Results for: {query}"
        return SearchOutput()

    @runtime.sandbox.tool("TextClassifier", effect="pure")
    def classifier_mock(content: str, threshold: float = 0.5):
        class ClassifierOutput:
            is_relevant = True
            confidence = 0.95
        return ClassifierOutput()

    @runtime.sandbox.tool("CacheStorage", effect="io")
    def cache_mock(key: str, value: str):
        class CacheOutput:
            status = "OK"
        return CacheOutput()

    res, traces = runtime.execute(code, entrypoint="orchestrate_search_and_cache", args=["machine learning", 0.7])
    assert res == 1
    assert len(traces) == 3
    assert traces[0].tool_name == "SearchEngine"
    assert traces[1].tool_name == "TextClassifier"
    assert traces[2].tool_name == "CacheStorage"


def test_example_12_agent_guardrails_compilation_and_execution():
    ex_path = Path("examples/12_agent_guardrails.jn")
    assert ex_path.exists()
    code = ex_path.read_text(encoding="utf-8")

    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    diags = TypeChecker().check(ast)
    errs = [d for d in diags if d.code.startswith("ERR")]
    assert not errs, f"Errori inattesi: {[e.message for e in errs]}"

    runtime = AgentRuntime()

    @runtime.sandbox.tool("SafeMathEvaluator", effect="pure")
    def math_mock(expression: str):
        class MathOutput:
            value = 42.0
            valid = True
        return MathOutput()

    @runtime.sandbox.tool("AuditTrail", effect="io")
    def audit_mock(action: str, actor: str, details: str):
        class AuditOutput:
            logged = True
        return AuditOutput()

    res, traces = runtime.execute(code, entrypoint="safe_audit_and_compute", args=["6 * 7", "user_admin"])
    assert res == 42.0
    assert len(traces) == 2
    assert traces[0].tool_name == "SafeMathEvaluator"
    assert traces[1].tool_name == "AuditTrail"
