"""
Test per la generazione di grammatiche GBNF per Piani di Agenti Multi-Tool (Fase A.2):
Verifica la generazione di grammatiche formali GBNF che vincolano l'intero script
dell'agente, impedendo allucinazioni di tool e parametri non conformi.
"""

import pytest
from janus.lexer import Lexer
from janus.parser import Parser
from janus.type_checker import TypeChecker
from janus.gbnf_gen import GBNFGenerator
from janus.gbnf_validator import GBNFValidator

def test_multi_tool_agent_gbnf_grammar_validity():
    code = """
    schema WebSearch io {
        query: str,
        limit: i32 = 10
    } -> {
        results: vec[str]
    }

    schema Calculator pure {
        expression: str
    } -> {
        value: f32
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    schemas = [d for d in ast.declarations if hasattr(d, "inputs")]

    gen = GBNFGenerator()
    gbnf_text = gen.generate_agent_grammar(schemas)

    # Verifica validità sintattica della grammatica GBNF (regole, parentesi, quote, riferimenti)
    assert GBNFValidator.validate_grammar_syntax(gbnf_text)
    assert "tool_call_WebSearch" in gbnf_text
    assert "tool_call_Calculator" in gbnf_text
    assert "root ::=" in gbnf_text

def test_agent_plan_conforms_to_schema_and_types():
    code = """
    schema WebSearch io {
        query: str,
        limit: i32 = 10
    } -> {
        results: vec[str]
    }

    schema Summarizer io {
        text: str
    } -> {
        summary: str
    }

    fn agent_plan(user_query:m: str) io {
        raw_results = call tool WebSearch(query = user_query:m, limit = 5)
        ans = call tool Summarizer(text = "Risultati trovati")
        ret ans
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    checker = TypeChecker()
    diags = checker.check(ast)
    errs = [d for d in diags if d.code.startswith("ERR")]
    assert not errs, f"Errori di tipo inattesi: {[e.message for e in errs]}"

    schemas = [d for d in ast.declarations if hasattr(d, "inputs")]
    gen = GBNFGenerator()
    gbnf_text = gen.generate_agent_grammar(schemas)

    # La grammatica deve definire la combinazione di entrambi i tool
    assert "tool_call_WebSearch" in gbnf_text
    assert "tool_call_Summarizer" in gbnf_text
