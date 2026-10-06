"""
Regression test for Bug B2:
The type checker fails to detect undefined variables in expressions.
For example:
  `y = undefined_var + xm`
passes type-checking with 0 errors and empty diagnostics.
"""

import pytest
from janus.lexer import Lexer
from janus.parser import Parser
from janus.type_checker import TypeChecker


def test_b2_undefined_variable_in_binary_expr():
    """An undefined variable used in an expression must produce an error diagnostic."""
    code = """
    fn test_fn(xm) pure {
        y = undefined_var + xm
        ret y
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    checker = TypeChecker()
    diags = checker.check(ast)

    errs = [d for d in diags if d.code.startswith("ERR") or "UNDEFINED" in d.code]
    assert len(errs) > 0, "Bug B2: TypeChecker failed to detect undefined variable 'undefined_var'"
    assert any("undefined_var" in d.message for d in errs), "Bug B2: Error message should mention 'undefined_var'"


def test_b2_undefined_variable_in_return_stmt():
    """Returning an undefined variable directly must produce an error diagnostic."""
    code = """
    fn test_ret() pure {
        ret non_existent_variable
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    checker = TypeChecker()
    diags = checker.check(ast)

    errs = [d for d in diags if d.code.startswith("ERR") or "UNDEFINED" in d.code]
    assert len(errs) > 0, "Bug B2: TypeChecker failed to detect undefined variable 'non_existent_variable'"
