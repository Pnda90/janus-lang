"""
Regression test for Bug B1:
The lexer/codegen strips the last letter from identifiers ending in n/m/b/t/s/v,
causing name collisions and overwriting parameters.
Examples:
  - `ab` and `bb` collapse to `a` and `b`.
  - `compute(ab, a)` produces duplicate parameter `a` in Python.
  - `xm = x + 1.0` in a function with parameter `xm` overwrites input `x = x + 1.0`.
  - `gw, gb = diff loss wrt (wb, bb)` produces `gw, g` in Python (gb collapses to g).
"""

import pytest
from janus.lexer import Lexer
from janus.parser import Parser
from janus.codegen import CodeGenerator


def test_b1_parameter_collision():
    """Parameters `ab` and `a` must remain distinct and not both collapse to `a`."""
    code = """
    fn compute(ab, a) pure {
        ret ab - a
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    py_code = CodeGenerator().generate(ast)

    # In Python, def compute(a, a) is a SyntaxError due to duplicate argument 'a'
    # The parameters MUST remain distinct.
    assert "def compute(a, a):" not in py_code, "Bug B1: 'ab' and 'a' collapsed into duplicate parameter 'a'"
    assert "ab" in py_code, "Bug B1: Parameter 'ab' should be preserved"


def test_b1_distinct_grad_variables():
    """`gb` must not collapse to `g` while `gw` remains `gw`."""
    code = """
    fn grad_step(loss, wb, bb) pure {
        gw, gb = diff loss wrt (wb, bb)
        ret (gw, gb)
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    py_code = CodeGenerator().generate(ast)

    # `gw, gb` should not generate `gw, g`
    assert "gw, g =" not in py_code, "Bug B1: 'gb' collapsed to 'g' in assignment targets"
    assert "return (gw, g)" not in py_code, "Bug B1: 'gb' collapsed to 'g' in return expression"


def test_b1_assignment_overwriting():
    """`xm = x + 1.0` where `xm` and `x` are distinct should not become `x = (x + 1.0)`."""
    code = """
    fn update(xm, x) pure {
        xm = x + 1.0
        ret xm
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    py_code = CodeGenerator().generate(ast)

    # If xm collapses to x, the generated Python overwrites x: `x = (x + 1.0)`
    assert "def update(x, x):" not in py_code
    assert "xm = " in py_code or ("x_m" in py_code), "Bug B1: 'xm' collapsed to 'x'"
