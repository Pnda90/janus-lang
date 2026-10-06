"""
Regression test for Bug B3:
examples/01_linreg.jn compiles `w = 0.0` and `b = 0.0` as Python primitive floats,
then attempts to call `torch.autograd.grad(loss, (w, b))`.
At runtime, autograd requires Tensors with `requires_grad=True`, so executing
the generated code crashes.
"""

import pytest
from janus.lexer import Lexer
from janus.parser import Parser
from janus.codegen import CodeGenerator


def test_b3_linreg_execution():
    """Compiling and executing 01_linreg.jn should succeed and train without autograd crash."""
    with open("examples/01_linreg.jn", "r", encoding="utf-8") as f:
        code = f.read()

    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    py_code = CodeGenerator().generate(ast)

    env = {}
    exec(py_code, env)
    assert "linreg" in env, "Generated code must define linreg function"
    linreg_fn = env["linreg"]

    # Trying to execute linreg: in current buggy codegen, w and b are initialized as 0.0 (float)
    # and diff generates torch.autograd.grad(loss, (w, b)) or fails on uninitialized torch.
    # The function call MUST succeed and return numerical weights (w, b).
    # This will fail under the current implementation.
    w, b = linreg_fn([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], 2, 0.01)
    assert isinstance(w, (float, int)), "w must be a numerical scalar"
    assert isinstance(b, (float, int)), "b must be a numerical scalar"
