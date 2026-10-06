"""
Property-based test using Hypothesis:
Verifies that distinct identifiers in JANUS source code remain distinct in the generated Python code,
and that parameter names never collide or cause duplicate argument syntax errors.
"""

import pytest
from hypothesis import given, strategies as st
from janus.lexer import Lexer
from janus.parser import Parser
from janus.codegen import CodeGenerator
from janus.type_checker import TypeChecker

KEYWORDS = {
    "fn", "ret", "let", "mut", "type", "schema", "kern", "if", "else", "loop",
    "for", "in", "brk", "cont", "on", "gpu", "call", "agentv", "wrt", "pure",
    "io", "stoc", "matmul", "dot", "add", "sub", "mul", "div", "relu", "smax",
    "gelu", "norm", "conv2d", "pool", "sum", "pow", "sqrt", "exp", "log", "flat",
    "trans", "diff", "alloc", "copy", "free", "rand", "true", "false"
}

# Strategy generating valid lowercase identifier base names
valid_base_names = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True).filter(
    lambda s: s not in KEYWORDS
)

case_suffixes = st.sampled_from(["m", "b", "t", "n", "s", "v"])


@st.composite
def distinct_janus_identifiers(draw):
    """Generates a list of 2 to 6 unique base identifiers, optionally with case annotations."""
    base_names = draw(st.lists(valid_base_names, min_size=2, max_size=6, unique=True))
    annotated = []
    for b in base_names:
        with_case = draw(st.booleans())
        if with_case:
            case = draw(case_suffixes)
            annotated.append(f"{b}:{case}")
        else:
            annotated.append(b)
    return base_names, annotated


@given(distinct_janus_identifiers())
def test_distinct_identifiers_remain_distinct_in_python(data):
    base_names, param_list = data
    params_str = ", ".join(param_list)
    code = f"""
    fn test_func({params_str}) pure {{
        ret {param_list[0]}
    }}
    """

    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()

    # 1. Type checker should report 0 duplicate parameter errors because base names are distinct
    checker = TypeChecker()
    diags = checker.check(ast)
    dup_errors = [d for d in diags if d.code == "ERR_DUPLICATE_PARAM"]
    assert len(dup_errors) == 0, f"Unexpected duplicate parameter error for {param_list}: {dup_errors}"

    # 2. Codegen must produce valid Python with all distinct parameters
    generator = CodeGenerator()
    py_code = generator.generate(ast)

    # 3. Executing the generated python code must succeed (no duplicate argument SyntaxError)
    env = {}
    exec(py_code, env)
    assert "test_func" in env
    func = env["test_func"]

    # 4. In Python inspection, parameter names must match the base names exactly and be unique
    import inspect
    sig = inspect.signature(func)
    py_params = list(sig.parameters.keys())

    assert len(py_params) == len(base_names), f"Parameter count mismatch: {py_params} vs {base_names}"
    assert len(set(py_params)) == len(py_params), f"Duplicate parameters generated in Python: {py_params}"
    assert py_params == base_names, f"Generated parameter names {py_params} differ from base names {base_names}"
