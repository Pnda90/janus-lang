"""
Regression test for Bug B4:
`janusc gbnf` (GBNFGenerator) generates malformed quoting in string literals,
such as `""status":"` and `""ok""` or unbalanced escaped quotes.
In GBNF grammar specification (llama.cpp):
Literals must be valid quoted strings `"..."`, without adjacent empty double-quotes `""`.
"""

import pytest
from janus.lexer import Lexer
from janus.parser import Parser
from janus.gbnf_gen import GBNFGenerator
from janus.ast_nodes import SchemaDecl


def test_b4_no_unbalanced_or_adjacent_double_quotes():
    """GBNF rules must not contain broken double-quote artifacts like `\"\"status\":`."""
    with open("examples/08_rag_pipeline.jn", "r", encoding="utf-8") as f:
        code = f.read()

    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()

    schemas = [d for d in ast.declarations if isinstance(d, SchemaDecl)]
    assert len(schemas) > 0, "examples/08_rag_pipeline.jn must have schema declarations"

    generator = GBNFGenerator()
    for s in schemas:
        gbnf_output = generator.generate_for_schema(s)
        # Bug B4 causes artifacts like:
        # ""status":" or ""ok"" or "\"docs\": ws number_list,\""
        assert '""status":' not in gbnf_output, "Bug B4: Malformed adjacent double quotes found in GBNF output"
        assert '""ok""' not in gbnf_output, "Bug B4: Malformed adjacent double quotes found in GBNF output"
        # In fact, adjacent empty quotes `""` before a label or value should never occur in valid GBNF:
        assert '""' not in gbnf_output, f"Bug B4: Found invalid adjacent double-quotes in GBNF:\n{gbnf_output}"
