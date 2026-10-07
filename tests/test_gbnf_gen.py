"""
Test di regressione e unitari per GBNFGenerator (Fase 1 della revisione finale):
Verifica:
a) Regola string JSON con escape e controllo U+0000..U+001F
b) Regola integer vs number (niente '007', mapping corretto i8/i32/i64 vs f16/f32/bf16)
c) Supporto a ListType e CustomType, con eccezione GBNFGenerationError per tipi non supportati
d) Ordine canonico dei parametri (obbligatori prima, opzionali poi) senza virgole iniziali
e) Grammatica agentica con if/else, accessi a campo (a.b) e spazio obbligatorio dopo ret
f) CLI janusc gbnf con modalità call, output, agent e flag --with-status
"""

import re
import pytest
from janus.lexer import Lexer
from janus.parser import Parser
from janus.ast_nodes import SchemaDecl, TypeDecl, FieldDecl, Param, PrimitiveType, CustomType, ListType
from janus.gbnf_gen import GBNFGenerator, GBNFGenerationError
from janus.gbnf_validator import GBNFValidator


def test_gbnf_string_rules_and_escapes():
    """La regola string deve escludere i controlli e supportare tutti gli escape JSON standard."""
    gen = GBNFGenerator()
    s = SchemaDecl(name="Echo", inputs=[Param(name="msg", type_expr=PrimitiveType(name="str"))])
    gbnf = gen.generate_tool_json_call_grammar(s)

    assert GBNFValidator.validate_grammar_syntax(gbnf)
    # Controllo presenza regola string rigorosa
    assert r'string ::= ["] ([^"\\\x00-\x1F] | escape)* ["]' in gbnf
    assert r'escape ::= [\\] (["] | [\\/bfnrt] | [u] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])' in gbnf


def test_gbnf_integer_vs_number_mapping():
    """i32 deve mappare a integer (senza leading zero come 007), f32 a number."""
    gen = GBNFGenerator()
    s = SchemaDecl(
        name="Compute",
        inputs=[
            Param(name="count", type_expr=PrimitiveType(name="i32")),
            Param(name="ratio", type_expr=PrimitiveType(name="f32"))
        ]
    )
    gbnf = gen.generate_tool_json_call_grammar(s)
    assert GBNFValidator.validate_grammar_syntax(gbnf)

    # integer e number devono essere definiti come da specifica
    assert 'integer ::= "-"? ("0" | [1-9] [0-9]*)' in gbnf
    assert 'number ::= integer ("." [0-9]+)? ([eE] [-+]? [0-9]+)?' in gbnf

    # count mappa a integer, ratio mappa a number
    assert '"\\"count\\": " ws integer' in gbnf
    assert '"\\"ratio\\": " ws number' in gbnf

    # Verifica con regex che la regola integer rifiuti '007'
    int_regex = re.compile(r'^-?(?:0|[1-9][0-9]*)$')
    assert int_regex.match("0")
    assert int_regex.match("42")
    assert int_regex.match("-10")
    assert not int_regex.match("007")
    assert not int_regex.match("01")


def test_gbnf_unsupported_type_raises_clear_error():
    """Qualsiasi tipo sconosciuto o non supportato deve sollevare GBNFGenerationError senza fallback a string."""
    gen = GBNFGenerator()
    s = SchemaDecl(
        name="Broken",
        inputs=[Param(name="unknown_data", type_expr=CustomType(name="UndeclaredType"))]
    )
    with pytest.raises(GBNFGenerationError) as exc_info:
        gen.generate_tool_json_call_grammar(s)
    assert "UndeclaredType" in str(exc_info.value)


def test_gbnf_custom_type_and_list_type_generation():
    """CustomType dichiarati in TypeDecl e ListType generano le rispettive regole formali."""
    point_type = TypeDecl(
        name="Point",
        fields=[
            FieldDecl(name="x", type_expr=PrimitiveType(name="f32")),
            FieldDecl(name="y", type_expr=PrimitiveType(name="f32"))
        ]
    )
    gen = GBNFGenerator(type_decls={"Point": point_type})

    schema = SchemaDecl(
        name="Draw",
        inputs=[
            Param(name="origin", type_expr=CustomType(name="Point")),
            Param(name="tags", type_expr=ListType(inner=PrimitiveType(name="str")))
        ]
    )
    gbnf = gen.generate_tool_json_call_grammar(schema)
    assert GBNFValidator.validate_grammar_syntax(gbnf)

    # Regola per il tipo CustomType Point
    assert 'Point ::= "{" ws "\\"x\\": " ws number "," ws "\\"y\\": " ws number ws "}"' in gbnf
    # Regola per ListType
    assert 'string_list ::= "[" ws (string ("," ws string)*)? ws "]"' in gbnf


def test_gbnf_canonical_parameter_ordering_no_leading_comma():
    """
    Parametri opzionali: se il primo dichiarato ha un default e il secondo è obbligatorio,
    l'ordine canonico impone obbligatorio prima, opzionale poi, impedendo virgole orfane.
    """
    from janus.ast_nodes import LiteralExpr

    gen = GBNFGenerator()
    # Dichiarato: prima opzionale (default 10), poi obbligatorio (senza default)
    schema = SchemaDecl(
        name="MixedParams",
        inputs=[
            Param(name="opt_val", type_expr=PrimitiveType(name="i32"), default=LiteralExpr(value=10)),
            Param(name="req_val", type_expr=PrimitiveType(name="str"), default=None)
        ]
    )
    gbnf = gen.generate_tool_json_call_grammar(schema)
    assert GBNFValidator.validate_grammar_syntax(gbnf)

    # In ordine canonico, req_val deve precedere opt_val
    assert 'MixedParamsArgs ::= "{" ws "\\"req_val\\": " ws string ws ("," ws "\\"opt_val\\": " ws integer)? ws "}"' in gbnf


def test_gbnf_all_optional_parameters_disjunctive_alternatives():
    """Quando tutti i parametri sono opzionali, le alternative disgiunte garantiscono nessun leading comma."""
    from janus.ast_nodes import LiteralExpr

    gen = GBNFGenerator()
    schema = SchemaDecl(
        name="AllOpt",
        inputs=[
            Param(name="a", type_expr=PrimitiveType(name="str"), default=LiteralExpr(value="hello")),
            Param(name="b", type_expr=PrimitiveType(name="i32"), default=LiteralExpr(value=1))
        ]
    )
    gbnf = gen.generate_tool_json_call_grammar(schema)
    assert GBNFValidator.validate_grammar_syntax(gbnf)

    # Deve contenere alternative sicure
    assert 'AllOptArgs ::=' in gbnf
    assert '"\\"a\\": " ws string' in gbnf
    assert '"\\"b\\": " ws integer' in gbnf


def test_gbnf_agent_grammar_supports_if_else_and_field_access():
    """La grammatica agentica deve supportare if/else, accessi a campi a.b e ret con spazio obbligatorio."""
    code = """
    schema SearchEngine io {
        query: str,
        top_k: i32 = 5
    } -> {
        raw_results: str
    }

    schema TextClassifier pure {
        content: str,
        threshold: f32 = 0.5
    } -> {
        is_relevant: bool,
        confidence: f32
    }
    """
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    schemas = [d for d in ast.declarations if isinstance(d, SchemaDecl)]

    gen = GBNFGenerator()
    gbnf = gen.generate_agent_grammar(schemas)
    assert GBNFValidator.validate_grammar_syntax(gbnf)

    assert 'if_stmt ::= "if" ws1 expr ws "{" ws block ws "}" (ws "else" ws "{" ws block ws "}")?' in gbnf
    assert 'field_access ::= ident ("." ident)*' in gbnf
    assert 'ret_stmt ::= "ret" ws1 (tool_call | expr)' in gbnf
    assert 'tool_call_SearchEngine' in gbnf
    assert 'tool_call_TextClassifier' in gbnf


def test_gbnf_output_grammar_clean_status():
    """generate_for_schema non include 'status': 'ok' a meno di with_status=True."""
    gen = GBNFGenerator()
    schema = SchemaDecl(
        name="ResultTool",
        outputs=[FieldDecl(name="answer", type_expr=PrimitiveType(name="str"))]
    )

    clean_gbnf = gen.generate_for_schema(schema, with_status=False)
    assert 'status' not in clean_gbnf
    assert 'ResultToolOutput ::= "{" ws "\\"answer\\": " ws string ws "}"' in clean_gbnf

    status_gbnf = gen.generate_for_schema(schema, with_status=True)
    assert '"\\"status\\": \\"ok\\""' in status_gbnf


def test_cli_gbnf_modes(capsys, monkeypatch, tmp_path):
    """Verifica l'interfaccia CLI janusc gbnf per le tre modalità call, output e agent."""
    import sys
    import json
    from janus.cli import main

    test_file = tmp_path / "test_tools.jn"
    test_file.write_text("""
    schema Ping io {
        host: str,
        count: i32 = 3
    } -> {
        latency: f32
    }
    """)

    # 1. Mode call (default)
    monkeypatch.setattr(sys, "argv", ["janusc", "gbnf", str(test_file), "--mode", "call"])
    main()
    out = capsys.readouterr().out
    assert "PingCall" in out
    assert '"\\"host\\": " ws string' in out

    # 2. Mode output (default without status)
    monkeypatch.setattr(sys, "argv", ["janusc", "gbnf", str(test_file), "--mode", "output"])
    main()
    out = capsys.readouterr().out
    assert "PingOutput" in out
    assert "status" not in out
    assert '"\\"latency\\": " ws number' in out

    # 3. Mode output con --with-status
    monkeypatch.setattr(sys, "argv", ["janusc", "gbnf", str(test_file), "--mode", "output", "--with-status"])
    main()
    out = capsys.readouterr().out
    assert '"\\"status\\": \\"ok\\""' in out

    # 4. Mode agent
    monkeypatch.setattr(sys, "argv", ["janusc", "gbnf", str(test_file), "--mode", "agent"])
    main()
    out = capsys.readouterr().out
    assert "tool_call_Ping" in out
    assert "if_stmt ::=" in out

    # 5. Errore di sintassi con --json
    bad_file = tmp_path / "bad.jn"
    bad_file.write_text("schema { broken")
    monkeypatch.setattr(sys, "argv", ["janusc", "gbnf", str(bad_file), "--json"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err_json = json.loads(capsys.readouterr().out)
    if isinstance(err_json, list):
        assert err_json[0].get("code") in ("ERR_SYNTAX", "ERR_PARSER")
    else:
        assert err_json.get("code") in ("ERR_SYNTAX", "ERR_PARSER")

