"""
Test differenziali e formali per grammatiche GBNF e JSON Schema (Fase 2):
1. Campionamento stocastico da GBNF (>= 500 campioni per schema) con validazione strict json.loads e jsonschema.
2. Accettazione di istanze valide da parte di GBNFRecognizer.
3. Rifiuto formale di istanze invalide (1.5 per i32, '007', newline letterale, campo mancante, campo extra).
4. Verifica opzionale con LlamaGrammar ufficiale (llama-cpp-python).
"""

import json
import pytest
import jsonschema
from hypothesis import given, settings, strategies as st

from janus.ast_nodes import SchemaDecl, Param, PrimitiveType, LiteralExpr
from janus.gbnf_gen import GBNFGenerator
from janus.jsonschema_export import schema_to_json_schema
from tests.support.gbnf_engine import GBNFRecognizer, GBNFSampler


def test_sampling_differential_500_samples():
    """
    Ogni stringa campionata direttamente dalla grammatica GBNF (500 campioni)
    deve superare json.loads (strict) E la validazione jsonschema Draft-07.
    """
    schemas = [
        SchemaDecl(
            name="EchoTool",
            inputs=[Param(name="message", type_expr=PrimitiveType(name="str"))]
        ),
        SchemaDecl(
            name="MathCalc",
            inputs=[
                Param(name="operation", type_expr=PrimitiveType(name="str")),
                Param(name="a", type_expr=PrimitiveType(name="f32")),
                Param(name="b", type_expr=PrimitiveType(name="f32"), default=LiteralExpr(value=0.0))
            ]
        ),
        SchemaDecl(
            name="ServerManager",
            inputs=[
                Param(name="action", type_expr=PrimitiveType(name="str")),
                Param(name="retries", type_expr=PrimitiveType(name="i32"), default=LiteralExpr(value=3)),
                Param(name="active", type_expr=PrimitiveType(name="bool"), default=LiteralExpr(value=True))
            ]
        )
    ]

    generator = GBNFGenerator()

    for schema in schemas:
        gbnf_text = generator.generate_tool_json_call_grammar(schema)
        json_schema = schema_to_json_schema(schema, mode="call")
        sampler = GBNFSampler(gbnf_text)

        # Esegui 500 campioni con seed fissi per riproducibilità deterministica
        for i in range(500):
            sample_str = sampler.sample(seed=42 + i)
            # 1. Verifica sintassi JSON stretta
            try:
                parsed_json = json.loads(sample_str)
            except Exception as e:
                pytest.fail(f"Stringa campionata da GBNF non valida per json.loads: {repr(sample_str)} | Errore: {e}")

            # 2. Verifica conformità a JSON Schema
            try:
                jsonschema.validate(instance=parsed_json, schema=json_schema)
            except jsonschema.ValidationError as ve:
                pytest.fail(f"Stringa campionata da GBNF viola JSON Schema: {repr(sample_str)} | Errore: {ve}")


@settings(max_examples=100)
@given(
    tool_arg=st.text(alphabet=st.characters(blacklist_categories=('Cs',), blacklist_characters=['\n', '\r', '"', '\\']), max_size=20),
    count_arg=st.integers(min_value=-1000, max_value=1000),
    ratio_arg=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e5, max_value=1e5)
)
def test_hypothesis_valid_json_accepted_by_recognizer(tool_arg, count_arg, ratio_arg):
    """Ogni istanza JSON valida generata deve essere accettata da GBNFRecognizer."""
    schema = SchemaDecl(
        name="TaskRunner",
        inputs=[
            Param(name="task", type_expr=PrimitiveType(name="str")),
            Param(name="count", type_expr=PrimitiveType(name="i32")),
            Param(name="ratio", type_expr=PrimitiveType(name="f32"))
        ]
    )
    generator = GBNFGenerator()
    gbnf_text = generator.generate_tool_json_call_grammar(schema)
    recognizer = GBNFRecognizer(gbnf_text)

    # Payload canonico
    payload = json.dumps({
        "tool": "TaskRunner",
        "args": {
            "task": tool_arg,
            "count": count_arg,
            "ratio": ratio_arg
        }
    })

    assert recognizer.recognize(payload) is True, f"Payload JSON valido rifiutato da GBNFRecognizer: {payload}"


def test_negative_instances_rejected_by_recognizer():
    """Istanze invalide devono essere respinte categoricamente dal recognizer."""
    schema = SchemaDecl(
        name="StrictTool",
        inputs=[
            Param(name="name", type_expr=PrimitiveType(name="str")),
            Param(name="count", type_expr=PrimitiveType(name="i32"))
        ]
    )
    generator = GBNFGenerator()
    gbnf_text = generator.generate_tool_json_call_grammar(schema)
    recognizer = GBNFRecognizer(gbnf_text)

    # 1. Float 1.5 al posto di intero i32
    bad_float = '{"tool": "StrictTool", "args": {"name": "test", "count": 1.5}}'
    assert recognizer.recognize(bad_float) is False, "Il recognizer non deve ammettere 1.5 per campo i32"

    # 2. Leading zero '007' non ammesso da integer
    bad_leading_zero = '{"tool": "StrictTool", "args": {"name": "test", "count": 007}}'
    assert recognizer.recognize(bad_leading_zero) is False, "Il recognizer non deve ammettere leading zero come '007'"

    # 3. Stringa con newline letterale non escaped (0x0A)
    bad_newline = '{"tool": "StrictTool", "args": {"name": "line1\nline2", "count": 10}}'
    assert recognizer.recognize(bad_newline) is False, "Il recognizer non deve ammettere newline letterali nelle stringhe"

    # 4. Campo obbligatorio mancante (count)
    bad_missing = '{"tool": "StrictTool", "args": {"name": "test"}}'
    assert recognizer.recognize(bad_missing) is False, "Il recognizer non deve ammettere payload con campi obbligatori mancanti"

    # 5. Campo extra non dichiarato
    bad_extra = '{"tool": "StrictTool", "args": {"name": "test", "count": 10, "extra": "unwanted"}}'
    assert recognizer.recognize(bad_extra) is False, "Il recognizer non deve ammettere campi extra non dichiarati"


def test_official_llama_grammar_cross_check():
    """Cross-verifica opzionale con LlamaGrammar ufficiale di llama-cpp-python se installato."""
    try:
        from llama_cpp import LlamaGrammar
    except ImportError:
        pytest.skip("llama-cpp-python non installato nell'ambiente: skip cross-check ufficiale LlamaGrammar")

    schema = SchemaDecl(name="LlamaCheck", inputs=[Param(name="q", type_expr=PrimitiveType(name="str"))])
    gbnf_text = GBNFGenerator().generate_tool_json_call_grammar(schema)
    grammar = LlamaGrammar.from_string(gbnf_text)
    assert grammar is not None
