"""
Generatore di Grammatiche GBNF per Grammar-Constrained Decoding con LLM (llama.cpp / vLLM / Outlines).
Converte definizioni SchemaDecl e TypeDecl in automi a stati finiti formali.
"""

from typing import List, Dict, Any
from janus.ast_nodes import SchemaDecl, TypeDecl, PrimitiveType, TensorType, CustomType

class GBNFGenerator:
    """
    Genera regole formali GBNF (Grammar-Based Context-Free Grammar)
    per costringere il modello a produrre output JSON strettamente conforme allo schema.
    """
    def __init__(self):
        pass

    def generate_for_schema(self, schema: SchemaDecl) -> str:
        field_parts = []
        for o in schema.outputs:
            type_rule = self._map_type(o.type_expr)
            field_parts.append(f'"\\"{o.name}\\": " ws {type_rule}')

        fields_str = ' "," ws '.join(field_parts)
        if fields_str:
            output_rule = f'{schema.name}Output ::= "{{" ws "\\"status\\": \\"ok\\"," ws {fields_str} ws "}}"'
        else:
            output_rule = f'{schema.name}Output ::= "{{" ws "\\"status\\": \\"ok\\"" ws "}}"'

        lines = [
            f"# GBNF Grammar for JANUS Schema: {schema.name}",
            f"root ::= {schema.name}Output",
            "",
            output_rule,
            "",
            'ws ::= [ \\t\\n\\r]*',
            'string ::= ["] [^"\\\\]* ["]',
            'number ::= ("-"? [0-9]+ ("." [0-9]+)?)',
            'boolean ::= ("true" | "false")',
            'string_list ::= "[" ws (string ("," ws string)*)? ws "]"',
            'number_list ::= "[" ws (number ("," ws number)*)? ws "]"',
        ]
        return "\n".join(lines)

    def _map_type(self, type_expr: Any) -> str:
        if isinstance(type_expr, PrimitiveType):
            if type_expr.name == "str":
                return "string"
            elif type_expr.name in ("f32", "f16", "i32", "i64"):
                return "number"
            elif type_expr.name == "bool":
                return "boolean"
        elif isinstance(type_expr, TensorType):
            if type_expr.dtype == "str":
                return "string_list"
            return "number_list"
        return "string"
