"""
Generatore di Grammatiche GBNF per Grammar-Constrained Decoding con LLM (llama.cpp / vLLM / Outlines).
Converte definizioni SchemaDecl e TypeDecl in automi a stati finiti formali.
Supporta:
1. Generazione di grammatiche JSON per singoli schemi di output.
2. Generazione di grammatiche multi-tool per script e piani di esecuzione di agenti.
"""

from typing import List, Dict, Any
from janus.ast_nodes import SchemaDecl, TypeDecl, PrimitiveType, TensorType, CustomType

class GBNFGenerator:
    """
    Genera regole formali GBNF (Grammar-Based Context-Free Grammar)
    per costringere il modello a produrre output JSON strettamente conforme allo schema
    oppure script di esecuzione agentica conformi agli schemi dei tool registrati.
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

    def generate_agent_grammar(self, schemas: List[SchemaDecl]) -> str:
        """
        Sintetizza una grammatica GBNF per un intero piano agentico vincolato a un catalogo di tool.
        L'LLM può chiamare solo i tool registrati con i loro argomenti tipizzati,
        eseguire assegnazioni, controlli condizionali e return.
        """
        lines = [
            "# GBNF Multi-Tool Agent Execution Grammar for JANUS",
            "root ::= ws (stmt (ws stmt)*)? ws",
            "",
            "stmt ::= assignment_stmt | tool_call | ret_stmt",
            'assignment_stmt ::= ident ws "=" ws (tool_call | expr)',
            'ret_stmt ::= "ret" ws (tool_call | expr)',
            "",
        ]

        tool_rules = []
        for s in schemas:
            tool_rule_name = f"tool_call_{s.name}"
            tool_rules.append(tool_rule_name)

            # Argomenti del tool
            arg_clauses = []
            for i, p in enumerate(s.inputs):
                ptype = self._map_type(p.type_expr)
                clause = f'"{p.name} = " ws ({ptype} | ident)'
                if p.default is not None:
                    # Parametro opzionale con default
                    if i == 0:
                        arg_clauses.append(f'({clause})?')
                    else:
                        arg_clauses.append(f'("," ws {clause})?')
                else:
                    # Parametro obbligatorio
                    if i == 0:
                        arg_clauses.append(f'({clause})')
                    else:
                        arg_clauses.append(f'("," ws {clause})')

            if arg_clauses:
                args_joined = " ws ".join(arg_clauses)
                lines.append(f'{tool_rule_name} ::= "call tool {s.name}(" ws {args_joined} ws ")"')
            else:
                lines.append(f'{tool_rule_name} ::= "call tool {s.name}()"')

        tool_choices = " | ".join(tool_rules) if tool_rules else '"[no_tools]"'
        lines.append(f"tool_call ::= {tool_choices}")
        lines.append("")
        lines.append('ident ::= [a-zA-Z_] [a-zA-Z0-9_:]*')
        lines.append('expr ::= string | number | boolean | ident')
        lines.append('ws ::= [ \\t\\n\\r]*')
        lines.append('string ::= ["] [^"\\\\]* ["]')
        lines.append('number ::= ("-"? [0-9]+ ("." [0-9]+)?)')
        lines.append('boolean ::= ("true" | "false")')
        lines.append('string_list ::= "[" ws (string ("," ws string)*)? ws "]"')
        lines.append('number_list ::= "[" ws (number ("," ws number)*)? ws "]"')

        return "\n".join(lines)

    def generate_tool_json_call_grammar(self, schema: SchemaDecl) -> str:
        """
        Sintetizza una grammatica GBNF per invocare un tool in formato JSON:
        {"tool": "NomeTool", "args": { ... }}
        """
        field_parts = []
        for i, p in enumerate(schema.inputs):
            ptype = self._map_type(p.type_expr)
            field_rule = f'"\\"{p.name}\\": " ws {ptype}'
            if p.default is not None:
                if i == 0:
                    field_parts.append(f'({field_rule})?')
                else:
                    field_parts.append(f'("," ws {field_rule})?')
            else:
                if i == 0:
                    field_parts.append(f'({field_rule})')
                else:
                    field_parts.append(f'("," ws {field_rule})')

        fields_str = ' ws '.join(field_parts) if field_parts else ''
        if fields_str:
            args_rule = f'{schema.name}Args ::= "{{" ws {fields_str} ws "}}"'
        else:
            args_rule = f'{schema.name}Args ::= "{{" ws "}}"'

        call_rule = f'{schema.name}Call ::= "{{" ws "\\"tool\\": \\"{schema.name}\\", \\"args\\": " ws {schema.name}Args ws "}}"'

        lines = [
            f"# GBNF JSON Tool Call Grammar for JANUS: {schema.name}",
            f"root ::= {schema.name}Call",
            "",
            call_rule,
            args_rule,
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
