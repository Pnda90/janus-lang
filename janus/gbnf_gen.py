"""
Generatore di Grammatiche GBNF per Grammar-Constrained Decoding con LLM (llama.cpp / vLLM / Outlines).
Converte definizioni SchemaDecl e TypeDecl in automi a stati finiti formali conformi allo standard GBNF.

Supporta:
1. `mode="call"`: Generazione di grammatiche JSON per la CHIAMATA al tool (l'input generato dall'LLM per invocare il tool).
2. `mode="output"`: Generazione di grammatiche JSON per l'OUTPUT del tool (risposta strutturata).
3. `mode="agent"`: Generazione di grammatiche per interi piani di esecuzione agentici multi-tool (script JANUS vincolato).
"""

from typing import List, Dict, Any, Optional, Union
from janus.ast_nodes import (
    SchemaDecl, TypeDecl, PrimitiveType, TensorType, CustomType, ListType, Param
)


class GBNFGenerationError(Exception):
    """Sollevata quando la generazione GBNF fallisce a causa di tipi non supportati o configurazioni non valide."""
    pass


CORE_RULES = [
    'ws ::= [ \\t\\n\\r]*',
    'ws1 ::= [ \\t\\n\\r]+',
    r'string ::= ["] ([^"\\\x00-\x1F] | escape)* ["]',
    r'escape ::= [\\] (["] | [\\/bfnrt] | [u] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])',
    'integer ::= "-"? ("0" | [1-9] [0-9]*)',
    'number ::= integer ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
    'boolean ::= ("true" | "false")',
    'string_list ::= "[" ws (string ("," ws string)*)? ws "]"',
    'integer_list ::= "[" ws (integer ("," ws integer)*)? ws "]"',
    'number_list ::= "[" ws (number ("," ws number)*)? ws "]"',
]


class GBNFGenerator:
    """
    Generatore formale GBNF (Grammar-Based Context-Free Grammar) per motori compatibili con llama.cpp.
    
    Regole di ordinamento canonico per argomenti:
    - Tutti i parametri obbligatori (senza default) vengono generati per primi, nell'ordine di dichiarazione.
    - I parametri opzionali (con default) vengono generati successivamente, ciascuno preceduto da una virgola opzionale.
    - Quando tutti i parametri sono opzionali, vengono generate alternative disgiunte per garantire
      che nessun valore opzionale iniziale ometta le parentesi/virgole in modo da produrre JSON malformato
      (nessuna virgola iniziale orfana).
    """

    def __init__(self, type_decls: Optional[Dict[str, TypeDecl]] = None):
        self.type_decls: Dict[str, TypeDecl] = dict(type_decls) if type_decls else {}
        self._custom_rules: Dict[str, str] = {}
        self._extra_rules: Dict[str, str] = {}

    def register_type_decls(self, type_decls: Dict[str, TypeDecl]) -> None:
        """Registra dichiarazioni di tipi custom per la risoluzione nei campi degli schemi."""
        self.type_decls.update(type_decls)

    def generate_for_schema(
        self,
        schema: SchemaDecl,
        with_status: bool = False,
        type_decls: Optional[Dict[str, TypeDecl]] = None
    ) -> str:
        """
        Sintetizza la grammatica GBNF per l'output strutturato di un tool.
        Di default non include campi fittizi non dichiarati.
        Se with_status=True, include opzionalmente `"status": "ok"` in testa all'oggetto.
        """
        if type_decls:
            self.type_decls.update(type_decls)
        self._custom_rules.clear()
        self._extra_rules.clear()

        field_parts = []
        if with_status:
            field_parts.append('"\\"status\\": \\"ok\\""')

        for o in schema.outputs:
            type_rule = self._map_type(o.type_expr)
            field_parts.append(f'"\\"{o.name}\\": " ws {type_rule}')

        fields_str = ' "," ws '.join(field_parts)
        if fields_str:
            output_rule = f'{schema.name}Output ::= "{{" ws {fields_str} ws "}}"'
        else:
            output_rule = f'{schema.name}Output ::= "{{" ws "}}"'

        lines = [
            f"# GBNF Grammar for JANUS Schema Output: {schema.name}",
            f"root ::= {schema.name}Output",
            "",
            output_rule,
            "",
        ]

        for extra in self._custom_rules.values():
            if extra:
                lines.append(extra)
        for extra in self._extra_rules.values():
            lines.append(extra)

        lines.extend(CORE_RULES)
        return "\n".join(lines)

    def generate_tool_json_call_grammar(
        self,
        schema_or_schemas: Union[SchemaDecl, List[SchemaDecl]],
        type_decls: Optional[Dict[str, TypeDecl]] = None
    ) -> str:
        """
        Sintetizza una grammatica GBNF per invocare tool in formato JSON:
        {"tool": "NomeTool", "args": { ... }}
        Supporta un singolo SchemaDecl o una lista di SchemaDecl (multi-tool call dispatch).
        Gli argomenti sono generati in ordine canonico (obbligatori prima, opzionali poi).
        """
        if type_decls:
            self.type_decls.update(type_decls)
        self._custom_rules.clear()
        self._extra_rules.clear()

        schemas = schema_or_schemas if isinstance(schema_or_schemas, list) else [schema_or_schemas]
        if not schemas:
            raise GBNFGenerationError("Nessuno schema fornito per la generazione della grammatica call JSON.")

        call_rules = []
        rule_definitions = []

        for schema in schemas:
            call_rule_name = f"{schema.name}Call"
            args_rule_name = f"{schema.name}Args"
            call_rules.append(call_rule_name)

            args_def = self._build_canonical_args_json_rule(args_rule_name, schema.inputs)
            call_def = f'{call_rule_name} ::= "{{" ws "\\"tool\\": \\"{schema.name}\\", \\"args\\": " ws {args_rule_name} ws "}}"'

            rule_definitions.append(call_def)
            rule_definitions.append(args_def)

        if len(call_rules) == 1:
            root_rule = f"root ::= {call_rules[0]}"
            title = schemas[0].name
        else:
            root_rule = f"root ::= {' | '.join(call_rules)}"
            title = "Multi-Tool Dispatch"

        lines = [
            f"# GBNF JSON Tool Call Grammar for JANUS: {title}",
            root_rule,
            "",
        ]
        lines.extend(rule_definitions)
        lines.append("")

        for extra in self._custom_rules.values():
            if extra:
                lines.append(extra)
        for extra in self._extra_rules.values():
            lines.append(extra)

        lines.extend(CORE_RULES)
        return "\n".join(lines)

    def generate_agent_grammar(
        self,
        schemas: List[SchemaDecl],
        type_decls: Optional[Dict[str, TypeDecl]] = None
    ) -> str:
        """
        Sintetizza una grammatica GBNF per un intero piano agentico vincolato a un catalogo di tool.
        L'LLM può chiamare solo i tool registrati con i loro argomenti tipizzati,
        eseguire assegnazioni, controlli condizionali (if/else), field access (a.b) e return.
        
        Costrutti supportati:
        - Assegnazione: `ident = call tool Tool(...)` o `ident = expr`
        - Chiamata tool: `call tool NomeTool(param = val, ...)` (ordine canonico: obbligatori poi opzionali)
        - Condizionale: `if expr { block } (else { block })?`
        - Return: `ret expr` (spazio obbligatorio ws1 dopo 'ret')
        - Espressioni: stringhe, interi, numeri, booleani, accessi a campo `ident.campo`, confronti (==, !=, <, <=, >, >=).
        """
        if type_decls:
            self.type_decls.update(type_decls)
        self._custom_rules.clear()
        self._extra_rules.clear()

        lines = [
            "# GBNF Multi-Tool Agent Execution Grammar for JANUS",
            "root ::= ws (stmt (ws stmt)*)? ws",
            "",
            "stmt ::= if_stmt | assignment_stmt | tool_call | ret_stmt",
            "block ::= (stmt (ws stmt)*)?",
            'if_stmt ::= "if" ws1 expr ws "{" ws block ws "}" (ws "else" ws "{" ws block ws "}")?',
            'assignment_stmt ::= ident ws "=" ws (tool_call | expr)',
            'ret_stmt ::= "ret" ws1 (tool_call | expr)',
            "",
            "expr ::= comparison_expr | primary_expr",
            "comparison_expr ::= primary_expr ws comp_op ws primary_expr",
            'comp_op ::= "==" | "!=" | "<=" | ">=" | "<" | ">"',
            "primary_expr ::= string | number | boolean | field_access",
            'field_access ::= ident ("." ident)*',
            "",
        ]

        tool_rules = []
        rule_definitions = []
        for s in schemas:
            tool_rule_name = f"tool_call_{s.name}"
            tool_rules.append(tool_rule_name)

            call_def = self._build_canonical_agent_tool_call_rule(tool_rule_name, s.name, s.inputs)
            rule_definitions.append(call_def)

        tool_choices = " | ".join(tool_rules) if tool_rules else '"[no_tools]"'
        lines.append(f"tool_call ::= {tool_choices}")
        lines.append("")
        lines.extend(rule_definitions)
        lines.append("")

        lines.append('ident ::= [a-zA-Z_] [a-zA-Z0-9_:]*')
        for extra in self._custom_rules.values():
            if extra:
                lines.append(extra)
        for extra in self._extra_rules.values():
            lines.append(extra)

        lines.extend(CORE_RULES)
        return "\n".join(lines)

    def _build_canonical_args_json_rule(self, rule_name: str, inputs: List[Param]) -> str:
        """
        Genera l'oggetto JSON per gli argomenti di un tool garantendo l'ordine canonico:
        1. Parametri obbligatori per primi in ordine di dichiarazione.
        2. Parametri opzionali dopo con virgola iniziale `("," ws ...)?`.
        3. Se tutti sono opzionali, raggruppamento disgiuntivo per evitare virgole orfane.
        """
        req_params = [p for p in inputs if p.default is None]
        opt_params = [p for p in inputs if p.default is not None]

        if not inputs:
            return f'{rule_name} ::= "{{" ws "}}"'

        def field_term(p: Param) -> str:
            ptype = self._map_type(p.type_expr)
            return f'"\\"{p.name}\\": " ws {ptype}'

        if req_params:
            req_terms = [field_term(p) for p in req_params]
            req_str = ' "," ws '.join(req_terms)
            opt_clauses = [f'("," ws {field_term(p)})?' for p in opt_params]
            all_parts = [req_str] + opt_clauses
            body = ' ws '.join(all_parts)
            return f'{rule_name} ::= "{{" ws {body} ws "}}"'
        else:
            n = len(opt_params)
            alts = []
            for k in range(n):
                first = field_term(opt_params[k])
                rest = [f'("," ws {field_term(opt_params[j])})?' for j in range(k + 1, n)]
                alt_parts = [first] + rest
                alts.append(' ws '.join(alt_parts))
            combined = ' | '.join(f'({alt})' for alt in alts)
            return f'{rule_name} ::= "{{" ws ({combined})? ws "}}"'

    def _build_canonical_agent_tool_call_rule(self, rule_name: str, tool_name: str, inputs: List[Param]) -> str:
        """Genera la regola GBNF per la chiamata DSL `call tool Tool(...)` in ordine canonico."""
        req_params = [p for p in inputs if p.default is None]
        opt_params = [p for p in inputs if p.default is not None]

        if not inputs:
            return f'{rule_name} ::= "call tool {tool_name}()"'

        def field_term(p: Param) -> str:
            ptype = self._map_type(p.type_expr)
            return f'"{p.name} = " ws ({ptype} | field_access)'

        if req_params:
            req_terms = [field_term(p) for p in req_params]
            req_str = ' "," ws '.join(req_terms)
            opt_clauses = [f'("," ws {field_term(p)})?' for p in opt_params]
            all_parts = [req_str] + opt_clauses
            body = ' ws '.join(all_parts)
            return f'{rule_name} ::= "call tool {tool_name}(" ws {body} ws ")"'
        else:
            n = len(opt_params)
            alts = []
            for k in range(n):
                first = field_term(opt_params[k])
                rest = [f'("," ws {field_term(opt_params[j])})?' for j in range(k + 1, n)]
                alt_parts = [first] + rest
                alts.append(' ws '.join(alt_parts))
            combined = ' | '.join(f'({alt})' for alt in alts)
            return f'{rule_name} ::= "call tool {tool_name}(" ws ({combined})? ws ")"'

    def _map_type(self, type_expr: Any) -> str:
        """
        Mappa tipi JANUS a regole GBNF senza fallback silenziosi.
        Solleva GBNFGenerationError se un tipo non è supportato o non trovato.
        """
        if type_expr is None:
            raise GBNFGenerationError("Tipo non specificato (None) in schema.")

        if isinstance(type_expr, PrimitiveType):
            name = type_expr.name
            if name == "str":
                return "string"
            elif name in ("i8", "i16", "i32", "i64", "int"):
                return "integer"
            elif name in ("f16", "f32", "f64", "bf16", "float"):
                return "number"
            elif name in ("bool", "boolean"):
                return "boolean"
            else:
                raise GBNFGenerationError(f"Tipo primitivo non supportato per GBNF: '{name}'")

        elif isinstance(type_expr, ListType):
            inner_rule = self._map_type(type_expr.inner)
            rule_name = f"{inner_rule}_list"
            if rule_name not in self._extra_rules:
                self._extra_rules[rule_name] = f'{rule_name} ::= "[" ws ({inner_rule} ("," ws {inner_rule})*)? ws "]"'
            return rule_name

        elif isinstance(type_expr, TensorType):
            dtype = type_expr.dtype
            if dtype in ("i8", "i16", "i32", "i64", "int"):
                return "integer_list"
            elif dtype in ("f16", "f32", "f64", "bf16", "float"):
                return "number_list"
            elif dtype == "str":
                return "string_list"
            elif dtype in ("bool", "boolean"):
                if "boolean_list" not in self._extra_rules:
                    self._extra_rules["boolean_list"] = 'boolean_list ::= "[" ws (boolean ("," ws boolean)*)? ws "]"'
                return "boolean_list"
            else:
                raise GBNFGenerationError(f"Tipo tensoriale con dtype non supportato per GBNF: '{dtype}'")

        elif isinstance(type_expr, CustomType):
            tname = type_expr.name
            if tname in self.type_decls:
                self._ensure_custom_type(tname, self.type_decls[tname])
                return tname
            else:
                raise GBNFGenerationError(f"CustomType non dichiarato o non trovato nei metadati: '{tname}'")

        else:
            raise GBNFGenerationError(f"Tipo non supportato per la generazione GBNF: {type_expr}")

    def _ensure_custom_type(self, type_name: str, type_decl: TypeDecl) -> None:
        if type_name in self._custom_rules:
            return
        self._custom_rules[type_name] = ""
        field_parts = []
        for f in type_decl.fields:
            ftype = self._map_type(f.type_expr)
            field_parts.append(f'"\\"{f.name}\\": " ws {ftype}')
        fields_str = ' "," ws '.join(field_parts)
        if fields_str:
            self._custom_rules[type_name] = f'{type_name} ::= "{{" ws {fields_str} ws "}}"'
        else:
            self._custom_rules[type_name] = f'{type_name} ::= "{{" ws "}}"'
