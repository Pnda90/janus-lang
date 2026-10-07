"""
Validatore formale per grammatiche GBNF e vincoli JSON di JANUS.
Implementa:
1. Verifica sintattica e strutturale delle regole GBNF (llama.cpp compliant).
2. Valutatore accept/reject per output JSON generati tramite decoding vincolato.
"""

import re
import json
from typing import Tuple, Dict, Any, List
from janus.ast_nodes import SchemaDecl, PrimitiveType, TensorType


class GBNFValidationError(Exception):
    pass


class GBNFValidator:
    """Valida la correttezza formale delle grammatiche GBNF e la conformità degli output JSON."""

    @staticmethod
    def validate_grammar_syntax(gbnf_text: str) -> bool:
        """
        Verifica la validità sintattica del testo GBNF:
        - Ogni regola ha la forma `name ::= expr`
        - Stringhe delimitate correttamente senza doppi apici vuoti `""` consecutivi
        - Tutte le regole referenziate devono essere definite nella grammatica
        """
        lines = [line.strip() for line in gbnf_text.splitlines() if line.strip() and not line.strip().startswith("#")]
        if not lines:
            raise GBNFValidationError("La grammatica GBNF è vuota.")

        defined_rules = set()
        rule_dependencies: Dict[str, List[str]] = {}

        rule_regex = re.compile(r"^([a-zA-Z0-9_-]+)\s*::=\s*(.+)$")

        for line in lines:
            m = rule_regex.match(line)
            if not m:
                raise GBNFValidationError(f"Sintassi regola GBNF non valida: '{line}'")
            rule_name = m.group(1)
            expr = m.group(2)

            # Verifica assenza di doppi apici vuoti/malformati
            if '""' in expr:
                raise GBNFValidationError(f"Doppi apici adiacenti non validi nella regola '{rule_name}': {expr}")

            # Verifica bilanciamento parentesi
            paren_count = 0
            in_quote = False
            in_bracket = False
            i = 0
            while i < len(expr):
                c = expr[i]
                if c == '\\':
                    i += 2
                    continue
                if c == '[' and not in_quote:
                    in_bracket = True
                elif c == ']' and not in_quote:
                    in_bracket = False
                elif c == '"' and not in_bracket:
                    in_quote = not in_quote
                elif not in_quote and not in_bracket:
                    if c == '(':
                        paren_count += 1
                    elif c == ')':
                        paren_count -= 1
                        if paren_count < 0:
                            raise GBNFValidationError(f"Parentesi chiusa non bilanciata nella regola '{rule_name}'")
                i += 1

            if paren_count != 0:
                raise GBNFValidationError(f"Parentesi non bilanciate nella regola '{rule_name}'")
            if in_bracket:
                raise GBNFValidationError(f"Classe di caratteri non chiusa nella regola '{rule_name}'")
            if in_quote:
                raise GBNFValidationError(f"Stringa letterale non chiusa nella regola '{rule_name}'")

            defined_rules.add(rule_name)

            # Estrai identificatori di non-terminali referenziati
            # Rimuovi prima le classi di caratteri [ ... ] (che possono contenere apici come ["])
            # e poi le stringhe letterali " ... "
            clean_expr = re.sub(r'\[(?:\\.|[^\]])*\]', ' ', expr)
            clean_expr = re.sub(r'"(?:\\.|[^"])*"', ' ', clean_expr)
            tokens = re.findall(r'\b[a-zA-Z0-9_-]+\b', clean_expr)
            rule_dependencies[rule_name] = tokens

        # Verifica che 'root' sia definita
        if "root" not in defined_rules:
            raise GBNFValidationError("La grammatica GBNF deve definire la regola iniziale 'root'")

        # Verifica che ogni simbolo non-terminale referenziato sia definito
        for rule, deps in rule_dependencies.items():
            for dep in deps:
                if dep not in defined_rules:
                    raise GBNFValidationError(f"Regola non definita '{dep}' referenziata in '{rule}'")

        return True

    @staticmethod
    def reference_validate_json_against_schema(schema: SchemaDecl, json_text: str, with_status: bool = False) -> Tuple[bool, str]:
        """
        Validatore di riferimento in Python per verificare se un payload JSON rispetta i campi di uno schema JANUS.
        
        NOTA METODOLOGICA:
        Questo è un controllo euristico di riferimento in Python, NON il motore di riconoscimento della grammatica GBNF.
        Per la verifica formale del riconoscimento sintattico della grammatica GBNF generata,
        utilizzare il recognizer dedicato `tests/support/gbnf_engine.py`.
        """
        try:
            data = json.loads(json_text)
        except Exception as e:
            return False, f"JSON non valido sintatticamente: {e}"

        if not isinstance(data, dict):
            return False, "La radice deve essere un oggetto JSON dict"

        expected_fields = set()
        if with_status:
            if data.get("status") != "ok":
                return False, f"Manca campo obbligatorio status='ok' (trovato: {data.get('status')})"
            expected_fields.add("status")

        # Verifica tutti i campi definiti in outputs
        for out in schema.outputs:
            fname = out.name
            expected_fields.add(fname)
            if fname not in data:
                return False, f"Campo obbligatorio mancante: '{fname}'"

            val = data[fname]
            texpr = out.type_expr

            if isinstance(texpr, PrimitiveType):
                if texpr.name == "str":
                    if not isinstance(val, str):
                        return False, f"Campo '{fname}' atteso str, trovato {type(val).__name__}"
                elif texpr.name in ("f32", "f16", "f64", "bf16"):
                    if not isinstance(val, (int, float)) or isinstance(val, bool):
                        return False, f"Campo '{fname}' atteso float/number, trovato {type(val).__name__}"
                elif texpr.name in ("i32", "i64", "i8", "i16"):
                    if not isinstance(val, int) or isinstance(val, bool):
                        return False, f"Campo '{fname}' atteso int, trovato {type(val).__name__}"
                elif texpr.name == "bool":
                    if not isinstance(val, bool):
                        return False, f"Campo '{fname}' atteso bool, trovato {type(val).__name__}"

            elif isinstance(texpr, TensorType):
                if not isinstance(val, list):
                    return False, f"Campo '{fname}' atteso lista, trovato {type(val).__name__}"
                if texpr.dtype == "str":
                    if not all(isinstance(item, str) for item in val):
                        return False, f"Tutti gli elementi di '{fname}' devono essere str"
                elif texpr.dtype in ("i32", "i64", "i8", "i16"):
                    if not all(isinstance(item, int) and not isinstance(item, bool) for item in val):
                        return False, f"Tutti gli elementi di '{fname}' devono essere int"
                else:
                    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in val):
                        return False, f"Tutti gli elementi di '{fname}' devono essere numeri"

        # Campi extra non ammessi dalla grammatica vincolata
        extra_fields = set(data.keys()) - expected_fields
        if extra_fields:
            return False, f"Campi inattesi non ammessi dalla grammatica: {extra_fields}"

        return True, "OK"

    # Alias retrocompatibile
    validate_json_against_schema = reference_validate_json_against_schema

