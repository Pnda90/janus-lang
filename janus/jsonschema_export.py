"""
Modulo di esportazione per JSON Schema da schemi tipizzati JANUS (Fase 2).
Fornisce funzioni per serializzare definizioni SchemaDecl verso JSON Schema Draft-07
sia per la chiamata a tool (mode="call") che per l'output strutturato (mode="output").
"""

from typing import Dict, Any, Optional, List
from janus.ast_nodes import (
    SchemaDecl, TypeDecl, PrimitiveType, TensorType, CustomType, ListType, TypeExpr
)


def type_to_json_schema(type_expr: Optional[TypeExpr], type_decls: Optional[Dict[str, TypeDecl]] = None) -> Dict[str, Any]:
    """Mappa un tipo formale JANUS in una specifica JSON Schema valida."""
    if type_expr is None:
        return {}

    if isinstance(type_expr, PrimitiveType):
        name = type_expr.name
        if name == "str":
            return {"type": "string"}
        elif name in ("i8", "i16", "i32", "i64", "int"):
            return {"type": "integer"}
        elif name in ("f16", "f32", "f64", "bf16", "float"):
            return {"type": "number"}
        elif name in ("bool", "boolean"):
            return {"type": "boolean"}
        else:
            raise ValueError(f"Tipo primitivo non supportato per JSON Schema: '{name}'")

    elif isinstance(type_expr, ListType):
        return {
            "type": "array",
            "items": type_to_json_schema(type_expr.inner, type_decls)
        }

    elif isinstance(type_expr, TensorType):
        dtype = type_expr.dtype
        if dtype == "str":
            item_schema = {"type": "string"}
        elif dtype in ("i8", "i16", "i32", "i64", "int"):
            item_schema = {"type": "integer"}
        elif dtype in ("bool", "boolean"):
            item_schema = {"type": "boolean"}
        else:
            item_schema = {"type": "number"}
        return {
            "type": "array",
            "items": item_schema
        }

    elif isinstance(type_expr, CustomType):
        tname = type_expr.name
        if type_decls and tname in type_decls:
            decl = type_decls[tname]
            props = {}
            req = []
            for f in decl.fields:
                props[f.name] = type_to_json_schema(f.type_expr, type_decls)
                req.append(f.name)
            return {
                "type": "object",
                "title": tname,
                "properties": props,
                "required": req,
                "additionalProperties": False
            }
        return {
            "type": "object",
            "title": tname,
            "additionalProperties": False
        }

    else:
        raise ValueError(f"Tipo non supportato per JSON Schema: {type_expr}")


def schema_to_json_schema(
    schema: SchemaDecl,
    mode: str = "call",
    type_decls: Optional[Dict[str, TypeDecl]] = None
) -> Dict[str, Any]:
    """
    Converte uno SchemaDecl JANUS in un JSON Schema formale (Draft-07).
    
    Argomenti:
        schema: Dichiarazione dello schema JANUS
        mode: "call" (chiamata al tool {"tool": ..., "args": ...}) oppure "output" (valori di ritorno)
        type_decls: Dizionario opzionale di TypeDecl per risolvere CustomType
    """
    if mode not in ("call", "output"):
        raise ValueError(f"Modalità non valida '{mode}': atteso 'call' o 'output'")

    if mode == "call":
        args_props: Dict[str, Any] = {}
        args_required: List[str] = []

        for p in schema.inputs:
            args_props[p.name] = type_to_json_schema(p.type_expr, type_decls)
            if p.default is None:
                args_required.append(p.name)

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": f"{schema.name}Call",
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "enum": [schema.name]
                },
                "args": {
                    "type": "object",
                    "properties": args_props,
                    "required": args_required,
                    "additionalProperties": False
                }
            },
            "required": ["tool", "args"],
            "additionalProperties": False
        }

    else:  # mode == "output"
        out_props: Dict[str, Any] = {}
        out_required: List[str] = []

        for o in schema.outputs:
            out_props[o.name] = type_to_json_schema(o.type_expr, type_decls)
            out_required.append(o.name)

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": f"{schema.name}Output",
            "type": "object",
            "properties": out_props,
            "required": out_required,
            "additionalProperties": False
        }
