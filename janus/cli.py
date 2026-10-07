"""
Command-Line Interface (CLI) per JANUS (janusc).
Fornisce comandi per compilare, eseguire, verificare la correttezza di tipo,
generare grammatiche GBNF e analizzare il conteggio dei token.
"""

import sys
import os
import argparse
import json
from typing import List

from janus.lexer import Lexer
from janus.parser import Parser, ParseError
from janus.type_checker import TypeChecker
from janus.codegen import CodeGenerator
from janus.gbnf_gen import GBNFGenerator, GBNFGenerationError
from janus.ast_nodes import SchemaDecl

def cmd_compile(args):
    source_path = args.file
    if not os.path.exists(source_path):
        print(f"Errore: File non trovato '{source_path}'", file=sys.stderr)
        sys.exit(1)

    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        
        # Type Check
        checker = TypeChecker()
        diags = checker.check(ast)
        if diags and not args.ignore_warnings:
            if args.json:
                print(json.dumps([d.to_dict() for d in diags], indent=2))
            else:
                for d in diags:
                    print(d.to_cli(source.splitlines()), file=sys.stderr)
            if any(d.code.startswith("ERR") for d in diags):
                sys.exit(1)

        generator = CodeGenerator(target=args.target)
        output_code = generator.generate(ast)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as out_f:
                out_f.write(output_code)
            if not args.quiet:
                print(f"Compilazione completata con successo: {args.output}")
        else:
            print(output_code)

    except ParseError as pe:
        if args.json:
            diag = {
                "status": "error",
                "code": "ERR_SYNTAX",
                "phase": "parser",
                "message": pe.message,
                "span": {"line": pe.token.line, "col": pe.token.col, "len": pe.token.length},
                "offending": pe.token.value
            }
            print(json.dumps(diag, indent=2))
        else:
            print(f"Errore di sintassi [{pe.token.line}:{pe.token.col}]: {pe.message} (token: '{pe.token.value}')", file=sys.stderr)
        sys.exit(1)

def cmd_run(args):
    source_path = args.file
    if not os.path.exists(source_path):
        print(f"Errore: File non trovato '{source_path}'", file=sys.stderr)
        sys.exit(1)

    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        
        checker = TypeChecker()
        diags = checker.check(ast)
        if any(d.code.startswith("ERR") for d in diags):
            for d in diags:
                print(d.to_cli(source.splitlines()), file=sys.stderr)
            sys.exit(1)

        generator = CodeGenerator()
        py_code = generator.generate(ast)
        
        # Esecuzione immediata in runtime Python
        env = {}
        exec(py_code, env)

    except Exception as e:
        print(f"Errore a runtime: {e}", file=sys.stderr)
        sys.exit(1)

def cmd_check(args):
    source_path = args.file
    if not os.path.exists(source_path):
        print(f"Errore: File non trovato '{source_path}'", file=sys.stderr)
        sys.exit(1)

    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        checker = TypeChecker()
        diags = checker.check(ast)

        if args.json:
            print(json.dumps([d.to_dict() for d in diags], indent=2))
        else:
            if not diags:
                print(f"Controllo semantico superato con successo: 0 errori ({source_path})")
            else:
                for d in diags:
                    print(d.to_cli(source.splitlines()), file=sys.stderr)
                if any(d.code.startswith("ERR") for d in diags):
                    sys.exit(1)

    except ParseError as pe:
        if args.json:
            diag = {
                "status": "error",
                "code": "ERR_SYNTAX",
                "phase": "parser",
                "message": pe.message,
                "span": {"line": pe.token.line, "col": pe.token.col, "len": pe.token.length},
                "offending": pe.token.value
            }
            print(json.dumps([diag], indent=2))
        else:
            print(f"Errore di sintassi [{pe.token.line}:{pe.token.col}]: {pe.message}", file=sys.stderr)
        sys.exit(1)

def cmd_gbnf(args):
    source_path = args.file
    json_mode = getattr(args, "json", False)
    if not os.path.exists(source_path):
        if json_mode:
            print(json.dumps([{"status": "error", "code": "ERR_IO", "phase": "cli", "message": f"File non trovato: '{source_path}'"}], indent=2))
        else:
            print(f"Errore: File non trovato '{source_path}'", file=sys.stderr)
        sys.exit(1)

    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
    except ParseError as pe:
        if json_mode:
            diag = {
                "status": "error",
                "code": "ERR_SYNTAX",
                "phase": "parser",
                "message": pe.message,
                "span": {"line": pe.token.line, "col": pe.token.col, "len": pe.token.length},
                "offending": pe.token.value
            }
            print(json.dumps([diag], indent=2))
        else:
            print(f"Errore di sintassi [{pe.token.line}:{pe.token.col}]: {pe.message}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if json_mode:
            diag = {"status": "error", "code": "ERR_PARSER", "phase": "parser", "message": str(e)}
            print(json.dumps([diag], indent=2))
        else:
            print(f"Errore durante l'analisi: {e}", file=sys.stderr)
        sys.exit(1)

    from janus.ast_nodes import TypeDecl
    schemas = [d for d in ast.declarations if isinstance(d, SchemaDecl)]
    type_decls = {d.name: d for d in ast.declarations if isinstance(d, TypeDecl)}
    gbnf_gen = GBNFGenerator(type_decls=type_decls)

    mode = getattr(args, "mode", "call")
    with_status = getattr(args, "with_status", False)

    try:
        if mode == "agent":
            print(gbnf_gen.generate_agent_grammar(schemas))
        elif mode == "output":
            if not schemas:
                print(f"# Nessuna dichiarazione 'schema' trovata in {source_path}")
                return
            outputs = [gbnf_gen.generate_for_schema(s, with_status=with_status) for s in schemas]
            print(("\n" + "=" * 50 + "\n").join(outputs))
        else:  # mode == "call"
            if not schemas:
                print(f"# Nessuna dichiarazione 'schema' trovata in {source_path}")
                return
            if len(schemas) == 1:
                print(gbnf_gen.generate_tool_json_call_grammar(schemas[0]))
            else:
                print(gbnf_gen.generate_tool_json_call_grammar(schemas))
    except GBNFGenerationError as ge:
        if json_mode:
            diag = {"status": "error", "code": "ERR_GBNF_GEN", "phase": "gbnf_gen", "message": str(ge)}
            print(json.dumps([diag], indent=2))
        else:
            print(f"Errore generazione GBNF: {ge}", file=sys.stderr)
        sys.exit(1)

def cmd_tokens(args):
    source_path = args.file
    if not os.path.exists(source_path):
        print(f"Errore: File non trovato '{source_path}'", file=sys.stderr)
        sys.exit(1)

    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    import re
    # Analisi dei token
    tokens = re.findall(r"[a-zA-Z_]+|[0-9]+|[:\.\,\;\(\)\[\]\{\}\=\+\-\*\/\@\>\<\_\~]|\s+", source)
    non_empty = [t for t in tokens if t.strip() or t == '\n']
    
    print(f"Analisi dei Token per: {source_path}")
    print(f"  Lunghezza caratteri: {len(source)}")
    print(f"  Righe di codice (LoC): {len([l for l in source.splitlines() if l.strip()])}")
    print(f"  Token BPE stimati:     {len(non_empty)}")

def main():
    parser = argparse.ArgumentParser(
        prog="janusc",
        description="JANUS Compiler & Toolchain: Token-Efficient, AI-Native Language"
    )
    subparsers = parser.add_subparsers(dest="command", help="Comando da eseguire")

    # compile
    p_comp = subparsers.add_parser("compile", help="Compila un sorgente .jn in Python/PyTorch")
    p_comp.add_argument("file", help="File sorgente .jn")
    p_comp.add_argument("-o", "--output", help="File Python di destinazione (.py)")
    p_comp.add_argument("--target", default="pytorch", choices=["pytorch", "numpy"], help="Backend target")
    p_comp.add_argument("--json", action="store_true", help="Emetti diagnostica in formato JSON")
    p_comp.add_argument("--quiet", action="store_true", help="Sopprimi messaggi informativi")
    p_comp.add_argument("--ignore-warnings", action="store_true", help="Ignora i warning")

    # run
    p_run = subparsers.add_parser("run", help="Compila ed esegue un file .jn immediatamente")
    p_run.add_argument("file", help="File sorgente .jn")

    # check
    p_check = subparsers.add_parser("check", help="Esegue type-checking e analisi semantica")
    p_check.add_argument("file", help="File sorgente .jn")
    p_check.add_argument("--json", action="store_true", help="Emetti diagnosi in formato JSON per LLM")

    # gbnf
    p_gbnf = subparsers.add_parser("gbnf", help="Genera grammatica GBNF per decodifica vincolata da uno schema")
    p_gbnf.add_argument("file", help="File sorgente .jn contenente dichiarazioni schema")
    p_gbnf.add_argument("--mode", default="call", choices=["call", "output", "agent"], help="Modalità di generazione: call (default, chiamata tool JSON), output (risultato tool JSON), agent (piano multi-tool DSL)")
    p_gbnf.add_argument("--with-status", action="store_true", help="Includi il campo 'status': 'ok' nella grammatica di output")
    p_gbnf.add_argument("--json", action="store_true", help="Emetti diagnostiche di errore in formato JSON")

    # tokens
    p_tok = subparsers.add_parser("tokens", help="Analizza la densità di token del sorgente")
    p_tok.add_argument("file", help="File sorgente .jn")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "compile": cmd_compile,
        "run": cmd_run,
        "check": cmd_check,
        "gbnf": cmd_gbnf,
        "tokens": cmd_tokens,
    }
    dispatch[args.command](args)

if __name__ == "__main__":
    main()
