"""
Unit test e Fuzz test per il Lexer e la gestione errori CLI di JANUS.
"""
import unittest
import tempfile
import os
import argparse
from io import StringIO
from unittest.mock import patch
from hypothesis import given, strategies as st, settings

from janus.lexer import Lexer, TokenType, LexError
from janus.parser import Parser, ParseError
from janus.cli import cmd_compile, cmd_run, cmd_check, cmd_gbnf, cmd_export_jsonschema

class TestLexer(unittest.TestCase):
    def test_case_identification(self):
        code = "x:m w:b b:b y:n q:m k:b v:b buf:t loss.m"
        tokens = Lexer(code).tokenize()
        case_toks = [t for t in tokens if t.type == TokenType.CASE_IDENT]
        self.assertEqual(len(case_toks), 9)
        self.assertEqual(case_toks[0].case, "m")
        self.assertEqual(case_toks[0].base_name, "x")
        self.assertEqual(case_toks[1].case, "b")
        self.assertEqual(case_toks[1].base_name, "w")
        self.assertEqual(case_toks[7].case, "t")
        self.assertEqual(case_toks[7].base_name, "buf")
        self.assertEqual(case_toks[8].case, "m")
        self.assertEqual(case_toks[8].base_name, "loss")

        # Verifica che gli identificatori ordinari NON vengano troncati
        plain_code = "ab bb gb loss total dim param"
        plain_tokens = Lexer(plain_code).tokenize()
        ident_toks = [t for t in plain_tokens if t.type == TokenType.IDENT]
        self.assertEqual([t.value for t in ident_toks], ["ab", "bb", "gb", "loss", "total", "dim", "param"])

    def test_scientific_floats(self):
        code = "1e-5 1e-6 1e-8 2.5e+3 42"
        tokens = Lexer(code).tokenize()
        self.assertEqual(tokens[0].type, TokenType.LIT_FLOAT)
        self.assertEqual(tokens[0].value, "1e-5")
        self.assertEqual(tokens[1].type, TokenType.LIT_FLOAT)
        self.assertEqual(tokens[1].value, "1e-6")
        self.assertEqual(tokens[2].type, TokenType.LIT_FLOAT)
        self.assertEqual(tokens[2].value, "1e-8")
        self.assertEqual(tokens[3].type, TokenType.LIT_FLOAT)
        self.assertEqual(tokens[3].value, "2.5e+3")
        self.assertEqual(tokens[4].type, TokenType.LIT_INT)
        self.assertEqual(tokens[4].value, "42")

    def test_keywords_and_effects(self):
        code = "fn pure io stoc diff wrt matmul dot relu smax"
        tokens = Lexer(code).tokenize()
        expected = [
            TokenType.KW_FN, TokenType.EFF_PURE, TokenType.EFF_IO, TokenType.EFF_STOC,
            TokenType.OP_DIFF, TokenType.KW_WRT, TokenType.OP_MATMUL, TokenType.OP_DOT,
            TokenType.OP_RELU, TokenType.OP_SMAX
        ]
        self.assertEqual([t.type for t in tokens[:-1]], expected)

    def test_illegal_character_raises_lex_error(self):
        code = "fn foo() pure { let x = $10; }"
        with self.assertRaises(LexError) as ctx:
            Lexer(code).tokenize()
        self.assertIn("Carattere illegale", str(ctx.exception))
        self.assertEqual(ctx.exception.char, "$")

    def test_unclosed_string_newline_raises_lex_error(self):
        code = 'x = "unclosed string without closing quote\n y = 2'
        with self.assertRaises(LexError) as ctx:
            Lexer(code).tokenize()
        self.assertIn("Stringa non chiusa prima della fine della riga", str(ctx.exception))

    def test_unclosed_string_eof_raises_lex_error(self):
        code = 'x = "string at eof without closing'
        with self.assertRaises(LexError) as ctx:
            Lexer(code).tokenize()
        self.assertIn("Stringa non chiusa", str(ctx.exception))

    def test_malformed_number_multiple_dots_raises_lex_error(self):
        code = "val = 1.2.3"
        with self.assertRaises(LexError) as ctx:
            Lexer(code).tokenize()
        self.assertIn("punti decimali multipli", str(ctx.exception))

    def test_malformed_number_incomplete_exponent_raises_lex_error(self):
        code = "val = 1e"
        with self.assertRaises(LexError) as ctx:
            Lexer(code).tokenize()
        self.assertIn("attese cifre dopo l'esponente", str(ctx.exception))

        code2 = "val = 2.5e+"
        with self.assertRaises(LexError) as ctx2:
            Lexer(code2).tokenize()
        self.assertIn("attese cifre dopo l'esponente", str(ctx2.exception))

    def test_malformed_number_trailing_dot_raises_lex_error(self):
        code = "val = 1."
        with self.assertRaises(LexError) as ctx:
            Lexer(code).tokenize()
        self.assertIn("attese cifre dopo il punto decimale", str(ctx.exception))

    def test_range_operator_number_not_malformed(self):
        code = "for i in 1..10 { }"
        tokens = Lexer(code).tokenize()
        types = [t.type for t in tokens]
        self.assertIn(TokenType.LIT_INT, types)
        self.assertIn(TokenType.RANGE, types)

class TestCLIErrorHandling(unittest.TestCase):
    def test_cli_exit_code_2_on_io_error(self):
        args = argparse.Namespace(file="/path/to/definitely/non_existent_file.jn", json=False)
        with self.assertRaises(SystemExit) as ctx:
            with patch("sys.stderr", new=StringIO()):
                cmd_check(args)
        self.assertEqual(ctx.exception.code, 2)

    def test_cli_exit_code_1_on_lex_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jn", delete=False) as f:
            f.write("let x = $bad_character;")
            tmp_path = f.name
        try:
            args = argparse.Namespace(file=tmp_path, json=False, effects=False)
            with self.assertRaises(SystemExit) as ctx:
                with patch("sys.stderr", new=StringIO()):
                    cmd_check(args)
            self.assertEqual(ctx.exception.code, 1)
        finally:
            os.remove(tmp_path)

    def test_cli_exit_code_1_on_parse_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jn", delete=False) as f:
            f.write("fn { { {")
            tmp_path = f.name
        try:
            args = argparse.Namespace(file=tmp_path, json=False, effects=False)
            with self.assertRaises(SystemExit) as ctx:
                with patch("sys.stderr", new=StringIO()):
                    cmd_check(args)
            self.assertEqual(ctx.exception.code, 1)
        finally:
            os.remove(tmp_path)

    def test_cli_exit_code_0_on_success(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jn", delete=False) as f:
            f.write("fn ok() pure { ret 42; }")
            tmp_path = f.name
        try:
            args = argparse.Namespace(file=tmp_path, json=False, effects=False)
            with patch("sys.stdout", new=StringIO()):
                cmd_check(args)  # non solleva SystemExit, termina con 0
        finally:
            os.remove(tmp_path)

    def test_cli_compile_exit_code_1_on_syntax_error_no_traceback(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jn", delete=False) as f:
            f.write("fn syntax_error(")
            tmp_path = f.name
        try:
            args = argparse.Namespace(file=tmp_path, json=False, target="pytorch", output=None, quiet=True, ignore_warnings=False)
            with self.assertRaises(SystemExit) as ctx:
                with patch("sys.stderr", new=StringIO()) as fake_err:
                    cmd_compile(args)
                    self.assertIn("Errore di sintassi", fake_err.getvalue())
            self.assertEqual(ctx.exception.code, 1)
        finally:
            os.remove(tmp_path)

    def test_cli_run_exit_code_1_on_syntax_error_no_traceback(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jn", delete=False) as f:
            f.write("fn syntax_error(")
            tmp_path = f.name
        try:
            args = argparse.Namespace(file=tmp_path)
            with self.assertRaises(SystemExit) as ctx:
                with patch("sys.stderr", new=StringIO()):
                    cmd_run(args)
            self.assertEqual(ctx.exception.code, 1)
        finally:
            os.remove(tmp_path)

# =========================================================================
# Fuzz Testing con Hypothesis: garantisce l'assenza di crash non gestiti
# =========================================================================
@given(st.text(max_size=300))
@settings(max_examples=150, deadline=None)
def test_fuzz_lexer_and_parser(random_code):
    try:
        tokens = Lexer(random_code).tokenize()
        Parser(tokens).parse()
    except (LexError, ParseError):
        # Rifiuto corretto di testo casuale non conforme
        pass

if __name__ == "__main__":
    unittest.main()
