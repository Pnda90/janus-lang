"""
Unit test per il Lexer di JANUS.
"""
import unittest
from janus.lexer import Lexer, TokenType

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

if __name__ == "__main__":
    unittest.main()
