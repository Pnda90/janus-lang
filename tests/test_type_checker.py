"""
Unit test per il Type Checker e Analisi Semantica di JANUS.
"""
import unittest
from janus.lexer import Lexer
from janus.parser import Parser
from janus.type_checker import TypeChecker

class TestTypeChecker(unittest.TestCase):
    def test_immutable_reassignment_error(self):
        code = """
        fn bad_mutate(x:m) pure {
            val = x:m * 2
            val = val + 1
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        checker = TypeChecker()
        diags = checker.check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_AFFINE_IMMUTABLE_MUTATION")
        self.assertIsNotNone(diags[0].patch)
        self.assertEqual(diags[0].patch.replacement, "mut val =")

    def test_mutable_reassignment_success(self):
        code = """
        fn good_mutate(x:m) pure {
            mut val = x:m * 2
            val = val + 1
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        checker = TypeChecker()
        diags = checker.check(ast)
        self.assertEqual(len(diags), 0)

    def test_effect_purity_violation(self):
        code = """
        fn bad_call(query:m: str) pure {
            ans = call agentv LLM prompt:m query:m
            ret ans
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        checker = TypeChecker()
        diags = checker.check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_EFFECT_PURITY_VIOLATION")

if __name__ == "__main__":
    unittest.main()
