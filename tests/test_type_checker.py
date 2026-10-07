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

    def test_transitive_effect_direct_tool_call_violation(self):
        code = """
        schema ReadFile [io] {
            path: str
        } -> {
            content: str
        }

        fn read_pure(p: str) pure {
            res = call tool ReadFile(path = p)
            ret res
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_EFFECT_PURITY_VIOLATION")
        self.assertIn("ReadFile", diags[0].message)

    def test_transitive_effect_indirect_violation(self):
        code = """
        schema FetchDoc [io] {
            query: str
        } -> {
            result: str
        }

        fn fetch_impl(q: str) io {
            ans = call tool FetchDoc(query = q)
            ret ans
        }

        fn pure_caller(q: str) pure {
            w = fetch_impl(q)
            ret w
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_EFFECT_PURITY_VIOLATION")
        self.assertIn("fetch_impl", diags[0].message)
        self.assertIn("effetto transitivo 'io'", diags[0].message)

    def test_transitive_effect_deep_chain(self):
        code = """
        schema RemoteLog [io] {
            msg: str
        } -> {
            ok: bool
        }

        fn step3(m: str) io {
            r = call tool RemoteLog(msg = m)
            ret r
        }

        fn step2(m: str) io {
            ret step3(m)
        }

        fn step1(m: str) io {
            ret step2(m)
        }

        fn pure_top(m: str) pure {
            res = step1(m)
            ret res
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_EFFECT_PURITY_VIOLATION")
        self.assertIn("step1", diags[0].message)

    def test_transitive_effect_nested_call_violation(self):
        code = """
        schema FetchValue [io] {
            k: str
        } -> {
            v: int
        }

        fn impure_source(k: str) io {
            r = call tool FetchValue(k = k)
            ret r
        }

        fn pure_math(x: int) pure {
            ret x * 2
        }

        fn pure_wrapper(k: str) pure {
            out = pure_math(impure_source(k))
            ret out
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_EFFECT_PURITY_VIOLATION")
        self.assertIn("impure_source", diags[0].message)

    def test_transitive_effect_mutual_recursion(self):
        code = """
        schema FetchNode [io] {
            id: int
        } -> {
            data: str
        }

        fn ping(n: int) io {
            if n > 0 {
                r = pong(n - 1)
                ret r
            }
            ret "done"
        }

        fn pong(n: int) io {
            doc = call tool FetchNode(id = n)
            p = ping(n)
            ret doc
        }

        fn pure_caller(n: int) pure {
            res = ping(n)
            ret res
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_EFFECT_PURITY_VIOLATION")
        self.assertIn("ping", diags[0].message)

    def test_transitive_effect_pure_calls_pure_success(self):
        code = """
        fn double(x: int) -> int pure {
            ret x * 2
        }

        fn increment(x: int) -> int pure {
            ret x + 1
        }

        fn compute_pipeline(x: int) -> int pure {
            d = double(x)
            i = increment(d)
            ret i
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 0)

    def test_transitive_effect_stochastic_violation(self):
        code = """
        schema RollDice [stoc] {
            sides: int
        } -> {
            roll: int
        }

        fn random_worker(s: int) stoc {
            r = call tool RollDice(sides = s)
            ret r
        }

        fn pure_user(s: int) pure {
            res = random_worker(s)
            ret res
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_EFFECT_PURITY_VIOLATION")
        self.assertIn("random_worker", diags[0].message)
        self.assertIn("effetto transitivo 'stoc'", diags[0].message)

if __name__ == "__main__":
    unittest.main()

