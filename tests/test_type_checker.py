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
    def test_tool_arg_type_mismatch_primitive(self):
        code = """
        schema ConfigTool [pure] {
            count: i32,
            query: str
        } -> {
            ok: bool
        }

        fn run_bad() pure {
            r = call tool ConfigTool(count = "not_a_number", query = 123)
            ret r
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 2)
        self.assertTrue(all(d.code == "ERR_TOOL_ARG_TYPE_MISMATCH" for d in diags))

    def test_tool_arg_widening_int_to_float_allowed(self):
        code = """
        schema MetricTool [pure] {
            threshold: f32
        } -> {
            valid: bool
        }

        fn run_widen() pure {
            r = call tool MetricTool(threshold = 5)
            ret r
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 0)

    def test_tool_arg_narrowing_float_to_int_rejected(self):
        code = """
        schema IntTool [pure] {
            counter: i32
        } -> {
            valid: bool
        }

        fn run_narrow() pure {
            r = call tool IntTool(counter = 3.14)
            ret r
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_TOOL_ARG_TYPE_MISMATCH")
        self.assertIn("atteso 'i32', ottenuto 'float'", diags[0].message)

    def test_tool_arg_type_mismatch_list(self):
        code = """
        schema ListTool [pure] {
            tags: [str]
        } -> {
            count: i32
        }

        fn run_list() pure {
            r = call tool ListTool(tags = [1, 2, 3])
            ret r
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_TOOL_ARG_TYPE_MISMATCH")

    def test_tool_arg_gradual_typing_no_false_positive(self):
        code = """
        schema SearchTool [pure] {
            q: str
        } -> {
            ok: bool
        }

        fn untyped_producer(x) pure {
            ret x
        }

        fn run_gradual(v) pure {
            r = call tool SearchTool(q = untyped_producer(v))
            ret r
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 0)

    def test_tool_positional_arg_type_mismatch(self):
        code = """
        schema TwoParams [pure] {
            num: i32,
            label: str
        } -> {
            ok: bool
        }

        fn run_pos() pure {
            r = call tool TwoParams("wrong", 99)
            ret r
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 2)
        self.assertTrue(all(d.code == "ERR_TOOL_ARG_TYPE_MISMATCH" for d in diags))

    def test_tool_duplicate_named_args(self):
        code = """
        schema QueryTool [pure] {
            query: str
        } -> {
            ok: bool
        }

        fn run_dup() pure {
            r = call tool QueryTool(query = "first", query = "second")
            ret r
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_DUPLICATE_TOOL_ARGUMENT")

    def test_tool_duplicate_positional_and_named_args(self):
        code = """
        schema StepTool [pure] {
            step: i32
        } -> {
            ok: bool
        }

        fn run_pos_named_dup() pure {
            r = call tool StepTool(10, step = 20)
            ret r
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_DUPLICATE_TOOL_ARGUMENT")
        self.assertIn("passato sia posizionalmente che per nome", diags[0].message)

    def test_tool_positional_arity_error(self):
        code = """
        schema SingleArg [pure] {
            x: i32
        } -> {
            ok: bool
        }

        fn run_too_many() pure {
            r = call tool SingleArg(1, 2, 3)
            ret r
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_TOOL_ARITY")

    def test_output_field_access_success_and_propagation(self):
        code = """
        schema DatabaseFetch [io] {
            id: i32
        } -> {
            count: i32,
            name: str
        }

        schema NextTool [io] {
            c: i32,
            n: str
        } -> {
            done: bool
        }

        fn pipeline(doc_id: i32) io {
            res = call tool DatabaseFetch(id = doc_id)
            c = res.count
            n = res.name
            out = call tool NextTool(c = c, n = n)
            ret out
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 0)

    def test_output_field_access_unknown_field_error(self):
        code = """
        schema GetRecord [io] {
            id: i32
        } -> {
            data: str
        }

        fn test_unknown_field(i: i32) io {
            rec = call tool GetRecord(id = i)
            bad = rec.non_existent_field
            ret bad
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_UNKNOWN_OUTPUT_FIELD")
        self.assertIn("non_existent_field", diags[0].message)

    def test_output_field_propagated_type_mismatch(self):
        code = """
        schema SourceTool [pure] {
            x: i32
        } -> {
            text_result: str
        }

        schema ExpectsInt [pure] {
            val: i32
        } -> {
            ok: bool
        }

        fn mismatch_pipeline(a: i32) pure {
            res = call tool SourceTool(x = a)
            out = call tool ExpectsInt(val = res.text_result)
            ret out
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_TOOL_ARG_TYPE_MISMATCH")
        self.assertIn("atteso 'i32', ottenuto 'str'", diags[0].message)

    def test_effect_lattice_stoc_calling_io_tool_escalation(self):
        code = """
        schema WriteFile [io] {
            path: str
        } -> {
            ok: bool
        }

        fn bad_stoc_tool(p: str) stoc {
            r = call tool WriteFile(path = p)
            ret r
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_EFFECT_ESCALATION")
        self.assertIn("non può invocare il tool 'io' 'WriteFile'", diags[0].message)

    def test_effect_lattice_stoc_calling_io_func_escalation(self):
        code = """
        schema HttpGet [io] {
            url: str
        } -> {
            body: str
        }

        fn io_helper(u: str) io {
            r = call tool HttpGet(url = u)
            ret r
        }

        fn stoc_caller(u: str) stoc {
            res = io_helper(u)
            ret res
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_EFFECT_ESCALATION")
        self.assertIn("non può invocare la funzione 'io' 'io_helper'", diags[0].message)

    def test_effect_lattice_stoc_calling_agent_escalation(self):
        code = """
        fn bad_stoc_agent(prompt: str) stoc {
            ans = call agentv LLM prompt:m prompt
            ret ans
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_EFFECT_ESCALATION")

    def test_effect_lattice_stoc_calling_pure_or_stoc_allowed(self):
        code = """
        schema RandomChoice [stoc] {
            k: i32
        } -> {
            choice: i32
        }

        fn pure_calc(x: i32) pure {
            ret x + 1
        }

        fn valid_stoc(k: i32) stoc {
            c = call tool RandomChoice(k = k)
            p = pure_calc(c.choice)
            ret p
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 0)

    def test_undefined_function_call_error(self):
        code = """
        fn test_call() pure {
            r = non_existent_function(123)
            ret r
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        diags = TypeChecker().check(ast)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "ERR_UNDEFINED_FUNCTION")
        self.assertIn("non_existent_function", diags[0].message)

    def test_effects_summary_computation(self):
        code = """
        schema ReadDisk [io] {
            f: str
        } -> {
            data: str
        }

        schema CoinFlip [stoc] {
            bias: f32
        } -> {
            heads: bool
        }

        fn pure_math(a: i32) pure {
            ret a * 2
        }

        fn stoc_action(b: f32) stoc {
            r = call tool CoinFlip(bias = b)
            ret r
        }

        fn io_action(file: str) io {
            d = call tool ReadDisk(f = file)
            ret d
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        checker = TypeChecker()
        diags = checker.check(ast)
        self.assertEqual(len(diags), 0)

        summary = checker.get_effects_summary()
        summary_by_fn = {s["function"]: s for s in summary}

        self.assertEqual(summary_by_fn["pure_math"]["declared_effect"], "pure")
        self.assertEqual(summary_by_fn["pure_math"]["computed_effect"], "pure")
        self.assertEqual(summary_by_fn["pure_math"]["reachable_tools"], [])

        self.assertEqual(summary_by_fn["stoc_action"]["declared_effect"], "stoc")
        self.assertEqual(summary_by_fn["stoc_action"]["computed_effect"], "stoc")
        self.assertEqual(summary_by_fn["stoc_action"]["reachable_tools"], ["CoinFlip"])

        self.assertEqual(summary_by_fn["io_action"]["declared_effect"], "io")
        self.assertEqual(summary_by_fn["io_action"]["computed_effect"], "io")
        self.assertEqual(summary_by_fn["io_action"]["reachable_tools"], ["ReadDisk"])

    def test_cli_check_effects_plain_text(self):
        import tempfile
        import os
        from io import StringIO
        import sys
        from unittest.mock import patch
        from janus.cli import cmd_check
        import argparse

        code = """
        schema FetchNode [io] {
            id: i32
        } -> {
            data: str
        }

        fn fetch_fn(i: i32) io {
            r = call tool FetchNode(id = i)
            ret r
        }

        fn calc(x: i32) pure {
            ret x + 1
        }
        """
        with tempfile.NamedTemporaryFile("w", suffix=".jn", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            args = argparse.Namespace(file=tmp_path, json=False, effects=True)
            with patch("sys.stdout", new=StringIO()) as fake_out:
                cmd_check(args)
                output = fake_out.getvalue()
                self.assertIn("Tabella degli Effetti", output)
                self.assertIn("fetch_fn", output)
                self.assertIn("FetchNode", output)
                self.assertIn("calc", output)
        finally:
            os.remove(tmp_path)

    def test_cli_check_effects_json_output(self):
        import tempfile
        import os
        import json
        from io import StringIO
        from unittest.mock import patch
        from janus.cli import cmd_check
        import argparse

        code = """
        schema ReadConfig [io] {
            path: str
        } -> {
            val: str
        }

        fn load_cfg(p: str) io {
            r = call tool ReadConfig(path = p)
            ret r
        }
        """
        with tempfile.NamedTemporaryFile("w", suffix=".jn", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            args = argparse.Namespace(file=tmp_path, json=True, effects=True)
            with patch("sys.stdout", new=StringIO()) as fake_out:
                cmd_check(args)
                raw = fake_out.getvalue()
                parsed = json.loads(raw)
                self.assertIn("effects", parsed)
                self.assertIn("diagnostics", parsed)
                self.assertTrue(parsed["valid"])
                effects = parsed["effects"]
                self.assertEqual(len(effects), 1)
                self.assertEqual(effects[0]["function"], "load_cfg")
                self.assertEqual(effects[0]["declared_effect"], "io")
                self.assertEqual(effects[0]["computed_effect"], "io")
                self.assertEqual(effects[0]["reachable_tools"], ["ReadConfig"])
        finally:
            os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()

