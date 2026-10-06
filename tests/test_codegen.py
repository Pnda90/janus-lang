"""
Unit test per il Code Generator e Transpiler di JANUS.
"""
import unittest
from janus.lexer import Lexer
from janus.parser import Parser
from janus.codegen import CodeGenerator

class TestCodegen(unittest.TestCase):
    def test_transpile_and_exec_simple(self):
        code = """
        type Point { x: f32, y: f32 }

        fn add_coords(p1: Point, p2: Point) pure {
            ret p1.x + p2.x
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        gen = CodeGenerator()
        py_code = gen.generate(ast)
        
        env = {}
        exec(py_code, env)
        Point = env["Point"]
        add_coords = env["add_coords"]
        
        pt1 = Point(x=3.0, y=4.0)
        pt2 = Point(x=1.0, y=2.0)
        res = add_coords(pt1, pt2)
        self.assertEqual(res, 4.0)

    def test_transpile_attention(self):
        code = """
        fn mha(qm, kb, vb) pure {
            scale = 1.0 / (kb.dim_last.sqrt)
            ret (qm @ kb.trans) * scale smax @ vb
        }
        """
        ast = Parser(Lexer(code).tokenize()).parse()
        gen = CodeGenerator()
        py_code = gen.generate(ast)
        self.assertIn("def mha(q, k, v):", py_code)
        self.assertIn("k.transpose(-2, -1)", py_code)

if __name__ == "__main__":
    unittest.main()
