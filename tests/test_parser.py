"""
Unit test per il Parser di JANUS.
"""
import unittest
from janus.lexer import Lexer
from janus.parser import Parser
from janus.ast_nodes import FnDecl, TypeDecl, SchemaDecl, BindingStmt, PipelineExpr

class TestParser(unittest.TestCase):
    def test_type_decl(self):
        code = "type Lin { w: mat[f32, D, H], b: vec[f32, H] }"
        ast = Parser(Lexer(code).tokenize()).parse()
        self.assertEqual(len(ast.declarations), 1)
        self.assertIsInstance(ast.declarations[0], TypeDecl)
        self.assertEqual(ast.declarations[0].name, "Lin")
        self.assertEqual(len(ast.declarations[0].fields), 2)

    def test_pipeline_chaining(self):
        code = "fn fwd(xm, wb, bb) pure { ret xm matmul wb add bb relu }"
        ast = Parser(Lexer(code).tokenize()).parse()
        fn = ast.declarations[0]
        self.assertIsInstance(fn, FnDecl)
        ret_stmt = fn.body[0]
        self.assertIsInstance(ret_stmt.expr, PipelineExpr)
        self.assertEqual(len(ret_stmt.expr.steps), 3)
        self.assertEqual(ret_stmt.expr.steps[0].op, "matmul")
        self.assertEqual(ret_stmt.expr.steps[1].op, "add")
        self.assertEqual(ret_stmt.expr.steps[2].op, "relu")

    def test_schema_decl(self):
        code = "schema QueryTool { q: str, top_k: i32 = 5 } -> { docs: vec[str] }"
        ast = Parser(Lexer(code).tokenize()).parse()
        self.assertEqual(len(ast.declarations), 1)
        self.assertIsInstance(ast.declarations[0], SchemaDecl)
        self.assertEqual(ast.declarations[0].name, "QueryTool")
        self.assertEqual(len(ast.declarations[0].inputs), 2)
        self.assertEqual(len(ast.declarations[0].outputs), 1)

if __name__ == "__main__":
    unittest.main()
