"""
Code Generator / Transpiler per JANUS:
Traduce l'AST di JANUS in codice Python 3.13 / PyTorch / NumPy ad alte prestazioni.
"""

from typing import List, Optional, Any
from janus.ast_nodes import (
    Program, ASTNode, TypeDecl, FieldDecl, SchemaDecl, FnDecl, KernelDecl, Param,
    Stmt, BindingStmt, RetStmt, IfStmt, ForStmt, LoopStmt, BreakStmt, ContinueStmt, ExprStmt,
    Expr, LiteralExpr, IdentExpr, CaseIdentExpr, BinaryExpr, UnaryExpr,
    CaseArg, PipeStep, PipelineExpr, CallExpr, AgentCallExpr, DiffExpr,
    IndexExpr, FieldAccessExpr, TupleExpr, ListExpr, RangeExpr
)

class CodeGenerator:
    def __init__(self, target: str = "pytorch"):
        self.target = target
        self.indent_level = 0

    def _indent(self) -> str:
        return "    " * self.indent_level

    def generate(self, program: Program) -> str:
        lines = [
            "# Generated automatically by JANUS Compiler (janusc v0.1.0)",
            "# Target: Python 3.13 / PyTorch / NumPy Acceleration",
            "",
            "import math",
            "try:",
            "    import torch",
            "    import torch.nn.functional as F",
            "    HAS_TORCH = True",
            "except ImportError:",
            "    HAS_TORCH = False",
            "",
            "try:",
            "    import numpy as np",
            "    HAS_NUMPY = True",
            "except ImportError:",
            "    HAS_NUMPY = False",
            "",
            "# Runtime helper e fallback di JANUS",
            "def _janus_agent_call(agent, prompt, tool=None):",
            "    return f'[Agent {agent}: prompt=\"{prompt}\"]'",
            "",
            "from dataclasses import dataclass",
            "from typing import List, Dict, Any, Tuple, Optional",
            "",
        ]

        for decl in program.declarations:
            lines.append(self._gen_top_level(decl))
            lines.append("")

        return "\n".join(lines)

    def _gen_top_level(self, node: ASTNode) -> str:
        if isinstance(node, TypeDecl):
            return self._gen_type_decl(node)
        elif isinstance(node, SchemaDecl):
            return self._gen_schema_decl(node)
        elif isinstance(node, FnDecl):
            return self._gen_fn_decl(node)
        elif isinstance(node, KernelDecl):
            return self._gen_kernel_decl(node)
        elif isinstance(node, Stmt):
            return self._gen_stmt(node)
        return ""

    def _gen_type_decl(self, node: TypeDecl) -> str:
        lines = [f"@dataclass", f"class {node.name}:"]
        if not node.fields:
            lines.append("    pass")
        else:
            for f in node.fields:
                clean_name = f.name
                lines.append(f"    {clean_name}: Any")
        return "\n".join(lines)

    def _gen_schema_decl(self, node: SchemaDecl) -> str:
        lines = [
            f"# Schema definition for tool-calling validation: {node.name}",
            f"@dataclass",
            f"class {node.name}Input:",
        ]
        if not node.inputs:
            lines.append("    pass")
        else:
            for p in node.inputs:
                def_val = f" = {self._gen_expr(p.default)}" if p.default else ""
                lines.append(f"    {p.name}: Any{def_val}")
        lines.append("")
        lines.append(f"@dataclass")
        lines.append(f"class {node.name}Output:")
        if not node.outputs:
            lines.append("    pass")
        else:
            for o in node.outputs:
                lines.append(f"    {o.name}: Any")
        return "\n".join(lines)

    def _gen_fn_decl(self, node: FnDecl) -> str:
        params_str = []
        for p in node.params:
            p_name = p.base_name or p.name
            if p.default:
                params_str.append(f"{p_name}={self._gen_expr(p.default)}")
            else:
                params_str.append(p_name)
        params_joined = ", ".join(params_str)
        
        lines = [f"def {node.name}({params_joined}):"]
        self.indent_level += 1
        
        if not node.body:
            lines.append(f"{self._indent()}pass")
        else:
            for s in node.body:
                lines.append(f"{self._indent()}{self._gen_stmt(s)}")

        self.indent_level -= 1
        return "\n".join(lines)

    def _gen_kernel_decl(self, node: KernelDecl) -> str:
        params_joined = ", ".join(p.base_name or p.name for p in node.params)
        lines = [
            f"# GPU SIMT Kernel mapping",
            f"def {node.name}({params_joined}):",
        ]
        self.indent_level += 1
        lines.append(f"{self._indent()}# Vectorized SIMD/GPU execution")
        for s in node.body:
            lines.append(f"{self._indent()}{self._gen_stmt(s)}")
        self.indent_level -= 1
        return "\n".join(lines)

    def _gen_stmt(self, stmt: Stmt) -> str:
        if isinstance(stmt, BindingStmt):
            clean_targets = []
            for t in stmt.targets:
                idx_suffix = ""
                base_t = t
                if "[" in t and t.endswith("]"):
                    base_t = t.split("[")[0]
                    idx_suffix = t[len(base_t):]

                if "." in base_t and base_t.split(".")[-1] in ("m", "b", "t", "n", "s", "v"):
                    c_base = base_t.split(".")[0]
                elif len(base_t) == 2 and base_t[1] in ("m", "b", "t", "n", "s", "v") and base_t[0].isalpha():
                    c_base = base_t[0]
                elif base_t.endswith("m") and base_t in ("datam", "labelsm", "imgm", "lossm", "querym", "taskm", "xm", "ym"):
                    c_base = base_t[:-1]
                elif base_t.endswith("b") and base_t in ("paramb", "linb", "mlpb", "noise_predb", "wb", "bb", "kb", "vb"):
                    c_base = base_t[:-1]
                elif base_t.endswith("t") and base_t in ("buft", "yt", "wt", "mlpt"):
                    c_base = base_t[:-1]
                else:
                    c_base = base_t
                clean_targets.append(f"{c_base}{idx_suffix}")
            targets_str = ", ".join(clean_targets)
            return f"{targets_str} = {self._gen_expr(stmt.expr)}"
        
        elif isinstance(stmt, RetStmt):
            if stmt.expr:
                return f"return {self._gen_expr(stmt.expr)}"
            return "return"

        elif isinstance(stmt, IfStmt):
            lines = [f"if {self._gen_expr(stmt.cond)}:"]
            self.indent_level += 1
            for s in stmt.then_branch:
                lines.append(f"{self._indent()}{self._gen_stmt(s)}")
            self.indent_level -= 1
            if stmt.else_branch:
                lines.append(f"{self._indent()}else:")
                self.indent_level += 1
                for s in stmt.else_branch:
                    lines.append(f"{self._indent()}{self._gen_stmt(s)}")
                self.indent_level -= 1
            return "\n".join(lines)

        elif isinstance(stmt, ForStmt):
            iter_str = self._gen_expr(stmt.iterable)
            lines = [f"for {stmt.var_name} in {iter_str}:"]
            self.indent_level += 1
            for s in stmt.body:
                lines.append(f"{self._indent()}{self._gen_stmt(s)}")
            self.indent_level -= 1
            return "\n".join(lines)

        elif isinstance(stmt, LoopStmt):
            lines = [f"while True:"]
            self.indent_level += 1
            for s in stmt.body:
                lines.append(f"{self._indent()}{self._gen_stmt(s)}")
            self.indent_level -= 1
            return "\n".join(lines)

        elif isinstance(stmt, BreakStmt):
            return "break"

        elif isinstance(stmt, ContinueStmt):
            return "continue"

        elif isinstance(stmt, ExprStmt):
            return self._gen_expr(stmt.expr)

        return ""

    def _gen_expr(self, expr: Expr) -> str:
        if isinstance(expr, LiteralExpr):
            if expr.lit_type == "str":
                return f'"{expr.value}"'
            elif expr.lit_type == "bool":
                return "True" if expr.value else "False"
            return str(expr.value)

        elif isinstance(expr, IdentExpr):
            return expr.name

        elif isinstance(expr, CaseIdentExpr):
            # Normalizza rimuovendo suffisso morfologico nel binding runtime
            return expr.base_name or expr.name

        elif isinstance(expr, BinaryExpr):
            lhs_s = self._gen_expr(expr.lhs)
            rhs_s = self._gen_expr(expr.rhs)
            return f"({lhs_s} {expr.op} {rhs_s})"

        elif isinstance(expr, UnaryExpr):
            return f"{expr.op}{self._gen_expr(expr.operand)}"

        elif isinstance(expr, RangeExpr):
            return f"range({self._gen_expr(expr.start)}, {self._gen_expr(expr.end)})"

        elif isinstance(expr, TupleExpr):
            el_str = ", ".join(self._gen_expr(e) for e in expr.elements)
            return f"({el_str})"

        elif isinstance(expr, ListExpr):
            el_str = ", ".join(self._gen_expr(e) for e in expr.elements)
            return f"[{el_str}]"

        elif isinstance(expr, IndexExpr):
            idx_str = ", ".join(self._gen_expr(e) for e in expr.indices)
            return f"{self._gen_expr(expr.target)}[{idx_str}]"

        elif isinstance(expr, FieldAccessExpr):
            # Mappature speciali di proprietà
            if expr.field_name == "trans":
                return f"{self._gen_expr(expr.target)}.transpose(-2, -1)"
            if expr.field_name == "sqrt":
                return f"math.sqrt({self._gen_expr(expr.target)})"
            if expr.field_name == "dim_last":
                return f"{self._gen_expr(expr.target)}.size(-1)"
            if expr.field_name == "len":
                return f"len({self._gen_expr(expr.target)})"
            return f"{self._gen_expr(expr.target)}.{expr.field_name}"

        elif isinstance(expr, CallExpr):
            fn_s = self._gen_expr(expr.func)
            args_s = ", ".join(self._gen_expr(a) for a in expr.args)
            return f"{fn_s}({args_s})"

        elif isinstance(expr, DiffExpr):
            target_s = self._gen_expr(expr.target)
            wrt_s = self._gen_expr(expr.wrt)
            return f"torch.autograd.grad({target_s}, {wrt_s})"

        elif isinstance(expr, AgentCallExpr):
            prompt_s = self._gen_expr(expr.prompt)
            tool_s = self._gen_expr(expr.tool) if expr.tool else "None"
            return f"_janus_agent_call(agent='{expr.agent_name}', prompt={prompt_s}, tool={tool_s})"

        elif isinstance(expr, PipelineExpr):
            current = self._gen_expr(expr.head)
            for step in expr.steps:
                args = [self._gen_expr(a.expr) for a in step.args]
                
                if step.op == "matmul":
                    current = f"torch.matmul({current}, {args[0]})"
                elif step.op == "dot":
                    current = f"torch.dot({current}, {args[0]})"
                elif step.op == "add":
                    current = f"({current} + {args[0]})"
                elif step.op == "sub":
                    current = f"({current} - {args[0]})"
                elif step.op == "mul":
                    current = f"({current} * {args[0]})"
                elif step.op == "div":
                    current = f"({current} / {args[0]})"
                elif step.op == "relu":
                    current = f"torch.relu({current})"
                elif step.op == "smax":
                    current = f"torch.softmax({current}, dim=-1)"
                elif step.op == "gelu":
                    current = f"F.gelu({current})"
                elif step.op == "sum":
                    current = f"torch.sum({current})"
                elif step.op == "sqrt":
                    current = f"torch.sqrt({current})"
                elif step.op == "pow":
                    current = f"torch.pow({current}, {args[0]})"
                elif step.op == "flat":
                    start_dim = args[0] if args else "1"
                    current = f"torch.flatten({current}, start_dim={start_dim})"
                elif step.op == "conv2d":
                    current = f"F.conv2d({current}, {args[0]})"
                elif step.op == "pool":
                    current = f"F.max_pool2d({current}, kernel_size=2, stride=2)"
                elif step.op == "norm":
                    current = f"F.layer_norm({current}, {current}.shape[-1:])"
                else:
                    args_s = ", ".join(args)
                    current = f"{step.op}({current}, {args_s})"
            return current

        return ""
