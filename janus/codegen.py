"""
Code Generator / Transpiler per JANUS:
Traduce l'AST di JANUS in codice Python (3.10+) / PyTorch / NumPy ad alte prestazioni.
"""

import keyword
from typing import List, Optional, Any
from janus import __version__
from janus.ast_nodes import (
    Program, ASTNode, TypeDecl, FieldDecl, SchemaDecl, FnDecl, KernelDecl, Param,
    Stmt, BindingStmt, RetStmt, IfStmt, ForStmt, LoopStmt, BreakStmt, ContinueStmt, ExprStmt,
    Expr, LiteralExpr, IdentExpr, CaseIdentExpr, BinaryExpr, UnaryExpr,
    CaseArg, PipeStep, PipelineExpr, CallExpr, AgentCallExpr, ToolCallExpr, DiffExpr,
    IndexExpr, FieldAccessExpr, TupleExpr, ListExpr, RangeExpr
)

class JanusToolNotRegistered(Exception):
    """Sollevata quando un tool invocato non è registrato nel runtime."""
    pass


class CodeGenerator:
    def __init__(self, target: str = "pytorch"):
        self.target = target
        self.indent_level = 0

    def _py_ident(self, name: str) -> str:
        if keyword.iskeyword(name):
            return f"{name}_"
        return name

    def _indent(self) -> str:
        return "    " * self.indent_level

    def generate(self, program: Program) -> str:
        lines = [
            f"# Generated automatically by JANUS Compiler (janusc v{__version__})",
            "# Target: Python (3.10+) / PyTorch / NumPy Acceleration",
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
            "from dataclasses import dataclass, fields",
            "from typing import List, Dict, Any, Tuple, Optional",
            "",
            "def _janus_agent_call(agent, prompt, tool=None):",
            "    return f'[Agent {agent}: prompt=\"{prompt}\"]'",
            "",
            "try:",
            "    from janus.codegen import JanusToolNotRegistered",
            "except ImportError:",
            "    class JanusToolNotRegistered(Exception):",
            "        \"\"\"Sollevata quando un tool invocato non è registrato nel runtime.\"\"\"",
            "        pass",
            "",
            "_JANUS_TOOL_REGISTRY = {}",
            "",
            "def register_janus_tool(name):",
            "    def decorator(fn):",
            "        _JANUS_TOOL_REGISTRY[name] = fn",
            "        return fn",
            "    return decorator",
            "",
            "def _janus_call_tool(tool_name, *args, **kwargs):",
            "    if tool_name in _JANUS_TOOL_REGISTRY:",
            "        return _JANUS_TOOL_REGISTRY[tool_name](*args, **kwargs)",
            "    raise JanusToolNotRegistered(f\"Tool '{tool_name}' non registrato nel runtime.\")",
            "",
            "class JanusStruct:",
            "    def __sub__(self, other):",
            "        if isinstance(other, self.__class__):",
            "            res = {}",
            "            for f in self.__dataclass_fields__:",
            "                v_s = getattr(self, f)",
            "                v_o = getattr(other, f)",
            "                if HAS_TORCH and isinstance(v_s, torch.Tensor) and isinstance(v_o, torch.Tensor):",
            "                    res[f] = (v_s - v_o).detach().requires_grad_()",
            "                else:",
            "                    res[f] = v_s - v_o",
            "            return self.__class__(**res)",
            "        return NotImplemented",
            "",
            "    def __mul__(self, scalar):",
            "        res = {}",
            "        for f in self.__dataclass_fields__:",
            "            v = getattr(self, f)",
            "            res[f] = v * scalar",
            "        return self.__class__(**res)",
            "",
            "    def __rmul__(self, scalar):",
            "        res = {}",
            "        for f in self.__dataclass_fields__:",
            "            v = getattr(self, f)",
            "            res[f] = scalar * v",
            "        return self.__class__(**res)",
            "",
            "    def __add__(self, other):",
            "        if isinstance(other, self.__class__):",
            "            res = {}",
            "            for f in self.__dataclass_fields__:",
            "                v_s = getattr(self, f)",
            "                v_o = getattr(other, f)",
            "                if HAS_TORCH and isinstance(v_s, torch.Tensor) and isinstance(v_o, torch.Tensor):",
            "                    res[f] = (v_s + v_o).detach().requires_grad_()",
            "                else:",
            "                    res[f] = v_s + v_o",
            "            return self.__class__(**res)",
            "        return NotImplemented",
            "",
            "def _janus_to_tensor(val, requires_grad=False):",
            "    if HAS_TORCH and not isinstance(val, torch.Tensor):",
            "        return torch.tensor(val, dtype=torch.float32, requires_grad=requires_grad)",
            "    elif HAS_TORCH and requires_grad and not val.requires_grad:",
            "        return val.clone().detach().requires_grad_(True)",
            "    return val",
            "",
            "def _janus_param(val):",
            "    if HAS_TORCH:",
            "        if isinstance(val, (int, float)):",
            "            return torch.tensor(float(val), requires_grad=True, dtype=torch.float32)",
            "        elif isinstance(val, torch.Tensor):",
            "            return val.clone().detach().requires_grad_(True)",
            "    return val",
            "",
            "def _janus_step_val(val):",
            "    if HAS_TORCH and isinstance(val, torch.Tensor):",
            "        return val.detach().requires_grad_()",
            "    elif hasattr(val, '__dataclass_fields__'):",
            "        res = {}",
            "        for f in val.__dataclass_fields__:",
            "            v = getattr(val, f)",
            "            if HAS_TORCH and isinstance(v, torch.Tensor):",
            "                res[f] = v.detach().requires_grad_()",
            "            else:",
            "                res[f] = v",
            "        return val.__class__(**res)",
            "    return val",
            "",
            "def _janus_diff(loss, wrt):",
            "    if not HAS_TORCH:",
            "        raise RuntimeError('PyTorch is required for autodiff')",
            "    if hasattr(wrt, '__dataclass_fields__'):",
            "        f_names = list(wrt.__dataclass_fields__.keys())",
            "        f_tensors = tuple(getattr(wrt, f) for f in f_names)",
            "        grads = torch.autograd.grad(loss, f_tensors, allow_unused=True)",
            "        return wrt.__class__(**{f: (g if g is not None else torch.zeros_like(getattr(wrt, f))) for f, g in zip(f_names, grads)})",
            "    elif isinstance(wrt, (tuple, list)):",
            "        grads = torch.autograd.grad(loss, tuple(wrt), allow_unused=True)",
            "        return tuple(g if g is not None else torch.zeros_like(w) for g, w in zip(grads, wrt)) if isinstance(wrt, tuple) else list(grads)",
            "    elif isinstance(wrt, torch.Tensor):",
            "        g = torch.autograd.grad(loss, wrt, allow_unused=True)[0]",
            "        return g if g is not None else torch.zeros_like(wrt)",
            "    return 0.0",
            "",
            "def _janus_result(val):",
            "    if HAS_TORCH and isinstance(val, torch.Tensor):",
            "        if val.numel() == 1:",
            "            return val.item()",
            "        return val.detach()",
            "    elif isinstance(val, tuple):",
            "        return tuple(_janus_result(v) for v in val)",
            "    elif isinstance(val, list):",
            "        return [_janus_result(v) for v in val]",
            "    return val",
            "",
            "def _janus_sqrt(val):",
            "    if HAS_TORCH and isinstance(val, torch.Tensor):",
            "        return torch.sqrt(torch.clamp(val, min=0.0))",
            "    return math.sqrt(max(0.0, float(val)))",
            "",
            "def gauss(*args):",
            "    shape = None",
            "    seed = None",
            "    for a in args:",
            "        if isinstance(a, (tuple, list)) or hasattr(a, 'shape'):",
            "            shape = a.shape if hasattr(a, 'shape') else a",
            "        elif isinstance(a, int):",
            "            seed = a",
            "    return ('gauss', shape, seed)",
            "",
            "def rand(desc, *args):",
            "    shape = None",
            "    seed = None",
            "    if isinstance(desc, tuple) and desc and desc[0] == 'gauss':",
            "        shape = desc[1]",
            "        seed = desc[2]",
            "    else:",
            "        for a in (desc,) + args:",
            "            if isinstance(a, (tuple, list)) or hasattr(a, 'shape'):",
            "                shape = a.shape if hasattr(a, 'shape') else a",
            "            elif isinstance(a, int):",
            "                seed = a",
            "    if HAS_TORCH:",
            "        eps = torch.randn(shape) if shape is not None else torch.randn(1)",
            "    else:",
            "        eps = 0.0",
            "    return eps, (seed + 1 if seed is not None else 0)",
            "",
            "shapes = None",
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
        lines = [f"@dataclass", f"class {node.name}(JanusStruct):"]
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
        tensor_inits = []
        for p in node.params:
            p_name = p.base_name or p.name
            if p.default:
                params_str.append(f"{p_name}={self._gen_expr(p.default)}")
            else:
                params_str.append(p_name)
            if p.case == "m" or ":m" in p.name:
                tensor_inits.append(p_name)
        params_joined = ", ".join(params_str)
        
        lines = [f"def {node.name}({params_joined}):"]
        self.indent_level += 1
        
        for t_name in tensor_inits:
            lines.append(f"{self._indent()}{t_name} = _janus_to_tensor({t_name})")

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

                if ":" in base_t:
                    c_base = base_t.split(":")[0]
                elif "." in base_t and base_t.split(".")[-1] in ("m", "b", "t", "n", "s", "v"):
                    c_base = base_t.split(".")[0]
                else:
                    c_base = base_t
                clean_targets.append(f"{c_base}{idx_suffix}")
            targets_str = ", ".join(clean_targets)
            expr_val = self._gen_expr(stmt.expr)
            if stmt.is_mut:
                if isinstance(stmt.expr, LiteralExpr):
                    expr_val = f"_janus_param({expr_val})"
                else:
                    expr_val = f"_janus_step_val({expr_val})"
            return f"{targets_str} = {expr_val}"
        
        elif isinstance(stmt, RetStmt):
            if stmt.expr:
                return f"return _janus_result({self._gen_expr(stmt.expr)})"
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
            return f"range(int({self._gen_expr(expr.start)}), int({self._gen_expr(expr.end)}))"

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
                return f"_janus_sqrt({self._gen_expr(expr.target)})"
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
            return f"_janus_diff({target_s}, {wrt_s})"

        elif isinstance(expr, AgentCallExpr):
            prompt_s = self._gen_expr(expr.prompt)
            tool_s = self._gen_expr(expr.tool) if expr.tool else "None"
            return f"_janus_agent_call(agent='{expr.agent_name}', prompt={prompt_s}, tool={tool_s})"

        elif isinstance(expr, ToolCallExpr):
            kwargs_list = [f"{k}={self._gen_expr(v)}" for k, v in expr.named_args.items()]
            pos_list = [self._gen_expr(v) for v in expr.positional_args]
            all_args = ", ".join([f"'{expr.tool_name}'"] + pos_list + kwargs_list)
            return f"_janus_call_tool({all_args})"

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
                    dim_val = None
                    keepdim_val = None
                    i = 0
                    while i < len(args):
                        a_str = str(args[i])
                        if "ax" in a_str and i + 1 < len(args):
                            dim_val = args[i + 1]
                            i += 2
                        elif "keep" in a_str and i + 1 < len(args):
                            keepdim_val = args[i + 1]
                            i += 2
                        elif a_str in ("True", "False"):
                            keepdim_val = a_str
                            i += 1
                        elif a_str.lstrip("-").isdigit():
                            dim_val = a_str
                            i += 1
                        else:
                            i += 1
                    opts = []
                    if dim_val is not None:
                        opts.append(f"dim={dim_val}")
                    if keepdim_val is not None:
                        opts.append(f"keepdim={keepdim_val}")
                    opts_s = f", {', '.join(opts)}" if opts else ""
                    current = f"torch.sum({current}{opts_s})"
                elif step.op == "sqrt":
                    current = f"_janus_sqrt({current})"
                elif step.op == "pow":
                    current = f"torch.pow({current}, {args[0]})"
                elif step.op == "flat":
                    start_dim = args[0] if args else "1"
                    current = f"torch.flatten({current}, start_dim={start_dim})"
                elif step.op == "conv2d":
                    k = args[0] if args else "None"
                    stride = "[1, 1]"
                    padding = "[1, 1]"
                    for a_str in args[1:]:
                        s = str(a_str)
                        if "str[" in s:
                            sub = s[s.index("["):]
                            stride = sub[:sub.index("]") + 1] if "]" in sub else sub
                        elif "pad[" in s:
                            sub = s[s.index("["):]
                            padding = sub[:sub.index("]") + 1] if "]" in sub else sub
                    current = f"F.conv2d({current}, {k}, stride={stride}, padding={padding})"
                elif step.op == "pool":
                    stride = "[2, 2]"
                    for a_str in args:
                        s = str(a_str)
                        if "str[" in s:
                            sub = s[s.index("["):]
                            stride = sub[:sub.index("]") + 1] if "]" in sub else sub
                    current = f"F.max_pool2d({current}, kernel_size=2, stride={stride})"
                elif step.op == "norm":
                    current = f"F.layer_norm({current}, {current}.shape[-1:])"
                else:
                    args_s = ", ".join(args)
                    current = f"{step.op}({current}, {args_s})"
            return current

        return ""
