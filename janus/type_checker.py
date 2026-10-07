"""
Type Checker & Semantic Analyzer per JANUS:
Verifica delle forme simboliche, rispetto dei ruoli morfologici ed algebra degli effetti.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from janus.ast_nodes import (
    Program, ASTNode, TypeDecl, SchemaDecl, FnDecl, KernelDecl, Param,
    Stmt, BindingStmt, RetStmt, IfStmt, ForStmt, LoopStmt, ExprStmt,
    Expr, LiteralExpr, IdentExpr, CaseIdentExpr, BinaryExpr, UnaryExpr,
    RangeExpr, TupleExpr, ListExpr, IndexExpr, FieldAccessExpr,
    PipelineExpr, PipeStep, CallExpr, AgentCallExpr, ToolCallExpr, DiffExpr,
    TypeExpr, PrimitiveType, TensorType, CustomType, ListType, ToolOutputType
)
from janus.diagnostics import Diagnostic, SourceSpan, DiagnosticPatch

BUILTINS = {
    "len", "dim_last", "sum", "pow", "sqrt", "exp", "log", "max", "min",
    "range", "tid", "blk", "print", "math", "torch", "true", "false",
    "gauss", "cat", "Seed", "step", "conv2d", "matmul", "dot", "norm", "pool",
    "str", "pad", "ax", "keep", "shapes", "seed", "rand"
}

EFFECT_ORDER = {
    "pure": 0,
    "stoc": 1,
    "io": 2,
}

INT_TYPES = {"i8", "i16", "i32", "i64", "int"}
FLOAT_TYPES = {"f16", "f32", "f64", "bf16", "float"}
NUMERIC_TYPES = INT_TYPES | FLOAT_TYPES
STR_TYPES = {"str", "string"}
BOOL_TYPES = {"bool", "boolean"}

class TypeCheckError(Exception):
    def __init__(self, diagnostic: Diagnostic):
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic

class SymbolTable:
    def __init__(self, parent: Optional['SymbolTable'] = None):
        self.parent = parent
        self.variables: Dict[str, Dict[str, Any]] = {}

    def _clean_name(self, name: str) -> str:
        if ":" in name:
            return name.split(":")[0]
        if "." in name and name.split(".")[-1] in ('m', 'b', 't', 'n', 's', 'v'):
            return name.split(".")[0]
        return name

    def define(self, name: str, var_type: Any, is_mut: bool = False, case: Optional[str] = None):
        cname = self._clean_name(name)
        self.variables[cname] = {
            "type": var_type,
            "is_mut": is_mut,
            "case": case
        }

    def lookup(self, name: str) -> Optional[Dict[str, Any]]:
        cname = self._clean_name(name)
        if cname in self.variables:
            return self.variables[cname]
        if self.parent:
            return self.parent.lookup(name)
        return None

class TypeChecker:
    def __init__(self):
        self.types: Dict[str, TypeDecl] = {}
        self.schemas: Dict[str, SchemaDecl] = {}
        self.functions: Dict[str, FnDecl] = {}
        self.transitive_effects: Dict[str, Set[str]] = {}
        self.operational_effects: Dict[str, Set[str]] = {}
        self.reachable_tools: Dict[str, Set[str]] = {}
        self.inferred_fn_returns: Dict[str, Optional[TypeExpr]] = {}
        self.diagnostics: List[Diagnostic] = []
        self.current_fn: Optional[FnDecl] = None

    def _format_type(self, t: Any) -> str:
        if t is None:
            return "unknown"
        if isinstance(t, PrimitiveType):
            return t.name
        if isinstance(t, ListType):
            return f"[{self._format_type(t.inner)}]"
        if isinstance(t, CustomType):
            return t.name
        if isinstance(t, ToolOutputType):
            return f"{t.tool_name}Output"
        if isinstance(t, TensorType):
            return f"{t.kind}<{t.dtype}>"
        if isinstance(t, str):
            return t
        return str(t)

    def _types_compatible(self, expected: Optional[TypeExpr], actual: Optional[TypeExpr]) -> bool:
        if expected is None or actual is None:
            return True  # Gradual typing: non dare errore falso se il tipo non è inferibile

        if isinstance(expected, PrimitiveType) and isinstance(actual, PrimitiveType):
            e_name = expected.name.lower()
            a_name = actual.name.lower()
            if e_name in INT_TYPES and a_name in INT_TYPES:
                return True
            if e_name in FLOAT_TYPES and a_name in FLOAT_TYPES:
                return True
            # Widening lecito solo tra int e float (es. i32 compatibile con f32/f16/bf16, ma non il contrario)
            if e_name in FLOAT_TYPES and a_name in INT_TYPES:
                return True
            if e_name in INT_TYPES and a_name in FLOAT_TYPES:
                return False
            if e_name in STR_TYPES and a_name in STR_TYPES:
                return True
            if e_name in BOOL_TYPES and a_name in BOOL_TYPES:
                return True
            return False

        if isinstance(expected, ListType) and isinstance(actual, ListType):
            return self._types_compatible(expected.inner, actual.inner)

        if isinstance(expected, CustomType) and isinstance(actual, CustomType):
            return expected.name == actual.name

        if isinstance(expected, CustomType) and isinstance(actual, ToolOutputType):
            return expected.name == actual.tool_name or expected.name == f"{actual.tool_name}Output"

        if isinstance(expected, ToolOutputType) and isinstance(actual, ToolOutputType):
            return expected.tool_name == actual.tool_name

        if isinstance(expected, TensorType) and isinstance(actual, TensorType):
            if expected.kind != actual.kind:
                return False
            e_dt = expected.dtype.lower()
            a_dt = actual.dtype.lower()
            if e_dt == a_dt:
                return True
            if e_dt in FLOAT_TYPES and a_dt in INT_TYPES:
                return True
            return False

        return False

    def check(self, program: Program) -> List[Diagnostic]:
        self.diagnostics.clear()
        self.types.clear()
        self.schemas.clear()
        self.functions.clear()
        self.transitive_effects.clear()
        self.operational_effects.clear()
        self.reachable_tools.clear()
        self.inferred_fn_returns.clear()

        # Pass 1: Registrazione tipi, schemi e funzioni
        for decl in program.declarations:
            if isinstance(decl, TypeDecl):
                self.types[decl.name] = decl
            elif isinstance(decl, SchemaDecl):
                self.schemas[decl.name] = decl
            elif isinstance(decl, FnDecl):
                self.functions[decl.name] = decl

        # Pass 1.5: Calcolo della propagazione transitiva degli effetti (chiusura fissa)
        self._compute_transitive_effects()

        # Pass 2: Controllo semantico delle funzioni e kernel
        for decl in program.declarations:
            if isinstance(decl, FnDecl):
                self._check_fn(decl)
            elif isinstance(decl, KernelDecl):
                self._check_kernel(decl)

        return self.diagnostics

    def _collect_direct_calls_and_effects(self, fn: FnDecl) -> Tuple[Set[str], Set[str], Set[str]]:
        called_fns: Set[str] = set()
        direct_tools: Set[str] = set()
        effects: Set[str] = set()

        def visit_expr(expr: Any):
            if expr is None:
                return
            if isinstance(expr, ToolCallExpr):
                direct_tools.add(expr.tool_name)
                if expr.tool_name in self.schemas:
                    s_eff = self.schemas[expr.tool_name].effect
                    if s_eff in ("io", "stoc"):
                        effects.add(s_eff)
                for val in expr.named_args.values():
                    visit_expr(val)
                for pos in expr.positional_args:
                    visit_expr(pos)
            elif isinstance(expr, AgentCallExpr):
                effects.add("io")
                visit_expr(expr.prompt)
                visit_expr(expr.tool)
                for v in expr.extra_args.values():
                    visit_expr(v)
            elif isinstance(expr, CallExpr):
                func_name = None
                if isinstance(expr.func, IdentExpr):
                    func_name = expr.func.name
                elif isinstance(expr.func, CaseIdentExpr):
                    func_name = expr.func.base_name or expr.func.name
                if func_name:
                    clean_name = func_name.split(":")[0]
                    called_fns.add(clean_name)
                visit_expr(expr.func)
                for a in expr.args:
                    visit_expr(a)
            elif isinstance(expr, PipelineExpr):
                visit_expr(expr.head)
                for step in expr.steps:
                    if step.op == "rand":
                        effects.add("stoc")
                    for a in step.args:
                        visit_expr(a.expr)
            elif isinstance(expr, BinaryExpr):
                visit_expr(expr.lhs)
                visit_expr(expr.rhs)
            elif isinstance(expr, UnaryExpr):
                visit_expr(expr.operand)
            elif isinstance(expr, TupleExpr):
                for e in expr.elements:
                    visit_expr(e)
            elif isinstance(expr, ListExpr):
                for e in expr.elements:
                    visit_expr(e)
            elif isinstance(expr, IndexExpr):
                visit_expr(expr.target)
                for idx in expr.indices:
                    visit_expr(idx)
            elif isinstance(expr, FieldAccessExpr):
                visit_expr(expr.target)
            elif isinstance(expr, DiffExpr):
                visit_expr(expr.target)
                visit_expr(expr.wrt)
            elif isinstance(expr, RangeExpr):
                visit_expr(expr.start)
                visit_expr(expr.end)

        def visit_stmt(stmt: Any):
            if stmt is None:
                return
            if isinstance(stmt, BindingStmt):
                visit_expr(stmt.expr)
            elif isinstance(stmt, RetStmt):
                visit_expr(stmt.expr)
            elif isinstance(stmt, ExprStmt):
                visit_expr(stmt.expr)
            elif isinstance(stmt, IfStmt):
                visit_expr(stmt.cond)
                for s in stmt.then_branch:
                    visit_stmt(s)
                if stmt.else_branch:
                    for s in stmt.else_branch:
                        visit_stmt(s)
            elif isinstance(stmt, ForStmt):
                visit_expr(stmt.iterable)
                for s in stmt.body:
                    visit_stmt(s)
            elif isinstance(stmt, LoopStmt):
                for s in stmt.body:
                    visit_stmt(s)

        for s in fn.body:
            visit_stmt(s)

        return called_fns, direct_tools, effects

    def _compute_transitive_effects(self):
        direct_calls: Dict[str, Set[str]] = {}
        self.reachable_tools = {}
        self.operational_effects = {}
        self.transitive_effects = {}

        for name, fn in self.functions.items():
            called_fns, direct_tools, effects = self._collect_direct_calls_and_effects(fn)
            direct_calls[name] = called_fns
            self.reachable_tools[name] = set(direct_tools)
            self.operational_effects[name] = set(effects)
            eff_contract = set(effects)
            if fn.effect in ("io", "stoc"):
                eff_contract.add(fn.effect)
            self.transitive_effects[name] = eff_contract

        # Chiusura transitiva con algoritmo a punto fisso (gestisce cicli e ricorsione)
        changed = True
        while changed:
            changed = False
            for name, called_set in direct_calls.items():
                for called in called_set:
                    if called in self.reachable_tools:
                        new_tools = self.reachable_tools[called] - self.reachable_tools[name]
                        if new_tools:
                            self.reachable_tools[name].update(new_tools)
                            changed = True

                    if called in self.operational_effects:
                        new_op_eff = self.operational_effects[called] - self.operational_effects[name]
                        if new_op_eff:
                            self.operational_effects[name].update(new_op_eff)
                            changed = True

                    if called in self.transitive_effects:
                        new_eff = self.transitive_effects[called] - self.transitive_effects[name]
                        if new_eff:
                            self.transitive_effects[name].update(new_eff)
                            changed = True

                    if called in self.functions:
                        decl_eff = self.functions[called].effect
                        if decl_eff in ("io", "stoc") and decl_eff not in self.transitive_effects[name]:
                            self.transitive_effects[name].add(decl_eff)
                            changed = True

    def get_effects_summary(self) -> List[Dict[str, Any]]:
        summary = []
        for name, fn in self.functions.items():
            op_effs = self.operational_effects.get(name, set())
            if "io" in op_effs:
                computed = "io"
            elif "stoc" in op_effs:
                computed = "stoc"
            else:
                computed = "pure"
            tools = sorted(list(self.reachable_tools.get(name, set())))
            summary.append({
                "function": name,
                "declared_effect": fn.effect,
                "computed_effect": computed,
                "reachable_tools": tools,
            })
        return summary

    def _check_fn(self, fn: FnDecl):
        self.current_fn = fn
        scope = SymbolTable()

        seen_params = set()
        for param in fn.params:
            p_base = param.base_name if (param.case or ":" in param.name) else param.name
            clean_base = scope._clean_name(p_base)
            if clean_base in seen_params:
                self.diagnostics.append(Diagnostic(
                    code="ERR_DUPLICATE_PARAM",
                    phase="type_check",
                    message=f"Parametro duplicato o in conflitto '{clean_base}' nella funzione '{fn.name}'.",
                    span=SourceSpan(param.line, param.col, len(param.name)),
                    offending=param.name,
                    patch=None
                ))
            else:
                seen_params.add(clean_base)
                scope.define(clean_base, param.type_expr, is_mut=False, case=param.case)

        self._check_block(fn.body, scope)
        self.current_fn = None

    def _check_kernel(self, kern: KernelDecl):
        scope = SymbolTable()
        scope.define("tid", PrimitiveType(name="i32"))
        scope.define("blk", PrimitiveType(name="i32"))
        for param in kern.params:
            p_base = param.base_name if (param.case or ":" in param.name) else param.name
            scope.define(p_base, param.type_expr, is_mut=True, case=param.case)
        self._check_block(kern.body, scope)

    def _check_block(self, stmts: List[Stmt], scope: SymbolTable):
        for stmt in stmts:
            self._check_stmt(stmt, scope)

    def _check_stmt(self, stmt: Stmt, scope: SymbolTable):
        if isinstance(stmt, BindingStmt):
            expr_type = self._check_expr(stmt.expr, scope)
            for target in stmt.targets:
                clean_target = scope._clean_name(target)
                if "[" in clean_target:
                    clean_target = clean_target.split("[")[0]
                existing = scope.lookup(clean_target)
                if existing:
                    # Riassegnazione: verifica mutabilità
                    if not existing["is_mut"] and not stmt.is_mut:
                        self.diagnostics.append(Diagnostic(
                            code="ERR_AFFINE_IMMUTABLE_MUTATION",
                            phase="type_check",
                            message=f"La variabile '{clean_target}' è stata dichiarata immutabile e non può essere riassegnata.",
                            span=SourceSpan(stmt.line, stmt.col, len(target)),
                            offending=target,
                            patch=DiagnosticPatch(target=f"{target} =", replacement=f"mut {target} =")
                        ))
                    else:
                        existing["type"] = expr_type or existing["type"]
                else:
                    scope.define(clean_target, expr_type, is_mut=stmt.is_mut)

        elif isinstance(stmt, RetStmt):
            if stmt.expr:
                ret_t = self._check_expr(stmt.expr, scope)
                if self.current_fn and ret_t is not None:
                    self.inferred_fn_returns[self.current_fn.name] = ret_t

        elif isinstance(stmt, IfStmt):
            self._check_expr(stmt.cond, scope)
            sub_then = SymbolTable(scope)
            self._check_block(stmt.then_branch, sub_then)
            if stmt.else_branch:
                sub_else = SymbolTable(scope)
                self._check_block(stmt.else_branch, sub_else)

        elif isinstance(stmt, ForStmt):
            self._check_expr(stmt.iterable, scope)
            loop_scope = SymbolTable(scope)
            loop_scope.define(stmt.var_name, PrimitiveType(name="i32"))
            self._check_block(stmt.body, loop_scope)

        elif isinstance(stmt, LoopStmt):
            loop_scope = SymbolTable(scope)
            self._check_block(stmt.body, loop_scope)

        elif isinstance(stmt, ExprStmt):
            self._check_expr(stmt.expr, scope)

    def _check_expr(self, expr: Expr, scope: SymbolTable) -> Any:
        if isinstance(expr, LiteralExpr):
            return PrimitiveType(name=expr.lit_type)

        if isinstance(expr, IdentExpr):
            var = scope.lookup(expr.name)
            if var:
                return var["type"]
            if (expr.name in self.functions or 
                expr.name in self.types or 
                expr.name in self.schemas or 
                expr.name in BUILTINS):
                return None

            self.diagnostics.append(Diagnostic(
                code="ERR_UNDEFINED_VARIABLE",
                phase="type_check",
                message=f"Variabile non definita '{expr.name}' nello scope corrente.",
                span=SourceSpan(expr.line, expr.col, len(expr.name)),
                offending=expr.name,
                patch=None
            ))
            return None

        if isinstance(expr, CaseIdentExpr):
            base = expr.base_name or expr.name
            var = scope.lookup(base)
            if var:
                return var["type"]
            if (base in self.functions or 
                base in self.types or 
                base in self.schemas or 
                base in BUILTINS):
                return None

            self.diagnostics.append(Diagnostic(
                code="ERR_UNDEFINED_VARIABLE",
                phase="type_check",
                message=f"Variabile non definita '{expr.name}' nello scope corrente.",
                span=SourceSpan(expr.line, expr.col, len(expr.name)),
                offending=expr.name,
                patch=None
            ))
            return None

        if isinstance(expr, BinaryExpr):
            t1 = self._check_expr(expr.lhs, scope)
            t2 = self._check_expr(expr.rhs, scope)
            if expr.op in ("==", "!=", "<", "<=", ">", ">="):
                return PrimitiveType(name="bool")
            if expr.op in ("+", "-", "*", "/", "%", "^", "@"):
                def is_flt(t):
                    return isinstance(t, PrimitiveType) and t.name.lower() in FLOAT_TYPES
                def is_int(t):
                    return isinstance(t, PrimitiveType) and t.name.lower() in INT_TYPES
                def is_str(t):
                    return isinstance(t, PrimitiveType) and t.name.lower() in STR_TYPES

                if is_flt(t1) or is_flt(t2):
                    return PrimitiveType(name="f32")
                if is_int(t1) and is_int(t2):
                    return PrimitiveType(name="i32")
                if expr.op == "+" and (is_str(t1) or is_str(t2)):
                    return PrimitiveType(name="str")
            return t1 or t2

        if isinstance(expr, UnaryExpr):
            t = self._check_expr(expr.operand, scope)
            if expr.op == "!":
                return PrimitiveType(name="bool")
            return t

        if isinstance(expr, RangeExpr):
            self._check_expr(expr.start, scope)
            self._check_expr(expr.end, scope)
            return None

        if isinstance(expr, TupleExpr):
            for e in expr.elements:
                self._check_expr(e, scope)
            return None

        if isinstance(expr, ListExpr):
            elem_types = [self._check_expr(e, scope) for e in expr.elements]
            typed_elems = [t for t in elem_types if t is not None]
            if typed_elems:
                return ListType(inner=typed_elems[0])
            return ListType(inner=PrimitiveType(name="any"))

        if isinstance(expr, IndexExpr):
            self._check_expr(expr.target, scope)
            for i in expr.indices:
                self._check_expr(i, scope)
            return None

        if isinstance(expr, FieldAccessExpr):
            target_type = self._check_expr(expr.target, scope)
            if target_type is None:
                return None
            if isinstance(target_type, ToolOutputType):
                if expr.field_name in target_type.fields:
                    return target_type.fields[expr.field_name]
                else:
                    self.diagnostics.append(Diagnostic(
                        code="ERR_UNKNOWN_OUTPUT_FIELD",
                        phase="type_check",
                        message=f"Campo sconosciuto '{expr.field_name}' per l'output del tool '{target_type.tool_name}'.",
                        span=SourceSpan(expr.line, expr.col, len(expr.field_name)),
                        offending=expr.field_name,
                        patch=None
                    ))
                    return None
            if isinstance(target_type, CustomType) and target_type.name in self.types:
                t_decl = self.types[target_type.name]
                fields_map = {f.name: f.type_expr for f in t_decl.fields}
                if expr.field_name in fields_map:
                    return fields_map[expr.field_name]
                else:
                    self.diagnostics.append(Diagnostic(
                        code="ERR_UNKNOWN_OUTPUT_FIELD",
                        phase="type_check",
                        message=f"Campo sconosciuto '{expr.field_name}' per il tipo '{target_type.name}'.",
                        span=SourceSpan(expr.line, expr.col, len(expr.field_name)),
                        offending=expr.field_name,
                        patch=None
                    ))
                    return None
            return None

        if isinstance(expr, CallExpr):
            func_name = None
            if isinstance(expr.func, IdentExpr):
                func_name = expr.func.name
            elif isinstance(expr.func, CaseIdentExpr):
                func_name = expr.func.base_name or expr.func.name

            if func_name:
                clean_func_name = func_name.split(":")[0]
                if clean_func_name not in self.functions and clean_func_name not in BUILTINS:
                    if scope.lookup(clean_func_name) is None:
                        self.diagnostics.append(Diagnostic(
                            code="ERR_UNDEFINED_FUNCTION",
                            phase="type_check",
                            message=f"Funzione non definita '{clean_func_name}'.",
                            span=SourceSpan(expr.line, expr.col, len(clean_func_name)),
                            offending=clean_func_name,
                            patch=None
                        ))
                        for a in expr.args:
                            self._check_expr(a, scope)
                        return None

                if clean_func_name in self.functions:
                    target_fn = self.functions[clean_func_name]
                    target_trans = self.transitive_effects.get(clean_func_name, set())
                    active_eff = "io" if ("io" in target_trans or target_fn.effect == "io") else (
                        "stoc" if ("stoc" in target_trans or target_fn.effect == "stoc") else "pure"
                    )

                    if self.current_fn:
                        caller_eff = self.current_fn.effect
                        if EFFECT_ORDER[active_eff] > EFFECT_ORDER[caller_eff]:
                            if caller_eff == "pure":
                                self.diagnostics.append(Diagnostic(
                                    code="ERR_EFFECT_PURITY_VIOLATION",
                                    phase="type_check",
                                    message=f"Invocazione della funzione non pura '{clean_func_name}' (effetto transitivo '{active_eff}') proibita in funzione 'pure'.",
                                    span=SourceSpan(expr.line, expr.col, len(clean_func_name)),
                                    offending=clean_func_name,
                                    patch=DiagnosticPatch(target=f"fn {self.current_fn.name}", replacement=f"fn {self.current_fn.name} ... {active_eff}")
                                ))
                            elif caller_eff == "stoc" and active_eff == "io":
                                self.diagnostics.append(Diagnostic(
                                    code="ERR_EFFECT_ESCALATION",
                                    phase="type_check",
                                    message=f"Escalation dell'effetto: la funzione 'stoc' '{self.current_fn.name}' non può invocare la funzione 'io' '{clean_func_name}' (effetto transitivo '{active_eff}').",
                                    span=SourceSpan(expr.line, expr.col, len(clean_func_name)),
                                    offending=clean_func_name,
                                    patch=DiagnosticPatch(target=f"fn {self.current_fn.name}", replacement=f"fn {self.current_fn.name} ... io")
                                ))

                    for a in expr.args:
                        self._check_expr(a, scope)
                    return target_fn.ret_type or self.inferred_fn_returns.get(clean_func_name)

            self._check_expr(expr.func, scope)
            for a in expr.args:
                self._check_expr(a, scope)
            return None

        if isinstance(expr, PipelineExpr):
            current = self._check_expr(expr.head, scope)
            for step in expr.steps:
                # Controlla violazioni di purezza di effetto
                if step.op == "rand" and self.current_fn and self.current_fn.effect == "pure":
                    self.diagnostics.append(Diagnostic(
                        code="ERR_EFFECT_PURITY_VIOLATION",
                        phase="type_check",
                        message="L'operazione stocastica 'rand' non è ammessa in una funzione 'pure'.",
                        span=SourceSpan(step.line, step.col, len(step.op)),
                        offending=step.op,
                        patch=DiagnosticPatch(target=f"fn {self.current_fn.name}", replacement=f"fn {self.current_fn.name} ... stoc")
                    ))
                for arg in step.args:
                    self._check_expr(arg.expr, scope)
            return current

        if isinstance(expr, AgentCallExpr):
            if self.current_fn:
                caller_eff = self.current_fn.effect
                if caller_eff == "pure":
                    self.diagnostics.append(Diagnostic(
                        code="ERR_EFFECT_PURITY_VIOLATION",
                        phase="type_check",
                        message="Chiamata ad agente cognitivo 'call agentv' proibita in una funzione 'pure'.",
                        span=SourceSpan(expr.line, expr.col, len(expr.agent_name)),
                        offending=expr.agent_name,
                        patch=DiagnosticPatch(target="pure", replacement="io")
                    ))
                elif caller_eff == "stoc":
                    self.diagnostics.append(Diagnostic(
                        code="ERR_EFFECT_ESCALATION",
                        phase="type_check",
                        message=f"Escalation dell'effetto: chiamata ad agente cognitivo 'io' non ammessa nella funzione 'stoc' '{self.current_fn.name}'.",
                        span=SourceSpan(expr.line, expr.col, len(expr.agent_name)),
                        offending=expr.agent_name,
                        patch=DiagnosticPatch(target="stoc", replacement="io")
                    ))
            if expr.prompt:
                self._check_expr(expr.prompt, scope)
            if expr.tool:
                self._check_expr(expr.tool, scope)
            for v in expr.extra_args.values():
                self._check_expr(v, scope)
            return PrimitiveType(name="str")

        if isinstance(expr, ToolCallExpr):
            if expr.tool_name not in self.schemas:
                self.diagnostics.append(Diagnostic(
                    code="ERR_UNDEFINED_TOOL",
                    phase="type_check",
                    message=f"Tool o Schema '{expr.tool_name}' non dichiarato nel programma.",
                    span=SourceSpan(expr.line, expr.col, len(expr.tool_name)),
                    offending=expr.tool_name,
                    patch=None
                ))
                return None

            schema = self.schemas[expr.tool_name]

            # Controllo lattice effetti
            if self.current_fn:
                caller_eff = self.current_fn.effect
                tool_eff = schema.effect
                if EFFECT_ORDER[tool_eff] > EFFECT_ORDER[caller_eff]:
                    if caller_eff == "pure":
                        self.diagnostics.append(Diagnostic(
                            code="ERR_EFFECT_PURITY_VIOLATION",
                            phase="type_check",
                            message=f"Invocazione del tool con effetto '{tool_eff}' '{expr.tool_name}' proibita in funzione 'pure'.",
                            span=SourceSpan(expr.line, expr.col, len(expr.tool_name)),
                            offending=expr.tool_name,
                            patch=DiagnosticPatch(target="pure", replacement=tool_eff)
                        ))
                    elif caller_eff == "stoc" and tool_eff == "io":
                        self.diagnostics.append(Diagnostic(
                            code="ERR_EFFECT_ESCALATION",
                            phase="type_check",
                            message=f"Escalation dell'effetto: la funzione 'stoc' '{self.current_fn.name}' non può invocare il tool 'io' '{expr.tool_name}'.",
                            span=SourceSpan(expr.line, expr.col, len(expr.tool_name)),
                            offending=expr.tool_name,
                            patch=DiagnosticPatch(target="stoc", replacement="io")
                        ))

            # 1. Controllo arità posizionale (ERR_TOOL_ARITY)
            if len(expr.positional_args) > len(schema.inputs):
                self.diagnostics.append(Diagnostic(
                    code="ERR_TOOL_ARITY",
                    phase="type_check",
                    message=f"Numero di argomenti posizionali ({len(expr.positional_args)}) supera i parametri dichiarati ({len(schema.inputs)}) per il tool '{expr.tool_name}'.",
                    span=SourceSpan(expr.line, expr.col, len(expr.tool_name)),
                    offending=expr.tool_name,
                    patch=None
                ))

            # 2. Controllo argomenti duplicati (ERR_DUPLICATE_TOOL_ARGUMENT)
            # a) Duplicati raccolti durante il parsing (es. tool(arg=1, arg=2))
            for dup_arg in getattr(expr, "duplicate_args", []):
                self.diagnostics.append(Diagnostic(
                    code="ERR_DUPLICATE_TOOL_ARGUMENT",
                    phase="type_check",
                    message=f"Argomento duplicato '{dup_arg}' specificato per il tool '{expr.tool_name}'.",
                    span=SourceSpan(expr.line, expr.col, len(dup_arg)),
                    offending=dup_arg,
                    patch=None
                ))

            # b) Duplicati tra posizionali e per nome
            for i, p_expr in enumerate(expr.positional_args):
                if i < len(schema.inputs):
                    param_name = schema.inputs[i].name
                    if param_name in expr.named_args:
                        self.diagnostics.append(Diagnostic(
                            code="ERR_DUPLICATE_TOOL_ARGUMENT",
                            phase="type_check",
                            message=f"Argomento duplicato '{param_name}' passato sia posizionalmente che per nome per il tool '{expr.tool_name}'.",
                            span=SourceSpan(expr.line, expr.col, len(param_name)),
                            offending=param_name,
                            patch=None
                        ))

            # 3. Controllo argomenti sconosciuti (ERR_UNKNOWN_TOOL_ARGUMENT)
            expected_inputs = {p.name: p for p in schema.inputs}
            for arg_name in expr.named_args:
                if arg_name not in expected_inputs:
                    self.diagnostics.append(Diagnostic(
                        code="ERR_UNKNOWN_TOOL_ARGUMENT",
                        phase="type_check",
                        message=f"Argomento sconosciuto '{arg_name}' per il tool '{expr.tool_name}'.",
                        span=SourceSpan(expr.line, expr.col, len(arg_name)),
                        offending=arg_name,
                        patch=None
                    ))

            # 4. Controllo argomenti obbligatori mancanti (ERR_MISSING_TOOL_ARGUMENT)
            provided_arg_names = set(expr.named_args.keys())
            for i in range(min(len(expr.positional_args), len(schema.inputs))):
                provided_arg_names.add(schema.inputs[i].name)

            for param_name, param in expected_inputs.items():
                if param.default is None and param_name not in provided_arg_names:
                    self.diagnostics.append(Diagnostic(
                        code="ERR_MISSING_TOOL_ARGUMENT",
                        phase="type_check",
                        message=f"Argomento obbligatorio mancante '{param_name}' per il tool '{expr.tool_name}'.",
                        span=SourceSpan(expr.line, expr.col, len(expr.tool_name)),
                        offending=param_name,
                        patch=None
                    ))

            # 5. Type checking degli argomenti (ERR_TOOL_ARG_TYPE_MISMATCH)
            # a) Argomenti posizionali
            for i, p_expr in enumerate(expr.positional_args):
                act_type = self._check_expr(p_expr, scope)
                if i < len(schema.inputs):
                    param = schema.inputs[i]
                    if not self._types_compatible(param.type_expr, act_type):
                        self.diagnostics.append(Diagnostic(
                            code="ERR_TOOL_ARG_TYPE_MISMATCH",
                            phase="type_check",
                            message=f"Tipo incompatibile per l'argomento posizionale {i} ('{param.name}') del tool '{expr.tool_name}': atteso '{self._format_type(param.type_expr)}', ottenuto '{self._format_type(act_type)}'.",
                            span=SourceSpan(p_expr.line, p_expr.col, len(param.name)),
                            offending=param.name,
                            patch=None
                        ))

            # b) Argomenti nominati
            for arg_name, arg_expr in expr.named_args.items():
                act_type = self._check_expr(arg_expr, scope)
                if arg_name in expected_inputs:
                    param = expected_inputs[arg_name]
                    if not self._types_compatible(param.type_expr, act_type):
                        self.diagnostics.append(Diagnostic(
                            code="ERR_TOOL_ARG_TYPE_MISMATCH",
                            phase="type_check",
                            message=f"Tipo incompatibile per l'argomento '{arg_name}' del tool '{expr.tool_name}': atteso '{self._format_type(param.type_expr)}', ottenuto '{self._format_type(act_type)}'.",
                            span=SourceSpan(arg_expr.line, arg_expr.col, len(arg_name)),
                            offending=arg_name,
                            patch=None
                        ))

            outputs_map = {f.name: f.type_expr for f in schema.outputs}
            return ToolOutputType(tool_name=expr.tool_name, fields=outputs_map)

        if isinstance(expr, DiffExpr):
            self._check_expr(expr.target, scope)
            self._check_expr(expr.wrt, scope)
            return PrimitiveType(name="f32")

        return None
