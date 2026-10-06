"""
Type Checker & Semantic Analyzer per JANUS:
Verifica delle forme simboliche, rispetto dei ruoli morfologici ed algebra degli effetti.
"""

from typing import Dict, List, Optional, Any, Set
from janus.ast_nodes import (
    Program, ASTNode, TypeDecl, SchemaDecl, FnDecl, KernelDecl, Param,
    Stmt, BindingStmt, RetStmt, IfStmt, ForStmt, LoopStmt, ExprStmt,
    Expr, LiteralExpr, IdentExpr, CaseIdentExpr, BinaryExpr, UnaryExpr,
    PipelineExpr, PipeStep, CallExpr, AgentCallExpr, DiffExpr,
    TypeExpr, PrimitiveType, TensorType, CustomType
)
from janus.diagnostics import Diagnostic, SourceSpan, DiagnosticPatch

class TypeCheckError(Exception):
    def __init__(self, diagnostic: Diagnostic):
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic

class SymbolTable:
    def __init__(self, parent: Optional['SymbolTable'] = None):
        self.parent = parent
        self.variables: Dict[str, Dict[str, Any]] = {}

    def define(self, name: str, var_type: Any, is_mut: bool = False, case: Optional[str] = None):
        self.variables[name] = {
            "type": var_type,
            "is_mut": is_mut,
            "case": case
        }

    def lookup(self, name: str) -> Optional[Dict[str, Any]]:
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

class TypeChecker:
    def __init__(self):
        self.types: Dict[str, TypeDecl] = {}
        self.schemas: Dict[str, SchemaDecl] = {}
        self.functions: Dict[str, FnDecl] = {}
        self.diagnostics: List[Diagnostic] = []
        self.current_fn: Optional[FnDecl] = None

    def check(self, program: Program) -> List[Diagnostic]:
        self.diagnostics.clear()

        # Pass 1: Registrazione tipi, schemi e funzioni
        for decl in program.declarations:
            if isinstance(decl, TypeDecl):
                self.types[decl.name] = decl
            elif isinstance(decl, SchemaDecl):
                self.schemas[decl.name] = decl
            elif isinstance(decl, FnDecl):
                self.functions[decl.name] = decl

        # Pass 2: Controllo semantico delle funzioni e kernel
        for decl in program.declarations:
            if isinstance(decl, FnDecl):
                self._check_fn(decl)
            elif isinstance(decl, KernelDecl):
                self._check_kernel(decl)

        return self.diagnostics

    def _check_fn(self, fn: FnDecl):
        self.current_fn = fn
        scope = SymbolTable()

        # Registra parametri
        for param in fn.params:
            scope.define(param.base_name or param.name, param.type_expr, is_mut=False, case=param.case)
            # Registra anche con il nome completo se diverso
            if param.name != (param.base_name or param.name):
                scope.define(param.name, param.type_expr, is_mut=False, case=param.case)

        # Controlla corpo
        self._check_block(fn.body, scope)
        self.current_fn = None

    def _check_kernel(self, kern: KernelDecl):
        scope = SymbolTable()
        # Thread id e block dims intrinseci
        scope.define("tid", PrimitiveType("i32"))
        scope.define("blk", PrimitiveType("i32"))
        for param in kern.params:
            scope.define(param.base_name or param.name, param.type_expr, is_mut=True, case=param.case)
        self._check_block(kern.body, scope)

    def _check_block(self, stmts: List[Stmt], scope: SymbolTable):
        for stmt in stmts:
            self._check_stmt(stmt, scope)

    def _check_stmt(self, stmt: Stmt, scope: SymbolTable):
        if isinstance(stmt, BindingStmt):
            expr_type = self._check_expr(stmt.expr, scope)
            for target in stmt.targets:
                existing = scope.lookup(target)
                if existing:
                    # Riassegnazione: verifica mutabilità
                    if not existing["is_mut"] and not stmt.is_mut:
                        self.diagnostics.append(Diagnostic(
                            code="ERR_AFFINE_IMMUTABLE_MUTATION",
                            phase="type_check",
                            message=f"La variabile '{target}' è stata dichiarata immutabile e non può essere riassegnata.",
                            span=SourceSpan(stmt.line, stmt.col, len(target)),
                            offending=target,
                            patch=DiagnosticPatch(target=f"{target} =", replacement=f"mut {target} =")
                        ))
                else:
                    scope.define(target, expr_type, is_mut=stmt.is_mut)

        elif isinstance(stmt, RetStmt):
            if stmt.expr:
                self._check_expr(stmt.expr, scope)

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
            loop_scope.define(stmt.var_name, PrimitiveType("i32"))
            self._check_block(stmt.body, loop_scope)

        elif isinstance(stmt, LoopStmt):
            loop_scope = SymbolTable(scope)
            self._check_block(stmt.body, loop_scope)

        elif isinstance(stmt, ExprStmt):
            self._check_expr(stmt.expr, scope)

    def _check_expr(self, expr: Expr, scope: SymbolTable) -> Any:
        if isinstance(expr, LiteralExpr):
            return PrimitiveType(expr.lit_type)

        if isinstance(expr, IdentExpr):
            var = scope.lookup(expr.name)
            if var:
                return var["type"]
            return None

        if isinstance(expr, CaseIdentExpr):
            var = scope.lookup(expr.base_name or expr.name)
            if var:
                return var["type"]
            return None

        if isinstance(expr, BinaryExpr):
            t1 = self._check_expr(expr.lhs, scope)
            t2 = self._check_expr(expr.rhs, scope)
            return t1 or t2

        if isinstance(expr, UnaryExpr):
            return self._check_expr(expr.operand, scope)

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
            # Verifica che la funzione corrente abbia effetto 'io'
            if self.current_fn and self.current_fn.effect == "pure":
                self.diagnostics.append(Diagnostic(
                    code="ERR_EFFECT_PURITY_VIOLATION",
                    phase="type_check",
                    message="Chiamata ad agente cognitivo 'call agentv' proibita in una funzione 'pure'.",
                    span=SourceSpan(expr.line, expr.col, len(expr.agent_name)),
                    offending=expr.agent_name,
                    patch=DiagnosticPatch(target="pure", replacement="io")
                ))
            return PrimitiveType("str")

        if isinstance(expr, DiffExpr):
            self._check_expr(expr.target, scope)
            self._check_expr(expr.wrt, scope)
            return PrimitiveType("f32")

        return None
