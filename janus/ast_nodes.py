"""
Definizioni dei nodi dell'Abstract Syntax Tree (AST) di JANUS.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

@dataclass
class ASTNode:
    line: int = 1
    col: int = 1

# =========================================================================
# Tipi
# =========================================================================

@dataclass
class TypeExpr(ASTNode):
    pass

@dataclass
class PrimitiveType(TypeExpr):
    name: str = "f32"  # f32, f16, bf16, i32, i64, bool, str

@dataclass
class TensorType(TypeExpr):
    kind: str = "tens"  # tens, mat, vec, scal
    dtype: str = "f32"
    shape: List[str] = field(default_factory=list)

@dataclass
class CustomType(TypeExpr):
    name: str = ""

@dataclass
class ListType(TypeExpr):
    inner: TypeExpr = field(default_factory=PrimitiveType)

# =========================================================================
# Espressioni
# =========================================================================

@dataclass
class Expr(ASTNode):
    pass

@dataclass
class LiteralExpr(Expr):
    value: Any = None
    lit_type: str = "int"  # int, float, str, bool

@dataclass
class IdentExpr(Expr):
    name: str = ""

@dataclass
class CaseIdentExpr(Expr):
    name: str = ""
    case: str = "m"        # m, b, t, n, s, v
    base_name: str = ""

@dataclass
class BinaryExpr(Expr):
    op: str = "+"
    lhs: Expr = field(default_factory=Expr)
    rhs: Expr = field(default_factory=Expr)

@dataclass
class UnaryExpr(Expr):
    op: str = "-"
    operand: Expr = field(default_factory=Expr)

@dataclass
class CaseArg(ASTNode):
    case: str = "m"
    expr: Expr = field(default_factory=Expr)
    name: Optional[str] = None

@dataclass
class PipeStep(ASTNode):
    op: str = ""
    args: List[CaseArg] = field(default_factory=list)

@dataclass
class PipelineExpr(Expr):
    head: Expr = field(default_factory=Expr)
    steps: List[PipeStep] = field(default_factory=list)

@dataclass
class CallExpr(Expr):
    func: Expr = field(default_factory=Expr)
    args: List[Expr] = field(default_factory=list)

@dataclass
class AgentCallExpr(Expr):
    agent_name: str = ""
    prompt: Expr = field(default_factory=Expr)
    tool: Optional[Expr] = None
    timeout: Optional[Expr] = None
    extra_args: Dict[str, Expr] = field(default_factory=dict)

@dataclass
class DiffExpr(Expr):
    target: Expr = field(default_factory=Expr)
    wrt: Expr = field(default_factory=Expr)

@dataclass
class IndexExpr(Expr):
    target: Expr = field(default_factory=Expr)
    indices: List[Expr] = field(default_factory=list)

@dataclass
class FieldAccessExpr(Expr):
    target: Expr = field(default_factory=Expr)
    field_name: str = ""

@dataclass
class TupleExpr(Expr):
    elements: List[Expr] = field(default_factory=list)

@dataclass
class ListExpr(Expr):
    elements: List[Expr] = field(default_factory=list)

@dataclass
class RangeExpr(Expr):
    start: Expr = field(default_factory=Expr)
    end: Expr = field(default_factory=Expr)

# =========================================================================
# Istruzioni (Statements)
# =========================================================================

@dataclass
class Stmt(ASTNode):
    pass

@dataclass
class BindingStmt(Stmt):
    targets: List[str] = field(default_factory=list)
    expr: Expr = field(default_factory=Expr)
    is_mut: bool = False

@dataclass
class RetStmt(Stmt):
    expr: Optional[Expr] = None

@dataclass
class IfStmt(Stmt):
    cond: Expr = field(default_factory=Expr)
    then_branch: List[Stmt] = field(default_factory=list)
    else_branch: Optional[List[Stmt]] = None

@dataclass
class ForStmt(Stmt):
    var_name: str = ""
    iterable: Expr = field(default_factory=Expr)
    body: List[Stmt] = field(default_factory=list)

@dataclass
class LoopStmt(Stmt):
    body: List[Stmt] = field(default_factory=list)

@dataclass
class BreakStmt(Stmt):
    pass

@dataclass
class ContinueStmt(Stmt):
    pass

@dataclass
class ExprStmt(Stmt):
    expr: Expr = field(default_factory=Expr)

# =========================================================================
# Dichiarazioni di Livello Superiore (Top-Level)
# =========================================================================

@dataclass
class FieldDecl(ASTNode):
    name: str = ""
    case: Optional[str] = None
    type_expr: Optional[TypeExpr] = None

@dataclass
class Param(ASTNode):
    name: str = ""
    case: Optional[str] = None
    base_name: str = ""
    type_expr: Optional[TypeExpr] = None
    default: Optional[Expr] = None

@dataclass
class TypeDecl(ASTNode):
    name: str = ""
    fields: List[FieldDecl] = field(default_factory=list)

@dataclass
class SchemaDecl(ASTNode):
    name: str = ""
    inputs: List[Param] = field(default_factory=list)
    outputs: List[FieldDecl] = field(default_factory=list)

@dataclass
class FnDecl(ASTNode):
    name: str = ""
    params: List[Param] = field(default_factory=list)
    ret_type: Optional[TypeExpr] = None
    effect: str = "pure"
    body: List[Stmt] = field(default_factory=list)

@dataclass
class KernelDecl(ASTNode):
    name: str = ""
    params: List[Param] = field(default_factory=list)
    grid_expr: Expr = field(default_factory=Expr)
    block_expr: Expr = field(default_factory=Expr)
    body: List[Stmt] = field(default_factory=list)

@dataclass
class Program(ASTNode):
    declarations: List[ASTNode] = field(default_factory=list)
