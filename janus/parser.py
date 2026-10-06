"""
Parser deterministico a discesa ricorsiva per JANUS.
"""

from typing import List, Optional, Tuple, Any, Dict
from janus.lexer import Token, TokenType
from janus.ast_nodes import (
    Program, ASTNode, TypeDecl, FieldDecl, SchemaDecl, FnDecl, KernelDecl, Param,
    Stmt, BindingStmt, RetStmt, IfStmt, ForStmt, LoopStmt, BreakStmt, ContinueStmt, ExprStmt,
    Expr, LiteralExpr, IdentExpr, CaseIdentExpr, BinaryExpr, UnaryExpr,
    CaseArg, PipeStep, PipelineExpr, CallExpr, AgentCallExpr, ToolCallExpr, DiffExpr,
    IndexExpr, FieldAccessExpr, TupleExpr, ListExpr, RangeExpr,
    TypeExpr, PrimitiveType, TensorType, CustomType, ListType
)

class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        super().__init__(f"[{token.line}:{token.col}] {message} (token: '{token.value}')")
        self.message = message
        self.token = token

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.cursor = 0

    def _current(self) -> Token:
        if self.cursor < len(self.tokens):
            return self.tokens[self.cursor]
        return self.tokens[-1]

    def _peek(self, offset: int = 1) -> Token:
        idx = self.cursor + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def _match(self, *types: TokenType) -> bool:
        if self._current().type in types:
            self.cursor += 1
            return True
        return False

    def _expect(self, token_type: TokenType, err_msg: str) -> Token:
        tok = self._current()
        if tok.type == token_type:
            self.cursor += 1
            return tok
        raise ParseError(err_msg, tok)

    def _expect_ident(self, err_msg: str) -> Token:
        tok = self._current()
        if tok.type in (TokenType.IDENT, TokenType.CASE_IDENT):
            self.cursor += 1
            return tok
        raise ParseError(err_msg, tok)

    def parse(self) -> Program:
        decls = []
        while not self._match(TokenType.EOF):
            decls.append(self._parse_top_level())
        return Program(declarations=decls)

    def _parse_top_level(self) -> ASTNode:
        tok = self._current()
        if tok.type == TokenType.KW_TYPE:
            return self._parse_type_decl()
        elif tok.type == TokenType.KW_SCHEMA:
            return self._parse_schema_decl()
        elif tok.type == TokenType.KW_FN:
            return self._parse_fn_decl()
        elif tok.type == TokenType.KW_KERN:
            return self._parse_kernel_decl()
        else:
            return self._parse_stmt()

    # =========================================================================
    # Dichiarazioni
    # =========================================================================

    def _parse_type_decl(self) -> TypeDecl:
        kw = self._expect(TokenType.KW_TYPE, "Atteso 'type'")
        name_tok = self._expect_ident("Atteso nome tipo")
        self._expect(TokenType.LBRACE, "Atteso '{' nella dichiarazione type")
        fields = []
        while not self._match(TokenType.RBRACE):
            fname = self._expect_ident("Atteso nome campo")
            case = fname.case if fname.type == TokenType.CASE_IDENT else None
            
            ftype = None
            if self._match(TokenType.COLON):
                ftype = self._parse_type_expr()
            elif self._current().type in (TokenType.IDENT, TokenType.LBRACKET):
                ftype = self._parse_type_expr()
            
            fields.append(FieldDecl(name=fname.value, case=case, type_expr=ftype, line=fname.line, col=fname.col))
            self._match(TokenType.COMMA)
            self._match(TokenType.SEMICOLON)
        return TypeDecl(name=name_tok.value, fields=fields, line=kw.line, col=kw.col)

    def _parse_schema_decl(self) -> SchemaDecl:
        kw = self._expect(TokenType.KW_SCHEMA, "Atteso 'schema'")
        name_tok = self._expect_ident("Atteso nome schema")
        effect = "io"
        if self._current().type in (TokenType.EFF_PURE, TokenType.EFF_IO, TokenType.EFF_STOC):
            effect = self._current().value
            self.cursor += 1

        self._expect(TokenType.LBRACE, "Atteso '{'")
        inputs = []
        while not self._match(TokenType.RBRACE):
            pname = self._expect_ident("Atteso nome parametro schema")
            self._expect(TokenType.COLON, "Atteso ':'")
            ptype = self._parse_type_expr()
            pdefault = None
            if self._match(TokenType.ASSIGN):
                pdefault = self._parse_expr()
            inputs.append(Param(name=pname.value, type_expr=ptype, default=pdefault, line=pname.line, col=pname.col))
            self._match(TokenType.COMMA)

        self._expect(TokenType.ARROW, "Atteso '->' tra input e output dello schema")
        self._expect(TokenType.LBRACE, "Atteso '{' per output schema")
        outputs = []
        while not self._match(TokenType.RBRACE):
            oname = self._expect_ident("Atteso nome campo output")
            self._expect(TokenType.COLON, "Atteso ':'")
            otype = self._parse_type_expr()
            outputs.append(FieldDecl(name=oname.value, type_expr=otype, line=oname.line, col=oname.col))
            self._match(TokenType.COMMA)

        return SchemaDecl(name=name_tok.value, effect=effect, inputs=inputs, outputs=outputs, line=kw.line, col=kw.col)

    def _parse_fn_decl(self) -> FnDecl:
        kw = self._expect(TokenType.KW_FN, "Atteso 'fn'")
        name_tok = self._expect_ident("Atteso nome funzione")
        self._expect(TokenType.LPAREN, "Atteso '('")
        params = []
        while not self._match(TokenType.RPAREN):
            ptok = self._current()
            is_mut = self._match(TokenType.KW_MUT)
            pname_tok = self._expect_ident("Atteso identificatore parametro")
            case = pname_tok.case if pname_tok.type == TokenType.CASE_IDENT else None
            base = pname_tok.base_name if pname_tok.type == TokenType.CASE_IDENT else pname_tok.value
            
            ptype = None
            if self._match(TokenType.COLON):
                ptype = self._parse_type_expr()
            
            pdef = None
            if self._match(TokenType.ASSIGN):
                pdef = self._parse_expr()

            params.append(Param(
                name=pname_tok.value, case=case, base_name=base,
                type_expr=ptype, default=pdef, line=pname_tok.line, col=pname_tok.col
            ))
            self._match(TokenType.COMMA)

        ret_type = None
        if self._match(TokenType.ARROW):
            ret_type = self._parse_type_expr()

        effect = "pure"
        if self._current().type in (TokenType.EFF_PURE, TokenType.EFF_IO, TokenType.EFF_STOC):
            eff_tok = self._current()
            self.cursor += 1
            effect = eff_tok.value

        body = self._parse_block()
        return FnDecl(name=name_tok.value, params=params, ret_type=ret_type, effect=effect, body=body, line=kw.line, col=kw.col)

    def _parse_kernel_decl(self) -> KernelDecl:
        kw = self._expect(TokenType.KW_KERN, "Atteso 'kern'")
        name_tok = self._expect(TokenType.IDENT, "Atteso nome kernel")
        self._expect(TokenType.LPAREN, "Atteso '('")
        params = []
        while not self._match(TokenType.RPAREN):
            ptok = self._current()
            case = ptok.case if ptok.type == TokenType.CASE_IDENT else None
            base = ptok.base_name if ptok.type == TokenType.CASE_IDENT else ptok.value
            self.cursor += 1
            ptype = None
            if self._match(TokenType.COLON):
                ptype = self._parse_type_expr()
            params.append(Param(name=ptok.value, case=case, base_name=base, type_expr=ptype, line=ptok.line, col=ptok.col))
            self._match(TokenType.COMMA)

        self._expect(TokenType.KW_ON, "Atteso 'on'")
        self._expect(TokenType.KW_GPU, "Atteso 'gpu'")
        self._expect(TokenType.LBRACKET, "Atteso '[' per grid e block")
        grid = self._parse_expr()
        self._expect(TokenType.COMMA, "Atteso ','")
        block_dim = self._parse_expr()
        self._expect(TokenType.RBRACKET, "Atteso ']'")
        self._match(TokenType.COLON)
        body = self._parse_block()
        return KernelDecl(name=name_tok.value, params=params, grid_expr=grid, block_expr=block_dim, body=body, line=kw.line, col=kw.col)

    # =========================================================================
    # Tipi
    # =========================================================================

    def _parse_type_expr(self) -> TypeExpr:
        tok = self._current()
        if tok.value in ("tens", "mat", "vec", "scal"):
            self.cursor += 1
            kind = tok.value
            self._expect(TokenType.LBRACKET, f"Atteso '[' dopo {kind}")
            dtype_tok = self._expect(TokenType.IDENT, "Atteso dtype primitivo")
            shapes = []
            while self._match(TokenType.COMMA):
                # dimensione simbolica o numerica
                s = self._parse_dim_expr()
                shapes.append(s)
            self._expect(TokenType.RBRACKET, "Atteso ']' chiusura tipo tensore")
            return TensorType(kind=kind, dtype=dtype_tok.value, shape=shapes, line=tok.line, col=tok.col)
        
        if tok.value in ("f32", "f16", "bf16", "i32", "i64", "i8", "bool", "str", "Seed"):
            self.cursor += 1
            return PrimitiveType(name=tok.value, line=tok.line, col=tok.col)

        self.cursor += 1
        return CustomType(name=tok.value, line=tok.line, col=tok.col)

    def _parse_dim_expr(self) -> str:
        # Parsa dimensione numerica o simbolica semplice
        parts = []
        while self._current().type not in (TokenType.COMMA, TokenType.RBRACKET, TokenType.EOF):
            parts.append(self._current().value)
            self.cursor += 1
        return "".join(parts)

    # =========================================================================
    # Blocchi e Istruzioni
    # =========================================================================

    def _parse_block(self) -> List[Stmt]:
        stmts = []
        if self._match(TokenType.LBRACE):
            while not self._match(TokenType.RBRACE) and not self._match(TokenType.EOF):
                stmts.append(self._parse_stmt())
                self._match(TokenType.SEMICOLON)
        elif self._match(TokenType.COLON):
            stmts.append(self._parse_stmt())
            self._match(TokenType.SEMICOLON)
        else:
            tok = self._current()
            raise ParseError("Atteso '{' o ':' per inizio blocco", tok)
        return stmts

    def _parse_stmt(self) -> Stmt:
        tok = self._current()

        if self._match(TokenType.KW_RET):
            expr = None
            if self._current().type not in (TokenType.SEMICOLON, TokenType.RBRACE, TokenType.EOF):
                expr = self._parse_expr()
            self._match(TokenType.SEMICOLON)
            return RetStmt(expr=expr, line=tok.line, col=tok.col)

        if self._match(TokenType.KW_IF):
            cond = self._parse_expr()
            then_b = self._parse_block()
            else_b = None
            if self._match(TokenType.KW_ELSE):
                else_b = self._parse_block()
            return IfStmt(cond=cond, then_branch=then_b, else_branch=else_b, line=tok.line, col=tok.col)

        if self._match(TokenType.KW_FOR):
            var_name = self._expect(TokenType.IDENT, "Atteso identificatore loop").value
            self._expect(TokenType.KW_IN, "Atteso 'in'")
            iterable = self._parse_expr()
            body = self._parse_block()
            return ForStmt(var_name=var_name, iterable=iterable, body=body, line=tok.line, col=tok.col)

        if self._match(TokenType.KW_LOOP):
            body = self._parse_block()
            return LoopStmt(body=body, line=tok.line, col=tok.col)

        if self._match(TokenType.KW_BRK):
            self._match(TokenType.SEMICOLON)
            return BreakStmt(line=tok.line, col=tok.col)

        if self._match(TokenType.KW_CONT):
            self._match(TokenType.SEMICOLON)
            return ContinueStmt(line=tok.line, col=tok.col)

        # Gestione mut x = expr
        if self._match(TokenType.KW_MUT):
            target = self._current().value
            self.cursor += 1
            self._expect(TokenType.ASSIGN, "Atteso '=' dopo mut identifier")
            val = self._parse_expr()
            self._match(TokenType.SEMICOLON)
            return BindingStmt(targets=[target], expr=val, is_mut=True, line=tok.line, col=tok.col)

        # Verifica se è un binding: var = expr oppure (a, b) = expr oppure a, b = expr
        # Lookahead per '='
        saved = self.cursor
        is_binding = False
        targets = []

        if tok.type in (TokenType.IDENT, TokenType.CASE_IDENT):
            target_str = tok.value
            self.cursor += 1
            if self._match(TokenType.LBRACKET):
                idx_parts = []
                while not self._match(TokenType.RBRACKET) and not self._match(TokenType.EOF):
                    idx_parts.append(self._current().value)
                    self.cursor += 1
                target_str = f"{target_str}[{''.join(idx_parts)}]"
            targets.append(target_str)
            while self._match(TokenType.COMMA):
                t = self._current()
                if t.type in (TokenType.IDENT, TokenType.CASE_IDENT):
                    targets.append(t.value)
                    self.cursor += 1
                else:
                    break
            if self._match(TokenType.ASSIGN):
                is_binding = True
        elif tok.type == TokenType.LPAREN:
            self.cursor += 1
            while not self._match(TokenType.RPAREN) and not self._match(TokenType.EOF):
                t = self._current()
                if t.type in (TokenType.IDENT, TokenType.CASE_IDENT):
                    targets.append(t.value)
                    self.cursor += 1
                self._match(TokenType.COMMA)
            if self._match(TokenType.ASSIGN):
                is_binding = True

        if is_binding:
            val = self._parse_expr()
            self._match(TokenType.SEMICOLON)
            return BindingStmt(targets=targets, expr=val, is_mut=False, line=tok.line, col=tok.col)

        # Fallback a espressione pura
        self.cursor = saved
        e = self._parse_expr()
        self._match(TokenType.SEMICOLON)
        return ExprStmt(expr=e, line=tok.line, col=tok.col)

    # =========================================================================
    # Espressioni e Pipeline Chaining
    # =========================================================================

    def _parse_expr(self) -> Expr:
        return self._parse_pipeline_or_binary()

    def _parse_pipeline_or_binary(self) -> Expr:
        head = self._parse_binary_expr()
        
        while True:
            steps = []
            while True:
                cur = self._current()
                if cur.type in (
                    TokenType.OP_MATMUL, TokenType.OP_DOT, TokenType.OP_ADD, TokenType.OP_SUB,
                    TokenType.OP_MUL, TokenType.OP_DIV, TokenType.OP_RELU, TokenType.OP_SMAX,
                    TokenType.OP_GELU, TokenType.OP_NORM, TokenType.OP_CONV2D, TokenType.OP_POOL,
                    TokenType.OP_SUM, TokenType.OP_POW, TokenType.OP_SQRT, TokenType.OP_EXP,
                    TokenType.OP_LOG, TokenType.OP_FLAT, TokenType.OP_TRANS
                ):
                    op_tok = cur
                    self.cursor += 1
                    args = self._parse_case_args()
                    steps.append(PipeStep(op=op_tok.value, args=args, line=op_tok.line, col=op_tok.col))
                else:
                    break

            if steps:
                head = PipelineExpr(head=head, steps=steps, line=head.line, col=head.col)

            # Controlla se continua con un operatore binario (+, -, *, /, @, ..)
            cur = self._current()
            op_prec = self._get_precedence(cur.type)
            if op_prec > 0:
                op_tok = cur
                self.cursor += 1
                rhs = self._parse_binary_expr(op_prec)
                head = BinaryExpr(op=op_tok.value, lhs=head, rhs=rhs, line=op_tok.line, col=op_tok.col)
            else:
                break

        return head

    def _parse_case_args(self) -> List[CaseArg]:
        args = []
        while True:
            cur = self._current()
            if self._peek(1).type == TokenType.ASSIGN:
                break
            if cur.type == TokenType.CASE_IDENT:
                if cur.case == "n":
                    break
                case_val = cur.case or "m"
                arg_expr = self._parse_postfix_expr()
                args.append(CaseArg(case=case_val, expr=arg_expr, line=cur.line, col=cur.col))
            elif cur.type == TokenType.IDENT:
                if self._peek(1).type in (TokenType.ASSIGN, TokenType.COMMA):
                    break
                if self.cursor > 0 and cur.line > self.tokens[self.cursor - 1].line:
                    break
                arg_expr = self._parse_postfix_expr()
                args.append(CaseArg(case="b", expr=arg_expr, line=cur.line, col=cur.col))
            elif cur.type == TokenType.MINUS and self._peek(1).type in (TokenType.LIT_INT, TokenType.LIT_FLOAT):
                self.cursor += 2
                nxt = self.tokens[self.cursor - 1]
                val = -float(nxt.value) if '.' in nxt.value or 'e' in nxt.value else -int(nxt.value)
                args.append(CaseArg(case="b", expr=LiteralExpr(value=val, lit_type="num", line=cur.line, col=cur.col), line=cur.line, col=cur.col))
            elif cur.type in (TokenType.LIT_INT, TokenType.LIT_FLOAT):
                self.cursor += 1
                val = float(cur.value) if '.' in cur.value or 'e' in cur.value else int(cur.value)
                args.append(CaseArg(case="b", expr=LiteralExpr(value=val, lit_type="num", line=cur.line, col=cur.col), line=cur.line, col=cur.col))
            elif cur.type == TokenType.LIT_BOOL:
                self.cursor += 1
                args.append(CaseArg(case="b", expr=LiteralExpr(value=(cur.value == "true"), lit_type="bool", line=cur.line, col=cur.col), line=cur.line, col=cur.col))
            elif cur.type == TokenType.LBRACKET:
                # lista come argomento [1, 1]
                list_expr = self._parse_primary()
                args.append(CaseArg(case="b", expr=list_expr, line=cur.line, col=cur.col))
            else:
                break
        return args

    def _parse_binary_expr(self, prec: int = 0) -> Expr:
        lhs = self._parse_unary_expr()
        
        while True:
            cur = self._current()
            op_prec = self._get_precedence(cur.type)
            if op_prec <= prec:
                break
            
            op_tok = cur
            self.cursor += 1

            # Gestione speciale del range ..
            if op_tok.type == TokenType.RANGE:
                rhs = self._parse_binary_expr(op_prec)
                lhs = RangeExpr(start=lhs, end=rhs, line=op_tok.line, col=op_tok.col)
                continue

            rhs = self._parse_binary_expr(op_prec)
            lhs = BinaryExpr(op=op_tok.value, lhs=lhs, rhs=rhs, line=op_tok.line, col=op_tok.col)

        return lhs

    def _get_precedence(self, token_type: TokenType) -> int:
        precedences = {
            TokenType.RANGE: 1,
            TokenType.EQ: 2, TokenType.NEQ: 2, TokenType.LT: 2, TokenType.LTE: 2, TokenType.GT: 2, TokenType.GTE: 2,
            TokenType.PLUS: 3, TokenType.MINUS: 3,
            TokenType.STAR: 4, TokenType.SLASH: 4, TokenType.AT: 4,
        }
        return precedences.get(token_type, 0)

    def _parse_unary_expr(self) -> Expr:
        cur = self._current()
        if cur.type in (TokenType.MINUS,):
            self.cursor += 1
            op = self._parse_unary_expr()
            return UnaryExpr(op=cur.value, operand=op, line=cur.line, col=cur.col)
        return self._parse_postfix_expr()

    def _parse_postfix_expr(self) -> Expr:
        expr = self._parse_primary()

        while True:
            if self._match(TokenType.DOT):
                # Accesso a campo o metodo/proprietà unaria
                member = self._current()
                self.cursor += 1
                expr = FieldAccessExpr(target=expr, field_name=member.value, line=member.line, col=member.col)
            elif self._match(TokenType.LBRACKET):
                # Indicizzazione o slicing
                indices = []
                while not self._match(TokenType.RBRACKET) and not self._match(TokenType.EOF):
                    indices.append(self._parse_expr())
                    self._match(TokenType.COMMA)
                expr = IndexExpr(target=expr, indices=indices, line=expr.line, col=expr.col)
            elif self._current().type == TokenType.LPAREN and isinstance(expr, (IdentExpr, CaseIdentExpr)):
                self.cursor += 1
                args = []
                while not self._match(TokenType.RPAREN) and not self._match(TokenType.EOF):
                    args.append(self._parse_expr())
                    self._match(TokenType.COMMA)
                expr = CallExpr(func=expr, args=args, line=expr.line, col=expr.col)
            else:
                break

        return expr

    def _parse_primary(self) -> Expr:
        cur = self._current()

        # Letterali
        if cur.type == TokenType.LIT_INT:
            self.cursor += 1
            return LiteralExpr(value=int(cur.value), lit_type="int", line=cur.line, col=cur.col)
        if cur.type == TokenType.LIT_FLOAT:
            self.cursor += 1
            return LiteralExpr(value=float(cur.value), lit_type="float", line=cur.line, col=cur.col)
        if cur.type == TokenType.LIT_STR:
            self.cursor += 1
            return LiteralExpr(value=cur.value, lit_type="str", line=cur.line, col=cur.col)
        if cur.type == TokenType.LIT_BOOL:
            self.cursor += 1
            return LiteralExpr(value=(cur.value == "true"), lit_type="bool", line=cur.line, col=cur.col)

        # Gestione speciale autodiff: diff loss wrt target
        if cur.type == TokenType.OP_DIFF:
            diff_tok = cur
            self.cursor += 1
            target = self._parse_expr()
            self._expect(TokenType.KW_WRT, "Atteso 'wrt' dopo target di diff")
            wrt = self._parse_expr()
            return DiffExpr(target=target, wrt=wrt, line=diff_tok.line, col=diff_tok.col)

        # Gestione speciale chiamata: call tool ToolName(...) oppure call agentv Agent ...
        if cur.type == TokenType.KW_CALL:
            call_tok = cur
            self.cursor += 1

            # 1. Chiamata a Tool esplicita: call tool ToolName(arg = val, ...)
            if self._current().value == "tool":
                self.cursor += 1
                tool_tok = self._expect_ident("Atteso nome tool dopo 'call tool'")
                tool_name = tool_tok.value
                named_args = {}
                positional_args = []
                if self._match(TokenType.LPAREN):
                    while not self._match(TokenType.RPAREN):
                        tok = self._current()
                        if (tok.type in (TokenType.IDENT, TokenType.CASE_IDENT) and 
                            self._peek(1).type in (TokenType.ASSIGN, TokenType.COLON)):
                            arg_name = tok.base_name if tok.type == TokenType.CASE_IDENT else tok.value
                            self.cursor += 2
                            val = self._parse_expr()
                            named_args[arg_name] = val
                        else:
                            val = self._parse_expr()
                            positional_args.append(val)
                        self._match(TokenType.COMMA)
                return ToolCallExpr(
                    tool_name=tool_name,
                    named_args=named_args,
                    positional_args=positional_args,
                    line=call_tok.line,
                    col=call_tok.col
                )

            # 2. Chiamata ad Agente: call agentv Agent ... o call agent:v Agent ...
            if self._match(TokenType.KW_AGENTV) or (self._current().type == TokenType.CASE_IDENT and self._current().case == "v" and self._current().base_name == "agent"):
                if self._current().type == TokenType.CASE_IDENT:
                    self.cursor += 1
                agent_name = self._expect_ident("Atteso nome agente").value
                
                prompt_expr = None
                tool_expr = None
                timeout_expr = None
                extra = {}

                # Legge argomenti dell'agente
                while self._current().type in (TokenType.IDENT, TokenType.CASE_IDENT):
                    if self._peek(1).type == TokenType.ASSIGN:
                        break
                    arg_label = self._current()
                    is_label = any(k in arg_label.value for k in ("prompt", "tool", "timeout", "ctx")) or arg_label.type == TokenType.CASE_IDENT
                    if not is_label:
                        break
                    self.cursor += 1
                    val_expr = self._parse_postfix_expr()
                    
                    if "prompt" in arg_label.value:
                        prompt_expr = val_expr
                    elif "tool" in arg_label.value:
                        tool_expr = val_expr
                    elif "timeout" in arg_label.value:
                        timeout_expr = val_expr
                    else:
                        extra[arg_label.value] = val_expr

                return AgentCallExpr(
                    agent_name=agent_name, prompt=prompt_expr or LiteralExpr(value=""),
                    tool=tool_expr, timeout=timeout_expr, extra_args=extra,
                    line=call_tok.line, col=call_tok.col
                )

            raise ParseError("Atteso 'tool' o 'agentv' dopo 'call'", self._current())

        # Chiamata a funzione definita dall'utente in notazione prefissa (es. step x:b y:b model:t lr)
        AGENT_LABELS = {"promptm", "toolb", "timeoutb", "ctxb", "prompt:m", "tool:b", "timeout:b", "ctx:b"}
        if (cur.type == TokenType.IDENT and 
            self._peek(1).type == TokenType.CASE_IDENT and 
            self._peek(1).value not in AGENT_LABELS and
            self._peek(2).type != TokenType.ASSIGN):
            func_name = cur.value
            self.cursor += 1
            args = self._parse_case_args()
            return CallExpr(
                func=IdentExpr(name=func_name, line=cur.line, col=cur.col),
                args=[a.expr for a in args],
                line=cur.line, col=cur.col
            )

        # Operazioni intrinseche in notazione prefissa (es. dot am bb, conv2d imgm kb ..., norm xm ...)
        if cur.type in (
            TokenType.OP_DOT, TokenType.OP_MATMUL, TokenType.OP_CONV2D,
            TokenType.OP_NORM, TokenType.OP_POOL, TokenType.OP_RAND, TokenType.OP_SUM
        ):
            op_tok = cur
            self.cursor += 1
            args = self._parse_case_args()
            if args:
                first_arg = args[0].expr
                return PipelineExpr(head=first_arg, steps=[PipeStep(op=op_tok.value, args=args[1:], line=op_tok.line, col=op_tok.col)], line=op_tok.line, col=op_tok.col)
            return IdentExpr(name=op_tok.value, line=op_tok.line, col=op_tok.col)

        # Identificatori
        if cur.type == TokenType.CASE_IDENT:
            self.cursor += 1
            return CaseIdentExpr(
                name=cur.value, case=cur.case or "m", base_name=cur.base_name or cur.value,
                line=cur.line, col=cur.col
            )
        if cur.type == TokenType.IDENT:
            self.cursor += 1
            return IdentExpr(name=cur.value, line=cur.line, col=cur.col)

        # Parentesi / Tuple
        if self._match(TokenType.LPAREN):
            elements = []
            while not self._match(TokenType.RPAREN) and not self._match(TokenType.EOF):
                elements.append(self._parse_expr())
                self._match(TokenType.COMMA)
            if len(elements) == 1:
                return elements[0]
            return TupleExpr(elements=elements, line=cur.line, col=cur.col)

        # Liste [a, b, c]
        if self._match(TokenType.LBRACKET):
            elements = []
            while not self._match(TokenType.RBRACKET) and not self._match(TokenType.EOF):
                elements.append(self._parse_expr())
                self._match(TokenType.COMMA)
            return ListExpr(elements=elements, line=cur.line, col=cur.col)

        raise ParseError(f"Token inatteso: '{cur.value}'", cur)
