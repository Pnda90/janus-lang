"""
Lexer per JANUS: tokenizzazione e riconoscimento della morfologia dei casi.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional

class TokenType(Enum):
    # Keywords di controllo e definizione
    KW_FN = auto()
    KW_RET = auto()
    KW_LET = auto()
    KW_MUT = auto()
    KW_TYPE = auto()
    KW_SCHEMA = auto()
    KW_KERN = auto()
    KW_IF = auto()
    KW_ELSE = auto()
    KW_LOOP = auto()
    KW_FOR = auto()
    KW_IN = auto()
    KW_BRK = auto()
    KW_CONT = auto()
    KW_ON = auto()
    KW_GPU = auto()
    KW_CALL = auto()
    KW_AGENTV = auto()
    KW_TOOL = auto()
    KW_WRT = auto()

    # Effetti
    EFF_PURE = auto()
    EFF_IO = auto()
    EFF_MUT = auto()
    EFF_STOC = auto()

    # Operazioni Intrinseche
    OP_MATMUL = auto()
    OP_DOT = auto()
    OP_ADD = auto()
    OP_SUB = auto()
    OP_MUL = auto()
    OP_DIV = auto()
    OP_RELU = auto()
    OP_SMAX = auto()
    OP_GELU = auto()
    OP_NORM = auto()
    OP_CONV2D = auto()
    OP_POOL = auto()
    OP_SUM = auto()
    OP_POW = auto()
    OP_SQRT = auto()
    OP_EXP = auto()
    OP_LOG = auto()
    OP_FLAT = auto()
    OP_TRANS = auto()
    OP_DIFF = auto()
    OP_ALLOC = auto()
    OP_COPY = auto()
    OP_FREE = auto()
    OP_RAND = auto()

    # Identificatori e Morfologia
    IDENT = auto()          # Identificatore generico
    CASE_IDENT = auto()     # Identificatore con caso saldato (base, case)
    
    # Letterali
    LIT_INT = auto()
    LIT_FLOAT = auto()
    LIT_STR = auto()
    LIT_BOOL = auto()

    # Simboli e Operatori
    ASSIGN = auto()         # =
    PLUS = auto()           # +
    MINUS = auto()          # -
    STAR = auto()           # *
    SLASH = auto()          # /
    AT = auto()             # @
    EQ = auto()             # ==
    NEQ = auto()            # !=
    LT = auto()             # <
    LTE = auto()            # <=
    GT = auto()             # >
    GTE = auto()            # >=
    ARROW = auto()          # ->
    COLON = auto()          # :
    SEMICOLON = auto()      # ;
    COMMA = auto()          # ,
    DOT = auto()            # .
    RANGE = auto()          # ..
    
    # Delimitatori
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    LBRACE = auto()         # {
    RBRACE = auto()         # }
    LBRACKET = auto()       # [
    RBRACKET = auto()       # ]

    EOF = auto()

KEYWORDS = {
    "fn": TokenType.KW_FN,
    "ret": TokenType.KW_RET,
    "let": TokenType.KW_LET,
    "mut": TokenType.KW_MUT,
    "type": TokenType.KW_TYPE,
    "schema": TokenType.KW_SCHEMA,
    "kern": TokenType.KW_KERN,
    "if": TokenType.KW_IF,
    "else": TokenType.KW_ELSE,
    "loop": TokenType.KW_LOOP,
    "for": TokenType.KW_FOR,
    "in": TokenType.KW_IN,
    "brk": TokenType.KW_BRK,
    "cont": TokenType.KW_CONT,
    "on": TokenType.KW_ON,
    "gpu": TokenType.KW_GPU,
    "call": TokenType.KW_CALL,
    "agentv": TokenType.KW_AGENTV,
    "wrt": TokenType.KW_WRT,
    
    "pure": TokenType.EFF_PURE,
    "io": TokenType.EFF_IO,
    "stoc": TokenType.EFF_STOC,

    "matmul": TokenType.OP_MATMUL,
    "dot": TokenType.OP_DOT,
    "add": TokenType.OP_ADD,
    "sub": TokenType.OP_SUB,
    "mul": TokenType.OP_MUL,
    "div": TokenType.OP_DIV,
    "relu": TokenType.OP_RELU,
    "smax": TokenType.OP_SMAX,
    "gelu": TokenType.OP_GELU,
    "norm": TokenType.OP_NORM,
    "conv2d": TokenType.OP_CONV2D,
    "pool": TokenType.OP_POOL,
    "sum": TokenType.OP_SUM,
    "pow": TokenType.OP_POW,
    "sqrt": TokenType.OP_SQRT,
    "exp": TokenType.OP_EXP,
    "log": TokenType.OP_LOG,
    "flat": TokenType.OP_FLAT,
    "trans": TokenType.OP_TRANS,
    "diff": TokenType.OP_DIFF,
    "alloc": TokenType.OP_ALLOC,
    "copy": TokenType.OP_COPY,
    "free": TokenType.OP_FREE,
    "rand": TokenType.OP_RAND,

    "true": TokenType.LIT_BOOL,
    "false": TokenType.LIT_BOOL,
}

CASE_LETTERS = {'m', 'b', 't', 'n', 's', 'v'}

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int
    length: int
    case: Optional[str] = None       # 'm', 'b', 't', 'n', 's', 'v' se CASE_IDENT
    base_name: Optional[str] = None  # Nome base senza desinenza

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.cursor = 0
        self.line = 1
        self.col = 1

    def _peek(self, offset: int = 0) -> str:
        idx = self.cursor + offset
        if idx < self.length:
            return self.source[idx]
        return '\0'

    def _advance(self) -> str:
        ch = self._peek()
        self.cursor += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self) -> List[Token]:
        tokens = []
        while self.cursor < self.length:
            start_col = self.col
            start_line = self.line
            ch = self._peek()

            # Salta spazi e tabulazioni
            if ch in ' \t\r\n':
                self._advance()
                continue

            # Commenti tipo // o #
            if ch == '#' or (ch == '/' and self._peek(1) == '/'):
                while self._peek() not in ('\n', '\0'):
                    self._advance()
                continue

            # Simboli composti
            if ch == '-' and self._peek(1) == '>':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.ARROW, "->", start_line, start_col, 2))
                continue
            if ch == '.' and self._peek(1) == '.':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.RANGE, "..", start_line, start_col, 2))
                continue
            if ch == '=' and self._peek(1) == '=':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.EQ, "==", start_line, start_col, 2))
                continue
            if ch == '!' and self._peek(1) == '=':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.NEQ, "!=", start_line, start_col, 2))
                continue
            if ch == '<' and self._peek(1) == '=':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.LTE, "<=", start_line, start_col, 2))
                continue
            if ch == '>' and self._peek(1) == '=':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.GTE, ">=", start_line, start_col, 2))
                continue

            # Simboli a singolo carattere
            single_symbols = {
                '=': TokenType.ASSIGN,
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.STAR,
                '/': TokenType.SLASH,
                '@': TokenType.AT,
                '<': TokenType.LT,
                '>': TokenType.GT,
                ':': TokenType.COLON,
                ';': TokenType.SEMICOLON,
                ',': TokenType.COMMA,
                '.': TokenType.DOT,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
            }

            if ch in single_symbols:
                self._advance()
                tokens.append(Token(single_symbols[ch], ch, start_line, start_col, 1))
                continue

            # Stringhe letterali
            if ch == '"':
                self._advance()
                str_val = []
                while self._peek() != '"' and self._peek() != '\0':
                    if self._peek() == '\\':
                        self._advance()
                    str_val.append(self._advance())
                if self._peek() == '"':
                    self._advance()
                text = "".join(str_val)
                tokens.append(Token(TokenType.LIT_STR, text, start_line, start_col, len(text) + 2))
                continue

            # Numeri (Interi e Float, inclusa notazione scientifica tipo 1e-5)
            if ch.isdigit():
                num_chars = []
                is_float = False
                while self._peek().isdigit() or self._peek() in ('.', 'e', 'E'):
                    if self._peek() == '.' and self._peek(1) == '.':
                        # Interrompi se è un operatore di range ..
                        break
                    if self._peek() == '.':
                        is_float = True
                    if self._peek() in ('e', 'E'):
                        is_float = True
                        num_chars.append(self._advance())
                        if self._peek() in ('+', '-'):
                            num_chars.append(self._advance())
                        continue
                    num_chars.append(self._advance())
                num_str = "".join(num_chars)
                ttype = TokenType.LIT_FLOAT if is_float else TokenType.LIT_INT
                tokens.append(Token(ttype, num_str, start_line, start_col, len(num_str)))
                continue

            # Identificatori, parole chiave e casi morfologici
            if ch.isalpha() or ch == '_':
                ident_chars = []
                while self._peek().isalnum() or self._peek() == '_':
                    ident_chars.append(self._advance())
                ident_str = "".join(ident_chars)

                # Verifica keyword esatta
                if ident_str in KEYWORDS:
                    tokens.append(Token(KEYWORDS[ident_str], ident_str, start_line, start_col, len(ident_str)))
                    continue

                # Verifica se seguito da :case (es. x:m o w:b o loss:m)
                if self._peek() == ':' and self._peek(1) in CASE_LETTERS and not (self._peek(2).isalnum() or self._peek(2) == '_'):
                    self._advance() # salta ':'
                    case_char = self._advance() # consuma lettera di caso
                    tokens.append(Token(
                        TokenType.CASE_IDENT, f"{ident_str}:{case_char}",
                        start_line, start_col, len(ident_str) + 2,
                        case=case_char, base_name=ident_str
                    ))
                    continue

                # Verifica se seguito da .case (es. dim.s o x.b)
                if self._peek() == '.' and self._peek(1) in CASE_LETTERS and not (self._peek(2).isalnum() or self._peek(2) == '_'):
                    self._advance() # salta '.'
                    case_char = self._advance()
                    tokens.append(Token(
                        TokenType.CASE_IDENT, f"{ident_str}.{case_char}",
                        start_line, start_col, len(ident_str) + 2,
                        case=case_char, base_name=ident_str
                    ))
                    continue

                # Identificatore generico (nessun troncamento euristico)
                tokens.append(Token(TokenType.IDENT, ident_str, start_line, start_col, len(ident_str)))
                continue

            # Carattere sconosciuto
            self._advance()

        tokens.append(Token(TokenType.EOF, "", self.line, self.col, 0))
        return tokens
