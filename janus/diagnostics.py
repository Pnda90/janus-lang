"""
Modulo di diagnostica strutturata LLM-Friendly per JANUS.
Emette errori sia in formato JSON canonico con patch correttiva, sia per CLI.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class SourceSpan:
    line: int
    col: int
    length: int

@dataclass
class DiagnosticPatch:
    target: str
    replacement: str

@dataclass
class Diagnostic:
    code: str
    phase: str             # "lexer", "parser", "type_check", "codegen"
    message: str
    span: SourceSpan
    offending: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    patch: Optional[DiagnosticPatch] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "status": "error",
            "code": self.code,
            "phase": self.phase,
            "span": asdict(self.span),
            "offending": self.offending,
            "message": self.message,
        }
        if self.expected is not None:
            d["expected"] = self.expected
        if self.actual is not None:
            d["actual"] = self.actual
        if self.patch is not None:
            d["patch"] = asdict(self.patch)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_cli(self, source_lines: Optional[list] = None) -> str:
        lines = [
            f"[\033[1;31mERROR\033[0m] {self.code}: {self.message}",
            f"  --> riga {self.span.line}, colonna {self.span.col} (token: '{self.offending}')"
        ]
        if source_lines and 1 <= self.span.line <= len(source_lines):
            src_line = source_lines[self.span.line - 1]
            lines.append(f"   |")
            lines.append(f"{self.span.line:3d}| {src_line}")
            indent = " " * (self.span.col - 1)
            marker = "^" * max(1, self.span.length)
            lines.append(f"   | {indent}\033[1;31m{marker}\033[0m")
        if self.patch:
            lines.append(f"  \033[1;32mSuggerimento Correttivo (Patch):\033[0m Sostituisci '{self.patch.target}' con '{self.patch.replacement}'")
        return "\n".join(lines)
