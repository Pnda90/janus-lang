"""
Runtime di esecuzione e sandbox per agenti e tool in JANUS.
Fase A.3: Fornisce isolamento, registrazione tipizzata, tracciamento granulare
e intercettazione dry-run per chiamate a tool e schemi agentici.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple

from janus.lexer import Lexer
from janus.parser import Parser
from janus.type_checker import TypeChecker
from janus.codegen import CodeGenerator


class ToolNotFoundError(Exception):
    """Sollevata quando un tool invocato non è registrato nella sandbox."""
    pass


class ToolExecutionError(Exception):
    """Sollevata quando l'esecuzione di un tool fallisce all'interno della sandbox."""
    pass


class ToolValidationError(Exception):
    """Sollevata quando gli argomenti forniti a un tool violano lo schema."""
    pass


@dataclass
class ToolTrace:
    """Rappresenta una singola esecuzione di un tool catturata dalla sandbox."""
    tool_name: str
    inputs: Dict[str, Any]
    output: Any = None
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    effect: str = "io"


class ToolSandbox:
    """
    Sandbox per l'esecuzione sicura e isolata di tool.
    Gestisce la registrazione delle funzioni, la validazione dei vincoli
    e la registrazione dettagliata di ogni invocazione (audit trail).
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._traces: List[ToolTrace] = []
        self._scope_globals: Optional[Dict[str, Any]] = None

    def register_tool(
        self,
        name: str,
        func: Callable,
        effect: str = "io",
        schema: Optional[Any] = None
    ) -> None:
        """Registra un tool callable con i relativi metadati di effetto e schema."""
        self._tools[name] = {
            "func": func,
            "effect": effect,
            "schema": schema,
        }

    def tool(self, name: str, effect: str = "io"):
        """Decorator helper per registrare un tool."""
        def decorator(fn: Callable):
            self.register_tool(name, fn, effect=effect)
            return fn
        return decorator

    def call_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Esegue un tool registrato all'interno della sandbox catturando metriche e anomalie.
        Se dry_run è attivo e il tool non è registrato, sintetizza un output mock.
        """
        start = time.perf_counter()

        if tool_name not in self._tools:
            if self.dry_run:
                # Mock fallback in dry run
                mock_out = None
                if self._scope_globals:
                    out_cls = self._scope_globals.get(f"{tool_name}Output")
                    if out_cls and hasattr(out_cls, "__dataclass_fields__"):
                        dummy_args = {}
                        for fname, ffield in out_cls.__dataclass_fields__.items():
                            t = getattr(ffield, "type", Any)
                            if t in (int, "int", "i32", "i64"):
                                dummy_args[fname] = 0
                            elif t in (float, "float", "f32", "f64"):
                                dummy_args[fname] = 0.0
                            elif t in (str, "str"):
                                dummy_args[fname] = ""
                            elif t in (bool, "bool"):
                                dummy_args[fname] = False
                            elif t in (list, "list", "List"):
                                dummy_args[fname] = []
                            else:
                                dummy_args[fname] = None
                        try:
                            mock_out = out_cls(**dummy_args)
                        except Exception:
                            mock_out = None
                if mock_out is None:
                    mock_out = {"tool": tool_name, "mock": True, "inputs": kwargs}
                duration_ms = (time.perf_counter() - start) * 1000.0
                trace = ToolTrace(
                    tool_name=tool_name,
                    inputs=kwargs,
                    output=mock_out,
                    duration_ms=duration_ms,
                    success=True,
                    effect="io"
                )
                self._traces.append(trace)
                return mock_out
            raise ToolNotFoundError(f"Tool '{tool_name}' is not registered in the sandbox")

        entry = self._tools[tool_name]
        fn = entry["func"]
        effect = entry.get("effect", "io")

        try:
            res = fn(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000.0
            trace = ToolTrace(
                tool_name=tool_name,
                inputs=kwargs,
                output=res,
                duration_ms=duration_ms,
                success=True,
                effect=effect
            )
            self._traces.append(trace)
            return res
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            trace = ToolTrace(
                tool_name=tool_name,
                inputs=kwargs,
                error=str(e),
                duration_ms=duration_ms,
                success=False,
                effect=effect
            )
            self._traces.append(trace)
            raise ToolExecutionError(f"Tool '{tool_name}' failed: {e}") from e

    def get_traces(self) -> List[ToolTrace]:
        """Restituisce la cronologia completa dei trace registrati."""
        return list(self._traces)

    def clear_traces(self) -> None:
        """Pulisce la cronologia dei trace."""
        self._traces.clear()


class AgentRuntime:
    """
    Coordinatore di alto livello per compilare ed eseguire piani agentici in JANUS.
    Garantisce il binding con la ToolSandbox e produce trace di esecuzione deterministici.
    """

    def __init__(self, sandbox: Optional[ToolSandbox] = None):
        self.sandbox = sandbox or ToolSandbox()

    def register_tool(self, name: str, func: Callable, effect: str = "io") -> None:
        """Registra un tool direttamente nella sandbox associata."""
        self.sandbox.register_tool(name, func, effect=effect)

    def compile(self, code: str) -> str:
        """Compila codice sorgente JANUS in Python garantendo l'assenza di errori di tipo."""
        tokens = Lexer(code).tokenize()
        ast = Parser(tokens).parse()
        checker = TypeChecker()
        diags = checker.check(ast)
        errors = [d for d in diags if d.code.startswith("ERR")]
        if errors:
            raise ValueError(f"Agent plan compilation failed: {[e.message for e in errors]}")
        return CodeGenerator().generate(ast)

    def execute(
        self,
        code: str,
        entrypoint: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        dry_run: bool = False
    ) -> Tuple[Any, List[ToolTrace]]:
        """
        Compila ed esegue una funzione agentica catturando tutti i tool trace.
        """
        prev_dry_run = self.sandbox.dry_run
        self.sandbox.dry_run = dry_run
        self.sandbox.clear_traces()

        py_code = self.compile(code)

        scope_globals: Dict[str, Any] = {}
        exec(py_code, scope_globals)

        # Bridge con la sandbox
        self.sandbox._scope_globals = scope_globals
        scope_globals["_janus_call_tool"] = self.sandbox.call_tool

        if entrypoint not in scope_globals:
            raise AttributeError(f"Entrypoint function '{entrypoint}' not found in compiled plan")

        fn = scope_globals[entrypoint]
        fn_args = args or []
        fn_kwargs = kwargs or {}

        try:
            result = fn(*fn_args, **fn_kwargs)
            return result, self.sandbox.get_traces()
        finally:
            self.sandbox.dry_run = prev_dry_run
