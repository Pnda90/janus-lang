"""
Runtime di esecuzione e sandbox per agenti e tool in JANUS.
Fase A.3 / Revisione finale: Fornisce isolamento, registrazione tipizzata, tracciamento granulare,
validazione a runtime contro schema, true dry-run e intercettazione thread-safe di operazioni I/O in tool 'pure'.
"""

import time
import threading
import builtins
import io
import os
import socket
import subprocess
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Any, Tuple

from janus.lexer import Lexer
from janus.parser import Parser
from janus.type_checker import TypeChecker
from janus.codegen import CodeGenerator, JanusToolNotRegistered
from janus.ast_nodes import SchemaDecl, PrimitiveType, ListType, CustomType


class ToolNotFoundError(Exception):
    """Sollevata quando un tool invocato non è registrato nella sandbox."""
    pass


class ToolExecutionError(Exception):
    """Sollevata quando l'esecuzione di un tool fallisce all'interno della sandbox."""
    pass


class StrictPurityViolationError(ToolExecutionError):
    """
    Sollevata da ToolSandbox quando la modalità opzionale strict_effects è attiva
    e una funzione registrata come 'pure' tenta di eseguire operazioni di I/O (file, socket, subprocess).
    """
    pass


class ToolValidationError(Exception):
    """Sollevata quando gli argomenti o l'output di un tool violano lo schema dichiarato."""
    pass


class ToolEffectMismatchError(Exception):
    """Sollevata quando l'effetto registrato per un tool non corrisponde a quello dichiarato nello schema."""
    pass


# =========================================================================
# Thread-safe Strict Effects Guard (Defense-in-depth cooperativa)
# =========================================================================

_thread_local = threading.local()

def _is_guard_active() -> bool:
    return getattr(_thread_local, "strict_guard_active", False)

def _get_active_tool_name() -> str:
    return getattr(_thread_local, "strict_tool_name", "unknown")

_patches_installed = False
_orig_builtins_open = builtins.open
_orig_io_open = io.open
_orig_os_open = os.open
_orig_os_remove = getattr(os, "remove", None)
_orig_os_unlink = getattr(os, "unlink", None)
_orig_os_rename = getattr(os, "rename", None)
_orig_os_system = getattr(os, "system", None)
_orig_popen = subprocess.Popen
_orig_socket_socket = socket.socket
_orig_socket_connect = socket.socket.connect


def _install_strict_guard_patches():
    """
    Installa una sola volta i monkey patch a livello di interprete.
    L'intercettazione è effettivamente attiva SOLO sul thread corrente
    quando _is_guard_active() restituisce True.
    """
    global _patches_installed
    if _patches_installed:
        return
    _patches_installed = True

    def guarded_builtins_open(*args, **kwargs):
        if _is_guard_active():
            raise StrictPurityViolationError(
                f"Violazione di purezza: operazione I/O 'open' tentata dal tool pure '{_get_active_tool_name()}' (strict_effects attivo)."
            )
        return _orig_builtins_open(*args, **kwargs)

    def guarded_io_open(*args, **kwargs):
        if _is_guard_active():
            raise StrictPurityViolationError(
                f"Violazione di purezza: operazione I/O 'open' tentata dal tool pure '{_get_active_tool_name()}' (strict_effects attivo)."
            )
        return _orig_io_open(*args, **kwargs)

    def guarded_os_open(*args, **kwargs):
        if _is_guard_active():
            raise StrictPurityViolationError(
                f"Violazione di purezza: operazione I/O 'os.open' tentata dal tool pure '{_get_active_tool_name()}' (strict_effects attivo)."
            )
        return _orig_os_open(*args, **kwargs)

    def guarded_os_remove(*args, **kwargs):
        if _is_guard_active():
            raise StrictPurityViolationError(
                f"Violazione di purezza: operazione filesystem 'os.remove' tentata dal tool pure '{_get_active_tool_name()}' (strict_effects attivo)."
            )
        return _orig_os_remove(*args, **kwargs)

    def guarded_os_unlink(*args, **kwargs):
        if _is_guard_active():
            raise StrictPurityViolationError(
                f"Violazione di purezza: operazione filesystem 'os.unlink' tentata dal tool pure '{_get_active_tool_name()}' (strict_effects attivo)."
            )
        return _orig_os_unlink(*args, **kwargs)

    def guarded_os_rename(*args, **kwargs):
        if _is_guard_active():
            raise StrictPurityViolationError(
                f"Violazione di purezza: operazione filesystem 'os.rename' tentata dal tool pure '{_get_active_tool_name()}' (strict_effects attivo)."
            )
        return _orig_os_rename(*args, **kwargs)

    def guarded_os_system(*args, **kwargs):
        if _is_guard_active():
            raise StrictPurityViolationError(
                f"Violazione di purezza: chiamata di sistema 'os.system' tentata dal tool pure '{_get_active_tool_name()}' (strict_effects attivo)."
            )
        return _orig_os_system(*args, **kwargs)

    def guarded_popen(*args, **kwargs):
        if _is_guard_active():
            raise StrictPurityViolationError(
                f"Violazione di purezza: operazione 'subprocess.Popen' tentata dal tool pure '{_get_active_tool_name()}' (strict_effects attivo)."
            )
        return _orig_popen(*args, **kwargs)

    def guarded_socket(*args, **kwargs):
        if _is_guard_active():
            raise StrictPurityViolationError(
                f"Violazione di purezza: operazione I/O 'apertura socket' tentata dal tool pure '{_get_active_tool_name()}' (strict_effects attivo)."
            )
        return _orig_socket_socket(*args, **kwargs)

    def guarded_socket_connect(self, *args, **kwargs):
        if _is_guard_active():
            raise StrictPurityViolationError(
                f"Violazione di purezza: connessione socket di rete tentata dal tool pure '{_get_active_tool_name()}' (strict_effects attivo)."
            )
        return _orig_socket_connect(self, *args, **kwargs)

    builtins.open = guarded_builtins_open
    io.open = guarded_io_open
    os.open = guarded_os_open
    if _orig_os_remove is not None:
        os.remove = guarded_os_remove
    if _orig_os_unlink is not None:
        os.unlink = guarded_os_unlink
    if _orig_os_rename is not None:
        os.rename = guarded_os_rename
    if _orig_os_system is not None:
        os.system = guarded_os_system
    subprocess.Popen = guarded_popen
    socket.socket = guarded_socket
    socket.socket.connect = guarded_socket_connect


@contextmanager
def _strict_purity_guard(tool_name: str):
    """
    Attiva l'intercettazione strict_effects solo ed esclusivamente per il thread corrente.
    Altri thread che eseguono contemporaneamente I/O legittimo non vengono bloccati.
    
    AVVISO DI SICUREZZA:
    Questo meccanismo è una difesa cooperativa a livello di interprete Python (defense-in-depth).
    NON garantisce isolamento a livello di processo o di sistema operativo
    (bypass tramite ctypes, moduli C compilati o syscall dirette non sono prevenuti).
    """
    _install_strict_guard_patches()
    prev_active = getattr(_thread_local, "strict_guard_active", False)
    prev_name = getattr(_thread_local, "strict_tool_name", "")
    _thread_local.strict_guard_active = True
    _thread_local.strict_tool_name = tool_name
    try:
        yield
    finally:
        _thread_local.strict_guard_active = prev_active
        _thread_local.strict_tool_name = prev_name


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
    mocked: bool = False


class ToolSandbox:
    """
    Sandbox per l'esecuzione sicura e tracciata di tool.
    Gestisce la registrazione delle funzioni, la validazione a runtime contro schema,
    true dry-run tipizzato e la difesa cooperativa strict_effects per tool 'pure'.
    """

    def __init__(self, dry_run: bool = False, strict_effects: bool = False):
        self.dry_run = dry_run
        self.strict_effects = strict_effects
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._traces: List[ToolTrace] = []
        self._scope_globals: Optional[Dict[str, Any]] = None
        self._schemas: Dict[str, SchemaDecl] = {}

    def register_tool(
        self,
        name: str,
        func: Callable,
        effect: str = "io",
        schema: Optional[Any] = None
    ) -> None:
        """Registra un tool callable con metadati di effetto e schema."""
        if self._schemas and name in self._schemas:
            declared_effect = self._schemas[name].effect
            if effect != declared_effect:
                raise ToolEffectMismatchError(
                    f"Effetto non corrispondente per il tool '{name}': lo schema dichiara '{declared_effect}', ma è stato registrato come '{effect}'."
                )

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

    def _validate_inputs(self, tool_name: str, schema: SchemaDecl, kwargs: Dict[str, Any]):
        for param in schema.inputs:
            pname = param.name
            if pname in kwargs:
                val = kwargs[pname]
                if not self._check_type(val, param.type_expr):
                    raise ToolValidationError(
                        f"Tipo non valido per l'argomento '{pname}' del tool '{tool_name}': atteso {param.type_expr}, ottenuto {type(val).__name__}"
                    )
            elif param.default is None:
                raise ToolValidationError(
                    f"Argomento obbligatorio mancante '{pname}' per il tool '{tool_name}'"
                )

    def _validate_output(self, tool_name: str, schema: SchemaDecl, result: Any):
        if len(schema.outputs) == 1:
            out_field = schema.outputs[0]
            fname = out_field.name
            if hasattr(result, fname):
                val = getattr(result, fname)
            elif isinstance(result, dict) and fname in result:
                val = result[fname]
            else:
                val = result
            if not self._check_type(val, out_field.type_expr):
                raise ToolValidationError(
                    f"Tipo non valido per il campo di output '{fname}' del tool '{tool_name}': atteso {out_field.type_expr}, ottenuto {type(val).__name__}"
                )
            return

        for out_field in schema.outputs:
            fname = out_field.name
            if hasattr(result, fname):
                val = getattr(result, fname)
            elif isinstance(result, dict) and fname in result:
                val = result[fname]
            else:
                raise ToolValidationError(
                    f"Output del tool '{tool_name}' privo del campo richiesto '{fname}'"
                )
            if not self._check_type(val, out_field.type_expr):
                raise ToolValidationError(
                    f"Tipo non valido per il campo di output '{fname}' del tool '{tool_name}': atteso {out_field.type_expr}, ottenuto {type(val).__name__}"
                )

    def _check_type(self, val: Any, type_expr: Any) -> bool:
        if type_expr is None or val is None:
            return True
        type_name = None
        if isinstance(type_expr, PrimitiveType):
            type_name = type_expr.name.lower()
        elif isinstance(type_expr, CustomType):
            type_name = type_expr.name.lower()

        if type_name:
            if type_name in ("i8", "i16", "i32", "i64", "int"):
                return isinstance(val, int) and not isinstance(val, bool)
            if type_name in ("f16", "f32", "f64", "bf16", "float"):
                return isinstance(val, (int, float)) and not isinstance(val, bool)
            if type_name in ("str", "string"):
                return isinstance(val, str)
            if type_name in ("bool", "boolean"):
                return isinstance(val, bool)

        if isinstance(type_expr, ListType):
            if not isinstance(val, list):
                return False
            if val and getattr(type_expr, "inner", None) is not None:
                return all(self._check_type(elem, type_expr.inner) for elem in val)
            return True
        return True

    def _create_mock_output(self, tool_name: str, schema: Optional[SchemaDecl], kwargs: Dict[str, Any]) -> Any:
        out_cls = None
        if self._scope_globals:
            out_cls = self._scope_globals.get(f"{tool_name}Output")

        dummy_args = {}
        if out_cls and hasattr(out_cls, "__dataclass_fields__"):
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
                return out_cls(**dummy_args)
            except Exception:
                pass

        if schema:
            for f in schema.outputs:
                t = f.type_expr
                if isinstance(t, PrimitiveType):
                    name = t.name.lower()
                    if name in ("i8", "i16", "i32", "i64", "int"):
                        dummy_args[f.name] = 0
                    elif name in ("f16", "f32", "f64", "bf16", "float"):
                        dummy_args[f.name] = 0.0
                    elif name in ("str", "string"):
                        dummy_args[f.name] = ""
                    elif name in ("bool", "boolean"):
                        dummy_args[f.name] = False
                    else:
                        dummy_args[f.name] = None
                elif isinstance(t, ListType):
                    dummy_args[f.name] = []
                else:
                    dummy_args[f.name] = None
            if out_cls:
                try:
                    return out_cls(**dummy_args)
                except Exception:
                    pass
            return dummy_args

        return {"tool": tool_name, "mock": True, "inputs": kwargs}

    def call_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Esegue un tool registrato all'interno della sandbox catturando metriche e tracce.
        - True dry-run: esegue solo tool 'pure' registrati; mocka 'io' e 'stoc' con mocked=True.
        - Non dry-run: fallisce immediatamente se il tool non è registrato.
        - Valida gli argomenti e l'output contro lo schema se disponibile.
        """
        start = time.perf_counter()

        schema = self._schemas.get(tool_name) if self._schemas else None

        # Verifica corrispondenza dell'effetto registrato con lo schema dichiarato
        if schema and tool_name in self._tools:
            declared_effect = schema.effect
            registered_effect = self._tools[tool_name].get("effect", "io")
            if registered_effect != declared_effect:
                raise ToolEffectMismatchError(
                    f"Effetto non corrispondente per il tool '{tool_name}': lo schema dichiara '{declared_effect}', ma è stato registrato come '{registered_effect}'."
                )

        # Se non registrato
        if tool_name not in self._tools:
            if not self.dry_run:
                raise ToolNotFoundError(f"Tool '{tool_name}' is not registered in the sandbox")

            # True dry-run per tool non registrato: sintetizza mock tipizzato
            mock_out = self._create_mock_output(tool_name, schema, kwargs)
            duration_ms = (time.perf_counter() - start) * 1000.0
            trace = ToolTrace(
                tool_name=tool_name,
                inputs=kwargs,
                output=mock_out,
                duration_ms=duration_ms,
                success=True,
                effect=schema.effect if schema else "io",
                mocked=True
            )
            self._traces.append(trace)
            return mock_out

        entry = self._tools[tool_name]
        fn = entry["func"]
        effect = entry.get("effect", "io")

        # True dry-run per tool registrato:
        # I tool 'io' e 'stoc' NON vengono invocati (vengono mockati con mocked=True).
        # Solo i tool 'pure' vengono eseguiti realmente.
        if self.dry_run and effect in ("io", "stoc"):
            mock_out = self._create_mock_output(tool_name, schema, kwargs)
            duration_ms = (time.perf_counter() - start) * 1000.0
            trace = ToolTrace(
                tool_name=tool_name,
                inputs=kwargs,
                output=mock_out,
                duration_ms=duration_ms,
                success=True,
                effect=effect,
                mocked=True
            )
            self._traces.append(trace)
            return mock_out

        # Validazione degli argomenti di input prima della chiamata
        if schema:
            self._validate_inputs(tool_name, schema, kwargs)

        try:
            if self.strict_effects and effect == "pure":
                with _strict_purity_guard(tool_name):
                    res = fn(*args, **kwargs)
            else:
                res = fn(*args, **kwargs)

            # Validazione dell'output dopo la chiamata
            if schema:
                self._validate_output(tool_name, schema, res)

            duration_ms = (time.perf_counter() - start) * 1000.0
            trace = ToolTrace(
                tool_name=tool_name,
                inputs=kwargs,
                output=res,
                duration_ms=duration_ms,
                success=True,
                effect=effect,
                mocked=False
            )
            self._traces.append(trace)
            return res

        except StrictPurityViolationError as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            trace = ToolTrace(
                tool_name=tool_name,
                inputs=kwargs,
                error=str(e),
                duration_ms=duration_ms,
                success=False,
                effect=effect,
                mocked=False
            )
            self._traces.append(trace)
            raise
        except ToolValidationError as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            trace = ToolTrace(
                tool_name=tool_name,
                inputs=kwargs,
                error=str(e),
                duration_ms=duration_ms,
                success=False,
                effect=effect,
                mocked=False
            )
            self._traces.append(trace)
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            trace = ToolTrace(
                tool_name=tool_name,
                inputs=kwargs,
                error=str(e),
                duration_ms=duration_ms,
                success=False,
                effect=effect,
                mocked=False
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
    Garantisce il binding con la ToolSandbox, associazione schema e trace di esecuzione.
    """

    def __init__(self, sandbox: Optional[ToolSandbox] = None, strict_effects: bool = False):
        self.sandbox = sandbox or ToolSandbox(strict_effects=strict_effects)

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
        dry_run: Optional[bool] = None,
        strict_effects: Optional[bool] = None
    ) -> Tuple[Any, List[ToolTrace]]:
        """
        Compila ed esegue una funzione agentica catturando tutti i tool trace.
        Ripristina sempre lo stato precedente della sandbox in try/finally.
        """
        prev_dry_run = self.sandbox.dry_run
        prev_strict = self.sandbox.strict_effects

        try:
            if dry_run is not None:
                self.sandbox.dry_run = dry_run
            if strict_effects is not None:
                self.sandbox.strict_effects = strict_effects
            self.sandbox.clear_traces()

            # Estrai schemi per l'associazione runtime e la validazione
            tokens = Lexer(code).tokenize()
            ast = Parser(tokens).parse()
            self.sandbox._schemas = {d.name: d for d in ast.declarations if isinstance(d, SchemaDecl)}

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

            result = fn(*fn_args, **fn_kwargs)
            return result, self.sandbox.get_traces()

        finally:
            self.sandbox.dry_run = prev_dry_run
            self.sandbox.strict_effects = prev_strict
