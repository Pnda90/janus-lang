"""
Test per il runtime di esecuzione sicura di agenti (ToolSandbox e AgentRuntime).
Fase A.3: Verifica l'esecuzione isolata, la cattura dei trace e la validazione dei tool.
"""

import pytest
import subprocess
import socket
import tempfile
from typing import Dict, Any

from janus.agent_runtime import (
    ToolSandbox,
    AgentRuntime,
    ToolExecutionError,
    ToolNotFoundError,
    StrictPurityViolationError,
    ToolValidationError,
    ToolEffectMismatchError,
    JanusToolNotRegistered,
)


def test_tool_sandbox_registration_and_execution():
    sandbox = ToolSandbox()

    @sandbox.tool("SearchEngine", effect="io")
    def search(query: str, limit: int = 5) -> Dict[str, Any]:
        return {"query": query, "results": [f"Result for {query} #{i}" for i in range(limit)]}

    res = sandbox.call_tool("SearchEngine", query="quantum computing", limit=2)
    assert res["query"] == "quantum computing"
    assert len(res["results"]) == 2

    # Verifica trace
    traces = sandbox.get_traces()
    assert len(traces) == 1
    t = traces[0]
    assert t.tool_name == "SearchEngine"
    assert t.inputs == {"query": "quantum computing", "limit": 2}
    assert t.success is True
    assert t.error is None
    assert t.duration_ms >= 0.0


def test_tool_sandbox_unregistered_tool():
    sandbox = ToolSandbox()
    with pytest.raises(ToolNotFoundError) as exc_info:
        sandbox.call_tool("DatabaseWriter", table="users")
    assert "DatabaseWriter" in str(exc_info.value)


def test_tool_sandbox_exception_handling():
    sandbox = ToolSandbox()

    @sandbox.tool("FailingTool")
    def bad_tool(x: int):
        if x < 0:
            raise ValueError("Negative numbers not allowed")
        return x * 2

    with pytest.raises(ToolExecutionError) as exc_info:
        sandbox.call_tool("FailingTool", x=-5)
    assert "Negative numbers" in str(exc_info.value)

    traces = sandbox.get_traces()
    assert len(traces) == 1
    assert traces[0].success is False
    assert "Negative numbers not allowed" in traces[0].error


def test_agent_runtime_compile_and_execute():
    code = """
    schema AddTool [pure] {
        a: float,
        b: float
    } -> {
        sum: float
    }

    schema LogMessage [io] {
        msg: str
    } -> {
        status: str
    }

    fn run_agent_workflow(x: float, y: float) -> float io {
        s = call tool AddTool(a = x, b = y)
        logged = call tool LogMessage(msg = "Computed sum")
        ret s
    }
    """
    runtime = AgentRuntime()

    # Registrazione tool nel runtime
    runtime.register_tool("AddTool", lambda a, b: a + b, effect="pure")
    runtime.register_tool("LogMessage", lambda msg: f"Logged: {msg}", effect="io")

    # Esecuzione del workflow compilato da JANUS
    result, traces = runtime.execute(code, entrypoint="run_agent_workflow", args=[3.5, 6.5])

    assert result == 10.0
    assert len(traces) == 2
    assert traces[0].tool_name == "AddTool"
    assert traces[0].output == 10.0
    assert traces[1].tool_name == "LogMessage"
    assert traces[1].output == "Logged: Computed sum"


def test_agent_runtime_dry_run_mode():
    code = """
    schema MockSearch [io] {
        query: str
    } -> {
        results_count: int
    }

    fn search_workflow(q: str) -> int io {
        res = call tool MockSearch(query = q)
        ret res
    }
    """
    runtime = AgentRuntime()
    # In modalità dry_run non serve registrare il tool: restituisce valori predefiniti
    result, traces = runtime.execute(code, entrypoint="search_workflow", args=["deep learning"], dry_run=True)

    assert len(traces) == 1
    assert traces[0].tool_name == "MockSearch"
    assert traces[0].inputs == {"query": "deep learning"}
    # Dry run produce mock in base ai campi di output definiti nello schema o default
    assert result is not None


def test_strict_effects_pure_tool_raises_on_file_io():
    sandbox = ToolSandbox(strict_effects=True)

    @sandbox.tool("ImpureFileTool", effect="pure")
    def impure_file(path: str):
        with open(path, "w") as f:
            f.write("data")
        return "written"

    with pytest.raises(StrictPurityViolationError) as exc_info:
        sandbox.call_tool("ImpureFileTool", path="/tmp/test_should_fail.txt")
    assert "operazione I/O 'open' tentata dal tool pure" in str(exc_info.value)

    traces = sandbox.get_traces()
    assert len(traces) == 1
    assert traces[0].success is False
    assert "open" in traces[0].error


def test_strict_effects_pure_tool_raises_on_subprocess():
    sandbox = ToolSandbox(strict_effects=True)

    @sandbox.tool("ImpureSubprocessTool", effect="pure")
    def impure_sub():
        subprocess.run(["echo", "exploit"])
        return "done"

    with pytest.raises(StrictPurityViolationError) as exc_info:
        sandbox.call_tool("ImpureSubprocessTool")
    assert "subprocess.Popen" in str(exc_info.value)


def test_strict_effects_pure_tool_raises_on_socket():
    sandbox = ToolSandbox(strict_effects=True)

    @sandbox.tool("ImpureSocketTool", effect="pure")
    def impure_sock():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.close()
        return "connected"

    with pytest.raises(StrictPurityViolationError) as exc_info:
        sandbox.call_tool("ImpureSocketTool")
    assert "apertura socket" in str(exc_info.value)


def test_strict_effects_pure_tool_allows_pure_computation():
    sandbox = ToolSandbox(strict_effects=True)

    @sandbox.tool("PureMathTool", effect="pure")
    def pure_math(a: int, b: int) -> int:
        return (a * b) + 42

    res = sandbox.call_tool("PureMathTool", a=3, b=4)
    assert res == 54

    traces = sandbox.get_traces()
    assert len(traces) == 1
    assert traces[0].success is True
    assert traces[0].output == 54


def test_strict_effects_io_tool_allows_io():
    sandbox = ToolSandbox(strict_effects=True)

    @sandbox.tool("LegitIoTool", effect="io")
    def legit_io(data: str) -> str:
        with tempfile.NamedTemporaryFile("w+", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            tmp.seek(0)
            return tmp.read()

    res = sandbox.call_tool("LegitIoTool", data="legit_content")
    assert res == "legit_content"


def test_strict_effects_disabled_by_default():
    sandbox = ToolSandbox(strict_effects=False)

    @sandbox.tool("UncheckedPureTool", effect="pure")
    def unchecked(data: str) -> str:
        with tempfile.NamedTemporaryFile("w+", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            tmp.seek(0)
            return tmp.read()

    # Senza strict_effects=True, la sandbox applicativa non intercetta builtins.open
    res = sandbox.call_tool("UncheckedPureTool", data="allowed_because_strict_is_false")
    assert res == "allowed_because_strict_is_false"


def test_agent_runtime_with_strict_effects_flag():
    code = """
    schema RoguePureTool [pure] {
        tag: str
    } -> {
        res: str
    }

    fn agent_task(tag: str) -> str pure {
        r = call tool RoguePureTool(tag = tag)
        ret r
    }
    """
    runtime = AgentRuntime(strict_effects=True)

    def rogue_fn(tag: str):
        with open("/tmp/sneaky.txt", "w") as f:
            f.write(tag)
        return "written"

    runtime.register_tool("RoguePureTool", rogue_fn, effect="pure")

    with pytest.raises(StrictPurityViolationError):
        runtime.execute(code, entrypoint="agent_task", args=["payload"], strict_effects=True)


def test_schema_effect_mismatch_raises_error():
    code = """
    schema SafeTool [pure] {
        v: i32
    } -> {
        r: i32
    }

    fn run(x: i32) -> i32 pure {
        res = call tool SafeTool(v = x)
        ret res
    }
    """
    runtime = AgentRuntime()
    # Schema dice pure, ma il tool viene registrato con effect="io"
    runtime.register_tool("SafeTool", lambda v: v * 2, effect="io")

    with pytest.raises(ToolEffectMismatchError) as exc_info:
        runtime.execute(code, entrypoint="run", args=[5])
    assert "SafeTool" in str(exc_info.value)
    assert "pure" in str(exc_info.value)
    assert "io" in str(exc_info.value)


def test_tool_validation_error_on_invalid_inputs():
    from janus.lexer import Lexer
    from janus.parser import Parser

    code = """
    schema TypeStrictTool [io] {
        count: i32,
        name: str
    } -> {
        status: str
    }

    fn run() -> str io {
        res = call tool TypeStrictTool(count = 10, name = "ok")
        ret res
    }
    """
    runtime = AgentRuntime()
    runtime.register_tool("TypeStrictTool", lambda count, name: {"status": "ok"}, effect="io")

    # Invocazione diretta tramite sandbox con tipo errato per 'count' (passando stringa invece di int)
    ast = Parser(Lexer(code).tokenize()).parse()
    runtime.sandbox._schemas = {d.name: d for d in ast.declarations if hasattr(d, "name")}

    with pytest.raises(ToolValidationError) as exc_info:
        runtime.sandbox.call_tool("TypeStrictTool", count="not_a_number", name="test")
    assert "count" in str(exc_info.value)


def test_tool_validation_error_on_missing_required_input():
    from janus.lexer import Lexer
    from janus.parser import Parser

    sandbox = ToolSandbox()
    code = """
    schema StrictParamTool [io] {
        req: str
    } -> {
        ack: bool
    }
    """
    ast = Parser(Lexer(code).tokenize()).parse()
    sandbox._schemas = {d.name: d for d in ast.declarations if hasattr(d, "name")}
    sandbox.register_tool("StrictParamTool", lambda req: {"ack": True}, effect="io")

    with pytest.raises(ToolValidationError) as exc_info:
        sandbox.call_tool("StrictParamTool")
    assert "req" in str(exc_info.value)


def test_tool_validation_error_on_invalid_outputs():
    code = """
    schema StrictOutputTool [pure] {
        val: i32
    } -> {
        computed: i32
    }

    fn run(v: i32) -> i32 pure {
        res = call tool StrictOutputTool(val = v)
        ret res
    }
    """
    runtime = AgentRuntime()
    # Output tipizzato male: restituisce stringa anziché i32
    runtime.register_tool("StrictOutputTool", lambda val: {"computed": "not_an_int"}, effect="pure")

    with pytest.raises(ToolValidationError) as exc_info:
        runtime.execute(code, entrypoint="run", args=[42])
    assert "computed" in str(exc_info.value)


def test_true_dry_run_pure_executed_and_io_mocked():
    code = """
    schema PureCalc [pure] {
        x: i32,
        y: i32
    } -> {
        res: i32
    }

    schema RemoteApi [io] {
        endpoint: str
    } -> {
        data: str
    }

    fn workflow(a: i32, b: i32) -> str io {
        sum_val = call tool PureCalc(x = a, y = b)
        net_val = call tool RemoteApi(endpoint = "https://api.example.com")
        ret net_val
    }
    """
    runtime = AgentRuntime()

    # Tool pure che esegue vero calcolo
    runtime.register_tool("PureCalc", lambda x, y: x + y, effect="pure")

    # Tool io che esploderebbe se invocato
    def dangerous_remote(endpoint: str):
        raise RuntimeError("RemoteApi non deve essere chiamato in dry-run!")
    runtime.register_tool("RemoteApi", dangerous_remote, effect="io")

    result, traces = runtime.execute(code, entrypoint="workflow", args=[10, 20], dry_run=True)

    assert len(traces) == 2
    # PureCalc è stato eseguito realmente
    assert traces[0].tool_name == "PureCalc"
    assert traces[0].mocked is False
    assert traces[0].output == 30

    # RemoteApi è stato mockato senza sollevare l'eccezione
    assert traces[1].tool_name == "RemoteApi"
    assert traces[1].mocked is True
    assert traces[1].output is not None


def test_fail_closed_unregistered_tool_raises_not_found():
    code = """
    schema UnregisteredIoTool [io] {
        arg: str
    } -> {
        out: str
    }

    fn run() -> str io {
        res = call tool UnregisteredIoTool(arg = "hello")
        ret res
    }
    """
    runtime = AgentRuntime()
    # In non-dry-run mode, chiamare un tool non registrato solleva ToolNotFoundError
    with pytest.raises(ToolNotFoundError) as exc_info:
        runtime.execute(code, entrypoint="run")
    assert "UnregisteredIoTool" in str(exc_info.value)


def test_fail_closed_unregistered_tool_standalone_codegen():
    from janus.lexer import Lexer
    from janus.parser import Parser
    from janus.codegen import CodeGenerator

    code = """
    schema ExternalTool [io] {
        msg: str
    } -> {
        resp: str
    }

    fn run() -> str io {
        r = call tool ExternalTool(msg = "ping")
        ret r
    }
    """
    ast = Parser(Lexer(code).tokenize()).parse()
    py_code = CodeGenerator().generate(ast)
    env = {}
    exec(py_code, env)

    # Senza passare per AgentRuntime o registrare il tool in _JANUS_TOOL_REGISTRY:
    with pytest.raises(JanusToolNotRegistered) as exc_info:
        env["run"]()
    assert "ExternalTool" in str(exc_info.value)


def test_agent_runtime_sandbox_state_restoration():
    code = """
    fn fail_job() -> i32 pure {
        ret 1 / 0
    }
    """
    runtime = AgentRuntime()
    assert runtime.sandbox.dry_run is False
    assert runtime.sandbox.strict_effects is False

    with pytest.raises(ZeroDivisionError):
        runtime.execute(code, entrypoint="fail_job", dry_run=True, strict_effects=True)

    # I flag devono essere ripristinati anche a fronte di eccezione
    assert runtime.sandbox.dry_run is False
    assert runtime.sandbox.strict_effects is False


def test_strict_effects_concurrency_thread_safety():
    """
    Verifica che strict_effects sia isolato sul thread corrente:
    il thread A (in strict pure) non impedisce al thread B di fare legittimo I/O su file.
    """
    import threading
    import time

    sandbox = ToolSandbox(strict_effects=True)

    @sandbox.tool("PureWaitTool", effect="pure")
    def pure_wait():
        time.sleep(0.05)
        return "pure_done"

    thread_b_success = []
    thread_b_error = []

    def thread_b_worker():
        try:
            with tempfile.NamedTemporaryFile("w+", delete=True) as tmp:
                tmp.write("concurrent io")
                tmp.flush()
                tmp.seek(0)
                content = tmp.read()
                thread_b_success.append(content == "concurrent io")
        except Exception as e:
            thread_b_error.append(e)

    # Avvia thread B e thread A
    ta = threading.Thread(target=lambda: sandbox.call_tool("PureWaitTool"))
    tb = threading.Thread(target=thread_b_worker)

    ta.start()
    time.sleep(0.01)  # Lascia che Thread A attivi il guard
    tb.start()

    ta.join()
    tb.join()

    assert not thread_b_error, f"Thread B ha subito interferenza da strict guard: {thread_b_error}"
    assert thread_b_success == [True]


@pytest.mark.xfail(reason="Strict effects è una difesa applicativa cooperativa a livello Python: bypass tramite ctypes/syscall C è un limite noto documentato.")
def test_strict_effects_ctypes_bypass_known_limitation():
    import ctypes

    sandbox = ToolSandbox(strict_effects=True)

    @sandbox.tool("CtypesBypassTool", effect="pure")
    def bypass_tool():
        libc = ctypes.CDLL(None)
        libc.puts(b"bypass test")
        return "bypassed"

    # Questo test dichiara xfail: ci aspettiamo che NON sollevi StrictPurityViolationError
    # proprio a dimostrazione che monkey-patching in-process non isola a livello di syscall/OS.
    with pytest.raises(StrictPurityViolationError):
        sandbox.call_tool("CtypesBypassTool")


