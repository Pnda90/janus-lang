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

