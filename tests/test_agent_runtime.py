"""
Test per il runtime di esecuzione sicura di agenti (ToolSandbox e AgentRuntime).
Fase A.3: Verifica l'esecuzione isolata, la cattura dei trace e la validazione dei tool.
"""

import pytest
from typing import Dict, Any

from janus.agent_runtime import ToolSandbox, AgentRuntime, ToolExecutionError, ToolNotFoundError


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
