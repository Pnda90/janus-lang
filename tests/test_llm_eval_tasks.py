"""
Test di Accettazione per i 30 Task dell'Harness di Valutazione LLM (Fase 4):
Verifica che tutte le 30 specifiche di task abbiano implementazioni di riferimento
valide e conformi in JANUS e Python che superano i test esecutivi PyTorch.
"""

import pytest
from benchmarks.llm_eval import TASKS, evaluate_janus, evaluate_python

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch non installato")

def test_tasks_count():
    assert len(TASKS) == 30, f"Attesi esattamente 30 compiti tensoriali, trovati {len(TASKS)}"

@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_task_reference_implementations(task):
    # 1. Verifica compilazione ed esecuzione JANUS
    janus_ok, py_code, janus_err = evaluate_janus(task["janus_ref"], task)
    assert janus_ok, f"JANUS fallito per {task['id']}: {janus_err}"

    # 2. Verifica esecuzione riferimento Python
    python_ok, _, py_err = evaluate_python(task["python_ref"], task)
    assert python_ok, f"Python fallito per {task['id']}: {py_err}"
