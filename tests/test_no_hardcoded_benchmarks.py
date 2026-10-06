"""
Test di Audit Integrità Benchmark (Fase 5 / CI):
Verifica che:
1. I file in benchmarks/ non contengano tabelle hardcoded di token inventati.
2. I file dei risultati (benchmarks/results/*.json) siano generati e validi.
3. Tutti i 10 esempi compilino senza errori sintattici o di tipo.
"""

import os
import re
import json
import pytest

def test_no_hardcoded_tables_in_benchmark_scripts():
    bench_dir = "benchmarks"
    forbidden_patterns = [
        r"reference_data\s*=",
        r"baseline_tokens\s*=",
        r"-50\.0%",
        r"-50\.7%",
        r"\"savings_pct\":\s*-5",
    ]
    
    for root, _, files in os.walk(bench_dir):
        if "results" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as src_file:
                    content = src_file.read()
                for pat in forbidden_patterns:
                    match = re.search(pat, content)
                    assert not match, f"Trovato pattern sospetto o hardcoded '{pat}' in {path}"

def test_token_benchmark_artifact_integrity():
    artifact_path = "benchmarks/results/tokens_benchmark.json"
    assert os.path.exists(artifact_path), f"File {artifact_path} mancante."
    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "metadata" in data
    assert "totals" in data
    assert "programs" in data
    assert len(data["programs"]) == 10
    # Verifica che la percentuale calcolata sia positiva (JANUS usa PIÙ token di Python sui tokenizer reali)
    diff_cl100k = data["totals"]["tokens_cl100k"]["diff_vs_idiomatic_pct"]
    assert diff_cl100k > 0, f"Atteso diff_pct positivo (JANUS > Python), trovato {diff_cl100k}"

def test_all_10_examples_compile_cleanly():
    from janus.lexer import Lexer
    from janus.parser import Parser
    from janus.type_checker import TypeChecker
    from janus.codegen import CodeGenerator

    examples_dir = "examples"
    files = sorted([f for f in os.listdir(examples_dir) if f.endswith(".jn")])
    assert len(files) == 10, f"Attesi 10 esempi, trovati {len(files)}"

    for f_name in files:
        f_path = os.path.join(examples_dir, f_name)
        with open(f_path, "r", encoding="utf-8") as f:
            code = f.read()
        tokens = Lexer(code).tokenize()
        ast = Parser(tokens).parse()
        diags = TypeChecker().check(ast)
        errs = [d for d in diags if d.code.startswith("ERR")]
        assert not errs, f"Errori di tipo in {f_name}: {[e.message for e in errs]}"
        py = CodeGenerator().generate(ast)
        assert len(py) > 0, f"Codice Python vuoto generato per {f_name}"
