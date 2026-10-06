# JANUS: Deterministic Agentic Execution Graph DSL & Constrained Decoding
### Technical Abstract & Architectural Pivot Report

**Author:** Pnda90  
**Date:** October 2026  
**Repository:** [https://github.com/Pnda90/janus-lang](https://github.com/Pnda90/janus-lang)  
**Status:** Verification, Hardening & Agentic DSL Pivot Complete (84/84 Automated Tests Passing, 0 Failures)

---

## 📄 Abstract

As autonomous agents powered by Large Language Models (LLMs) increasingly orchestrate external tools, databases, and APIs, traditional unconstrained JSON tool calling exhibits significant failure rates: hallucinated arguments, missing required fields, runtime type mismatches, and undefined side-effects.

We present **JANUS**, an LL(1) Domain-Specific Language designed for **deterministic agentic execution graphs, zero-hallucination tool calling via GBNF constrained decoding, and formal effect tracking**.

### Architectural Pillars
1. **First-Class Tool Schemas with Monadic Effects (`schema Name [effect]`):**
   Explicit interface contracts tracking side-effect boundaries (`pure` for deterministic verified calculations, `io` for external actions and network mutations, `stoc` for probabilistic operations).
2. **Multi-Tool GBNF Grammar Synthesis:**
   Deterministic compilation of tool contracts into formal GBNF grammars for inference engines (*llama.cpp*, *vLLM*, *Ollama*). Token masking at inference time mathematically guarantees 100% schema conformance and zero JSON syntax errors.
3. **Isolated Tool Sandbox Runtime (`janus.agent_runtime`):**
   Sandboxed execution environment providing typed argument validation, execution profiling, step-by-step audit trails (`ToolTrace`), and transparent mock/dry-run capabilities.
4. **Machine-Readable Diagnostics:**
   JSON-standardized compiler diagnostics (`./janusc check --json`) with stable error codes, exact byte spans, and automated corrective patches for compiler-guided multi-turn agent self-repair loops.
5. **Autodiff & Tensor Computation Backend:**
   High-performance transpilation pipeline to Python 3.13 / PyTorch with automatic differentiation (`diff loss wrt w`) and vectorized dataclass structures (`JanusStruct`).

---

## 🔬 Empirical Findings & Benchmark Results

All metrics are experimentally measured and reproducible via dedicated scripts in `benchmarks/` with raw data stored in `benchmarks/results/`:

### 1. Agentic Tool Calling Benchmark (20 Realistic Agent Tasks)
Evaluated via `benchmarks/agent_eval.py` across diverse agent scenarios (search, databases, payments, filesystems, cryptography, networking):

| Metric | Unconstrained JSON Tool Calling | JANUS GBNF Constrained | Guarantee / Improvement |
| :--- | :---: | :---: | :--- |
| **Valid JSON Syntax** | 95.0% | **100.0%** | Zero truncated or malformed JSON |
| **Schema Conformance** | 80.0% | **100.0%** | Zero hallucinated or omitted fields |
| **Argument Type Accuracy** | 75.0% | **100.0%** | Strict types enforced via token masking |
| **First-Attempt Perfect Calls** | 75.0% | **100.0%** | Mathematical determinism |
| **Tokens Consumed (`cl100k_base`)** | 484 tokens | **411 tokens** | **-15.1%** (elimination of rambling keys & noise) |

### 2. Lexical Token Efficiency for Pure Tensor Computing (10 Reference Programs)
Evaluated via `benchmarks/tokens.py` on real Foundation Model tokenizers:
- **Refuted Hypothesis:** On standard off-the-shelf BPE tokenizers (`cl100k_base`, `o200k_base`), pure tensor code in JANUS consumes **+51.5% to +52.2% more tokens** than idiomatic Python. Standard tokenizers heavily compress mainstream Python keywords into single tokens, whereas custom DSL tokens undergo subword fragmentation.
- **Strategic Pivot:** While DSLs do not compress tokens for generic tensor computing without a dedicated tokenizer vocabulary, they provide unmatched **safety, determinism, and zero-hallucination guarantees** for agentic tool execution.

---

## 🎯 Verified Engineering Status

* **Test Suite:** **84 automated tests passing (0 failures)** covering lexing, LL(1) parsing, type & effect checking, autodiff numeric convergence, GBNF grammar validation, agent runtime sandboxing, and benchmark integrity audit.
* **Continuous Integration:** Fully automated GitHub Actions workflow on Python 3.11 and 3.12 with zero warnings or errors.

---

## 📌 Summary Recommendation

JANUS demonstrates that the genuine value of an AI-targeted DSL is **not lexical compression**, but **grammatical determinism, formal effect boundaries, and inference-time token masking**. By turning tool calling from unconstrained text generation into a typed, verifiable execution graph, JANUS eliminates prompt-level tool hallucinations by compiler construction.
