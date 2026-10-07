# JANUS: Agentic Execution Graph DSL with Type & Effect Systems & Constrained Decoding
### Technical Abstract & Architectural Pivot Report

**Author:** Pnda90  
**Date:** October 2026  
**Repository:** [https://github.com/Pnda90/janus-lang](https://github.com/Pnda90/janus-lang)  
**Status:** Verification, Hardening & Agentic DSL Pivot Complete (Automated Test Suite Passing, 0 Failures)

---

## 📄 Abstract

As autonomous agents powered by Large Language Models (LLMs) increasingly orchestrate external tools, databases, and APIs, traditional unconstrained JSON tool calling exhibits significant failure rates: hallucinated arguments, missing required fields, runtime type mismatches, and undefined side-effects.

We present **JANUS**, an LL(1) Domain-Specific Language designed for **agentic execution graphs, formal effect tracking, and GBNF grammar-constrained decoding**.

Constrained decoding mathematically enforces syntax and type conformance during generation through token masking. However, it does not guarantee semantic correctness of tool choice or argument values, as sampling remains stochastic. JANUS combines grammar synthesis with static effect checking to establish rigorous boundaries around agent execution.

### Architectural Pillars
1. **First-Class Tool Schemas with Monadic Effects (`schema Name [effect]`):**
   Explicit interface contracts tracking side-effect boundaries (`pure` for verified deterministic calculations, `io` for external actions and network mutations, `stoc` for probabilistic operations).
2. **Multi-Tool GBNF Grammar Synthesis:**
   Compilation of tool contracts into formal GBNF grammars for inference engines (*llama.cpp*, *vLLM*, *Ollama*). Token masking at inference time guarantees 100% schema conformance and zero JSON syntax errors by construction.
3. **Isolated Tool Sandbox Runtime (`janus.agent_runtime`):**
   Execution environment providing typed argument validation, execution profiling, step-by-step audit trails (`ToolTrace`), and transparent mock/dry-run capabilities.
4. **Machine-Readable Diagnostics:**
   JSON-standardized compiler diagnostics (`./janusc check --json`) with stable error codes, exact byte spans, and automated corrective patches for compiler-guided multi-turn agent self-repair loops.
5. **Autodiff & Tensor Computation Backend:**
   Transpilation pipeline to Python (3.10+) / PyTorch with automatic differentiation (`diff loss wrt w`) and vectorized dataclass structures (`JanusStruct`).

---

## 🔬 Empirical Findings & Benchmark Methodology

All metrics are experimentally reproducible via dedicated scripts in `benchmarks/` with raw data stored in `benchmarks/results/`:

### 1. Agentic Tool Calling Benchmark (20 Realistic Agent Tasks)
Evaluated via `benchmarks/agent_eval.py` across diverse agent scenarios (search, databases, payments, filesystems, cryptography, networking):

| Metric | Unconstrained JSON (Simulated Baseline) | JANUS GBNF Constrained | Guarantee / Note |
| :--- | :---: | :---: | :--- |
| **Valid JSON Syntax** | 90.0% – 95.0% | **100.0%** | Zero truncated or malformed JSON (token masking) |
| **Schema Conformance** | 75.0% – 80.0% | **100.0%** | Zero omitted required fields or hallucinated keys |
| **Argument Type Accuracy** | 70.0% – 75.0% | **100.0%** | Strict scalar/primitive types enforced by grammar |
| **Tool Choice & Semantic Args** | Dependent on prompt | Dependent on prompt | Sampling remains stochastic; grammar constrains syntax |
| **Tokens Consumed (`cl100k_base`)** | 686 tokens | **615 tokens** | Elimination of rambling keys & noise |

*Note on methodology:* Dry-run benchmark numbers reflect simulated failure modes documented in literature. Live inference evaluation requires running local inference backends (`llama-server`, `ollama`, `vllm`) via `benchmarks/agent_eval.py --backend`.

### 2. Lexical Token Efficiency for Pure Tensor Computing (10 Reference Programs)
Evaluated via `benchmarks/tokens.py` on real Foundation Model tokenizers:
- **Refuted Hypothesis:** On standard off-the-shelf BPE tokenizers (`cl100k_base`, `o200k_base`), pure tensor code in JANUS consumes **+51.5% to +52.2% more tokens** than idiomatic Python. Standard tokenizers heavily compress mainstream Python keywords into single tokens, whereas custom DSL tokens undergo subword fragmentation.
- **Strategic Pivot:** While DSLs do not compress tokens for generic tensor computing without a dedicated tokenizer vocabulary, they provide unmatched **syntactic safety, static effect boundaries, and verifiable execution graphs** for agentic tool orchestration.

---

## 🎯 Verified Engineering Status

* **Test Suite:** Automated tests covering lexing, LL(1) parsing, type & effect checking, autodiff numeric convergence, GBNF grammar validation, agent runtime sandboxing, and benchmark integrity audit.
* **Continuous Integration:** Automated GitHub Actions workflow on Python 3.10, 3.11, 3.12, and 3.13 with zero warnings or errors.

---

## 📌 Summary Recommendation

JANUS demonstrates that the genuine value of an AI-targeted DSL is **not lexical compression**, but **grammatical structure, formal effect boundaries, and inference-time token masking**. By turning tool calling from unconstrained text generation into a typed, verifiable execution graph, JANUS eliminates syntactic tool call failures by compiler construction while bounding side-effects through static verification.
