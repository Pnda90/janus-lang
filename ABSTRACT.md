# JANUS: A Domain-Specific Language for Neural Computing and Constrained Decoding
### Technical Abstract & Investigation Report

**Authors:** Pnda90 & The Janus Engineering Team  
**Date:** October 2026  
**Repository:** [https://github.com/Pnda90/janus-lang](https://github.com/Pnda90/janus-lang)  
**Status:** Verification & Hardening Complete (LL(1) Transpiler, PyTorch Runtime Interop, GBNF Generation, Honest Benchmark Suite)

---

## 📄 Abstract

As Foundation Models (Large Language Models) increasingly generate executable code, domain-specific languages (DSLs) have been proposed to overcome syntactic verbosity, delimiter overhead, and non-deterministic parsing in general-purpose languages like Python. We present an empirical investigation of **JANUS**, a domain-specific programming language designed for neural computing, automatic differentiation, and grammar-constrained decoding.

JANUS features:
1. **Agglutinative Latin Case Morphology (`ident:case`):** Explicit semantic role tagging on variables (`:m` accusative operands, `:b` ablative parameters, `:t` dative buffers, `:n` nominative definitions, `:s` genitive shapes, `:v` vocative agents).
2. **Strict LL(1) Grammar:** A deterministic grammar enabling linear-time parsing and the automatic synthesis of valid GBNF grammars for constrained decoding in inference engines (*llama.cpp*, *vLLM*).
3. **Structured JSON Diagnostics:** Machine-readable diagnostics with standardized error codes, precise source coordinates, offending tokens, and corrective patches for automated multi-turn LLM repair loops.
4. **PyTorch Transpilation & Execution Runtime:** A transpilation pipeline supporting native autodiff (`diff loss wrt (wb, bb)`), tensor arithmetic dataclasses (`JanusStruct`), and full execution against PyTorch references.

### Empirical Evaluation of Core Hypotheses
We investigated the central hypothesis: *"An LL(1) DSL with machine-readable diagnostics enables LLMs to produce correct tensor computation with fewer total tokens and higher correctness compared to Python."*

Our empirical measurements reveal:
- **Token Efficiency (Refuted):** On standard BPE tokenizers (`cl100k_base` and `o200k_base`), JANUS code consumes **+51.5% to +52.2% more tokens** than equivalent idiomatic Python across 10 complete reference architectures. This is caused by BPE subword fragmentation: standard tokenizers have single-token vocabulary entries for common Python keywords, whereas novel DSL constructs undergo multi-token splitting. The in-context prompt overhead (+229 tokens) is never amortized on generic tokenizers.
- **Parsing Determinism & Constrained Decoding (Confirmed):** The LL(1) grammar allows robust export of clean GBNF grammars for schema validation without quotation imbalance or non-deterministic lookahead.
- **Automated Self-Correction (Confirmed):** Machine-readable JSON diagnostics allow LLMs to systematically rectify syntax and typing errors within multi-turn agentic loops.

---

## 🎯 Verified Empirical Metrics

| Metric | Python (Idiomatic PyTorch) | Python (Compact) | JANUS (`.jn`) | Delta vs Idiomatic |
| :--- | :---: | :---: | :---: | :---: |
| **Lines of Code (10 Programs)** | 86 LoC | 40 LoC | 99 LoC | **+15.1%** |
| **Tokens `cl100k_base` (GPT-4)** | 929 tokens | 718 tokens | 1,414 tokens | **+52.2%** |
| **Tokens `o200k_base` (GPT-4o)** | 933 tokens | 720 tokens | 1,413 tokens | **+51.5%** |
| **System Prompt In-Context Cost** | 0 tokens | 0 tokens | 229 tokens | +229 tokens |
| **Pass Rate on Test Suite** | 100% (Reference) | - | 100% (66/66 tests) | Parity |
| **Grammar Class** | Context-Free / Indented | Context-Free | Strict LL(1) | Deterministic |

---

## 📌 Conclusions
A custom DSL without a dedicated BPE vocabulary cannot yield token savings on off-the-shelf Foundation Models. However, the value of JANUS lies in **constrained decoding guarantees (GBNF)** and **structured machine-readable feedback for agentic self-repair**, demonstrating that grammar determinism and diagnostic precision, rather than token compression, are the genuine advantages of specialized AI-targeted languages.
