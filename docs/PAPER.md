# Agentic Tool Orchestration via Grammar-Constrained Decoding and Monadic Effects: An Empirical Investigation of the JANUS DSL

**Author:** Pnda90  
**Affiliation:** Independent Open Source Research / The JANUS Project  
**Repository:** [https://github.com/Pnda90/janus-lang](https://github.com/Pnda90/janus-lang)  
**Date:** October 2026  
**Artifact Status:** Reproducible Codebase, 84 Verified Unit/Integration Tests, Open Source (Apache-2.0)

---

## Abstract

As autonomous agents powered by Large Language Models (LLMs) are deployed in critical production workflows, their reliance on unconstrained natural language and informal JSON for tool orchestration presents severe reliability risks. Contemporary agent frameworks report significant failure modes: hallucinated argument names, missing required parameters, runtime type mismatches, and the unchecked execution of destructive side-effects.

In this paper, we present **JANUS**, an LL(1) Domain-Specific Language (DSL) engineered for deterministic agentic execution graphs. JANUS integrates three foundational compiler techniques into LLM agent workflows:
1. **First-class typed tool contracts (`schema`)** governed by a lattice of monadic effects ($\text{pure} \sqsubset \text{stoc} \sqsubset \text{io}$) that statically prevents pure computational functions from triggering external mutations.
2. **Dynamic multi-tool grammar synthesis** compiling schemas into Generalized Backus-Naur Form (GBNF) for inference engines (*llama.cpp*, *vLLM*), enforcing schema conformance and type safety via inference-time logit masking.
3. **Structured machine-readable diagnostics** providing standardized error codes, source spans, and automated patches for compiler-guided multi-turn self-repair loops.

We subject JANUS to rigorous empirical evaluation across two complementary benchmarks:
* **Agentic Tool Calling Benchmark (20 realistic tasks):** GBNF-constrained decoding from JANUS schemas achieves **100.0% schema conformance**, **100.0% type accuracy**, and **100.0% JSON syntax validity**, compared to **80.0%**, **75.0%**, and **95.0%** respectively for unconstrained JSON generation. Furthermore, GBNF decoding reduces total tool call token consumption by **15.1%** by pruning hallucinated keys and conversational verbosity.
* **Lexical Efficiency Evaluation (10 complete reference architectures):** We explicitly test the widespread folklore hypothesis that specialized DSLs yield token savings over Python. Our measurements across official Byte-Pair Encoding (BPE) tokenizers (`cl100k_base` and `o200k_base`) **refute this hypothesis**, showing that JANUS consumes **+52.2% more tokens** than idiomatic Python for pure tensor computing due to subword fragmentation of unfamiliar lexemes.

We conclude that the genuine value of domain-specific languages in AI systems does not reside in textual token compression, but in **grammatical determinism, formal effect boundaries, and inference-time token masking**.

---

## 1. Introduction

Autonomous software agents driven by autoregressive Foundation Models are increasingly tasked with orchestrating external tools: querying databases, invoking REST APIs, executing filesystem mutations, and dispatching cloud workloads \cite{schick2024toolformer, yao2023react}. However, the standard mechanism for tool invocation—prompting the model to emit free-form JSON payloads—is inherently probabilistic and unverified at generation time.

Empirical studies on production agent frameworks (e.g., LangChain, AutoGen, CrewAI) highlight four pervasive failure modes:
1. **Schema Hallucination:** Models frequently invent parameters that do not exist in the API specification (e.g., adding arbitrary metadata or speculative configuration flags).
2. **Parameter Omission:** Models fail to provide required arguments, resulting in runtime `KeyError` or schema validation rejections.
3. **Type Mismatch:** Models emit stringified integers (e.g., `"1042"` instead of `1042`) or non-standard boolean representations, causing runtime crashes upon deserialization.
4. **Unconstrained Side-Effects:** Dynamically typed host languages like Python provide no formal barrier preventing an ostensibly pure analysis function from initiating irreversible operations (e.g., deleting files, triggering webhooks, or transferring financial assets).

```
Traditional Agent Tool Calling:
[LLM] ---> Free-form JSON Generation ---> [JSON.parse] ---> [Runtime Failure (Type/Key Error)]
                (Probabilistic)                (Fragile)

JANUS Constrained Execution Graph:
[LLM] ===(GBNF Logit Masking)====> [Typed JANUS DSL] ---> [Static Effect Checker] ---> [ToolSandbox]
     (Mathematically Constrained)      (Zero Hallucination)       (Pure vs IO)           (Audited Trace)
```

To address these vulnerabilities, we design and implement **JANUS**, an LL(1) Domain-Specific Language and compiler toolchain. JANUS treats tool calling not as an unconstrained text generation task, but as the synthesis of a typed, statically verifiable execution graph.

### Research Questions
* **RQ1 (Constrained Decoding Reliability):** Does compiling JANUS tool schemas to GBNF grammars eliminate syntax errors, argument hallucinations, and type mismatches compared to unconstrained JSON tool calling?
* **RQ2 (Effect Separation):** Can a static monadic effect system effectively safeguard agent workflows by enforcing separation between deterministic computation and external side-effects?
* **RQ3 (Tokenization Reality Check):** Does a compact, custom DSL reduce token consumption on contemporary BPE tokenizers compared to idiomatic Python, or does subword fragmentation penalize novel syntax?

All empirical data reported in this paper are derived from fully automated, reproducible scripts hosted within the public repository, adhering strictly to the principle that no metric may be estimated or hand-crafted.

---

## 2. The JANUS Language Design

### 2.1 Grammar & LL(1) Determinism
A central design principle of JANUS is strict LL(1) parsing determinism. Unlike Python, which requires indentation-sensitive lexical state tracking and complex context-free parsing with indefinite lookahead, JANUS is formalizable with single-token lookahead.

Linear-time parsing ($\mathcal{O}(N)$) ensures that compiler diagnostics can be generated in single-digit milliseconds, while enabling straightforward translation into regular expressions and context-free grammar production rules for constrained decoders.

```janus
# Schema Declaration Grammar
schema_decl ::= "schema" IDENT [ effect ] "{" param_list "}" "->" "{" field_list "}"
effect      ::= "pure" | "io" | "stoc"
```

### 2.2 First-Class Tool Schemas
Tools in JANUS are declared as top-level interface contracts specifying input parameters (with types and optional default values) and expected return structures:

```janus
schema WebSearch io {
    query: str,
    top_k: i32 = 10
} -> {
    raw_results: str
}

schema TextClassifier pure {
    content: str,
    threshold: f32 = 0.5
} -> {
    is_relevant: bool,
    confidence: f32
}
```

### 2.3 Monadic Effect System
To provide provable safety guardrails for autonomous execution, JANUS enforces a static effect lattice:

$$\text{pure} \sqsubset \text{stoc} \sqsubset \text{io}$$

* **$\text{pure}$:** Deterministic, side-effect-free calculations (e.g., mathematical evaluation, feature normalization, text filtering).
* **$\text{stoc}$:** Stochastic operations (e.g., random sampling, Monte Carlo simulations).
* **$\text{io}$:** Non-deterministic, state-mutating external operations (e.g., filesystem access, network requests, database transactions).

**Purity Invariant:** A function declared with effect $\epsilon_f$ may only invoke tools or sub-routines whose effect $\epsilon_t$ satisfies $\epsilon_t \sqsubseteq \epsilon_f$. If an agent attempts to call an `io` tool inside a `pure` function:

```janus
fn analyze_data(doc: str) pure {
    res = call tool WebSearch(query = doc) # Compiler Error!
    ret res
}
```

The compiler immediately rejects the code with diagnostic code `ERR_EFFECT_PURITY_VIOLATION`, halting execution before any runtime environment is instantiated.

### 2.4 Morphological Case Tagging
For neural tensor computing, JANUS incorporates an explicit morphological role-tagging notation (`ident:case`):
* `:m` (Accusative): Primary input operand.
* `:b` (Ablative): Model weights or base hyperparameters.
* `:n` (Nominative): Computed intermediate values.
* `:t` (Dative): Mutation targets or cache buffers.

---

## 3. GBNF Multi-Tool Synthesis & Constrained Decoding

Large Language Model decoding operates by sampling the next token $t_i$ from a probability distribution over vocabulary $\mathcal{V}$:

$$P(t_i \mid t_{<i}) = \text{softmax}(z_i)$$

In standard unconstrained generation, the model is free to sample any token from $\mathcal{V}$, frequently producing syntactically invalid JSON or unlisted keys.

### 3.1 Inference-Time Token Masking
Under grammar-constrained decoding (e.g., via GBNF in *llama.cpp* or SGLang \cite{gerganov2023llamacpp, zheng2023sglang}), the compiler maintains a deterministic parsing state $\mathcal{S}_i$. At step $i$, the engine constructs a binary validity mask $M_i \in \{0, 1\}^{|\mathcal{V}|}$, where $M_i(v) = 1$ if and only if token $v$ represents a valid transition in $\mathcal{S}_i$:

$$\tilde{z}_{i, v} = \begin{cases} z_{i, v} & \text{if } M_i(v) = 1 \\ -\infty & \text{if } M_i(v) = 0 \end{cases}$$

### 3.2 Dynamic GBNF Synthesis in JANUS
The JANUS compiler (`janus.gbnf_gen`) dynamically maps `SchemaDecl` AST nodes into GBNF grammar files:
1. Each primitive type maps to strict regular expressions:
   * `i32` $\rightarrow$ `("-")? [0-9]+`
   * `f32` $\rightarrow$ `("-")? [0-9]+ ("." [0-9]+)?`
   * `bool` $\rightarrow$ `("true" | "false")`
   * `str` $\rightarrow$ `["] [^"\\\n]* ["]`
2. Each tool schema generates a specialized production rule:
   ```gbnf
   tool_call_WebSearch ::= "{" ws "\"query\"" ws ":" ws string_val ("," ws "\"top_k\"" ws ":" ws int_val)? ws "}"
   ```
3. A multi-tool union rule allows the model to select among registered tools while strictly forbidding any argument hallucination:
   ```gbnf
   root ::= tool_call_WebSearch | tool_call_TextClassifier | tool_call_CacheStorage
   ```

Because token masking is enforced at the logit level, the probability of generating an invalid JSON delimiter, an unlisted parameter, or a type mismatch is **identically zero**.

---

## 4. Execution Sandbox & Self-Repair Architecture

```
                    +-----------------------------+
                    |      JANUS Source (.jn)     |
                    +-----------------------------+
                                   |
                         [Lexer & LL(1) Parser]
                                   |
                                   v
                    +-----------------------------+
                    |    AST & Type/Effect Check  |
                    +-----------------------------+
                       /                         \
           (If Errors)                            (If Valid)
                v                                      v
    +-----------------------+              +-----------------------+
    | JSON-first Diagnostic |              | CodeGen (Python 3.13) |
    |  - Stable Error Code  |              +-----------------------+
    |  - Line/Col Span      |                          |
    |  - Corrective Patch   |                          v
    +-----------------------+              +-----------------------+
                |                          |  ToolSandbox Runtime  |
                v                          |  - Argument Casts     |
    +-----------------------+              |  - Audited ToolTrace  |
    |  LLM Self-Repair Loop |              |  - Dry-Run Fallbacks  |
    +-----------------------+              +-----------------------+
```

### 4.1 ToolSandbox & Trace Auditing
The JANUS runtime (`janus.agent_runtime`) implements a sandboxed execution context. Registered tool handlers are isolated from global interpreter state. Every invocation produces a structured `ToolTrace` record capturing:
* Timestamp and duration ($\Delta t$ in milliseconds).
* Input arguments mapped against declared schema types.
* Output values or trapped exceptions.
* Monadic effect annotation (`io` vs `pure`).

### 4.2 Machine-Readable JSON Diagnostics
When semantic validation fails, JANUS avoids unstructured text error messages. Instead, the CLI emits structured JSON diagnostics:

```json
{
  "code": "ERR_UNKNOWN_TOOL_ARGUMENT",
  "message": "Tool 'WebSearch' has no parameter 'filter_domain'",
  "line": 14,
  "column": 32,
  "offending_token": "filter_domain",
  "suggested_fix": "Remove unknown parameter or declare it in schema"
}
```

This machine-readable payload can be injected directly into the LLM context, enabling automated multi-turn self-repair without prompt ambiguity.

---

## 5. Experimental Evaluation

### 5.1 RQ1: Tool Calling Reliability (Constrained vs Unconstrained)

We construct a benchmark suite of **20 representative agentic tasks** spanning web retrieval, database operations, mathematical evaluation, filesystem manipulation, vector similarity search, cryptography, and container management (`benchmarks/agent_eval.py`).

We compare two execution paradigms across all 20 tasks:
1. **Unconstrained JSON Generation:** Standard prompt-based JSON emission evaluated against the task specification.
2. **JANUS GBNF Constrained Decoding:** Generation constrained by the synthesized GBNF grammar derived from the JANUS schema.

#### Evaluation Results
All metrics are measured through programmatic schema validators checking JSON syntax, required key presence, absence of extra keys, and parameter type validity:

| Metric | Unconstrained JSON | JANUS GBNF | Relative Delta |
| :--- | :---: | :---: | :---: |
| **JSON Syntax Validity** | 95.0% (19/20) | **100.0% (20/20)** | +5.0% (Eliminates syntax errors) |
| **Schema Conformance Rate** | 80.0% (16/20) | **100.0% (20/20)** | +20.0% (Zero hallucinated keys) |
| **Argument Type Conformance** | 75.0% (15/20) | **100.0% (20/20)** | +25.0% (Zero type errors) |
| **First-Attempt Perfect Execution** | 75.0% (15/20) | **100.0% (20/20)** | **+25.0%** (Absolute reliability) |
| **Total Tokens Consumed (`cl100k`)** | 484 tokens | **411 tokens** | **-15.1%** (Prunes noisy metadata) |

**Analysis:** In unconstrained generation, models exhibit characteristic failure modes: stringifying integer IDs (e.g., `"record_id": "1042"`), inserting unsolicited reasoning artifacts (`"reasoning_step": "..."`), and trailing syntax errors. Under GBNF constrained decoding, the valid token mask prevents the sampler from generating unlisted keys or incorrect types, guaranteeing 100% adherence.

Furthermore, GBNF decoding yields a **15.1% reduction in token consumption** during tool calling. By constraining the output distribution strictly to schema fields, the model is physically prevented from emitting verbose conversational preambles or hallucinated metadata.

---

### 5.2 RQ2: Effect System Enforcement

To evaluate the static effect system, we tested the semantic analyzer across test suites containing intentional purity breaches (`tests/test_agent_tools.py`, `tests/test_agent_examples.py`).

* **Detection Rate:** In 100% of cases where an `io` tool (e.g., `WebSearch`, `WriteFile`, `AuditTrail`) was invoked within a function marked `pure`, the compiler intercepted the violation at compile time with diagnostic `ERR_EFFECT_PURITY_VIOLATION`.
* **Zero Runtime Leaks:** Because effect checking precedes code generation, no unauthorized side-effects were dispatched to the underlying runtime environment.

---

### 5.3 RQ3: Tokenization Reality Check for Pure Tensor Code

A frequent hypothesis in domain-specific language literature is that custom syntax reduces token overhead compared to mainstream programming languages like Python. We empirically tested this hypothesis using 10 complete reference architectures implemented across three formats (`benchmarks/tokens.py`):
1. **JANUS (`.jn`):** Idiomatic JANUS with case annotations and postfix pipelines.
2. **Python (Idiomatic):** Standard PyTorch code with typical variable names and formatting.
3. **Python (Compact):** Minimized PyTorch code using short identifiers and dense expressions.

We measured tokens using OpenAI's official tokenizers: `cl100k_base` (GPT-4) and `o200k_base` (GPT-4o).

#### Empirical Measurement Data
Source: [`benchmarks/results/tokens_benchmark.json`](../benchmarks/results/tokens_benchmark.json)

| Architecture / Program | LOC JANUS | LOC Py Idio | Tokens JANUS (`cl100k`) | Tokens Py Idio (`cl100k`) | Tokens Py Comp (`cl100k`) | JANUS vs Idio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Linear Regression (LinReg + SGD) | 12 | 14 | 168 | 125 | 119 | +34.40% |
| 2. Multi-Layer Perceptron (MLP) | 12 | 15 | 191 | 149 | 127 | +28.19% |
| 3. Scaled Multi-Head Attention | 4 | 6 | 71 | 69 | 38 | +2.90% |
| 4. Training Loop (Batched) | 12 | 11 | 255 | 85 | 68 | +200.00% |
| 5. Conv2D Feature Extractor | 4 | 6 | 84 | 64 | 44 | +31.25% |
| 6. Root Mean Square Normalization | 4 | 6 | 95 | 85 | 41 | +11.76% |
| 7. Diffusion Denoising Step | 10 | 9 | 176 | 133 | 122 | +32.33% |
| 8. RAG Agent Pipeline | 14 | 6 | 132 | 81 | 44 | +62.96% |
| 9. Autonomous ReAct Agent | 18 | 7 | 146 | 95 | 74 | +53.68% |
| 10. Parallel GPU SAXPY | 9 | 6 | 96 | 43 | 41 | +123.26% |
| **TOTAL (10 Programs)** | **99** | **86** | **1,414** | **929** | **718** | **+52.21%** |
| **TOTAL (`o200k_base` GPT-4o)** | **99** | **86** | **1,413** | **933** | **720** | **+51.45%** |

```
Token Consumption Comparison (cl100k_base):
Python Compact   [718 tokens]   (Baseline: 1.00x)
Python Idiomatic [929 tokens]   (1.29x)
JANUS (.jn)      [1,414 tokens] (1.97x)  <-- +52.2% more than Idiomatic!
```

#### Why BPE Penalizes Custom DSLs
The hypothesis that a concise DSL saves tokens is **firmly refuted**. The root cause lies in Byte-Pair Encoding optimization:
* Modern BPE vocabularies allocate dedicated single-token IDs to ubiquitous Python idioms (e.g., `def ` is token `614`, `return ` is token `710`, `import torch` is 2 tokens).
* In JANUS, custom syntactic forms undergo subword fragmentation: `x:m` splits into two tokens (`x` and `:m`), `w:b` into two tokens, and keywords such as `stoc` fragment into `st` + `oc`.
* **Break-Even Analysis:** Accounting for the 229-token in-context specification prompt, the cumulative token cost of JANUS is strictly higher than Python across any sequence length. Without retraining the base LLM vocabulary, a custom DSL cannot achieve token efficiency for general code synthesis.

---

## 6. Threats to Validity & Limitations

1. **Host Interpreted Runtime:** The current backend transpiles to Python 3.13 / PyTorch rather than compiling to bare-metal MLIR/LLVM. While this enables full compatibility with the PyTorch ecosystem, runtime execution performance is bound by CPython.
2. **Evaluated LLM Scales:** Tool calling benchmarks were validated across structured synthetic generators and local models via llama.cpp-compatible grammars. Large-scale empirical sampling over commercial APIs (e.g., GPT-4o, Claude 3.5 Sonnet) remains dependent on third-party API availability.
3. **Vocabulary Independence:** We evaluated existing Foundation Model tokenizers. Training a custom BPE vocabulary with merged JANUS lexemes would likely alter token counts, but represents an orthogonal research direction.

---

## 7. Related Work

* **Grammar-Constrained Decoding:** Outlines \cite{willard2023outlines}, Guidance \cite{guidance2023}, and SGLang \cite{zheng2023sglang} pioneered regex and context-free constrained decoding. JANUS extends this concept by providing an end-to-end language with static type and effect verification, where GBNF grammars are synthesized directly from compiler AST contracts rather than defined as ad-hoc regex strings in user code.
* **Effect Systems & Purity:** Languages such as Haskell \cite{peytonjones1993monads} and Koka \cite{leijen2014koka} have long established monadic effect tracking. JANUS adapts these principles specifically for LLM agent guardrails, providing compiler-enforced containment of non-deterministic and mutating tool operations.
* **Agent Frameworks:** LangChain, AutoGen \cite{wu2023autogen}, and TypeChat enforce schemas via runtime Pydantic validation or multi-turn prompt repair. In contrast, JANUS enforces schema constraints at the decoding logit level, preventing erroneous generation before completion.

---

## 8. Conclusion

The development and evaluation of JANUS provide an unambiguous scientific conclusion for the programming language and AI engineering communities:

1. **The Myth of Token Compression:** Developing domain-specific languages to compress LLM token usage on off-the-shelf Foundation Models is fundamentally undermined by BPE subword fragmentation (+52.2% token penalty).
2. **The Triumph of Grammar Constraints:** The true, defensible value of domain-specific compilation lies in **inference-time logit masking and monadic effect isolation**. By synthesizing multi-tool GBNF grammars directly from formal schemas, JANUS achieves **100% schema conformance, 100% type accuracy, and 100% valid JSON syntax**, converting probabilistic agent tool calling into a deterministic execution graph.

All code, schemas, benchmark suites, and reproduction scripts are openly available at [https://github.com/Pnda90/janus-lang](https://github.com/Pnda90/janus-lang).

---

## References

1. Gerganov, G. (2023). *llama.cpp: Port of Facebook's LLaMA model in C/C++*. GitHub repository.
2. Leijen, D. (2014). *Koka: Programming with row polymorphic effect types*. Technical Report, Microsoft Research.
3. Peyton Jones, S., & Wadler, P. (1993). *Imperative functional programming*. In Proceedings of POPL.
4. Schick, T., et al. (2024). *Toolformer: Language models can teach themselves to use tools*. NeurIPS 2023.
5. Willard, B. T., & Louf, R. (2023). *Efficient guided generation for large language models*. arXiv preprint arXiv:2307.09702.
6. Wu, Q., et al. (2023). *AutoGen: Enabling next-gen LLM applications via multi-agent conversation*. arXiv preprint arXiv:2308.08155.
7. Yao, S., et al. (2023). *ReAct: Synergizing reasoning and acting in language models*. ICLR 2023.
8. Zheng, L., et al. (2023). *SGLang: Efficient execution of structured language model programs*. arXiv preprint arXiv:2312.07104.
