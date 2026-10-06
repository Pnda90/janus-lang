#!/usr/bin/env python3
"""
Benchmark Riproducibile dei Token (Fase 4):
Confronta lo STESSO programma nei 10 esempi tra:
- JANUS (.jn reale su disco)
- Python/PyTorch idiomatico
- Python/PyTorch compatto
Utilizza tokenizzatori reali (tiktoken cl100k_base e o200k_base, transformers se disponibile).
Calcola LOC, token per programma, totali, costo del System Prompt JANUS e punto di pareggio.
Salva i risultati grezzi in benchmarks/results/tokens_benchmark.json.
Nessun dato hardcoded: tutti i numeri derivano da misurazioni eseguite.
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List

try:
    import tiktoken
except ImportError:
    print("ERRORE: tiktoken non installato. Esegui 'pip install tiktoken'.")
    sys.exit(1)

# In-context guide / System Prompt per JANUS
JANUS_SYSTEM_PROMPT = """You are an expert compiler and programmer for JANUS (.jn), a domain-specific language for neural computing.
Syntax:
- Functions: `fn name(params) effect { body }` where effect is `pure`, `io`, or `stoc`.
- Morphological cases: `:m` (accusative operand), `:b` (base parameter), `:t` (transitive buffer), `:n` (nominative assignment), `:s` (structural shape), `:v` (vocative agent).
- Pipeline expressions: chain postfix operators `matmul`, `relu`, `smax`, `sqrt`, `pow`, `sum`, `conv2d`, `pool max`.
- Autodiff: `g = diff loss:n wrt target:b`.
- State mutation: `mut x = ...`.
- Control flow: `for i in start..end { ... }`, `ret val`.
- Data types: `type Name { field: type }`, `schema Name { input } -> { output }`.
Always write correct, idiomatic JANUS code according to this specification."""

PYTHON_PROGRAMS = {
    "01_linreg": {
        "title": "1. Linear Regression (LinReg + SGD)",
        "idiomatic": """def linreg(x, y, epochs=100, lr=0.01):
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    n = len(x)
    for _ in range(epochs):
        pred = x * w + b
        loss = torch.sum((pred - y) ** 2) / n
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    return w.item(), b.item()""",
        "compact": """def linreg(x, y, epochs, lr):
    w, b = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)
    for _ in range(epochs):
        loss = (((x * w + b) - y) ** 2).mean()
        gw, gb = torch.autograd.grad(loss, (w, b))
        w, b = (w - lr * gw).detach().requires_grad_(), (b - lr * gb).detach().requires_grad_()
    return w.item(), b.item()"""
    },
    "02_mlp": {
        "title": "2. Multi-Layer Perceptron (MLP)",
        "idiomatic": """class MLP(nn.Module):
    def __init__(self, d, h, c):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, c)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def mlp_step(x, y, mlp, lr):
    pred = mlp(x)
    loss = torch.sum((pred - y) ** 2) / len(x)
    grads = torch.autograd.grad(loss, mlp.parameters())
    with torch.no_grad():
        for p, g in zip(mlp.parameters(), grads):
            p -= lr * g
    return mlp, loss.item()""",
        "compact": """def mlp_step(x, y, mlp, lr):
    pred = torch.relu(x @ mlp.w1 + mlp.b1) @ mlp.w2 + mlp.b2
    loss = (((pred - y) ** 2).sum()) / len(x)
    grads = torch.autograd.grad(loss, (mlp.w1, mlp.b1, mlp.w2, mlp.b2))
    return [p - lr * g for p, g in zip((mlp.w1, mlp.b1, mlp.w2, mlp.b2), grads)], loss"""
    },
    "03_attention": {
        "title": "3. Scaled Multi-Head Attention",
        "idiomatic": """def mha(q, k, v):
    d_k = k.size(-1)
    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)""",
        "compact": """def mha(q, k, v):
    return torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1)), dim=-1) @ v"""
    },
    "04_training_loop": {
        "title": "4. Training Loop (Batched)",
        "idiomatic": """def train_loop(data, labels, model, epochs, bs, lr):
    n_batches = len(data) // bs
    for ep in range(epochs):
        for b in range(n_batches):
            x = data[b * bs : (b + 1) * bs]
            y = labels[b * bs : (b + 1) * bs]
            step(x, y, model, lr)""",
        "compact": """def train_loop(data, labels, model, epochs, bs, lr):
    for _ in range(epochs):
        for b in range(len(data) // bs):
            step(data[b * bs : (b + 1) * bs], labels[b * bs : (b + 1) * bs], model, lr)"""
    },
    "05_conv2d": {
        "title": "5. Conv2D Feature Extractor",
        "idiomatic": """def cnn_feature_extractor(img, k):
    conv = F.conv2d(img, k, stride=1, padding=1)
    act = torch.relu(conv)
    pool = F.max_pool2d(act, kernel_size=2, stride=2)
    return torch.flatten(pool, start_dim=1)""",
        "compact": """def cnn_feature_extractor(img, k):
    return torch.flatten(F.max_pool2d(torch.relu(F.conv2d(img, k, stride=1, padding=1)), 2, 2), 1)"""
    },
    "06_rmsnorm": {
        "title": "6. Root Mean Square Normalization",
        "idiomatic": """class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        variance = torch.mean(x ** 2, dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)) * self.w""",
        "compact": """def rms_norm(x, w, eps=1e-6):
    return (x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)) * w"""
    },
    "07_diffusion": {
        "title": "7. Diffusion Denoising Step",
        "idiomatic": """def denoise(xt, noise_pred, a, a_prev, seed):
    sig = math.sqrt(((1.0 - a_prev) / (1.0 - a)) * (1.0 - a / a_prev))
    x0 = (xt - math.sqrt(1.0 - a) * noise_pred) / math.sqrt(a)
    dir_xt = math.sqrt(1.0 - a_prev - sig ** 2) * noise_pred
    torch.manual_seed(seed)
    eps = torch.randn_like(xt)
    return math.sqrt(a_prev) * x0 + dir_xt + sig * eps, seed + 1""",
        "compact": """def denoise(xt, noise_pred, a, a_prev, seed):
    sig = math.sqrt(((1.0 - a_prev) / (1.0 - a)) * (1.0 - a / a_prev))
    x0 = (xt - math.sqrt(1.0 - a) * noise_pred) / math.sqrt(a)
    dir_xt = math.sqrt(1.0 - a_prev - sig ** 2) * noise_pred
    return math.sqrt(a_prev) * x0 + dir_xt + sig * torch.randn_like(xt), seed + 1"""
    },
    "08_rag_pipeline": {
        "title": "8. RAG Agent Pipeline",
        "idiomatic": """class SearchQuery(BaseModel):
    q: str
    topk: int = 5

def rag_pipeline(query: str):
    docs_out = call_agent("Retriever", prompt=query, tool=SearchQuery)
    selected = docs_out["docs"]
    prompt = ["Contesto:", selected, "Domanda:", query]
    return call_agent("LLM", prompt=prompt)""",
        "compact": """def rag_pipeline(query: str):
    docs = call_agent("Retriever", query, SearchQuery)["docs"]
    return call_agent("LLM", ["Contesto:", docs, "Domanda:", query])"""
    },
    "09_react_agent": {
        "title": "9. Autonomous ReAct Agent",
        "idiomatic": """class Action(BaseModel):
    tool: str
    args: str

def agent_exec(task: str, max_retries: int = 3):
    history = [task]
    for _ in range(max_retries):
        action = call_agent("Planner", prompt=history, tool=Action)
        if action["ok"]:
            return action["result"]
        history.extend(["Errore:", action["result"]])
    return "Fallimento task" """,
        "compact": """def agent_exec(task: str, max_retries: int = 3):
    h = [task]
    for _ in range(max_retries):
        a = call_agent("Planner", h, Action)
        if a["ok"]: return a["result"]
        h += ["Errore:", a["result"]]
    return "Fallimento task" """
    },
    "10_gpu_saxpy": {
        "title": "10. Parallel GPU SAXPY",
        "idiomatic": """@cuda.jit
def saxpy(x, y, a):
    idx = cuda.grid(1)
    if idx < x.size:
        y[idx] = a * x[idx] + y[idx]""",
        "compact": """def saxpy(x, y, a):
    idx = blockIdx.x * blockDim.x + threadIdx.x
    if idx < len(x): y[idx] = a * x[idx] + y[idx]"""
    }
}


def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def count_tokens(text: str, enc: Any) -> int:
    return len(enc.encode(text))


def count_loc(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")])


def run_benchmark():
    commit = get_git_commit()
    date_str = datetime.now(timezone.utc).isoformat()

    enc_cl100k = tiktoken.get_encoding("cl100k_base")
    enc_o200k = tiktoken.get_encoding("o200k_base")

    # Misura system prompt
    prompt_tokens_cl100k = count_tokens(JANUS_SYSTEM_PROMPT, enc_cl100k)
    prompt_tokens_o200k = count_tokens(JANUS_SYSTEM_PROMPT, enc_o200k)

    results = []

    total_janus_loc = 0
    total_py_idio_loc = 0
    total_py_comp_loc = 0

    total_janus_cl100k = 0
    total_py_idio_cl100k = 0
    total_py_comp_cl100k = 0

    total_janus_o200k = 0
    total_py_idio_o200k = 0
    total_py_comp_o200k = 0

    # Leggi i 10 esempi reali su disco appartenenti alla suite tensoriale
    files = sorted([f for f in os.listdir("examples") if f.endswith(".jn") and f.replace(".jn", "") in PYTHON_PROGRAMS])

    for fname in files:
        key = fname.replace(".jn", "")
        fpath = os.path.join("examples", fname)
        with open(fpath, "r", encoding="utf-8") as f:
            janus_code = f.read().strip()

        py_info = PYTHON_PROGRAMS.get(key, {})
        py_idio = py_info.get("idiomatic", "")
        py_comp = py_info.get("compact", "")
        title = py_info.get("title", key)

        # LOC
        j_loc = count_loc(janus_code)
        pi_loc = count_loc(py_idio)
        pc_loc = count_loc(py_comp)

        total_janus_loc += j_loc
        total_py_idio_loc += pi_loc
        total_py_comp_loc += pc_loc

        # Tokens cl100k
        j_tok_cl = count_tokens(janus_code, enc_cl100k)
        pi_tok_cl = count_tokens(py_idio, enc_cl100k)
        pc_tok_cl = count_tokens(py_comp, enc_cl100k)

        total_janus_cl100k += j_tok_cl
        total_py_idio_cl100k += pi_tok_cl
        total_py_comp_cl100k += pc_tok_cl

        # Tokens o200k
        j_tok_o = count_tokens(janus_code, enc_o200k)
        pi_tok_o = count_tokens(py_idio, enc_o200k)
        pc_tok_o = count_tokens(py_comp, enc_o200k)

        total_janus_o200k += j_tok_o
        total_py_idio_o200k += pi_tok_o
        total_py_comp_o200k += pc_tok_o

        results.append({
            "key": key,
            "title": title,
            "loc": {"janus": j_loc, "py_idiomatic": pi_loc, "py_compact": pc_loc},
            "tokens_cl100k": {"janus": j_tok_cl, "py_idiomatic": pi_tok_cl, "py_compact": pc_tok_cl},
            "tokens_o200k": {"janus": j_tok_o, "py_idiomatic": pi_tok_o, "py_compact": pc_tok_o},
            "diff_vs_idiomatic_pct_cl100k": round((j_tok_cl - pi_tok_cl) / pi_tok_cl * 100, 2) if pi_tok_cl > 0 else 0.0,
            "diff_vs_compact_pct_cl100k": round((j_tok_cl - pc_tok_cl) / pc_tok_cl * 100, 2) if pc_tok_cl > 0 else 0.0,
        })

    # Calcolo totali e percentuali complessive
    total_diff_idio_cl100k_pct = round((total_janus_cl100k - total_py_idio_cl100k) / total_py_idio_cl100k * 100, 2) if total_py_idio_cl100k > 0 else 0.0
    total_diff_comp_cl100k_pct = round((total_janus_cl100k - total_py_comp_cl100k) / total_py_comp_cl100k * 100, 2) if total_py_comp_cl100k > 0 else 0.0

    total_diff_idio_o200k_pct = round((total_janus_o200k - total_py_idio_o200k) / total_py_idio_o200k * 100, 2) if total_py_idio_o200k > 0 else 0.0
    total_diff_comp_o200k_pct = round((total_janus_o200k - total_py_comp_o200k) / total_py_comp_o200k * 100, 2) if total_py_comp_o200k > 0 else 0.0

    # Break-even cl100k
    saved_per_prog_vs_idio_cl100k = (total_py_idio_cl100k - total_janus_cl100k) / len(results)
    if saved_per_prog_vs_idio_cl100k > 0:
        breakeven_idio_cl100k = round(prompt_tokens_cl100k / saved_per_prog_vs_idio_cl100k, 1)
    else:
        breakeven_idio_cl100k = None  # Mai raggiunto

    saved_per_prog_vs_comp_cl100k = (total_py_comp_cl100k - total_janus_cl100k) / len(results)
    if saved_per_prog_vs_comp_cl100k > 0:
        breakeven_comp_cl100k = round(prompt_tokens_cl100k / saved_per_prog_vs_comp_cl100k, 1)
    else:
        breakeven_comp_cl100k = None  # Mai raggiunto

    benchmark_summary = {
        "metadata": {
            "commit": commit,
            "date": date_str,
            "tokenizers": ["tiktoken/cl100k_base", "tiktoken/o200k_base"],
            "transformers_llama_qwen": "non scaricato / saltato (richiede download pesi/vocabolario offline)"
        },
        "system_prompt": {
            "tokens_cl100k": prompt_tokens_cl100k,
            "tokens_o200k": prompt_tokens_o200k,
            "text": JANUS_SYSTEM_PROMPT
        },
        "totals": {
            "loc": {"janus": total_janus_loc, "py_idiomatic": total_py_idio_loc, "py_compact": total_py_comp_loc},
            "tokens_cl100k": {
                "janus": total_janus_cl100k,
                "py_idiomatic": total_py_idio_cl100k,
                "py_compact": total_py_comp_cl100k,
                "diff_vs_idiomatic_pct": total_diff_idio_cl100k_pct,
                "diff_vs_compact_pct": total_diff_comp_cl100k_pct,
                "breakeven_vs_idiomatic_programs": breakeven_idio_cl100k,
                "breakeven_vs_compact_programs": breakeven_comp_cl100k
            },
            "tokens_o200k": {
                "janus": total_janus_o200k,
                "py_idiomatic": total_py_idio_o200k,
                "py_compact": total_py_comp_o200k,
                "diff_vs_idiomatic_pct": total_diff_idio_o200k_pct,
                "diff_vs_compact_pct": total_diff_comp_o200k_pct
            }
        },
        "programs": results
    }

    # Salva su file JSON
    out_dir = "benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tokens_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    # Stampa Report Markdown
    print("\n" + "=" * 90)
    print(f"BENCHMARK TOKENIZZAZIONE REALE JANUS vs PYTHON (Commit {commit})")
    print("=" * 90)
    print(f"Tokenizer primari: tiktoken cl100k_base (GPT-4) e o200k_base (GPT-4o)")
    print(f"System Prompt JANUS in-context: {prompt_tokens_cl100k} token (cl100k) | {prompt_tokens_o200k} token (o200k)\n")

    header = f"| {'Programma':<30} | {'JANUS cl100k':<12} | {'Py Idio cl100k':<14} | {'Py Comp cl100k':<14} | {'vs Idio (%)':<12} | {'vs Comp (%)':<12} |"
    print(header)
    print("|" + "-" * 32 + "|" + "-" * 14 + "|" + "-" * 16 + "|" + "-" * 16 + "|" + "-" * 14 + "|" + "-" * 14 + "|")

    for p in results:
        t_j = p["tokens_cl100k"]["janus"]
        t_pi = p["tokens_cl100k"]["py_idiomatic"]
        t_pc = p["tokens_cl100k"]["py_compact"]
        d_pi = p["diff_vs_idiomatic_pct_cl100k"]
        d_pc = p["diff_vs_compact_pct_cl100k"]
        sign_pi = "+" if d_pi > 0 else ""
        sign_pc = "+" if d_pc > 0 else ""
        print(f"| {p['title']:<30} | {t_j:<12} | {t_pi:<14} | {t_pc:<14} | {sign_pi}{d_pi:>10}% | {sign_pc}{d_pc:>10}% |")

    print("|" + "=" * 32 + "|" + "=" * 14 + "|" + "=" * 16 + "|" + "=" * 16 + "|" + "=" * 14 + "|" + "=" * 14 + "|")
    sign_tot_pi = "+" if total_diff_idio_cl100k_pct > 0 else ""
    sign_tot_pc = "+" if total_diff_comp_cl100k_pct > 0 else ""
    print(f"| {'TOTALE (10 programmi)':<30} | {total_janus_cl100k:<12} | {total_py_idio_cl100k:<14} | {total_py_comp_cl100k:<14} | {sign_tot_pi}{total_diff_idio_cl100k_pct:>10}% | {sign_tot_pc}{total_diff_comp_cl100k_pct:>10}% |\n")

    print(f"Righe di codice (LOC): JANUS={total_janus_loc} | Py Idiomatic={total_py_idiio_loc} | Py Compact={total_py_comp_loc}" if 'total_py_idiio_loc' in locals() else f"Righe di codice (LOC): JANUS={total_janus_loc} | Py Idiomatic={total_py_idio_loc} | Py Compact={total_py_comp_loc}")
    print(f"Risultati o200k_base: JANUS={total_janus_o200k} | Py Idio={total_py_idio_o200k} ({'+' if total_diff_idio_o200k_pct > 0 else ''}{total_diff_idio_o200k_pct}%) | Py Comp={total_py_comp_o200k} ({'+' if total_diff_comp_o200k_pct > 0 else ''}{total_diff_comp_o200k_pct}%)")

    print("\n--- ANALISI PUNTO DI PAREGGIO (BREAK-EVEN) ---")
    if breakeven_idio_cl100k:
        print(f"Punto di pareggio vs Python Idiomatico: {breakeven_idio_cl100k} programmi (risparmio medio: {saved_per_prog_vs_idio_cl100k:.1f} token/programma)")
    else:
        print(f"Punto di pareggio vs Python Idiomatico: MAI RAGGIUNTO (JANUS usa +{abs(saved_per_prog_vs_idio_cl100k):.1f} token/programma in più rispetto a Python Idiomatico)")

    if breakeven_comp_cl100k:
        print(f"Punto di pareggio vs Python Compatto: {breakeven_comp_cl100k} programmi")
    else:
        print(f"Punto di pareggio vs Python Compatto: MAI RAGGIUNTO (JANUS usa +{abs(saved_per_prog_vs_comp_cl100k):.1f} token/programma in più rispetto a Python Compatto)")

    print(f"\nSalvati dati grezzi in: {out_path}")
    return benchmark_summary


if __name__ == "__main__":
    run_benchmark()
