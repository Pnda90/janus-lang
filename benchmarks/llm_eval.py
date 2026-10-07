#!/usr/bin/env python3
"""
benchmarks/llm_eval.py

Harness di valutazione LLM per JANUS vs Python su 30 task di calcolo tensoriale.
Supporta:
- Misurazione pass@1 e pass@5 con n campioni indipendenti.
- Ciclo di autoriparazione multi-turn guidato da diagnostica JSON strutturata (per JANUS)
  e traceback di runtime/sintassi (per Python).
- Tracciamento completo di token (prompt, sistema, completamento, riparazione) via tiktoken.
- Provider configurabili da variabile d'ambiente (OpenAI, Anthropic, Ollama, o --dry-run simulato).
- Salvataggio dei risultati grezzi e metadati in benchmarks/results/llm_eval_*.json.
"""

import os
import sys
import json
import time
import math
import argparse
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Callable

# Tokenizer per calcolo accurato dei token
try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(ENC.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return len(text.split())

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from janus.lexer import Lexer
from janus.parser import Parser, ParseError
from janus.type_checker import TypeChecker
from janus.codegen import CodeGenerator

# ---------------------------------------------------------------------------
# 1. DEFINIZIONE DEI 30 TASK TENSORIALI
# ---------------------------------------------------------------------------

TASKS: List[Dict[str, Any]] = [
    {
        "id": "task_01",
        "name": "Linear Regression Forward",
        "fn_name": "linreg_fwd",
        "description": "Write a pure function `linreg_fwd(x, w, b)` that computes element-wise linear prediction `x * w + b`.",
        "janus_sig": "fn linreg_fwd(x:m, w:b, b:b) pure",
        "python_sig": "def linreg_fwd(x, w, b):",
        "janus_ref": "fn linreg_fwd(x:m, w:b, b:b) pure {\n    ret x:m * w:b + b:b\n}",
        "python_ref": "def linreg_fwd(x, w, b):\n    return x * w + b",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([1.0, 2.0]), torch.tensor(0.5), torch.tensor(1.0))), torch.tensor([1.5, 2.0]))
    },
    {
        "id": "task_02",
        "name": "Mean Squared Error Loss",
        "fn_name": "mse",
        "description": "Write a pure function `mse(y_pred, y_true)` computing mean squared error: `((y_pred - y_true) pow 2) sum / y_pred.len`.",
        "janus_sig": "fn mse(y_pred:m, y_true:m) pure",
        "python_sig": "def mse(y_pred, y_true):",
        "janus_ref": "fn mse(y_pred:m, y_true:m) pure {\n    ret ((y_pred:m - y_true:m) pow 2) sum / y_pred:m.len\n}",
        "python_ref": "def mse(y_pred, y_true):\n    return ((y_pred - y_true) ** 2).sum() / len(y_pred)",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]))), torch.tensor(2.0))
    },
    {
        "id": "task_03",
        "name": "ReLU Activation",
        "fn_name": "relu_op",
        "description": "Write a pure function `relu_op(x)` that applies ReLU activation: `x relu`.",
        "janus_sig": "fn relu_op(x:m) pure",
        "python_sig": "def relu_op(x):",
        "janus_ref": "fn relu_op(x:m) pure {\n    ret x:m relu\n}",
        "python_ref": "def relu_op(x):\n    return torch.relu(x)",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([-1.0, 2.0]))), torch.tensor([0.0, 2.0]))
    },
    {
        "id": "task_04",
        "name": "Matrix Multiplication",
        "fn_name": "mat_mul",
        "description": "Write a pure function `mat_mul(a, b)` that performs matrix multiplication: `a matmul b`.",
        "janus_sig": "fn mat_mul(a:m, b:b) pure",
        "python_sig": "def mat_mul(a, b):",
        "janus_ref": "fn mat_mul(a:m, b:b) pure {\n    ret a:m matmul b:b\n}",
        "python_ref": "def mat_mul(a, b):\n    return a @ b",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.ones(2, 3), torch.ones(3, 2))), torch.full((2, 2), 3.0))
    },
    {
        "id": "task_05",
        "name": "Vector Dot Product",
        "fn_name": "dot_product",
        "description": "Write a pure function `dot_product(a, b)` computing inner dot product of 1D vectors: `(a * b) sum`.",
        "janus_sig": "fn dot_product(a:m, b:m) pure",
        "python_sig": "def dot_product(a, b):",
        "janus_ref": "fn dot_product(a:m, b:m) pure {\n    ret (a:m * b:m) sum\n}",
        "python_ref": "def dot_product(a, b):\n    return (a * b).sum()",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))), torch.tensor(11.0))
    },
    {
        "id": "task_06",
        "name": "Squared L2 Norm",
        "fn_name": "l2_sq",
        "description": "Write a pure function `l2_sq(w)` computing squared L2 norm: `(w pow 2) sum`.",
        "janus_sig": "fn l2_sq(w:m) pure",
        "python_sig": "def l2_sq(w):",
        "janus_ref": "fn l2_sq(w:m) pure {\n    ret (w:m pow 2) sum\n}",
        "python_ref": "def l2_sq(w):\n    return (w ** 2).sum()",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([2.0, 3.0]))), torch.tensor(13.0))
    },
    {
        "id": "task_07",
        "name": "SGD Parameter Step",
        "fn_name": "sgd_step",
        "description": "Write a pure function `sgd_step(w, g, lr)` performing one SGD parameter update: `w - lr * g`.",
        "janus_sig": "fn sgd_step(w:m, g:m, lr: f32) pure",
        "python_sig": "def sgd_step(w, g, lr):",
        "janus_ref": "fn sgd_step(w:m, g:m, lr: f32) pure {\n    ret w:m - lr * g:m\n}",
        "python_ref": "def sgd_step(w, g, lr):\n    return w - lr * g",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([1.0, 2.0]), torch.tensor([0.1, 0.2]), 0.1)), torch.tensor([0.99, 1.98]))
    },
    {
        "id": "task_08",
        "name": "RMSNorm",
        "fn_name": "rmsnorm",
        "description": "Write a pure function `rmsnorm(x, w, eps)` computing RMS Normalization: `variance = (x pow 2) sum ax:b -1 keep:b true / x.dim_last; ret (x / (variance + eps sqrt)) * w`.",
        "janus_sig": "fn rmsnorm(x:m, w:b, eps: f32) pure",
        "python_sig": "def rmsnorm(x, w, eps):",
        "janus_ref": "fn rmsnorm(x:m, w:b, eps: f32) pure {\n    variance = (x:m pow 2) sum ax:b -1 keep:b true / x:m.dim_last\n    ret (x:m / (variance + eps sqrt)) * w:b\n}",
        "python_ref": "def rmsnorm(x, w, eps):\n    v = (x ** 2).sum(dim=-1, keepdim=True) / x.size(-1)\n    return (x / torch.sqrt(v + eps)) * w",
        "test": lambda fn: fn(torch.tensor([[1.0, 2.0]]), torch.tensor([1.0, 1.0]), 1e-5).shape == (1, 2)
    },
    {
        "id": "task_09",
        "name": "Softmax",
        "fn_name": "softmax_last",
        "description": "Write a pure function `softmax_last(x)` computing softmax along the last dimension: `x smax`.",
        "janus_sig": "fn softmax_last(x:m) pure",
        "python_sig": "def softmax_last(x):",
        "janus_ref": "fn softmax_last(x:m) pure {\n    ret x:m smax\n}",
        "python_ref": "def softmax_last(x):\n    return torch.softmax(x, dim=-1)",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([0.0, 0.0]))), torch.tensor([0.5, 0.5]))
    },
    {
        "id": "task_10",
        "name": "Scaled Dot-Product Attention",
        "fn_name": "scaled_attn",
        "description": "Write a pure function `scaled_attn(q, k, v, scale)` computing attention: `scores = (q matmul k.trans) * scale; weights = scores smax; ret weights matmul v`.",
        "janus_sig": "fn scaled_attn(q:m, k:b, v:b, scale: f32) pure",
        "python_sig": "def scaled_attn(q, k, v, scale):",
        "janus_ref": "fn scaled_attn(q:m, k:b, v:b, scale: f32) pure {\n    scores = (q:m matmul k:b.trans) * scale\n    weights = scores smax\n    ret weights matmul v:b\n}",
        "python_ref": "def scaled_attn(q, k, v, scale):\n    scores = (q @ k.transpose(-2, -1)) * scale\n    return torch.softmax(scores, dim=-1) @ v",
        "test": lambda fn: fn(torch.ones(1, 2, 4), torch.ones(1, 2, 4), torch.ones(1, 2, 4), 0.5).shape == (1, 2, 4)
    },
    {
        "id": "task_11",
        "name": "Residual Addition",
        "fn_name": "res_add",
        "description": "Write a pure function `res_add(x, res)` adding residual connection `x + res`.",
        "janus_sig": "fn res_add(x:m, res:m) pure",
        "python_sig": "def res_add(x, res):",
        "janus_ref": "fn res_add(x:m, res:m) pure {\n    ret x:m + res:m\n}",
        "python_ref": "def res_add(x, res):\n    return x + res",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0]))), torch.tensor([3.0, 5.0]))
    },
    {
        "id": "task_12",
        "name": "Scale Tensor by Constant",
        "fn_name": "scale_tensor",
        "description": "Write a pure function `scale_tensor(x, alpha)` multiplying tensor `x` by scalar `alpha`.",
        "janus_sig": "fn scale_tensor(x:m, alpha: f32) pure",
        "python_sig": "def scale_tensor(x, alpha):",
        "janus_ref": "fn scale_tensor(x:m, alpha: f32) pure {\n    ret x:m * alpha\n}",
        "python_ref": "def scale_tensor(x, alpha):\n    return x * alpha",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([2.0, 4.0]), 3.0)), torch.tensor([6.0, 12.0]))
    },
    {
        "id": "task_13",
        "name": "Vector Subtraction",
        "fn_name": "vec_sub",
        "description": "Write a pure function `vec_sub(a, b)` returning vector difference `a - b`.",
        "janus_sig": "fn vec_sub(a:m, b:m) pure",
        "python_sig": "def vec_sub(a, b):",
        "janus_ref": "fn vec_sub(a:m, b:m) pure {\n    ret a:m - b:m\n}",
        "python_ref": "def vec_sub(a, b):\n    return a - b",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([5.0, 8.0]), torch.tensor([3.0, 2.0]))), torch.tensor([2.0, 6.0]))
    },
    {
        "id": "task_14",
        "name": "Single Dense Layer with ReLU",
        "fn_name": "mlp_layer",
        "description": "Write a pure function `mlp_layer(x, w, b)` computing `(x matmul w + b) relu`.",
        "janus_sig": "fn mlp_layer(x:m, w:b, b:b) pure",
        "python_sig": "def mlp_layer(x, w, b):",
        "janus_ref": "fn mlp_layer(x:m, w:b, b:b) pure {\n    ret (x:m matmul w:b + b:b) relu\n}",
        "python_ref": "def mlp_layer(x, w, b):\n    return torch.relu(x @ w + b)",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([[-1.0, 2.0]]), torch.tensor([[1.0], [0.5]]), torch.tensor([0.5]))), torch.tensor([[0.5]]))
    },
    {
        "id": "task_15",
        "name": "Cosine Similarity",
        "fn_name": "cos_sim",
        "description": "Write a pure function `cos_sim(u, v)` computing cosine similarity between 1D vectors: `((u * v) sum) / (((u pow 2) sum sqrt) * ((v pow 2) sum sqrt))`.",
        "janus_sig": "fn cos_sim(u:m, v:m) pure",
        "python_sig": "def cos_sim(u, v):",
        "janus_ref": "fn cos_sim(u:m, v:m) pure {\n    prod = (u:m * v:m) sum\n    nu = (u:m pow 2) sum sqrt\n    nv = (v:m pow 2) sum sqrt\n    ret prod / (nu * nv)\n}",
        "python_ref": "def cos_sim(u, v):\n    return (u * v).sum() / (torch.sqrt((u**2).sum()) * torch.sqrt((v**2).sum()))",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))), torch.tensor(1.0))
    },
    {
        "id": "task_16",
        "name": "GELU Activation",
        "fn_name": "gelu_act",
        "description": "Write a pure function `gelu_act(x)` computing GELU activation: `x gelu`.",
        "janus_sig": "fn gelu_act(x:m) pure",
        "python_sig": "def gelu_act(x):",
        "janus_ref": "fn gelu_act(x:m) pure {\n    ret x:m gelu\n}",
        "python_ref": "def gelu_act(x):\n    import torch.nn.functional as F\n    return F.gelu(x)",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([0.0, 1.0]))), torch.nn.functional.gelu(torch.tensor([0.0, 1.0])))
    },
    {
        "id": "task_17",
        "name": "Power Operation",
        "fn_name": "pow_op",
        "description": "Write a pure function `pow_op(x, p)` computing element-wise power: `x pow p`.",
        "janus_sig": "fn pow_op(x:m, p: f32) pure",
        "python_sig": "def pow_op(x, p):",
        "janus_ref": "fn pow_op(x:m, p: f32) pure {\n    ret x:m pow p\n}",
        "python_ref": "def pow_op(x, p):\n    return torch.pow(x, p)",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([3.0, 2.0]), 2.0)), torch.tensor([9.0, 4.0]))
    },
    {
        "id": "task_18",
        "name": "Square Root",
        "fn_name": "sqrt_op",
        "description": "Write a pure function `sqrt_op(x)` computing element-wise square root: `x sqrt`.",
        "janus_sig": "fn sqrt_op(x:m) pure",
        "python_sig": "def sqrt_op(x):",
        "janus_ref": "fn sqrt_op(x:m) pure {\n    ret x:m sqrt\n}",
        "python_ref": "def sqrt_op(x):\n    return torch.sqrt(x)",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([16.0, 9.0]))), torch.tensor([4.0, 3.0]))
    },
    {
        "id": "task_19",
        "name": "Sum All Elements",
        "fn_name": "sum_all",
        "description": "Write a pure function `sum_all(x)` computing total sum of tensor elements: `x sum`.",
        "janus_sig": "fn sum_all(x:m) pure",
        "python_sig": "def sum_all(x):",
        "janus_ref": "fn sum_all(x:m) pure {\n    ret x:m sum\n}",
        "python_ref": "def sum_all(x):\n    return torch.sum(x)",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([1.0, 2.0, 3.0]))), torch.tensor(6.0))
    },
    {
        "id": "task_20",
        "name": "Weight Decay Step",
        "fn_name": "weight_decay",
        "description": "Write a pure function `weight_decay(g, w, decay)` returning decayed gradient `g + decay * w`.",
        "janus_sig": "fn weight_decay(g:m, w:b, decay: f32) pure",
        "python_sig": "def weight_decay(g, w, decay):",
        "janus_ref": "fn weight_decay(g:m, w:b, decay: f32) pure {\n    ret g:m + decay * w:b\n}",
        "python_ref": "def weight_decay(g, w, decay):\n    return g + decay * w",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]), 0.1)), torch.tensor([1.2, 2.4]))
    },
    {
        "id": "task_21",
        "name": "Linear Projection",
        "fn_name": "linear_proj",
        "description": "Write a pure function `linear_proj(x, w)` computing linear matrix projection `x matmul w`.",
        "janus_sig": "fn linear_proj(x:m, w:b) pure",
        "python_sig": "def linear_proj(x, w):",
        "janus_ref": "fn linear_proj(x:m, w:b) pure {\n    ret x:m matmul w:b\n}",
        "python_ref": "def linear_proj(x, w):\n    return x @ w",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0], [1.0]]))), torch.tensor([[3.0]]))
    },
    {
        "id": "task_22",
        "name": "Squared Difference",
        "fn_name": "diff_sq",
        "description": "Write a pure function `diff_sq(a, b)` returning element-wise squared difference `(a - b) pow 2`.",
        "janus_sig": "fn diff_sq(a:m, b:m) pure",
        "python_sig": "def diff_sq(a, b):",
        "janus_ref": "fn diff_sq(a:m, b:m) pure {\n    ret (a:m - b:m) pow 2\n}",
        "python_ref": "def diff_sq(a, b):\n    return (a - b) ** 2",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([4.0, 5.0]), torch.tensor([1.0, 2.0]))), torch.tensor([9.0, 9.0]))
    },
    {
        "id": "task_23",
        "name": "1D Mean",
        "fn_name": "mean_1d",
        "description": "Write a pure function `mean_1d(x)` returning the mean of a 1D tensor `x sum / x.len`.",
        "janus_sig": "fn mean_1d(x:m) pure",
        "python_sig": "def mean_1d(x):",
        "janus_ref": "fn mean_1d(x:m) pure {\n    ret x:m sum / x:m.len\n}",
        "python_ref": "def mean_1d(x):\n    return x.sum() / len(x)",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([2.0, 4.0]))), torch.tensor(3.0))
    },
    {
        "id": "task_24",
        "name": "Weighted Sum",
        "fn_name": "weighted_sum",
        "description": "Write a pure function `weighted_sum(x, w)` computing `(x * w) sum`.",
        "janus_sig": "fn weighted_sum(x:m, w:b) pure",
        "python_sig": "def weighted_sum(x, w):",
        "janus_ref": "fn weighted_sum(x:m, w:b) pure {\n    ret (x:m * w:b) sum\n}",
        "python_ref": "def weighted_sum(x, w):\n    return (x * w).sum()",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.5]))), torch.tensor(1.5))
    },
    {
        "id": "task_25",
        "name": "Euclidean Norm (L2)",
        "fn_name": "norm_l2",
        "description": "Write a pure function `norm_l2(x)` returning Euclidean norm: `((x pow 2) sum) sqrt`.",
        "janus_sig": "fn norm_l2(x:m) pure",
        "python_sig": "def norm_l2(x):",
        "janus_ref": "fn norm_l2(x:m) pure {\n    ret ((x:m pow 2) sum) sqrt\n}",
        "python_ref": "def norm_l2(x):\n    return torch.sqrt((x ** 2).sum())",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([3.0, 4.0]))), torch.tensor(5.0))
    },
    {
        "id": "task_26",
        "name": "Sum Last Dimension with Keepdim",
        "fn_name": "sum_last_dim",
        "description": "Write a pure function `sum_last_dim(x)` computing `x sum ax:b -1 keep:b true`.",
        "janus_sig": "fn sum_last_dim(x:m) pure",
        "python_sig": "def sum_last_dim(x):",
        "janus_ref": "fn sum_last_dim(x:m) pure {\n    ret x:m sum ax:b -1 keep:b true\n}",
        "python_ref": "def sum_last_dim(x):\n    return torch.sum(x, dim=-1, keepdim=True)",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))), torch.tensor([[3.0], [7.0]]))
    },
    {
        "id": "task_27",
        "name": "Flatten 2D Tensor",
        "fn_name": "flatten_op",
        "description": "Write a pure function `flatten_op(x)` flattening tensor from dimension 1: `x flat 1`.",
        "janus_sig": "fn flatten_op(x:m) pure",
        "python_sig": "def flatten_op(x):",
        "janus_ref": "fn flatten_op(x:m) pure {\n    ret x:m flat 1\n}",
        "python_ref": "def flatten_op(x):\n    return torch.flatten(x, start_dim=1)",
        "test": lambda fn: fn(torch.zeros(2, 3, 4)).shape == (2, 12)
    },
    {
        "id": "task_28",
        "name": "Add Bias Vector",
        "fn_name": "add_bias",
        "description": "Write a pure function `add_bias(x, b)` computing `x + b`.",
        "janus_sig": "fn add_bias(x:m, b:b) pure",
        "python_sig": "def add_bias(x, b):",
        "janus_ref": "fn add_bias(x:m, b:b) pure {\n    ret x:m + b:b\n}",
        "python_ref": "def add_bias(x, b):\n    return x + b",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0]))), torch.tensor([3.0, 5.0]))
    },
    {
        "id": "task_29",
        "name": "Element-wise Multiplication",
        "fn_name": "mul_elem",
        "description": "Write a pure function `mul_elem(a, b)` returning `a * b`.",
        "janus_sig": "fn mul_elem(a:m, b:m) pure",
        "python_sig": "def mul_elem(a, b):",
        "janus_ref": "fn mul_elem(a:m, b:m) pure {\n    ret a:m * b:m\n}",
        "python_ref": "def mul_elem(a, b):\n    return a * b",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([2.0, 3.0]), torch.tensor([3.0, 4.0]))), torch.tensor([6.0, 12.0]))
    },
    {
        "id": "task_30",
        "name": "Euclidean Distance (Norm of Difference)",
        "fn_name": "norm_diff",
        "description": "Write a pure function `norm_diff(a, b)` computing L2 distance between vectors `a` and `b`: `((a - b) pow 2) sum sqrt`.",
        "janus_sig": "fn norm_diff(a:m, b:m) pure",
        "python_sig": "def norm_diff(a, b):",
        "janus_ref": "fn norm_diff(a:m, b:m) pure {\n    ret ((a:m - b:m) pow 2) sum sqrt\n}",
        "python_ref": "def norm_diff(a, b):\n    return torch.sqrt(((a - b) ** 2).sum())",
        "test": lambda fn: torch.allclose(torch.as_tensor(fn(torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]))), torch.tensor(5.0))
    }
]

# ---------------------------------------------------------------------------
# 2. SYSTEM PROMPT & SPECIFICHE
# ---------------------------------------------------------------------------

JANUS_SYSTEM_PROMPT = """You are an expert compiler and programmer for JANUS (.jn), a domain-specific language for neural computing.
Syntax:
- Functions: `fn name(params) effect { body }` where effect is `pure`, `io`, or `stoc`.
- Morphological cases: `:m` (accusative operand), `:b` (base parameter), `:t` (transitive buffer), `:n` (nominative assignment), `:s` (structural shape), `:v` (vocative agent).
- Pipeline expressions: chain postfix operators `matmul`, `relu`, `smax`, `sqrt`, `pow`, `sum`, `conv2d`, `pool max`, `flat`.
- Autodiff: `g = diff loss:n wrt target:b`.
- State mutation: `mut x:b = ...`.
- Control flow: `for i in start..end { ... }`, `ret val`.
Return ONLY the raw JANUS function definition within ```janus ... ``` or plain text, without conversational fluff."""

PYTHON_SYSTEM_PROMPT = """You are an expert Python and PyTorch engineer.
Write clean, idiomatic PyTorch functions without classes or scripts.
Return ONLY the function code within ```python ... ``` or plain text, without conversational fluff."""

# ---------------------------------------------------------------------------
# 3. CLIENT LLM & GESTIONE ERRORI
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, provider: str, model: str, dry_run: bool = False):
        self.provider = provider
        self.model = model
        self.dry_run = dry_run

    def generate(self, messages: List[Dict[str, str]], task_id: str, lang: str, sample_idx: int, repair_attempt: int) -> str:
        if self.dry_run or self.provider == "mock":
            return self._mock_generate(messages, task_id, lang, sample_idx, repair_attempt)
        elif self.provider == "openai":
            return self._call_openai(messages)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages)
        elif self.provider == "ollama":
            return self._call_ollama(messages)
        else:
            raise ValueError(f"Provider sconosciuto: {self.provider}")

    def _mock_generate(self, messages: List[Dict[str, str]], task_id: str, lang: str, sample_idx: int, repair_attempt: int) -> str:
        """
        Simulatore deterministico per --dry-run:
        - Per verificare il ciclo di autoriparazione: al tentativo 0 per il campione 0 del task_01,
          restituisce intenzionalmente un errore di sintassi (mancanza di annotazione caso),
          e al tentativo di riparazione restituisce il codice corretto.
        - Per tutti gli altri campioni restituisce l'implementazione di riferimento.
        """
        task = next(t for t in TASKS if t["id"] == task_id)
        if task_id == "task_01" and sample_idx == 0 and repair_attempt == 0:
            if lang == "janus":
                # Codice volutamente errato (manca operando destro: scatena ERR_SYNTAX nel parser)
                return "fn linreg_fwd(x:m, w:b, b:b) pure {\n    ret x:m * w:b + \n}"
            else:
                return "def linreg_fwd(x, w, b):\n    ret x * w + b" # sintassi errata: 'ret' non valido in Python

        # Risposta corretta di riferimento
        if lang == "janus":
            return f"```janus\n{task['janus_ref']}\n```"
        else:
            return f"```python\n{task['python_ref']}\n```"

    def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        import requests
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY non impostata nell'ambiente.")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2
        }
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_anthropic(self, messages: List[Dict[str, str]]) -> str:
        import requests
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY non impostata nell'ambiente.")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        # Separazione eventuale del messaggio di sistema
        system_content = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                user_msgs.append(m)
        payload = {
            "model": self.model,
            "system": system_content,
            "messages": user_msgs,
            "max_tokens": 1024,
            "temperature": 0.2
        }
        resp = requests.post(f"{base_url}/messages", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        import requests
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2
        }
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

# ---------------------------------------------------------------------------
# 4. VALUTATORE DEL CODICE (COMPILAZIONE & TEST ESEGUIBILE)
# ---------------------------------------------------------------------------

def extract_code(raw: str, lang: str) -> str:
    """Estrae il codice pulito da eventuali blocchi markdown triple-backtick."""
    raw = raw.strip()
    fence = f"```{lang}"
    if fence in raw:
        start = raw.index(fence) + len(fence)
        end = raw.find("```", start)
        return raw[start:end].strip() if end != -1 else raw[start:].strip()
    elif "```" in raw:
        start = raw.index("```") + 3
        newline = raw.find("\n", start)
        if newline != -1 and newline < start + 15:
            start = newline + 1
        end = raw.find("```", start)
        return raw[start:end].strip() if end != -1 else raw[start:].strip()
    return raw

def evaluate_janus(code_str: str, task: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Verifica ed esegue il codice JANUS.
    Restituisce: (successo, python_generato_o_none, json_diagnostica_se_fallito)
    """
    try:
        tokens = Lexer(code_str).tokenize()
        ast = Parser(tokens).parse()
        checker = TypeChecker()
        diags = checker.check(ast)
        errs = [d for d in diags if d.code.startswith("ERR")]
        if errs:
            diag_json = json.dumps([d.to_dict() for d in errs], indent=2)
            return False, None, diag_json

        py_code = CodeGenerator().generate(ast)
        scope = {}
        exec(py_code, scope)

        if task["fn_name"] not in scope:
            err_diag = json.dumps([{
                "status": "error",
                "code": "ERR_ENTRYPOINT_MISSING",
                "message": f"Attesa funzione '{task['fn_name']}' non trovata nel sorgente generato."
            }], indent=2)
            return False, py_code, err_diag

        fn = scope[task["fn_name"]]
        test_passed = task["test"](fn)
        if not test_passed:
            err_diag = json.dumps([{
                "status": "error",
                "code": "ERR_TEST_ASSERTION_FAILED",
                "message": "Il test di accettazione PyTorch ha fallito l'uguaglianza numerica."
            }], indent=2)
            return False, py_code, err_diag

        return True, py_code, None

    except ParseError as pe:
        diag = {
            "status": "error",
            "code": "ERR_SYNTAX",
            "phase": "parser",
            "message": pe.message,
            "span": {"line": pe.token.line, "col": pe.token.col, "len": pe.token.length},
            "offending": pe.token.value
        }
        return False, None, json.dumps([diag], indent=2)
    except Exception as e:
        diag = {
            "status": "error",
            "code": "ERR_RUNTIME",
            "message": f"{type(e).__name__}: {str(e)}"
        }
        return False, None, json.dumps([diag], indent=2)

def evaluate_python(code_str: str, task: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Verifica ed esegue il codice Python/PyTorch nativo.
    Restituisce: (successo, python_codice, diagnostica_se_fallito)
    """
    try:
        compiled = compile(code_str, "<llm_generated>", "exec")
        scope = {"torch": torch, "F": F, "math": math}
        exec(compiled, scope)

        if task["fn_name"] not in scope:
            err_diag = f"Entrypoint function '{task['fn_name']}' not found in defined namespace."
            return False, code_str, err_diag

        fn = scope[task["fn_name"]]
        test_passed = task["test"](fn)
        if not test_passed:
            err_diag = "AssertionError: PyTorch numerical acceptance test failed."
            return False, code_str, err_diag

        return True, code_str, None

    except SyntaxError as se:
        err_diag = f"SyntaxError at line {se.lineno}, col {se.offset}: {se.msg}"
        return False, code_str, err_diag
    except Exception as e:
        err_diag = f"{type(e).__name__}: {str(e)}"
        return False, code_str, err_diag

# ---------------------------------------------------------------------------
# 5. HARNESS DI VALUTAZIONE CON PASS@K E CICLO DI AUTORIPARAZIONE
# ---------------------------------------------------------------------------

def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Calcola lo stimatore non distorto di pass@k (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

def run_evaluation(
    client: LLMClient,
    tasks: List[Dict[str, Any]],
    languages: List[str] = ["janus", "python"],
    num_samples: int = 5,
    max_repairs: int = 2
) -> Dict[str, Any]:
    """Esegue la valutazione per tutti i compiti e tutte le lingue specificate."""
    results_by_lang: Dict[str, Any] = {}

    for lang in languages:
        print(f"\n=======================================================")
        print(f"Valutazione Lingua: {lang.upper()} (Campioni per task: {num_samples})")
        print(f"=======================================================")

        sys_prompt = JANUS_SYSTEM_PROMPT if lang == "janus" else PYTHON_SYSTEM_PROMPT
        sys_tokens = count_tokens(sys_prompt)

        task_results = []
        total_tokens_consumed = 0
        total_attempts_recorded = []
        repairs_attempted = 0
        repairs_succeeded = 0

        for t_idx, task in enumerate(tasks, 1):
            print(f"[{t_idx:02d}/{len(tasks):02d}] {task['name']} ({task['id']})...", end="", flush=True)

            sample_outcomes = []
            c_correct = 0

            for s_idx in range(num_samples):
                prompt_text = (
                    f"Implement the following function in {lang.upper()}:\n"
                    f"Name: {task['fn_name']}\n"
                    f"Description: {task['description']}\n"
                    f"Signature expected: {task['janus_sig'] if lang == 'janus' else task['python_sig']}"
                )

                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt_text}
                ]

                sample_tokens = sys_tokens + count_tokens(prompt_text)
                success = False
                attempts_taken = 0
                error_diag = None
                code_history = []

                for attempt in range(max_repairs + 1):
                    attempts_taken += 1
                    raw_reply = client.generate(messages, task["id"], lang, s_idx, attempt)
                    sample_tokens += count_tokens(raw_reply)
                    extracted = extract_code(raw_reply, lang)
                    code_history.append(extracted)

                    if lang == "janus":
                        passed, py_code, error_diag = evaluate_janus(extracted, task)
                    else:
                        passed, py_code, error_diag = evaluate_python(extracted, task)

                    if passed:
                        success = True
                        if attempt > 0:
                            repairs_succeeded += 1
                        break
                    else:
                        if attempt < max_repairs:
                            if attempt == 0:
                                repairs_attempted += 1
                            # Prepara il feedback diagnostico per il turno successivo
                            if lang == "janus":
                                repair_msg = (
                                    f"The compilation/execution failed with the following machine-readable JSON diagnostics:\n"
                                    f"```json\n{error_diag}\n```\n"
                                    f"Please correct the JANUS code according to these diagnostics."
                                )
                            else:
                                repair_msg = (
                                    f"Execution failed with error:\n"
                                    f"{error_diag}\n"
                                    f"Please fix the Python code."
                                )
                            messages.append({"role": "assistant", "content": raw_reply})
                            messages.append({"role": "user", "content": repair_msg})
                            sample_tokens += count_tokens(repair_msg)

                if success:
                    c_correct += 1

                total_tokens_consumed += sample_tokens
                total_attempts_recorded.append(attempts_taken)

                sample_outcomes.append({
                    "sample_idx": s_idx,
                    "success": success,
                    "attempts": attempts_taken,
                    "tokens": sample_tokens,
                    "final_code": code_history[-1] if code_history else "",
                    "last_error": error_diag if not success else None
                })

            pass_1 = compute_pass_at_k(num_samples, c_correct, 1)
            pass_5 = compute_pass_at_k(num_samples, c_correct, 5) if num_samples >= 5 else None

            print(f" Corretti: {c_correct}/{num_samples} (pass@1: {pass_1*100:.1f}%)")

            task_results.append({
                "task_id": task["id"],
                "task_name": task["name"],
                "correct_samples": c_correct,
                "total_samples": num_samples,
                "pass@1": round(pass_1, 4),
                "pass@5": round(pass_5, 4) if pass_5 is not None else None,
                "samples": sample_outcomes
            })

        mean_pass_1 = sum(t["pass@1"] for t in task_results) / len(task_results)
        mean_pass_5 = (
            sum(t["pass@5"] for t in task_results if t["pass@5"] is not None) / len(task_results)
            if num_samples >= 5 else None
        )
        avg_attempts = sum(total_attempts_recorded) / len(total_attempts_recorded) if total_attempts_recorded else 1.0
        repair_rate = (repairs_succeeded / repairs_attempted) if repairs_attempted > 0 else 0.0

        results_by_lang[lang] = {
            "system_prompt_tokens": sys_tokens,
            "total_tokens_consumed": total_tokens_consumed,
            "mean_pass@1": round(mean_pass_1, 4),
            "mean_pass@5": round(mean_pass_5, 4) if mean_pass_5 is not None else None,
            "avg_attempts_to_solve": round(avg_attempts, 2),
            "repair_attempts": repairs_attempted,
            "repairs_succeeded": repairs_succeeded,
            "repair_success_rate": round(repair_rate, 4),
            "tasks": task_results
        }

    return results_by_lang

# ---------------------------------------------------------------------------
# 6. MAIN CLI & SALVATAGGIO ARTEFATTI
# ---------------------------------------------------------------------------

def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()[:7]
    except Exception:
        return "unknown"

def main():
    parser = argparse.ArgumentParser(description="Harness di Valutazione LLM (JANUS vs Python)")
    parser.add_argument("--dry-run", action="store_true", help="Esegue simulazione sintetica senza chiamate API reali")
    parser.add_argument("--provider", choices=["mock", "openai", "anthropic", "ollama"], default=None,
                        help="Provider LLM da usare (default: 'mock' se --dry-run o assenza di API key)")
    parser.add_argument("--model", type=str, default=None, help="Modello LLM (es. gpt-4o-mini, claude-3-5-sonnet-20241022)")
    parser.add_argument("--samples", "-n", type=int, default=5, help="Numero di campioni per task (default: 5)")
    parser.add_argument("--max-repairs", type=int, default=2, help="Numero massimo di tentativi di riparazione con diagnostica")
    parser.add_argument("--task", type=str, default=None, help="Esegui solo uno specifico task (es. task_01)")
    parser.add_argument("--output", type=str, default=None, help="File JSON di destinazione per i risultati grezzi")
    args = parser.parse_args()

    # Rilevamento automatico provider/modalità
    if args.dry_run or args.provider == "mock":
        provider = "mock"
        model = args.model or "simulated-mock-v1"
        is_dry_run = True
    else:
        if args.provider:
            provider = args.provider
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            print("[INFO] Nessuna API key rilevata (OPENAI_API_KEY / ANTHROPIC_API_KEY non presenti).")
            print("       Attivazione automatica modalita' --dry-run (risposte simulate).")
            provider = "mock"
            is_dry_run = True

        if provider == "openai":
            model = args.model or "gpt-4o-mini"
            is_dry_run = False
        elif provider == "anthropic":
            model = args.model or "claude-3-5-sonnet-20241022"
            is_dry_run = False
        elif provider == "ollama":
            model = args.model or "llama3"
            is_dry_run = False
        else:
            model = args.model or "simulated-mock-v1"
            is_dry_run = True

    tasks_to_run = TASKS
    if args.task:
        tasks_to_run = [t for t in TASKS if t["id"] == args.task]
        if not tasks_to_run:
            print(f"Errore: Task non trovato '{args.task}'.")
            sys.exit(1)

    print("=" * 60)
    print("JANUS vs Python - LLM Evaluation Benchmark")
    print(f"Commit: {get_git_commit()} | Data: {datetime.now(timezone.utc).isoformat()}")
    print(f"Provider: {provider} | Modello: {model} | Dry-run: {is_dry_run}")
    print(f"Task totali: {len(tasks_to_run)} | Campioni (n): {args.samples} | Max Riparazioni: {args.max_repairs}")
    if is_dry_run:
        print("[ATTENZIONE] Esecuzione in DRY-RUN: le risposte sono simulate e i dati risultanti")
        print("             NON DEVONO MAI essere citati come misure reali nel paper/report!")
    print("=" * 60)

    client = LLMClient(provider=provider, model=model, dry_run=is_dry_run)
    start_time = time.time()
    results = run_evaluation(
        client=client,
        tasks=tasks_to_run,
        languages=["janus", "python"],
        num_samples=args.samples,
        max_repairs=args.max_repairs
    )
    elapsed = time.time() - start_time

    # Output JSON grezzo
    os.makedirs("benchmarks/results", exist_ok=True)
    if args.output:
        out_path = args.output
    elif is_dry_run:
        out_path = "benchmarks/results/simulated_llm_eval.json"
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = f"benchmarks/results/llm_eval_{model}_{ts}.json"

    payload = {
        "metadata": {
            "commit": get_git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "is_dry_run": is_dry_run,
            "warning": "SIMULATED / DRY-RUN DATA: DO NOT REPORT AS REAL MEASUREMENTS" if is_dry_run else None,
            "samples_n": args.samples,
            "max_repairs": args.max_repairs,
            "num_tasks": len(tasks_to_run),
            "elapsed_seconds": round(elapsed, 2)
        },
        "summary": {
            "janus": {
                "mean_pass@1": results["janus"]["mean_pass@1"],
                "mean_pass@5": results["janus"]["mean_pass@5"],
                "avg_attempts": results["janus"]["avg_attempts_to_solve"],
                "repair_success_rate": results["janus"]["repair_success_rate"],
                "total_tokens": results["janus"]["total_tokens_consumed"]
            },
            "python": {
                "mean_pass@1": results["python"]["mean_pass@1"],
                "mean_pass@5": results["python"]["mean_pass@5"],
                "avg_attempts": results["python"]["avg_attempts_to_solve"],
                "repair_success_rate": results["python"]["repair_success_rate"],
                "total_tokens": results["python"]["total_tokens_consumed"]
            }
        },
        "results": results
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Valutazione completata in {elapsed:.2f}s!")
    print(f"Risultati grezzi salvati in: {out_path}")
    print("=" * 60)
    print("RIEPILOGO:")
    print(f"  JANUS  : pass@1={results['janus']['mean_pass@1']*100:.1f}%, pass@5={((results['janus']['mean_pass@5'] or 0)*100):.1f}%, avg_attempts={results['janus']['avg_attempts_to_solve']}, tokens={results['janus']['total_tokens_consumed']}")
    print(f"  Python : pass@1={results['python']['mean_pass@1']*100:.1f}%, pass@5={((results['python']['mean_pass@5'] or 0)*100):.1f}%, avg_attempts={results['python']['avg_attempts_to_solve']}, tokens={results['python']['total_tokens_consumed']}")
    if is_dry_run:
        print("[DRY-RUN] Risultati simulati generati a scopo di verifica architetturale dell'harness.")

if __name__ == "__main__":
    main()
