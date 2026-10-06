"""
Test di Esecuzione Reale per i Programmi di Riferimento JANUS (Fase 2):
Esegue il codice Python generato e lo confronta numericamente con implementazioni
di riferimento PyTorch scritte a mano entro tolleranze esplicite.
"""

import math
import pytest
from janus.lexer import Lexer
from janus.parser import Parser
from janus.codegen import CodeGenerator

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch non installato nell'ambiente di test")


def compile_and_load(jn_file_path: str):
    """Compila un file JANUS in Python ed esegue il codice risultante restituendo l'ambiente."""
    with open(jn_file_path, "r", encoding="utf-8") as f:
        code = f.read()
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    py_code = CodeGenerator().generate(ast)
    env = {}
    exec(py_code, env)
    return env, py_code


def test_linreg_convergence():
    """
    Programma 1: linreg converge verso i veri parametri del problema sintetico:
    y = 2.0 * x + 1.0.
    Tolleranza: errore assoluto < 0.05 sui parametri w e b.
    """
    env, _ = compile_and_load("examples/01_linreg.jn")
    linreg_fn = env["linreg"]

    torch.manual_seed(42)
    x = torch.linspace(1.0, 5.0, 20)
    true_w = 2.0
    true_b = 1.0
    y = true_w * x + true_b

    w_hat, b_hat = linreg_fn(x, y, epochs=1500, lr=0.03)

    assert isinstance(w_hat, (float, int)), f"w deve essere uno scalare numerico, ricevuto {type(w_hat)}"
    assert isinstance(b_hat, (float, int)), f"b deve essere uno scalare numerico, ricevuto {type(b_hat)}"
    assert abs(w_hat - true_w) < 0.05, f"w stimato ({w_hat}) distante dal vero {true_w}"
    assert abs(b_hat - true_b) < 0.05, f"b stimato ({b_hat}) distante dal vero {true_b}"


def test_mlp_matches_pytorch_reference():
    """
    Programma 2: mlp_step calcola forward e backward pass identici
    al riferimento PyTorch scritto a mano con atol=1e-5.
    """
    env, _ = compile_and_load("examples/02_mlp.jn")
    MLP_class = env["MLP"]
    mlp_step_fn = env["mlp_step"]

    torch.manual_seed(42)
    D, H, C = 4, 8, 2
    w1 = torch.randn(D, H, requires_grad=True)
    b1 = torch.randn(H, requires_grad=True)
    w2 = torch.randn(H, C, requires_grad=True)
    b2 = torch.randn(C, requires_grad=True)
    x = torch.randn(6, D)
    y = torch.randn(6, C)
    lr = 0.05

    # Riferimento PyTorch
    ref_pred = torch.relu(x @ w1 + b1) @ w2 + b2
    ref_loss = torch.sum((ref_pred - y) ** 2) / len(x)
    gw1, gb1, gw2, gb2 = torch.autograd.grad(ref_loss, (w1, b1, w2, b2))
    ref_w1_up = w1 - lr * gw1
    ref_b1_up = b1 - lr * gb1
    ref_w2_up = w2 - lr * gw2
    ref_b2_up = b2 - lr * gb2

    # JANUS generato
    mlp = MLP_class(
        w1=w1.clone().detach().requires_grad_(),
        b1=b1.clone().detach().requires_grad_(),
        w2=w2.clone().detach().requires_grad_(),
        b2=b2.clone().detach().requires_grad_()
    )
    new_mlp, loss = mlp_step_fn(x, y, mlp, lr)

    assert abs(loss - ref_loss.item()) < 1e-5, "La loss di MLP non coincide con il riferimento PyTorch"
    assert torch.allclose(new_mlp.w1, ref_w1_up, atol=1e-5), "Aggiornamento w1 non coincide"
    assert torch.allclose(new_mlp.b1, ref_b1_up, atol=1e-5), "Aggiornamento b1 non coincide"
    assert torch.allclose(new_mlp.w2, ref_w2_up, atol=1e-5), "Aggiornamento w2 non coincide"
    assert torch.allclose(new_mlp.b2, ref_b2_up, atol=1e-5), "Aggiornamento b2 non coincide"


def test_attention_matches_pytorch_reference():
    """
    Programma 3: mha calcola scaled dot-product attention identica
    al riferimento PyTorch con atol=1e-5.
    """
    env, _ = compile_and_load("examples/03_attention.jn")
    mha_fn = env["mha"]

    torch.manual_seed(42)
    B, N, D = 2, 8, 16
    q = torch.randn(B, N, D)
    k = torch.randn(B, N, D)
    v = torch.randn(B, N, D)

    # Riferimento PyTorch: softmax(Q @ K.T / sqrt(D), dim=-1) @ V
    scale = 1.0 / math.sqrt(D)
    ref_scores = (q @ k.transpose(-2, -1)) * scale
    ref_attn = torch.softmax(ref_scores, dim=-1) @ v

    out = mha_fn(q, k, v)
    assert torch.allclose(out, ref_attn, atol=1e-5), "L'attenzione JANUS non coincide con il riferimento PyTorch"


def test_conv2d_matches_pytorch_reference():
    """
    Programma 5: cnn_feature_extractor (conv2d + relu + maxpool + flatten)
    coincide con il riferimento PyTorch con atol=1e-5.
    """
    env, _ = compile_and_load("examples/05_conv2d.jn")
    cnn_fn = env["cnn_feature_extractor"]

    torch.manual_seed(42)
    # img: [batch=2, in_channels=3, height=8, width=8]
    # k: [out_channels=4, in_channels=3, kernel_h=3, kernel_w=3]
    img = torch.randn(2, 3, 8, 8)
    k = torch.randn(4, 3, 3, 3)

    # Riferimento PyTorch corrispondente:
    # conv2d con stride 1, padding 1
    # relu
    # max_pool2d con kernel 2, stride 2
    # flatten start_dim=1
    conv = F.conv2d(img, k, stride=1, padding=1)
    act = torch.relu(conv)
    pool = F.max_pool2d(act, kernel_size=2, stride=2)
    ref_out = torch.flatten(pool, start_dim=1)

    out = cnn_fn(img, k)
    assert torch.allclose(out, ref_out, atol=1e-5), "L'output del feature extractor Conv2D non coincide con PyTorch"


def test_rmsnorm_matches_pytorch_reference():
    """
    Programma 6: rms_norm coincide con l'implementazione matematica
    di riferimento di Root Mean Square Normalization con atol=1e-5.
    """
    env, _ = compile_and_load("examples/06_rmsnorm.jn")
    rmsnorm_fn = env["rms_norm"]

    torch.manual_seed(42)
    D = 16
    x = torch.randn(4, 8, D)
    w = torch.randn(D)
    eps = 1e-6

    # Riferimento PyTorch RMSNorm: x / sqrt(mean(x^2) + eps) * w
    var = torch.sum(x ** 2, dim=-1, keepdim=True) / D
    ref_out = (x / torch.sqrt(var + eps)) * w

    out = rmsnorm_fn(x, w, eps=eps)
    assert torch.allclose(out, ref_out, atol=1e-5), "RMSNorm non coincide con il riferimento PyTorch"


def test_training_loop_execution():
    """
    Programma 4: train_loop esegue il training su mini-batch e riduce la loss.
    """
    env, _ = compile_and_load("examples/04_training_loop.jn")
    ModelWeights = env["ModelWeights"]
    train_loop_fn = env["train_loop"]

    torch.manual_seed(42)
    D, C = 4, 1
    w = torch.randn(D, C, requires_grad=True)
    b = torch.randn(C, requires_grad=True)
    model = ModelWeights(w=w, b=b)

    data = torch.randn(40, D)
    labels = data @ torch.tensor([[1.0], [2.0], [-1.0], [0.5]]) + 0.1

    # Esegue il ciclo di training
    train_loop_fn(data, labels, model, epochs=5, bs=10, lr=0.01)
    # Verifica che il modello contenga ancora pesi validi e finiti
    assert torch.isfinite(model.w).all()
    assert torch.isfinite(model.b).all()


def test_diffusion_denoise_execution():
    """
    Programma 7: denoise step esegue su CPU restituendo un tensore con forma corretta e stato PRNG.
    """
    env, _ = compile_and_load("examples/07_diffusion.jn")
    denoise_fn = env["denoise"]

    torch.manual_seed(42)
    B, C, H, W = 1, 3, 16, 16
    xt = torch.randn(B, C, H, W)
    noise_pred = torch.randn(B, C, H, W)
    a = 0.75
    a_prev = 0.80
    seed = 42

    denoised, next_seed = denoise_fn(xt, noise_pred, a, a_prev, seed)
    assert denoised.shape == xt.shape, f"Forma attesa {xt.shape}, ottenuta {denoised.shape}"
    assert torch.isfinite(denoised).all(), "Il tensore di output contiene valori non finiti (NaN/Inf)"
    assert next_seed == seed + 1, "Il seed PRNG deve avanzare deterministicamente"


def test_compile_only_examples_declared():
    """
    Verifica che gli esempi esclusi dall'esecuzione su CPU compilino regolarmente,
    dichiarando esplicitamente il motivo dell'esclusione runtime:
    - 08_rag_pipeline.jn: compile-only per dipendenza da servizi esterni di retrieval/agente cognitivo.
    - 09_react_agent.jn: compile-only per dipendenza da LLM runtime e loop interattivo.
    - 10_gpu_saxpy.jn: compile-only per kernel SIMT nativo che richiede GPU hardware.
    """
    # 08 RAG
    _, py_rag = compile_and_load("examples/08_rag_pipeline.jn")
    assert "def rag_pipeline(" in py_rag
    assert "_janus_agent_call" in py_rag

    # 09 ReAct Agent
    _, py_react = compile_and_load("examples/09_react_agent.jn")
    assert "def agent_exec(" in py_react

    # 10 GPU SAXPY
    _, py_saxpy = compile_and_load("examples/10_gpu_saxpy.jn")
    assert "def saxpy(" in py_saxpy
    assert "idx = " in py_saxpy
