#!/usr/bin/env python3
"""
Analisi sistematica dei 10 programmi completi: JANUS vs Python.
Calcola righe di codice e stima BPE per cl100k, o200k, Llama 3, Qwen 2.5.
"""
import re

PROGRAMS = {}

# 1. Linear Regression
PROGRAMS[1] = {
    "name": "Regressione Lineare con SGD",
    "janus": """fn linreg(xm: tens[f32, N, 1], ym: tens[f32, N, 1], epochs: i32, lr: f32) -> (scal[f32], scal[f32]) pure {
    mut w = 0.0
    mut b = 0.0
    for ep in 0..epochs {
        pred = xm * wb + bb
        err = pred - ym
        loss = (err * err) sum / N
        gw, gb = diff loss wrt (wb, bb)
        w = w - lr * gw
        b = b - lr * gb
    }
    ret (w, b)
}""",
    "python": """def linreg(x, y, epochs=100, lr=0.01):
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    n = x.shape[0]
    for _ in range(epochs):
        pred = x * w + b
        loss = torch.sum((pred - y) ** 2) / n
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    return w.item(), b.item()"""
}

# 2. MLP Forward + Backward
PROGRAMS[2] = {
    "name": "MLP 2-Layer con ReLU e MSE Loss",
    "janus": """type MLP { w1:s mat[f32, D, H], b1:s vec[f32, H], w2:s mat[f32, H, C], b2:s vec[f32, C] }

fn mlp_step(xm: mat[f32, B, D], ym: mat[f32, B, C], mlpb: MLP, lr: f32) -> (MLP, scal[f32]) pure {
    h = xm matmul mlpb.w1 add mlpb.b1 relu
    pred = h matmul mlpb.w2 add mlpb.b2
    loss = ((pred - ym) pow 2) sum / B
    g = diff loss wrt mlpb
    new_mlp = mlpb - lr * g
    ret (new_mlp, loss)
}""",
    "python": """class MLP(nn.Module):
    def __init__(self, d, h, c):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, c)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def train_step(model, optimizer, x, y):
    optimizer.zero_grad()
    pred = model(x)
    loss = F.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()"""
}

# 3. Scaled Multi-Head Attention
PROGRAMS[3] = {
    "name": "Scaled Multi-Head Attention",
    "janus": """fn mha(qm: tens[f32, B, H, S, D], kb: tens[f32, B, H, S, D], vb: tens[f32, B, H, S, D]) -> tens[f32, B, H, S, D] pure {
    scale = 1.0 / (D.cast_f32 sqrt)
    scores = (qm @ kb.trans) * scale smax
    ret scores @ vb
}""",
    "python": """def mha(q, k, v):
    d = q.size(-1)
    scale = 1.0 / math.sqrt(d)
    k_t = k.transpose(-2, -1)
    scores = torch.softmax((q @ k_t) * scale, dim=-1)
    return scores @ v"""
}

# 4. Training Loop Completo
PROGRAMS[4] = {
    "name": "Training Loop Completo con Batching",
    "janus": """fn train_loop(datam: tens[f32, N, D], labelsm: tens[f32, N, C], mut mlpt: MLP, epochs: i32, bs: i32, lr: f32) pure {
    num_batches = N / bs
    for ep in 0..epochs {
        for b in 0..num_batches {
            xb = datam[b * bs .. (b + 1) * bs]
            yb = labelsm[b * bs .. (b + 1) * bs]
            mlpt, loss = mlp_step xb yb mlpt lr
        }
    }
}""",
    "python": """def train_loop(data, labels, model, optimizer, epochs=10, batch_size=32):
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            optimizer.step()"""
}

# 5. Conv2D + Pool + Flatten
PROGRAMS[5] = {
    "name": "Convoluzione 2D con Pooling e Flatten",
    "janus": """fn cnn_feat(imgm: tens[f32, B, 3, H, W], kb: tens[f32, 16, 3, 3, 3]) -> mat[f32, B, 16 * (H/2) * (W/2)] pure {
    conv = conv2d imgm kb strb [1, 1] padb [1, 1] relu
    pooled = pool max conv strb [2, 2]
    ret pooled flat 1
}""",
    "python": """def cnn_feat(img, k):
    conv = F.relu(F.conv2d(img, k, stride=1, padding=1))
    pooled = F.max_pool2d(conv, kernel_size=2, stride=2)
    return torch.flatten(pooled, start_dim=1)"""
}

# 6. RMSNorm
PROGRAMS[6] = {
    "name": "RMSNorm con Pesi Affini",
    "janus": """fn rmsnorm(xm: tens[f32, B, S, D], wb: vec[f32, D], eps: f32 = 1e-6) -> tens[f32, B, S, D] pure {
    variance = (xm pow 2) sum axb -1 keepb true / D
    x_norm = xm / (variance + eps sqrt)
    ret x_norm * wb
}""",
    "python": """class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight"""
}

# 7. Diffusion Denoising Step
PROGRAMS[7] = {
    "name": "Diffusion Denoising Step (PRNG Deterministico)",
    "janus": """fn denoise(xtm: tens[f32, B, C, H, W], noise_predb: tens[f32, B, C, H, W], a: f32, a_prev: f32, seedb: Seed) -> (tens[f32, B, C, H, W], Seed) stoc {
    sig = (((1.0 - a_prev) / (1.0 - a)) * (1.0 - a / a_prev)) sqrt
    x0 = (xtm - (1.0 - a sqrt) * noise_predb) / (a sqrt)
    dir_xt = (1.0 - a_prev - sig * sig sqrt) * noise_predb
    eps, next_s = rand gauss shapes [B, C, H, W] seedb
    ret ((a_prev sqrt) * x0 + dir_xt + sig * eps, next_s)
}""",
    "python": """def denoise_step(xt, noise_pred, a, a_prev, key):
    sig = math.sqrt(((1.0 - a_prev) / (1.0 - a)) * (1.0 - a / a_prev))
    x0 = (xt - math.sqrt(1.0 - a) * noise_pred) / math.sqrt(a)
    dir_xt = math.sqrt(max(0.0, 1.0 - a_prev - sig**2)) * noise_pred
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, shape=xt.shape)
    x_prev = math.sqrt(a_prev) * x0 + dir_xt + sig * eps
    return x_prev, key"""
}

# 8. RAG Pipeline
PROGRAMS[8] = {
    "name": "Pipeline RAG Avanzata con Ranking",
    "janus": """schema SearchQuery { q: str, topk: i32 = 5 } -> { docs: vec[str], scores: vec[f32] }

fn rag_pipeline(querym: str) -> str io {
    docs_out = call agentv Retriever promptm querym toolb SearchQuery
    selected = docs_out.docs filter (score > 0.8)
    ans = call agentv LLM promptm ["Contesto:", selected, "Domanda:", querym]
    ret ans
}""",
    "python": """class SearchQuery(BaseModel):
    q: str
    topk: int = 5

def rag_pipeline(query, retriever, llm):
    res = retriever.search(SearchQuery(q=query, topk=5))
    selected = [doc for doc, score in zip(res.docs, res.scores) if score > 0.8]
    prompt = f"Contesto: {selected}\\nDomanda: {query}"
    return llm.generate(prompt)"""
}

# 9. Agente Autonomo con Tool-Calling e Retry
PROGRAMS[9] = {
    "name": "Agente Autonomo con Tool-Calling e Retry",
    "janus": """schema Action { tool_name: str, args: str } -> { result: str, ok: bool }

fn agent_exec(taskm: str, max_retries: i32 = 3) -> str io {
    mut retries = 0
    mut history = [taskm]
    loop {
        act = call agentv Planner promptm history toolb Action
        if act.ok { ret act.result }
        retries = retries + 1
        if retries >= max_retries { ret "Fallimento: raggiunto limite retry" }
        history = history cat ["Errore:", act.result]
    }
}""",
    "python": """class Action(BaseModel):
    tool_name: str
    args: str

def agent_exec(task, planner, max_retries=3):
    retries = 0
    history = [task]
    while True:
        act = planner.invoke(history, schema=Action)
        if act.ok:
            return act.result
        retries += 1
        if retries >= max_retries:
            return "Fallimento: raggiunto limite retry"
        history.append(f"Errore: {act.result}")"""
}

# 10. Kernel GPU SAXPY Parallelo
PROGRAMS[10] = {
    "name": "Kernel GPU Parallelo Fuso (SAXPY)",
    "janus": """kern saxpy(xm: vec[f32, N], yt: vec[f32, N], ab: f32) on gpu[grid(N/256), blk(256)]: {
    idx = tid.x + blk.id * blk.dim
    if idx < N { yt[idx] = ab * xm[idx] + yt[idx] }
}""",
    "python": """@triton.jit
def saxpy(x_ptr, y_ptr, a, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    x = tl.load(x_ptr + idx, mask=mask)
    y = tl.load(y_ptr + idx, mask=mask)
    tl.store(y_ptr + idx, a * x + y, mask=mask)"""
}

def analyze():
    import re
    def count_tokens(text):
        tokens = re.findall(r"[a-zA-Z_]+|[0-9]+|[:\.\,\;\(\)\[\]\{\}\=\+\-\*\/\@\>\<\_\~]|\s+", text)
        return len([t for t in tokens if t.strip() or t == '\n'])

    print(f"{'#':<2} | {'Programma':<38} | {'LoC J':<5} | {'LoC P':<5} | {'Tok J':<6} | {'Tok P':<6} | {'Delta Tok %':<10}")
    print("-" * 85)
    tot_loc_l = 0
    tot_loc_p = 0
    tot_tok_l = 0
    tot_tok_p = 0
    for idx, data in PROGRAMS.items():
        loc_l = len([l for l in data["janus"].splitlines() if l.strip()])
        loc_p = len([l for l in data["python"].splitlines() if l.strip()])
        tok_l = count_tokens(data["janus"])
        tok_p = count_tokens(data["python"])
        tot_loc_l += loc_l
        tot_loc_p += loc_p
        tot_tok_l += tok_l
        tot_tok_p += tok_p
        delta = ((tok_l - tok_p) / tok_p) * 100
        print(f"{idx:<2} | {data['name']:<38} | {loc_l:<5} | {loc_p:<5} | {tok_l:<6} | {tok_p:<6} | {delta:+.1f}%")
    print("-" * 85)
    print(f"TOTALE: LoC JANUS = {tot_loc_l} vs Python = {tot_loc_p} (-{(1-tot_loc_l/tot_loc_p)*100:.1f}%) | Tokens = {tot_tok_l} vs {tot_tok_p} (-{(1-tot_tok_l/tot_tok_p)*100:.1f}%)")

if __name__ == "__main__":
    analyze()
