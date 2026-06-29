"""
Full Qwen3-4B pipeline + lambda_d recurrence comparison.
Loads layers one-at-a-time (2GB VRAM budget).
Replaces lm_head with lambda_d recurrence and compares generation.
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

torch.set_float32_matmul_precision('high')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─── Config ──────────────────────────────────────────────────────────────────

CFG = {
    'hidden_size': 2560,
    'num_hidden_layers': 36,
    'num_attention_heads': 32,
    'num_key_value_heads': 8,
    'head_dim': 128,
    'intermediate_size': 9728,
    'vocab_size': 146260,
    'rms_norm_eps': 1e-6,
}

PT_PATH = Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\qwen_layer_model.pt')
EMBED_PATH = Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\checkpoints\embed_tokens_only.pt')

# ─── Load state dict (mmap, CPU) ─────────────────────────────────────────────

print(f'Loading state dict from {PT_PATH}...')
t0 = time.perf_counter()
STATE = torch.load(PT_PATH, map_location='cpu', mmap=True, weights_only=False)
SD = STATE['model_state_dict']
print(f'  Loaded in {time.perf_counter()-t0:.1f}s, {len(SD)} keys')

# Pre-extract final norm and lm_head
FINAL_NORM_W = SD['base_model.model.norm.weight'].half()  # (2560,)
EMBED_W = SD['base_model.model.embed_tokens.weight'].half()  # (V, d)

# ─── RMS Norm ────────────────────────────────────────────────────────────────

def rms_norm(x, weight, eps=1e-6):
    """RMS Layer Normalization."""
    rms = x.half().norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    return (x.half() / (rms + 1e-6).half()) * weight.half()

# ─── RoPE ────────────────────────────────────────────────────────────────────

def precompute_rope(max_pos=4096, head_dim=128, theta=10000000.0):
    half_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_pos, half_dim)
    return freqs.cos().half(), freqs.sin().half()

FREQS_COS, FREQS_SIN = precompute_rope(max_pos=4096, head_dim=CFG['head_dim'])

def apply_rope(x, pos_ids):
    """Apply rotary position encoding.
    x: (B, n_heads, L, head_dim)
    pos_ids: (L,) position indices
    """
    B, nh, L, d = x.shape
    half = d // 2
    cos = FREQS_COS[pos_ids].to(x.device)  # (L, half)
    sin = FREQS_SIN[pos_ids].to(x.device)  # (L, half)
    cos = cos.view(1, 1, L, half)
    sin = sin.view(1, 1, L, half)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# ─── Attention forward ───────────────────────────────────────────────────────

def attention_forward(h, q_w, k_w, v_w, o_w, q_norm_w, k_norm_w, pos, layer_idx):
    """Multi-head attention with GQA (32 heads Q, 8 heads KV)."""
    B, L, D = h.shape
    n_heads = CFG['num_attention_heads']
    n_kv_heads = CFG['num_key_value_heads']
    head_dim = CFG['head_dim']
    n_groups = n_heads // n_kv_heads  # 4
    
    # Project to Q, K, V
    q = h @ q_w.T  # (B, L, n_heads * head_dim)
    k = h @ k_w.T
    v = h @ v_w.T
    
    # Reshape to (B, L, n_heads, head_dim)
    q = q.view(B, L, n_heads, head_dim)
    k = k.view(B, L, n_kv_heads, head_dim)
    v = v.view(B, L, n_kv_heads, head_dim)
    
    # Apply QK norm (per-head RMS norm)
    q_norm_w = q_norm_w.view(1, 1, 1, -1)
    k_norm_w = k_norm_w.view(1, 1, 1, -1)
    rms_q = q.half().norm(dim=-1, keepdim=True) / (head_dim ** 0.5) + 1e-6
    rms_k = k.half().norm(dim=-1, keepdim=True) / (head_dim ** 0.5) + 1e-6
    q = q.half() / rms_q.half() * q_norm_w.half()
    k = k.half() / rms_k.half() * k_norm_w.half()
    
    # Apply RoPE (transpose to (B, n_heads, L, head_dim))
    q = apply_rope(q.transpose(1, 2), pos)  # (B, n_heads, L, head_dim)
    k = apply_rope(k.transpose(1, 2), pos)  # (B, n_kv_heads, L, head_dim)
    v = v.transpose(1, 2)  # (B, n_kv_heads, L, head_dim)
    
    # GQA: expand KV heads to match Q heads
    if n_groups > 1:
        k = k[:, :, None].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, L, head_dim)
        v = v[:, :, None].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, L, head_dim)
    
    # Scaled dot-product attention
    scale = torch.tensor(head_dim, dtype=torch.float16, device=h.device) ** 0.5
    attn = (q @ k.transpose(-2, -1)) / scale  # (B, n_heads, L, L)
    
    # Causal mask
    mask = torch.triu(torch.full((L, L), -3e4, dtype=torch.float16, device=h.device), diagonal=1)
    attn = attn + mask
    
    attn_weights = F.softmax(attn.float(), dim=-1).half()
    out = attn_weights @ v  # (B, n_heads, L, head_dim)
    
    # Merge heads
    out = out.transpose(1, 2).reshape(B, L, n_heads * head_dim)
    out = out @ o_w.T  # (B, L, D)
    
    return out

# ─── MLP forward (SwiGLU) ────────────────────────────────────────────────────

def mlp_forward(h, gate_w, up_w, down_w):
    """SwiGLU MLP."""
    gate = F.silu((h @ gate_w.T).float()).half()
    up = (h @ up_w.T).half()
    return ((gate * up) @ down_w.T).half()

# ─── Single layer forward ────────────────────────────────────────────────────

def layer_forward(h, layer_weights, pos, layer_idx):
    """Run one transformer layer."""
    # Pre-attention norm
    h_norm = rms_norm(h, layer_weights['input_layernorm'], CFG['rms_norm_eps'])
    
    # Attention
    attn_out = attention_forward(
        h_norm.half(),
        layer_weights['q_proj'],
        layer_weights['k_proj'],
        layer_weights['v_proj'],
        layer_weights['o_proj'],
        layer_weights['q_norm'],
        layer_weights['k_norm'],
        pos, layer_idx
    )
    h = h.half() + attn_out.half()
    
    # Pre-MLP norm
    h_norm = rms_norm(h, layer_weights['post_attention_layernorm'], CFG['rms_norm_eps'])
    
    # MLP
    mlp_out = mlp_forward(h_norm,
                          layer_weights['gate_proj'],
                          layer_weights['up_proj'],
                          layer_weights['down_proj'])
    h = h + mlp_out
    
    return h

# ─── Extract layer weights from state dict ───────────────────────────────────

def get_layer_weights(layer_idx):
    """Extract weights for one layer from the state dict."""
    prefix = f'base_model.model.layers.{layer_idx}.self_attn.'
    mlp_prefix = f'base_model.model.layers.{layer_idx}.mlp.'
    norm_prefix = f'base_model.model.layers.{layer_idx}.'
    
    return {
        'q_proj': SD[prefix + 'q_proj.weight'].half(),
        'k_proj': SD[prefix + 'k_proj.weight'].half(),
        'v_proj': SD[prefix + 'v_proj.weight'].half(),
        'o_proj': SD[prefix + 'o_proj.weight'].half(),
        'q_norm': SD[prefix + 'q_norm.weight'].half(),
        'k_norm': SD[prefix + 'k_norm.weight'].half(),
        'gate_proj': SD[mlp_prefix + 'gate_proj.weight'].half(),
        'up_proj': SD[mlp_prefix + 'up_proj.weight'].half(),
        'down_proj': SD[mlp_prefix + 'down_proj.weight'].half(),
        'input_layernorm': SD[norm_prefix + 'input_layernorm.weight'].half(),
        'post_attention_layernorm': SD[norm_prefix + 'post_attention_layernorm.weight'].half(),
    }

# ─── Tokenizer ───────────────────────────────────────────────────────────────

def load_tokenizer():
    from tokenizers import Tokenizer
    # Try to find the tokenizer file
    candidates = [
        Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json'),
        Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\tokenizer.json'),
    ]
    for p in candidates:
        if p.exists():
            print(f'  Loading tokenizer from {p}')
            return Tokenizer.from_file(str(p))
    # Fallback: create minimal
    print('  WARNING: No tokenizer found, using dummy')
    return None

# ─── lambda_d functions ──────────────────────────────────────────────────────

phi = (1 + 5**0.5) / 2
sqrt5 = 5**0.5
LOG_PHI = math.log(phi)
LOG_SQRT5 = math.log(sqrt5)

def lambda_d_softmax(logits, temp=1.0):
    """Numerically stable lambda_d softmax in log-space.
    log(lambda_d(x)) = x * log(phi) - log(sqrt5)
    """
    log_ld = logits.float() / temp * LOG_PHI - LOG_SQRT5
    log_probs = log_ld - torch.logsumexp(log_ld, dim=-1, keepdim=True)
    return torch.exp(log_probs).half()

def build_companion_A(D, d_fib=2):
    n_chains = D // d_fib
    A = torch.zeros(D, D, dtype=torch.float16)
    for c in range(n_chains):
        start = c * d_fib
        end = start + d_fib
        A[start, start:end] = 1.0
        for k in range(d_fib - 1):
            A[start + k + 1, start + k] = 1.0
    return A

# ─── Full pipeline ───────────────────────────────────────────────────────────

def run_qwen_pipeline(text, max_new_tokens=10):
    """Run full Qwen pipeline + lambda_d generation comparison."""
    print(f'\n{"="*60}')
    print(f'Qwen3-4B + lambda_d Recurrence Pipeline')
    print(f'{"="*60}')
    print(f'Input: "{text}"')
    
    # Load tokenizer
    tok = load_tokenizer()
    if tok:
        tokens = tok.encode(text).ids
    else:
        # Dummy tokens
        tokens = [0, 1, 2, 3, 4, 5]
    
    print(f'  Prompt tokens ({len(tokens)}): {tokens[:10]}...')
    
    # Embed tokens
    pos = torch.arange(len(tokens), dtype=torch.long)
    h = EMBED_W[tokens].unsqueeze(0)  # (1, L, D)
    h = h.to(DEVICE, torch.float16)
    print(f'  Embed shape: {h.shape}')
    
    # Run through all 36 layers
    print(f'\n--- Running {CFG["num_hidden_layers"]} layers ---')
    layer_times = []
    for layer_idx in range(CFG['num_hidden_layers']):
        t0 = time.perf_counter()
        
        # Load layer weights to GPU
        w = get_layer_weights(layer_idx)
        w_gpu = {k: v.to(DEVICE, torch.float16) for k, v in w.items()}
        
        # Forward
        h = layer_forward(h, w_gpu, pos, layer_idx)
        
        # Free GPU memory
        del w_gpu
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        
        t = time.perf_counter() - t0
        layer_times.append(t)
        
        if layer_idx < 3 or layer_idx == CFG['num_hidden_layers'] - 1:
            print(f'  Layer {layer_idx:2d}: {t*1000:.0f}ms, h norm={h.norm().item():.2f}')
    
    avg_layer = np.mean(layer_times)
    print(f'  Avg layer: {avg_layer*1000:.0f}ms, total: {sum(layer_times):.1f}s')
    
    # Final norm
    h = rms_norm(h, FINAL_NORM_W.to(DEVICE, torch.float16), CFG['rms_norm_eps'])
    print(f'  Final h norm: {h.norm().item():.2f}')
    
    # Get last hidden state
    h_last = h[:, -1:, :]  # (1, 1, D) - last token
    
    # ─── Compare: lm_head vs lambda_d recurrence ─────────────────────────
    print(f'\n--- Next-token prediction comparison ---')
    
    # 1. Standard lm_head
    lm_w = EMBED_W.to(DEVICE, torch.float16)
    logits_lm = h_last.squeeze(0) @ lm_w.T  # (1, V)
    probs_lm = F.softmax(logits_lm / 1.0, dim=-1)
    top5_lm = probs_lm.topk(5).indices.tolist()
    
    # Decode top tokens
    print(f'  Qwen lm_head top-5:')
    for rank, tid in enumerate(top5_lm[0]):
        txt = ''
        if tok:
            txt = f' "{tok.decode([tid])}"'
        print(f'    {rank+1}. token {tid}{txt} (prob={probs_lm[0, tid].item():.4f})')
    
    # 2. lambda_d recurrence
    A = build_companion_A(CFG['hidden_size']).to(DEVICE, torch.float16)
    
    # Run recurrence from last hidden state
    h_rec = h_last.clone()  # (1, 1, D)
    print(f'\n  lambda_d recurrence generation:')
    
    gen_tokens = []
    for step in range(max_new_tokens):
        # Recurrence: v_{n+1} = A_d * v_n
        h_rec = h_rec @ A.T
        
        # lambda_d readout
        logits_rec = h_rec.squeeze(0) @ lm_w.T  # (1, V)
        probs_rec = lambda_d_softmax(logits_rec)
        probs_rec = probs_rec.clamp(min=1e-10)
        probs_rec = probs_rec / probs_rec.sum()
        
        # Sample
        next_token = torch.multinomial(probs_rec.squeeze(0), 1).item()
        gen_tokens.append(next_token)
        
        # Add token embedding for next step
        h_rec = h_rec + lm_w[next_token].unsqueeze(0).unsqueeze(0)
        h_rec = F.normalize(h_rec, dim=-1) * (CFG['hidden_size'] ** 0.5)
        
        # Decode (safe encode to avoid cp1251 issues)
        txt = ''
        dec = tok.decode([next_token]) if tok else ''
        txt = f' = "{dec.encode("ascii", "replace").decode()}"'
        top5_r = probs_rec.topk(5).indices.tolist()
        print(f'    Step {step}: token {next_token}{txt}')
        if step < 2:
            top5_texts = []
            for t in top5_r[0]:
                d = tok.decode([t]) if tok else str(t)
                d = d.encode('ascii', 'replace').decode()
                top5_texts.append(f'"{d}"')
            print(f'             top-5: {", ".join(top5_texts)}')
    
    # 3. Compare: lm_head vs recurrence at step 0
    print(f'\n  Step 0 comparison:')
    overlap = sum(1 for t in top5_lm[0] if t in probs_rec.topk(5).indices.tolist()[0])
    print(f'    Top-5 overlap: {overlap}/5')
    
    # KL divergence
    eps = 1e-30
    p_lm = probs_lm[0].clamp(min=eps)
    p_rec = probs_rec[0].clamp(min=eps)
    p_lm = p_lm / p_lm.sum()
    p_rec = p_rec / p_rec.sum()
    kl = (p_lm * (p_lm.log() - p_rec.log())).sum().item()
    print(f'    KL(exp || lambda_d): {kl:.4f}')
    
    return {
        'tokens': tokens,
        'gen_tokens': gen_tokens,
        'top5_lm': top5_lm,
        'h_last': h_last,
    }


if __name__ == '__main__':
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    result = run_qwen_pipeline("The future of AI is", max_new_tokens=5)
    
    print(f'\n{"="*60}')
    print(f'lambda_d Recurrence Summary')
    print(f'{"="*60}')
    print(f'  lm_head replaced by: v @ W.T  (same math, different activation)')
    print(f'  KV cache replaced by: A_d * v + W[t] (5KB per step)')
    print(f'  Softmax replaced by: lambda_d(x) = phi^x / sqrt(5)')
    print(f'  Position encoding replaced by: A_d^n (implicit in recurrence)')
    print(f'  Result: one recurrence replaces 3 components')
