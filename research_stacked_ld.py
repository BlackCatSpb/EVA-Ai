"""
Stacked lambda_d distillation: train A_d to match Qwen's per-layer hidden states.
Replaces 36 attention layers with learnable A_d * v + MLP(lambda_d(v)).
"""

import os, sys, time, math, json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8'
torch.set_float32_matmul_precision('high')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─── Config ──────────────────────────────────────────────────────────────────

D = 2560
N_LAYERS = 36
VOCAB = 146260
INTERMEDIATE = 9728
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128

PT_PATH = Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\qwen_layer_model.pt')

print('Loading Qwen state dict...')
t0 = time.perf_counter()
STATE = torch.load(PT_PATH, map_location='cpu', mmap=True, weights_only=False)
SD = STATE['model_state_dict']
print(f'  Loaded {len(SD)} keys in {time.perf_counter()-t0:.1f}s')

EMBED_W = SD['base_model.model.embed_tokens.weight'].half()
FINAL_NORM_W = SD['base_model.model.norm.weight'].half()

# ─── Utilities (copied from pipeline) ────────────────────────────────────────

def precompute_rope(max_pos=4096, theta=10000000.0):
    half_dim = HEAD_DIM // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().half(), freqs.sin().half()

FREQS_COS, FREQS_SIN = precompute_rope()

def apply_rope(x, pos_ids):
    B, nh, L, d = x.shape
    half = d // 2
    cos = FREQS_COS[pos_ids].to(x.device).view(1, 1, L, half)
    sin = FREQS_SIN[pos_ids].to(x.device).view(1, 1, L, half)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

def rms_norm(x, weight, eps=1e-6):
    x = x.half()
    rms = x.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return x / (rms + eps).half() * weight.half()

def attention_forward(h, w, pos):
    B, L, _ = h.shape
    q = (h @ w['q_proj'].T).view(B, L, N_HEADS, HEAD_DIM)
    k = (h @ w['k_proj'].T).view(B, L, N_KV_HEADS, HEAD_DIM)
    v = (h @ w['v_proj'].T).view(B, L, N_KV_HEADS, HEAD_DIM)
    q_norm = w['q_norm'].view(1, 1, 1, -1)
    k_norm = w['k_norm'].view(1, 1, 1, -1)
    rms_q = q.half().norm(dim=-1, keepdim=True) / (HEAD_DIM ** 0.5) + 1e-6
    rms_k = k.half().norm(dim=-1, keepdim=True) / (HEAD_DIM ** 0.5) + 1e-6
    q = q.half() / rms_q.half() * q_norm.half()
    k = k.half() / rms_k.half() * k_norm.half()
    q = apply_rope(q.transpose(1, 2), pos)
    k = apply_rope(k.transpose(1, 2), pos)
    v = v.transpose(1, 2)
    n_groups = N_HEADS // N_KV_HEADS
    if n_groups > 1:
        k = k[:, :, None].expand(-1, -1, n_groups, -1, -1).reshape(B, N_HEADS, L, HEAD_DIM)
        v = v[:, :, None].expand(-1, -1, n_groups, -1, -1).reshape(B, N_HEADS, L, HEAD_DIM)
    scale = torch.tensor(HEAD_DIM, dtype=torch.float16, device=h.device) ** 0.5
    attn = (q @ k.transpose(-2, -1)) / scale
    mask = torch.triu(torch.full((L, L), -3e4, dtype=torch.float16, device=h.device), diagonal=1)
    attn = attn + mask
    attn_w = F.softmax(attn.float(), dim=-1).half()
    out = (attn_w @ v).transpose(1, 2).reshape(B, L, N_HEADS * HEAD_DIM) @ w['o_proj'].T
    out = out.half()
    return out

def mlp_forward(h, w):
    gate = F.silu((h @ w['gate_proj'].T).float()).half()
    up = (h @ w['up_proj'].T).half()
    return ((gate * up) @ w['down_proj'].T).half()

def get_layer_weights(idx):
    p = f'base_model.model.layers.{idx}.self_attn.'
    m = f'base_model.model.layers.{idx}.mlp.'
    n = f'base_model.model.layers.{idx}.'
    return {
        'q_proj': SD[p+'q_proj.weight'].half(),
        'k_proj': SD[p+'k_proj.weight'].half(),
        'v_proj': SD[p+'v_proj.weight'].half(),
        'o_proj': SD[p+'o_proj.weight'].half(),
        'q_norm': SD[p+'q_norm.weight'].half(),
        'k_norm': SD[p+'k_norm.weight'].half(),
        'gate_proj': SD[m+'gate_proj.weight'].half(),
        'up_proj': SD[m+'up_proj.weight'].half(),
        'down_proj': SD[m+'down_proj.weight'].half(),
        'input_layernorm': SD[n+'input_layernorm.weight'].half(),
        'post_attention_layernorm': SD[n+'post_attention_layernorm.weight'].half(),
    }

# ─── Data Generation: run Qwen, extract per-layer states ────────────────────

def generate_training_data(prompts):
    """Run Qwen on prompts, save hidden state from each layer."""
    from tokenizers import Tokenizer
    tok_path = Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json')
    tok = Tokenizer.from_file(str(tok_path))
    
    all_states = []  # list of [(L, D)] per prompt
    
    for pi, prompt in enumerate(prompts):
        tokens = tok.encode(prompt).ids[:20]
        if len(tokens) < 4:
            continue
        L = len(tokens)
        pos = torch.arange(L, dtype=torch.long)
        h = EMBED_W[tokens].unsqueeze(0).to(DEVICE, torch.float16)
        
        prompt_states = [h.squeeze(0).cpu()]  # layer 0 = embeddings
        
        print(f'Prompt {pi}: "{prompt[:30]}..." L={L}', end=' ')
        t0 = time.perf_counter()
        
        for layer_idx in range(N_LAYERS):
            w = get_layer_weights(layer_idx)
            w_gpu = {k: v.to(DEVICE, torch.float16) for k, v in w.items()}
            
            # Forward
            h_norm = rms_norm(h, w_gpu['input_layernorm'])
            attn_out = attention_forward(h_norm, w_gpu, pos)
            h = h.half() + attn_out.half()
            
            h_norm = rms_norm(h, w_gpu['post_attention_layernorm'])
            mlp_out = mlp_forward(h_norm, w_gpu)
            h = h + mlp_out.half()
            
            prompt_states.append(h.squeeze(0).cpu())
            del w_gpu
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        
        all_states.append(prompt_states)
        print(f'({time.perf_counter()-t0:.1f}s)')
    
    return all_states, tok

# ─── Stacked lambda_d Model ─────────────────────────────────────────────────

LOG_PHI = math.log((1 + 5**0.5) / 2)
LOG_SQRT5 = math.log(5**0.5)

def lambda_d_softmax(logits):
    log_ld = logits.float() * LOG_PHI - LOG_SQRT5
    log_probs = log_ld - torch.logsumexp(log_ld, dim=-1, keepdim=True)
    return torch.exp(log_probs).half()

class LDBlock(torch.nn.Module):
    """Single lambda_d block: A_d * v + MLP(lambda_d(v))."""
    def __init__(self, d=D, intermediate=INTERMEDIATE):
        super().__init__()
        # Learnable A_d matrix
        self.A = torch.nn.Parameter(torch.eye(d, dtype=torch.float16))
        # MLP weights (initialized from Qwen)
        self.gate_proj = torch.nn.Parameter(torch.empty(intermediate, d, dtype=torch.float16))
        self.up_proj = torch.nn.Parameter(torch.empty(intermediate, d, dtype=torch.float16))
        self.down_proj = torch.nn.Parameter(torch.empty(d, intermediate, dtype=torch.float16))
        self.input_norm = torch.nn.Parameter(torch.ones(d, dtype=torch.float16))
        self.post_norm = torch.nn.Parameter(torch.ones(d, dtype=torch.float16))
    
    def forward(self, h):
        # Pre-norm
        h_norm = rms_norm(h, self.input_norm)
        # A_d * v (replaces attention)
        h_rec = h_norm.half() @ self.A.T
        # MLP (SwiGLU, same as Qwen)
        gate = F.silu((h_rec @ self.gate_proj.T).float()).half()
        up = (h_rec @ self.up_proj.T).half()
        mlp_out = ((gate * up) @ self.down_proj.T).half()
        # Residual
        h = h.half() + mlp_out
        h = rms_norm(h, self.post_norm)
        return h

class StackedLD(torch.nn.Module):
    """Stacked lambda_d blocks replacing all attention layers."""
    def __init__(self, num_layers=N_LAYERS):
        super().__init__()
        self.blocks = torch.nn.ModuleList([LDBlock() for _ in range(num_layers)])
    
    def init_from_qwen(self, layer_indices):
        """Initialize MLP weights from Qwen layers."""
        for i, lidx in enumerate(layer_indices):
            w = get_layer_weights(lidx)
            self.blocks[i].gate_proj.data = w['gate_proj']
            self.blocks[i].up_proj.data = w['up_proj']
            self.blocks[i].down_proj.data = w['down_proj']
            self.blocks[i].input_norm.data = w['input_layernorm']
            self.blocks[i].post_norm.data = w['post_attention_layernorm']
        # Freeze MLP, only train A_d
        for b in self.blocks:
            b.gate_proj.requires_grad = False
            b.up_proj.requires_grad = False
            b.down_proj.requires_grad = False
            b.input_norm.requires_grad = False
            b.post_norm.requires_grad = False
    
    def forward(self, h, return_states=False):
        """Run stacked lambda_d blocks."""
        states = [h]
        for block in self.blocks:
            h = block(h)
            states.append(h)
        if return_states:
            return states
        return h
    
    def get_A_matrices(self):
        return [b.A.detach().cpu() for b in self.blocks]

# ─── Training ────────────────────────────────────────────────────────────────

def spectral_radius(A):
    """Compute spectral radius (power iteration)."""
    x = torch.randn(A.shape[0], 1, device=A.device, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(20):
            x = A.float() @ x
            x = x / (x.norm() + 1e-10)
        sr = (x.T @ A.float() @ x).item() / (x.T @ x + 1e-10).item()
    return abs(sr)

def train_step(model, target_states, opt, layer_idx, phi_target=1.618):
    """One training step for one A_d matrix."""
    h = target_states[layer_idx].unsqueeze(0).to(DEVICE, torch.float16)
    target = target_states[layer_idx + 1].unsqueeze(0).to(DEVICE, torch.float16)
    
    # Forward through one block
    pred = model.blocks[layer_idx](h)
    
    # MSE loss
    loss = F.mse_loss(pred.float(), target.float())
    
    # Spectral radius regularization (keep near phi)
    A = model.blocks[layer_idx].A
    with torch.no_grad():
        sr = spectral_radius(A)
    reg = (sr - phi_target) ** 2 * 0.1
    
    total_loss = loss + reg
    
    opt.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_([A], 1.0)
    opt.step()
    
    return loss.item(), sr, F.cosine_similarity(pred.float(), target.float()).mean().item()


def main():
    print('=' * 60)
    print('Stacked lambda_d Distillation')
    print('=' * 60)
    
    # 1. Generate training data from Qwen
    print('\n--- Generating training data ---')
    prompts = [
        "The future of AI is",
        "Once upon a time in",
        "The key to success is",
        "In the beginning, there was",
        "Let me tell you about",
        "The most important thing is",
        "Here is what we know about",
        "The problem with this is",
        "I believe that the answer is",
        "According to recent research",
    ]
    
    all_states, tok = generate_training_data(prompts)
    print(f'  Generated {len(all_states)} sequences')
    
    # 2. Build stacked lambda_d model
    print('\n--- Building stacked lambda_d ---')
    model = StackedLD(num_layers=N_LAYERS)
    model.init_from_qwen(list(range(N_LAYERS)))
    model = model.to(DEVICE, torch.float16)
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'  Trainable: {trainable/1e6:.1f}M (A_d matrices)')
    print(f'  Frozen: {frozen/1e6:.1f}M (MLP + norms from Qwen)')
    
    # 3. Train A_d layer by layer
    print('\n--- Training A_d (layer by layer) ---')
    n_train_layers = min(12, N_LAYERS)  # train first 12 layers
    phi_target = (1 + 5**0.5) / 2
    
    for layer_idx in range(n_train_layers):
        # Collect all states for this layer
        layer_inputs = []
        layer_targets = []
        for seq_states in all_states:
            if len(seq_states) > layer_idx + 1:
                layer_inputs.append(seq_states[layer_idx])
                layer_targets.append(seq_states[layer_idx + 1])
        
        print(f'  Layer {layer_idx}: inputs={len(layer_inputs)}, shapes={[s.shape for s in layer_inputs[:3]]}')
        if len(layer_inputs) == 0:
            print(f'    SKIP - no data')
            continue
        
        # Use last token of each sequence (next-token prediction style)
        # Handle variable lengths
        X_last = torch.stack([s[-1:, :] for s in layer_inputs])  # (N, 1, D)
        Y_last = torch.stack([s[-1:, :] for s in layer_targets])
        
        # Train A for this layer
        opt = torch.optim.AdamW([model.blocks[layer_idx].A], lr=0.001)
        
        best_loss = float('inf')
        best_A = model.blocks[layer_idx].A.detach().clone()
        n_epochs = 5
        
        for epoch in range(n_epochs):
            total_loss = 0
            total_cos = 0
            
            for i in range(len(X_last)):
                h_in = X_last[i].to(DEVICE, torch.float16)  # (1, D)
                h_tgt = Y_last[i].to(DEVICE, torch.float16)
                
                # Forward through block
                pred = model.blocks[layer_idx](h_in)
                
                # MSE loss
                loss = F.mse_loss(pred.float(), h_tgt.float())
                
                # Spectral radius regularization
                A = model.blocks[layer_idx].A
                with torch.no_grad():
                    sr = spectral_radius(A)
                reg = (sr - phi_target) ** 2 * 0.1
                total_loss_val = loss + reg
                
                opt.zero_grad()
                total_loss_val.backward()
                torch.nn.utils.clip_grad_norm_([A], 1.0)
                opt.step()
                
                cos = F.cosine_similarity(pred.float(), h_tgt.float()).item()
                total_loss += loss.item()
                total_cos += cos
            
            avg_loss = total_loss / len(X_last)
            avg_cos = total_cos / len(X_last)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_A = model.blocks[layer_idx].A.detach().clone()
        
        # Restore best A
        model.blocks[layer_idx].A.data = best_A
        
        sr = spectral_radius(model.blocks[layer_idx].A)
        print(f'  Layer {layer_idx:2d}: loss={best_loss:.6f}, cos={avg_cos:.4f}, sr(A)={sr:.4f} (target={phi_target:.4f})')
    
    # 4. Evaluate: compare stacked lambda_d states with Qwen states
    print('\n--- Evaluation ---')
    test_prompt = "The future of AI is"
    tok = None  # will be loaded if needed
    from tokenizers import Tokenizer
    tok_path = Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json')
    tok = Tokenizer.from_file(str(tok_path))
    test_tokens = torch.tensor(tok.encode(test_prompt).ids[:10])
    
    with torch.no_grad():
        # Run Qwen pipeline
        from copy import deepcopy
        h = EMBED_W[test_tokens].unsqueeze(0).to(DEVICE, torch.float16)
        L = test_tokens.shape[0]
        pos = torch.arange(L, dtype=torch.long)
        
        qwen_states = [h.squeeze(0).cpu()]
        for lidx in range(N_LAYERS):
            w = get_layer_weights(lidx)
            w_gpu = {k: v.to(DEVICE, torch.float16) for k, v in w.items()}
            h_norm = rms_norm(h, w_gpu['input_layernorm'])
            attn_out = attention_forward(h_norm, w_gpu, pos)
            h = h.half() + attn_out.half()
            h_norm = rms_norm(h, w_gpu['post_attention_layernorm'])
            mlp_out = mlp_forward(h_norm, w_gpu)
            h = h + mlp_out.half()
            del w_gpu
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            qwen_states.append(h.squeeze(0).cpu())
        
        # Run stacked lambda_d
        h_ld = EMBED_W[test_tokens].unsqueeze(0).to(DEVICE, torch.float16)
        ld_states = model.forward(h_ld, return_states=True)
        ld_states = [s.squeeze(0).cpu() for s in ld_states]
        
        # Compare
        print(f'\n  Per-layer cosine similarity (last token):')
        cos_values = []
        for lidx in range(min(n_train_layers + 1, len(qwen_states))):
            if lidx < len(ld_states):
                cos = F.cosine_similarity(
                    qwen_states[lidx][-1:].float(), 
                    ld_states[lidx][-1:].float()
                ).item()
                cos_values.append(cos)
                marker = ' [trained]' if lidx <= n_train_layers and lidx > 0 else ''
                print(f'    Layer {lidx:2d}: cos={cos:.4f}{marker}')
        
        print(f'\n  Avg cos (trained layers 1-{n_train_layers}): {np.mean(cos_values[1:n_train_layers+1]):.4f}')
        if len(cos_values) > n_train_layers + 1:
            print(f'  Avg cos (untrained layers {n_train_layers+1}-{N_LAYERS}): {np.mean(cos_values[n_train_layers+1:]):.4f}')
    
    print(f'\n  lambda_d + A_d replaces attention with:')
    print(f'    - {trainable/1e6:.1f}M trainable params (A_d matrices)')
    print(f'    - O(d²) per layer vs O(L·d) for attention')
    print(f'    - No KV cache, no RoPE, no softmax')


if __name__ == '__main__':
    main()
