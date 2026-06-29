"""
Stacked lambda_d v2: train A_d directly on attention deltas.
Block: h_out = h_in + A_d@rms_norm(h_in) + MLP(rms_norm(h_in + A_d@rms_norm(h_in)))
"""

import os, sys, time, math
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8'
torch.set_float32_matmul_precision('high')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

D = 2560
N_LAYERS = 36
VOCAB = 146260
INTERMEDIATE = 9728
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128

PT_PATH = Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\qwen_layer_model.pt')

print('Loading Qwen state...')
t0 = time.perf_counter()
STATE = torch.load(PT_PATH, map_location='cpu', mmap=True, weights_only=False)
SD = STATE['model_state_dict']
EMBED_W = SD['base_model.model.embed_tokens.weight'].half()
FINAL_NORM_W = SD['base_model.model.norm.weight'].half()
print(f'  {time.perf_counter()-t0:.1f}s')

# ─── Qwen forward utils ──────────────────────────────────────────────────────

def precompute_rope(max_pos=4096, theta=10000000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM//2, dtype=torch.float32) / (HEAD_DIM//2)))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().half(), freqs.sin().half()
FREQS_COS, FREQS_SIN = precompute_rope()

def rope(x, pos):
    B, nh, L, d = x.shape
    cos = FREQS_COS[pos].to(x.device).view(1,1,L,d//2)
    sin = FREQS_SIN[pos].to(x.device).view(1,1,L,d//2)
    x1, x2 = x[...,:d//2], x[...,d//2:]
    return torch.cat([x1*cos-x2*sin, x1*sin+x2*cos], dim=-1)

def rms(x, w):
    x = x.half()
    r = x.norm(dim=-1,keepdim=True)/(D**0.5)
    return x/(r+1e-6).half()*w.half()

def attn_forward(h, w, pos):
    B,L,_=h.shape
    q=(h@w['q_proj'].T).view(B,L,N_HEADS,HEAD_DIM)
    k=(h@w['k_proj'].T).view(B,L,N_KV_HEADS,HEAD_DIM)
    v=(h@w['v_proj'].T).view(B,L,N_KV_HEADS,HEAD_DIM)
    qn=w['q_norm'].view(1,1,1,-1); kn=w['k_norm'].view(1,1,1,-1)
    q=q.half()/(q.half().norm(dim=-1,keepdim=True)/(HEAD_DIM**0.5)+1e-6).half()*qn.half()
    k=k.half()/(k.half().norm(dim=-1,keepdim=True)/(HEAD_DIM**0.5)+1e-6).half()*kn.half()
    q=rope(q.transpose(1,2),pos); k=rope(k.transpose(1,2),pos); v=v.transpose(1,2)
    g=N_HEADS//N_KV_HEADS
    if g>1:
        k=k[:,:,None].expand(-1,-1,g,-1,-1).reshape(B,N_HEADS,L,HEAD_DIM)
        v=v[:,:,None].expand(-1,-1,g,-1,-1).reshape(B,N_HEADS,L,HEAD_DIM)
    s=torch.tensor(HEAD_DIM,dtype=torch.float16,device=h.device)**0.5
    a=(q@k.transpose(-2,-1))/s+torch.triu(torch.full((L,L),-3e4,dtype=torch.float16,device=h.device),1)
    o=F.softmax(a.float(),dim=-1).half()@v
    return o.transpose(1,2).reshape(B,L,N_HEADS*HEAD_DIM)@w['o_proj'].T

def mlp_forward(h, w):
    g=F.silu((h@w['gate_proj'].T).float()).half()
    u=(h@w['up_proj'].T).half()
    return ((g*u)@w['down_proj'].T).half()

def get_w(l):
    p=f'base_model.model.layers.{l}.self_attn.'
    m=f'base_model.model.layers.{l}.mlp.'
    n=f'base_model.model.layers.{l}.'
    return {k:SD[p+k+'.weight'].half() for k in ['q_proj','k_proj','v_proj','o_proj','q_norm','k_norm']} | \
           {k:SD[m+k+'.weight'].half() for k in ['gate_proj','up_proj','down_proj']} | \
           {k:SD[n+k+'.weight'].half() for k in ['input_layernorm','post_attention_layernorm']}

# ─── Data generation: extract attention deltas ──────────────────────────────

def gen_data(prompts):
    """Extract (h_norm, attn_delta) for each layer and prompt."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json')))
    
    data = {l: {'inputs': [], 'targets': []} for l in range(N_LAYERS)}
    
    for pi, p in enumerate(prompts):
        tokens = torch.tensor(tok.encode(p).ids[:10])
        L = len(tokens)
        pos = torch.arange(L, dtype=torch.long)
        h = EMBED_W[tokens].unsqueeze(0).to(DEVICE, torch.float16)
        
        print(f'Prompt {pi}: L={L}', end=' ')
        t0 = time.perf_counter()
        
        for lidx in range(N_LAYERS):
            w = get_w(lidx); wg = {k:v.to(DEVICE,torch.float16) for k,v in w.items()}
            
            # Extract attention inputs and outputs FROM THIS LAYER
            h_norm = rms(h, wg['input_layernorm'])
            attn_delta = attn_forward(h_norm, wg, pos)
            h = h.half() + attn_delta.half()
            
            h_norm2 = rms(h, wg['post_attention_layernorm'])
            mlp_delta = mlp_forward(h_norm2, wg)
            h = h + mlp_delta.half()
            
            # Save LAST token's h_norm (input to attention) and attn_delta
            # These are the training data for A_d
            data[lidx]['inputs'].append(h_norm[0, -1:, :].cpu())  # (1, D)
            data[lidx]['targets'].append(attn_delta[0, -1:, :].cpu())  # (1, D)
            
            del wg
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        
        print(f'{time.perf_counter()-t0:.1f}s')
    
    # Stack all
    for l in range(N_LAYERS):
        data[l]['X'] = torch.cat(data[l]['inputs'], dim=0)  # (N, D)
        data[l]['Y'] = torch.cat(data[l]['targets'], dim=0)
        print(f'  Layer {l}: X={data[l]["X"].shape}, Y={data[l]["Y"].shape}')
    
    return data

# ─── Linear A_d training ─────────────────────────────────────────────────────

def train_A_d(data, n_layers=12, lr=0.01, steps=50):
    """Train A_d matrices as linear regression: minimize ||A@X - Y||^2 + reg."""
    A_mats = {}
    
    for lidx in range(n_layers):
        X = data[lidx]['X'].to(DEVICE, torch.float32)  # (N, D)
        Y = data[lidx]['Y'].to(DEVICE, torch.float32)
        
        # Initialize A_d as identity with small random perturbation
        A = torch.nn.Parameter(torch.eye(D, dtype=torch.float32, device=DEVICE) * 0.01)
        opt = torch.optim.AdamW([A], lr=lr)
        
        phi = (1 + 5**0.5) / 2
        best_loss = float('inf')
        best_A = A.detach().clone()
        
        for step in range(steps):
            pred = X @ A.T  # (N, D) @ (D, D) -> (N, D)
            loss = F.mse_loss(pred, Y)
            
            # Spectral radius regularization: keep near phi
            with torch.no_grad():
                x = torch.randn(D, 1, device=DEVICE, dtype=torch.float32)
                for _ in range(10):
                    x = A @ x
                    x = x / (x.norm() + 1e-10)
                sr = (x.T @ A @ x).item() / (x.T @ x + 1e-10).item()
            loss = loss + 0.1 * (abs(sr) - phi) ** 2
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([A], 1.0)
            opt.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_A = A.detach().clone()
        
        # Final metrics
        with torch.no_grad():
            pred_final = X @ best_A.T
            cos = F.cosine_similarity(pred_final, Y, dim=-1).mean().item()
        
        A_mats[lidx] = best_A.cpu().half()
        print(f'  Layer {lidx:2d}: loss={best_loss:.6f}, cos={cos:.4f}, |sr|={abs(sr):.4f}')
    
    return A_mats

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print('='*60)
    print('Stacked lambda_d Distillation v2')
    print('='*60)
    
    prompts = [
        "The future of AI is","Once upon a time in","The key to success is",
        "In the beginning there was","Let me tell you about","The most important thing",
        "Here is what we know","The problem with this","I believe that the answer",
        "According to recent research","This suggests that","We have found that",
        "The results show that","Our analysis indicates","Based on these findings",
        "Scientists have discovered","The new technology enables","Researchers found that",
        "The data suggests a","Our model predicts that","In this paper we",
        "The algorithm works by","Neural networks can learn","Deep learning has revolutionized",
        "The transformer architecture uses","Attention is all you need",
        "Language models understand context","Training requires large datasets",
        "The embedding layer maps","Tokenization splits text into",
        "Gradient descent optimizes the","Backpropagation computes gradients through",
        "The loss function measures","Regularization prevents overfitting by",
        "Batch normalization stabilizes training","Dropout randomly masks neurons",
        "The optimizer updates weights","Learning rate scheduling improves convergence",
        "The hidden state encodes","The output layer predicts tokens",
        "The key to intelligence is","Understanding language requires reasoning",
        "The future will bring","В будущем искусственный интеллект",
        "Развитие технологий приводит к","Новые методы машинного обучения",
        "Исследования показывают что","Модели понимают естественный язык",
        "Обучение на больших данных позволяет",
        "Результаты экспериментов подтверждают","Наш подход основан на",
    ]
    
    print('\n--- Generating training data ---')
    data = gen_data(prompts)
    
    print('\n--- Training A_d matrices ---')
    A_mats = train_A_d(data, n_layers=N_LAYERS, lr=0.01, steps=50)
    
    # ─── Test: forward with trained A_d + MLP ──────────────────────────
    print('\n--- Testing trained stacked lambda_d ---')
    test = "The future of AI is"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json')))
    tokens = torch.tensor(tok.encode(test).ids[:8])
    L = len(tokens)
    pos = torch.arange(L, dtype=torch.long)
    
    # Run full Qwen for comparison
    print('  Running Qwen reference...')
    h_q = EMBED_W[tokens].unsqueeze(0).to(DEVICE, torch.float16)
    for lidx in range(N_LAYERS):
        w = get_w(lidx); wg = {k:v.to(DEVICE,torch.float16) for k,v in w.items()}
        h_norm = rms(h_q, wg['input_layernorm'])
        attn_delta = attn_forward(h_norm, wg, pos)
        h_q = h_q.half() + attn_delta.half()
        h_norm2 = rms(h_q, wg['post_attention_layernorm'])
        mlp_delta = mlp_forward(h_norm2, wg)
        h_q = h_q + mlp_delta.half()
        del wg
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    h_q_last = rms(h_q, FINAL_NORM_W.to(DEVICE,torch.float16))[:, -1:, :]
    
    # Run stacked lambda_d with trained A_d
    print('  Running stacked lambda_d...')
    h_ld = EMBED_W[tokens].unsqueeze(0).to(DEVICE, torch.float16)
    for lidx in range(N_LAYERS):
        w = get_w(lidx)
        wg = {k:v.to(DEVICE,torch.float16) for k,v in w.items()}
        
        # Load trained A_d or use zero (untrained layers -> no attention)
        if lidx in A_mats:
            A = A_mats[lidx].to(DEVICE, torch.float16)
        else:
            A = torch.zeros(D, D, dtype=torch.float16, device=DEVICE)
        
        # lambda_d block: h_out = h_in + A@rms_norm(h_in) + MLP(rms_norm(h_in + A@rms_norm(h_in)))
        h_norm = rms(h_ld, wg['input_layernorm'])
        attn_delta = h_norm @ A.T  # A_d replaces attention
        h_ld = h_ld.half() + attn_delta.half()
        h_norm2 = rms(h_ld, wg['post_attention_layernorm'])
        mlp_delta = mlp_forward(h_norm2, wg)
        h_ld = h_ld + mlp_delta.half()
        
        del wg
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    h_ld_last = rms(h_ld, FINAL_NORM_W.to(DEVICE,torch.float16))[:, -1:, :]
    
    # Compare
    print(f'  Shapes: h_q_last={h_q_last.shape}, h_ld_last={h_ld_last.shape}')
    cos_final = F.cosine_similarity(h_q_last.float(), h_ld_last.float(), dim=-1).item()
    print(f'\n  Final hidden state cos: {cos_final:.4f}')
    
    # lm_head readout comparison
    lm_w = EMBED_W.to(DEVICE, torch.float16)
    logits_q = h_q_last.squeeze(0) @ lm_w.T
    logits_ld = h_ld_last.squeeze(0) @ lm_w.T
    top5_q = logits_q.topk(5).indices.tolist()
    top5_ld = logits_ld.topk(5).indices.tolist()
    overlap = sum(1 for t in top5_q[0] if t in top5_ld[0])
    
    ids_q = [int(top5_q[0][j]) for j in range(min(5, len(top5_q[0])))]
    ids_ld = [int(top5_ld[0][j]) for j in range(min(5, len(top5_ld[0])))]
    print(f'  Qwen lm_head top-5: tokens={ids_q}')
    print(f'  lambda_d top-5:     tokens={ids_ld}')
    print(f'  Top-5 overlap: {overlap}/5')
    print(f'  Total trainable: {len(A_mats)*D*D/1e6:.1f}M params ({len(A_mats)} A_d matrices)')
    
    # ─── Multi-token generation test ─────────────────────────────────────
    print('\n--- Multi-token generation ---')
    gen_prompts = [
        "The future of AI is",
        "The key to success is",
        "In this paper we",
    ]
    for gp in gen_prompts:
        print(f'\n  Prompt: {gp!r}')
        
        # Qwen reference
        tokens = torch.tensor(tok.encode(gp).ids[:8])
        h_q = EMBED_W[tokens].unsqueeze(0).to(DEVICE, torch.float16)
        L_orig = len(tokens)
        pos = torch.arange(L_orig, dtype=torch.long)
        for lidx in range(N_LAYERS):
            w = get_w(lidx); wg = {k:v.to(DEVICE,torch.float16) for k,v in w.items()}
            h_norm = rms(h_q, wg['input_layernorm'])
            attn_delta = attn_forward(h_norm, wg, pos)
            h_q = h_q.half() + attn_delta.half()
            h_norm2 = rms(h_q, wg['post_attention_layernorm'])
            mlp_delta = mlp_forward(h_norm2, wg)
            h_q = h_q + mlp_delta.half()
            del wg
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        h_last = rms(h_q, FINAL_NORM_W.to(DEVICE,torch.float16))[:, -1:, :]
        logits = h_last.squeeze(0) @ EMBED_W.to(DEVICE, torch.float16).T
        top5 = logits.topk(5).indices.tolist()[0]
        decoded = tok.decode_batch([[t] for t in top5[:3]])
        print(f'    Qwen next-3: {[(t, tok.decode([t])) for t in top5[:3]]}')
        
        # lambda_d stack
        tokens = torch.tensor(tok.encode(gp).ids[:8])
        h_ld = EMBED_W[tokens].unsqueeze(0).to(DEVICE, torch.float16)
        for lidx in range(N_LAYERS):
            w = get_w(lidx); wg = {k:v.to(DEVICE,torch.float16) for k,v in w.items()}
            if lidx in A_mats:
                A = A_mats[lidx].to(DEVICE, torch.float16)
            else:
                A = torch.zeros(D, D, dtype=torch.float16, device=DEVICE)
            h_norm = rms(h_ld, wg['input_layernorm'])
            attn_delta = h_norm @ A.T
            h_ld = h_ld.half() + attn_delta.half()
            h_norm2 = rms(h_ld, wg['post_attention_layernorm'])
            mlp_delta = mlp_forward(h_norm2, wg)
            h_ld = h_ld + mlp_delta.half()
            del wg
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        h_last = rms(h_ld, FINAL_NORM_W.to(DEVICE,torch.float16))[:, -1:, :]
        logits = h_last.squeeze(0) @ EMBED_W.to(DEVICE, torch.float16).T
        top5 = logits.topk(5).indices.tolist()[0]
        print(f'    ld next-3:    {[(t, tok.decode([t])) for t in top5[:3]]}')
    
    return A_mats


def ld_forward(tokens, A_mats):
    """Run λ_d stack: returns (1, 1, D) final hidden state."""
    L = len(tokens)
    pos = torch.arange(L, dtype=torch.long)
    h = EMBED_W[tokens].unsqueeze(0).to(DEVICE, torch.float16)
    for lidx in range(N_LAYERS):
        w = get_w(lidx); wg = {k:v.to(DEVICE,torch.float16) for k,v in w.items()}
        if lidx in A_mats:
            A = A_mats[lidx].to(DEVICE, torch.float16)
        else:
            A = torch.zeros(D, D, dtype=torch.float16, device=DEVICE)
        h_norm = rms(h, wg['input_layernorm'])
        attn_delta = h_norm @ A.T
        h = h.half() + attn_delta.half()
        h_norm2 = rms(h, wg['post_attention_layernorm'])
        mlp_delta = mlp_forward(h_norm2, wg)
        h = h + mlp_delta.half()
        del wg
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    return rms(h, FINAL_NORM_W.to(DEVICE,torch.float16))[:, -1:, :]


def ld_generate_with_bandit(prompt, A_mats, max_new_tokens=20, epsilon=0.1,
                           repeat_penalty=2.0, window=10, temp=0.8, use_ld_readout=False):
    """Generate with ε-greedy bandit + repetition penalty.
    
    ε-greedy: with prob ε, explore (sample from full distribution);
    otherwise exploit (greedy argmax).
    Repetition penalty: penalizes recently generated tokens.
    """
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json')))
    
    gen = tok.encode(prompt).ids[:max_new_tokens]
    recent = []
    
    lm_w = EMBED_W.to(DEVICE, torch.float16)
    
    for step in range(max_new_tokens):
        h_last = ld_forward(torch.tensor(gen), A_mats)
        logits = h_last.squeeze(0) @ lm_w.T  # (1, V)
        
        # Apply repetition penalty
        for t in recent:
            logits[0, t] -= repeat_penalty
        
        # ε-greedy
        if use_ld_readout:
            ld = logits.float() / temp * LOG_PHI - LOG_SQRT5
            probs = torch.exp(ld - torch.logsumexp(ld, dim=-1, keepdim=True))
        else:
            probs = F.softmax(logits.float() / temp, dim=-1)
        
        if torch.rand(1).item() < epsilon:
            # Explore: sample from full distribution
            probs = probs.clamp(min=1e-10)
            probs = probs / probs.sum()
            next_token = torch.multinomial(probs.squeeze(0), 1).item()
        else:
            # Exploit: greedy
            if use_ld_readout:
                next_token = logits.topk(1).indices[0, 0].item()
            else:
                next_token = probs.topk(1).indices[0, 0].item()
        
        gen.append(next_token)
        recent.append(next_token)
        if len(recent) > window:
            recent.pop(0)
        
        decoded = tok.decode([next_token]).encode('ascii', 'replace').decode()
        sys.stdout.write(f'\r  Step {step+1:2d}: "{decoded}" (ε={epsilon:.1f}, t={temp:.1f})')
        sys.stdout.flush()
    
    print()
    return tok.decode(gen)


def ab_test_readout(prompts, A_mats, max_new_tokens=8):
    """A/B test: λ_d readout vs exp softmax on same λ_d stack."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json')))
    
    print('\n' + '='*60)
    print('A/B Test: λ_d readout vs exp softmax on λ_d stack')
    print('='*60)
    
    lm_w = EMBED_W.to(DEVICE, torch.float16)
    
    for prompt in prompts:
        print(f'\n  Prompt: {prompt!r}')
        tokens = torch.tensor(tok.encode(prompt).ids[:max_new_tokens])
        
        # Get final hidden state from λ_d stack
        h_last = ld_forward(tokens, A_mats)
        logits = h_last.squeeze(0) @ lm_w.T
        
        # exp softmax
        probs_exp = F.softmax(logits.float(), dim=-1)
        top5_exp = probs_exp.topk(5).indices.tolist()[0]
        
        # λ_d readout
        ld = logits.float() * LOG_PHI - LOG_SQRT5
        probs_ld = torch.exp(ld - torch.logsumexp(ld, dim=-1, keepdim=True))
        top5_ld = probs_ld.topk(5).indices.tolist()[0]
        
        # Decode
        exp_top = [(t, tok.decode([t]).encode('ascii','replace').decode()) for t in top5_exp]
        ld_top = [(t, tok.decode([t]).encode('ascii','replace').decode()) for t in top5_ld]
        
        overlap = sum(1 for t in top5_exp if t in top5_ld)
        kl = 0.5 * ((probs_exp[0] + 1e-10).log() - (probs_ld[0] + 1e-10).log()).abs().mean().item()
        
        print(f'    exp top-5: {exp_top}')
        print(f'    λ_d top-5: {ld_top}')
        print(f'    overlap: {overlap}/5, mean|log-ratio|: {kl:.4f}')


def save_A_mats(A_mats, path='checkpoints/A_d_mats_36.pkl'):
    import pickle
    ckpt_path = Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai') / path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    A_cpu = {k: v.cpu() for k, v in A_mats.items()}
    with open(ckpt_path, 'wb') as f:
        pickle.dump(A_cpu, f)
    print(f'\n  Saved A_d matrices to {ckpt_path} ({sum(v.numel() for v in A_cpu.values())/1e6:.0f}M params)')


def ld_generate(prompt: str, A_mats: dict, max_new_tokens: int = 20, top_k: int = 5, temperature: float = 0.7):
    """Autoregressive generation with stacked λ_d."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json')))
    
    tokens = torch.tensor(tok.encode(prompt).ids[:max_new_tokens])
    generated = tokens.tolist()
    
    for step in range(max_new_tokens):
        L = len(generated)
        pos = torch.arange(L, dtype=torch.long)
        h = EMBED_W[torch.tensor(generated)].unsqueeze(0).to(DEVICE, torch.float16)
        
        for lidx in range(N_LAYERS):
            w = get_w(lidx); wg = {k:v.to(DEVICE,torch.float16) for k,v in w.items()}
            if lidx in A_mats:
                A = A_mats[lidx].to(DEVICE, torch.float16)
            else:
                A = torch.zeros(D, D, dtype=torch.float16, device=DEVICE)
            h_norm = rms(h, wg['input_layernorm'])
            attn_delta = h_norm @ A.T
            h = h.half() + attn_delta.half()
            h_norm2 = rms(h, wg['post_attention_layernorm'])
            mlp_delta = mlp_forward(h_norm2, wg)
            h = h + mlp_delta.half()
            del wg
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        
        h_last = rms(h, FINAL_NORM_W.to(DEVICE,torch.float16))[:, -1:, :]
        logits = h_last.squeeze(0) @ EMBED_W.to(DEVICE, torch.float16).T
        
        # Sample with temperature + top-k
        if temperature > 0:
            logits = logits / temperature
        probs = F.softmax(logits.float(), dim=-1)
        topk_probs, topk_indices = probs.topk(top_k, dim=-1)
        next_token = topk_indices[0, torch.multinomial(topk_probs[0], 1).item()].item()
        generated.append(next_token)
        
        sys.stdout.write(f'\r  Step {step+1}/{max_new_tokens}: tok={next_token}')
        sys.stdout.flush()
    
    print()
    return tok.decode(generated)

def ld_generate_both(prompt: str, A_mats: dict, max_new_tokens: int = 10):
    """Compare Qwen and λ_d generation side by side."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json')))
    
    print(f'\n  Prompt: {prompt!r}')
    
    # Qwen reference
    tokens = torch.tensor(tok.encode(prompt).ids[:max_new_tokens])
    gen_q = tokens.tolist()
    for step in range(max_new_tokens):
        L = len(gen_q)
        pos = torch.arange(L, dtype=torch.long)
        h = EMBED_W[torch.tensor(gen_q)].unsqueeze(0).to(DEVICE, torch.float16)
        for lidx in range(N_LAYERS):
            w = get_w(lidx); wg = {k:v.to(DEVICE,torch.float16) for k,v in w.items()}
            h_norm = rms(h, wg['input_layernorm'])
            attn_delta = attn_forward(h_norm, wg, pos)
            h = h.half() + attn_delta.half()
            h_norm2 = rms(h, wg['post_attention_layernorm'])
            mlp_delta = mlp_forward(h_norm2, wg)
            h = h + mlp_delta.half()
            del wg
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        h_last = rms(h, FINAL_NORM_W.to(DEVICE,torch.float16))[:, -1:, :]
        logits = h_last.squeeze(0) @ EMBED_W.to(DEVICE, torch.float16).T
        top5_indices = logits.topk(5).indices.tolist()[0]
        gen_q.append(top5_indices[0])
    text_q = tok.decode(gen_q)
    
    # λ_d stack
    tokens = torch.tensor(tok.encode(prompt).ids[:max_new_tokens])
    gen_ld = tokens.tolist()
    for step in range(max_new_tokens):
        L = len(gen_ld)
        pos = torch.arange(L, dtype=torch.long)
        h = EMBED_W[torch.tensor(gen_ld)].unsqueeze(0).to(DEVICE, torch.float16)
        for lidx in range(N_LAYERS):
            w = get_w(lidx); wg = {k:v.to(DEVICE,torch.float16) for k,v in w.items()}
            if lidx in A_mats:
                A = A_mats[lidx].to(DEVICE, torch.float16)
            else:
                A = torch.zeros(D, D, dtype=torch.float16, device=DEVICE)
            h_norm = rms(h, wg['input_layernorm'])
            attn_delta = h_norm @ A.T
            h = h.half() + attn_delta.half()
            h_norm2 = rms(h, wg['post_attention_layernorm'])
            mlp_delta = mlp_forward(h_norm2, wg)
            h = h + mlp_delta.half()
            del wg
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        h_last = rms(h, FINAL_NORM_W.to(DEVICE,torch.float16))[:, -1:, :]
        logits = h_last.squeeze(0) @ EMBED_W.to(DEVICE, torch.float16).T
        top5_indices = logits.topk(5).indices.tolist()[0]
        gen_ld.append(top5_indices[0])
    text_ld = tok.decode(gen_ld)
    
    print(f'    Qwen: {text_q}')
    print(f'    ld:   {text_ld}')
    return text_q, text_ld


if __name__ == '__main__':
    A_mats = main()
    save_A_mats(A_mats)
    
    print('\n' + '='*60)
    print('Full generation comparison (greedy, 8 new tokens)')
    print('='*60)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json')))
    
    test_prompts = [
        "The future of AI is",
        "The key to success is",
        "In this paper we",
        "According to recent research",
        "The results show that",
    ]
    for p in test_prompts:
        ld_generate_both(p, A_mats, max_new_tokens=8)
    
    # ─── Bandit generation ─────────────────────────────────────────────
    print('\n' + '='*60)
    print('Bandit λ_d generation (ε-greedy + repeat penalty)')
    print('='*60)
    for p in test_prompts[:3]:
        print(f'\n  Prompt: {p!r}')
        text = ld_generate_with_bandit(p, A_mats, max_new_tokens=12, 
                                        epsilon=0.15, repeat_penalty=2.0, 
                                        window=8, temp=0.8)
        print(f'  Result: {text}')
    
    # ─── A/B test: λ_d vs exp readout ─────────────────────────────────
    ab_test_readout(test_prompts, A_mats)
