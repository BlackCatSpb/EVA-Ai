"""
Bandit λ_d generation + A/B readout test.
Loads trained A_d from checkpoint, no re-training needed.
"""
import os, sys, time, math, pickle
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
phi = (1 + 5**0.5) / 2
LOG_PHI = math.log(phi)
LOG_SQRT5 = math.log(5**0.5)

PT_PATH = Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\qwen_layer_model.pt')
CKPT_PATH = Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\checkpoints\A_d_mats_36.pkl')

print('Loading Qwen state...')
t0 = time.perf_counter()
STATE = torch.load(PT_PATH, map_location='cpu', mmap=True, weights_only=False)
SD = STATE['model_state_dict']
EMBED_W = SD['base_model.model.embed_tokens.weight'].half()
FINAL_NORM_W = SD['base_model.model.norm.weight'].half()
print(f'  {time.perf_counter()-t0:.1f}s')

print('Loading A_d matrices...')
with open(CKPT_PATH, 'rb') as f:
    A_mats = pickle.load(f)
print(f'  Loaded {len(A_mats)} A_d matrices ({sum(v.numel() for v in A_mats.values())/1e6:.0f}M params)')

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

def ld_forward(tokens):
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

def qwen_forward(tokens):
    """Run full Qwen reference: returns (1, 1, D) final hidden state."""
    L = len(tokens)
    pos = torch.arange(L, dtype=torch.long)
    h = EMBED_W[tokens].unsqueeze(0).to(DEVICE, torch.float16)
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
    return rms(h, FINAL_NORM_W.to(DEVICE,torch.float16))[:, -1:, :]

def qwen_generate(prompt, max_new_tokens=8):
    """Qwen reference generation (greedy)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json')))
    gen = tok.encode(prompt).ids[:max_new_tokens]
    lm_w = EMBED_W.to(DEVICE, torch.float16)
    for _ in range(max_new_tokens):
        h_last = qwen_forward(torch.tensor(gen))
        logits = h_last.squeeze(0) @ lm_w.T
        next_token = logits.topk(1).indices[0, 0].item()
        gen.append(next_token)
    return tok.decode(gen)


def ld_generate_bandit(prompt, max_new_tokens=20, epsilon=0.15,
                      repeat_penalty=2.0, window=8, temp=0.8, use_ld_readout=False):
    """Generate with ε-greedy bandit + repetition penalty."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json')))
    
    gen = tok.encode(prompt).ids[:max_new_tokens]
    recent = []
    lm_w = EMBED_W.to(DEVICE, torch.float16)
    
    for step in range(max_new_tokens):
        h_last = ld_forward(torch.tensor(gen))
        logits = h_last.squeeze(0) @ lm_w.T  # (1, V)
        
        for t in recent:
            logits[0, t] -= repeat_penalty
        
        if use_ld_readout:
            ld = logits.float() / temp * LOG_PHI - LOG_SQRT5
            probs = torch.exp(ld - torch.logsumexp(ld, dim=-1, keepdim=True))
        else:
            probs = F.softmax(logits.float() / temp, dim=-1)
        
        if torch.rand(1).item() < epsilon:
            probs = probs.clamp(min=1e-10)
            probs = probs / probs.sum()
            next_token = torch.multinomial(probs.squeeze(0), 1).item()
        else:
            if use_ld_readout:
                next_token = logits.topk(1).indices[0, 0].item()
            else:
                next_token = probs.topk(1).indices[0, 0].item()
        
        gen.append(next_token)
        recent.append(next_token)
        if len(recent) > window:
            recent.pop(0)
        
        try:
            dec = tok.decode([next_token]).encode('ascii', 'replace').decode(errors='replace')
        except:
            dec = f'[tok {next_token}]'
        sys.stdout.buffer.write(f'\r  Step {step+1:2d}: "{dec}"'.encode('utf-8', errors='replace'))
        sys.stdout.buffer.flush()
        sys.stdout.flush()
    
    print()
    return tok.decode(gen)


# ─── Main test ────────────────────────────────────────────────────────

def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB\tokenizer.json')))
    lm_w = EMBED_W.to(DEVICE, torch.float16)
    
    test_prompts = [
        "The future of AI is",
        "The key to success is",
        "In this paper we",
        "According to recent research",
        "The results show that",
    ]
    
    # ─── 1. Baseline: Greedy λ_d generation (no bandit) ──────────────
    print('\n' + '='*60)
    print('1. Greedy λ_d (baseline, no bandit)')
    print('='*60)
    for p in test_prompts:
        h_last = ld_forward(torch.tensor(tok.encode(p).ids[:8]))
        logits = h_last.squeeze(0) @ lm_w.T
        top3 = [(t, tok.decode([t]).encode('ascii','replace').decode()) for t in logits.topk(3).indices.tolist()[0]]
        print(f'  {p!r}  ->  {top3}')
    
    # ─── 2. Bandit λ_d generation (ε=0.15, repeat penalty) ───────────
    print('\n' + '='*60)
    print('2. Bandit λ_d (ε=0.15, repeat_penalty=2.0)')
    print('='*60)
    for p in test_prompts[:3]:
        print(f'\n  Prompt: {p!r}')
        text = ld_generate_bandit(p, max_new_tokens=12, epsilon=0.15,
                                  repeat_penalty=2.0, window=8, temp=0.8,
                                  use_ld_readout=False)
        print(f'  Result: {text}')
    
    # ─── 3. Bandit λ_d with λ_d readout ───────────────────────────────
    print('\n' + '='*60)
    print('3. Bandit λ_d + λ_d readout (no exp softmax)')
    print('='*60)
    for p in test_prompts[:3]:
        print(f'\n  Prompt: {p!r}')
        text = ld_generate_bandit(p, max_new_tokens=12, epsilon=0.15,
                                  repeat_penalty=2.0, window=8, temp=0.8,
                                  use_ld_readout=True)
        print(f'  Result: {text}')
    
    # ─── 4. A/B test: λ_d vs exp at same hidden state ────────────────
    print('\n' + '='*60)
    print('4. A/B Test: λ_d readout vs exp softmax')
    print('='*60)
    for p in test_prompts:
        tokens = torch.tensor(tok.encode(p).ids[:8])
        h_last = ld_forward(tokens)
        logits = h_last.squeeze(0) @ lm_w.T
        
        probs_exp = F.softmax(logits.float(), dim=-1)
        top5_exp = probs_exp.topk(5).indices.tolist()[0]
        
        ld = logits.float() * LOG_PHI - LOG_SQRT5
        probs_ld = torch.exp(ld - torch.logsumexp(ld, dim=-1, keepdim=True))
        top5_ld = probs_ld.topk(5).indices.tolist()[0]
        
        exp_top = [(t, tok.decode([t]).encode('ascii','replace').decode()) for t in top5_exp]
        ld_top = [(t, tok.decode([t]).encode('ascii','replace').decode()) for t in top5_ld]
        overlap = sum(1 for t in top5_exp if t in top5_ld)
        
        print(f'\n  {p!r}')
        print(f'    exp top-5: {exp_top}')
        print(f'    λ_d top-5: {ld_top}')
        print(f'    overlap: {overlap}/5')
    
    # ─── 5. Compare: Qwen vs best λ_d generation ─────────────────────
    print('\n' + '='*60)
    print('5. Qwen vs Bandit λ_d (full generation comparison)')
    print('='*60)
    for p in test_prompts:
        print(f'\n  Prompt: {p!r}')
        qwen_text = qwen_generate(p, max_new_tokens=8)
        ld_text = ld_generate_bandit(p, max_new_tokens=8, epsilon=0.1,
                                     repeat_penalty=1.5, window=6, temp=0.7,
                                     use_ld_readout=False)
        print(f'    Qwen:      {qwen_text}')
        print(f'    λ_d-bandit: {ld_text}')

if __name__ == '__main__':
    main()
