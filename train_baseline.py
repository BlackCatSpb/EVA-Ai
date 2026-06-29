"""
Baseline: Single Transformer attention layer (from scratch).
Same data/hyperparams as train_phase1.py for fair comparison.
"""

import os, sys, math, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─── Config (matches Phase 1) ──────────────────────────────────────────
D = 256
VOCAB = 50000
N_HEADS = 4
INTERMEDIATE = 1024
BATCH_SIZE = 8
SEQ_LEN = 128
LR = 3e-4
EPOCHS = 3
WARMUP_STEPS = 100
GRAD_CLIP = 1.0
LOG_EVERY = 10
EVAL_EVERY = 50
MAX_TRAIN = 5000
MAX_EVAL = 200

TOKENIZER_PATH = 'C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/mlearning/eva_models/qwen3.5-0.8b'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
VOCAB = min(VOCAB, tokenizer.vocab_size)
print(f'Vocab: {VOCAB}')

# ─── Single Attention Layer ────────────────────────────────────────────
class SingleAttentionLayer(nn.Module):
    """One transformer block: RoPE attn + SwiGLU MLP + pre-RMSNorm."""
    def __init__(self, d_model, n_heads, intermediate):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.input_layernorm = nn.RMSNorm(d_model, eps=1e-6)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

        self.post_attention_layernorm = nn.RMSNorm(d_model, eps=1e-6)
        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)

    @staticmethod
    def precompute_freqs(dim, max_len=512, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, freqs)
        return torch.cos(freqs), torch.sin(freqs)

    def apply_rotary(self, x, cos, sin):
        # x: (B, L, n_heads, head_dim)
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        cos = cos[:x.shape[1], :half].unsqueeze(0).unsqueeze(2)
        sin = sin[:x.shape[1], :half].unsqueeze(0).unsqueeze(2)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        # Pre-attention norm
        h = self.input_layernorm(x)
        q = self.q_proj(h).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.apply_rotary(q, cos, sin)
        k = self.apply_rotary(k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, L, self.d_model)
        h = x + self.o_proj(attn)
        # MLP (SwiGLU)
        r = self.post_attention_layernorm(h)
        gate = F.silu(self.gate_proj(r))
        h = h + self.down_proj(gate * self.up_proj(r))
        return h

# ─── Full Model ────────────────────────────────────────────────────────
class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.block = SingleAttentionLayer(D, N_HEADS, INTERMEDIATE)
        self.final_norm = nn.LayerNorm(D, eps=1e-6)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)
        cos, sin = SingleAttentionLayer.precompute_freqs(D // N_HEADS * 2, SEQ_LEN + 1)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, input_ids):
        B, L = input_ids.shape
        h = self.embed(input_ids)
        h = self.block(h, self.cos, self.sin)
        h = self.final_norm(h.float())
        logits = self.lm_head(h)
        return logits

model = BaselineModel().to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Model: {trainable/1e3:.0f}K / {total/1e3:.0f}K')

# ─── Data (identical to Phase 1) ───────────────────────────────────────
def tokenize_fn(examples):
    texts = [t for t in examples['text'] if len(t.strip()) > 0]
    if not texts:
        return {'input_ids': []}
    enc = tokenizer(texts, truncation=True, max_length=SEQ_LEN + 1, padding=False)
    return {'input_ids': [[min(i, VOCAB - 1) for i in ids] for ids in enc['input_ids']]}

print('Loading wikitext-103...')
ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
ds = ds.filter(lambda x: len(x['text'].strip()) > 0)
ds = ds.map(tokenize_fn, batched=True, remove_columns=['text'])

def chunk_fn(examples):
    all_ids = []
    for ids in examples['input_ids']:
        all_ids.extend(ids)
    chunks = []
    for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN // 2):
        chunk = all_ids[i:i + SEQ_LEN + 1]
        if len(chunk) == SEQ_LEN + 1:
            chunks.append(chunk)
    return {'chunk': chunks}

ds = ds.map(chunk_fn, batched=True, remove_columns=['input_ids'])
ds = ds.flatten()
print(f'Total chunks: {len(ds)}')

ds = ds.select(range(min(len(ds), MAX_TRAIN + MAX_EVAL)))
train_ds = ds.select(range(MAX_TRAIN))
eval_ds = ds.select(range(MAX_TRAIN, min(len(ds), MAX_TRAIN + MAX_EVAL)))
print(f'Train: {len(train_ds)}, Eval: {len(eval_ds)}')

def collate_fn(batch):
    ids = torch.tensor([b['chunk'] for b in batch], dtype=torch.long)
    return ids[:, :-1].to(DEVICE), ids[:, 1:].to(DEVICE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

# ─── Training ──────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * EPOCHS)

step = 0
best_ppl = float('inf')
metrics_log = []

print(f'\nTraining: {len(train_loader)} batches/epoch, {EPOCHS} epochs')
t0 = time.perf_counter()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), batch_y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        n_batches += 1
        step += 1

        if step % LOG_EVERY == 0:
            ppl = math.exp(loss.item())
            lr_now = scheduler.get_last_lr()[0]
            print(f'  Step {step:5d} | loss={loss.item():.4f} | ppl={ppl:.1f} | lr={lr_now:.2e}')

    avg_loss = epoch_loss / n_batches
    print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}, ppl={math.exp(avg_loss):.1f}')

    # Eval
    model.eval()
    eval_loss = 0.0
    eval_batches = 0
    with torch.no_grad():
        for bx, by in eval_loader:
            logits = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
            eval_loss += loss.item()
            eval_batches += 1
            if eval_batches >= 50:
                break

    eval_ppl = math.exp(eval_loss / eval_batches)
    print(f'  Eval: loss={eval_loss/eval_batches:.4f}, ppl={eval_ppl:.1f}')

    metrics_log.append({
        'epoch': epoch + 1,
        'train_loss': avg_loss,
        'train_ppl': math.exp(avg_loss),
        'eval_loss': eval_loss / eval_batches,
        'eval_ppl': eval_ppl,
    })

    if eval_ppl < best_ppl:
        best_ppl = eval_ppl
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), 'checkpoints/baseline_best.pt')
        print(f'  [New best: ppl={best_ppl:.1f}]')

t_train = time.perf_counter() - t0
print(f'\nTraining time: {t_train:.0f}s')
print(f'Best eval ppl: {best_ppl:.1f}')

# ─── Summary ───────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('BASELINE RESULTS (Transformer Attention)')
print('=' * 60)
print(f'D={D}, heads={N_HEADS}, lr={LR}')
print(f'Trainable: {trainable/1e3:.0f}K')
for m in metrics_log:
    print(f'  Epoch {m["epoch"]}: train_ppl={m["train_ppl"]:.1f}, eval_ppl={m["eval_ppl"]:.1f}')
print(f'Best ppl: {best_ppl:.1f}')

with open('checkpoints/baseline_metrics.json', 'w') as f:
    json.dump(metrics_log, f, indent=2)

print('Done.')
