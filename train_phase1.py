"""
Phase 1: Single-layer CE training.
Trains one LDBlock on next-token prediction using wikitext-103.
"""

import os, sys, math, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDBlock, LDStack, fibonacci_roots

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─── Config ──────────────────────────────────────────────────────────────

D = 256
VOCAB = 50000       # limited vocab for fast training
N_MODES = 4
INTERMEDIATE = 1024
LORA_RANK = 0       # start without LoRA
BATCH_SIZE = 8
SEQ_LEN = 128
LR = 3e-4
EPOCHS = 3
WARMUP_STEPS = 100
GRAD_CLIP = 1.0
LOG_EVERY = 10
EVAL_EVERY = 50
MAX_TRAIN = 5000     # small subset for fast iteration
MAX_EVAL = 200

# Tokenizer
TOKENIZER_PATH = 'C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/mlearning/eva_models/qwen3.5-0.8b'
print(f'Loading tokenizer from {TOKENIZER_PATH}...')
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
VOCAB = min(VOCAB, tokenizer.vocab_size)
print(f'Vocab size: {VOCAB}')

# ─── lambda_d model ───────────────────────────────────────────────────────────

cfg = LDConfig()
cfg.D = D
cfg.n_layers = 1        # single layer for Phase 1
cfg.n_modes = N_MODES
cfg.vocab = VOCAB
cfg.intermediate = INTERMEDIATE
cfg.lora_rank = LORA_RANK
cfg.use_lora = LORA_RANK > 0

lambdas = fibonacci_roots(cfg.n_modes + 1)
print(f'lambda_k: {[f"{l:.4f}" for l in lambdas.tolist()]}')

class SingleLayerModel(nn.Module):
    """Embed → 1 LDBlock → LayerNorm → lm_head (untied)."""
    def __init__(self, cfg, lambdas):
        super().__init__()
        self.D = cfg.D
        self.vocab = cfg.vocab
        self.embed = nn.Embedding(cfg.vocab, cfg.D)
        self.block = LDBlock(cfg, layer_idx=0, lambda_roots=lambdas)
        self.final_norm = nn.LayerNorm(cfg.D, eps=1e-6)
        # Separate lm_head (untied) — standard practice, avoids auto-correlation issues
        self.lm_head = nn.Linear(cfg.D, cfg.vocab, bias=False)
    
    def forward(self, input_ids, return_gates=False):
        B, L = input_ids.shape
        h = self.embed(input_ids)
        h_out, alpha = self.block(h, return_gates=True)
        h_normed = self.final_norm(h_out.float())
        logits = self.lm_head(h_normed)  # (B, L, V)
        if return_gates:
            return logits, alpha
        return logits

model = SingleLayerModel(cfg, lambdas).to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Model: {trainable/1e3:.0f}K trainable / {total/1e3:.0f}K total')

# ─── Data ────────────────────────────────────────────────────────────────

def tokenize_fn(examples):
    texts = [t for t in examples['text'] if len(t.strip()) > 0]
    if not texts:
        return {'input_ids': []}
    enc = tokenizer(texts, truncation=True, max_length=SEQ_LEN + 1, padding=False)
    # Clamp token IDs to vocab range
    return {'input_ids': [[min(i, VOCAB - 1) for i in ids] for ids in enc['input_ids']]}

print('Loading wikitext-103...')
ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
# Filter empty lines
ds = ds.filter(lambda x: len(x['text'].strip()) > 0)
# Tokenize
ds = ds.map(tokenize_fn, batched=True, remove_columns=['text'])
# Flatten sequences into chunks of SEQ_LEN + 1
def chunk_fn(examples):
    all_ids = []
    for ids in examples['input_ids']:
        all_ids.extend(ids)
    chunks = []
    for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN // 2):  # overlap 50%
        chunk = all_ids[i:i + SEQ_LEN + 1]
        if len(chunk) == SEQ_LEN + 1:
            chunks.append(chunk)
    return {'chunk': chunks}

ds = ds.map(chunk_fn, batched=True, remove_columns=['input_ids'])
ds = ds.flatten()
print(f'Total chunks: {len(ds)}')

# Split
ds = ds.select(range(min(len(ds), MAX_TRAIN + MAX_EVAL)))
train_ds = ds.select(range(MAX_TRAIN))
eval_ds = ds.select(range(MAX_TRAIN, min(len(ds), MAX_TRAIN + MAX_EVAL)))
print(f'Train: {len(train_ds)}, Eval: {len(eval_ds)}')

def collate_fn(batch):
    ids = torch.tensor([b['chunk'] for b in batch], dtype=torch.long)
    x = ids[:, :-1]
    y = ids[:, 1:]
    return x.to(DEVICE), y.to(DEVICE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

# ─── Training ────────────────────────────────────────────────────────────

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
    epoch_gate_ent = 0.0
    n_batches = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        
        logits, alpha = model(batch_x, return_gates=True)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), batch_y.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        
        # Gate entropy
        gate_ent = -(alpha * torch.log(alpha.clamp(min=1e-10))).sum(dim=-1).mean().item()
        
        epoch_loss += loss.item()
        epoch_gate_ent += gate_ent
        n_batches += 1
        step += 1
        
        if step % LOG_EVERY == 0:
            ppl = math.exp(loss.item())
            lr_now = scheduler.get_last_lr()[0]
            print(f'  Step {step:5d} | loss={loss.item():.4f} | ppl={ppl:.2f} | '
                  f'H(alpha)={gate_ent:.3f} | lr={lr_now:.2e}')
    
    # End of epoch
    avg_loss = epoch_loss / n_batches
    avg_ent = epoch_gate_ent / n_batches
    print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}, ppl={math.exp(avg_loss):.2f}, H(alpha)={avg_ent:.3f}')
    
    # Evaluation
    model.eval()
    eval_loss = 0.0
    eval_batches = 0
    with torch.no_grad():
        for bx, by in eval_loader:
            logits, _ = model(bx, return_gates=True)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
            eval_loss += loss.item()
            eval_batches += 1
            if eval_batches >= 50:  # limit eval time
                break
    
    eval_ppl = math.exp(eval_loss / eval_batches)
    print(f'  Eval: loss={eval_loss/eval_batches:.4f}, ppl={eval_ppl:.2f}')
    
    metrics_log.append({
        'epoch': epoch + 1,
        'train_loss': avg_loss,
        'train_ppl': math.exp(avg_loss),
        'gate_entropy': avg_ent,
        'eval_loss': eval_loss / eval_batches,
        'eval_ppl': eval_ppl,
    })
    
    if eval_ppl < best_ppl:
        best_ppl = eval_ppl
        torch.save(model.state_dict(), 'checkpoints/phase1_best.pt')
        print(f'  [New best: ppl={best_ppl:.2f}]')

t_train = time.perf_counter() - t0
print(f'\nTraining time: {t_train:.0f}s ({t_train/step*1000:.1f}ms/step)')
print(f'Best eval ppl: {best_ppl:.2f}')

# ─── Summary ─────────────────────────────────────────────────────────────

print('\n' + '=' * 60)
print('PHASE 1 RESULTS')
print('=' * 60)
print(f'D={D}, K={N_MODES}, LoRA rank={LORA_RANK}, lr={LR}')
print(f'Trainable: {trainable/1e3:.0f}K')
print(f'Data: wikitext-103 ({len(train_ds)} train, {len(eval_ds)} eval)')
print(f'Steps: {step}, Time: {t_train:.0f}s')
for m in metrics_log:
    print(f'  Epoch {m["epoch"]}: train_ppl={m["train_ppl"]:.2f}, '
          f'eval_ppl={m["eval_ppl"]:.2f}, H(alpha)={m["gate_entropy"]:.3f}')
print(f'Best ppl: {best_ppl:.2f}')

# Save metrics
os.makedirs('checkpoints', exist_ok=True)
with open('checkpoints/phase1_metrics.json', 'w') as f:
    json.dump(metrics_log, f, indent=2)

print('Done.')
