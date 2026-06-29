"""
Phase 1b: K4_L0 with 50K chunks. Optimized data pipeline.
"""

import os, sys, math, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDBlock, fibonacci_roots

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

D = 256
VOCAB = 50000
N_MODES = 4
INTERMEDIATE = 1024
LORA_RANK = 0
BATCH_SIZE = 8
SEQ_LEN = 128
LR = 3e-4
EPOCHS = 3
GRAD_CLIP = 1.0
LOG_EVERY = 200
N_CHUNKS_TRAIN = 50000
N_CHUNKS_EVAL = 500

TOKENIZER_PATH = 'C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/mlearning/eva_models/qwen3.5-0.8b'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
VOCAB = min(VOCAB, tokenizer.vocab_size)

# Model
cfg = LDConfig()
cfg.D = D
cfg.n_layers = 1
cfg.n_modes = N_MODES
cfg.vocab = VOCAB
cfg.intermediate = INTERMEDIATE
cfg.use_lora = False

lambdas = fibonacci_roots(N_MODES + 1)

class SingleLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.block = LDBlock(cfg, layer_idx=0, lambda_roots=lambdas)
        self.final_norm = nn.LayerNorm(D, eps=1e-6)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        h_out, _ = self.block(h, return_gates=True)
        h_normed = self.final_norm(h_out.float())
        return self.lm_head(h_normed)

model = SingleLayerModel().to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable: {trainable/1e3:.0f}K')

# ─── Fast data pipeline ────────────────────────────────────────────────
# Wikitext-103 has 1.1M rows, each ~1 sentence. We need ~50K rows for 50K chunks.
# Strategy: select first rows, filter empties, tokenize, chunk — all sequentially.
print('Loading wikitext-103 (first rows only)...')
ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')

# Take enough rows to produce N_CHUNKS_TRAIN chunks
n_rows_needed = N_CHUNKS_TRAIN + N_CHUNKS_EVAL + 150000
ds_small = ds.select(range(min(len(ds), n_rows_needed)))

# Filter empties and tokenize in one pass
def process(examples):
    texts = [t for t in examples['text'] if len(t.strip()) > 0]
    if not texts:
        return {'input_ids': []}
    enc = tokenizer(texts, truncation=True, max_length=SEQ_LEN + 1, padding=False)
    ids = [[min(i, VOCAB - 1) for i in row] for row in enc['input_ids']]
    # Build chunks on the fly
    all_ids = []
    for row in ids:
        all_ids.extend(row)
    chunks = []
    for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN // 2):
        chunk = all_ids[i:i + SEQ_LEN + 1]
        if len(chunk) == SEQ_LEN + 1:
            chunks.append(chunk)
    return {'chunk': chunks}

ds_chunked = ds_small.map(process, batched=True, remove_columns=['text'], batch_size=1000)
ds_chunked = ds_chunked.flatten()
total = len(ds_chunked)
print(f'Chunks from {n_rows_needed} source rows: {total}')

# Select
n_total = min(total, N_CHUNKS_TRAIN + N_CHUNKS_EVAL)
n_train = min(N_CHUNKS_TRAIN, n_total - N_CHUNKS_EVAL)
n_eval = min(N_CHUNKS_EVAL, n_total - n_train)
ds_sel = ds_chunked.select(range(n_total))
train_ds = ds_sel.select(range(n_train))
eval_ds = ds_sel.select(range(n_train, n_train + n_eval))
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
    train_ppl = math.exp(avg_loss)

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
    print(f'>> Epoch {epoch+1}: train_ppl={train_ppl:.1f}, eval_ppl={eval_ppl:.1f}')

    if eval_ppl < best_ppl:
        best_ppl = eval_ppl

print(f'\nTime: {time.perf_counter()-t0:.0f}s, Best eval PPL: {best_ppl:.1f}')
print('Done.')
