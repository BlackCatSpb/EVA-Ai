"""
Fast training from pre-cached numpy chunks.
Usage: python train_fast.py [N_chunks]
"""

import os, sys, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDBlock, fibonacci_roots

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

D = 256
VOCAB = 50000
N_MODES = 4
BATCH_SIZE = 8
SEQ_LEN = 128
LR = 3e-4
EPOCHS = 3
GRAD_CLIP = 1.0
LOG_EVERY = 200

N_CHUNKS_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
N_CHUNKS_EVAL = 500
print(f'Training chunks: {N_CHUNKS_TRAIN}')

# Load pre-cached data
print('Loading pre-cached chunks...')
t0 = time.perf_counter()
arr = np.load('wikitext_chunks.npy')  # (N, 129)
print(f'  Loaded {arr.shape[0]} chunks in {time.perf_counter()-t0:.1f}s')

n_total = min(arr.shape[0], N_CHUNKS_TRAIN + N_CHUNKS_EVAL)
n_train = min(N_CHUNKS_TRAIN, n_total - N_CHUNKS_EVAL)
n_eval = min(N_CHUNKS_EVAL, n_total - n_train)

train_ids = torch.tensor(arr[:n_train], dtype=torch.long)
eval_ids = torch.tensor(arr[n_train:n_train + n_eval], dtype=torch.long)
print(f'Train: {n_train}, Eval: {n_eval}')

# Pre-compute x and y
train_x = train_ids[:, :-1].to(DEVICE)
train_y = train_ids[:, 1:].to(DEVICE)
eval_x = eval_ids[:, :-1].to(DEVICE)
eval_y = eval_ids[:, 1:].to(DEVICE)

train_ds = TensorDataset(train_x, train_y)
eval_ds = TensorDataset(eval_x, eval_y)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

# Model
cfg = LDConfig()
cfg.D = D
cfg.n_layers = 1
cfg.n_modes = N_MODES
cfg.vocab = VOCAB
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

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * EPOCHS)

step = 0
best_ppl = float('inf')
t_start = time.perf_counter()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for bx, by in train_loader:
        optimizer.zero_grad()
        logits = model(bx)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
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

    train_ppl = math.exp(epoch_loss / n_batches)

    model.eval()
    eval_loss = 0.0
    for bx, by in eval_loader:
        logits = model(bx)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
        eval_loss += loss.item()
    eval_ppl = math.exp(eval_loss / len(eval_loader))

    print(f'>> Epoch {epoch+1}: train_ppl={train_ppl:.1f}, eval_ppl={eval_ppl:.1f}')
    if eval_ppl < best_ppl:
        best_ppl = eval_ppl

print(f'\nTime: {time.perf_counter()-t_start:.0f}s, Best eval PPL: {best_ppl:.1f}')
print('Done.')
