"""
Phase 2: 12-layer LDStack at D=896 with causal conv + dense bottleneck MLP.
Training: gradient accumulation (eff batch=32) + linear warmup (5%).
"""

import os, sys, math, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─── Config ──────────────────────────────────────────────────────────────
D = 896
VOCAB = 50000
N_MODES = 4
N_LAYERS = 12
BATCH_SIZE = 4
ACCUM_STEPS = 8          # effective batch = 32
SEQ_LEN = 128
LR = 1e-3
WARMUP_FRAC = 0.05
EPOCHS = 3
GRAD_CLIP = 1.0
LOG_EVERY = 100
CKPT_DIR = 'checkpoints'

# ─── Argparse ────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='wikitext',
                    choices=['wikitext', 'russian', 'auto'],
                    help='Dataset to use (wikitext, russian, or auto-detect)')
parser.add_argument('--train_chunks', type=int, default=None,
                    help='Number of training chunks (default: all available)')
parser.add_argument('--eval_chunks', type=int, default=500)
parser.add_argument('--epochs', type=int, default=EPOCHS)
args = parser.parse_args()
if args.epochs != EPOCHS:
    EPOCHS = args.epochs

# ─── Load data ───────────────────────────────────────────────────────────
def choose_data(data_choice):
    if data_choice == 'auto':
        if os.path.exists('russian_chunks.npy'):
            return 'russian_chunks.npy'
        return 'wikitext_chunks.npy'
    return {'wikitext': 'wikitext_chunks.npy', 'russian': 'russian_chunks.npy'}[data_choice]

data_file = choose_data(args.data)
print(f'Data file: {data_file}')
t0 = time.perf_counter()
arr = np.load(data_file)
print(f'  Loaded {arr.shape[0]} chunks in {time.perf_counter()-t0:.1f}s')

N_CHUNKS_TRAIN = args.train_chunks if args.train_chunks else arr.shape[0] - args.eval_chunks
N_CHUNKS_EVAL = min(args.eval_chunks, arr.shape[0] - 10000) if args.train_chunks is None else args.eval_chunks

n_total = min(arr.shape[0], N_CHUNKS_TRAIN + N_CHUNKS_EVAL)
n_train = min(N_CHUNKS_TRAIN, n_total - N_CHUNKS_EVAL)
n_eval = min(N_CHUNKS_EVAL, n_total - n_train)

train_ids = torch.tensor(arr[:n_train], dtype=torch.long)
eval_ids = torch.tensor(arr[n_train:n_train + n_eval], dtype=torch.long)
print(f'Train: {n_train} chunks ({n_train * SEQ_LEN / 1e6:.1f}M tok), '
      f'Eval: {n_eval} chunks ({n_eval * SEQ_LEN / 1e6:.1f}M tok)')

train_x = train_ids[:, :-1].to(DEVICE)
train_y = train_ids[:, 1:].to(DEVICE)
eval_x = eval_ids[:, :-1].to(DEVICE)
eval_y = eval_ids[:, 1:].to(DEVICE)

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(TensorDataset(eval_x, eval_y), batch_size=BATCH_SIZE)

# ─── Model ───────────────────────────────────────────────────────────────
class Phase2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        cfg = LDConfig()
        cfg.D = D
        cfg.n_layers = N_LAYERS
        cfg.n_modes = N_MODES
        cfg.vocab = VOCAB
        cfg.bottleneck = 256
        self.stack = LDStack(cfg)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)

    def forward(self, input_ids, return_gates=False):
        h = self.embed(input_ids)
        if return_gates:
            h, gates = self.stack(h, return_gates=True)
            return self.lm_head(h), gates
        h = self.stack(h)
        return self.lm_head(h)

model = Phase2Model().to(DEVICE)
n_all = sum(p.numel() for p in model.parameters())
n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model: {n_all/1e6:.1f}M params ({n_t/1e6:.1f}M trainable)')

# ─── Checkpoint helpers ──────────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler, step, epoch, best_ppl, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step, 'epoch': epoch, 'best_ppl': best_ppl, 'stats': stats,
        'config': {
            'D': D, 'VOCAB': VOCAB, 'N_MODES': N_MODES,
            'N_LAYERS': N_LAYERS, 'BATCH_SIZE': BATCH_SIZE,
            'ACCUM_STEPS': ACCUM_STEPS, 'SEQ_LEN': SEQ_LEN, 'LR': LR,
            'EPOCHS': EPOCHS,
        }
    }
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(ckpt, path)
    print(f'  [CKPT] Saved {path}')

def find_latest_ckpt():
    files = sorted(glob.glob(os.path.join(CKPT_DIR, 'phase2_*.pt')))
    return files[-1] if files else None

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt

# ─── Sanity ──────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    bx_test = next(iter(train_loader))[0][:1]
    h = model.embed(bx_test)
    print(f'  sanity: embed range=[{h.min():.4f},{h.max():.4f}]', flush=True)
    h = model.stack(h)
    n, inf = torch.isnan(h).any().item(), torch.isinf(h).any().item()
    print(f'  sanity: stack range=[{h.min():.4f},{h.max():.4f}] nan={n} inf={inf}', flush=True)
    logits = model.lm_head(h)
    print(f'  sanity: logits range=[{logits.min():.4f},{logits.max():.4f}]', flush=True)

# ─── Training ────────────────────────────────────────────────────────────
total_steps = (n_train // BATCH_SIZE) * EPOCHS
warmup_steps = int(total_steps * WARMUP_FRAC)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

def get_lr(step):
    if step < warmup_steps:
        return LR * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return LR * 0.5 * (1.0 + math.cos(math.pi * progress))

step = 0
start_epoch = 0
best_ppl = float('inf')
t_start = time.perf_counter()

# Resume
ckpt_path = find_latest_ckpt()
if ckpt_path:
    print(f'Resuming from {ckpt_path}...')
    ckpt = load_checkpoint(ckpt_path, model, optimizer)
    step = ckpt['step']
    start_epoch = ckpt['epoch']
    best_ppl = ckpt['best_ppl']
    print(f'  Resumed step={step}, epoch={start_epoch}, best_ppl={best_ppl:.1f}')

print(f'\nTraining: {n_train//BATCH_SIZE} steps/epoch, {EPOCHS} epochs')
print(f'  total_steps={total_steps}, warmup_steps={warmup_steps}')

for epoch in range(start_epoch, EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for bx, by in train_loader:
        logits = model(bx)

        if torch.isnan(logits).any():
            print(f'  [NAN] logits at step {step}, skipping')
            continue

        loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))

        if torch.isnan(loss):
            print(f'  [NAN] loss at step {step}, skipping')
            continue

        loss = loss / ACCUM_STEPS
        loss.backward()

        epoch_loss += loss.item() * ACCUM_STEPS
        n_batches += 1
        step += 1

        if step % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()

        if step % LOG_EVERY == 0:
            ppl = math.exp(epoch_loss / max(n_batches, 1))
            lr_now = optimizer.param_groups[0]['lr']
            print(f'  Step {step:5d} | loss={epoch_loss/max(n_batches,1):.4f} | ppl={ppl:.1f} | lr={lr_now:.2e}')

    # Flush accumulated gradients at epoch end
    if step % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()

    if n_batches > 0:
        train_ppl = math.exp(epoch_loss / n_batches)
    else:
        train_ppl = float('inf')
        print(f'  [WARN] No valid batches in epoch {epoch+1}')

    model.eval()
    eval_loss = 0.0
    all_entropies = []
    with torch.no_grad():
        for bx, by in eval_loader:
            logits, gates = model(bx, return_gates=True)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
            eval_loss += loss.item()
            # gates: (n_layers, B, L, K) → entropy per layer
            H = -(gates * (gates + 1e-10).log()).sum(dim=-1).mean(dim=(1, 2))
            all_entropies.append(H)
    eval_ppl = math.exp(eval_loss / len(eval_loader))
    avg_entropy = torch.stack(all_entropies).mean(dim=0).cpu().tolist()
    max_entropy = math.log(N_MODES)
    print(f'  Gate entropy per layer: {[f"{e:.3f}" for e in avg_entropy]}')
    print(f'  Max possible H={max_entropy:.3f}, mean H={sum(avg_entropy)/len(avg_entropy):.3f}')

    print(f'>> Epoch {epoch+1}: train_ppl={train_ppl:.1f}, eval_ppl={eval_ppl:.1f}')
    is_best = eval_ppl < best_ppl
    if is_best:
        best_ppl = eval_ppl
        save_checkpoint(os.path.join(CKPT_DIR, 'phase2_best.pt'),
                        model, optimizer, None, step, epoch+1, best_ppl,
                        {'train_ppl': train_ppl, 'eval_ppl': eval_ppl})
    save_checkpoint(os.path.join(CKPT_DIR, f'phase2_epoch{epoch+1}.pt'),
                    model, optimizer, None, step, epoch+1, best_ppl,
                    {'train_ppl': train_ppl, 'eval_ppl': eval_ppl, 'is_best': is_best})

print(f'\nTime: {time.perf_counter()-t_start:.0f}s')
print(f'Best eval PPL: {best_ppl:.1f}')
print('Phase 2 complete.')
