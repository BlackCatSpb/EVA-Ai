"""
Phase 2-ZK: 12-layer LDStack at D=896 with ZeckendorfReadout instead of lm_head.
Continues from phase2 checkpoint or trains from scratch.
"""

import os, sys, math, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack
from ld_model.readout import ZeckendorfReadout

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─── Config ──────────────────────────────────────────────────────────────
D = 896
VOCAB = 50000
N_MODES = 4
N_LAYERS = 12
BATCH_SIZE = 4
ACCUM_STEPS = 8
SEQ_LEN = 128
LR = 1e-3
WARMUP_FRAC = 0.05
EPOCHS = 3
GRAD_CLIP = 1.0
LOG_EVERY = 100
CKPT_DIR = 'checkpoints_zk'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='russian', choices=['wikitext', 'russian', 'auto'])
parser.add_argument('--train_chunks', type=int, default=None)
parser.add_argument('--eval_chunks', type=int, default=500)
parser.add_argument('--epochs', type=int, default=EPOCHS)
parser.add_argument('--resume', type=str, default=None,
                    help='Phase2 checkpoint to resume from (loads stack + embed, resets readout)')
args = parser.parse_args()
if args.epochs != EPOCHS:
    EPOCHS = args.epochs

# ─── Data ────────────────────────────────────────────────────────────────
def choose_data(data_choice):
    if data_choice == 'auto':
        for fn in ['russian_chunks.npy', 'wikitext_chunks.npy']:
            if os.path.exists(fn):
                return fn
        raise FileNotFoundError('No data file found')
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
class Phase2ZKModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        cfg = LDConfig()
        cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = VOCAB; cfg.bottleneck = 256
        self.stack = LDStack(cfg)
        self.readout = ZeckendorfReadout(cfg)

    def forward(self, input_ids):
        h = self.stack(self.embed(input_ids))
        return h

    def compute_loss(self, input_ids, target_ids):
        h = self.forward(input_ids)
        B, L, D = h.shape
        log_p = self.readout.log_probs_for_target(h.reshape(-1, D), target_ids.reshape(-1))
        return -log_p.mean()

model = Phase2ZKModel().to(DEVICE)

# Resume from Phase2 checkpoint (loads embed + stack, resets Zeckendorf)
if args.resume:
    print(f'Loading Phase2 checkpoint: {args.resume}')
    ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=True)
    sd = ckpt.get('model_state_dict', ckpt.get('model_fp16', ckpt))
    # Filter only embed.* and stack.* keys
    load_sd = model.state_dict()
    for k in list(sd.keys()):
        new_k = k
        if k in load_sd and load_sd[k].shape == sd[k].shape:
            load_sd[k] = sd[k].float() if sd[k].dtype == torch.float16 else sd[k]
    model.load_state_dict(load_sd, strict=False)
    print(f'  Loaded embed + stack from checkpoint (Zeckendorf freshly initialized)')

n_all = sum(p.numel() for p in model.parameters())
n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_zk = sum(p.numel() for p in model.readout.parameters())
n_stack = n_all - n_zk - model.embed.weight.numel()
print(f'Model: {n_all/1e6:.1f}M params ({n_t/1e6:.1f}M trainable)')
print(f'  Stack: {n_stack/1e6:.1f}M, Embed: {model.embed.weight.numel()/1e6:.1f}M, Zeckendorf: {n_zk:,}')

# ─── Sanity ──────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    bx_test = next(iter(train_loader))[0][:1]
    h = model.embed(bx_test)
    print(f'  sanity: embed range=[{h.min():.4f},{h.max():.4f}]', flush=True)
    h = model.stack(h)
    n, inf = torch.isnan(h).any().item(), torch.isinf(h).any().item()
    print(f'  sanity: stack range=[{h.min():.4f},{h.max():.4f}] nan={n} inf={inf}', flush=True)
    log_p = model.readout.log_probs_for_target(h.reshape(-1, D), bx_test.reshape(-1))
    print(f'  sanity: log_p mean={log_p.mean():.4f} nan={torch.isnan(log_p).any()}', flush=True)

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

# Resume from ZK checkpoint
os.makedirs(CKPT_DIR, exist_ok=True)
ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, 'phase2zk_*.pt')))
if ckpt_files:
    ckpt_path = ckpt_files[-1]
    print(f'Resuming from {ckpt_path}...')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    step = ckpt['step']; start_epoch = ckpt['epoch']; best_ppl = ckpt['best_ppl']
    print(f'  Resumed step={step}, epoch={start_epoch}, best_ppl={best_ppl:.1f}')

print(f'\nTraining: {n_train//BATCH_SIZE} steps/epoch, {EPOCHS} epochs')
print(f'  total_steps={total_steps}, warmup_steps={warmup_steps}')

for epoch in range(start_epoch, EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for bx, by in train_loader:
        log_p = model.readout.log_probs_for_target(
            model.stack(model.embed(bx)).reshape(-1, D), by.reshape(-1))
        loss = -log_p.mean()

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
            v_info = ''
            if model.stack.cfg.learnable_V:
                norms = [l.V_cay_A.norm().item() + l.V_cay_B.norm().item()
                         for l in model.stack.layers if l.V_cay_A is not None]
                v_info = f' |A+B|≈[{", ".join(f"{n:.2f}" for n in norms)}]'
            print(f'  Step {step:5d} | loss={epoch_loss/max(n_batches,1):.4f} | ppl={ppl:.1f} | lr={lr_now:.2e}{v_info}')

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
    with torch.no_grad():
        for bx, by in eval_loader:
            log_p = model.readout.log_probs_for_target(
                model.stack(model.embed(bx)).reshape(-1, D), by.reshape(-1))
            eval_loss += -log_p.mean().item()
    eval_ppl = math.exp(eval_loss / len(eval_loader))

    print(f'>> Epoch {epoch+1}: train_ppl={train_ppl:.1f}, eval_ppl={eval_ppl:.1f}')
    is_best = eval_ppl < best_ppl
    if is_best:
        best_ppl = eval_ppl

    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step, 'epoch': epoch+1, 'best_ppl': best_ppl,
        'train_ppl': train_ppl, 'eval_ppl': eval_ppl,
        'config': {'D': D, 'VOCAB': VOCAB, 'N_LAYERS': N_LAYERS, 'LR': LR},
    }
    torch.save(ckpt, os.path.join(CKPT_DIR, f'phase2zk_epoch{epoch+1}.pt'))
    if is_best:
        torch.save(ckpt, os.path.join(CKPT_DIR, 'phase2zk_best.pt'))
    print(f'  [CKPT] Saved to {CKPT_DIR}/phase2zk_epoch{epoch+1}.pt')

print(f'\nTime: {time.perf_counter()-t_start:.0f}s')
print(f'Best eval PPL: {best_ppl:.1f}')
print('Phase 2-ZK complete.')
