"""
Phase 2: 12-layer LDStack at D=896 (Qwen2.5-0.5B scale).
Frozen embedding + MLP + norms. Train only LDBlock gates + LoRA.
"""

import os, sys, math, time, json, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDBlock, LDMLP, fibonacci_roots

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─── Config ──────────────────────────────────────────────────────────────
D = 896
VOCAB = 50000
N_MODES = 4
N_LAYERS = 12
INTERMEDIATE = 4864  # Qwen2.5-0.5B
BATCH_SIZE = 4
SEQ_LEN = 128
LR = 1e-3
EPOCHS = 3
GRAD_CLIP = 1.0
LOG_EVERY = 100
N_CHUNKS_TRAIN = 50000
N_CHUNKS_EVAL = 500
CKPT_DIR = 'checkpoints'

# ─── Load data ───────────────────────────────────────────────────────────
print('Loading pre-cached chunks...')
t0 = time.perf_counter()
arr = np.load('wikitext_chunks.npy')
print(f'  Loaded {arr.shape[0]} chunks in {time.perf_counter()-t0:.1f}s')

n_total = min(arr.shape[0], N_CHUNKS_TRAIN + N_CHUNKS_EVAL)
n_train = min(N_CHUNKS_TRAIN, n_total - N_CHUNKS_EVAL)
n_eval = min(N_CHUNKS_EVAL, n_total - n_train)

train_ids = torch.tensor(arr[:n_train], dtype=torch.long)
eval_ids = torch.tensor(arr[n_train:n_train + n_eval], dtype=torch.long)
print(f'Train: {n_train} chunks, Eval: {n_eval} chunks')

train_x = train_ids[:, :-1].to(DEVICE)
train_y = train_ids[:, 1:].to(DEVICE)
eval_x = eval_ids[:, :-1].to(DEVICE)
eval_y = eval_ids[:, 1:].to(DEVICE)

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(TensorDataset(eval_x, eval_y), batch_size=BATCH_SIZE)

# ─── Model ───────────────────────────────────────────────────────────────
class Qwen2LDBlock(nn.Module):
    """Qwen2-like layer: LDBlock (internal norm+residual) → RMSNorm → SwiGLU MLP."""
    def __init__(self, d_model, intermediate, n_modes, lambda_roots, layer_idx, lora_rank=16):
        super().__init__()
        self.d_model = d_model
        # LDBlock handles its own pre-norm and residual
        cfg = LDConfig()
        cfg.D = d_model
        cfg.n_modes = n_modes
        cfg.vocab = VOCAB
        cfg.intermediate = intermediate
        cfg.lora_rank = lora_rank
        cfg.use_lora = lora_rank > 0
        self.ld = LDBlock(cfg, layer_idx, lambda_roots)
        # Post-attn norm
        self.post_norm = nn.RMSNorm(d_model, eps=1e-6)
        # MLP (SwiGLU with LoRA)
        self.mlp = LDMLP(cfg, use_lora=lora_rank > 0)

    def forward(self, x):
        delta = self.ld(x, residual=False)  # just delta from LDBlock
        h = x + delta                       # external residual
        r = self.post_norm(h)
        h = h + self.mlp(r)
        return h

class Phase2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        lambdas = fibonacci_roots(N_MODES + 1)
        self.layers = nn.ModuleList([
            Qwen2LDBlock(D, INTERMEDIATE, N_MODES, lambdas, i, lora_rank=16)
            for i in range(N_LAYERS)
        ])
        self.final_norm = nn.RMSNorm(D, eps=1e-6)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)
        self._freeze_non_gate_params()

    def _freeze_non_gate_params(self):
        n_frozen = 0
        n_trainable = 0
        for name, p in self.named_parameters():
            is_gate = 'W_gate' in name or 'b_gate' in name
            is_lora = 'A' in name or 'B' in name
            is_head = 'lm_head' in name
            is_norm = 'final_norm' in name or 'post_norm' in name
            if is_gate or is_lora or is_head or is_norm:
                p.requires_grad = True
                n_trainable += p.numel()
            else:
                p.requires_grad = False
                n_frozen += p.numel()
        print(f'Params: {n_trainable/1e6:.1f}M trainable, {n_frozen/1e6:.1f}M frozen '
              f'({n_trainable/(n_trainable+n_frozen)*100:.1f}% trainable)')

    def forward(self, input_ids):
        B, L = input_ids.shape
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        logits = self.lm_head(h)
        return logits

model = Phase2Model().to(DEVICE)

def save_checkpoint(path, model, optimizer, scheduler, step, epoch, best_ppl, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'epoch': epoch,
        'best_ppl': best_ppl,
        'stats': stats,
        'config': {
            'D': D, 'VOCAB': VOCAB, 'N_MODES': N_MODES,
            'N_LAYERS': N_LAYERS, 'INTERMEDIATE': INTERMEDIATE,
            'BATCH_SIZE': BATCH_SIZE, 'SEQ_LEN': SEQ_LEN,
            'LR': LR, 'EPOCHS': EPOCHS,
        }
    }, path)
    print(f'  [CKPT] Saved {path}')

def find_latest_ckpt():
    pattern = os.path.join(CKPT_DIR, 'phase2_epoch*.pt')
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt

if __name__ == '__main__':
    # Quick sanity check: trace per-layer before training
    model.eval()
    with torch.no_grad():
        bx_test = next(iter(train_loader))[0][:1]
        h = model.embed(bx_test)
        print(f'  sanity: embed range=[{h.min():.4f},{h.max():.4f}]', flush=True)
        for i, layer in enumerate(model.layers):
            h = layer(h)
            n, inf = torch.isnan(h).any().item(), torch.isinf(h).any().item()
            print(f'  sanity: layer {i} range=[{h.min():.4f},{h.max():.4f}] nan={n} inf={inf}', flush=True)
            if n or inf: break
        h = model.final_norm(h)
        logits = model.lm_head(h)
        print(f'  sanity: logits range=[{logits.min():.4f},{logits.max():.4f}] nan={torch.isnan(logits).any().item()}', flush=True)
    
    # ─── Training ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * EPOCHS)
    
    step = 0
    start_epoch = 0
    best_ppl = float('inf')
    t_start = time.perf_counter()
    
    # Try resume from checkpoint
    ckpt_path = find_latest_ckpt()
    if ckpt_path:
        print(f'Resuming from {ckpt_path}...')
        ckpt = load_checkpoint(ckpt_path, model, optimizer, scheduler)
        step = ckpt['step']
        start_epoch = ckpt['epoch']
        best_ppl = ckpt['best_ppl']
        print(f'  Resumed at step {step}, epoch {start_epoch}, best_ppl={best_ppl:.1f}')
    
    print(f'\nTraining: {len(train_loader)} batches/epoch, {EPOCHS} epochs')
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
    
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            
            # NaN detection
            if torch.isnan(logits).any():
                nan_frac = torch.isnan(logits).float().mean().item()
                print(f'  [NAN] logits contain {nan_frac*100:.1f}% NaN at step {step}')
                print(f'  [NAN] Skipping batch')
                continue
            
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
            
            if torch.isnan(loss):
                print(f'  [NAN] loss is NaN at step {step}, skipping')
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
    
            epoch_loss += loss.item()
            n_batches += 1
            step += 1
    
            if step % LOG_EVERY == 0:
                ppl = math.exp(loss.item())
                lr_now = scheduler.get_last_lr()[0]
                print(f'  Step {step:5d} | loss={loss.item():.4f} | ppl={ppl:.1f} | lr={lr_now:.2e}')
    
        if n_batches > 0:
            train_ppl = math.exp(epoch_loss / n_batches)
        else:
            train_ppl = float('inf')
            print(f'  [WARN] No valid batches in epoch {epoch+1}')
    
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for bx, by in eval_loader:
                logits = model(bx)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
                eval_loss += loss.item()
        eval_ppl = math.exp(eval_loss / len(eval_loader))
    
        print(f'>> Epoch {epoch+1}: train_ppl={train_ppl:.1f}, eval_ppl={eval_ppl:.1f}')
        is_best = eval_ppl < best_ppl
        if is_best:
            best_ppl = eval_ppl
            save_checkpoint(
                os.path.join(CKPT_DIR, 'phase2_best.pt'),
                model, optimizer, scheduler, step, epoch+1, best_ppl,
                {'train_ppl': train_ppl, 'eval_ppl': eval_ppl}
            )
        save_checkpoint(
            os.path.join(CKPT_DIR, f'phase2_epoch{epoch+1}.pt'),
            model, optimizer, scheduler, step, epoch+1, best_ppl,
            {'train_ppl': train_ppl, 'eval_ppl': eval_ppl, 'is_best': is_best}
        )
    
    print(f'\nTime: {time.perf_counter()-t_start:.0f}s')
    print(f'Best eval PPL: {best_ppl:.1f}')
    print('Phase 2 complete.')
