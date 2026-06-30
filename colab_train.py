"""
Colab training for Phase 2.
Usage: python colab_train.py --ckpt_dir /path/to/checkpoints
"""
import os, sys, math, time, json, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDBlock, LDMLP, fibonacci_roots

# ─── Config ──────────────────────────────────────────────────────────────
D = 896
VOCAB = 50000
N_MODES = 4
N_LAYERS = 12
INTERMEDIATE = 4864
SEQ_LEN = 128
LR = 1e-3
EPOCHS = 3
GRAD_CLIP = 1.0
LOG_EVERY = 100
N_CHUNKS_TRAIN = 50000
N_CHUNKS_EVAL = 500

# ─── Model (copied from train_phase2.py) ─────────────────────────────────
class Qwen2LDBlock(nn.Module):
    def __init__(self, d_model, intermediate, n_modes, lambda_roots, layer_idx, lora_rank=16):
        super().__init__()
        cfg = LDConfig()
        cfg.D = d_model
        cfg.n_modes = n_modes
        cfg.vocab = VOCAB
        cfg.intermediate = intermediate
        cfg.lora_rank = lora_rank
        cfg.use_lora = lora_rank > 0
        self.ld = LDBlock(cfg, layer_idx, lambda_roots)
        self.post_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.mlp = LDMLP(cfg, use_lora=lora_rank > 0)

    def forward(self, x):
        delta = self.ld(x, residual=False)
        h = x + delta
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
        for name, p in self.named_parameters():
            is_gate = 'W_gate' in name or 'b_gate' in name
            is_lora = 'A' in name or 'B' in name
            is_head = 'lm_head' in name
            is_norm = 'final_norm' in name or 'post_norm' in name
            p.requires_grad = bool(is_gate or is_lora or is_head or is_norm)

    def forward(self, input_ids):
        B, L = input_ids.shape
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.lm_head(h)

# ─── Checkpoint utils ────────────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler, step, epoch, best_ppl, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step, 'epoch': epoch, 'best_ppl': best_ppl, 'stats': stats,
    }, path)
    print(f'  [CKPT] {path}')

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location='cuda', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt

def find_latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

# ─── Main ────────────────────────────────────────────────────────────────
def main(ckpt_dir='checkpoints', batch_size=8, n_train=50000):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')

    # Load data
    print('Loading chunks...')
    if not os.path.exists('wikitext_chunks.npy'):
        print('Generating chunks from HuggingFace...')
        from datasets import load_dataset
        from transformers import AutoTokenizer
        ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
        text = '\n\n'.join(ds['text'])
        tokens = tokenizer(text)['input_ids']
        n = len(tokens) // SEQ_LEN
        arr = np.array(tokens[:n * SEQ_LEN], dtype=np.int32).reshape(n, SEQ_LEN)
        np.save('wikitext_chunks.npy', arr)
        print(f'  Created {arr.shape[0]} chunks')
    else:
        arr = np.load('wikitext_chunks.npy')
    print(f'  Total chunks: {arr.shape[0]}')

    n_total = min(arr.shape[0], n_train + N_CHUNKS_EVAL)
    n_tr = min(n_train, n_total - N_CHUNKS_EVAL)
    n_ev = min(N_CHUNKS_EVAL, n_total - n_tr)

    train_ids = torch.tensor(arr[:n_tr], dtype=torch.long)
    eval_ids = torch.tensor(arr[n_tr:n_tr + n_ev], dtype=torch.long)
    train_loader = DataLoader(TensorDataset(train_ids[:, :-1].to(DEVICE),
                                            train_ids[:, 1:].to(DEVICE)),
                              batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(eval_ids[:, :-1].to(DEVICE),
                                           eval_ids[:, 1:].to(DEVICE)),
                             batch_size=batch_size)

    # Model
    model = Phase2Model().to(DEVICE)

    # Sanity
    model.eval()
    with torch.no_grad():
        bx = next(iter(train_loader))[0][:1]
        h = model.embed(bx)
        print(f'  embed: [{h.min():.2f}, {h.max():.2f}]')
        for i, layer in enumerate(model.layers):
            h = layer(h)
            n, inf = torch.isnan(h).any().item(), torch.isinf(h).any().item()
            print(f'  layer {i}: [{h.min():.2f}, {h.max():.2f}] nan={n}')
            if n or inf: break
        print(f'  logits: [{model.lm_head(model.final_norm(h)).min():.2f},'
              f'{model.lm_head(model.final_norm(h)).max():.2f}]')

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * EPOCHS)

    # Resume
    step, start_epoch, best_ppl = 0, 0, float('inf')
    ckpt = find_latest(os.path.join(ckpt_dir, 'phase2_epoch*.pt'))
    if ckpt:
        print(f'Resuming {ckpt}...')
        data = load_checkpoint(ckpt, model, optimizer, scheduler)
        step, start_epoch, best_ppl = data['step'], data['epoch'], data['best_ppl']
        print(f'  step={step}, epoch={start_epoch}, best_ppl={best_ppl:.1f}')

    # Graceful interrupt: save on Ctrl+C / Colab stop
    import signal
    _ckpt_saved = [False]
    def _save_and_exit(signum, frame):
        if _ckpt_saved[0]:
            return
        _ckpt_saved[0] = True
        print('\n  [SIGINT] Saving checkpoint...')
        save_checkpoint(
            os.path.join(ckpt_dir, f'phase2_interrupt_step{step}.pt'),
            model, optimizer, scheduler, step, epoch+1, best_ppl,
            {'note': 'interrupted'})
        print('  [SIGINT] Saved. Resume by re-running cell 5.')
        sys.exit(0)
    signal.signal(signal.SIGINT, _save_and_exit)

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            # Periodic save every 1000 steps
            if step > 0 and step % 1000 == 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f'phase2_autosave_step{step}.pt'),
                    model, optimizer, scheduler, step, epoch+1, best_ppl,
                    {'train_ppl': math.exp(epoch_loss / max(n_batches, 1))})

            if step % LOG_EVERY == 0:
                ppl = math.exp(loss.item())
                lr_now = scheduler.get_last_lr()[0]
                print(f'  Step {step:5d} | loss={loss.item():.4f} | ppl={ppl:.1f} | lr={lr_now:.2e}')

        train_ppl = math.exp(epoch_loss / n_batches) if n_batches else float('inf')

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for bx, by in eval_loader:
                loss = F.cross_entropy(model(bx).reshape(-1, VOCAB), by.reshape(-1))
                eval_loss += loss.item()
        eval_ppl = math.exp(eval_loss / len(eval_loader))

        print(f'>> Epoch {epoch+1}: train_ppl={train_ppl:.1f}, eval_ppl={eval_ppl:.1f}')
        is_best = eval_ppl < best_ppl
        if is_best:
            best_ppl = eval_ppl
            save_checkpoint(os.path.join(ckpt_dir, 'phase2_best.pt'),
                           model, optimizer, scheduler, step, epoch+1, best_ppl,
                           {'train_ppl': train_ppl, 'eval_ppl': eval_ppl})
        save_checkpoint(os.path.join(ckpt_dir, f'phase2_epoch{epoch+1}.pt'),
                       model, optimizer, scheduler, step, epoch+1, best_ppl,
                       {'train_ppl': train_ppl, 'eval_ppl': eval_ppl})

    print(f'Best eval PPL: {best_ppl:.1f}')
    return best_ppl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_train', type=int, default=50000)
    args = parser.parse_args()
    main(args.ckpt_dir, args.batch_size, args.n_train)
