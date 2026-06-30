"""
Colab training for Phase 2 with causal conv + dense bottleneck MLP + grad accum + warmup.
Usage: python colab_train.py --ckpt_dir /path/to/checkpoints --batch_size 8
"""

import os, sys, math, time, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack

# ─── Config ──────────────────────────────────────────────────────────────
D = 896
VOCAB = 50000
N_MODES = 4
N_LAYERS = 12
SEQ_LEN = 128
LR = 1e-3
WARMUP_FRAC = 0.05
EPOCHS = 3
GRAD_CLIP = 1.0
LOG_EVERY = 100
N_CHUNKS_TRAIN = 50000
N_CHUNKS_EVAL = 500

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

    def forward(self, input_ids):
        h = self.embed(input_ids)
        h = self.stack(h)
        return self.lm_head(h)

# ─── Checkpoint utils ────────────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, step, epoch, best_ppl, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step, 'epoch': epoch, 'best_ppl': best_ppl, 'stats': stats,
    }, path)
    print(f'  [CKPT] {path}')

def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location='cuda', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt

def find_latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

# ─── Main ────────────────────────────────────────────────────────────────
def main(ckpt_dir='checkpoints', batch_size=8, n_train=50000):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')
    ACCUM_STEPS = max(1, 32 // batch_size)  # eff batch ≈ 32

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
        h = model.stack(h)
        print(f'  stack: [{h.min():.2f}, {h.max():.2f}] nan={torch.isnan(h).any().item()}')
        print(f'  logits: [{model.lm_head(h).min():.2f}, {model.lm_head(h).max():.2f}]')

    # Optimizer + LR schedule
    total_steps = (n_tr // batch_size) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def get_lr(step):
        if step < warmup_steps:
            return LR * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return LR * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Resume
    step, start_epoch, best_ppl = 0, 0, float('inf')
    ckpt = find_latest(os.path.join(ckpt_dir, 'phase2_epoch*.pt'))
    if ckpt:
        print(f'Resuming {ckpt}...')
        data = load_checkpoint(ckpt, model, optimizer)
        step, start_epoch, best_ppl = data['step'], data['epoch'], data['best_ppl']
        print(f'  step={step}, epoch={start_epoch}, best_ppl={best_ppl:.1f}')

    # Graceful interrupt
    import signal
    _ckpt_saved = [False]
    def _save_and_exit(signum, frame):
        if _ckpt_saved[0]: return
        _ckpt_saved[0] = True
        print('\n  [SIGINT] Saving checkpoint...')
        save_checkpoint(os.path.join(ckpt_dir, f'phase2_interrupt_step{step}.pt'),
                        model, optimizer, step, epoch+1, best_ppl,
                        {'note': 'interrupted'})
        sys.exit(0)
    signal.signal(signal.SIGINT, _save_and_exit)

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad()

        for bx, by in train_loader:
            logits = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
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

            if step > 0 and step % 1000 == 0:
                save_checkpoint(os.path.join(ckpt_dir, f'phase2_autosave_step{step}.pt'),
                                model, optimizer, step, epoch+1, best_ppl,
                                {'train_ppl': math.exp(epoch_loss / max(n_batches, 1))})

            if step % LOG_EVERY == 0:
                ppl = math.exp(epoch_loss / max(n_batches, 1))
                lr_now = optimizer.param_groups[0]['lr']
                print(f'  Step {step:5d} | loss={epoch_loss/max(n_batches,1):.4f} | ppl={ppl:.1f} | lr={lr_now:.2e}')

        if step % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()

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
                            model, optimizer, step, epoch+1, best_ppl,
                            {'train_ppl': train_ppl, 'eval_ppl': eval_ppl})
        save_checkpoint(os.path.join(ckpt_dir, f'phase2_epoch{epoch+1}.pt'),
                        model, optimizer, step, epoch+1, best_ppl,
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
