"""Fresh start: train Phase2Model from scratch with all features.
Tests whether adaptive_depth + learnable_V produce balanced layer usage
when trained from step 0 (vs loaded old checkpoint)."""

import os, sys, math, time, glob, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

D = 896; VOCAB = 50000; N_MODES = 4; N_LAYERS = 12
BATCH_SIZE = 4; ACCUM_STEPS = 4; SEQ_LEN = 64
EFF_BATCH = BATCH_SIZE * ACCUM_STEPS
LR = 1e-3; WARMUP_FRAC = 0.05; EPOCHS = 1; GRAD_CLIP = 1.0
TOTAL_STEPS = 5000; LOG_EVERY = 100; CKPT_EVERY = 100

# ─── Data: first N chunks from russian ───────────────────────────────────
arr = np.load('russian_chunks.npy')
N_CHUNKS = 20000  # enough for 5000 steps
ids = torch.tensor(arr[:N_CHUNKS], dtype=torch.long)
x = ids[:, :-1].to(DEVICE)
y = ids[:, 1:].to(DEVICE)
loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)
print(f'Data: {N_CHUNKS} chunks of {SEQ_LEN}, {N_CHUNKS*SEQ_LEN/1e6:.1f}M tok')

# ─── Model (fresh, all features enabled) ─────────────────────────────────
class Phase2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        cfg = LDConfig()
        cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = VOCAB; cfg.bottleneck = 256
        self.stack = LDStack(cfg)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=False)
    def forward(self, input_ids):
        return self.lm_head(self.stack(self.embed(input_ids)))

model = Phase2Model().to(DEVICE)
n_all = sum(p.numel() for p in model.parameters())
n_cayley = sum(p.numel() for n, p in model.named_parameters() if 'V_cay' in n)
n_depth = sum(p.numel() for p in model.stack.depth_logits) if model.stack.depth_logits is not None else 0
print(f'Model: {n_all/1e6:.1f}M params | Cayley: {n_cayley:,} | depth_logits: {n_depth}')

# ─── Sanity ──────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    bx = next(iter(loader))[0][:1]
    h = model.embed(bx)
    h = model.stack(h)
    logits = model.lm_head(h)
    print(f'sanity: stack=[{h.min():.2f},{h.max():.2f}] logits=[{logits.min():.2f},{logits.max():.2f}]')

# ─── Training ────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

def get_lr(step):
    if step < TOTAL_STEPS * WARMUP_FRAC:
        return LR * (step + 1) / max(TOTAL_STEPS * WARMUP_FRAC, 1)
    progress = (step - TOTAL_STEPS * WARMUP_FRAC) / max(TOTAL_STEPS - TOTAL_STEPS * WARMUP_FRAC, 1)
    return LR * 0.5 * (1.0 + math.cos(math.pi * progress))

step = 0; n_batches = 0; epoch_loss = 0.0
optimizer.zero_grad()
t0 = time.perf_counter()

print(f'\nTraining: {TOTAL_STEPS} steps, B={BATCH_SIZE}, accum={ACCUM_STEPS} (eff={EFF_BATCH})')
print(f'  warmup={int(TOTAL_STEPS*WARMUP_FRAC)} steps')

while step < TOTAL_STEPS:
    for bx, by in loader:
        if step >= TOTAL_STEPS: break
        logits = model(bx)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
        loss = loss / ACCUM_STEPS
        loss.backward()
        epoch_loss += loss.item() * ACCUM_STEPS
        n_batches += 1
        step += 1

        if step % ACCUM_STEPS == 0 or step == TOTAL_STEPS:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            lr = get_lr(step)
            for g in optimizer.param_groups: g['lr'] = lr
            optimizer.step(); optimizer.zero_grad()

        if step % LOG_EVERY == 0:
            ppl = math.exp(epoch_loss / max(n_batches, 1))
            norms = [l.V_cay_A.norm().item() + l.V_cay_B.norm().item()
                     for l in model.stack.layers if l.V_cay_A is not None]
            depth_vals = [f'{torch.sigmoid(d).item():.2f}' for d in model.stack.depth_logits] if model.stack.depth_logits is not None else []
            v = f' |A+B|≈[{", ".join(f"{n:.2f}" for n in norms[:4])}...]' if norms else ''
            if depth_vals: v += f' depth≈[{", ".join(depth_vals)}]'
            print(f'  Step {step:4d} | loss={epoch_loss/max(n_batches,1):.4f} | ppl={ppl:.1f} | lr={lr:.2e}{v}')

        if step % CKPT_EVERY == 0:
            ckpt_path = f'checkpoints/fresh_start_{step}.pt'
            torch.save({'model_state_dict': model.state_dict(), 'step': step, 'loss': epoch_loss / max(n_batches,1)}, ckpt_path)

# ─── Final evaluation ───────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    bx_eval, by_eval = next(iter(loader))
    logits, gates = model(bx_eval), None
    # We need gates: recreate forward with return_gates
    # Quick hack: use the model's stack directly
    h = model.embed(bx_eval)
    h, gates = model.stack(h, return_gates=True)
    logits = model.lm_head(h)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), by_eval.reshape(-1))
    print(f'\n>> Final: PPL={math.exp(loss.item()):.1f}, Loss={loss.item():.4f}')

    # Pass rates
    gates_np = gates.cpu().numpy()
    depth_vals = [torch.sigmoid(d).item() for d in model.stack.depth_logits] if model.stack.depth_logits is not None else []
    print(f'\n  Layer | Gate entropy | Spread | Pass% | threshold')
    for lidx in range(N_LAYERS):
        gate_avg = gates_np[lidx].mean(axis=(0, 1))
        H = -(gate_avg * np.log(gate_avg + 1e-10)).sum()
        spread = gates_np[lidx].std(axis=-1).mean()
        if lidx < N_LAYERS - 1 and depth_vals:
            th = depth_vals[lidx]
            pr = (gates_np[lidx].std(axis=-1).mean(axis=-1) > th).mean()
        else:
            th, pr = 0, 1.0
        print(f'  L{lidx:>3} | {H:.3f} | {spread:.4f} | {pr*100:>4.0f}% | {th:.2f}')

print(f'\nTime: {time.perf_counter()-t0:.0f}s')
print('Fresh start test complete.')

# Save checkpoint for analyzer
torch.save({'model_state_dict': model.state_dict(), 'step': step},
           'checkpoints/fresh_start_test.pt')
print('Saved checkpoints/fresh_start_test.pt')
