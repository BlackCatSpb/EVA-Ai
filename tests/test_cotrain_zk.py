"""Test: co-train stack + ZeckendorfReadout (200 steps)."""
import torch, numpy as np, time, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ld_model.core import LDConfig, LDStack
from ld_model.readout import ZeckendorfReadout

DEVICE = 'cuda'; D=896; VOCAB=50000; L=128
cfg = LDConfig(); cfg.D=D; cfg.n_layers=12; cfg.n_modes=4; cfg.vocab=VOCAB
cfg.bottleneck=256; cfg.adaptive_depth=True; cfg.learnable_V=True; cfg.V_rank=8

print('Loading checkpoint...')
t0 = time.time()
stack = LDStack(cfg).to(DEVICE)
zk = ZeckendorfReadout(cfg).to(DEVICE)
ckpt = torch.load('checkpoints/model_step25000.pt', map_location=DEVICE, weights_only=True)
sd = {k: v.float() if v.dtype==torch.float16 else v for k,v in ckpt['model_fp16'].items()}

stack_sd = {k.replace('stack.', ''): v for k,v in sd.items() if k.startswith('stack.')}
stack.load_state_dict(stack_sd, strict=False)

arr = np.load('russian_chunks.npy')
print(f'Loaded: {time.time()-t0:.1f}s')

embed = torch.nn.Embedding(VOCAB, D).to(DEVICE)
embed_sd = {k.replace('embed.', ''): v for k,v in sd.items() if k.startswith('embed.')}
embed.load_state_dict(embed_sd, strict=False)
del ckpt

# Use Zeckendorf to compute loss
def compute_loss(tokens):
    h = stack(embed(tokens))
    B, L, D = h.shape
    log_p = zk.log_probs_for_target(h.reshape(-1, D), tokens.reshape(-1))
    return -log_p.mean()

params = list(embed.parameters()) + list(stack.parameters()) + list(zk.parameters())
n_total = sum(p.numel() for p in params)
n_trainable = sum(p.numel() for p in params if p.requires_grad)
print(f'Params: {n_total:,} total, {n_trainable:,} trainable')
print(f'Zeckendorf: {sum(p.numel() for p in zk.parameters()):,}')

# Warmup
with torch.no_grad():
    x_w = torch.zeros(2, 8, dtype=torch.long, device=DEVICE)
    _ = compute_loss(x_w)

opt = torch.optim.AdamW(params, lr=5e-5)
N_STEPS = 200; B = 4
losses = []
t0 = time.time()

for step in range(1, N_STEPS + 1):
    idx = np.random.randint(0, len(arr), B)
    x = torch.from_numpy(arr[idx, :L].copy()).long().to(DEVICE)
    opt.zero_grad()
    loss = compute_loss(x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    losses.append(loss.item())

    if step % 20 == 0:
        avg = np.mean(losses[-20:])
        ppl = math.exp(avg)
        time_per = (time.time() - t0) / step
        print(f'  step {step:4d}/{N_STEPS} | loss={avg:.4f} | ppl={ppl:.0f} | {time_per:.1f}s/step')

print(f'\nDone: {time.time()-t0:.1f}s total')
print(f'Loss: {losses[0]:.4f} -> {np.mean(losses[-20:]):.4f}')
