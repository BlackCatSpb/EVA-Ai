"""Efficient ZeckendorfReadout training -- precompute hidden states first."""
import torch, numpy as np, time, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ld_model.core import LDConfig, LDStack
from ld_model.readout import ZeckendorfReadout

DEVICE = 'cuda'
D, N_LAYERS, VOCAB = 896, 12, 50000
B, L = 8, 64  # 512 tok/sample
N_SAMPLES = 100  # 100 x 512 = 51,200 tokens total
N_EPOCHS = 20
BATCH_TRAIN = 2048  # per gradient step
CKPT_PATH = 'checkpoints/model_step25000.pt'
SAVE_PATH = 'checkpoints/zeckendorf_trained.pt'

cfg = LDConfig()
for k, v in [('D', D), ('n_layers', N_LAYERS), ('n_modes', 4),
             ('vocab', VOCAB), ('bottleneck', 256)]:
    setattr(cfg, k, v)
cfg.adaptive_depth = True; cfg.learnable_V = True; cfg.V_rank = 8

# --- Build embed + stack (both frozen) ---------------------------------
print('Loading embedding + LDStack from checkpoint...')
embed = torch.nn.Embedding(VOCAB, D).to(DEVICE)
stack = LDStack(cfg).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
sd = {k: v.float() if v.dtype == torch.float16 else v
      for k, v in ckpt['model_fp16'].items()}
for module, name in [(embed, 'embed'), (stack, 'stack')]:
    mod_sd = {k.replace(f'{name}.', ''): v for k, v in sd.items()
              if k.startswith(f'{name}.') or k == name}
    compat = {k: v for k, v in mod_sd.items()
              if k in module.state_dict() and module.state_dict()[k].shape == v.shape}
    module.load_state_dict(compat, strict=False)
    module.requires_grad_(False)
    module.eval()

# Warmup
print('Warmup...')
with torch.no_grad():
    _ = stack(embed(torch.zeros(2, 8, dtype=torch.long, device=DEVICE)))
print('  done')

# --- Precompute hidden states ------------------------------------------
arr = np.load('russian_chunks.npy')
N = arr.shape[0]
print(f'Data: {N:,} chunks x {L} tok')

all_h = []
all_target = []
total_tokens = 0

for i in range(N_SAMPLES):
    idx = np.random.randint(0, N, B)
    x = torch.from_numpy(arr[idx].copy()).long().to(DEVICE)
    with torch.no_grad():
        h = stack(embed(x))
    all_h.append(h.cpu())
    all_target.append(x.cpu())
    total_tokens += B * L
    if (i + 1) % 40 == 0:
        print(f'  precompute {i+1}/{N_SAMPLES} ({total_tokens:,} tokens)')

h_all = torch.cat(all_h, dim=0).reshape(-1, D).float()  # (total_tokens, D)
target_all = torch.cat(all_target, dim=0).reshape(-1)    # (total_tokens,)
del embed, stack, all_h, all_target
print(f'Precomputed: {h_all.shape[0]:,} hidden states x {h_all.shape[1]}')

# --- Train ZeckendorfReadout -------------------------------------------
print('\nTraining ZeckendorfReadout...')
readout = ZeckendorfReadout(cfg).to(DEVICE)
opt = torch.optim.AdamW(readout.parameters(), lr=3e-3)

N_STEPS = (h_all.shape[0] // BATCH_TRAIN) * N_EPOCHS
step = 0
for epoch in range(N_EPOCHS):
    perm = torch.randperm(h_all.shape[0])
    h_perm = h_all[perm].to(DEVICE)
    t_perm = target_all[perm].to(DEVICE)

    for start in range(0, h_all.shape[0], BATCH_TRAIN):
        end = min(start + BATCH_TRAIN, h_all.shape[0])
        h_batch = h_perm[start:end]
        t_batch = t_perm[start:end]

        readout.train()
        opt.zero_grad()
        log_probs = readout.log_probs_for_target(h_batch, t_batch)
        loss = -log_probs.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
        opt.step()

        step += 1
        if step % 200 == 0:
            ppl = math.exp(loss.item())
            print(f'  step {step:5d} / ~{N_STEPS} | loss={loss.item():.4f} | ppl={ppl:.2f}')

torch.save(readout.state_dict(), SAVE_PATH)
print(f'Saved to {SAVE_PATH}')

# --- Compare with lm_head ----------------------------------------------
print('\n' + '='*60)
print('COMPARISON: Zeckendorf (trained) vs lm_head')
print('='*60)

# Load original model with lm_head
class OrigModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        self.stack2 = LDStack(cfg)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=False)
    def forward(self, x):
        return self.lm_head(self.stack2(self.embed(x)))

orig = OrigModel().to(DEVICE)
ckpt2 = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
sd2 = {k: v.float() if v.dtype==torch.float16 else v for k,v in ckpt2['model_fp16'].items()}
compat2 = {k: v for k,v in sd2.items() if k in orig.state_dict()
           and orig.state_dict()[k].shape == v.shape}
orig.load_state_dict(compat2, strict=False)
orig.eval()

# Eval: precompute a fresh eval set
print('Precomputing eval hiddens...')
idx_eval = np.random.randint(0, N, 16)
x_eval = torch.from_numpy(arr[idx_eval, :64].copy()).long().to(DEVICE)
with torch.no_grad():
    h_eval = orig.stack2(orig.embed(x_eval))
    h_eval_flat = h_eval.reshape(-1, D)
    logits_lm = orig.lm_head(h_eval_flat)
    probs_lm = torch.softmax(logits_lm.float(), dim=-1)

readout.eval()
with torch.no_grad():
    log_probs_zk = readout.forward_log_probs(h_eval_flat)
    probs_zk = torch.exp(log_probs_zk)
    probs_zk = probs_zk / probs_zk.sum(dim=-1, keepdim=True)

    V_common = min(probs_lm.shape[1], probs_zk.shape[1])

    # Top-k overlap
    top_lm = probs_lm[:, :V_common].topk(10).indices
    top_zk = probs_zk[:, :10].topk(10).indices
    overlap = 0
    for b in range(probs_lm.shape[0]):
        overlap += len(set(top_lm[b].tolist()) & set(top_zk[b].tolist()))
    overlap_avg = overlap / probs_lm.shape[0]

    # KL
    p_lm = probs_lm[:, :V_common].clamp(min=1e-30)
    p_zk = probs_zk[:, :V_common].clamp(min=1e-30)
    p_lm = p_lm / p_lm.sum(dim=-1, keepdim=True)
    p_zk = p_zk / p_zk.sum(dim=-1, keepdim=True)
    kl = (p_lm * (p_lm.log() - p_zk.log())).sum(dim=-1).mean().item()

    # P@1
    hit1 = (top_lm[:, 0] == top_zk[:, 0]).float().mean().item()

print(f'  Top-10 overlap:      {overlap_avg:.1f} / 10')
print(f'  Top-1 match (P@1):   {hit1:.3f}')
print(f'  KL(lm || zk):        {kl:.4f}')

# Compare losses on a batch
print('\nLoss comparison on 50 eval batches...')
eval_loss_lm = 0.0
eval_loss_zk = 0.0
for i in range(50):
    idx = np.random.randint(0, N, 8)
    x = torch.from_numpy(arr[idx, :128].copy()).long().to(DEVICE)
    with torch.no_grad():
        logits = orig(x)
        loss_lm = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1))
        eval_loss_lm += loss_lm.item()

        h = orig.stack2(orig.embed(x))
        log_p = readout.log_probs_for_target(h.reshape(-1, D), x.reshape(-1))
        loss_zk = -log_p.mean().item()
        eval_loss_zk += loss_zk
eval_loss_lm /= 50; eval_loss_zk /= 50

print(f'  lm_head loss:   {eval_loss_lm:.4f} (ppl={math.exp(eval_loss_lm):.2f})')
print(f'  Zeckendorf loss: {eval_loss_zk:.4f} (ppl={math.exp(eval_loss_zk):.2f})')

# Generation comparison
print('\nSample generations:')
readout.eval()
with torch.no_grad():
    idx = np.random.randint(0, N, 4)
    x = torch.from_numpy(arr[idx, :16].copy()).long().to(DEVICE)
    h = orig.stack2(orig.embed(x))
    h_flat = h.reshape(-1, D)
    tokens_zk = readout.predict(h_flat)
    logits_lm = orig.lm_head(h_flat)
    tokens_lm = logits_lm.argmax(dim=-1)
    for i in range(min(8, h_flat.shape[0])):
        match = 'OK' if tokens_lm[i]==tokens_zk[i] else 'XX'
        print(f'  [{i:2d}] lm={tokens_lm[i].item():5d}  zk={tokens_zk[i].item():5d}  {match}')

# Parameter count
zk_params = sum(p.numel() for p in readout.parameters())
lm_params = orig.lm_head.weight.numel()
print(f'\n  {"Readout params":20} {lm_params:>10,} (lm_head)  {zk_params:>10,} (Zeckendorf)')
print(f'  {"Reduction":20} {"":>10} {lm_params/zk_params:>10.0f}x')
