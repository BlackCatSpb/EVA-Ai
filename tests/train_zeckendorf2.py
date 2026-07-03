"""Full ZeckendorfReadout train + compare."""
import torch, numpy as np, time, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ld_model.core import LDConfig, LDStack
from ld_model.readout import ZeckendorfReadout

DEVICE = 'cuda'; D=896; VOCAB=50000
cfg = LDConfig(); cfg.D=D; cfg.n_layers=12; cfg.n_modes=4; cfg.vocab=VOCAB
cfg.bottleneck=256; cfg.adaptive_depth=True; cfg.learnable_V=True; cfg.V_rank=8

print('Loading stack...')
t0 = time.time()
embed = torch.nn.Embedding(VOCAB, D).to(DEVICE)
stack = LDStack(cfg).to(DEVICE)
ckpt = torch.load('checkpoints/model_step25000.pt', map_location=DEVICE, weights_only=True)
sd = {k: v.float() if v.dtype==torch.float16 else v for k,v in ckpt['model_fp16'].items()}
for module, name in [(embed, 'embed'), (stack, 'stack')]:
    mod_sd = {k.replace(f'{name}.', ''): v for k,v in sd.items() if k.startswith(f'{name}.')}
    module.load_state_dict(mod_sd, strict=False)
stack.requires_grad_(False); stack.eval()

print('Warmup + precompute...')
with torch.no_grad():
    _ = stack(embed(torch.zeros(2, 8, dtype=torch.long, device=DEVICE)))
arr = np.load('russian_chunks.npy')
all_h, all_t = [], []
for i in range(100):
    idx = np.random.randint(0, len(arr), 8)
    x = torch.from_numpy(arr[idx, :64].copy()).long().to(DEVICE)
    with torch.no_grad():
        h = stack(embed(x))
    all_h.append(h.cpu()); all_t.append(x.cpu())
h_all = torch.cat(all_h).reshape(-1, D).float()
t_all = torch.cat(all_t).reshape(-1)
del embed, stack
print(f'Precomputed {h_all.shape[0]:,} hiddens in {time.time()-t0:.1f}s')

print('Training ZeckendorfReadout...')
readout = ZeckendorfReadout(cfg).to(DEVICE)
opt = torch.optim.AdamW(readout.parameters(), lr=3e-3)
BATCH = 2048; EPOCHS = 20
N_STEPS = (h_all.shape[0] // BATCH) * EPOCHS
step = 0
for epoch in range(EPOCHS):
    perm = torch.randperm(h_all.shape[0])
    hp = h_all[perm].to(DEVICE); tp = t_all[perm].to(DEVICE)
    for start in range(0, h_all.shape[0], BATCH):
        end = min(start + BATCH, h_all.shape[0])
        opt.zero_grad()
        log_p = readout.log_probs_for_target(hp[start:end], tp[start:end])
        loss = -log_p.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
        opt.step()
        step += 1
    ppl = math.exp(loss.item())
    print(f'  epoch {epoch+1:2d}/{EPOCHS} | step {step:4d}/{N_STEPS} | loss={loss.item():.3f} | ppl={ppl:.0f}')

torch.save(readout.state_dict(), 'checkpoints/zeckendorf_trained.pt')
print(f'Done in {time.time()-t0:.1f}s')

# ---- Quick comparison ----
print('\n=== Comparison ===')
orig = torch.nn.Linear(D, VOCAB, bias=False).to(DEVICE)
orig_sd = {k.replace('lm_head.', ''): v for k,v in sd.items() if k.startswith('lm_head.')}
orig.load_state_dict(orig_sd)

embed2 = torch.nn.Embedding(VOCAB, D).to(DEVICE)
stack2 = LDStack(cfg).to(DEVICE)
for module, name in [(embed2, 'embed'), (stack2, 'stack')]:
    mod_sd = {k.replace(f'{name}.', ''): v for k,v in sd.items() if k.startswith(f'{name}.')}
    module.load_state_dict(mod_sd, strict=False)
stack2.requires_grad_(False); stack2.eval()

idx = np.random.randint(0, len(arr), 4)
x = torch.from_numpy(arr[idx, :32].copy()).long().to(DEVICE)
with torch.no_grad():
    h_fresh = stack2(embed2(x))
    h_flat = h_fresh.reshape(-1, D)
    logits_lm = orig(h_flat)
    probs_lm = torch.softmax(logits_lm.float(), dim=-1)
    loss_lm = torch.nn.functional.cross_entropy(logits_lm, x.reshape(-1)).item()
    log_p_zk = readout.forward_log_probs(h_flat)
    probs_zk = torch.exp(log_p_zk)
    probs_zk = probs_zk / probs_zk.sum(dim=-1, keepdim=True)
    loss_zk = -readout.log_probs_for_target(h_flat, x.reshape(-1)).mean().item()
    Vc = min(probs_lm.shape[1], probs_zk.shape[1])
    top_lm = probs_lm[:, :Vc].topk(10).indices
    top_zk = probs_zk[:, :10].topk(10).indices
    overlap = 0
    for b in range(probs_lm.shape[0]):
        overlap += len(set(top_lm[b].tolist()) & set(top_zk[b].tolist()))
    p_lm = probs_lm[:, :Vc].clamp(1e-30); p_zk = probs_zk[:, :Vc].clamp(1e-30)
    p_lm = p_lm / p_lm.sum(dim=-1, keepdim=True)
    p_zk = p_zk / p_zk.sum(dim=-1, keepdim=True)
    kl = (p_lm * (p_lm.log() - p_zk.log())).sum(dim=-1).mean().item()
    tokens_zk = readout.predict(h_flat[:8])
    tokens_lm = logits_lm[:8].argmax(dim=-1)
    matches = (tokens_zk == tokens_lm).sum().item()

print(f'  lm_head  loss={loss_lm:.4f}  ppl={math.exp(loss_lm):.1f}')
print(f'  Zeckendorf loss={loss_zk:.4f}  ppl={math.exp(loss_zk):.1f}')
print(f'  Top-10 overlap: {overlap:.0f}/{probs_lm.shape[0]*10}')
print(f'  KL(lm || zk):   {kl:.4f}')
print(f'  Token match: {matches}/8')
for i in range(min(8, h_flat.shape[0])):
    m = 'OK' if tokens_zk[i].item() == tokens_lm[i].item() else 'XX'
    print(f'    [{i}] lm={tokens_lm[i].item():5d}  zk={tokens_zk[i].item():5d} {m}')

zk_params = sum(p.numel() for p in readout.parameters())
lm_params = orig.weight.numel()
print(f'  lm_head params: {lm_params:,}  Zeckendorf: {zk_params:,} ({lm_params/zk_params:.0f}x)')
