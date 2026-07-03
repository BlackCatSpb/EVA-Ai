"""Hybrid: learned per-token routing between Zeckendorf and lm_head."""

import torch, numpy as np, time, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ld_model.core import LDConfig, LDStack
from ld_model.readout import ZeckendorfReadout

DEVICE = 'cuda'; D=896; VOCAB=50000
cfg = LDConfig(); cfg.D=D; cfg.n_layers=12; cfg.n_modes=4; cfg.vocab=VOCAB
cfg.bottleneck=256; cfg.adaptive_depth=True; cfg.learnable_V=True; cfg.V_rank=8

# --- Precompute hiddens (once) -----------------------------------------
print('Loading...')
t0 = time.time()
embed = torch.nn.Embedding(VOCAB, D).to(DEVICE)
stack = LDStack(cfg).to(DEVICE)
ckpt = torch.load('checkpoints/model_step25000.pt', map_location=DEVICE, weights_only=True)
sd = {k: v.float() if v.dtype==torch.float16 else v for k,v in ckpt['model_fp16'].items()}
for module, name in [(embed, 'embed'), (stack, 'stack')]:
    mod_sd = {k.replace(f'{name}.', ''): v for k,v in sd.items() if k.startswith(f'{name}.')}
    module.load_state_dict(mod_sd, strict=False)
stack.requires_grad_(False); stack.eval()

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
H = torch.cat(all_h).reshape(-1, D).float()
T = torch.cat(all_t).reshape(-1)
del embed, stack
N = H.shape[0]
print(f'Precomputed {N:,} hiddens in {time.time()-t0:.1f}s')

# --- Build components ---------------------------------------------------
print('Building hybrid: Zeckendorf + lm_head + router...')

# ZeckendorfReadout
zk = ZeckendorfReadout(cfg).to(DEVICE)

# lm_head (frozen)
lm = torch.nn.Linear(D, VOCAB, bias=False).to(DEVICE)
lm_w = {k.replace('lm_head.', ''): v for k,v in sd.items() if k.startswith('lm_head.')}
lm.load_state_dict(lm_w)
for p in lm.parameters():
    p.requires_grad = False

# Router: MLP D → 2 → α
router = torch.nn.Sequential(
    torch.nn.Linear(D, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
).to(DEVICE)

# Optimizer: train zk + router (lm frozen)
params = list(zk.parameters()) + list(router.parameters())
opt = torch.optim.AdamW(params, lr=3e-3)
print(f'  Trainable params: {sum(p.numel() for p in params):,}')

# --- Precompute lm_head logits (frozen, deterministic) -----------------
print('Precomputing lm_head logits (frozen)...')
t0 = time.time()
# Process in chunks to avoid OOM
chunk_size = 8192
logits_lm_all = []
lm.eval()
with torch.no_grad():
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        hb = H[start:end].to(DEVICE)
        logits_lm_all.append(lm(hb).cpu())
logits_lm_all = torch.cat(logits_lm_all, dim=0)  # (N, V)
log_p_lm_all = logits_lm_all.log_softmax(dim=-1)  # (N, V)
logits_lm_for_target = log_p_lm_all[torch.arange(N), T]  # (N,)
print(f'  done in {time.time()-t0:.1f}s')

# --- Training -----------------------------------------------------------
print('Training...')
BATCH = 2048; EPOCHS = 20
N_STEPS = (N // BATCH) * EPOCHS
step = 0
all_alphas = []

for epoch in range(EPOCHS):
    perm = torch.randperm(N)
    hp = H[perm].to(DEVICE)
    tp = T[perm].to(DEVICE)
    lp = logits_lm_for_target[perm].to(DEVICE)

    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        hb = hp[start:end]; tb = tp[start:end]; lb = lp[start:end]

        opt.zero_grad()

        # Zeckendorf log P(target)
        logit_zk = zk.log_probs_for_target(hb, tb)  # (B,)
        logit_lm = lb  # precomputed

        # Router α
        alpha = torch.sigmoid(router(hb)).squeeze(-1)  # (B,)

        # Log-mixed probability
        log_mixed = torch.logaddexp(logit_zk + torch.log(alpha + 1e-8),
                                    logit_lm + torch.log(1 - alpha + 1e-8))
        loss = -log_mixed.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % 100 == 0:
            with torch.no_grad():
                avg_a = alpha.mean().item()
                zk_win = (logit_zk > logit_lm).float().mean().item()
            ppl = math.exp(loss.item())
            print(f'  step {step:5d}/{N_STEPS} | loss={loss.item():.4f} ppl={ppl:.0f} | avg_α={avg_a:.3f} zk_better={zk_win*100:.0f}%')
        step += 1

    with torch.no_grad():
        all_alphas.append(torch.sigmoid(router(hp[:BATCH])).squeeze(-1).cpu())

all_alphas = torch.cat(all_alphas) if all_alphas else torch.tensor([0.0])
print(f'\nFinal α distribution: mean={all_alphas.mean():.3f} std={all_alphas.std():.3f} '
      f'median={all_alphas.median():.3f}')

# --- Analysis -----------------------------------------------------------
print('\n' + '='*60)
print('ANALYSIS')
print('='*60)

with torch.no_grad():
    # Eval on fresh data
    idx = np.random.randint(0, len(arr), 4)
    x = torch.from_numpy(arr[idx, :32].copy()).long().to(DEVICE)

    embed2 = torch.nn.Embedding(VOCAB, D).to(DEVICE)
    stack2 = LDStack(cfg).to(DEVICE)
    for module, name in [(embed2, 'embed'), (stack2, 'stack')]:
        mod_sd = {k.replace(f'{name}.', ''): v for k,v in sd.items() if k.startswith(f'{name}.')}
        module.load_state_dict(mod_sd, strict=False)
    stack2.eval()

    h_eval = stack2(embed2(x))
    h_flat = h_eval.reshape(-1, D)
    t_flat = x.reshape(-1)

    # All three losses
    logits_lm_eval = lm(h_flat)
    loss_lm = torch.nn.functional.cross_entropy(logits_lm_eval, t_flat).item()

    log_p_zk_eval = zk.log_probs_for_target(h_flat, t_flat)
    loss_zk = -log_p_zk_eval.mean().item()

    log_p_lm_eval = logits_lm_eval.log_softmax(dim=-1)
    alpha_eval = torch.sigmoid(router(h_flat)).squeeze(-1)
    logit_lm_eval = log_p_lm_eval[range(len(t_flat)), t_flat]
    log_mixed_eval = torch.logaddexp(log_p_zk_eval + torch.log(alpha_eval + 1e-8),
                                      logit_lm_eval + torch.log(1 - alpha_eval + 1e-8))
    loss_mixed = -log_mixed_eval.mean().item()

    print(f'  {"Model":<25} {"Loss":<10} {"PPL":<10} {"α mean":<10}')
    print(f'  {"-"*55}')
    print(f'  {"lm_head only":<25} {loss_lm:<10.4f} {math.exp(loss_lm):<10.1f} {"N/A":<10}')
    print(f'  {"Zeckendorf only":<25} {loss_zk:<10.4f} {math.exp(loss_zk):<10.1f} {"N/A":<10}')
    print(f'  {"Hybrid (routed)":<25} {loss_mixed:<10.4f} {math.exp(loss_mixed):<10.1f} {alpha_eval.mean():<10.3f}')

    # When does router prefer Zeckendorf?
    zk_preferred = (alpha_eval > 0.5).sum().item()
    total = len(alpha_eval)
    print(f'\n  Router prefers Zeckendorf: {zk_preferred}/{total} '
          f'({zk_preferred/total*100:.0f}%)')

    # Gate spread correlation
    # Re-use existing adaptive depth gates from the stack2 forward
    # We can't easily extract gate spread without modifying forward()

    # Token match analysis
    zk_tokens = zk.predict(h_flat)
    lm_tokens = logits_lm_eval.argmax(dim=-1)

    zk_correct = (zk_tokens == t_flat).float().mean().item()
    lm_correct = (lm_tokens == t_flat).float().mean().item()

    # For tokens where router chose Zeckendorf, how often does ZK beat LM?
    zk_win = 0
    lm_win = 0
    for i in range(len(t_flat)):
        if alpha_eval[i] > 0.5:
            if zk_tokens[i] == t_flat[i]:
                zk_win += 1
        else:
            if lm_tokens[i] == t_flat[i]:
                lm_win += 1
    print(f'\n  When router -> ZK: ZK correct = {zk_win}/{zk_preferred} ({zk_win/max(zk_preferred,1)*100:.0f}%)')
    print(f'  When router -> LM: LM correct = {lm_win}/{total-zk_preferred} ({lm_win/max(total-zk_preferred,1)*100:.0f}%)')

    # Parameter counts
    zk_p = sum(p.numel() for p in zk.parameters())
    lm_p = sum(p.numel() for p in lm.parameters())
    rx_p = sum(p.numel() for p in router.parameters())
    print(f'\n  {"Component":<20} {"Params":<12} {"Memory":<12}')
    print(f'  {"-"*44}')
    print(f'  {"Zeckendorf":<20} {zk_p:<12,} {zk_p*4/1e6:<12.2f}MB')
    print(f'  {"lm_head":<20} {lm_p:<12,} {lm_p*4/1e6:<12.2f}MB')
    print(f'  {"Router":<20} {rx_p:<12,} {rx_p*4/1e6:<12.2f}MB')
    print(f'  {"Total":<20} {zk_p+rx_p:<12,} {(zk_p+rx_p)*4/1e6:<12.2f}MB')
    print(f'  {"vs lm_head alone":<20} {"":>12} {lm_p*4/1e6/(zk_p+rx_p)/4e-6:<12.1f}x reduction')

    print(f'\n  => Hybrid with routing adds only {rx_p:,} router params')
    print(f'  => Inference cost: 2× forward (ZK + LM) but router decides which to trust')
    print(f'  => Practical: run LM only when α < 0.1, otherwise ZK suffices')
