"""Fast V_delta divergence test on random data, small dim."""
import torch, sys
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack
import torch.nn as nn

cfg = LDConfig()
cfg.D = 256; cfg.n_layers = 6; cfg.n_modes = 4; cfg.vocab = 50000
cfg.bottleneck = 64; cfg.adaptive_depth = False
cfg.learnable_V = True; cfg.V_rank = 16; cfg.V_delta_max_norm = 0.1

model = LDStack(cfg)
model.train()
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
v_params = sum(p.numel() for n, p in model.named_parameters() if 'V_delta' in n)
print(f'V_delta params: {v_params:,}')

# Use lm_head to mimic real training
lm_head = nn.Linear(256, 50000, bias=False)
opt = torch.optim.Adam(list(model.parameters()) + list(lm_head.parameters()), lr=1e-3)

x = torch.randint(0, 50000, (8, 32))

# Track cosine similarity over time
for step in range(200):
    opt.zero_grad()
    h = model(lm_head.weight[x])
    logits = lm_head(h)
    loss = nn.functional.cross_entropy(logits.view(-1, 50000), x.view(-1))
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    
    # Clip V_delta
    for l in model.layers:
        U, V = l.V_delta_U.data, l.V_delta_V.data
        n = (U @ V.T).norm().item()
        if n > cfg.V_delta_max_norm:
            s = (cfg.V_delta_max_norm / n) ** 0.5
            l.V_delta_U.data *= s
            l.V_delta_V.data *= s
    
    if step % 50 == 0:
        norms = [(l.V_delta_U @ l.V_delta_V.T).norm().item() for l in model.layers]
        # Cosine similarity between adjacent layers
        coses = []
        for i in range(len(model.layers)-1):
            Vi = model.layers[i].V_delta_U @ model.layers[i].V_delta_V.T
            Vj = model.layers[i+1].V_delta_U @ model.layers[i+1].V_delta_V.T
            cos = (Vi / Vi.norm() * Vj / Vj.norm()).sum().item()
            coses.append(f'{cos:.3f}')
        print(f'step {step:3d}: loss={loss.item():.3f} norms={[f"{n:.4f}" for n in norms]}')
        print(f'  cos(adjacent): [{", ".join(coses)}]')

# Final stats
print(f'\nFinal V_delta norms:')
for i, l in enumerate(model.layers):
    uv = l.V_delta_U @ l.V_delta_V.T
    n = uv.norm().item()
    s = torch.linalg.svdvals(uv.float())
    eff_rank = (s / s.max()).sum().item()
    print(f'  L{i}: ||UV||={n:.4f} eff_rank={eff_rank:.1f}')
