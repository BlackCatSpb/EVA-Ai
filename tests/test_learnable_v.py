"""Test learnable V delta: training stability + gradient flow."""
import torch, sys
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack, rms_norm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# Config with learnable_V
cfg = LDConfig()
cfg.D = 256; cfg.n_layers = 4; cfg.n_modes = 4; cfg.vocab = 10000
cfg.bottleneck = 64; cfg.adaptive_depth = False
cfg.learnable_V = True; cfg.V_rank = 8; cfg.V_delta_max_norm = 0.1

model = LDStack(cfg).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

B, L = 4, 32
x = torch.randn(B, L, 256, device=DEVICE)

print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')
v_delta_params = 0
for l in model.layers:
    if l.V_delta_U is not None:
        v_delta_params += l.V_delta_U.numel() + l.V_delta_V.numel()
print(f'V_delta params: {v_delta_params:,}')

for step in range(30):
    opt.zero_grad()
    h, gates = model(x, return_gates=True)

    total_norm = 0
    orth_loss = 0
    for l in model.layers:
        if l.V_delta_U is not None:
            delta_norm = (l.V_delta_U @ l.V_delta_V.T).norm().item()
            total_norm += delta_norm
            orth_loss = orth_loss + l.orth_loss()

    loss = h.norm() + gates.mean() + 0.01 * orth_loss
    loss.backward()

    # Check V_delta gradients
    v_grad_norm = 0
    for l in model.layers:
        if l.V_delta_U is not None and l.V_delta_U.grad is not None:
            v_grad_norm += l.V_delta_U.grad.norm().item()
            v_grad_norm += l.V_delta_V.grad.norm().item()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    # Apply norm clipping
    for l in model.layers:
        if l.V_delta_U is not None:
            U = l.V_delta_U.data
            V = l.V_delta_V.data
            norm_est = U.norm() * V.norm() + 1e-10
            if norm_est > cfg.V_delta_max_norm:
                scale = (cfg.V_delta_max_norm / norm_est) ** 0.5
                l.V_delta_U.data *= scale
                l.V_delta_V.data *= scale

    if step % 10 == 0:
        print(f'step {step:3d}: loss={loss.item():.3f} |V_delta|={total_norm:.4f} '
              f'orth={orth_loss.item():.6f} V_grad={v_grad_norm:.4f} grad={grad_norm:.4f}')

# Verify no NaN
for name, p in model.named_parameters():
    if torch.isnan(p).any():
        print(f'NaN in {name}!')
        break
else:
    print('No NaN detected.')

# Verify V_eff is near-orthogonal
print(f'\nV_eff orthogonality check (layer 0):')
l0 = model.layers[0]
v = torch.randn(1, 256, device=DEVICE)
with torch.no_grad():
    v_frozen = v @ l0.V
    if l0.V_delta_U is not None:
        delta_v = (v @ l0.V_delta_U) @ l0.V_delta_V.T
        v_eff = v_frozen + delta_v
    else:
        v_eff = v_frozen
    print(f'  ||v||={v.norm().item():.4f} ||V_frozen·v||={v_frozen.norm().item():.4f}')
    if l0.V_delta_U is not None:
        print(f'  ||V_eff·v||={v_eff.norm().item():.4f} (should be ~{v.norm().item():.4f})')

print('\nAll tests passed.')
