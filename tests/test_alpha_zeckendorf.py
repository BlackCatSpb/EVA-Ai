"""
Synthetic verification: α-entropy regularization + Zeckendorf readout.
No gradient training; we only verify formulas and distributions.
"""

import os, sys, math
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ld_model.core import LDConfig, LDStack, fibonacci_roots
from ld_model.readout import ZeckendorfReadout, fibonacci_bases, zeckendorf_code

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# Config
cfg = LDConfig()
cfg.D = 256
cfg.n_layers = 6
cfg.n_modes = 4
cfg.vocab = 1000
cfg.intermediate = 1024
cfg.lora_rank = 32
cfg.use_lora = True

lambdas = fibonacci_roots(cfg.n_modes + 1)
print(f'Config: D={cfg.D}, layers={cfg.n_layers}, modes={cfg.n_modes}')
print(f'  lambdas: {[f"{l:.4f}" for l in lambdas.tolist()]}')

# Model
model = LDStack(cfg).to(DEVICE, torch.float16)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable params: {trainable/1e3:.0f}K')

# ====== Test 1: α-entropy formula verified by construction ======
print('\n' + '=' * 60)
print('TEST 1: α-entropy formula and gate sparsity via gate_scale')
print('=' * 60)

B, L = 4, 8
h = F.normalize(torch.randn(B, L, cfg.D, device=DEVICE, dtype=torch.float16), dim=-1) * (cfg.D ** 0.5)

with torch.no_grad():
    _, alpha = model(h, return_gates=True)

# α distribution: should be approximately uniform (W_gate ~ N(0,0.01), scale=4)
# Even with scale=4, small random weights won't produce extreme gates
mean_alpha = alpha.float().mean(dim=(0, 1, 2))  # mean over all tokens and layers
entropy = -(alpha.float() * torch.log(alpha.float().clamp(min=1e-10))).sum(dim=-1).mean().item()
max_H = math.log(cfg.n_modes)
print(f'  Mean α per mode: {mean_alpha.tolist()}')
print(f'  α-entropy: {entropy:.4f} / {max_H:.4f}')

# Verify that increasing gate_scale reduces entropy
# Manually compute what happens with scale=1 vs scale=4
layer0 = model.layers[0]
with torch.no_grad():
    gate_logits = (h.float() @ layer0.W_gate.float()) + layer0.b_gate.float()
    a_scale1 = F.softmax(gate_logits * 1.0, dim=-1)
    a_scale4 = F.softmax(gate_logits * 4.0, dim=-1)
    H1 = -(a_scale1 * torch.log(a_scale1.clamp(min=1e-10))).sum(dim=-1).mean().item()
    H4 = -(a_scale4 * torch.log(a_scale4.clamp(min=1e-10))).sum(dim=-1).mean().item()
    
print(f'  gate_scale=1 → H(α)={H1:.4f}')
print(f'  gate_scale=4 → H(α)={H4:.4f}')
if H4 < H1:
    print('  [PASS] Higher gate_scale → lower entropy (sparser gates)')
else:
    print('  [INFO] Entropy unchanged (random weights too small to matter)')

# ====== Test 2: Gate differentiation ======
print('\n' + '=' * 60)
print('TEST 2: Gate differentiation (α(h₁) ≠ α(h₂))')
print('=' * 60)

h1 = F.normalize(torch.randn(B, L, cfg.D, device=DEVICE, dtype=torch.float16), dim=-1) * (cfg.D ** 0.5)
h2 = F.normalize(torch.randn(B, L, cfg.D, device=DEVICE, dtype=torch.float16), dim=-1) * (cfg.D ** 0.5)

with torch.no_grad():
    _, g1 = model(h1, return_gates=True)
    _, g2 = model(h2, return_gates=True)

cos = F.cosine_similarity(
    g1.float().reshape(-1, cfg.n_modes),
    g2.float().reshape(-1, cfg.n_modes), dim=-1
).mean().item()
print(f'  Gate cos(h₁, h₂): {cos:.4f}')
if cos < 0.95:
    print('  [PASS] Gates differ between inputs')
else:
    print('  [COLLAPSE] Gates too similar')

# Per-layer breakdown
for li in range(cfg.n_layers):
    c = F.cosine_similarity(
        g1[li].float().reshape(-1, cfg.n_modes),
        g2[li].float().reshape(-1, cfg.n_modes), dim=-1
    ).mean().item()
    print(f'  Layer {li}: cos={c:.4f}')

# ====== Test 3: Full-stack gate distribution ======
print('\n' + '=' * 60)
print('TEST 3: Per-layer gate distribution')
print('=' * 60)

h_inf = F.normalize(torch.randn(2, 16, cfg.D, device=DEVICE, dtype=torch.float16), dim=-1) * (cfg.D ** 0.5)
with torch.no_grad():
    _, gates_all = model(h_inf, return_gates=True)

for li in range(cfg.n_layers):
    g = gates_all[li].float().mean(dim=(0, 1))
    ent = -(g * torch.log(g.clamp(min=1e-10))).sum().item()
    dom = g.argmax().item()
    print(f'  Layer {li}: α=[{" ".join(f"{v.item():.3f}" for v in g)}]  H={ent:.3f}  dom=d={dom+2}')

# ====== Test 4: Zeckendorf readout vs lm_head ======
print('\n' + '=' * 60)
print('TEST 4: Zeckendorf readout vs lm_head')
print('=' * 60)

zk = ZeckendorfReadout(cfg).to(DEVICE, torch.float16)
print(f'  ZK params: {sum(p.numel() for p in zk.parameters())/1e3:.0f}K')

W_embed = torch.randn(cfg.vocab, cfg.D, device=DEVICE, dtype=torch.float16) * 0.1

results = []
for q in range(10):
    hq = F.normalize(torch.randn(1, cfg.D, device=DEVICE, dtype=torch.float16), dim=-1) * (cfg.D ** 0.5)
    m = zk.compare_with_lm_head(hq, W_embed, top_k=10)
    results.append(m)
    if q < 3:
        print(f'  Q{q}: overlap={m["top_k_overlap"]:.1f}/10, KL={m["kl_div"]:.4f}')

avg_ol = np.mean([r['top_k_overlap'] for r in results])
avg_kl = np.mean([r['kl_div'] for r in results])
print(f'\n  Average: overlap={avg_ol:.1f}/10, KL={avg_kl:.4f}')

# ====== Test 5: Zeckendorf generation ======
print('\n' + '=' * 60)
print('TEST 5: Zeckendorf tree traversal (generation)')
print('=' * 60)

hq = F.normalize(torch.randn(5, cfg.D, device=DEVICE, dtype=torch.float16), dim=-1) * (cfg.D ** 0.5)
t_greedy = zk.predict(hq, greedy=True)

fibs = fibonacci_bases(cfg.vocab)
all_good = True
for t in t_greedy.tolist():
    bits = zeckendorf_code(t, fibs)
    for i in range(len(bits) - 1):
        if bits[i] == 1 and bits[i+1] == 1:
            print(f'  [FAIL] Token {t}: adjacent 1s at {i},{i+1}')
            all_good = False
    if t >= cfg.vocab:
        print(f'  [FAIL] Token {t} >= {cfg.vocab}')
        all_good = False

if all_good:
    print(f'  [PASS] All {len(t_greedy)} tokens respect Zeckendorf constraint')
    print(f'  Greedy tokens: {t_greedy.tolist()[:10]}')

# ====== Summary ======
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'  α-entropy (scale=4): {H4:.3f} / {max_H:.3f} (scale=1: {H1:.3f})')
print(f'  Gate diff cos: {cos:.4f}')
print(f'  Zeckendorf vs lm_head: overlap={avg_ol:.1f}/10, KL={avg_kl:.4f}')
print(f'  Zeckendorf constraint: {"PASS" if all_good else "FAIL"}')
print('Done.')
