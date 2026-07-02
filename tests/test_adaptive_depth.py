"""Test adaptive depth routing in LDStack."""
import torch, sys
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack

D, K, N_LAYERS = 256, 4, 6
cfg = LDConfig()
cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = K; cfg.vocab = 10000
cfg.bottleneck = 64
cfg.adaptive_depth = True
cfg.depth_threshold_low = 0.05
cfg.depth_threshold_high = 0.20

model = LDStack(cfg)
model.eval()

B, L = 2, 16
x = torch.randn(B, L, D)

with torch.no_grad():
    h, gates = model(x, return_gates=True)
    print(f"Output shape: {h.shape}")
    print(f"Gates shape:  {gates.shape}")

    # Spread per layer
    print(f"\n=== Per-layer spread ===")
    for lidx in range(N_LAYERS):
        spread = gates[lidx].std(dim=-1)
        thresh = torch.sigmoid(model.depth_logits[lidx]).item() if lidx < N_LAYERS-1 else 0
        n_continue = (spread > thresh).sum().item() if lidx < N_LAYERS-1 else B*L
        print(f"  Layer {lidx}: spread={spread.mean():.3f} thresh={thresh:.3f} continue={n_continue}/{B*L}")

    print(f"\nDepth logits: {model.depth_logits.data}")
    print(f"Depth thresholds: {[f'{torch.sigmoid(model.depth_logits[lidx]).item():.3f}' for lidx in range(N_LAYERS-1)]}")

    # Test force_depth
    force = torch.randint(0, N_LAYERS, (B, L))
    h2 = model(x, force_depth=force)
    print(f"\nforce_depth output shape: {h2.shape}")
    print(f"force_depth range: {force.min().item()}-{force.max().item()}")

    # Backward compatibility: adaptive=False
    cfg2 = LDConfig()
    cfg2.D = D; cfg2.n_layers = N_LAYERS; cfg2.n_modes = K; cfg2.vocab = 10000
    cfg2.bottleneck = 64
    cfg2.adaptive_depth = False

    model2 = LDStack(cfg2)
    model2.eval()
    with torch.no_grad():
        h3 = model2(x)
        print(f"\nNon-adaptive output shape: {h3.shape}")

print("\nAll tests passed.")
