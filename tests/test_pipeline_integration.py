"""Verify pipeline integration: learnable_V + adaptive_depth default."""
import torch, sys, numpy as np
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack, clip_v_delta
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D, VOCAB, N_MODES, N_LAYERS = 896, 50000, 4, 6

cfg = LDConfig()
cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
cfg.vocab = VOCAB; cfg.bottleneck = 256
cfg.adaptive_depth = True

print(f'Defaults: learnable_V={cfg.learnable_V}, V_rank={cfg.V_rank}, V_delta_max_norm={cfg.V_delta_max_norm}')
print(f'Defaults: adaptive_depth={cfg.adaptive_depth}, thresholds={cfg.depth_threshold_low}-{cfg.depth_threshold_high}')

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.stack = LDStack(cfg)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)
    def forward(self, x):
        return self.lm_head(self.stack(self.embed(x)))

model = Model().to(DEVICE)
model.train()

ckpt = torch.load('checkpoints/model_step25000.pt', map_location=DEVICE, weights_only=True)
sd = {k: v.float() if v.dtype==torch.float16 else v for k,v in ckpt['model_fp16'].items()}
model_sd = model.state_dict()
compat_sd = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
missed = model.load_state_dict(compat_sd, strict=False)
del ckpt, sd

v_params = sum(p.numel() for n, p in model.named_parameters() if 'V_delta' in n)
print(f'Params: {sum(p.numel() for p in model.parameters()):,} total, {v_params:,} V_delta')
if missed.missing_keys:
    print(f'Missing (expected): {missed.missing_keys[:5]}')

arr = np.load('russian_chunks.npy')
x = torch.from_numpy(arr[:8, :128].copy()).long().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for step in range(5):
    opt.zero_grad()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, VOCAB), x.view(-1))
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    clip_v_delta(model.stack)

    v_norms = [(l.V_delta_U @ l.V_delta_V.T).norm().item()
               for l in model.stack.layers if l.V_delta_U is not None]
    print(f'step {step}: loss={loss.item():.2f} |Vd|={[f"{n:.4f}" for n in v_norms]}')

for name, p in model.named_parameters():
    if torch.isnan(p).any():
        print(f'NaN: {name}')
        break
else:
    print('No NaN. Pipeline integration OK.')
