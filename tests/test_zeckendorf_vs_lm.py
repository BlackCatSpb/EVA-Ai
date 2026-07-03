"""Test ZeckendorfReadout vs lm_head on checkpoint."""
import torch, sys, numpy as np, time, math
from copy import deepcopy

sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack
from ld_model.readout import ZeckendorfReadout

DEVICE = 'cuda'
D, N_LAYERS, VOCAB = 896, 12, 50000
B, L = 4, 32

cfg = LDConfig()
cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = 4; cfg.vocab = VOCAB
cfg.bottleneck = 256; cfg.adaptive_depth = True; cfg.learnable_V = True; cfg.V_rank = 8

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        self.stack = LDStack(cfg)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=False)
    def forward(self, x):
        return self.lm_head(self.stack(self.embed(x)))

model = Model().to(DEVICE)

ckpt = torch.load('checkpoints/model_step25000.pt', map_location=DEVICE, weights_only=True)
sd = {k: v.float() if v.dtype==torch.float16 else v for k,v in ckpt['model_fp16'].items()}
msd = model.state_dict()
compat = {k: v for k,v in sd.items() if k in msd and msd[k].shape == v.shape}
model.load_state_dict(compat, strict=False)

torch.save(deepcopy(ckpt), 'checkpoints/model_step25000_zk_test.pt')
del ckpt
print('Checkpoint copied.')

arr = np.load('russian_chunks.npy')
x = torch.from_numpy(arr[:B, :L].copy()).long().to(DEVICE)

model.eval()
with torch.no_grad():
    h = model.stack(model.embed(x))
    h_flat = h.reshape(-1, D)

    # === lm_head ===
    t0 = time.time()
    logits_lm = model.lm_head(h_flat)
    t_lm = time.time() - t0
    probs_lm = torch.softmax(logits_lm / 1.0, dim=-1)
    entropy_lm = -(probs_lm * probs_lm.log().clamp(min=-1e10)).sum(dim=-1).mean().item()

    print('\n=== lm_head (44.8M params) ===')
    print(f'  Time: {t_lm*1000:.1f}ms for {B*L} tokens')
    print(f'  Entropy: {entropy_lm:.4f} (max={math.log(VOCAB):.4f})')

    # === Zeckendorf Readout ===
    zk = ZeckendorfReadout(cfg).to(DEVICE)
    zk_params = sum(p.numel() for p in zk.parameters())
    print(f'\n=== ZeckendorfReadout ({zk_params:,} params) ===')

    h_one = h_flat[:1]
    t0 = time.time()
    comp = zk.compare_with_lm_head(h_one, model.lm_head.weight, top_k=10)
    t_zk = time.time() - t0
    print(f'  Time (1 token): {t_zk*1000:.1f}ms')
    overlap_key = 'top_k_overlap'
    print(f'  Top-10 overlap with lm_head: {comp[overlap_key]:.1f} / 10')
    print(f'  KL(lm_head || Zeckendorf): {comp["kl_div"]:.4f}')
    print(f'  Valid Zeckendorf tokens: {comp["V_common"]:,} / {comp["V_total"]:,}')

    tokens_zk = zk.predict(h_flat[:4])
    top1_lm = probs_lm[:4].argmax(dim=-1)
    print(f'  Zeckendorf generated: {tokens_zk.tolist()}')
    print(f'  lm_head argmax:       {top1_lm.tolist()}')

    lm_params = sum(p.numel() for p in model.lm_head.parameters())
    print(f'\n=== Size comparison ===')
    print(f'  lm_head: {lm_params:,} params ({lm_params/1e6:.1f}M)')
    print(f'  Zeckendorf: {zk_params:,} params ({zk_params/1e6:.1f}M)')
    print(f'  Reduction: {lm_params/zk_params:.0f}x')
    print(f'  Inference memory: lm_head=({VOCAB},{D})={VOCAB*D*4/1e6:.1f}MB vs'
          f' Zeckendorf={zk_params*4/1e6:.2f}MB')
