"""Debug NaN in Phase 2 model."""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDBlock, LDMLP, fibonacci_roots

D, K, VOCAB, I, N_LAYERS = 896, 4, 50000, 4864, 12
DEVICE = torch.device('cuda')

class Qwen2LDBlock(torch.nn.Module):
    def __init__(self, d_model, intermediate, n_modes, lambda_roots, layer_idx, lora_rank=16):
        super().__init__()
        cfg = LDConfig(); cfg.D = d_model; cfg.n_modes = n_modes; cfg.vocab = VOCAB
        cfg.intermediate = intermediate; cfg.lora_rank = lora_rank; cfg.use_lora = lora_rank > 0
        self.ld = LDBlock(cfg, layer_idx, lambda_roots)
        self.post_norm = torch.nn.RMSNorm(d_model, eps=1e-6)
        self.mlp = LDMLP(cfg, use_lora=lora_rank > 0)
    def forward(self, x):
        delta = self.ld(x, residual=False)
        h = x.float() + delta.float()
        r = self.post_norm(h)
        h = h + self.mlp(r).float()
        return h

class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        lambdas = fibonacci_roots(K + 1)
        self.layers = torch.nn.ModuleList([Qwen2LDBlock(D, I, K, lambdas, i, lora_rank=16) for i in range(N_LAYERS)])
        self.final_norm = torch.nn.RMSNorm(D, eps=1e-6)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=False)
    def forward(self, input_ids):
        h = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if torch.isnan(h).any():
                print(f'NAN at layer {i}!', flush=True)
                break
        h = self.final_norm(h.float())
        logits = self.lm_head(h)
        return logits

model = TestModel().to(DEVICE)
model.train()

arr = np.load('wikitext_chunks.npy')
bx = torch.tensor(arr[:2, :-1], dtype=torch.long).to(DEVICE)

# Phase2Model style freeze
for name, p in model.named_parameters():
    is_gate = 'W_gate' in name or 'b_gate' in name
    is_lora = 'A' in name or 'B' in name
    is_head = 'lm_head' in name
    p.requires_grad_(is_gate or is_lora or is_head)
    if 'gate' in name.lower() or 'lora' in name.lower():
        print(f'  trainable: {name} ({p.numel()})', flush=True)

logits = model(bx)
print(f'Train+freeze: nan={torch.isnan(logits).any().item()}, range=[{logits.min():.4f},{logits.max():.4f}]', flush=True)

loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), bx.reshape(-1))
loss.backward()
print(f'Loss={loss.item():.4f}, backward OK', flush=True)
print('Done', flush=True)
