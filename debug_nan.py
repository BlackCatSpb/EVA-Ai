"""Debug NaN in Phase 2 model (updated for new architecture)."""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack

D, K, VOCAB, N_LAYERS = 896, 4, 50000, 12
DEVICE = torch.device('cuda')

class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        cfg = LDConfig(); cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = K; cfg.vocab = VOCAB; cfg.bottleneck = 256
        self.stack = LDStack(cfg)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=False)
    def forward(self, input_ids):
        h = self.embed(input_ids)
        h = self.stack(h)
        return self.lm_head(h)

model = TestModel().to(DEVICE)
model.train()
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

arr = np.load('wikitext_chunks.npy')
bx = torch.tensor(arr[:2, :-1], dtype=torch.long).to(DEVICE)

logits = model(bx)
print(f'Forward: nan={torch.isnan(logits).any().item()}, range=[{logits.min():.4f},{logits.max():.4f}]', flush=True)

loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), torch.tensor(arr[:2, 1:], dtype=torch.long).to(DEVICE).reshape(-1))
loss.backward()
print(f'Loss={loss.item():.4f}, backward OK', flush=True)
print('Done', flush=True)
