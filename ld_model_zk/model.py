"""ZKModel: λ_d stack (frozen) + ZeckendorfReadout (trained)."""

import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ld_model.core import LDConfig, LDStack
from ld_model.readout import ZeckendorfReadout


class ZKModel(torch.nn.Module):
    """λ_d stack with Zeckendorf readout instead of lm_head.

    LDStack is frozen; only ZeckendorfReadout centroids are trained.
    """

    def __init__(self, cfg):
        super().__init__()
        self.embed = torch.nn.Embedding(cfg.vocab, cfg.D)
        self.stack = LDStack(cfg)
        self.readout = ZeckendorfReadout(cfg)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning log-probabilities for all tokens."""
        h = self.stack(self.embed(x))
        B, L, D = h.shape
        h_flat = h.reshape(-1, D)
        return self.readout.forward_log_probs(h_flat)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """NLL loss on Zeckendorf log-probabilities (efficient per-target)."""
        h = self.stack(self.embed(x))
        B, L, D = h.shape
        h_flat = h.reshape(-1, D)
        target = x.reshape(-1)
        log_probs = self.readout.log_probs_for_target(h_flat, target)
        return -log_probs.mean()

    def freeze_stack(self):
        for p in self.embed.parameters():
            p.requires_grad = False
        for p in self.stack.parameters():
            p.requires_grad = False


class ZKModelTrainer:
    """Lightweight trainer for ZKModel."""

    def __init__(self, model, lr=1e-3, device='cuda'):
        self.model = model.to(device)
        self.device = device
        params = [p for p in self.model.readout.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr)
        self.lr = lr

    def train_step(self, x: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model.compute_loss(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.readout.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def eval_step(self, x: torch.Tensor) -> float:
        self.model.eval()
        return self.model.compute_loss(x).item()

    def compare_with_lm(self, zk_model, lm_head: torch.nn.Linear,
                        h: torch.Tensor, top_k: int = 10) -> dict:
        """Compare Zeckendorf vs lm_head on hidden states."""
        return zk_model.readout.compare_with_lm_head(h, lm_head.weight, top_k=top_k)


def load_zk_from_checkpoint(ckpt_path: str, cfg, device='cuda'):
    """Load frozen LDStack from checkpoint + fresh ZeckendorfReadout."""
    model = ZKModel(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = {k: v.float() if v.dtype == torch.float16 else v
          for k, v in ckpt['model_fp16'].items()}
    msd = model.state_dict()
    compat = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(compat, strict=False)
    model.freeze_stack()
    return model.to(device)
