from __future__ import annotations

from typing import Any, Dict

import torch

from eva.adapters.torch_adapter import Batch


def example_model_fn(batch: Batch) -> Dict[str, Any]:
    """A trivial CPU-only model function demonstrating expected signature.

    It computes per-row sums for any LongTensor inputs and returns a dict
    with logits-like tensor and echoes back meta indices.
    """
    tensors = batch.tensors
    metas = batch.metas

    # Example: use input_ids if present; else sum all 1D long tensors
    if "input_ids" in tensors:
        x = tensors["input_ids"].to(dtype=torch.long)
        logits = x.sum(dim=1, keepdim=True).to(dtype=torch.float32)
    else:
        parts = []
        for k, v in tensors.items():
            if v.dtype in (torch.int32, torch.int64) and v.dim() == 2:
                parts.append(v)
        if parts:
            x = torch.stack([p.sum(dim=1) for p in parts], dim=1).sum(dim=1, keepdim=True)
            logits = x.to(dtype=torch.float32)
        else:
            # Fallback: batch size inferred from any tensor's first dim
            if not tensors:
                # No tensors available - return empty result
                logits = torch.zeros((len(metas), 1), dtype=torch.float32)
            else:
                any_t = next(iter(tensors.values()))
                logits = torch.zeros((any_t.shape[0], 1), dtype=torch.float32)

    return {
        "logits": logits,  # [B, 1]
        "meta_idx": torch.tensor([m.idx for m in metas], dtype=torch.long),
    }
