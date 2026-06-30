# λ_d Architecture Summary

## Goal
Self-contained language model based on λ_d recurrence: content-dependent A(h) = V·diag(α★λ)·V⁻¹ with block-wise Fibonacci spectrum + Zeckendorf tree readout.

## Phase 2 — Training

**12-layer LDStack, D=896, K=4, 50K chunks wikitext-103**
- ✅ Fixed spectral clamping bug (block-wise λ_eff, no identity)
- ✅ Causal Conv1d (kernel=4) added — cross-token mixing
- ✅ Dense Bottleneck MLP (D→256→D) replaces LoRA rank-16 SwiGLU
- ✅ Gradient accumulation (eff batch=32), linear warmup (5%)
- ✅ All params trainable (no frozen groups)

## Critical Bug Fixes

| Bug | Fix |
|-----|-----|
| Scalar λ_eff + clamping = identity | Block-wise λ_eff vector, no clamping |
| LoRA base = frozen random | Base = 0, pure A·B low-rank |
| No cross-token mixing | Causal Conv1d (kernel=4) in each LDBlock |
| LoRA rank-16 bottleneck (16/4864 dims) | Dense bottleneck 896→256→896, fully trainable |
| Small physical batch (4) | Gradient accumulation to eff batch=32 |
| No warmup | Linear warmup 5% of steps |

## Files
- `ld_model/core.py` — LDBlock (causal conv + block-wise λ_eff + V·Λ·Vᵀ), BottleneckMLP, LDStack
- `ld_model/readout.py` — ZeckendorfReadout (h·c logits, not φ)
- `train_phase2.py` — 12-layer training with grad accum, warmup, checkpoint save/load
- `colab_train.py` — Standalone T4/Colab training script
- `monitor.py` — GPU/CPU/RAM/Disk live monitor
- `LAMBDA_ARCHITECTURE.md` — full method documentation (Russian)
