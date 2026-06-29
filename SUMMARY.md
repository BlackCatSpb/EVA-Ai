# λ_d Architecture Summary

## Goal
Self-contained language model based on λ_d recurrence: content-dependent A(h) = V·diag(α★λ)·V⁻¹ with block-wise Fibonacci spectrum + Zeckendorf tree readout.

## Phase 2 — Training (step 200 / 37500)

**12-layer LDStack, D=896, K=4, 50K chunks wikitext-103**
- train PPL 599 at step 200, loss decreasing (6.39)
- No NaN (fp32 stable)
- Norm grows linearly: embed ±4.9 → layer 11 ±51 (~4.5/step)

## Critical Bug Fix (2026-06-29)

**Scalar λ_eff + clamping killed LDBlock.** Old code: `λ_eff = α·λ` (scalar per token), then `delta /= max(λ_eff)` where max = same scalar → division by identity. LDBlock = `h + V·V^T·norm(h)` ≈ identity. Model was learning only from lm_head + embed.

**Fix:** λ_eff is now a D-dimensional vector. Each of K groups (D//K dims) gets λ_k·α_k. No clamping. Content-dependent spectrum per eigen-direction.

## Phase 1 — Complete (frozen, bugs fixed)

### Results (D=256, single layer)

| Config | Train PPL | Eval PPL | Notes |
|--------|-----------|----------|-------|
| Random baseline | 50000 | — | |
| **LDBlock 5K** (NEW) | — | — | re-run needed with fixed arch |
| **LDBlock 50K** (NEW) | — | — | re-run needed with fixed arch |
| Transformer 5K | 125 | 354 | baseline |

### Old grid search (D=256, buggy arch)
- K and LoRA rank negligible (LDBlock was identity, so nothing mattered)
- All old PPL numbers are invalid — architecture was dead

## Key Decisions
- **Block-wise λ_eff** — K groups of D/K dimensions, each scaled by λ_k·α_k
- **No spectral clamping** — bounded by construction (||Δ|| ≤ max(λ)·√D)
- **LoRA frozen base = 0** — pure low-rank W_eff = A·B (not random + tiny correction)
- **fp32 only** — fp16 `.half()` in autograd causes NaN on MX550
- **Trainable norms** — final_norm, post_norm unfrozen for lm_head adaptation
- **Untied lm_head** — trainable, not from Qwen
- **Checkpoints** — saved per epoch + best, auto-resume

## Files
- `ld_model/core.py` — LDBlock (block-wise λ_eff), LoRALinear (zero base), LDMLP, LDStack
- `ld_model/readout.py` — ZeckendorfReadout (h·c logits, not φ)
- `train_phase2.py` — 12-layer training with checkpoint save/load
- `tests/test_synthetic.py` — stability, gates, sequential gen
- `tests/test_alpha_zeckendorf.py` — entropy, gates, Zeckendorf vs lm_head
- `LAMBDA_ARCHITECTURE.md` — full method documentation (Russian)
