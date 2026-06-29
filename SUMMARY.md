# λ_d Architecture Summary

## Goal
Self-contained language model based on λ_d recurrence: content-dependent A(h) = V·diag(α·λ)·V⁻¹ with Fibonacci mixture + Zeckendorf tree readout.

## Phase 2 — In Progress

**12-layer LDStack, D=896, K=4, 50K chunks wikitext-103**
- NaN fixed: fp16 `.half()` in autograd CUDA → 100% NaN. All fp32 now stable.
- Training running: loss 5.9→5.1 (PPL 625→164) at step 1.2K / 12.5K per epoch
- Eval at end of epoch 1 expected

## Phase 1 — Complete

### Results

| Config | Train PPL | Eval PPL | Notes |
|--------|-----------|----------|-------|
| Random baseline (ln V) | 50000 | — | vocab=50000 |
| **LDBlock 5K** | **206** | **581** | 1 layer D=256 K=4 |
| **LDBlock 50K** | **155** | **218** | 10× data → 2.7× better |
| Transformer 5K | 125 | 354 | 1 attn layer D=256 |
| Grid (K×LoRA) 5K e1 | ~2100 | ~750 | K=2..8, L=0/16/64 |

### Grid Search: K and LoRA Negligible
- K=2/3/4/6/8, LoRA rank=0/16/64 → all eval PPL ~750 at epoch 1
- Extended 3-epoch: K4_L0 best eval 567, K6_L0=584, K8_L64=578
- **Bottleneck is data quantity, not mode count**

### Key Finding: LDBlock Internal Residual + Deep Stack
LDBlock does `h_out = h + V·Λ̂·V⁻¹·norm(h)`. The internal residual is incompatible with deep stacking (2×/layer growth). Fixed with `residual=False` option + external residual in Qwen2LDBlock.

**Spectral radius clarification:** A = I + V·Λ̂·V⁻¹ has eigenvalues in (1.2]. This causes exponential blow-up if iterated as RNN. But LDBlock is a **residual feed-forward block**, not an RNN — applied once per layer in parallel across positions. RMS norm bounds δ to √D, giving **linear** (not exponential) growth across layers. Same stability mechanism as transformer pre-norm.

### NaN Root Cause
fp16 `.half()` casts in autograd graph on CUDA produce 100% NaN when mixed with fp32 operations. Fixed by removing all `.half()`/`.float()` casts. Model runs in pure fp32.

### Parameter Efficiency (D=256)
| Component | Params | vs Attention |
|-----------|--------|-------------|
| Attention QKV+O | 262K | baseline |
| LDBlock all (W_gate + V) | 67K | **3.9× fewer** |
| LDBlock trainable only | 1K | **255× fewer** |

### Baseline Comparison Note
LDBlock 50K (218) vs Transformer 5K (354) is **apples-to-oranges** (different data). Transformer baseline needs re-run on 50K for fair comparison.

## Key Decisions
- φ not debunked — it's a resonance
- A_d is a linear RNN family (SSM). Content-dependent gating is differentiator
- K does not matter for PPL
- LoRA rank does not help at D=256
- **Data scaling is primary lever** — 10× data → 2.7× PPL improvement
- Untied lm_head, gate_scale=4.0 critical for stable training
- Zeckendorf readout for inference only (25× faster)

## Files
- `ld_model/core.py` — LDBlock, LDStack, LoRALinear, LDMLP
- `ld_model/readout.py` — ZeckendorfReadout
- `train_phase1.py` — single-layer training (complete)
- `train_phase2.py` — 12-layer training (in progress)
- `train_baseline.py` — transformer attention baseline
- `grid_search.py` / `grid_search_extended.py` — K×LoRA sweep
- `tests/test_synthetic.py` — stability, gates, sequential gen
- `tests/test_alpha_zeckendorf.py` — entropy, gates, Zeckendorf vs lm_head
- `LAMBDA_ARCHITECTURE.md` — full method documentation
