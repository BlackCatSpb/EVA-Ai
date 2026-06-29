# λ_d Architecture: Methods & Goals

## Overview

λ_d (lambda-d) — content-dependent **residual feed-forward block** (not an RNN) where the effective linear transform A(h) is a function of the current hidden state. Unlike fixed-spectrum SSMs (Mamba, RWKV), λ_d computes a **per-token, per-layer spectrum** via gating over Fibonacci roots.

**Important stability note:** LDBlock is NOT iterated over time. Each layer applies one step in parallel across positions. The residual connection + RMS norm make the operation:
```
h_{l+1} = h_l + V·Λ̂·V⁻¹·rms_norm(h_l)
```
Since `rms_norm(h_l)` has RMS=1 (L2 norm = √D, independent of ||h_l||), each delta is bounded by √D, and the hidden state grows **linearly** with stack depth (not exponentially). This is the same stability mechanism as transformer pre-norm residual.

## Core Components

### `fibonacci_roots(max_k)` — `ld_model/core.py:24`

Computes λ_k — spectral radii of d-step Fibonacci companion matrices using Newton's method.

- **Goal**: Generate K eigenvalue bases with values strictly between 1 and 2
- λ₂ = φ ≈ 1.618, λ₃ ≈ 1.839, λ₄ ≈ 1.927, λ₅ ≈ 1.965, ...
- These form the spectrum from which each token selects per layer

### `random_orthogonal(D, n_reflections)` — `ld_model/core.py:53`

Random orthogonal matrix via Householder product.

- **Goal**: Create the eigenbasis V for each LDBlock
- Frozen after init, stored as buffer (not trainable)
- V⁻¹ = Vᵀ since orthogonal — cheap inverse

### `LDBlock` — `ld_model/core.py:82`

The core λ_d recurrence: one content-dependent linear layer.

**Forward pass:**
1. `rms_norm(h)` — stabilize hidden state magnitude
2. `α = softmax(scale · (W_gate · h_norm + b))` — content-dependent gating over K modes
3. `λ_eff = α · λ` — per-token effective spectral radius
4. `Δ = V · diag(λ_eff) · V⁻¹ · h_norm` — apply in eigenbasis
5. `Δ = Δ / max(λ_eff)` — clamp spectral radius ≤ 1
6. `h_out = h + Δ` — residual connection (optional: `residual=False` returns just Δ for external residual)

**Trainable params per layer:** 
- W_gate: D × K (e.g., 896×4 = 3.6K)
- b_gate: K (e.g., 4)
- Total: ~D·K parameters per layer (microscopic vs attention's 4·D²)

**Frozen per layer:**
- V: D × D orthogonal matrix (~0.8M at D=896, ~1M at D=1024)
- input_ln_w, post_ln_w: D each

**Parameter comparison (D=256, Phase 1):**
- Attention (QKV+O): 4·D² = 262K
- LDBlock all params (W_gate + V): D·K + D² = 1K + 66K = 67K (ratio ~3.9×)
- LDBlock trainable only (W_gate + b_gate): D·K + K = 1K (ratio ~255×)

**Goal**: Replace the QKV-attention sublayer with a parameter-efficient content-dependent block that captures per-token spectral dynamics via Fibonacci-gated recurrence.

### `LDMLP` — `ld_model/core.py:187`

SwiGLU MLP with optional LoRA adaptation.

- Gate, up, down projections
- LoRA: W′ = W + A·B (frozen W, trainable A∈ℝᵈ ʸ ˣ ʳ, B∈ℝʳ ˣ  ᵈ ʸ)
- **Goal**: Provide non-linear transformation complementary to LDBlock's linear recurrence

### `LDStack` — `ld_model/core.py:215`

Stacked λ_d layers with interleaved MLPs.

- **Goal**: Deep architecture (12-36 layers) for hierarchical representation learning

### `LoRALinear` — `ld_model/core.py:168`

Linear layer with LoRA: W_eff = W_buffer + A·B.

- **Goal**: Adapt frozen MLP weights with minimal trainable parameters

### `LDConfig` — `ld_model/core.py:9`

Configuration dataclass for D, n_layers, n_modes, vocab, intermediate, lora_rank.

---

## Zeckendorf Readout

### `fibonacci_bases(vocab_size)` — `ld_model/readout.py:13`

Generate Fibonacci bases for Zeckendorf representation.

- F = [1, 2, 3, 5, 8, ...]
- Every integer has a unique representation as sum of non-consecutive Fibonacci numbers
- **Goal**: Replace lm_head (D×V) with tree-structured readout (K×2×2×D)

### `zeckendorf_code(token_id, fibs)` — `ld_model/readout.py:27`

Compute Zeckendorf binary code for a token ID.

- Returns MSB-first bit array
- Constraint: no adjacent 1s (the non-linearity that distinguishes λ_d from softmax)

### `ZeckendorfReadout` — `ld_model/readout.py:48`

Tree-structured readout with learnable centroids.

- K levels, 2 states per level (prev=0 or prev=1), 2 digits (0 or 1)
- 3 centroids per level (c[k,0,0], c[k,0,1], c[k,1,0]; c[k,1,1] = 0 — invalid)
- h·c scoring (linear logits) with Zeckendorf constraint
- **Goal**: Reduce readout from D×V (hundreds of millions) to K×4×D (~hundreds of thousands)

### `predict(h, greedy, temperature)` — `ld_model/readout.py:153`

Generate next token by traversing the Zeckendorf tree.

- At each level k, choose digit 0 or 1 based on h·c[k,state,digit] softmax
- If state=1 (previous digit was 1), forced to 0 (Zeckendorf constraint)
- **Goal**: Decode hidden state to token ID via structured tree walk

### `compare_with_lm_head(h, W_embed)` — `ld_model/readout.py:205`

Compare Zeckendorf vs standard lm_head on top-k overlap and KL divergence.

- **Goal**: Verify Zeckendorf learns similar distribution to lm_head

---

## Training Scripts

### `train_phase1.py`

Single LDBlock (1 layer, D=256, K=4) trained on wikitext-103 CE loss.

- **Goal**: Validate basic learning capability of λ_d
- **Result**: train PPL 206 → eval PPL 581 (5K data), eval PPL 218 (50K data)
- **Key findings**: Uneeded lm_head critical; content-dependent gates differentiate

### `train_phase2.py`

12-layer LDStack at D=896 (Qwen2.5-0.5B scale). Frozen embedding + MLP + norms.

- **Goal**: Deep stack stability and layer specialization
- **Status**: In progress (NaN fix with external residual applied)

### `train_baseline.py`

Single transformer attention layer (RoPE + SwiGLU MLP) — apples-to-apples comparison.

- **Goal**: Baseline PPL for fair comparison with LDBlock
- **Result**: train PPL 125 → eval PPL 354 (vs LDBlock: train 206 → eval 581 on same 5K data)
- LDBlock is 1.6× worse on 5K data but has ~255× fewer trainable parameters (1K vs 262K)
- **Note**: 50K LDBlock (218) vs 5K transformer (354) is not a fair comparison — baseline must be re-run on 50K data to claim architectural advantage

### `train_fast.py`

Fast training from pre-cached numpy chunks (`wikitext_chunks.npy`).

- **Goal**: Skip dataset processing for rapid iteration
- **Usage**: `python train_fast.py [N_chunks]`

### `train_moredata.py`

Increased data regime (50K chunks vs 5K).

- **Goal**: Measure scaling behavior with more data
- **Result**: eval PPL 581→218, overfitting gap 2.8×→1.4×

---

## Grid Search

### `grid_search.py`

1-epoch sweep over K ∈ {2,3,4,6,8} × LoRA rank ∈ {0,16,64}.

### `grid_search_extended.py`

3-epoch runs for top configs: K4_L0, K6_L0, K8_L64.

- **Finding**: K and LoRA rank have negligible effect on epoch-1 PPL. All configs cluster at ~750 eval PPL.
- **Conclusion**: LDBlock bottleneck is data quantity and depth, not mode count

---

## Tests

### `tests/test_synthetic.py`

4 tests on random data (D=256, 6 layers):
- **test_stability**: No NaN/Inf, norm ratio ~1.0
- **test_gate_differs**: Gates differ between inputs (cos~0.80)
- **test_gate_distribution**: Gate entropy near max (1.26/1.39)
- **test_sequential_gen**: 32-step generation stable, norm constant

### `tests/test_alpha_zeckendorf.py`

5 tests for integrated components:
- **test_alpha_entropy_formula**: H(α) vs gate_scale matches theory
- **test_gate_differs**: Gate cos~0.42 between different inputs
- **test_per_layer_gate_distribution**: Each layer has distinct gate pattern
- **test_zeckendorf_vs_lm_head**: Low overlap (expected for untrained centroids)
- **test_zeckendorf_constraint**: Generated tokens all satisfy no-adjacent-1s

---

## Key Results (Phase 1 Complete)

| Setting | Train PPL | Eval PPL | Δ Overfitting |
|---|---|---|---|
| Random baseline (ln V) | 50000 | — | — |
| LDBlock 5K data | 206 | 581 | 2.8× |
| LDBlock 50K data | 155 | 218 | 1.4× |
| Transformer 5K data | 125 | 354 | 2.8× |
| K grid (all configs) | ~2100 (ep1) | ~750 (ep1) | — |

- **50K LDBlock (218) vs 5K transformer (354)** — not a fair comparison (apples-to-oranges, different data quantity). Re-run transformer on 50K needed for claim.
- Content-dependent gating confirmed (cos ≠ 1, H(α) < max)
- Model scales with data — no evidence of capacity ceiling yet

### Key Parameter Efficiency (D=256)

| Architecture | Trainable Params | Eval PPL (5K data) | Ratio |
|---|---|---|---|
| Transformer attention | 262K | 354 | baseline |
| LDBlock | 1K | 581 | 1.6× worse, 255× fewer params |
| LDBlock (50K data) | 1K | 218 | beats 5K baseline with 255× fewer params |
