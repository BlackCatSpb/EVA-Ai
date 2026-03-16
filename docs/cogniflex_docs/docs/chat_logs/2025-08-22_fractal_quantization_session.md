# CogniFlex – Fractal Quantization & Saving Session Archive

- Date/Time: 2025-08-22T21:05:20+03:00
- Location: `C:\Users\black\OneDrive\Desktop\CogniFlex`
- Objective: Fix overflow during quantization, add robust sharded/incremental saving with recovery.

## Key Code Changes

- Added `FractalWeightStore._safe_quantize_to_int8()` in `cogniflex/mlearning/storage/fractal_store.py`:
  - Replaces NaN/Inf with finite values via `np.nan_to_num`.
  - Detects all-zero blocks -> returns zeros with scale 1.0, sets `has_zero_scale`.
  - Clips to [-127, 127] before casting to `int8` to avoid overflow warnings.
- Refactored `FractalWeightStore._build_fractal_hierarchy()` `int8` path to use the helper.
- Metadata: now stores both `quant_scale` and `quantization_scale`, and `has_zero_scale` when applicable.

## Save & Recovery Methods Confirmed

- `save_to_disk_sharded(...)` – writes compressed `.npz` shards + `shards_manifest.jsonl`.
- `save_to_disk_incremental(..., resume=True/False)` – batched, resumable saving with `incremental_state.json`.
- `save_to_disk_with_recovery(...)` – retries with reduced batch sizes automatically.
- `resume_save(output_path)` – convenience wrapper to resume.

## Recommended Run (PowerShell one-liners)

- Use venv Python directly (no activation):
```powershell
$env:PYTHONPATH="C:\Users\black\OneDrive\Desktop\CogniFlex"; $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"; .\.venv311\Scripts\python.exe -c "from cogniflex.mlearning.storage.fractal_store import repack_model_to_fractal; repack_model_to_fractal(r'C:\path\to\hf-model', r'C:\Users\black\OneDrive\Desktop\CogniFlex\fractal_out', fractal_levels=4, block_size=64, device='cpu')"
```

- Save with recovery (if store is already built in-memory):
```python
from cogniflex.mlearning.storage.fractal_store import FractalWeightStore
store = FractalWeightStore(block_size=64, fractal_levels=4, device="cpu")
# ... pack weights ...
store.save_to_disk_with_recovery(r"C:\Users\black\OneDrive\Desktop\CogniFlex\fractal_out")
```

- Resume after interruption:
```python
from cogniflex.mlearning.storage.fractal_store import FractalWeightStore
store = FractalWeightStore(block_size=64, fractal_levels=4, device="cpu")
store.resume_save(r"C:\Users\black\OneDrive\Desktop\CogniFlex\fractal_out")
```

## Notes

- Prefer CPU for conversion to reduce GPU memory pressure.
- Reduce `block_size` (e.g., 32–64) for lower memory.
- Environment hint for CUDA: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Output directory contains `index.json` and shard manifests for reload.

## Next Steps

1. Run a small-model conversion to validate no overflow warnings.
2. Verify metadata (`quant_scale`/`quantization_scale`, `has_zero_scale`) present on int8 containers.
3. Optionally add granular progress logs in incremental save (every N shards).

---
This archive captures the essential context and commands so you can safely close the heavy chat panel to reduce IDE RAM usage without losing progress. The original chat remains untouched.
