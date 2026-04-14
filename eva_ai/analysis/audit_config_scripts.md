# Config & Scripts Audit

## 1. Configuration System

### 1.1 optimal_config.json

**Path:** eva_ai/config/optimal_config.json

**Contains:**
- system_info: 15.7 GB RAM, 8 CPU cores, no CUDA (HARDCODED)
- hybrid_cache: 22000 tokens, 0.5 GB target, LRU eviction
- fractal_model_manager: duplicates fractal_model_config.json
- text_quality: auto-improvement with learning_rate=3e-5
- gui: light theme, 5000ms refresh
- training: auto-training OFF, batch_size=1
- logging: INFO level, 100MB files, 5 backups

**Problems:**
- system_info contains HARDCODED values - not portable
- fractal_model_manager duplicates fractal_model_config.json
- cuda_available: false - config does not adapt to GPU

### 1.2 fractal_model_config.json

**Path:** eva_ai/config/fractal_model_config.json

**Contains:**
- device: cpu
- max_memory_tokens: 22000
- target_memory_gb: 0.5
- batch_size: 2, max_length: 128, overlap_tokens: 32
- cache_tokenization: true, parallel_tokenization: false
- tokenization_workers: 2
- memory_optimization: true, use_uint16: true
- tensor_pool_size: 500

**Status:** Used by current_manager.py and opt_core.py

### 1.3 gui_config.json

**Path:** eva_ai/config/gui_config.json

**Contains:**
- theme: light, compact_mode: false
- show_advanced_metrics: true, enable_gpu_monitoring: true
- auto_refresh_interval: 5000, notification_throttle_seconds: 30

### 1.4 apply_optimal_config.py

**Purpose:** Applies configs via console output only (does not modify code)

**Functions:**
- apply_fractal_config() - reads fractal_model_config.json
- apply_gui_config() - reads gui_config.json

**Problems:**
- Does NOT apply optimal_config.json - this is misinformation
- Does not modify runtime configuration
- Simply prints parameters to console
- Hardcoded paths os.getcwd() + eva - does not work from other directories

### 1.5 CoreBrain Integration

**Config Usage:**

| File | Reads | Problem |
|------|-------|---------|
| current_manager.py | fractal_model_config.json | Path eva/config does not work |
| opt_core.py | fractal_model_config.json | Path eva/config does not work |
| apply_optimal_config.py | Both JSONs | Not called from CoreBrain |

**Code path:**
`python
optimal_config_path = os.path.join(os.getcwd(), eva, config, fractal_model_config.json)
`

**Problem:** os.getcwd() returns working directory, not project root.

---

## 2. Scripts (Migration Utilities)

### 2.1 migrate_kg_to_fg.py

**Purpose:** Migrate data from KnowledgeGraph to FractalGraphV2

**Logic:**
1. Creates CoreBrain
2. Searches for knowledge_graph in brain.components
3. Calls migrate_knowledge_graph(brain)

**Status:** DEPRECATED - calls non-existent module kg_to_fg_migration

### 2.2 migrate_to_optimized.py

**Purpose:** Migrate to OptimizedFractalModelManager

**Functions:**
1. Tests OptimizedFractalModelManager
2. Updates unified_config.json - file does NOT exist in repo
3. Copies optimized_fractal_model_manager.py to current_manager.py

**Problems:**
- Calls OptimizedFractalModelManager from eva_ai.mlearning.optimized_fractal_model_manager - file NOT FOUND
- Updates unified_config.json which does not exist
- Creates crutch with current_manager.py

### 2.3 migrate_events.py

**Purpose:** Documentation of Event System to Event Bus migration

**Logic:**
- Simply logs changes
- Does not perform actual migration
- Status: INFO script, does not perform actions

### 2.4 load_gguf_to_fg.py

**Purpose:** Load GGUF model architecture into FractalGraph

**Logic:**
1. Creates FractalMemoryGraph
2. Calls create_architecture_mapper(fg)
3. Loads architecture via mapper.load_model_architecture(gguf_path)
4. Tests create_aci_concept_from_context()

**Problems:**
- GGUF path hardcoded: C:\\Users\\black\\...\\qwen2.5-3b-instruct-q4_k_m.gguf
- Uses fg_gguf_architecture_mapper - not verified to exist

### 2.5 activate_max_cache.py

**Purpose:** Activate maximum hybrid cache

**Functions:**
1. Tests UnifiedFractalManager
2. Checks cache parameters
3. Stress-test tokenization
4. Generation test

**Problems:**
- Calls UnifiedFractalManager from eva_ai.mlearning.unified_fractal_manager - not verified
- Calls manager.optimizations.optimized_tokenize() - method may not exist
- Hardcoded values in output (773,461 tokens, 3.0 GB) do not match configs

### 2.6 complete_fractal_solution.py

**Purpose:** Complete solution for cleanup and export

**Stages:**
1. cleanup_old_models() - deletes old models
2. create_unique_fractal_tokenizer() - creates custom tokenizer
3. download_and_export_rugpt3_with_custom_tokenizer() - exports ruGPT3
4. create_fractal_integration() - creates fractal_integration.py

**Problems:**
- Downloads sberbank-ai/rugpt3large_based_on_gpt2 - HUGE model
- Creates new fractal tokenizer architecture - NOT integrated with system
- Generates fractal_integration.py with code to add to manager - file not applied
- Uses export_hf_model_to_fractal from eva_ai.mlearning.storage.fractal_store - not verified

### 2.7 export_qwen.py

**Purpose:** Export Qwen model to fractal storage

**Logic:**
- Uses ModelExporter
- Exports qwen3.5-0.8b

**Problems:**
- Path eva\\mlearning\\ - should be eva_ai\\mlearning\\
- Uses ModelExporter - not verified to exist

### 2.8 simple_test.py

**Purpose:** Simple Qwen generation test

**Logic:**
1. Loads qwen3.5-0.8b from transformers
2. Generates 100 tokens
3. Outputs speed and result

**Problems:**
- Path eva\\mlearning\\ - INCORRECT, should be eva_ai\\

---

## 3. PyTorch Adapter

### 3.1 TorchBatchAdapter

**Path:** eva_ai/adapters/torch_adapter.py (230 lines)

**Purpose:** Batch processing of PyTorch tensors with support for:
- Sequence padding (pad_1d)
- Collate function for batches
- Pin memory for GPU
- Prefetch functions
- Timeouts and limits by tokens/items

**Integration:**
`python
from eva_ai.core.batch_wrapper import BatchEnvelope, unwrap_for_adapter
from eva_ai.core.device_resolver import resolve_device, should_pin_memory
`

**Capabilities:**

| Method | Description |
|--------|-------------|
| push(item) | Adds item to buffer |
| try_pop_batch() | Extracts batch by limits (max_items, max_tokens, timeout_ms) |
| flush() | Force extracts all items |

**Configuration:**
`python
TorchBatchAdapter(
    max_items=32,
    max_tokens=None,
    timeout_ms=5,
    key_for_length=input_ids,
    pad_value=0,
    pin_memory=False,
    non_blocking=True,
    prefetch_fn=None,
    allowed_tensor_keys=[input_ids, attention_mask, ...]
)
`

**Problems:**
- No integration with EVA AI pipeline (used for other purposes)
- allowed_tensor_keys hardcoded - not configurable via config
- No error handling for pin_memory() - silently fails

---

## 4. Overall Assessment

### 4.1 Configuration System

| Criterion | Score | Comment |
|-----------|-------|---------|
| Relevance | 4/10 | system_info hardcoded, conflict optimal vs fractal_model |
| Portability | 3/10 | Paths via os.getcwd(), not absolute |
| Usage | 2/10 | apply_optimal_config.py not called from CoreBrain |
| Integrity | 3/10 | Data duplication between JSON files |

**Recommendations:**
1. Remove optimal_config.json or make it source of truth
2. Use Path(__file__).parent.parent instead of os.getcwd()
3. Integrate apply_optimal_config.py into CoreBrain initialization

### 4.2 Scripts

| Script | Status | Problem |
|--------|--------|---------|
| migrate_kg_to_fg.py | BROKEN | Calls non-existent kg_to_fg_migration |
| migrate_to_optimized.py | BROKEN | Calls non-existent optimized_fractal_model_manager |
| migrate_events.py | DOCS | Does not perform actions, only documentation |
| load_gguf_to_fg.py | UNKNOWN | Depends on fg_gguf_architecture_mapper |
| activate_max_cache.py | UNKNOWN | Depends on unified_fractal_manager |
| complete_fractal_solution.py | DEPRECATED | Creates separate architecture, not integrated |
| export_qwen.py | UNKNOWN | Depends on ModelExporter |
| simple_test.py | WORKS | Direct transformers usage |

**Recommendations:**
1. Delete or fix migrate_kg_to_fg.py, migrate_to_optimized.py
2. Document dependencies in each script
3. Move hardcoded paths to configuration

### 4.3 PyTorch Adapter

| Criterion | Score | Comment |
|-----------|-------|---------|
| Code Quality | 8/10 | Clean, readable code |
| Functionality | 7/10 | Full set for batch processing |
| Integration | 3/10 | Not used in EVA AI pipeline |
| Documentation | 5/10 | Minimal docstrings |

**Recommendations:**
1. Integrate into generation pipeline
2. Add to config/adapters.json settings
3. Document usage with BatchEnvelope

### 4.4 Architectural Problems

1. Configuration as decoration: apply_optimal_config.py only prints, does not apply
2. Scripts as isolated utilities: No integration into main code
3. Hardcoded paths everywhere: Dependency on specific machine
4. Dead code: complete_fractal_solution.py creates what is not used
5. Gap between configs: optimal_config.json not related to fractal_model_config.json

---

## 5. Summary

Configuration System:     3/10 - Fragmented, not integrated, hardcoded paths
Scripts (8 files):        3/10 - Mostly broken or deprecated, poor integration
PyTorch Adapter:           7/10 - Quality code, poor integration with EVA
Overall Architecture:      4/10 - Many isolated components, low cohesion

Priority Fixes:
1. Remove duplication optimal_config.json vs fractal_model_config.json
2. Fix or delete migrate_kg_to_fg.py and migrate_to_optimized.py
3. Integrate TorchBatchAdapter into generation pipeline
4. Fix paths to absolute via Path(__file__).parent
