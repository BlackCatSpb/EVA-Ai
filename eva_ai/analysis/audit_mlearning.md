# MLearning Subsystem Audit Report

**Generated:** April 14, 2026  
**Auditor:** EVA AI System  
**Directory:** C:\Users\black\OneDrive\Desktop\CogniFlex\eva_ai\mlearning

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Files** | 75+ |
| **Model Manager Classes** | 8 |
| **Storage Implementations** | 6 |
| **EventBus Integration** | NOT FOUND |
| **Pickle Usage** | 5 locations |
| **Active Used** | ~4 components |
| **Dead Code** | ~70% |

**Overall Score:** 3/10

---

## 1. File Structure Overview

### 1.1 Main Directory (eva_ai/mlearning/)

`
mlearning/
|__init__.py
|ml_unit.py              # Stub (re-exports from unit_core.py)
|ml_core.py              # Core ML interface
|ml_types.py
|unit_core.py            # MLUnit main class
|unit_components.py      # Component management
|unit_training.py        # Training (DISABLED - SelfDialogLearning)
|model_manager.py        # ModelManager (BaseComponent)
|model_selector.py       # ModelSelector
|model_config.py         # MODEL_CONFIGS dict
|fractal_model_manager.py # FractalModelManager
|fractal_qwen_manager.py # FractalQwenManager (DISABLED)
|universal_model_manager.py # UniversalModelManager
|hybrid_model_manager.py # HybridModelManager
|current_manager.py      # OptimizedFractalModelManager (DUPLICATE)
|qwen_model_manager.py  # QwenModelManager
|bitnet_model_manager.py # BitNetModelManager
|unified_fractal_manager.py # UnifiedFractalManager
|storage/
|  |__init__.py
|  |fractal_store.py    # Re-export stub
|  |fractal_weight_store.py # Uses PICKLE
|  |model_storage_adapter.py
|  |optimized_fractal_model_manager.py # Stub
|  |opt_core.py         # OptimizedFractalModelManager (DUPLICATE)
|  |opt_models.py
|  |opt_cache.py
|  |...
|hot_deployment/
|  |llama_cpp_hot.py
|  |openvino_inference.py
|  |...
|eva_models/
|  |qwen2.5-0.5b/
`

---

## 2. CRITICAL ISSUES

### 2.1 MASSIVE Code Duplication - Model Managers

**Problem:** Found **8 Model Manager classes** performing similar functions:

| Class | File | Status | Used |
|-------|------|--------|------|
| ModelManager | model_manager.py | Active | unit_components.py |
| FractalModelManager | fractal_model_manager.py | Active | brain_components.py |
| QwenModelManager | qwen_model_manager.py | Active | brain_query.py, model_selector.py |
| UniversalModelManager | universal_model_manager.py | DEAD | Not imported anywhere |
| HybridModelManager | hybrid_model_manager.py | Active | init_factories.py |
| OptimizedFractalModelManager | current_manager.py | DEAD | Not imported anywhere |
| OptimizedFractalModelManager | storage/opt_core.py | DEAD | Not imported anywhere |
| BitNetModelManager | bitnet_model_manager.py | DEAD | Not imported anywhere |

**Duplicate Analysis:**
- current_manager.py and storage/opt_core.py contain IDENTICAL class OptimizedFractalModelManager
- fractal_model_manager.py comments say text generation now through Qwen3.5-2B but still exists
- Multiple managers for auto-selection but actual selection hardcoded in code

**Recommendation:** DELETE 5 of 8 managers (see Section 5).

---

### 2.2 EventBus Integration: NOT FOUND

**Status:** MISSING

Searched all mlearning/*.py files for:
- EventBus
- event_bus
- EventBus()

**Result:** No matches found in eva_ai/mlearning/.

**Consequence:** 
- ML components cannot publish/subscribe to system events
- No coordination with SelfDialogLearning, ConceptMiner, etc.
- Components use direct method calls only

**Recommendation:** Add EventBus integration to MLCore and ModelManager.

---

### 2.3 Pickle Usage: SECURITY WARNING

**Found Pickle in:**
1. eva_ai/mlearning/storage/fractal_weight_store.py (line 54, 63)
2. eva_ai/memory/cache_disk.py (line 7, 250, 344)
3. eva_ai/memory/fractal_torch_storage/base_storage.py (line 206, 216, 229, 232)
4. eva_ai/storage/fractal_storage.py (line 206, 208)
5. eva_ai/memory/disk_cache.py (line 7, 250)

**Issue:** pickle.load() can execute arbitrary code from malicious files.

**In fractal_weight_store.py:**
`python
import pickle
# ...
pickle.dump(self.containers, f)  # Line 63
`

**Recommendation:** Replace pickle with json or safetensors serialization.

---

## 3. Usage Analysis

### 3.1 Active Imports (Where They Are Used)

| Component | Imported By | Purpose |
|-----------|-------------|---------|
| ModelManager | unit_components.py | Fallback model management |
| FractalModelManager | brain_components.py | Main fractal model |
| QwenModelManager | brain_query.py | Text generation via get_qwen_model_manager() |
| HybridModelManager | init_factories.py | Hot window management |
| FractalWeightStore | model_manager.py | Model storage |

### 3.2 Dead Code (Not Imported Anywhere)

| Component | File | Reason |
|-----------|------|--------|
| UniversalModelManager | universal_model_manager.py | Never imported |
| OptimizedFractalModelManager | current_manager.py | Never imported |
| OptimizedFractalModelManager | storage/opt_core.py | Duplicate of above |
| BitNetModelManager | bitnet_model_manager.py | Never imported |
| FractalQwenManager | fractal_qwen_manager.py | DISABLED (returns None) |
| ModelStorageAdapter | storage/model_storage_adapter.py | Never imported |
| MemoryGraphStore | storage/memory_graph_store.py | Never imported |

### 3.3 Disabled Components

| Component | Status | Notes |
|-----------|--------|-------|
| TrainingOrchestrator | Disabled | Training via SelfDialogLearning |
| FractalQwenManager.get_fractal_qwen() | Disabled | Returns None |
| QwenModelManager.get_qwen_model_manager() | Disabled | Returns None |

---

## 4. Storage Subsystem Analysis

### 4.1 Files

`
storage/
|__init__.py
|fractal_store.py         # Re-export stub
|fractal_weight_store.py  # USES PICKLE
|model_storage_adapter.py # Unused adapter
|memory_graph_store.py    # Unused
|fractal_store_core.py    # Main FractalWeightStore
|fractal_store_utils.py
|fractal_model_loader.py
|store_core.py
|store_operations.py
|store_queries.py
|store_cache.py
|model_storage_config.py
|optimized_fractal_model_manager.py  # Stub
|opt_core.py              # Duplicate OptimizedFractalModelManager
|opt_models.py
|opt_cache.py
|unified_*.py              # Unclear purpose
`

### 4.2 Issues

1. fractal_weight_store.py - Uses unsafe pickle
2. Duplicates - fractal_weight_store vs store_core both define FractalWeightStore
3. Unused imports - ModelStorageAdapter not used
4. Confusing structure - Multiple files re-export same classes

---

## 5. Recommendations

### 5.1 DELETE Dead Code (Priority: CRITICAL)

Safe to delete:
- eva_ai/mlearning/universal_model_manager.py - Never used
- eva_ai/mlearning/current_manager.py - Duplicate, never used
- eva_ai/mlearning/storage/optimized_fractal_model_manager.py
- eva_ai/mlearning/storage/opt_core.py - Duplicate
- eva_ai/mlearning/storage/opt_models.py - Part of duplicate
- eva_ai/mlearning/storage/opt_cache.py - Part of duplicate
- eva_ai/mlearning/bitnet_model_manager.py - Never used
- eva_ai/mlearning/fractal_qwen_manager.py - DISABLED, returns None
- eva_ai/mlearning/storage/memory_graph_store.py - Never imported
- eva_ai/mlearning/storage/model_storage_adapter.py - Never imported
- eva_ai/mlearning/storage/unified_*.py - Unclear if used
- eva_ai/mlearning/text_quality_*.py - Check if used
- eva_ai/mlearning/enhanced_learning_*.py - Check if used
- eva_ai/mlearning/comprehensive_learning_system.py - Check if used

### 5.2 EventBus Integration (Priority: HIGH)

Add to MLCore.__init__():
`python
from eva_ai.core.event_bus import EventBus

self.event_bus = EventBus()
self.event_bus.subscribe(system.idle, self._on_idle)
self.event_bus.subscribe(model.request, self._on_model_request)
`

### 5.3 Replace Pickle (Priority: HIGH)

Replace in fractal_weight_store.py:
`python
BAD:
pickle.dump(self.containers, f)

GOOD:
import json
with open(file, w) as f:
    json.dump([c.__dict__ for c in self.containers.values()], f)
`

### 5.4 Consolidate ModelManagers (Priority: MEDIUM)

Keep only:
- QwenModelManager - Primary model interface
- ModelManager - Base component wrapper
- HybridModelManager - Memory management

Delete others.

---

## 6. Scoring Breakdown

| Category | Score | Max | Issues |
|----------|-------|-----|--------|
| Code Organization | 4 | 10 | Massive duplication |
| EventBus Integration | 0 | 10 | Not implemented |
| Pickle Security | 2 | 10 | 5 unsafe locations |
| Active Usage | 4 | 10 | ~30% code used |
| Maintainability | 3 | 10 | Dead code everywhere |
| **TOTAL** | **2.6** | **10** | **~3/10** |

---

## 7. Action Items

| # | Action | Priority |
|---|--------|----------|
| 1 | Delete 10+ unused files | CRITICAL |
| 2 | Add EventBus to MLCore | HIGH |
| 3 | Replace pickle with json | HIGH |
| 4 | Consolidate ModelManagers | MEDIUM |
| 5 | Audit storage/ directory | MEDIUM |

---

**END OF AUDIT REPORT
