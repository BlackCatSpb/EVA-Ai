# EVA AI Generation System - Implementation Plan V2
# Based on detailed code analysis

## Current Architecture Issues

| Issue | Root Cause | Impact |
|-------|-----------|-------|
| Duplicate code | `_init_two_model_pipeline()` duplicated | Maintenance burden |
| Threading blocking | Uses `threading.Thread` instead of async | Race conditions |
| No raw tokens | OpenVINO doesn't expose token IDs | A→B transfer loses precision |
| Complex initialization | 7+ abstraction layers | Hard to debug |
| No async | Sync generation blocks pipeline | Low throughput |

---

## Recommended Architecture

```
                    ┌─────────────────┐
                    │  brain_query.py │
                    └───────┬─────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │   GenerationCoordinator (NEW)            │
        │   - Async orchestration                  │
        │   - Model pool management               │
        │   - KV cache coordination              │
        └───────────────────┬───────────────────┘
                          │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │ Model A  │    │ Model B  │    │ Model C  │
    │ (LOGIC)  │    │(CONTEXT) │    │ (CODER)  │
    └───────────┘    └───────────┘    └───────────┘
```

---

## Phase 1: Quick Fixes (Current bugs)

### 1.1 Async Migration (hybrid_dialog_manager.py)

**Problem:** Uses blocking `threading.Thread`

**Fix:** Replace with `asyncio.to_thread()`

```python
# BEFORE
thread_a = threading.Thread(target=generate_model_a, daemon=True)
thread_a.start()

# AFTER  
result_a = await asyncio.to_thread(generate_model_a, prompt)
```

**Files:** `hybrid_dialog_manager.py`

**ETA:** 2 hours

---

### 1.2 Unified Pipeline Access

**Problem:** Can't access `_openvino_cpu` from HybridKnowledgeDialogManager

**Fix:** Add getter in pipeline_adapter.py

```python
# In PipelineAdapter
@property
def openvino_models(self):
    return {
        'cpu': self._generator._openvino_cpu if self._generator else None,
        'gpu': self._generator._openvino_gpu if self._generator else None
    }
```

**Files:** `pipeline_adapter.py`, `hybrid_dialog_manager.py`

**ETA:** 1 hour

---

## Phase 2: Performance

### 2.1 Parallel Generation

**Goal:** Run A and B in parallel, not sequential

```python
async def generate_dual_parallel(prompt):
    # Spawn both models simultaneously
    task_a = asyncio.to_thread(generate_logic, prompt)
    task_b = asyncio.to_thread(generate_context, prompt)
    
    # Wait for quick response from A
    result_a = await asyncio.wait_for(task_a, timeout=10)
    
    # Use A result for streaming
    yield result_a
    
    # Get B result for validation
    result_b = await task_b
    yield validate_and_merge(result_a, result_b)
```

**Expected improvement:** -50% perceived latency

**Files:** `hybrid_dialog_manager.py`

**ETA:** 4 hours

---

### 2.2 Token Caching

**Goal:** Cache tokens between A→B transfer

```python
class TokenCache:
    def __init__(self):
        self._cache = {}
    
    def get_tokens(self, session_id):
        return self._cache.get(session_id)
    
    def set_tokens(self, session_id, tokens):
        self._cache[session_id] = tokens
```

**Files:** `token_streaming.py`

**ETA:** 2 hours

---

## Phase 3: Advanced Features

### 3.1 Smart LoRA Routing (NEW)

**Current:** Manual selection via `force_mode`

**Goal:** Auto-select based on query

```python
class LoRARouter:
    """Auto-select based on query."""
    
    ROUTES = {
        'code': ['код', 'функц', 'def ', 'import ', 'class '],
        'creative': ['придумай', 'напиши', 'сочини', 'истор'],
        'knowledge': ['что такое', 'объясни', 'почему', 'как работает'],
        'logic': ['логика', 'рассужд', 'сравни', 'анализ'],
    }
    
    def route(self, query: str) -> str:
        query_lower = query.lower()
        scores = {name: sum(1 for p in patterns if p in query_lower) 
                 for name, patterns in self.ROUTES.items()}
        best = max(scores, key=scores.get)
        return f'eva_{best}' if scores[best] > 0 else 'eva_knowledge'
```

**Files:** NEW `smart_lora_router.py`

**ETA:** 2 hours

---

### 3.2 Generation Coordinator (NEW)

Central orchestrator for all generation:

```python
class GenerationCoordinator:
    """Central coordinator for model generation."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.token_cache = TokenCache()
        self.lora_router = LoRARouter()
    
    async def generate(self, query, config):
        # Auto-select LoRA
        lora = self.lora_router.route(query)
        
        # Get cached tokens if available
        cached = self.token_cache.get_tokens(query.hash)
        
        # Parallel generation
        result = await self._generate_parallel(query, lora, cached)
        
        # Cache result
        self.token_cache.set_tokens(query.hash, result.tokens)
        
        return result
```

**Files:** NEW `generation_coordinator.py` 

**ETA:** 1 day

---

## Phase 4: Architecture Cleanup

### 4.1 Remove Duplicate Code

**Problem:** `_init_two_model_pipeline()` appears twice

**Fix:**

```python
# brain_components.py
# Keep only ONE function
def _init_pipeline(brain):
    """Single entry point."""
    config = brain.config.get('model', {})
    
    if config.get('use_unified_generator', True):
        _init_unified_generator(brain)
    else:
        _init_legacy_pipeline(brain)  # Single function
```

**Files:** `brain_components.py`

**ETA:** 1 hour

---

### 4.2 Async Pipeline Initialization

**Problem:** Sync init blocks event loop

**Fix:**

```python
async def init_pipeline_async(brain):
    # Async model loading
    model = await asyncio.to_thread(load_model, path)
    return model
```

**Files:** CoreBrain initialization

**ETA:** 2 hours

---

## Implementation Checklist

| # | Task | Complexity | Files | ETA |
|-----|------|------------|-------|-----|
| 1 | Async threading fix | ⭐ | hybrid_dialog_manager.py | 2h |
| 2 | Pipeline access | ⭐ | pipeline_adapter.py | 1h |
| 3 | Parallel generation | ⭐⭐ | hybrid_dialog_manager.py | 4h |
| 4 | Token caching | ⭐ | token_streaming.py | 2h |
| 5 | Smart LoRA routing | ⭐ | NEW file | 2h |
| 6 | Generation coordinator | ⭐⭐ | NEW file | 1d |
| 7 | Remove duplicate code | ⭐ | brain_components.py | 1h |
| 8 | Async init | ⭐⭐ | core_brain.py | 2h |

---

## File Structure After

```
eva_ai/core/
├── __init__.py
├── brain_components.py      # Cleaned (remove duplicate)
├── pipeline_adapter.py     # Added openvino_models property
├── generation_coordinator.py  # NEW - Phase 3
├── smart_lora_router.py    # NEW - Phase 3
├── token_streaming.py       # Added caching
├── hybrid_dialog_manager.py # Async + Parallel
├── unified_generator.py
└── openvino_generator.py
```

---

## Testing

```bash
# Test parallel generation
python -c "
from eva_ai.core.generation_coordinator import GenerationCoordinator
coordinator = GenerationCoordinator(pipeline)

# Should run in parallel
result = await coordinator.generate('привет', {})
print(f'Response: {result.text}')
print(f'Models used: {result.models_used}')
"

# Test Smart LoRA
python -c "
from eva_ai.core.smart_lora_router import LoRARouter
router = LoRARouter()
assert router.route('напиши код') == 'eva_code'
assert router.route('что такое') == 'eva_knowledge'
print('Smart routing: OK')
"
```

---

## Start?

Confirm which tasks to begin: