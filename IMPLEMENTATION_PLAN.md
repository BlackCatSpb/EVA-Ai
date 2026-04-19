# EVA AI Architecture Improvements - Implementation Plan

## Priority Queue

### Phase 1: Quick Wins (1-2 days)
1. **Smart LoRA Routing** - Auto-select LoRA based on query analysis
2. **Fix integration issues** - Current bugs

### Phase 2: Performance (3-5 days)
3. **Parallel Model Launch** - A || B simultaneously
4. **Streaming A→B** - Partial token transfer

### Phase 3: Advanced (1-2 weeks)
5. **Ensemble Mode** - Logits merging
6. **KV-cache Persistence**

---

## Phase 1: Quick Wins

### 1.1 Smart LoRA Routing

**Goal:** Automatically select best LoRA based on query analysis.

**Current state:**
- Manual force_mode parameter
- Fixed LoRA per session

**Implementation:**

```python
# eva_ai/core/smart_lora_router.py - NEW FILE

class SmartLoRARouter:
    """Analyzes query and selects optimal LoRA."""
    
    # Query patterns
    PATTERNS = {
        'code': ['код', 'функци', 'def ', 'class ', 'import ', 'программ'],
        'creative': ['придумай', 'напиши', 'сочини', 'история', 'стих', 'рассказ'],
        'knowledge': ['что такое', 'кто такой', 'объясни', 'почему', 'как работает'],
        'logic': ['логика', 'рассуждение', 'анализ', 'сравни'],
    }
    
    def select_lora(self, query: str) -> str:
        """Analyze query and return best LoRA name."""
        query_lower = query.lower()
        scores = {}
        
        for lora_name, patterns in self.PATTERNS.items():
            score = sum(1 for p in patterns if p in query_lower)
            scores[lora_name] = score
        
        best = max(scores, key=scores.get)
        return f'eva_{best}' if scores[best] > 0 else 'eva_knowledge'
```

**Files to modify:**
- `hybrid_dialog_manager.py` - Add SmartLoRARouter
- `generate_streaming()` - Use auto-selection

**ETA:** 2-4 hours

---

### 1.2 Fix Integration Issues (Current bugs)

**Bugs to fix:**
1. ✓ TokenizedInputs error - DONE (fec39b3)
2. ✓ PipelineAdapter detection - DONE (3de70f5)
3. ✓ Fallback loops - Need verify

**ETA:** Done

---

## Phase 2: Performance

### 2.1 Parallel Model Launch

**Goal:** Run Model A and Model B simultaneously when independent.

**Current state:**
```
Query → A (wait) → B (wait) → Response
```

**Implementation:**

```python
# In hybrid_dialog_manager.py

async def generate_parallel(self, query):
    """Launch both models in parallel."""
    
    # Start both in threads
    result_a = asyncio.to_thread(generate_model_a, query)
    result_b = asyncio.to_thread(generate_model_b, query)
    
    # Wait for A first (for quick response)
    response_a = await result_a
    response_b = await result_b
    
    # Use A for streaming, B for validation
    return merge_responses(response_a, response_b)
```

**Trade-off:**
- Quick response from A immediately
- B provides depth/validation
- 50% faster perceived latency

**Files to modify:**
- `hybrid_dialog_manager.py` - Add `generate_parallel()`

**ETA:** 1 day

---

### 2.2 Streaming A→B Transfer

**Goal:** Transfer partial tokens from A to B without waiting.

**Current state:**
- Full A response → B prompt

**Implementation:**

```python
# Partial transfer - every N tokens
def token_callback(text):
    # Transfer to B every 50 tokens
    partial_buffer += text
    if len(partial_buffer) > 50:
        # Call B with partial context
        pipeline_b.generate(partial_context + partial_buffer)
```

**Files to modify:**
- `openvino_token_collector.py` - Add transfer callback
- `hybrid_dialog_manager.py` - Use partial transfer

**ETA:** 1-2 days

---

## Phase 3: Advanced

### 3.1 Ensemble Mode

**Goal:** Merge logits from both models for better output.

**Current state:**
- Sequential: A → B

**Implementation:**

```python
# eva_ai/core/ensemble_generator.py - NEW FILE

class EnsembleGenerator:
    """Merge logits from multiple models."""
    
    def generate(self, prompt, models):
        """Generate with ensemble."""
        
        # Get logits from each model
        logits_a = models[0].get_logits(prompt)
        logits_b = models[1].get_logits(prompt)
        
        # Average logits
        merged = (logits_a + logits_b) / 2
        
        # Decode
        return self.decoder.decode(merged)
```

**Challenge:** OpenVINO doesn't expose raw logits easily.

**Alternative:** Use last hidden states if available.

**Files to create:**
- `ensemble_generator.py`

**ETA:** 3-5 days (research needed)

---

### 3.2 KV-cache Persistence

**Goal:** Cache attention keys between sessions.

**Current state:**
- Each query starts from scratch

**Implementation:**

```python
class PersistentKVCache:
    """Save and load KV cache."""
    
    def save(self, session_id):
        """Save current KV state."""
        cache = self.model.get_kv_cache()
        self.storage.save(f"kv_{session_id}", cache)
    
    def load(self, session_id):
        """Load KV state."""
        cache = self.storage.load(f"kv_{session_id}")
        self.model.set_kv_cache(cache)
```

**Files to create:**
- `kv_cache_persistence.py`

**Challenge:** KV cache format varies by model.

**Alternative:** Use prefix caching (already implemented in EVA).

**ETA:** 1 week

---

## Priority Implementation Order

| # | Feature | Complexity | Priority | ETA |
|-----|---------|-----------|----------|-----|
| 1 | Smart LoRA Routing | ⭐ | HIGH | 4h |
| 2 | Parallel Launch | ⭐⭐ | HIGH | 1d |
| 3 | Streaming A→B | ⭐⭐ | MEDIUM | 2d |
| 4 | Ensemble | ⭐⭐⭐⭐ | LOW | 1w |
| 5 | KV Cache | ⭐⭐⭐ | LOW | 1w |

---

## File Structure After Changes

```
eva_ai/core/
├── hybrid_dialog_manager.py     # Main orchestrator
├── smart_lora_router.py      # NEW - Feature 1
├── parallel_generator.py   # NEW - Feature 2
├── ensemble_generator.py   # NEW - Feature 4
├── streaming_transfer.py   # NEW - Feature 3
├── token_streaming.py     # ✓ DONE
├── openvino_token_collector.py  # ✓ DONE
└── kv_cache_persistence.py # NEW - Feature 5
```

---

## Testing Plan

```bash
# Test Smart Routing
python -c "
router = SmartLoRARouter()
assert router.select_lora('напиши код') == 'eva_code'
assert router.select_lora('что такое') == 'eva_knowledge'
print('Smart Routing: OK')
"

# Test Parallel
# Start EVA, query "привет" - should respond immediately

# Test Ensemble
# Requires model.logits access - research phase
```

---

## Start Implementation?

Answer: Yes/No/Which feature first?