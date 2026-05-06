# Eva Pie Architecture - Two Qwen 3 Models

## Status: CONFIGURED

### Models Setup

**LOGIC Model (Condensed):**
- Path: `eva_pie_architecture/models/gguf_models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf`
- Purpose: Logic, reasoning, short answers
- Context: 4096 tokens

**CONTEXT Model (Extended):**
- Path: `eva_pie_architecture/models/gguf_models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m_model_b.gguf`
- Purpose: Long context, detailed answers
- Context: 32768 tokens

### Architecture Components

**1. UnifiedGenerator** (`eva_ai/core/unified_generator.py`)
```python
class ModelType(Enum):
    LOGIC = "logic"      # Model A - condensed
    CONTEXT = "context"  # Model B - extended

Features:
- Lazy loading (loads on first use)
- L2 routing (SimpleRouter)
- FractalGraph V2 integration
- Automatic model selection
```

**2. L2 Router** (`SimpleRouter`)
```python
CONTEXT_KEYWORDS = [
    'detailed', 'step by step', 'context',
    'analyze', 'compare', 'summary'
]

Route Logic:
- CONTEXT keywords -> CONTEXT model
- Default -> LOGIC model
```

**3. PipelineAdapter** (`eva_ai/core/pipeline_adapter.py`)
- Compatible with TwoModelPipeline interface
- Proxies calls to UnifiedGenerator
- Maintains backward compatibility

**4. Integration** (`brain_components.py`)
```python
_init_unified_generator(brain)
- Loads LOGIC model (condensed)
- Loads CONTEXT model (extended)
- Creates PipelineAdapter
- Sets two_model_pipeline_ready = True
```

### Configuration

```json
{
  "model": {
    "use_unified_generator": true,
    "logic_model_path": ".../qwen2.5-3b-instruct-q4_k_m.gguf",
    "context_model_path": ".../qwen2.5-3b-instruct-q4_k_m_model_b.gguf",
    "llama_cpp_n_ctx": 4096,
    "llama_cpp_threads": 4
  }
}
```

### Test Results

```
[OK] ModelType.LOGIC: logic
[OK] ModelType.CONTEXT: context
[OK] Router: 'Solve logic puzzle' -> LOGIC
[OK] Router: 'Explain in detail' -> CONTEXT
[OK] Generation: 63 tokens in 9.67s
```

### Files Modified

1. `eva_ai/core/unified_generator.py`
   - ModelType: LOGIC + CONTEXT
   - SimpleRouter with CONTEXT_KEYWORDS
   - _load_model_paths for dual models

2. `eva_ai/core/pipeline_adapter.py`
   - create_pipeline_adapter with logic/context paths

3. `eva_ai/core/brain_components.py`
   - _init_unified_generator with dual model setup

4. `eva_ai/core/__init__.py`
   - Export ModelType

5. `eva_ai/core/pie_model_paths.py`
   - Paths for qwen_3b condensed + extended

### Usage

```python
from eva_ai.core import UnifiedGenerator

gen = UnifiedGenerator()

# Automatic routing
result = gen.generate("Solve this logic problem")  # Uses LOGIC
result = gen.generate("Explain in detail...")      # Uses CONTEXT

# Force specific model
from eva_ai.core.unified_generator import ModelType
result = gen.generate("Query", force_model=ModelType.CONTEXT)
```

### Summary

✅ Two physical Qwen 3 models configured
✅ LOGIC (condensed) for reasoning
✅ CONTEXT (extended) for long contexts
✅ L2 routing automatic selection
✅ Full CoreBrain integration
✅ Backward compatibility maintained

System ready!
