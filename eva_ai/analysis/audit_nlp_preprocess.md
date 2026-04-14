# NLP & Preprocessing Audit

## 1. NLP System (text_processor.py)

### 1.1 Functions

**TextProcessor** (eva_ai/nlp/text_processor.py, 311 lines):
- **Tokenization**: tokenize(), encode(), decode()
- **Text preprocessing**: preprocess_text() - cleaning, space normalization, lowercasing
- **Batch processing**: batch_process()
- **Caching**: _encode_cached() with LRU cache (2048 entries)

**Uses**:
- Qwen2.5-0.5B tokenizer (default)
- TokenizerRegistry (singleton) for reuse
- Lazy loading of tokenizer

### 1.2 Integration

**Connection to brain_query**: MINIMAL
- NOT used directly in brain_query
- Connected via init_factories.create_text_processor()
- Attached to response_generator: text_processor, token_streamer, hybrid_cache
- Used in unified_text_processor (mlearning)

**Other integration points**:
- graph_ml_core.py - for getting tokenizer
- unit_components.py - TextProcessor initialization
- import_pipeline.py - normalize_text, segment_text

**EventBus**: NOT USED
- No event subscriptions
- No event publications

### 1.3 Issues

1. **STUB**: File eva_ai/nlp/text_processor.py is a legacy wrapper
2. **Actually works**: UnifiedTextProcessor from mlearning/unified_text_processor.py
3. **No entity extraction** - only tokenization
4. **brain_query does not use TextProcessor directly**
5. **No integration with reasoning**

---

## 2. Preprocessing Pipeline

### 2.1 Stages

**PreprocessingPipeline** (eva_ai/preprocess/preprocessing_pipeline.py, 335 lines):

| Stage | Method | Description |
|-------|--------|-------------|
| 1 | extract_entities() | Entity extraction via GGUF |
| 2 | check_clarification_needed() | Check if clarification needed |
| 3 | _extract_keywords() | Extract keywords |
| 4 | _build_raw_context() | Build context |
| 5 | _save_to_cache() | Save to hybrid cache |

**GGUFEntityExtractor**:
- Uses llama_instance (GGUF model)
- Russian prompts for entity extraction and clarification
- Fallback: simple capitalized word extraction

### 2.2 Integration

**brain_query**: USED
```python
# brain_query.py:587-611
if session_id and hasattr(self, preprocessing_pipeline) and self.preprocessing_pipeline:
    preprocessed_result = self.preprocessing_pipeline.process(
        query=query, session_context=session_context, session_id=session_id)
    # Returns clarification if needed
```

**Initialization** (brain_components.py:468-475):
```python
def _init_preprocessing(brain):
    brain.preprocessing_pipeline = None
    try:
        from ..preprocess.preprocessing_pipeline import PreprocessingPipeline
        llama_instance = brain.llama_cpp_deployment.llama if ...
        brain.preprocessing_pipeline = PreprocessingPipeline(llama_instance=llama_instance, hybrid_cache=brain.hybrid_cache)
    except Exception as e:
        query_logger.debug(f"PreprocessingPipeline not initialized: {e}")
```

**EventBus**: NOT USED
- No event subscriptions
- No event publications

### 2.3 Issues

1. **DEPENDENCY ON llama_cpp_deployment**: If llama not initialized - fallback to stub
2. **GGUFEntityExtractor - POTENTIAL STUB**: 
   - llama_instance passed but may be None
   - Fallback: _fallback_extraction() - trivial capitalized word extraction
3. **No integration with concept_extractor or concept_miner**
4. **PreprocessedQuery NOT used in further flow** - only for clarification

---

## 3. Entity Extraction - OTHER IMPLEMENTATIONS

There are THREE different EntityExtractors:

| Module | Class | Purpose |
|--------|-------|---------|
| preprocess/preprocessing_pipeline.py | GGUFEntityExtractor | Query preprocessing (GGUF) |
| reasoning/entity_extractor.py | EntityExtractor | Extraction for self-learning (regex) |
| gui/web_gui/server_main.py | EntityExtractor | GUI entity extraction |

**reasoning/entity_extractor.py** (398 lines):
- Pattern-based extraction (regex)
- Methods: extract_from_query(), extract_from_response(), extract_from_contradiction()
- Save to knowledge graph
- Used in: enhanced_reasoning_engine, contradiction_core_detection, memory_manager

---

## 4. Reasoning Integration

**preprocessing to reasoning connection**: ABSENT
- preprocessing_pipeline not connected to reasoning_engine
- Preprocessing results not passed to reasoning
- brain_query processes them separately

**brain_query flow** (simplified):
```
1. Preprocessing (if qwen_only_mode and session_id)
   -> clarification or entities
   
2. FractalGraphV2 context
   -> knowledge_context

3. Reasoning engines (if enabled)
   -> SelfReasoningEngine OR EnhancedReasoningEngine

4. Fallback: qwen_model_manager / llama_cpp
```

---

## 5. Overall Assessment

### Component Status

| Component | Status | Issues |
|-----------|--------|--------|
| TextProcessor | STUB | Legacy, not used in brain_query |
| UnifiedTextProcessor | WORKING | Real NLP processor |
| PreprocessingPipeline | PARTIALLY WORKING | Depends on llama_cpp_deployment |
| GGUFEntityExtractor | DEPENDENCY | Fallback to stub without GGUF |
| reasoning.EntityExtractor | WORKING | Pattern-based extraction |

### Architectural Problems

1. **Duplicate entity extraction**: 3 different implementations
2. **No unified preprocessing -> reasoning flow**
3. **PreprocessingResults ignored** after clarification check
4. **EventBus not used** in these modules
5. **brain_query - COUPLING**: everything in one method _handle_qwen_mode()

### What Actually Works

1. **Tokenization**: UnifiedTextProcessor (via TokenizerRegistry)
2. **Entity extraction**: reasoning.EntityExtractor (pattern-based)
3. **Clarification flow**: PreprocessingPipeline (if GGUF available)
4. **Context building**: FractalGraphV2 + knowledge_context

### Recommendations

1. Unify entity extraction into one module
2. Add EventBus events for preprocessing/reasoning
3. Use preprocessing results in reasoning
4. Remove duplication between TextProcessor and UnifiedTextProcessor