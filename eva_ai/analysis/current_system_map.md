# CogniFlex System Architecture Analysis

**Generated:** March 21, 2026
**Analysis Type:** Deep Code Analysis

---

## PART 1: Current System Structure Map

### 1.1 CoreBrain (core_brain.py)

| Method | Line | Purpose | Status |
|--------|------|---------|--------|
| __init__ | 94-356 | Initializes subsystems | ACTIVE |
| initialize | 420-597 | Initializes components | ACTIVE |
| _initialize_memory_manager | 357-387 | Memory init | ACTIVE |
| process_query | 794-936 | Query processing | ACTIVE |
| _generate_basic_fallback_response | 937-964 | Keyword fallback | STUB |
| start | 983-1067 | Start components | ACTIVE |
| stop | 1130-1180 | Stop components | ACTIVE |
| check_module_dependencies | 668-686 | Not called | STUB |
| get_module_activity | 654-666 | Not called | STUB |

**CONTRADICTION:** process_query claims 7 fallback levels but only 6 + keyword

### 1.2 QueryProcessor (query_processor.py)

| Method | Line | Purpose | Status |
|--------|------|---------|--------|
| process_query | 69-231 | Query pipeline | ACTIVE |
| _process_nlp | 233-309 | NLP+caching | ACTIVE |
| _search_knowledge_graph | 353-406 | KG search | ACTIVE |
| _parallel_search | 431-501 | Parallel search | ACTIVE |
| _generate_response | 503-540 | Response gen | ACTIVE |
| _check_ethics | 542-577 | Ethics check | ACTIVE |
| _check_contradictions | 579-606 | Contradictions | ACTIVE |
| _store_insight | 625-647 | Store insight | ACTIVE |

**COMPONENT DEPS:** hybrid_cache, web_search_engine, ml_unit, ethics_framework

### 1.3 ResponseGenerator (response_generator.py)

| Method | Line | Purpose | Status |
|--------|------|---------|--------|
| generate_response | 428-493 | Main generation | ACTIVE |
| _generate_fractal_response | 503-525 | Fractal gen | ACTIVE |
| _create_fallback_response | 706-735 | Fallback | ACTIVE |
| _map_reduce_context | 743-747 | Stub - unused | STUB |

**CONTRADICTION:** Line 528-534 calls brain.process_query recursively

### 1.4 ReasoningEngine (reasoning_engine.py)

| Method | Line | Purpose | Status |
|--------|------|---------|--------|
| reason | 112-218 | Main reasoning | ACTIVE |
| _memory_retrieval | 317-376 | Memory retrieval | ACTIVE |
| _check_contradictions | 378-411 | Check contradictions | ACTIVE |
| _web_search_if_needed | 413-441 | Web search | ACTIVE |
| _ethics_check | 510-538 | Ethics check | ACTIVE |
| _synthesize_answer | 734-767 | Synthesis | ACTIVE |
| _finalize_answer | 994-1024 | Finalize | ACTIVE |

**11 REASONING PHASES:** INITIAL_ANALYSIS, MEMORY_RETRIEVAL, CONTRADICTION_CHECK, WEB_SEARCH, KNOWLEDGE_GRAPH_QUERY, ETHICS_CHECK, ANALYTICS_CHECK, PERFORMANCE_ANALYSIS, SYNTHESIS, REFLECTION, FINAL_ANSWER

**CONTRADICTION:** Line 999 ignores synthesis results from line 767

### 1.5 GenerationCoordinator (generation_coordinator.py)

| Provider | Priority |
|---------|----------|
| HybridModelProvider | 1 |
| FractalModelProvider | 1 |
| ResponseGeneratorProvider | 2 |
| MLUnitProvider | 3 |

---

## PART 2: Manager Integration Map

### EthicsFramework - FULLY IMPLEMENTED

| Method | Called By | Status |
|--------|-----------|--------|
| analyze_content | QueryProcessor, ReasoningEngine, SelfDialogLearning | ACTIVE |
| analyze_request | Internal wrapper | ACTIVE |
| analyze_response | Not called | STUB |

### ContradictionManager - FULLY IMPLEMENTED

| Method | Called By | Status |
|--------|-----------|--------|
| detect_contradictions | QueryProcessor, ReasoningEngine, SelfDialogLearning | ACTIVE |
| get_contradiction_stats | CoreBrain | ACTIVE |
| resolve_contradiction | Not called | STUB |

**ISSUE:** QueryProcessor uses contradiction_resolver, ReasoningEngine uses contradiction_manager

### WebSearchEngine - FULLY IMPLEMENTED

| Method | Called By | Status |
|--------|-----------|--------|
| search | QueryProcessor, ReasoningEngine | ACTIVE |
| search_async | Not called | ACTIVE |
| web_search_and_learn | Not called | STUB |

### AnalyticsManager - HAS MISSING METHODS

| Method | Called By | Status |
|--------|-----------|--------|
| get_learning_insights | SelfDialogLearning | ACTIVE |
| get_performance_report | ReasoningEngine | ACTIVE |
| get_system_analytics | ReasoningEngine | MISSING |
| get_metrics | ReasoningEngine | MISSING |

### MemoryManager - PARTIALLY INTEGRATED

| Method | Called By | Status |
|--------|-----------|--------|
| add_to_episodic | SelfDialogLearning | STUB |
| store | SelfDialogLearning | STUB |
| get_hot_window_data | ReasoningEngine | MAY_NOT_EXIST |

---

## PART 3: SelfDialogLearningSystem Analysis

### 3.1 Managers Accessed (lines 141-148)

- _model_manager = fractal_model_manager OR model_manager
- _memory_manager = memory_manager
- _knowledge_graph = knowledge_graph
- _ethics_framework = ethics_framework
- _contradiction_manager = contradiction_manager
- _web_search_engine = web_search_engine
- _analytics_manager = analytics_manager

### 3.2 Manager Initialization in CoreBrain

Most managers are accessed via getattr() but NOT explicitly set in CoreBrain.__init__().
They are set via component_initializer.initialize_components().

**CRITICAL ISSUE:** Race condition - SelfDialogLearning may access managers before init.

### 3.3 Critical Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| MemoryManager.add_to_episodic missing | HIGH | Insights not stored |
| WebSearch result format mismatch | MEDIUM | Fact verify fails |
| Manager initialization race | HIGH | Returns None |

### 3.4 Graceful Fallbacks

- _check_ethics: Returns {approved: True} if unavailable (line 475)
- _detect_contradictions: Returns [] if unavailable (line 504)
- _verify_facts: Returns empty if unavailable (line 550)
- _analyze_quality: Falls back to direct analysis (line 573)

---

## PART 4: Contradictions Summary

| # | Location | Contradiction |
|---|----------|--------------|
| 1 | process_query:794 | Claims 7 fallbacks, implements 6+keyword |
| 2 | _generate_response:528 | Calls brain.process_query recursively |
| 3 | _finalize_answer:999 | Ignores _synthesize_answer results |
| 4 | _map_reduce_context:743 | Returns input unchanged, never used |
| 5 | contradiction_resolver vs contradiction_manager | Same functionality, different paths |

---

## PART 5: Stub Methods

| Module | Method | Issue |
|--------|--------|-------|
| CoreBrain | check_module_dependencies | Never called |
| CoreBrain | get_module_activity | Never called |
| ResponseGenerator | _map_reduce_context | Returns unchanged |
| EthicsFramework | analyze_response | Wraps analyze_content |
| AnalyticsManager | get_system_analytics | Called but missing |
| AnalyticsManager | get_metrics | Called but missing |
| WebSearchEngine | web_search_and_learn | Never called |

---

## PART 6: Recommendations

### HIGH PRIORITY:
1. Fix manager initialization order - component_initializer must run before SelfDialogLearning
2. Implement MemoryManager.add_to_episodic() and store() methods
3. Fix WebSearch result format mismatch in SelfDialogLearning line 525

### MEDIUM PRIORITY:
4. Fix ReasoningEngine._finalize_answer to use _synthesize_answer results
5. Unify contradiction_resolver/contradiction_manager - use single attribute
6. Fix QueryProcessor recursive call risk in _generate_response

### LOW PRIORITY:
7. Remove/implement stub methods
8. Document Manager APIs and expected return types

---

*End of Analysis*