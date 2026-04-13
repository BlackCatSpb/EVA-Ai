# CogniFlex System Flow - Current Architecture

**Document Version:** 3.0  
**Date:** April 13, 2026  
**Status:** ACTIVE DEVELOPMENT

---

## 1. Query to Response Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            USER QUERY                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  brain_query.py: process_query()                                             │
│  ├── Preprocessing (entities, intents)                                      │
│  ├── knowledge_context = fractal_graph_v2.get_context_for_query()           │
│  │       │                                                                    │
│  │       ▼                                                                    │
│  │   semantic_search(query) → get relevant nodes                            │
│  │       │                                                                    │
│  │       ▼                                                                    │
│  │   concept_extractor.get_concepts_for_prompt()                            │
│  │       │                                                                    │
│  │       ▼                                                                    │
│  │   contradiction_generator.get_contradictions_for_prompt()                  │
│  └── full_prompt = query + knowledge_context + concepts + contradictions     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ModelAccessManager - Coordinates model access (priority queue + locking)     │
│  ├── request_access(priority, task_type, callback)                          │
│  └── Events: model.request, model.completed, model.failed                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  UnifiedGenerator + OpenVINOGenerator (LAZY LOAD)                          │
│  ├── CPU Device (Logic/Context): LATENCY mode, NUM_STREAMS=AUTO             │
│  └── GPU Device (Coder/Self-dialog): THROUGHPUT mode                       │
│                                                                              │
│  OpenVINOGeneratorRegistry - Singleton for GPU model sharing                │
│  └── If model exists on device → reuse existing LLMPipeline instance        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Response Processing                                                         │
│  ├── concept_extractor.extract_concepts(query, response)                   │
│  │       │                                                                    │
│  │       ▼                                                                    │
│  │   save_concept_to_graph() → FGv2                                         │
│  │   queue_concept_for_dialog() → SelfDialogLearning                        │
│  │                                                                            │
│  ├── contradiction_generator.generate_contradiction() (if new concept)        │
│  │       │                                                                    │
│  │       ▼                                                                    │
│  │   save_contradiction() → ContradictionManager                            │
│  │   queue_contradiction_for_resolution() → SelfDialogLearning                │
│  │                                                                            │
│  └── save_experience(query, response, model_used) → FGv2                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Self-Learning Flow (Updated)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CONCEPT EXTRACTION PIPELINE                              │
│                                                                              │
│  ┌─ Fast Level: ConceptExtractor (sync, on every query)                    │
│  │   extract_concepts(query, response)                                       │
│  │       → Frequency analysis of terms                                       │
│  │       → Save to FGv2 as 'concept' nodes                                  │
│  │       → Generate basic facts (is_a, has_property, can, related_to)        │
│  │                                                                            │
│  └─ Deep Level: ConceptMiner (async, on system.idle)                        │
│      _detect_semantic_gaps(clusters)                                        │
│          ΔC = min(1 - cos(μC, v))  # Semantic gap detection               │
│      _generate_hypothesis() → via LLM                                       │
│      _validate_candidate() → NLI, Ontology, Ethics, Web                    │
│      _integrate_candidate() → FGv2                                          │
│      Lifecycle: PROVISIONAL → CONFIRMED → STABLE → ARCHIVED                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  CONTRADICTION DETECTION PIPELINE                           │
│                                                                              │
│  ┌─ Pattern Level: ContradictionGenerator (sync, on concept extract)       │
│  │   generate_contradiction(concept_name, domain)                           │
│  │       → Template-based (positive vs negative viewpoints)                   │
│  │       → Generate conflicting_facts                                       │
│  │                                                                            │
│  └─ Analysis Level: ContradictionMiner (async, on system.idle)              │
│      _detect_candidate_pairs()                                              │
│          sim(u,v) ≥ 0.75 AND contra(u,v) ≥ 0.65                            │
│      _cluster_pairs() → transitive closure                                   │
│      _generate_formulation() → CONTRADICTION_TITLE, DESCRIPTION              │
│      _create_contradiction_node() → FGv2 with 'contradicts' edges          │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SELF DIALOG LEARNING                                      │
│  ┌─ DialogConceptsMixin (integrated in SelfDialogLearning)                  │
│  │   queue_concept_for_dialog()                                             │
│  │   queue_contradiction_for_resolution()                                    │
│  │                                                                            │
│  └─ _worker_loop()                                                          │
│      Priority: contradictions > concepts > conversation history               │
│                                                                              │
│      For Concept:                                                            │
│          _run_concept_dialog()                                               │
│              ASSISTANT → CRITIC → LEARNER → TEACHER                        │
│                                                                              │
│      For Contradiction:                                                       │
│          _run_contradiction_dialog()                                         │
│              ASSISTANT → CRITIC → LEARNER → TEACHER                        │
│                                                                              │
│      Results → extract_knowledge_from_cache() → facts for future responses  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Event Bus Integration (Updated)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         EVENT BUS (Central Coordinator)                       │
│                                                                              │
│  PUBLISHERS:                                                                │
│  ┌─ ConceptMiner: concept.confirmed, concept.integrated                    │
│  ┌─ ContradictionMiner: contradiction.detected, contradiction.resolved     │
│  ┌─ SelfDialogLearning: self_dialog.scheduled, self_dialog.completed        │
│  ┌─ ModelAccessManager: model.request, model.completed, model.failed       │
│  ┌─ GraphCurator: curator.knowledge_extracted, curator.graph_optimized   │
│                                                                              │
│  SUBSCRIPTIONS:                                                             │
│  ┌─ ConceptMiner subscribes to:                                             │
│  │       memory.graph_updated, memory.clustering_complete                    │
│  │       pipeline.complete, system.ready, system.idle                      │
│  │                                                                            │
│  ┌─ ContradictionMiner subscribes to:                                      │
│  │       memory.node_created, memory.graph_updated, system.idle            │
│  │                                                                            │
│  └─ SelfDialogLearning subscribes to:                                       │
│          curator.knowledge_extracted, curator.graph_optimized               │
│          system.idle, system.state_changed                                  │
│          concept.confirmed, contradiction.detected                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Deferred Command System

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              DEFERRED COMMAND SYSTEM (Centralized Task Queue)                │
│                                                                              │
│  Priority Queue:                                                             │
│  ├── CRITICAL (0): Immediate operations - query generation                   │
│  ├── HIGH (1): Urgent learning - concept mining, contradiction resolution   │
│  ├── NORMAL (2): Regular operations - dialog processing                     │
│  └── LOW (3): Background optimization - context compaction                  │
│                                                                              │
│  INTEGRATION:                                                               │
│  ┌─ ConceptMiner: _schedule_mining_if_needed() → add_command()            │
│  ┌─ ContradictionMiner: _schedule_check() → add_command()                 │
│  └─ SelfDialogLearning: _on_system_idle() → add_command()                  │
│                                                                              │
│  Features:                                                                  │
│  ├── Load Shedding: Drop LOW priority when overloaded                        │
│  ├── Recovery: Auto-restart failed commands                                 │
│  └── EventBus: Publishes command events for monitoring                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Model Access Coordination

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MODEL ACCESS MANAGER                                       │
│                                                                              │
│  Purpose: Prevent conflicts when multiple components need model access          │
│                                                                              │
│  Priority Queue:                                                             │
│  ├── CRITICAL: User queries (immediate)                                      │
│  ├── HIGH: Self-dialog, concept mining, contradiction mining                 │
│  ├── NORMAL: Background analysis                                            │
│  └── LOW: Optimization tasks                                                │
│                                                                              │
│  Flow:                                                                       │
│  1. request_access(priority, task_type, callback) → request_id              │
│  2. ModelAccessManager.acquire_lock() - blocks other requests               │
│  3. Execute: callback(*args, **kwargs)                                     │
│  4. Publish: model.completed / model.failed                                 │
│  5. Release lock → next request                                              │
│                                                                              │
│  EventBus Events:                                                            │
│  ├── model.request: {request_id, priority, task_type, queue_size}         │
│  ├── model.completed: {request_id, task_type, elapsed}                     │
│  └── model.failed: {request_id, task_type, error}                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. OpenVINO Generator Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  OPEN_VINO_GENERATOR_REGISTRY (Singleton)                     │
│                                                                              │
│  Purpose: Share GPU model between multiple generators to save VRAM           │
│                                                                              │
│  Key = (model_path, device)                                                 │
│  Value = OpenVINOGenerator instance with loaded LLMPipeline                │
│                                                                              │
│  ┌─ get_or_create(model_path, device, creator_fn)                          │
│  │       If exists: return cached instance, ref_count++                     │
│  │       If new: create via creator_fn, cache, ref_count=1                 │
│  │                                                                            │
│  └─ release(model_path, device) → ref_count--                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    OPEN_VINO_GENERATOR (Lazy Load)                           │
│                                                                              │
│  __init__():                                                                │
│  └── Sets attributes, does NOT load model                                   │
│                                                                              │
│  _load_model():                                                            │
│  └── Loads ov_genai.LLMPipeline on FIRST generate() call                   │
│                                                                              │
│  Devices:                                                                   │
│  ├── CPU: Logic/Context models → LATENCY mode, NUM_STREAMS=AUTO           │
│  └── GPU: Coder/Self-dialog models → THROUGHPUT mode                       │
│                                                                              │
│  SchedulerConfig:                                                           │
│  ├── cache_size: KV-cache size (GB)                                        │
│  ├── max_num_seqs: Parallel slots                                          │
│  ├── max_num_batched_tokens: Batch size                                    │
│  └── enable_prefix_caching: Reuse KV for common prefixes                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Component Initialization Order

```
1. CoreBrain.__init__()
   ├── event_bus = EventBus()
   ├── deferred_system = DeferredCommandSystem(max_workers=6)
   │       └── set_event_bus(event_bus)  # Global event bus for commands
   │
   ├── _init_managers()
   │       ├── resource_manager
   │       ├── config_manager
   │       └── memory_manager
   │
   ├── _init_unified_generator()
   │       ├── OpenVINOGenerator for CPU (NOT loaded yet - lazy)
   │       ├── OpenVINOGenerator for GPU (NOT loaded yet - lazy)
   │       └── ModelAccessManager (started)
   │
   └── _init_background()
           ├── concept_miner = ConceptMiner()
           │       └── start() → subscribes to EventBus
           ├── contradiction_miner = ContradictionMiner()
           │       └── start() → subscribes to EventBus
           └── self_dialog_learning = SelfDialogLearning()
                   └── start() → subscribes to EventBus

2. CoreBrain.initialize()
   ├── _subscribe_to_system_events()
   ├── component_initializer.initialize_components()
   ├── _initialize_memory_manager()
   ├── _connect_components()
   └── _start_post_init_services()
           ├── fractal_graph_v2.start()
           ├── graph_curator.start()
           └── Start services that were waiting for system ready
```

---

## 8. Data Flow Summary

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│         EventBus: query_received        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│         ModelAccessManager              │
│   request_access(CRITICAL, query)      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│       UnifiedGenerator.generate()        │
│   (Lazy loads OpenVINO if needed)       │
└─────────────────────────────────────────┘
    │
    ├─► CPU OpenVINO: Logic/Context
    └─► GPU OpenVINO: Coder/Self-dialog
    │
    ▼
┌─────────────────────────────────────────┐
│   ConceptExtractor.extract_concepts()    │
│   ContradictionGenerator.generate()      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│         EventBus Publications            │
│   concept.confirmed (if new concept)     │
│   contradiction.detected (if new)        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│       SelfDialogLearning queue           │
│   (Uses DeferredCommandSystem)            │
└─────────────────────────────────────────┘
```

---

## 9. Configuration

### brain_config.json (Key Sections)

```json
{
  "model": {
    "use_openvino": true,
    "cpu_device": "CPU",
    "gpu_device": "GPU.0",
    "general_model_path": "path/to/model.gguf",
    "code_model_path": "path/to/coder.gguf"
  },
  "concept_system": {
    "enabled": true,
    "extractor": { "enabled": true },
    "miner": {
      "enabled": true,
      "mining_interval_seconds": 3600,
      "priority_queue": "NORMAL"
    }
  },
  "contradiction_system": {
    "enabled": true,
    "generator": { "enabled": true },
    "miner": {
      "enabled": true,
      "check_interval_seconds": 3600,
      "similarity_threshold": 0.75,
      "contradiction_threshold": 0.65
    }
  },
  "self_dialog_learning": {
    "enabled": true,
    "auto_dialog_interval": 300,
    "auto_learning_interval": 60
  }
}
```

---

## 10. Key Files

| Component | File | Status |
|-----------|------|--------|
| CoreBrain | `eva_ai/core/core_brain.py` | ACTIVE |
| UnifiedGenerator | `eva_ai/core/unified_generator.py` | ACTIVE |
| OpenVINOGenerator | `eva_ai/core/openvino_generator.py` | ACTIVE |
| ModelAccessManager | `eva_ai/core/model_access_manager.py` | ACTIVE |
| OpenVINOGeneratorRegistry | `eva_ai/core/openvino_generator.py` | ACTIVE |
| DeferredCommandSystem | `eva_ai/core/deferred_command_system.py` | ACTIVE |
| EventBus | `eva_ai/core/event_bus.py` | ACTIVE |
| ConceptExtractor | `eva_ai/knowledge/concept_extractor.py` | ACTIVE |
| ConceptMiner | `eva_ai/knowledge/concept_miner.py` | ACTIVE |
| ContradictionGenerator | `eva_ai/contradiction/contradiction_generator.py` | ACTIVE |
| ContradictionMiner | `eva_ai/contradiction/contradiction_miner.py` | ACTIVE |
| SelfDialogLearning | `eva_ai/learning/dialog_core.py` | ACTIVE |
| DialogConceptsMixin | `eva_ai/learning/dialog_concepts.py` | ACTIVE |

---

## 11. Testing

```bash
# Test imports
python -c "from eva_ai.core.openvino_generator import get_openvino_registry; print('OK')"

# Test registry
python -c "
from eva_ai.core.openvino_generator import get_openvino_registry
r = get_openvino_registry()
print(r.get_stats())
"

# Test CoreBrain import
python -c "from eva_ai.core.core_brain import CoreBrain; print('OK')"

# Run EVA
cd C:\Users\black\OneDrive\Desktop\CogniFlex && python -m eva_ai
```

---

*Last Updated: April 13, 2026*
