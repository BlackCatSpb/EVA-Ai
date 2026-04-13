# CogniFlex System Flow - Complete Architecture

**Document Version:** 4.0  
**Date:** April 14, 2026  
**Status:** ACTIVE DEVELOPMENT

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                     │
│                     Web Browser (Flask + JavaScript)                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Flask Server (WebGUI)                                 │
│  ├── /api/chat/stream - SSE streaming                                     │
│  ├── /api/sessions - Session management                                    │
│  ├── /api/knowledge-graph - FGv2 data                                     │
│  └── /api/analytics - Dashboard analytics                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         CoreBrain (Центральный координатор)                   │
│                                                                              │
│  Mixins: ConfigMixin, ComponentMixin, QueryMixin, MonitoringMixin          │
│          MemoryMixin, StateMixin, EventSubscriptionMixin                     │
│          CommandIssuerMixin, ProcessTrackerMixin                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│   EventBus          │   │ DeferredCommandSystem │   │ TwoModelPipeline     │
│   (Шина событий)    │   │ (Отложенные команды)   │   │ (UnifiedGenerator)  │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
                                      │                           │
                                      │           ┌───────────────┴───────────────┐
                                      │           │                               │
                                      │           ▼                               ▼
                                      │   ┌─────────────┐             ┌─────────────┐
                                      │   │  LOGIC      │             │  CODER      │
                                      │   │  (CPU)      │             │  (GPU)      │
                                      │   └─────────────┘             └─────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  ConceptExtractor    │   │  ContradictionGen   │   │ SelfDialogLearning │
│  (Fast, sync)      │   │  (Templates)        │   │ (DialogConcepts)    │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  ConceptMiner       │   │  ContradictionMiner │   │  FractalGraphV2    │
│  (Deep, async)     │   │  (Analysis)         │   │  (Storage)         │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

---

## 2. Query Processing Pipeline

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  brain_query.process_query()                                               │
│                                                                              │
│  1. Preprocessing: entities, intents                                        │
│  2. Context building:                                                       │
│     ├── concept_extractor.get_concepts_for_prompt()                         │
│     ├── contradiction_generator.get_contradictions_for_prompt()               │
│     └── extract_knowledge_from_cache()                                       │
│  3. Full prompt = query + concepts + contradictions + knowledge            │
└──────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ModelAccessManager.request_access()                                       │
│  Priority: CRITICAL (user query)                                           │
│  Events: model.request → model.completed / model.failed                     │
└──────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  UnifiedGenerator.generate_iterative()                                       │
│                                                                              │
│  1. LOGIC model (CPU) → краткий ответ                                   │
│  2. CONTEXT model (CPU) → расширенный ответ с контекстом                  │
│                                                                              │
│  OpenVINO devices (lazy loaded):                                            │
│  ├── CPU: Logic/Context → LATENCY mode, NUM_STREAMS=AUTO                 │
│  └── GPU: Coder/Self-dialog → THROUGHPUT mode                             │
└──────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Response Processing                                                       │
│                                                                              │
│  ├── concept_extractor.extract_concepts(query, response)                   │
│  │       → save_concept_to_graph() → FGv2                                 │
│  │       → queue_concept_for_dialog() → SelfDialogLearning                │
│  │                                                                            │
│  ├── contradiction_generator.generate_contradiction() (if new concept)        │
│  │       → save_contradiction() → ContradictionManager                  │
│  │       → queue_contradiction_for_resolution() → SelfDialogLearning        │
│  │                                                                            │
│  └── save_experience() → FGv2                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Concept System (Two-Level Architecture)

### Level 1: ConceptExtractor (Fast, Synchronous)

**File:** `eva_ai/knowledge/concept_extractor.py`

```python
class ConceptExtractor:
    def extract_concepts(query, response, context=None) -> List[Concept]
    
    # Flow:
    # 1. _extract_terms() → frequency analysis
    # 2. _create_concept() → domain, facts, related_terms
    # 3. save_concept_to_graph() → FGv2 node type='concept'
```

**Triggers:** Every query/response

**Output:** List[Concept] with facts (is_a, has_property, can, related_to)

---

### Level 2: ConceptMiner (Deep, Asynchronous)

**File:** `eva_ai/knowledge/concept_miner.py`

```python
class ConceptMiner:
    # Subscribes to: memory.graph_updated, memory.clustering_complete,
    #                pipeline.complete, system.ready, system.idle
    
    def _mining_cycle():
        # 1. _get_clusters() → from FGv2
        # 2. _detect_semantic_gaps() → ΔC = min(1 - cos(μC, v))
        # 3. _generate_hypothesis() → via LLM
        # 4. _validate_candidate() → NLI + Ontology + Ethics + Web
        # 5. _integrate_candidate() → FGv2
        # 6. queue_concept_for_dialog() → SelfDialogLearning
```

**Lifecycle:** PROVISIONAL → CONFIRMED → STABLE → ARCHIVED

**Configuration:**
```python
{
    "base_threshold": 0.30,
    "max_candidates_per_cycle": 3,
    "llm_temperature": 0.35,
    "enable_web_search_validation": True
}
```

---

## 4. Contradiction System (Two-Level Architecture)

### Level 1: ContradictionGenerator (Template-Based)

**File:** `eva_ai/contradiction/contradiction_generator.py`

```python
class ContradictionGenerator:
    def generate_contradiction(concept_name, domain='general') -> GeneratedContradiction
        # Uses templates for positive/negative viewpoints
        # Domain templates: general, technology, science, philosophy
    
    def get_contradictions_for_prompt(concept_name) -> str
        # Format for prompt context
```

---

### Level 2: ContradictionMiner (Analysis-Based)

**File:** `eva_ai/contradiction/contradiction_miner.py`

```python
class ContradictionMiner:
    # Subscribes to: memory.node_created, memory.graph_updated, system.idle
    
    def _detection_cycle():
        # 1. _detect_candidate_pairs()
        #    - sim(u,v) ≥ 0.75 AND contra(u,v) ≥ 0.65
        # 2. _cluster_pairs() → transitive closure
        # 3. _filter_and_prioritize()
        #    - priority = α·|C| + β·avg_confidence + γ·max_contra
        # 4. _generate_formulation() → LLM generation
        # 5. _validate_candidate()
        # 6. _create_contradiction_node() → FGv2 with 'contradicts' edges
```

**Configuration:**
```python
{
    "sim_threshold": 0.75,
    "contra_threshold": 0.65,
    "max_candidates_per_cycle": 5,
    "llm_temperature": 0.25
}
```

---

## 5. SelfDialogLearning System

**File:** `eva_ai/learning/dialog_core.py`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SelfDialogLearning (Main Class)                           │
│                                                                              │
│  Inherits: DialogTopicsMixin, DialogGenerationMixin,                        │
│             DialogLearningMixin, DialogConceptsMixin                          │
│                                                                              │
│  Queues:                                                                   │
│  ├── _concept_queue: concepts to discuss                                    │
│  ├── _contradiction_topics: contradictions to resolve                       │
│  └── dialog_queue: general dialogs                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Dialog Concepts Integration

**File:** `eva_ai/learning/dialog_concepts.py`

```python
class DialogConceptsMixin:
    def queue_concept_for_dialog(concept_name, priority=0.5)
    def queue_contradiction_for_resolution(contr_id, concept, priority=0.7)
    
    # Priority: contradictions > concepts > conversation history
    
    # 4-Turn Concept Dialog:
    # 1. ASSISTANT → presents concept
    # 2. CRITIC → finds problems
    # 3. LEARNER → proposes directions
    # 4. TEACHER → gives recommendations
    
    # 4-Turn Contradiction Dialog:
    # 1. ASSISTANT → presents both viewpoints
    # 2. CRITIC → analyzes sides
    # 3. LEARNER → finds synthesis
    # 4. TEACHER → formulates resolution
```

### EventBus Subscriptions

```python
# System events
"system.idle" → _on_system_idle
"system.state_changed" → _on_system_state_changed

# Concept events
"concept.confirmed" → _on_concept_confirmed

# Contradiction events
"contradiction.detected" → _on_contradiction_detected

# Curator events
"curator.knowledge_extracted" → _on_curator_knowledge_extracted
"curator.graph_optimized" → _on_curator_graph_optimized
"curator.cleanup_done" → _on_curator_cleanup_done
```

---

## 6. EventBus Architecture

**File:** `eva_ai/core/event_bus.py`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           EventBus                                          │
│                                                                              │
│  Priority: LOW(1) < NORMAL(2) < HIGH(3) < CRITICAL(4)                      │
│                                                                              │
│  Publishers:                                                               │
│  ├── ModelAccessManager: model.request, model.completed, model.failed        │
│  ├── ConceptMiner: concept.confirmed, concept.integrated                      │
│  ├── ContradictionMiner: contradiction.detected, contradiction.resolved     │
│  ├── SelfDialogLearning: self_dialog.scheduled, self_dialog.completed        │
│  ├── GraphCurator: curator.knowledge_extracted, curator.graph_optimized      │
│  └── DeferredCommandSystem: command.completed, command.failed               │
│                                                                              │
│  Key Methods:                                                               │
│  ├── publish(event_type, data)                                             │
│  ├── subscribe(event_type, handler, priority=5)                             │
│  ├── unsubscribe(event_type, handler_or_id)                                  │
│  └── get_subscribers_count(event_type) → int                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. DeferredCommandSystem Architecture

**File:** `eva_ai/core/deferred_command_system.py`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    DeferredCommandSystem                                      │
│                                                                              │
│  Priority Queues:                                                          │
│  ├── CRITICAL(0): Immediate operations (user queries)                        │
│  ├── HIGH(1): Urgent learning (concept/contradiction mining)                 │
│  ├── NORMAL(2): Regular operations (dialog processing)                       │
│  └── LOW(3): Background optimization (context compaction)                    │
│                                                                              │
│  Components:                                                               │
│  ├── PriorityQueue per priority level                                       │
│  ├── ThreadPoolExecutor (max_workers=6)                                    │
│  ├── Load Shedding: CPU/queue monitoring                                    │
│  └── Recovery: Auto-restart failed commands                                  │
│                                                                              │
│  Used By:                                                                   │
│  ├── ConceptMiner._schedule_mining_if_needed()                               │
│  ├── ContradictionMiner._schedule_check()                                   │
│  └── SelfDialogLearning._on_system_idle()                                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. ModelAccessManager Architecture

**File:** `eva_ai/core/model_access_manager.py`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ModelAccessManager (Singleton)                            │
│                                                                              │
│  Purpose: Coordinate model access, prevent conflicts                         │
│                                                                              │
│  Priority Queue:                                                            │
│  ├── CRITICAL: User queries (immediate)                                     │
│  ├── HIGH: Self-dialog, concept mining, contradiction mining                 │
│  ├── NORMAL: Background analysis                                           │
│  └── LOW: Optimization tasks                                               │
│                                                                              │
│  Flow:                                                                     │
│  1. request_access(priority, task_type, callback) → request_id              │
│  2. acquire_lock() → blocks other requests                                 │
│  3. Execute: callback(*args, **kwargs)                                      │
│  4. Publish: model.completed / model.failed                                 │
│  5. Release lock → next request                                            │
│                                                                              │
│  ThreadPoolExecutor: max_workers=4                                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. OpenVINO Architecture

**File:** `eva_ai/core/openvino_generator.py`

### OpenVINOGeneratorRegistry (Singleton)

```python
class OpenVINOGeneratorRegistry:
    # Key = (model_path, device)
    # Value = OpenVINOGenerator with loaded LLMPipeline
    
    def get_or_create(model_path, device, creator_fn) → generator
        # If exists: return cached, ref_count++
        # If new: create via creator_fn, cache, ref_count=1
    
    def release(model_path, device) → ref_count--
```

### OpenVINOGenerator (Lazy Load)

```python
class OpenVINOGenerator:
    # __init__(): Sets attributes, does NOT load model
    # _load_model(): Loads on FIRST generate() call
    
    def generate_streaming(prompt, max_tokens, temperature, chunk_size=25):
        # Uses ov_genai.LLMPipeline
        # Yields chunks via threading queue
    
    # Devices:
    # ├── CPU: Logic/Context → LATENCY mode, NUM_STREAMS=AUTO
    # └── GPU: Coder/Self-dialog → THROUGHPUT mode
    
    # SchedulerConfig:
    # ├── cache_size: KV-cache size (GB)
    # ├── max_num_seqs: Parallel slots
    # ├── max_num_batched_tokens: Batch size
    # └── enable_prefix_caching: Reuse KV for common prefixes
```

---

## 10. Component Initialization Order

```
1. CoreBrain.__init__()
   │
   ├── event_bus = EventBus()
   ├── deferred_system = DeferredCommandSystem(max_workers=6)
   │       └── set_event_bus(event_bus)
   │
   ├── _init_managers()
   │       ├── resource_manager
   │       ├── config_manager
   │       ├── memory_manager
   │       └── component_initializer
   │
   ├── _init_unified_generator()
   │       ├── OpenVINOGenerator for CPU (NOT loaded - lazy)
   │       ├── OpenVINOGenerator for GPU (NOT loaded - lazy)
   │       └── ModelAccessManager (started)
   │
   └── _init_background()
           ├── concept_miner = ConceptMiner()
           ├── contradiction_miner = ContradictionMiner()
           └── self_dialog_learning = SelfDialogLearning()

2. CoreBrain.initialize()
   ├── _subscribe_to_system_events()
   ├── component_initializer.initialize_components()
   ├── _initialize_memory_manager()
   ├── _connect_components()
   └── _start_post_init_services()
```

---

## 11. Web GUI Streaming Architecture

**File:** `eva_ai/gui/web_gui/server_routes_chat.py`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  POST /api/chat/stream                                                      │
│                                                                              │
│  1. Parse JSON: message, session_id, user_id, mode                          │
│  2. Get pipeline: brain.two_model_pipeline                                 │
│  3. Optional web search enhancement                                         │
│  4. SSE streaming:                                                         │
│     yield "data: {type: 'start', timestamp}\n\n"                         │
│     for chunk in pipeline.generate_streaming(prompt, chunk_size=25):        │
│         yield f"data: {type: 'chunk', text, is_final, ...}\n\n"          │
│     yield "data: {type: 'done', full_text}\n\n"                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### JavaScript Client (app.js)

```javascript
// XHR for POST streaming (EventSource only supports GET)
const xhr = new XMLHttpRequest();
xhr.open('POST', '/api/chat/stream', true);

xhr.onprogress = function() {
    // Parse SSE lines: "data: {...}"
    // data.type === 'chunk' → fullText += data.text
    // data.type === 'complete' → fullText = data.text (REPLACES, not +=)
    // data.type === 'done' → finalize
};
```

**SSE Event Types:** start, chunk, complete, done, error

---

## 12. Key Data Classes

### Concept (ConceptExtractor)

```python
@dataclass
class Concept:
    name: str
    description: str
    domain: str  # science, technology, philosophy, general
    source: str  # extraction, dialog, web
    confidence: float
    related_terms: List[str]
    facts: List[Dict]  # {relation_type, value, confidence}
```

### PhantomCandidate (ConceptMiner)

```python
@dataclass
class PhantomCandidate:
    id: str
    cluster_id: str
    centroid: List[float]
    variance: float
    semantic_gap: float
    title: str = ""
    definition: str = ""
    status: str = "provisional"  # provisional, confirmed, stable, archived
    confidence: float = 0.0
```

### GeneratedContradiction (ContradictionGenerator)

```python
@dataclass
class GeneratedContradiction:
    concept: str
    viewpoint_a: str
    viewpoint_b: str
    divergence_level: float
    reasoning_a: str
    reasoning_b: str
    resolution: Optional[str] = None
```

### ContradictionCandidate (ContradictionMiner)

```python
@dataclass
class ContradictionCandidate:
    id: str
    cluster_id: str
    node_ids: List[str]
    avg_similarity: float
    max_contradiction: float
    priority: float
    title: str = ""
    description: str = ""
    resolution_question: str = ""
    status: str = "active"
```

### SelfDialog

```python
@dataclass
class SelfDialog:
    id: str
    topic: str
    turns: List[DialogTurn]
    start_time: float
    end_time: Optional[float] = None
    outcome: Optional[str] = None
    learning_type: Optional[LearningType] = None

@dataclass
class DialogTurn:
    role: DialogRole  # ASSISTANT, CRITIC, LEARNER, TEACHER
    content: str
    timestamp: float
    quality_score: float = 0.0
```

---

## 13. API Endpoints Summary

### Chat & Streaming

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat` | POST | Non-streaming chat |
| `/api/chat/stream` | POST | SSE streaming |
| `/api/v1/chat` | POST | v1 API with threading |

### Sessions

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/sessions` | GET/POST/DELETE | Session management |

### System

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/status` | GET | System status |
| `/api/health` | GET | Health check |
| `/api/metrics` | GET | System metrics |

### Knowledge Graph

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/knowledge-graph` | GET | KG data |
| `/api/nodes` | GET | Graph nodes |
| `/api/edges` | GET | Graph edges |
| `/api/concepts` | GET | Concept nodes |
| `/api/contradictions` | GET | Contradiction nodes |

### Analytics

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/analytics` | GET | Dashboard data |
| `/api/learning` | GET | Learning stats |

---

## 14. File Structure

```
eva_ai/
├── core/
│   ├── core_brain.py              # Central coordinator
│   ├── unified_generator.py        # Unified generation
│   ├── openvino_generator.py      # OpenVINO + Registry + Lazy Load
│   ├── model_access_manager.py     # Model access coordination
│   ├── deferred_command_system.py  # Command queue system
│   ├── event_bus.py               # Event pub/sub
│   ├── pipeline_adapter.py         # Pipeline wrapper
│   ├── brain_query.py             # Query processing
│   ├── brain_components.py         # Component initialization
│   └── init_factories.py          # Factory functions
│
├── knowledge/
│   ├── concept_extractor.py        # Fast concept extraction
│   ├── concept_miner.py           # Deep concept mining
│   ├── kg_adapter.py              # KG → FGv2 adapter
│   ├── graph_curator.py           # Graph optimization
│   └── wikipedia_kb.py            # Wikipedia KB
│
├── contradiction/
│   ├── contradiction_generator.py  # Template-based generation
│   ├── contradiction_miner.py     # Analytical detection
│   └── contradiction_manager.py    # High-level coordination
│
├── learning/
│   ├── dialog_core.py             # SelfDialogLearning (MAIN)
│   ├── dialog_concepts.py         # DialogConceptsMixin
│   ├── dialog_types.py             # DialogRole, SelfDialog
│   ├── dialog_topics.py           # DialogTopicsMixin
│   ├── dialog_generation.py       # DialogGenerationMixin
│   └── dialog_learning.py         # DialogLearningMixin
│
└── gui/web_gui/
    ├── server_main.py             # Flask app
    ├── server_routes_chat.py     # Chat streaming endpoints
    ├── server_routes_core.py     # System endpoints
    ├── server_routes_graph.py    # KG endpoints
    ├── bridge.py                 # GUI-CORE bridge
    └── static/js/app.js          # Frontend JavaScript
```

---

## 15. Configuration (brain_config.json)

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
      "base_threshold": 0.30
    }
  },
  "contradiction_system": {
    "enabled": true,
    "generator": { "enabled": true },
    "miner": {
      "enabled": true,
      "check_interval_seconds": 3600,
      "sim_threshold": 0.75,
      "contra_threshold": 0.65
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

*Last Updated: April 14, 2026*
