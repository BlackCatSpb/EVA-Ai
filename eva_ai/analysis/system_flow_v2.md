# CogniFlex System Flow - Fractal Graph V2

**Document Version:** 2.0  
**Date:** April 6, 2026

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
│  │   Fallback: knowledge_graph.get_relevant_nodes()                          │
│  └── full_prompt = query + knowledge_context                                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LLM Generation (Qwen/LlamaCpp)                                             │
│  └── Response generation                                                      │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Response Processing                                                         │
│  ├── save_experience(query, response, model_used, quality)                  │
│  │       │                                                                    │
│  │       ▼                                                                    │
│  │   fractal_graph_v2.save_experience()                                       │
│  │       → add_node(query, type=query)                                       │
│  │       → add_node(response, type=response)                                 │
│  │       → add_edge(query → response, relation=generated_by)                 │
│  └── Return to user                                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Self-Learning Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         GRAPH CURATOR (Background)                          │
│  Runs every 60-600 seconds (adaptive)                                       │
│  ├── check_graph_health()                                                    │
│  ├── create_semantic_links() - SEMANTIC_ASSOCIATIONS dict                   │
│  ├── cleanup_orphans() - nodes with no edges, confidence < 0.3              │
│  ├── cleanup_duplicates() - similarity > 0.95                               │
│  ├── recluster_if_needed() - threshold < 0.4                                │
│  └── extract_from_gguf() - extract knowledge from GGUF models                │
│                                                                               │
│  Events published: curator.started, curator.completed,                      │
│                   curator.graph_optimized, curator.knowledge_extracted       │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SELF DIALOG LEARNING                                      │
│  ┌─ Periodic (auto_learning_interval)                                        │
│  │   _check_and_execute_learning_opportunities()                             │
│  │       → _get_learning_opportunities()                                     │
│  │       → _execute_learning_opportunity()                                   │
│  │              │                                                            │
│  │              ▼                                                           │
│  │       _perform_learning()                                                  │
│  │              │                                                            │
│  │              ▼                                                           │
│  │       ┌─ expansion: add_node() + check_contradiction()                  │
│  │       ├─ refinement: self_dialogue() verification                        │
│  │       ├─ updating: update existing nodes                                 │
│  │       └─ integration: merge new knowledge                                 │
│  │                                                                          │
│  └─ On user feedback (rating < 3)                                            │
│      analyze_interaction() → create_dialog()                                 │
│      → save_experience() with quality score                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       FRACTAL GRAPH V2                                       │
│  add_node()                                                                  │
│      → auto_vectorize (embeddings)                                         │
│      → auto_cluster (find nearest group, threshold 0.6)                    │
│      → save to SQLite                                                        │
│                                                                               │
│  check_contradiction()                                                      │
│      → semantic_search() → compare with group centroid                      │
│      → return {is_contradiction, distance}                                  │
│                                                                               │
│  self_dialogue()                                                            │
│      → check_contradiction()                                               │
│      → _search_related_facts()                                              │
│      → confirm/reject based on confirming facts count                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Event System Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         EVENT BUS                                            │
│                                                                               │
│  SUBSCRIPTIONS:                                                              │
│  ┌─ QueryMixin: query_received → triggers self-dialog check               │
│  ┌─ SelfDialogLearning: curator.events → learning triggers                 │
│  ┌─ GraphCurator: system.health_check → publish metrics                    │
│  └─ WebUI: pipeline.*, generation.*, curator.*, self_dialog.*             │
│                                                                               │
│  EVENT TYPES:                                                                │
│  ├── pipeline.start, pipeline.model_a_start, pipeline.complete             │
│  ├── generation.progress, generation.completed                              │
│  ├── curator.started, curator.completed, curator.error                     │
│  ├── curator.graph_optimized, curator.knowledge_extracted                  │
│  ├── self_dialog.started, self_dialog.completed                             │
│  └── memory_pressure, system_health_check                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      DEFERRED COMMANDS                                       │
│  Priority queue for background tasks                                         │
│  ├── CRITICAL: Immediate operations                                         │
│  ├── HIGH: Urgent learning                                                  │
│  ├── NORMAL: Regular curation                                               │
│  └── LOW: Background optimization                                           │
│                                                                               │
│  Used by: GraphCurator for scheduled curation                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Web UI Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         BROWSER                                             │
│                                                                               │
│  VIEW: learning                                                              │
│  └── loadLearning() → GET /api/learning                                    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─ SelfDialogLearning.stats (total_dialogs, successful_learning)         │
│  ├─ SelfDialogLearning._get_learning_opportunities()                       │
│  └─ SelfDialogLearning.dialog_history (recent dialogs)                     │
│                                                                               │
│  VIEW: knowledge                                                             │
│  └── loadKnowledge() → GET /api/knowledge                                   │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─ fractal_graph_v2.get_stats() (if available)                          │
│  ├─ knowledge_graph.get_stats() (fallback)                                  │
│  └─ Demo data (if nothing else)                                             │
│                                                                               │
│  VIEW: analytics                                                             │
│  └── loadAnalytics() → GET /api/analytics                                  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─ fractal_graph_v2 metrics (nodes, edges, groups)                       │
│  ├─ graph_curator metrics (cycles, state, next_run)                         │
│  ├─ self_dialog_learning stats                                               │
│  └─ system resources (CPU, memory, cache)                                   │
│                                                                               │
│  VIEW: monitor                                                                │
│  └── initMonitor() → EventSource /api/events/stream                        │
│       │                                                                      │
│       ▼                                                                      │
│  Pipeline events, generation events,                                        │
│  Curator events (new tab), self-dialog events                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Initialization Order

```
1. CoreBrain.__init__()
   ├── load_brain_config()
   ├── Initialize components via ComponentInitializer
   │   ├── event_bus
   │   ├── resource_manager
   │   ├── knowledge_graph (legacy)
   │   ├── text_processor
   │   ├── ml_unit
   │   ├── model_manager
   │   └── self_dialog_learning
   │
2. CoreBrain.initialize()
   ├── Subscribe to system events
   ├── Initialize memory_manager
   ├── Connect components
   │
3. _start_post_init_services()
   ├── Start self_dialog_learning (if enabled)
   ├── Initialize fractal_graph_v2 (NEW!)
   │   └── FractalMemoryGraph(storage + embeddings)
   ├── Start graph_curator
   │   └── GraphCurator(brain=brain, config=...)
   │       └── Subscribe to event_bus
   │       └── Connect to deferred_system
   └── Start gguf_training (if enabled)
```

---

## 6. Key Data Structures

### FractalNode
```python
@dataclass
class FractalNode:
    id: str
    content: str
    node_type: str  # concept, fact, detail, etc.
    level: int      # 0-3 (fractal level)
    confidence: float
    embedding: List[float]
    parent_group_id: str
    metadata: Dict
    created_at: float
    is_contradiction: bool
```

### SemanticGroup
```python
@dataclass
class SemanticGroup:
    id: str
    name: str
    level: int
    embedding: List[float]  # centroid
    member_ids: List[str]
    member_count: int
    avg_confidence: float
```

### CuratorMetrics
```python
@dataclass
class CuratorMetrics:
    cycles_completed: int
    nodes_curated: int
    links_created: int
    groups_created: int
    total_nodes: int
    orphan_nodes: int
    avg_cycle_time: float
    knowledge_extracted: int
    state: str
```

---

## 7. Configuration Files

### brain_config.json (new sections)
```json
{
  "fractal_graph_v2": {
    "enabled": true,
    "storage_dir": "eva/memory/fractal_graph_v2/data",
    "embedding_model": "intfloat/multilingual-e5-base",
    "embedding_device": "cuda",
    "embedding_dim": 768
  },
  "graph_curator": {
    "enabled": true,
    "adaptive_interval": true,
    "min_interval": 60,
    "max_interval": 600,
    "check_graph_health": true,
    "cleanup_orphans": true,
    "cleanup_duplicates": true,
    "recluster_threshold": 0.4,
    "extract_from_gguf": true,
    "gguf_models_dir": "eva/mlearning/eva_models"
  }
}
```

---

## 8. Fallback Chain

| Component | Primary | Fallback 1 | Fallback 2 |
|-----------|---------|------------|-----------|
| Knowledge Context | FractalGraphV2 | KnowledgeGraph | Empty string |
| Learning Stats | SelfDialogLearning | self_analyzer | Demo data (0) |
| Knowledge Stats | FractalGraphV2 | KnowledgeGraph | Demo nodes |
| Dialog History | SelfDialogLearning | None | Empty list |

---

## 9. Testing Commands

```bash
# Test imports
python -c "from eva.memory.fractal_graph_v2 import FractalMemoryGraph"
python -c "from eva.knowledge.graph_curator import GraphCurator"

# Test API endpoints (when running)
curl http://localhost:5555/api/learning
curl http://localhost:5555/api/knowledge
curl http://localhost:5555/api/analytics

# Check logs
tail -f eva.log | grep -E "(fractal_graph|curator|self_dialog)"
```