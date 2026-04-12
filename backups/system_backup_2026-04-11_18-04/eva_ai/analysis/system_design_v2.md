# CogniFlex System Design - Fractal Graph V2 Integration

**Document Version:** 2.0  
**Date:** April 6, 2026  
**Purpose:** Updated system architecture with Fractal Graph V2

---

## 1. System Overview

### 1.1 Core Philosophy
CogniFlex is a self-learning cognitive system that:
- Uses Fractal Graph V2 as primary knowledge storage
- Integrates with legacy KnowledgeGraph for backward compatibility
- Implements continuous self-improvement through GraphCurator
- Provides real-time metrics and monitoring via Web UI

### 1.2 Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        CoreBrain                                │
│  ┌──────────────┬──────────────┬──────────────┬─────────────┐ │
│  │ fractal_     │ knowledge_    │ self_dialog  │ graph_      │ │
│  │ graph_v2     │ graph         │ _learning    │ curator     │ │
│  └──────────────┴──────────────┴──────────────┴─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Event Bus + Deferred Commands                │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Web UI (Monitor, Learning, Knowledge)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Fractal Graph V2

### 2.1 Storage Layer
- **Location:** `eva/memory/fractal_graph_v2/`
- **Database:** SQLite (fractal_graph.db)
- **Embedding:** intfloat/multilingual-e5-base (768-dim)

### 2.2 Node Types
| Type | Description | Level |
|------|-------------|-------|
| concept | Core concept | 0-1 |
| fact | Verified fact | 1-2 |
| detail | Additional detail | 2-3 |
| context | Contextual info | 2-3 |
| attribute | Attribute/property | 3 |
| query | User query | 2 |
| response | System response | 2 |
| learned_knowledge | Self-learned | 1 |

### 2.3 Key Features
- Semantic clustering (agglomerative + DBSCAN)
- Contradiction detection via self-dialog
- Incremental vectorization with caching
- N-gram keyword indexing

### 2.4 API Methods
```python
# Core operations
add_node(content, node_type, level, confidence)
add_knowledge(subject, relation, object)
semantic_search(query, top_k, min_level)
keyword_search(query, top_k)
check_contradiction(content, group_id)
self_dialogue(new_knowledge)

# Integration
save_experience(query, response, model_used, quality_score)
get_context_for_query(query, max_length)
retrieve_knowledge(query, top_k)
```

---

## 3. GraphCurator

### 3.1 Purpose
Background maintenance and optimization of the knowledge graph.

### 3.2 Features
- **Adaptive intervals:** 60-600 seconds based on activity
- **Event bus integration:** Subscribes to system events
- **Deferred commands:** Uses priority queue for background tasks
- **GGUF extraction:** Extracts knowledge from GGUF models

### 3.3 Operations
1. Health check (orphan nodes, duplicate detection)
2. Semantic link creation
3. Orphan cleanup
4. Duplicate merging
5. Re-clustering when threshold reached

### 3.4 Metrics
```python
@dataclass
class CuratorMetrics:
    cycles_completed: int
    nodes_curated: int
    links_created/removed: int
    groups_created/merged: int
    total_nodes/edges/groups: int
    orphan_nodes: int
    avg_cycle_time: float
    knowledge_extracted: int
```

---

## 4. Integration Points

### 4.1 Brain Query (`brain_query.py`)
- Primary context source: FractalGraphV2
- Fallback: KnowledgeGraph
- Flow: Query → FG2.get_context_for_query() → LLM

### 4.2 Self-Dialog Learning (`dialog_learning.py`)
- Adds learned knowledge to FG2
- Checks contradictions before adding
- Uses self_dialogue() for verification
- Saves experiences via save_experience()

### 4.3 Web UI (`server_routes.py`)
- `/api/learning` - SDL stats and opportunities
- `/api/knowledge` - Knowledge stats (FG2 + KG)
- `/api/analytics` - System metrics including FG2
- `/api/events/stream` - Real-time events including curator

---

## 5. Data Flow

### 5.1 Query Processing
```
User Query
    ↓
brain_query.process()
    ↓
fractal_graph_v2.get_context_for_query()
    ↓
Context + Query → LLM
    ↓
Response + save_experience()
    ↓
SelfDialogLearning (if low rating)
```

### 5.2 Self-Learning
```
GraphCurator (periodic)
    ↓
Check knowledge gaps
    ↓
SelfDialogLearning.create_dialog()
    ↓
LLM generates self-questions
    ↓
dialog_learning._learn_expansion/refinement
    ↓
fractal_graph_v2.add_node() + check_contradiction()
```

### 5.3 Monitoring
```
EventBus (curator.*, self_dialog.*, generation.*)
    ↓
SSE /api/events/stream
    ↓
Web UI Monitor (real-time display)
```

---

## 6. Configuration

### 6.1 brain_config.json
```json
{
  "fractal_graph_v2": {
    "enabled": true,
    "storage_dir": "eva/memory/fractal_graph_v2/data",
    "embedding_model": "intfloat/multilingual-e5-base",
    "embedding_device": "cuda"
  },
  "graph_curator": {
    "enabled": true,
    "adaptive_interval": true,
    "min_interval": 60,
    "max_interval": 600,
    "extract_from_gguf": true
  }
}
```

---

## 7. Files Modified

### Core
- `eva/core/brain_components.py` - Graph API methods
- `eva/core/brain_init.py` - FG2 + Curator initialization
- `eva/core/brain_query.py` - FG2 context integration
- `eva/core/init_connections.py` - Dependency registration

### Learning
- `eva/learning/dialog_core.py` - Curator events, save_experience
- `eva/learning/dialog_learning.py` - FG2 integration

### Knowledge
- `eva/knowledge/graph_curator.py` - Full rewrite with metrics

### GUI
- `eva/gui/web_gui/server_routes.py` - Learning, Knowledge APIs
- `eva/gui/web_gui/static/js/app.js` - Curator handling
- `eva/gui/web_gui/static/css/style.css` - Curator styles
- `eva/gui/web_gui/templates/index.html` - Curator tab

### New Files
- `eva/memory/fractal_graph_v2/` - Complete module
- `migrate_old_graph.py` - Migration script
- `benchmark_generation.py` - Performance testing

---

## 8. Backward Compatibility

- KnowledgeGraph remains active for legacy components
- Fallback chain: FG2 → KG → Demo data
- All new features work without breaking existing functionality