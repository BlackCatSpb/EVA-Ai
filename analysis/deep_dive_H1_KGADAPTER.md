# H1 Deep Dive: KGAdapter Integration

## Problem Summary

**Issue**: KGAdapter (KnowledgeGraphAdapter) is not created in init_factories.py, although it should provide access to FGv2 through a unified compatibility interface.

**Priority**: H1 (HIGH)
**Status**: Critical architecture vulnerability

---

## 1. Where KGAdapter is Defined

### 1.1 Definition File
- **Path**: `C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\knowledge\kg_adapter.py`
- **Class**: `KnowledgeGraphAdapter`
- **Purpose**: Adapter for compatibility with code using KnowledgeGraph API by redirecting calls to FractalGraph v2

### 1.2 KGAdapter Methods

| Method | Description |
|--------|-------------|
| `nodes` (property) | Returns nodes as dictionary |
| `edges` (property) | Returns edges as dictionary |
| `get_recent_entities(limit)` | Get recent entities |
| `add_entity(name, type, props)` | Add entity |
| `add_concept(concept, content, domain, source)` | Add concept |
| `add_relation(source, target, relation_type)` | Add relation |
| `find_related(concept, limit)` | Find related concepts |
| `get_related_concepts(concept)` | Get list of related concepts |
| `find_path_between_concepts(c1, c2, max_depth)` | Find shortest path via BFS |
| `get_entity_facts(entity)` | Get facts about entity |
| `search_nodes(query, limit)` | Search nodes |
| `get_stats()` | Get statistics |

### 1.3 Constructor

```python
def __init__(self, fractal_graph):
    self._fg = fractal_graph
    self.stats = {
        total_nodes: 0,
        total_edges: 0,
        migrated_from_kg: True
    }
```

---

## 2. Where Components are Created

### 2.1 Current Version (init_factories.py)

**File**: `C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\init_factories.py`

**Function**: `create_knowledge_components(initializer)` (line 493)

This function creates:
- ConceptExtractor (fast level of concept extraction)
- ContradictionGenerator (contradiction generation via templates)
- ContradictionMiner (contradiction detection in graph)
- ConceptMiner (deep cluster analysis)
- Wikipedia Knowledge Base
- **KGAdapter IS NOT CREATED** (PROBLEM)

### 2.2 Previous Version (backup)

**File**: `C:\Users\black\OneDrive\Desktop\EVA-Ai\backups\system_backup_2026-04-11_18-04\eva_ai\core\init_factories.py`

**Function**: `create_knowledge_graph(initializer)` (line 484)

This function created:
- KnowledgeGraphAdapter
- ConceptExtractor
- ContradictionGenerator
- ContradictionMiner
- ConceptMiner
- Wikipedia Knowledge Base

### 2.3 Error Code in Current Version

```python
init_factories.py, lines 493-609
def create_knowledge_components(initializer):
    Creates components based on FractalGraph v2.
    WITHOUT KG adapter - using FGv2 directly.
    
    ... creates components ...
    
    initializer.logger.info("[OK] KnowledgeGraph adapter (FGv2) created")
    return kg_adapter  ERROR: kg_adapter is never defined!
```

**Bug**: Line 606 tries to return `kg_adapter` which was never created. This will cause `NameError` when function is called.

---

## 3. Why KGAdapter is Not Created

### 3.1 Refactoring Without Preserving Functionality

During refactoring (renaming `create_knowledge_graph()` to `create_knowledge_components()`):
- Function was renamed
- Comment changed to "WITHOUT KG adapter"
- KGAdapter creation was removed
- **BUT**: Return `kg_adapter` remained - this is a bug

### 3.2 Call Sequence

In `register_all_factories()` (line 966):
```python
initializer.component_factories = {
    No: knowledge_graph: lambda: create_knowledge_graph(initializer)
    No: kg_adapter: lambda: create_kg_adapter(initializer)
}
```

Factory `knowledge_graph` is no longer registered.

---

## 4. Impact on System

### 4.1 Number of knowledge_graph References

**Found 485 references** to `brain.knowledge_graph` in code:

| Module | References |
|--------|------------|
| learning/scheduler_tasks.py | ~50 |
| learning/dialog_learning.py | ~20 |
| learning/integration_manager.py | ~15 |
| reasoning/self_reasoning_engine.py | ~10 |
| memory/ltm_retrieval.py | ~5 |
| Other modules | ~385 |

### 4.2 What Breaks

1. **SelfReasoningEngine** (lines 598, 600, 896)
   ```python
   if self.brain and hasattr(self.brain, knowledge_graph) and self.brain.knowledge_graph:
       kg = self.brain.knowledge_graph
   ```
   - hasattr check returns False (attribute does not exist)
   - Code will not crash, but will not work

2. **SchedulerTasks** (lines 253-702)
   - Multiple calls to add_node(), add_edge(), search_nodes()
   - All these operations will fail

3. **DialogLearning** (lines 327-363)
   - Concept updates after self-dialog
   - Works only with hasattr checks

4. **HealthMonitor** (line 68)
   ```python
   kg_health = self.brain.knowledge_graph.get_system_health()
   ```
   - Will cause AttributeError

---

## 5. Proposed Solution

### 5.1 Add KGAdapter Creation

In `create_knowledge_components()` function after getting FGv2:

```python
def create_knowledge_components(initializer):
    try:
        Get FGv2
        fg = getattr(initializer.core_brain, fractal_graph_v2, None)
        if fg is None:
            components = getattr(initializer.core_brain, components, {})
            fg = components.get(fractal_graph_v2)
        
        if fg is None:
            initializer.logger.warning("[WARN] FGv2 not found, knowledge components not created")
            return None
        
        ADD: KGAdapter Creation
        from eva_ai.knowledge.kg_adapter import KnowledgeGraphAdapter
        kg_adapter = KnowledgeGraphAdapter(fg)
        initializer.core_brain.knowledge_graph = kg_adapter
        initializer.core_brain.components[knowledge_graph] = kg_adapter
        initializer.logger.info("[OK] KnowledgeGraphAdapter created")
        
        ... rest of code ...
```

### 5.2 Register Factory

In `register_all_factories()`:

```python
def register_all_factories(initializer):
    initializer.component_factories = {
        knowledge_graph: lambda: create_knowledge_graph(initializer),
    }
```

### 5.3 Fix Return

Remove line `return kg_adapter` or fix to:
```python
return initializer.core_brain.knowledge_graph
```

---

## 6. Summary

| Aspect | Description |
|--------|-------------|
| **Problem** | KGAdapter removed during refactoring, but 485 references expect it |
| **Root Cause** | Function create_knowledge_graph() renamed to create_knowledge_components() without preserving KGAdapter creation |
| **Impact** | Multiple AttributeError in runtime, hasattr checks prevent crashes |
| **Solution** | Restore KGAdapter creation in create_knowledge_components() |
| **Complexity** | Low (5-10 lines of code) |
| **Priority** | HIGH (affects system stability) |

---

## 7. Files to Modify

1. **`C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\init_factories.py`**
   - Line ~493: Add KGAdapter creation
   - Line ~606: Fix return
   - Line ~968: Register factory

---

*Date: 2026-04-27*
*Author: AI Architect Agent*
