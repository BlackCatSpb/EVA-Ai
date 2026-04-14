# SnapshotManager Audit Report

**Generated:** 2026-04-14 10:19:28  
**Auditor:** EVA AI System  
**Component:** SnapshotManager (fractal_graph_v2)

---

## 1. Component Overview

### 1.1 Purpose
SnapshotManager provides immutable memory snapshots for generation consistency in EVA AI system. It ensures that during response generation, the context remains unchanged even if the underlying graph is modified.

### 1.2 Location
- **Main Implementation:** C:\Users\black\OneDrive\Desktop\CogniFlex\eva_ai\memory\fractal_graph_v2\snapshot_manager.py
- **Integration:** C:\Users\black\OneDrive\Desktop\CogniFlex\eva_ai\core\hybrid_pipeline_adapter.py
- **Exports:** C:\Users\black\OneDrive\Desktop\CogniFlex\eva_ai\memory\fractal_graph_v2\__init__.py

### 1.3 Architecture Summary
`
SnapshotManager (snapshot_manager.py)
├── MemorySnapshot (dataclass) - immutable snapshot data
│   ├── snapshot_id: str
│   ├── created_at: float
│   ├── node_contents: Dict[str, str]
│   ├── node_confidences: Dict[str, float]
│   ├── node_metadata: Dict[str, Dict]
│   ├── dialogue_context: str
│   ├── session_id: str
│   └── ttl_seconds: float = 300.0
│
└── SnapshotManager (main class)
    ├── fractal_graph: reference to FractalGraph
    ├── ttl_seconds: TTL for snapshots
    ├── max_active_snapshots: max snapshots limit (default: 20)
    ├── _active_snapshots: Dict[str, MemorySnapshot]
    ├── _session_snapshots: Dict[str, str]
    └── _cleanup_thread: background cleanup thread
`

---

## 2. Snapshot Creation Analysis

### 2.1 Creation Flow
**Method:** create_snapshot(session_id, node_ids, dialogue_context)

**Process:**
1. Acquire lock (	hreading.RLock)
2. Generate snapshot_id via SHA256 hash of session_id:timestamp
3. For each node_id in node_ids:
   - Fetch node from fractal_graph
   - Extract content, confidence, metadata
   - Store in local dictionaries
4. Create MemorySnapshot dataclass
5. Store in _active_snapshots[snapshot_id]
6. Map session to snapshot: _session_snapshots[session_id] = snapshot_id
7. Call _evict_if_needed() to enforce max limit
8. Release lock

### 2.2 Code Analysis (lines 119-168)
`python
def create_snapshot(self, session_id, node_ids, dialogue_context="") -> MemorySnapshot:
    with self._lock:
        snapshot_id = self._generate_snapshot_id(session_id)
        
        node_contents = {}
        node_confidences = {}
        node_metadata = {}
        
        if self.fractal_graph:
            for node_id in node_ids:
                node = self._get_node(node_id)
                if node:
                    node_contents[node_id] = getattr(node, 'content', '')
                    node_confidences[node_id] = getattr(node, 'confidence', 0.5)
                    node_metadata[node_id] = getattr(node, 'metadata', {})
        
        snapshot = MemorySnapshot(...)
        self._active_snapshots[snapshot_id] = snapshot
        self._session_snapshots[session_id] = snapshot_id
        self._evict_if_needed()
        
        return snapshot
`

### 2.3 Node Retrieval Method (lines 242-255)
`python
def _get_node(self, node_id: str):
    if not self.fractal_graph:
        return None
    
    try:
        if hasattr(self.fractal_graph, 'get_node'):
            return self.fractal_graph.get_node(node_id)
        elif hasattr(self.fractal_graph, 'storage') and hasattr(self.fractal_graph.storage, 'nodes'):
            return self.fractal_graph.storage.nodes.get(node_id)
    except Exception as e:
        logger.warning(f"Failed to get node {node_id}: {e}")
    
    return None
`

### 2.4 Issues Found in Creation
| Issue | Severity | Description |
|-------|----------|-------------|
| **No deep copy** | HIGH | 
ode_contents, 
ode_confidences, 
ode_metadata store references. If the original node objects are modified later, the snapshot may become inconsistent. Should use deepcopy() for all nested structures. |
| **Silent failure** | MEDIUM | If _get_node() fails for a node, it's silently skipped. No error logged, no warning to caller. |
| **No validation** | LOW | 
ode_ids is not validated - can be empty list, None, or contain non-existent IDs. |

---

## 3. Snapshot Recovery Analysis

### 3.1 Recovery Methods

#### Method 1: get_snapshot(snapshot_id) (lines 170-186)
`python
def get_snapshot(self, snapshot_id: str) -> Optional[MemorySnapshot]:
    with self._lock:
        snapshot = self._active_snapshots.get(snapshot_id)
        
        if snapshot and snapshot.is_expired():
            del self._active_snapshots[snapshot_id]
            if snapshot.session_id in self._session_snapshots:
                del self._session_snapshots[snapshot.session_id]
            return None
        
        return snapshot
`

#### Method 2: get_session_snapshot(session_id) (lines 188-194)
`python
def get_session_snapshot(self, session_id: str) -> Optional[MemorySnapshot]:
    with self._lock:
        snapshot_id = self._session_snapshots.get(session_id)
        if snapshot_id:
            return self.get_snapshot(snapshot_id)
        return None
`

### 3.2 Recovery Flow
1. Lookup snapshot by ID or session_id
2. Check if expired via is_expired() method
3. If expired - delete and return None
4. If valid - return snapshot

### 3.3 Issues Found in Recovery
| Issue | Severity | Description |
|-------|----------|-------------|
| **Double lock** | MEDIUM | get_session_snapshot() calls get_snapshot() which also acquires lock. get_snapshot() acquires lock again via with self._lock. But RLock allows re-entry, so no deadlock. Still inefficient. |
| **Race window** | LOW | Between checking is_expired() and returning, another thread could delete the snapshot. Minor issue due to RLock. |

### 3.4 MemorySnapshot Data Access
The MemorySnapshot dataclass provides:
- get_content(node_id) - returns node content
- get_confidence(node_id) - returns confidence
- get_metadata(node_id) - returns metadata dict
- get_all_nodes() - list all node IDs in snapshot

---

## 4. TTL and Cleanup Analysis

### 4.1 TTL Configuration
| Parameter | Default | Configurable | Location |
|-----------|---------|--------------|----------|
| 	tl_seconds | 300.0 (5 min) | Yes | Constructor, create_snapshot() |
| max_active_snapshots | 20 | Yes | Constructor |

### 4.2 Expiration Check
`python
def is_expired(self) -> bool:
    return (time.time() - self.created_at) > self.ttl_seconds
`

### 4.3 Cleanup Loop (lines 110-117)
`python
def _cleanup_loop(self):
    while self._running:
        try:
            self.cleanup_expired()
        except Exception as e:
            logger.error(f"Snapshot cleanup error: {e}")
        time.sleep(30)
`

**Issues:**
| Issue | Severity | Description |
|-------|----------|-------------|
| **Hardcoded interval** | MEDIUM | Sleep interval is hardcoded to 30 seconds. Not configurable. |
| **No startup cleanup** | LOW | Cleanup only runs periodically. On startup, expired snapshots from previous run are not cleaned. |

### 4.4 Manual Cleanup (lines 196-218)
`python
def cleanup_expired(self) -> int:
    with self._lock:
        expired = [
            sid for sid, snap in self._active_snapshots.items()
            if snap.is_expired()
        ]
        
        for sid in expired:
            snap = self._active_snapshots[sid]
            if snap.session_id in self._session_snapshots:
                del self._session_snapshots[snap.session_id]
            del self._active_snapshots[sid]
        
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired snapshots")
        
        return len(expired)
`

### 4.5 Eviction on Limit (lines 220-235)
`python
def _evict_if_needed(self):
    if len(self._active_snapshots) > self.max_active_snapshots:
        sorted_snapshots = sorted(
            self._active_snapshots.items(),
            key=lambda x: x[1].created_at
        )
        
        to_remove = len(self._active_snapshots) - self.max_active_snapshots
        for i in range(to_remove):
            sid, snap = sorted_snapshots[i]
            if snap.session_id in self._session_snapshots:
                del self._session_snapshots[snap.session_id]
            del self._active_snapshots[sid]
        
        logger.debug(f"Evicted {to_remove} old snapshots")
`

### 4.6 Issues Found in TTL/Cleanup
| Issue | Severity | Description |
|-------|----------|-------------|
| **Eviction vs Expiration** | MEDIUM | Oldest snapshots are evicted first (LRU), not expired ones. This is correct for LRU but could evict non-expired snapshots when limit reached. |
| **No cleanup on access** | LOW | get_snapshot() checks expiration but only deletes when accessed. Background cleanup handles this. |
| **Memory not released immediately** | LOW | Expired snapshots remain in memory until next cleanup cycle (max 30s delay). |

---

## 5. Integration with FractalGraph

### 5.1 Integration Points

#### Point 1: HybridPipelineAdapter (hybrid_pipeline_adapter.py, lines 562-576)
`python
snapshot_mgr = create_snapshot_manager(
    fractal_graph=self.fractal_graph,
    ttl_seconds=300.0,
    max_active_snapshots=20
)

relevant_nodes = self._find_relevant_nodes(query, top_k=10)
node_ids = [n.get('id', n.get('node_id')) for n in relevant_nodes if n]
node_ids = [n for n in node_ids if n]

snapshot = snapshot_mgr.create_snapshot(
    session_id=session_id,
    node_ids=node_ids,
    dialogue_context=""
)
`

**Issue:** SnapshotManager is created fresh for each generation request! It doesn't persist across requests. This means:
- No shared state between generations
- Background cleanup thread starts/stops per request
- Memory snapshots are isolated per request

#### Point 2: VirtualTokenManager Integration
The snapshot is passed to VirtualTokenManager:
`python
virtual_token_mgr = create_virtual_token_manager(
    snapshot_or_contents=snapshot,  # SnapshotManager or dict
    llama_model=self.model_a or self.model_b,
    config=self._kwargs.get('virtual_tokens', {})
)
`

### 5.2 VirtualTokenHandler Usage (virtual_token_handler.py, lines 244-248)
`python
def _get_content(self, node_id: str) -> Optional[str]:
    if self._snapshot:
        return self._snapshot.get_content(node_id)
    return self._contents_dict.get(node_id)
`

### 5.3 Issues Found in Integration
| Issue | Severity | Description |
|-------|----------|-------------|
| **Per-request instantiation** | CRITICAL | SnapshotManager is created fresh in each call to generate_with_virtual_tokens(). The cleanup thread starts and stops for each request. This is inefficient and defeats the purpose of persistent snapshots. |
| **No singleton pattern** | CRITICAL | Should be created once and reused across requests, similar to how FractalMemoryGraph is used. |
| **Fragmented architecture** | HIGH | SnapshotManager in ractal_graph_v2/ is separate from SnapshotManager in graph_learning.py (export/import). These are two different classes with the same name. |

---

## 6. Alternative SnapshotManager: graph_learning.py

**Note:** There is ANOTHER SnapshotManager in va_ai/memory/graph_learning.py (lines 553-649).

### 6.1 Purpose
This SnapshotManager handles **export/import of knowledge snapshots** to JSON files for backup and transfer.

### 6.2 Methods
`python
class SnapshotManager:
    def __init__(self, fractal_memory, context_builder):
        self.snapshots_dir = os.path.join(fractal_memory.storage_dir, "snapshots")
    
    def export_snapshot(self, name: str = None) -> str:
        # Exports experiences and concepts to JSON
        
    def import_snapshot(self, snapshot_path: str) -> Dict:
        # Imports from JSON file
        
    def list_snapshots(self) -> List[Dict]:
        # Lists available snapshots
`

### 6.3 Issue: Name Collision
Both classes are named SnapshotManager but serve completely different purposes:
- ractal_graph_v2/snapshot_manager.py - runtime consistency snapshots
- graph_learning.py - persistent export/import for knowledge graphs

---

## 7. Thread Safety Analysis

### 7.1 Locking Strategy
`python
self._lock = threading.RLock()  # Reentrant lock
`

All public methods use with self._lock: to ensure thread safety.

### 7.2 Protected Operations
| Operation | Protected | Method |
|-----------|-----------|--------|
| Create snapshot | Yes | with self._lock: |
| Get snapshot | Yes | with self._lock: |
| Get session snapshot | Yes (via get_snapshot) | with self._lock: |
| Cleanup expired | Yes | with self._lock: |
| Evict if needed | Yes (called from create) | with self._lock: |
| Get stats | Yes | with self._lock: |
| Start/Stop | No | Thread management methods |

### 7.3 Thread Safety Issues
| Issue | Severity | Description |
|-------|----------|-------------|
| **start()/stop() not locked** | LOW | start() and stop() modify _running flag and thread without lock. Could cause race conditions if called simultaneously. |
| **Background thread exception** | MEDIUM | _cleanup_loop() catches all exceptions but only logs them. If cleanup repeatedly fails, snapshots accumulate indefinitely. |

---

## 8. Error Handling Analysis

### 8.1 Error Handling Quality
| Scenario | Handling | Issue |
|----------|----------|-------|
| Node not found | Silently skip | No warning logged |
| Fractal graph not set | Returns None | No error, just empty snapshot |
| Lock acquisition fails | RLock handles | N/A |
| Cleanup thread exception | Logs error only | Could accumulate memory |

### 8.2 Missing Error Handling
| Scenario | Expected Behavior | Current Behavior |
|----------|-------------------|------------------|
| Empty node_ids list | Warning or allow | Creates empty snapshot silently |
| Invalid session_id | Validation | Accepts any string |
| Snapshot ID not found | Return None | Works correctly |
| Max snapshots exceeded | LRU eviction | Works correctly |

---

## 9. Performance Analysis

### 9.1 Memory Usage
| Element | Size Estimation |
|---------|-----------------|
| Per-node storage | content (string) + confidence (float) + metadata (dict) |
| Snapshot overhead | ~200 bytes per snapshot (ID, timestamps, dicts) |
| Max memory (20 snapshots, 10 nodes each, 1KB content) | ~400KB |

### 9.2 Time Complexity
| Operation | Complexity |
|-----------|------------|
| Create snapshot | O(n) where n = number of node_ids |
| Get snapshot | O(1) dict lookup |
| Cleanup expired | O(n) where n = active snapshots |
| Eviction | O(n log n) due to sorting |

### 9.3 Performance Issues
| Issue | Severity | Description |
|-------|----------|-------------|
| **Sorting on every eviction** | LOW | Uses sorted() which is O(n log n). With 20 max snapshots, this is negligible. |
| **No snapshot reuse** | HIGH | Each generation request creates new snapshot. Old ones are discarded even if node_ids overlap. |
| **No snapshot caching** | MEDIUM | Could cache snapshots for identical node_ids sets. |

---

## 10. Testing Status

### 10.1 Implicit Testing
The system is used in production but no explicit unit tests found for:
- SnapshotManager
- MemorySnapshot
- Integration with VirtualTokenManager

### 10.2 Coverage Gaps
| Area | Coverage |
|------|----------|
| create_snapshot() | Implicit via usage |
| get_snapshot() | Implicit |
| get_session_snapshot() | Implicit |
| cleanup_expired() | Implicit |
| TTL expiration | Implicit |
| Concurrent access | Not tested |

---

## 11. Issues Summary

### Critical Issues
| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | Per-request instantiation of SnapshotManager | hybrid_pipeline_adapter.py:562 | Defeats purpose; no persistent state; thread thrashing |
| 2 | No deep copy of node data | snapshot_manager.py:147-149 | Potential data inconsistency if nodes modified |

### High Priority Issues
| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 3 | Name collision with graph_learning.SnapshotManager | Two files | Confusion, potential import errors |
| 4 | Silent node retrieval failures | snapshot_manager.py:144-149 | Missing data without warning |

### Medium Priority Issues
| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 5 | Hardcoded cleanup interval (30s) | snapshot_manager.py:117 | Not configurable |
| 6 | No startup cleanup of expired snapshots | _cleanup_loop | Stale data until first cleanup |
| 7 | Background cleanup swallows exceptions | snapshot_manager.py:115 | Silent failure accumulation |
| 8 | Double lock in get_session_snapshot | snapshot_manager.py:190-193 | Inefficient but not broken (RLock) |

### Low Priority Issues
| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 9 | No validation of input parameters | create_snapshot | Accepts invalid data |
| 10 | start()/stop() not thread-safe | snapshot_manager.py:93-108 | Potential race with _running flag |
| 11 | Memory not released immediately | cleanup_expired | Up to 30s delay |

---

## 12. Recommendations

### Must Fix (Critical)
1. **Persist SnapshotManager across requests**
   - Create singleton or store in HybridPipelineAdapter instance
   - Reuse same SnapshotManager for all generations
   - Start cleanup thread once at initialization

2. **Use deepcopy for node data**
   `python
   from copy import deepcopy
   node_contents[node_id] = deepcopy(getattr(node, 'content', ''))
   node_metadata[node_id] = deepcopy(getattr(node, 'metadata', {}))
   `

### Should Fix (High)
3. **Rename graph_learning.SnapshotManager**
   - Rename to KnowledgeSnapshotManager or ExportSnapshotManager
   - Avoid confusion with runtime snapshot manager

4. **Add logging for node retrieval failures**
   `python
   if not node:
       logger.warning(f"Node not found: {node_id}")
   `

### Could Fix (Medium)
5. **Make cleanup interval configurable**
6. **Add startup cleanup call**
7. **Add try/finally around cleanup thread to prevent hanging**

### Nice to Have (Low)
8. **Validate input parameters**
9. **Add lock to start()/stop()**
10. **Add unit tests**

---

## 13. Conclusion

### Overall Score: **5/10**

The SnapshotManager implementation has a solid foundation with proper thread safety and TTL mechanisms, but suffers from architectural issues that significantly impact its effectiveness:

**Strengths:**
- Thread-safe implementation using RLock
- Proper TTL-based expiration
- LRU eviction when max limit reached
- Clean separation of concerns (MemorySnapshot vs manager)

**Weaknesses:**
- Critical architectural flaw: per-request instantiation in HybridPipelineAdapter
- Missing deep copy of node data
- Name collision with another SnapshotManager class
- Silent error handling

### Quick Wins
1. Fix per-request instantiation (CRITICAL)
2. Add deepcopy for node data (CRITICAL)
3. Rename graph_learning SnapshotManager (HIGH)

---

## Appendix A: File Locations

| File | Path |
|------|------|
| Main SnapshotManager | C:\Users\black\OneDrive\Desktop\CogniFlex\eva_ai\memory\fractal_graph_v2\snapshot_manager.py |
| HybridPipelineAdapter | C:\Users\black\OneDrive\Desktop\CogniFlex\eva_ai\core\hybrid_pipeline_adapter.py |
| VirtualTokenHandler | C:\Users\black\OneDrive\Desktop\CogniFlex\eva_ai\memory\fractal_graph_v2\virtual_token_handler.py |
| Export SnapshotManager | C:\Users\black\OneDrive\Desktop\CogniFlex\eva_ai\memory\graph_learning.py |
| FractalGraph Init | C:\Users\black\OneDrive\Desktop\CogniFlex\eva_ai\memory\fractal_graph_v2\__init__.py |

---

## Appendix B: Related Classes

### MemorySnapshot (dataclass)
Lines: 21-61
Purpose: Immutable container for snapshot data

### VirtualTokenManager
Location: irtual_token_handler.py:259
Purpose: Coordinates virtual token handling with snapshot data

### VirtualTokenLogitsProcessor
Location: irtual_token_handler.py:32
Purpose: Boosts probabilities of virtual tokens during generation

### StreamingVirtualTokenHandler
Location: irtual_token_handler.py:101
Purpose: Replaces virtual tokens with content from snapshot during streaming

