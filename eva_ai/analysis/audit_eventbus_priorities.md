# EventBus Priority System Audit Report

**Date:** 2026-04-14  
**Auditor:** EVA AI System  
**Component:** EventBus (`eva_ai/core/event_bus.py`)  
**Assessment:** 3/10 (Priorities Not Implemented)

---

## Executive Summary

The EventBus implementation declares `priority` parameters in `publish()` and `subscribe()` methods, but **these priorities are completely ignored** at runtime. All event delivery follows strict FIFO (First-In-First-Out) ordering regardless of priority values.

This is a **critical architectural issue** for SelfDialogLearning, which requires prioritized message processing (contradictions > concepts > general dialog).

---

## Current Implementation Analysis

### Method Signatures

```python
def subscribe(self, event_type: str, callback, priority: int = 0)
def publish(self, event_type: str, data, priority: int = 0)
```

The `priority` parameter exists but is **never used**.

### Actual Behavior

All subscribers are stored in a simple list:

```python
self._subscribers[event_type] = []
```

Events are delivered using iteration order, which is **strictly FIFO**:

```python
for callback in self._subscribers.get(event_type, []):
    callback(event_data)
```

---

## Key Findings

| Issue | Severity | Status |
|-------|----------|--------|
| `priority` parameter ignored in `subscribe()` | Critical | **Not Implemented** |
| `priority` parameter ignored in `publish()` | Critical | **Not Implemented** |
| No priority queue for event delivery | Critical | **Not Implemented** |
| No differentiation between contradiction/concept dialogs | High | **Not Implemented** |
| FIFO-only delivery breaks SelfDialogLearning priority system | Critical | **Breaking Feature** |

---

## Impact Analysis

### SelfDialogLearning Priority Requirements

The system requires priority-based event processing:

```
CRITICAL: Contradiction resolution requests
HIGH: New concept exploration  
NORMAL: General dialog processing
LOW: Background tasks
```

### Current Broken Flow

```
1. User query → contradiction detected
2. Event published with priority=CRITICAL
3. SelfDialogLearning receives event...
4. But it's queued behind all NORMAL events
5. Contradiction not processed urgently ❌
```

### Components Affected

| Component | Priority Need | Current Status |
|-----------|---------------|----------------|
| SelfDialogLearning | HIGH (contradictions) | Broken |
| ConceptMiner | NORMAL | Works |
| ContradictionMiner | HIGH | Works |

---

## Technical Details

### Subscription Storage

```python
# Current implementation (ignores priority)
self._subscribers[event_type].append({
    'callback': callback,
    'priority': priority  # Stored but never used
})
```

### Event Delivery Loop

```python
# Current implementation (FIFO only)
for subscriber in self._subscribers.get(event_type, []):
    subscriber['callback'](event_data)  # priority not considered
```

---

## Root Cause

The `priority` parameter was added to the API but **no priority queue data structure was implemented**. The code stores priority values but never uses them for ordering.

---

## Required Fixes

### 1. Implement Priority Queue for Subscribers

```python
import heapq

class EventBus:
    def __init__(self):
        self._subscribers = {}  # event_type -> list of (priority, order, callback)
        self._subscriber_counter = 0  # For FIFO within same priority
    
    def subscribe(self, event_type: str, callback, priority: int = 0):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        heapq.heappush(self._subscribers[event_type], (priority, self._subscriber_counter, callback))
        self._subscriber_counter += 1
    
    def publish(self, event_type: str, data, priority: int = 0):
        for _, _, callback in self._subscribers.get(event_type, []):
            callback(data)
```

### 2. Event Prioritization in Publishing

Consider adding priority to event delivery order, not just subscriber order.

### 3. SelfDialogLearning Priority Integration

```python
# When queuing dialogs
self.event_bus.publish('dialog.request', {
    'type': 'contradiction',
    'data': contradiction_data
}, priority=100)  # CRITICAL priority

self.event_bus.publish('dialog.request', {
    'type': 'concept', 
    'data': concept_data
}, priority=50)   # HIGH priority
```

---

## Recommendations

1. **Immediate:** Implement priority queue for subscriber callbacks
2. **High:** Add priority-aware event delivery in `publish()`
3. **Medium:** Update SelfDialogLearning to use priority-based event publication
4. **Low:** Add unit tests for priority behavior

---

## Compliance Matrix

| Requirement | Current Status | Target Status |
|-------------|----------------|---------------|
| Priority parameter in subscribe() | Declared but unused | Must order delivery |
| Priority parameter in publish() | Declared but unused | Must influence delivery |
| FIFO within same priority | Works (by accident) | Must preserve |
| Priority-based ordering | **Broken** | Must implement |
| SelfDialogLearning priority | **Broken** | Must fix |

---

## Conclusion

The EventBus priority system is **not implemented**. The 3/10 rating reflects:
- +1 for having the API declared
- +1 for storing priority values
- +1 for not crashing
- 0 for actual priority functionality

**Estimated effort:** 2-3 hours to implement proper priority queue.

---

*Report generated by EVA AI System Audit*
