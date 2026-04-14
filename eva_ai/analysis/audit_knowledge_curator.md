# Audit Report: KnowledgeCurator System

**Date:** 2026-04-14
**Auditor:** EVA AI Agent
**Component:** KnowledgeCurator / GraphCurator
**Integration Score:** 2/10

---

## Executive Summary

**CRITICAL FINDING:** KnowledgeCurator does NOT exist in the codebase.

The codebase contains only GraphCurator (eva_ai/knowledge/graph_curator.py), which serves as the knowledge graph curator but lacks proper integration with EventBus and DeferredCommandSystem.

---

## 1. KnowledgeCurator Search Results

| Search Pattern | Found | Location |
|----------------|-------|----------|
| KnowledgeCurator class | NO | - |
| KnowledgeCurator import | NO | - |
| GraphCurator class | YES | eva_ai/knowledge/graph_curator.py |
| curator.* events | YES (subscriptions) | eva_ai/learning/dialog_core.py |

**Conclusion:** KnowledgeCurator is either planned but not implemented, or the user means GraphCurator.

---

## 2. GraphCurator Implementation Analysis

### 2.1 File Location
- **Path:** eva_ai/knowledge/graph_curator.py
- **Lines:** 460
- **Class:** GraphCurator

### 2.2 Core Functions

GraphCurator performs the following operations on FractalGraph v2:

| Function | Description | Status |
|----------|-------------|--------|
| _cleanup_garbage() | Removes orphaned and old nodes | Implemented |
| _process_level_promotions() | Promotes/demotes nodes | Implemented |
| _consolidate_nodes() | Creates semantic groups | Implemented |
| _update_metrics() | Updates curator metrics | Implemented |
| _merge_overlapping_groups() | Merges groups >70 overlap | Implemented |

### 2.3 Protected Node Types

GraphCurator protects these node types from deletion:
- concept, contradiction
- model_a, model_b, model_c, model_root
- semantic_group, domain_profile
- Nodes with is_static=True
- Nodes with high confidence (>0.7) and frequent access (>10)

### 2.4 Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| MIN_EFFECTIVE_CONFIDENCE | 0.15 | Minimum confidence |
| PROMOTE_THRESHOLD | 0.8 | For level promotion |
| DEMOTE_THRESHOLD | 0.2 | For level demotion |
| check_interval | 600 sec | Between curation cycles |

---

## 3. EventBus Integration Analysis

### 3.1 Current State: NO INTEGRATION

| Integration Point | Required | Implemented | Status |
|-------------------|----------|-------------|--------|
| EventBus subscription | YES | NO | FAIL |
| EventBus publication | YES | NO | FAIL |
| DeferredCommandSystem | YES | NO | FAIL |

### 3.2 SelfDialogLearning Subscriptions

SelfDialogLearning subscribes to these curator.* events:
- curator.knowledge_extracted - NEVER PUBLISHED
- curator.graph_optimized - NEVER PUBLISHED
- curator.cleanup_done - NEVER PUBLISHED

### 3.3 Evidence

No event_bus usage found in graph_curator.py:
- No self.event_bus assignment
- No subscribe() calls
- No publish() calls

---

## 4. DeferredCommandSystem Analysis

### 4.1 Assignment (brain_init.py:111)
brain.graph_curator._deferred_system = brain.deferred_system

But in graph_curator.py:
- NO usage of self._deferred_system
- Uses threading.Timer directly (line 151)
- Fixed 600 second interval

### 4.2 Comparison with Similar Components

| Component | Uses DeferredSystem | Uses EventBus |
|-----------|---------------------|---------------|
| GraphCurator | NO (unused) | NO |
| ConceptMiner | YES | YES |
| ContradictionMiner | YES | YES |

---

## 5. Operations Performed

### 5.1 Cleanup Operations
- Remove orphaned nodes (no edges)
- Remove nodes with effective_confidence < 0.15
- Remove nodes older than 7 days
- Maximum 50 nodes removed per cycle

### 5.2 Promotion/Demotion
- Promote: confidence > 0.8 AND access_count > 5 AND level < 3
- Demote: confidence < 0.2 AND access_count < 2 AND level > 0

### 5.3 Consolidation
- Cluster nodes by cosine similarity (threshold 0.6)
- Create SemanticGroup for clusters with 3+ members
- Merge groups with >70% member overlap

---

## 6. Duplication Analysis

### 6.1 Functional Overlap

| Component | Domain | Function | Overlap |
|-----------|--------|----------|---------|
| GraphCurator | Knowledge Graph | Optimization | - |
| ConceptMiner | Knowledge | Gap detection | LOW |
| ContradictionMiner | Knowledge | Contradictions | LOW |

### 6.2 Conclusion

NO SIGNIFICANT DUPLICATION - Components serve different purposes.

---

## 7. Critical Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| No EventBus Integration | CRITICAL | Cannot communicate |
| Orphaned Event Subscriptions | CRITICAL | Dead code |
| Unused DeferredSystem | HIGH | Memory waste |
| Hardcoded Timing | MEDIUM | No adaptation |
| Missing is_running() | MEDIUM | Breaks brain_init |

---

## 8. Score Calculation

| Criterion | Max | Score | Notes |
|-----------|-----|-------|-------|
| EventBus Integration | 3 | 0 | No integration |
| DeferredCommandSystem | 2 | 0 | Assigned but unused |
| Proper Timing | 1 | 0 | Hardcoded Timer |
| Event Publishing | 2 | 0 | Never published |
| Coordination | 1 | 0 | Isolated |
| Code Quality | 1 | 1 | Well-structured |
| TOTAL | 10 | 1 | Critical failure |

---

## 9. Recommendations

### Critical
1. Add EventBus Integration
2. Use DeferredCommandSystem
3. Add is_running() Method

### High Priority
4. Remove Hardcoded Interval
5. Fix Event Publishing

### Medium Priority
6. Add ModelAccessManager Coordination

---

## 10. Comparison

| Component | File | Lines | EventBus | DeferredSystem |
|-----------|------|-------|----------|----------------|
| GraphCurator | graph_curator.py | 460 | NO | NO |
| ConceptMiner | concept_miner.py | 1040 | YES | YES |
| ContradictionMiner | contradiction_miner.py | 922 | YES | YES |
| SelfDialogLearning | dialog_core.py | 982 | YES | YES |

---

## 11. Final Verdict

**FINAL SCORE: 2/10**

Reason: KnowledgeCurator does not exist. GraphCurator exists but:
- Has NO EventBus integration
- Has unused DeferredCommandSystem reference
- Publishes NO events
- Creates dead subscriptions in SelfDialogLearning
- Uses deprecated threading.Timer pattern

Action Required: Complete re-architecture of GraphCurator integration.

---

*Report generated: 2026-04-14*
*Auditor: EVA AI Agent*
