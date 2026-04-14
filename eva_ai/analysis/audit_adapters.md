# EVA AI Adapters Subsystem Audit Report

**Generated:** Tue Apr 14 2026  
**Auditor:** Automated Code Audit  
**Scope:** All adapter implementations in EVA AI  

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Adapters Found** | 9 |
| **Active/Used** | 5 |
| **Deprecated/Orphaned** | 2 |
| **Needs Review** | 2 |
| **Overall Score** | **6.5/10** |

---

## 1. Adapters Inventory

### 1.1 TorchBatchAdapter
**Location:** va_ai/adapters/torch_adapter.py (230 lines)

| Attribute | Value |
|-----------|-------|
| **Class** | TorchBatchAdapter |
| **Purpose** | PyTorch batch accumulation with timeout/token limits |
| **Dependencies** | 	orch, va_ai.core.batch_wrapper |
| **EventBus** | Via vents parameter (optional) |
| **Status** | ORPHANED - Not imported anywhere |

**Issues:**
- Not imported/used anywhere in codebase
- Only referenced in docstrings of other files
- events parameter exists but no actual EventBus integration

---

### 1.2 PipelineAdapter
**Location:** va_ai/core/pipeline_adapter.py (312 lines)

| Attribute | Value |
|-----------|-------|
| **Class** | PipelineAdapter |
| **Purpose** | Compatibility layer for UnifiedGenerator to TwoModelPipeline API |
| **Dependencies** | va_ai.core.unified_generator |
| **EventBus** | Yes - passed to UnifiedGenerator |
| **Status** | ACTIVE - Used as fallback |

**Usage Locations:**
- va_ai/core/brain_components.py (line 1025) - fallback creation
- va_ai/learning/dialog_concepts.py (line 113) - type checking

**Issues:**
- Wrapper around HybridPipelineAdapter/UnifiedGenerator
- Code duplication with HybridPipelineAdapter

---

### 1.3 HybridPipelineAdapter
**Location:** va_ai/core/hybrid_pipeline_adapter.py (720 lines)

| Attribute | Value |
|-----------|-------|
| **Class** | HybridPipelineAdapter |
| **Purpose** | MAIN ADAPTER - Hybrid routing between Fractal/Dual/Recursive pipelines |
| **Dependencies** | FractalPipeline, DualGenerator, RecursiveModelPipeline |
| **EventBus** | No direct usage - accesses via brain parameter |
| **Status** | PRIMARY - Main pipeline adapter |

**Modes:** fractal, dual, recursive, hybrid

**Usage Locations:**
- va_ai/core/brain_components.py (lines 278, 399) - PRIMARY CREATION
- va_ai/contradiction/contradiction_miner.py - EVAGenerator calls
- va_ai/knowledge/concept_miner.py - EVAGenerator calls

**Issues:**
- No direct EventBus integration - relies on brain passed in constructor
- Large class (720 lines) - could be split

---

### 1.4 KnowledgeGraphAdapter
**Location:** va_ai/knowledge/kg_adapter.py (171 lines)

| Attribute | Value |
|-----------|-------|
| **Class** | KnowledgeGraphAdapter |
| **Purpose** | Compatibility layer - redirects KG API calls to FractalGraph v2 |
| **Dependencies** | FractalMemoryGraph |
| **EventBus** | No |
| **Status** | ACTIVE - Used for KG backward compatibility |

**Usage Locations:**
- va_ai/core/init_factories.py (line 504)
- va_ai/knowledge/knowledge_graph.py (line 28)

**Issues:**
- __getattr__ swallows all errors silently
- No EventBus integration for graph updates

---

### 1.5 PieIntegration
**Location:** va_ai/memory/pie_integration/pie_adapter.py (373 lines)

| Attribute | Value |
|-----------|-------|
| **Class** | PieIntegration |
| **Purpose** | Integration of L1 (ActivationProfiler) + L2 (RoutingEngine) + L3 (FGv2) |
| **EventBus** | No |
| **Status** | ACTIVE - Used in DualGenerator |

**Issues:**
- No EventBus integration
- L1 entropy estimation is TODO (hardcoded 0.5)

---

### 1.6 ModelStorageAdapter
**Location:** va_ai/mlearning/storage/model_storage_adapter.py (111 lines)

| Attribute | Value |
|-----------|-------|
| **Class** | ModelStorageAdapter |
| **Purpose** | Integrate MemoryGraphStore with ModelManager |
| **Status** | ORPHANED - Not imported anywhere |

**Issues:**
- Not imported anywhere in codebase
- No tests found

---

### 1.7 FractalPipelineAdapter
**Location:** va_ai/core/fractal_pipeline.py (line 190-267)

| Attribute | Value |
|-----------|-------|
| **Status** | WRAPPER - Used internally by HybridPipelineAdapter |

**Issues:**
- Not actively used
- Fallback not implemented (NotImplementedError)

---

### 1.8 OpenVINOCacheAdapter
**Location:** va_ai/core/openvino_generator.py (line 702+)

| Status | PARTIAL - Only partial code shown |
|--------|-----------------------------------|

**Issues:**
- Only 50 lines shown
- Not found in any imports

---

### 1.9 LlamaCppTokenizerAdapter
**Location:** va_ai/mlearning/tokenizer_registry.py (line 147-212)

| Status | PARTIAL - Only in docstring |
|--------|------------------------------|

---

### 1.10 integration_adapters.py
**Location:** va_ai/core/integration_adapters.py (330 lines)

| Type | MODULE with standalone functions |
|------|----------------------------------|

**Issues:**
- Functions need to be attached to a class instance
- _setup_adapters is empty (pass)

---

## 2. EventBus Integration Analysis

| Adapter | EventBus Usage |
|---------|----------------|
| PipelineAdapter | Indirect (via UnifiedGenerator) |
| HybridPipelineAdapter | None |
| KnowledgeGraphAdapter | None |
| PieIntegration | None |
| integration_adapters.py | Yes (handlers designed for it) |
| TorchBatchAdapter | Optional |

**Critical Gaps:**
1. HybridPipelineAdapter - No EventBus despite being central
2. KnowledgeGraphAdapter - Graph updates not published

---

## 3. Usage Statistics

`
ACTIVE:
- HybridPipelineAdapter: brain_components (primary)
- KnowledgeGraphAdapter: init_factories, knowledge_graph
- PieIntegration: dual_generator_pie

FALLBACK:
- PipelineAdapter: brain_components, dialog_concepts

ORPHANED:
- TorchBatchAdapter: nowhere used
- ModelStorageAdapter: nowhere used
- FractalPipelineAdapter: docstring only
`

---

## 4. Critical Issues

| ID | Issue | Severity |
|----|-------|----------|
| C1 | No EventBus in HybridPipelineAdapter | HIGH |
| C2 | No EventBus in KnowledgeGraphAdapter | HIGH |
| C3 | TorchBatchAdapter completely orphaned | HIGH |
| C4 | ModelStorageAdapter completely orphaned | HIGH |

---

## 5. Final Score

| Category | Score | Max | Weight |
|----------|-------|-----|--------|
| Code Quality | 7 | 10 | 30% |
| EventBus Integration | 4 | 10 | 25% |
| Usage/Active | 7 | 10 | 25% |
| Documentation | 6 | 10 | 10% |
| Test Coverage | 5 | 10 | 10% |
| **OVERALL** | **6.5** | 10 | 100% |

---

## 6. Recommendations

1. Add EventBus to HybridPipelineAdapter
2. Add EventBus to KnowledgeGraphAdapter  
3. Remove or document TorchBatchAdapter
4. Remove or document ModelStorageAdapter
5. Merge PipelineAdapter into HybridPipelineAdapter

---

END OF REPORT
