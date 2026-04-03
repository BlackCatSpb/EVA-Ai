# SYSTEM AUDIT — CogniFlex / EVA AI
*Generated: 2026-04-03*
*Last Updated: 2026-04-03 (Phase 1 fixes applied)*
*Auditors: AI Architecture Team (3 agents)*

---

## FIX STATUS

| ID | Issue | Status | Fixed In |
|----|-------|--------|----------|
| C1 | GraphCoordinator import broken | ✅ FIXED | Phase 1 |
| C2 | Method name build_context_array → build_context | ✅ FIXED | Phase 1 |
| C3 | God Method process_query (~900 lines) | ⏳ PLANNED | Phase 3 |
| C4 | Class-level constant mutation | ✅ FIXED | Phase 1 |
| C5 | os.chdir() side effect | ✅ FIXED | Phase 1 |
| C6 | Model B file existence check | ✅ FIXED | Phase 1 |
| X1 | Circular dependency pattern | ⏳ ACCEPTED | Architecture |
| X2 | Duplicate component creation | ✅ FIXED | Phase 1c |
| X3 | Configuration drift | ⏳ PLANNED | Phase 4 |
| X4 | Model B path risk | ✅ FIXED | Phase 1 (C6) |

## CLEANUP SUMMARY

- **Imports removed:** 13 dead imports across 6 files
- **Methods removed:** 7 dead methods + 4 dead factories
- **Fields removed:** 8 unused fields across 3 files
- **Duplicate code removed:** 2 duplicate component creations, 1 duplicate handler call
- **Broken code fixed:** 4 broken try/except blocks, BOM character

## FILE SIZE REDUCTION

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| core_brain.py | 3676 lines | ~3550 lines | ~126 lines |
| component_initializer.py | 1277 lines | ~1190 lines | ~87 lines |
| recursive_model_pipeline.py | 640 lines | 639 lines | 1 line |
| unified_fractal_memory.py | 545 lines | ~540 lines | ~5 lines |
| gguf_fractal_exporter.py | 345 lines | 344 lines | 1 line |
| graph_learning.py | 492 lines | ~485 lines | ~7 lines |

---

## TABLE OF CONTENTS
1. [Architecture Overview](#1-architecture-overview)
2. [Critical Issues](#2-critical-issues)
3. [Module Audits](#3-module-audits)
4. [Cross-Module Issues](#4-cross-module-issues)
5. [Dead Code Inventory](#5-dead-code-inventory)
6. [Optimization Plan](#6-optimization-plan)
7. [Fix Priority Matrix](#7-fix-priority-matrix)

---

## 1. ARCHITECTURE OVERVIEW

### Current System Structure
```
brain_config.json
    │
    ▼
core_brain.py (~3550 lines) ──────► component_initializer.py (~1190 lines)
    │                                      │
    ├─► recursive_model_pipeline.py (639 lines)
    ├─► unified_fractal_memory.py (~540 lines)
    │       ├─► graph_learning.py (~485 lines)
    │       └─► gguf_fractal_exporter.py (344 lines)
    ├─► self_reasoning_engine.py (1799 lines)
    └─► 20+ other modules
```

### Three-Model Pipeline
- **Model A**: Qwen 2.5 3B (Logic) — n_ctx=2048, temp=0.3
- **Model B**: Qwen 2.5 3B clone (Concepts) — n_ctx=2048, temp=0.3
- **Model C**: Qwen 2.5 Coder 1.5B (Code) — lazy load, n_ctx=2048

### Graph Learning (FIXED ✅)
- ExperienceNode: Q&A pairs with embeddings
- ConceptNode: Clustered knowledge patterns
- DynamicContextBuilder: Context assembly from graph
- GraphLearningLoop: Background clustering (daemon thread)
- SnapshotManager: Export/import knowledge snapshots

---

## 2. CRITICAL ISSUES

### C1: Graph Learning Never Initializes ✅ FIXED
**Was:** Imported non-existent `GraphCoordinator` class.
**Fix:** Replaced with `DynamicContextBuilder` in `unified_fractal_memory.py:171`.

### C2: Method Name Mismatch ✅ FIXED
**Was:** Called `build_context_array()` which doesn't exist.
**Fix:** Changed to `build_context()` in `unified_fractal_memory.py:196`.

### C3: God Method — process_query (~900 lines) ⏳ PLANNED
**File:** `core_brain.py:1345-2248`
**Status:** Requires major refactoring — planned for Phase 3.

### C4: Class-Level Constant Mutation ✅ FIXED
**Was:** `self.MODEL_A_MAX_TOKENS = params_a.get(...)` mutated class attributes.
**Fix:** Uses local variables `a_max_tokens`, `a_temperature`, etc. in `recursive_model_pipeline.py`.

### C5: Working Directory Side Effect ✅ FIXED
**Was:** `_ensure_eva_path()` called `os.chdir(eva_root)` at module load time.
**Fix:** Removed `os.chdir()` call. All paths now absolute.

### C6: Model B Path Risk ✅ FIXED
**Was:** No file existence check — silent failure if Model B file missing.
**Fix:** Added `os.path.exists()` checks with fallback to Model A path.

---

## 3. MODULE AUDITS

### 3.1 core_brain.py (~3550 lines)

**Removed Imports:**
- ~~`datetime`~~ ✅
- ~~`TYPE_CHECKING`~~ ✅
- ~~`get_generation_coordinator`~~ ✅
- ~~`LearningOpportunityDetector`~~ ✅
- ~~`AutopilotCache`~~ ✅
- ~~`WebDiscoveryDetector`~~ ✅
- ~~`ModuleRecoveryDetector`~~ ✅

**Removed Methods:**
- ~~`_check_system_ready_for_training`~~ ✅
- ~~`setup_smart_cache_eviction`~~ ✅
- ~~Duplicate `_register_deferred_system_handlers` call~~ ✅
- ~~Duplicate `ReasoningIntegration` creation~~ ✅

**Remaining Issues:**
- `[CRITICAL]` `process_query` method is ~900 lines — needs refactoring
- `[HIGH]` `__init__` is ~490 lines — should be split into phases
- `[MEDIUM]` `reboot` method may cause resource leaks on re-initialization

### 3.2 recursive_model_pipeline.py (639 lines)

**Removed:**
- ~~`Tuple` import~~ ✅

**Remaining Issues:**
- `[HIGH]` `check_quality` flags "Конечно!" as gibberish — overly aggressive
- `[HIGH]` `check_quality` flags >30% English as gibberish — breaks code queries
- `[MEDIUM]` No backoff delay between retries
- `[MEDIUM]` `_sanitize_response` has overly specific replacements

### 3.3 component_initializer.py (~1190 lines)

**Removed:**
- ~~`threading` import~~ ✅
- ~~`create_training_orchestrator` factory~~ ✅
- ~~`create_learning_manager` factory~~ ✅
- ~~`create_learning_scheduler` factory~~ ✅
- ~~`create_gui` factory~~ ✅
- ~~`gui` from component_dependencies~~ ✅

**Remaining Issues:**
- `[HIGH]` Mixed language log messages
- `[MEDIUM]` Single early failure cascades to all dependent components

### 3.4 unified_fractal_memory.py (~540 lines)

**Removed:**
- ~~`Tuple` import~~ ✅
- ~~`MemoryNode.version` field~~ ✅
- ~~`MemoryNode.relations` field~~ ✅

**Remaining Issues:**
- `[HIGH]` Hot/warm node tracking sets never updated after initial load
- `[MEDIUM]` `close()` does not stop `GraphLearningLoop` daemon thread

### 3.5 gguf_fractal_exporter.py (344 lines)

**Removed:**
- ~~`hashlib` import~~ ✅
- ~~`Optional` import~~ ✅
- ~~`self.tensors` field~~ ✅

**Remaining Issues:**
- `[HIGH]` Hard top-level import of `llama_cpp`
- `[MEDIUM]` Uses `rope_freq_base` where `rms_norm_eps` is semantically correct

### 3.6 graph_learning.py (~485 lines)

**Removed:**
- ~~`Optional`, `Tuple` imports~~ ✅
- ~~`ExperienceNode.tags` field~~ ✅
- ~~`ExperienceNode.related_experiences` field~~ ✅
- ~~`ConceptNode.updated_at` field~~ ✅
- ~~`ConceptNode.generation` field~~ ✅
- ~~`ConceptNode.parent_concepts` field~~ ✅

**Remaining Issues:**
- `[HIGH]` Write side-effects in read-only `_find_relevant_experiences`
- `[MEDIUM]` Direct access to private methods

### 3.7 brain_config.json (130 lines)

**Remaining Issues:**
- `[CRITICAL]` `disable_pytorch: true` but `type: "qwen"` refers to PyTorch model
- `[CRITICAL]` `enable_training: false` AND `training_disabled: true` — redundant
- `[CRITICAL]` `self_dialog_learning.enabled: true` but `disable_learning_threads: true` — contradiction

---

## 4. CROSS-MODULE ISSUES

### X1: Circular Dependency Pattern ⏳ ACCEPTED
`CoreBrain.__init__` creates `ComponentInitializer(self)`, and factories set attributes on `self.core_brain`. Tight coupling but functional.

### X2: Duplicate Component Creation ✅ FIXED
- ~~`QueryProcessor` created twice~~ — ComponentInitializer handles it
- ~~`ReasoningIntegration` created twice~~ — Removed from core_brain.py

### X3: Configuration Drift ⏳ PLANNED
`brain_config.json` has settings no module reads. Planned for Phase 4.

### X4: Model B File Dependency ✅ FIXED
File existence check added with fallback to Model A path.

---

## 5. DEAD CODE INVENTORY (CLEANED)

### Imports Removed ✅
| File | Import | 
|------|--------|
| core_brain.py | `datetime`, `TYPE_CHECKING`, `get_generation_coordinator`, `LearningOpportunityDetector`, `AutopilotCache`, `WebDiscoveryDetector`, `ModuleRecoveryDetector` |
| recursive_model_pipeline.py | `Tuple` |
| component_initializer.py | `threading` |
| unified_fractal_memory.py | `Tuple` |
| gguf_fractal_exporter.py | `hashlib`, `Optional` |
| graph_learning.py | `Optional`, `Tuple` |

### Methods Removed ✅
| File | Method |
|------|--------|
| core_brain.py | `_check_system_ready_for_training`, `setup_smart_cache_eviction`, duplicate `_register_deferred_system_handlers`, duplicate `ReasoningIntegration` creation |
| component_initializer.py | `create_training_orchestrator`, `create_learning_manager`, `create_learning_scheduler`, `create_gui` |

### Fields Removed ✅
| File | Field |
|------|-------|
| unified_fractal_memory.py | `MemoryNode.version`, `MemoryNode.relations` |
| graph_learning.py | `ExperienceNode.tags`, `ExperienceNode.related_experiences`, `ConceptNode.updated_at`, `ConceptNode.generation`, `ConceptNode.parent_concepts` |
| gguf_fractal_exporter.py | `self.tensors` |

---

## 6. OPTIMIZATION PLAN

### Phase 1: Fix Critical ✅ COMPLETE
- [x] C1: Fix graph learning import
- [x] C2: Fix method name mismatch
- [x] C4: Fix class-level constant mutation
- [x] C5: Remove os.chdir() side effect
- [x] C6: Add Model B file existence check
- [x] Dead code cleanup (imports, methods, fields, factories)

### Phase 2: Quality Improvements ⏳ PLANNED
- [ ] Fix `check_quality` — don't flag polite phrases as gibberish
- [ ] Fix `check_quality` — handle code queries with English keywords
- [ ] Add backoff delay between retries
- [ ] Fix `_sanitize_response` — remove overly specific replacements
- [ ] Fix `rope_freq_base` vs `rms_norm_eps` semantic error

### Phase 3: Refactor God Methods ⏳ PLANNED
- [ ] Break `process_query` (~900 lines) into strategy pattern
- [ ] Split `CoreBrain.__init__` (~490 lines) into phases

### Phase 4: Configuration Cleanup ⏳ PLANNED
- [ ] Remove redundant config keys
- [ ] Add missing config sections
- [ ] Fix contradictions

### Phase 5: Threading & Safety ⏳ PLANNED
- [ ] Add graceful shutdown for `GraphLearningLoop` daemon thread
- [ ] Fix thread-safety in `_find_relevant_experiences`
- [ ] Fix encapsulation violations

---

## 7. FIX PRIORITY MATRIX

| Priority | Issue | Effort | Impact | Status |
|----------|-------|--------|--------|--------|
| P0 | C1: Graph learning import broken | 5min | HIGH | ✅ DONE |
| P0 | C2: Method name mismatch | 2min | HIGH | ✅ DONE |
| P0 | C4: Class-level constant mutation | 10min | HIGH | ✅ DONE |
| P0 | C5: Working directory side effect | 15min | MEDIUM | ✅ DONE |
| P0 | C6: Model B file existence check | 15min | MEDIUM | ✅ DONE |
| P1 | Dead code cleanup | 30min | LOW | ✅ DONE |
| P1 | Duplicate component creation | 30min | MEDIUM | ✅ DONE |
| P2 | Quality improvements | 1h | MEDIUM | ⏳ PLANNED |
| P3 | God method refactoring | 2h | MEDIUM | ⏳ PLANNED |
| P4 | Configuration cleanup | 30min | LOW | ⏳ PLANNED |
| P5 | Threading & safety | 2h | MEDIUM | ⏳ PLANNED |

---

## 8. TOKENIZER OPTIMIZATION (PLANNED)

### Current State
- Tokenizer: Qwen tokenizer loaded via transformers/AutoTokenizer
- Location: `eva/mlearning/eva_models/qwen3.5-0.8b` (PyTorch model path)
- Used by: `UnifiedTextProcessor`, `ResponseGenerator`, `MLUnit`
- Issue: Loads full PyTorch model just for tokenizer weights

### Optimization Opportunities
1. **Extract tokenizer only** — Don't load full PyTorch model, just tokenizer.json
2. **Use llama_cpp tokenizer** — GGUF models have built-in tokenizers, no need for separate PyTorch tokenizer
3. **Pre-compute token cache** — Cache frequent query tokenizations
4. **Batch tokenization** — Tokenize multiple queries at once

### Proposed Architecture
```
Current:  Query → PyTorch Tokenizer (slow, loads full model)
Proposed: Query → llama_cpp tokenizer (fast, already in memory)
```

---

*This document serves as the living audit cache. Update as fixes are applied.*
