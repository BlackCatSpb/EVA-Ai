# EVA AI System - Import and Connection Issues Report

**Date:** 2026-03-30  
**System:** EVA AI (formerly CogniFlex)  
**Purpose:** Comprehensive audit of imports, module connections, and initialization logic

---

## Executive Summary

After parallel investigation by 3 AI Architects, **multiple critical issues** were found:

1. **FractalAttentionSystem** not connected to CoreBrain
2. **Event System API mismatch** - `.on()` vs `.subscribe()`
3. **DummyAttentionSystem** missing attributes
4. **SystemOptimizer** wrong parameter passed
5. **Integrator** never passed to GUI
6. **TrainingOrchestrator** initialization failures

---

## CRITICAL ISSUES

### 1. FractalAttentionSystem - Not Connected to CoreBrain

**Severity:** CRITICAL  
**Files:** 
- `eva/core/fractal_attention_system.py` (line 78)
- `eva/core/component_initializer.py` (lines 531-538)
- `eva/core/integration_layer.py` (line 117)

**Issue:** `FractalAttentionSystem` is created in `integration_layer.py` but never registered on `CoreBrain`, so `brain.attention_system` always returns `None`.

**component_initializer.py:531-538:**
```python
attention_system = getattr(self.core_brain, 'attention_system', None)
if attention_system is None:
    self.logger.warning("[WARN] attention_system не найден - используется DummyAttentionSystem")
    attention_system = DummyAttentionSystem()
```

**Fix Required:** Either:
- Initialize `FractalAttentionSystem` in `CoreBrain.__init__` and register it, OR
- Have `integration_layer.py` register it in `brain.components`

---

### 2. Event System API Mismatch

**Severity:** CRITICAL  
**Files:**
- `eva/gui/web_gui/bridge.py` (lines 43-55)
- `eva/gui/core_gui.py` (lines 635, 639, 645)
- `eva/gui/chat_module.py` (lines 170, 172)

**Issue:** Code uses `.on()` method which doesn't exist on `EventSystem`. Only `.subscribe()` exists.

**What Exists:** `CoreBrain` creates `self.events = EventSystem()` with:
- `.subscribe()` - NOT `.on()`
- `.trigger()` - NOT `.emit()`

**bridge.py:43-47:**
```python
self.brain.events.on('query_received', self._on_query_received)      # WRONG!
self.brain.events.on('response_generated', ...)                        # WRONG!
self.brain.events.on('training_progress', ...)                         # WRONG!
```

**Fix Required:** Replace all `.on()` with `.subscribe()` in:
- `bridge.py` lines 43-47, 51-55
- `core_gui.py` lines 635, 639, 645
- `chat_module.py` lines 170, 172

---

### 3. SystemOptimizer Wrong Parameter

**Severity:** CRITICAL  
**File:** `eva/core/fractal_attention_system.py:78`

**Issue:** `SystemOptimizer` expects `core_brain` but receives `FractalAttentionSystem` instance:

```python
def __init__(self, core_brain):
    ...
    self.system_optimizer = SystemOptimizer(self)  # BUG: passes self (FractalAttentionSystem), not self.core_brain
```

**Compare with system_optimizer.py:21:**
```python
def __init__(self, core_brain):  # Expects CoreBrain!
```

**Fix:** Change line 78 to `SystemOptimizer(self.core_brain)`

---

### 4. DummyAttentionSystem Missing core_brain Attribute

**Severity:** MODERATE  
**File:** `eva/core/component_initializer.py:533-537`

**Issue:** `DummyAttentionSystem` doesn't have `.core_brain` attribute that `LearningScheduler` may access:

```python
class DummyAttentionSystem:
    def __init__(self):
        self.pending_opportunities = []
    # Missing: self.core_brain = None
```

**Fix:** Add `self.core_brain = None` to `DummyAttentionSystem` class

---

### 5. Integrator Never Passed to GUI

**Severity:** MODERATE  
**Files:**
- `start_webgui.py` (line 68)
- `eva/run.py` (line 48)

**Issue:** GUI accepts `integrator` parameter but it's never passed from entry points.

**start_webgui.py:68:**
```python
gui = server.create_app(brain=brain)  # No integrator!
```

**Result:** `self.integrator` is always `None` in GUI, forcing fallback to direct brain access.

---

## MODERATE ISSUES

### 6. LearningScheduler Missing Methods

**Severity:** MODERATE  
**File:** `eva/core/learning_scheduler.py`

**Issue:** Missing methods called by integration layer:
- `identify_learning_opportunities()`
- `schedule_learning_session()`
- `get_high_priority_opportunities()`

**Fix:** Add these methods to `LearningScheduler` class

---

### 7. eva_tokenizer.py Missing Imports

**Severity:** LOW  
**File:** `eva/mlearning/eva_tokenizer.py`

**Issue:** Missing standard library imports that may be used:
- `re`
- `asyncio`
- `json`
- `ThreadPoolExecutor`

---

## COMPONENT DEPENDENCY MATRIX

| Component | Expected Dependency | Status |
|-----------|---------------------|--------|
| `learning_scheduler` | `attention_system` | BROKEN - Uses Dummy |
| `training_orchestrator` | `ml_unit`, `hybrid_cache` | PARTIAL - May fail |
| `knowledge_graph` | `event_bus` | OK |
| `fractal_attention` | `core_brain` | NOT REGISTERED |
| `gui` | `integrator` | NOT PASSED |

---

## SPECIFIC LINE REFERENCES

### bridge.py
| Line | Issue |
|------|-------|
| 43 | `self.brain.events.on('query_received', ...)` - `.on()` doesn't exist |
| 44-47 | Same `.on()` issue |
| 68 | `gui = server.create_app(brain=brain)` - No integrator |

### core_gui.py
| Line | Issue |
|------|-------|
| 635 | `self.brain.events.on('model_load', ...)` - `.on()` doesn't exist |
| 639 | `self.brain.events.on('models_ready', ...)` - `.on()` doesn't exist |
| 645 | `self.brain.events.on('request_gui_reload', ...)` - `.on()` doesn't exist |

### chat_module.py
| Line | Issue |
|------|-------|
| 170 | `brain.events.on('model_load', ...)` - `.on()` doesn't exist |
| 172 | `brain.events.on('models_ready', ...)` - `.on()` doesn't exist |

### fractal_attention_system.py
| Line | Issue |
|------|-------|
| 78 | `SystemOptimizer(self)` - passes wrong parameter |

### component_initializer.py
| Line | Issue |
|------|-------|
| 531-538 | `attention_system` always None, uses DummyAttentionSystem |
| 533-537 | DummyAttentionSystem missing `.core_brain` attribute |

---

## VERIFIED WORKING COMPONENTS

The following imports and modules are verified to exist and work:

- `eva.learning.self_dialog_learning` - EXISTS
- `eva.knowledge.online_knowledge` - EXISTS
- `eva.learning.self_analyzer` - EXISTS
- `eva.reasoning.integration` - EXISTS
- `eva.knowledge.knowledge_graph_integrated` - EXISTS
- `eva.knowledge.qwen_api_enhancer` - EXISTS
- `eva.learning.learning_manager` - EXISTS
- `eva.mlearning.training_orchestrator` - EXISTS
- `eva.gui.core_gui` - EXISTS (ЕВАGUI class)
- `eva.core.event_bus` - EXISTS
- `eva.security.security_framework` - EXISTS
- `eva.memory.memory_manager` - EXISTS
- `eva.memory.hybrid_token_cache` - EXISTS

---

## RECOMMENDED FIXES ORDER

### Priority 1 (Critical):
1. Fix `fractal_attention_system.py:78` - pass correct parameter to SystemOptimizer
2. Connect `FractalAttentionSystem` to CoreBrain (register in components)
3. Replace `.on()` with `.subscribe()` in all GUI files

### Priority 2 (Moderate):
4. Add `core_brain = None` to DummyAttentionSystem
5. Pass integrator to GUI in entry points
6. Add missing methods to LearningScheduler

### Priority 3 (Low):
7. Add missing imports to eva_tokenizer.py

---

## END OF REPORT

Report generated by parallel AI Architect investigation.
3 agents investigated: Core System, ML System, GUI System
