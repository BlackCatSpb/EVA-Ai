# AUDIT REPORT: Training System in EVA AI

**Date:** 2026-04-14
**Auditor:** EVA AI System Audit
**Files Analyzed:** 5

---

## 1. STRUCTURE OVERVIEW

### eva_ai/training/ (1 component)

| File | Size | Status |
|------|------|--------|
| __init__.py | 5 lines | OK |
| gguf_training_system.py | 789 lines | MAIN COMPONENT |
| __pycache__/ | compiled | - |

### eva_ai/core/background_jobs/ (related)

| File | Size | Purpose |
|------|------|----------|
| training_job.py | 45 lines | Background job wrapper |
| training_job.py.bak | 45 lines | Backup (orphaned) |
| base_job.py | 34 lines | Base class |

### eva_ai/mlearning/ (parallel training systems)

| File | Classes | Notes |
|------|---------|-------|
| text_quality_trainer.py | TextQualityTrainer | Has train_async() |
| fractal_trainer.py | FractalKnowledgeTrainer | Alternative trainer |

---

## 2. GGUFTrainingSystem ANALYSIS

### 2.1 Architecture

**Components:**
- TrainingStatus enum (IDLE, EXTRACTING, PREPARING, TRAINING, VERIFYING, MERGING, COMPLETED, FAILED)
- TrainingMetrics dataclass (loss, accuracy, knowledge_volume, etc.)
- VerifiedKnowledge dataclass (concept, description, links, source, confidence)
- GGUFTrainingSystem class (789 lines)

**Key Features:**
- Knowledge distillation via llama_cpp
- LoRA adapters by domain (programming, science, history, geography, general)
- Auto-start capability with configurable interval
- Model deployment and verification

### 2.2 Configuration Paths

| Parameter | Value | Status |
|-----------|-------|--------|
| base_model_path | C:\Users\black\OneDrive\Desktop\CogniFlex\eva_pie_architecture\models\gguf_models\ruadapt_qwen3_4b_q4_k_m.gguf | HARDCODED WINDOWS |
| training_model_path | eva_ai/models/training_qwen.gguf | RELATIVE |
| lora_path | eva_ai/models/lora_adapters | RELATIVE |
| verified_model_path | eva_ai/models/verified_qwen.gguf | RELATIVE |

### 2.3 Training Flow

1. _extract_verified_knowledge() - from knowledge graph
2. _prepare_training_data() - Q/A pairs generation
3. _train_separate_instance() - llama_cpp inference OR simulation
4. _verify_training_quality() - integrity + quality checks
5. _save_lora_adapters() - to disk

---

## 3. USAGE ANALYSIS

### 3.1 GGUFTrainingSystem Usage

**Initialization Points:**
- brain_init.py:119-128 - created in _init_training_system()
- brain.gguf_training = GGUFTrainingSystem(...)
- Auto-start called if auto_start=True and model verified

**Direct Usage:**
- NO direct usage found in other modules
- Only initialization + auto_start

### 3.2 TrainingJob (Background) Usage

**Job Configuration:**
- resource_class = GPU
- default_priority = CommandPriority.MEDIUM
- Looks for brain.memory_graph_trainer

**CRITICAL PROBLEM:**
MemoryGraphTrainer is explicitly set to None in SelfAnalyzer (line 64)!

TrainingJob looks for memory_graph_trainer which is explicitly set to None!

### 3.3 Parallel Training Systems

| System | Location | Method | Status |
|--------|----------|--------|--------|
| GGUFTrainingSystem | training/ | knowledge distillation | NOT USED |
| TextQualityTrainer | mlearning/ | train_async() | ACTIVE |
| FractalKnowledgeTrainer | mlearning/ | - | PRESENT |
| TrainingJob | background_jobs/ | memory_graph_trainer | BROKEN |

---

## 4. EVENTBUS INTEGRATION ANALYSIS

### 4.1 GGUFTrainingSystem EventBus Status

**Result: NO EVENTBUS INTEGRATION**

| Check | Status |
|-------|--------|
| Import EventBus | NO |
| subscribe() calls | NO |
| publish() calls | NO |
| EventBus parameter | NO |

### 4.2 TrainingJob EventBus Status

**Result: NO EVENTBUS INTEGRATION**

- No EventBus imports
- No event publishing
- No event subscriptions

### 4.3 Events Published

GGUFTrainingSystem publishes NO events:
- No training.started
- No training.completed
- No training.failed
- No lora.saved

Other components cannot react to training state changes.

---

## 5. DUPLICATION ANALYSIS

### 5.1 Training Concepts Duplicated

| Concept | Locations | Conflict |
|---------|-----------|----------|
| Training Loop | GGUFTrainingSystem, mlearning trainers | Yes |
| LoRA Adapters | GGUFTrainingSystem only | - |
| Model Paths | Multiple hardcoded | Yes |
| Training Status | Multiple enums | Yes |

### 5.2 Related Classes Duplicated

| Class | Locations |
|-------|-----------|
| TrainingStatus | training/gguf_training_system.py, core/enhanced_self_learning.py |
| TrainingConfig | mlearning/training_types.py, mlearning/ml_types.py |
| TrainingMetrics | training/gguf_training_system.py, mlearning/training_types.py |

### 5.3 File Duplication

- training_job.py.bak - orphaned backup file
- Multiple adaptation managers (see audit_adaptation.md)

---

## 6. CRITICAL BUGS FOUND

### BUG #1: self.output_dir NOT DEFINED

**Location:** gguf_training_system.py:640, 661

Line 640: adapters_dir = Path(self.output_dir) / "lora_adapters"
Line 661: prepared_path = Path(self.output_dir) / "training_data_prepared"

**Problem:** self.output_dir is NEVER defined in __init__ or anywhere else!

**Impact:** _save_lora_adapters() and _train_simulation() will crash with AttributeError

**Severity:** CRITICAL

---

### BUG #2: TrainingJob references REMOVED component

**Location:** training_job.py:20-23

trainer = getattr(brain, "memory_graph_trainer", None)
if trainer is None:
    sa = getattr(brain, "self_analyzer", None)
    trainer = getattr(sa, "memory_graph_trainer", None) if sa else None

**Problem:** memory_graph_trainer is explicitly set to None in SelfAnalyzer (line 64)

**Impact:** TrainingJob always returns early, never executes training

**Severity:** CRITICAL

---

### BUG #3: Hardcoded Windows Path

**Location:** gguf_training_system.py:84

base_dir = r"C:\Users\black\OneDrive\Desktop\CogniFlex\eva_pie_architecture\models\gguf_models"

**Problem:** Not portable, uses absolute Windows path

**Severity:** HIGH

---

### BUG #4: Two _save_lora_adapters() Methods

**Location:** gguf_training_system.py:637, 733

Line 637: def _save_lora_adapters(self):  # Uses self.output_dir
Line 733: def _save_lora_adapters(self):  # Uses self.lora_path

**Problem:** Duplicate method names - second one shadows first

**Severity:** MEDIUM

---

## 7. INTEGRATION GAPS

### 7.1 What is Missing

| Integration Point | Status | Notes |
|-------------------|--------|-------|
| GGUFTrainingSystem -> SelfDialogLearning | NONE | No connection |
| TrainingJob -> TrainingSystem | BROKEN | references removed component |
| LoRA -> Production Model | NONE | LoRA never applied |
| Training -> EventBus | NONE | No events published |
| Knowledge Graph -> Training | PARTIAL | Extracts but does not write back |

### 7.2 Unused Code Paths

gguf_training_system.py - never executed:

if model:
    self._train_via_distillation(model, training_data)  # if llama_cpp available
else:
    self._train_simulation(training_data)  # fallback - BROKEN due to output_dir

---

## 8. ASSESSMENT

### Overall Score: 3.5 / 10

| Category | Score | Notes |
|----------|-------|-------|
| Implementation | 5/10 | Theory OK, bugs block execution |
| Integration | 2/10 | Isolated, no EventBus |
| Usability | 3/10 | Not connected to main system |
| Reliability | 3/10 | Critical bugs present |
| Documentation | 5/10 | Clear docstrings |

### Detailed Breakdown

**Strengths:**
- Well-structured knowledge distillation concept
- LoRA adapter architecture is sound
- Auto-start mechanism is good
- Comprehensive logging

**Weaknesses:**
- NOT integrated with main learning system
- NO EventBus integration
- Critical bugs prevent operation
- Hardcoded paths
- Duplicate classes

---

## 9. RECOMMENDATIONS

### Immediate (Must Fix)

1. **Fix self.output_dir bug**
   - Add self.output_dir = self.config.get("output_dir", "eva_ai/models") in __init__

2. **Fix TrainingJob**
   - Update to use GraphCurator or remove dead code
   - Or reconnect to GGUFTrainingSystem

3. **Add EventBus integration**
   - Publish: training.started, training.completed, training.failed
   - Subscribe to: system.idle, graph.updated

### Short-term

4. **Connect LoRA to production**
   - Add method to apply LoRA adapters to main model

5. **Remove hardcoded paths**
   - Use relative paths or config

6. **Delete orphaned files**
   - training_job.py.bak

### Long-term

7. **Unify training systems**
   - Choose ONE training approach
   - Remove duplicates

8. **Add training triggers**
   - Connect to SelfDialogLearning for continuous learning

---

## 10. CONCLUSION

The GGUFTrainingSystem is a well-designed but poorly integrated training subsystem. It has critical bugs that prevent execution and lacks any connection to the main EVA AI learning pipeline. The parallel TrainingJob system references components that have been removed, making it completely non-functional.

**Priority:** MEDIUM-HIGH
**Risk:** HIGH (broken code in production path)
**Effort to Fix:** MEDIUM
