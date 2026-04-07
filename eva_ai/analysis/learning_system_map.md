# CogniFlex Learning System Analysis

**Date:** March 21, 2026  
**Status:** CRITICAL - Significant Fake/Stub Implementation Found

---

## PART 1: All Learning Files Analysis

### File: `cogniflex/learning/self_dialog_learning.py`

**Class: `SelfDialogLearningSystem`** (NEW learning system)
- Inherits from `BaseComponent`
- **Status: ACTIVE** (enabled in config)

**Key Methods:**
| Method | Purpose | Real/Fake |
|--------|---------|-----------|
| `run_self_dialog_cycle()` | Executes full dialog cycle (question -> response -> analysis) | REAL |
| `generate_question()` | Uses model.generate() or fallback string | REAL with fallback |
| `generate_response()` | Uses model.generate() or fallback string | REAL with fallback |
| `_check_ethics()` | Calls ethics_framework.analyze_content() | REAL with fallback |
| `_detect_contradictions()` | Calls contradiction_manager.detect_contradictions() | REAL with fallback |
| `_verify_facts()` | Calls web_search_engine.search() | REAL with fallback |
| `_analyze_quality()` | Uses analytics_manager or direct analysis | REAL |
| `store_learning_insight()` | Stores insights to memory/knowledge_graph | REAL |
| `_save_insight_to_file()` | Saves JSON to cache/self_dialog_insights/ | REAL |

**Manager Dependencies:**
- `_model_manager` (fractal_model_manager)
- `_memory_manager`
- `_knowledge_graph`
- `_ethics_framework`
- `_contradiction_manager`
- `_web_search_engine`
- `_analytics_manager`

**Used By:**
- CoreBrain (initialized and started)

---

### File: `cogniflex/learning/memory_graph_trainer.py`

**Class: `MemoryGraphNetwork`** (PyTorch neural network)
- Real neural network architecture for graph training

**Class: `MemoryGraphTrainer`**
- **Status: DISABLED** by config (`training_disabled=True`, `enable_training=False`)

**Key Methods:**
| Method | Purpose | Real/Fake |
|--------|---------|-----------|
| `prepare_training_data()` | Gets nodes/edges from knowledge_graph | REAL with synthetic fallback |
| `_create_synthetic_data()` | Creates random tensors when no graph data | **FAKE DATA** |
| `_train_epoch()` | Real PyTorch training loop | REAL |
| `train_async()` | Runs training in thread | REAL |
| `_train_worker()` | Main training worker | REAL |

**Critical Finding:**
```python
# Lines 845-868: Early exit when disabled
if training_disabled:
    self.disabled = True
    self.model = None
    self.optimizer = None
    self.training_stats = {"disabled": True}
    return
```

---

### File: `cogniflex/learning/learning_manager.py`

**Class: `LearningManager`**
- **Status: ACTIVE** but mostly wrapper

**Key Methods:**
| Method | Purpose | Real/Fake |
|--------|---------|-----------|
| `train_model()` | Delegates to TrainingOrchestrator | REAL |
| `evaluate_model()` | Returns hardcoded metrics | **FAKE** |
| `get_model_status()` | Returns hardcoded status | **FAKE** |
| `stop_training()` | Always returns True | **FAKE** |

**Stub Methods:**
```python
def evaluate_model(self, model_id: str, data: Any, **kwargs):
    # Lines 192-196: FAKE METRICS
    metrics = {
        "accuracy": 0.0,
        "loss": 0.0,
        # Другие метрики...
    }

def get_model_status(self, model_id: str):
    # Lines 223-228: FAKE STATUS
    return {
        "model_id": model_id,
        "status": "ready",
        "progress": 1.0,
        "metrics": {}
    }
```

---

### File: `cogniflex/learning/self_analyzer.py`

**Class: `SelfAnalyzer`**
- Wrapper that coordinates: `AnalyzerCore`, `HealthMonitor`, `LearningOpportunityManager`, `PerformanceAnalyzer`, `MemoryGraphTrainer`

**Key Methods:**
| Method | Purpose | Real/Fake |
|--------|---------|-----------|
| `analyze_system()` | Coordinates health + performance + evolution analysis | REAL |
| `analyze_knowledge_gaps()` | Returns `{"status": "not_implemented"}` | **STUB** |
| `analyze_contradictions()` | Returns `{"status": "not_implemented"}` | **STUB** |
| `start_learning_process()` | Delegates to memory_graph_trainer | REAL (but disabled) |

---

### File: `cogniflex/learning/analyzer_core.py`

**Class: `AnalyzerCore`**
- SQLite-backed learning opportunity tracking
- **Status: ACTIVE**

**Key Methods:**
| Method | Purpose | Real/Fake |
|--------|---------|-----------|
| `get_learning_opportunities()` | SQL query to DB | REAL |
| `add_learning_opportunity()` | SQL INSERT | REAL |
| `start_background_analysis()` | Runs `_analysis_worker` thread | REAL |
| `_perform_periodic_analysis()` | Calls feedback/knowledge/performance analyzers | REAL |

---

### File: `cogniflex/learning/performance_analyzer.py`

**Class: `PerformanceAnalyzer`**
- Analyzes component performance
- **Status: ACTIVE**

**Key Methods:**
| Method | Purpose | Real/Fake |
|--------|---------|-----------|
| `analyze_performance()` | Iterates over components | REAL |
| `analyze_user_feedback()` | Uses adaptation_manager | REAL |

---

### File: `cogniflex/learning/integrated_learning_manager.py`

**Class: `IntegratedLearningManager`**
- Coordinates training across fractal_model_manager, hybrid_cache, knowledge_graph
- **Status: ACTIVE**

**Key Methods:**
| Method | Purpose | Real/Fake |
|--------|---------|-----------|
| `_train_from_document()` | Delegates to TrainingOrchestrator | REAL |
| `_train_knowledge_graph()` | Extracts nodes/edges, saves to fractal storage | REAL |
| `_fine_tune_model()` | Saves checkpoint (no actual training) | **STUB** |
| `_simple_document_training()` | Caches text, no training | **FAKE** |

**Training Loop: `_learning_worker()`**
- Pulls tasks from queue
- Processes: document_training, knowledge_graph_training, model_fine_tuning
- No actual model training implemented

---

### File: `cogniflex/learning/learning_processor.py`

**Class: `LearningProcessor`**
- Coordinates DataProcessor, LearningTaskGenerator, IntegrationManager
- **Status: ACTIVE** (but main task processing is stubbed)

**Critical Finding:**
```python
# Lines 251-275: SIMULATED LEARNING
def _simulate_learning_process(self, task: LearningTask) -> Optional[Dict[str, Any]]:
    """Симуляция процесса обучения (заглушка)"""
    time.sleep(0.1)  # Симуляция времени обработки
    
    return {
        'model_updates': {
            'weights_updated': True,
            'bias_adjusted': True,
            'learning_rate': 0.001
        },
        'performance_metrics': {
            'accuracy_improvement': 0.05,
            'loss_reduction': 0.1,
            'convergence_rate': 0.8
        }
    }
```

---

## PART 2: Old vs New Learning System

### OLD System: Legacy Components

#### `EnhancedSelfLearningSystem` (`cogniflex/core/enhanced_self_learning.py`)

**Status: DISABLED** by config (`training_disabled=True`)

**Key Methods:**
| Method | Purpose | Real/Fake |
|--------|---------|-----------|
| `_train_epoch()` | Simulates metrics with numpy | **FAKE** |
| `_validate_model()` | Simulates accuracy | **FAKE** |
| `_perform_training_session()` | Only runs if not disabled | REAL (but disabled) |
| `_collect_system_data()` | Collects from chat/memory/knowledge | REAL |
| `_save_model_with_entities()` | Calls model_manager.save_model() | REAL |

**Critical Finding (Line 501-524):**
```python
def _train_epoch(self, epoch_number: int, training_data: List[str]) -> EpochMetrics:
    # Имитация обучения (в реальной системе - вызов model.train())
    # Для демонстрации генерируем реалистичные метрики
    
    base_loss = 2.5 - (epoch_number * 0.3)  # Decreases with epochs
    noise = np.random.normal(0, 0.05)
    loss = max(0.5, base_loss + noise)
    
    accuracy = min(0.95, 0.4 + (epoch_number * 0.15) + np.random.normal(0, 0.02))
    perplexity = np.exp(loss)
    learning_rate = self.default_params['learning_rate'] * (0.9 ** (epoch_number - 1))
    duration = time.time() - start_time + np.random.uniform(5, 15)  # Имитация времени
```

---

#### `SelfLearningSystem` (`cogniflex/core/self_learning_system.py`)

**Status: LEGACY** (still imported in core_brain)

**Key Methods:**
| Method | Purpose | Real/Fake |
|--------|---------|-----------|
| `_train_model_on_data()` | Writes to temp file, sleeps 2 seconds | **FAKE** 
| `_expand_training_texts()` | Creates variations | REAL |
| `_save_updated_model()` | Logs only | **FAKE** |

**Critical Finding (Line 234-240):**
```python
# Имитируем процесс обучения
time.sleep(2)  # Имитация обучения

# Очищаем временный файл
if training_file.exists():
    training_file.unlink()

logger.info("Обучение модели завершено (имитация)")
```

---

#### `MemoryGraphML` (`cogniflex/core/memory_graph_ml.py`)

**Status: ACTIVE** but limited functionality

**Key Methods:**
| Method | Purpose | Real/Fake |
|--------|---------|-----------|
| `_compute_embeddings()` | Uses sentence-transformers or random | REAL with fallback |
| `_compute_fallback_embedding()` | Random deterministic embedding | **FAKE** |
| `_extract_patterns()` | Finds frequent paths | REAL |
| `add_insight()` | Tokenizes, computes embedding, stores | REAL |

---

### NEW System: `SelfDialogLearningSystem`

**Status: ACTIVE** (enabled in config)

**Advantages over OLD:**
1. No fake training loops - uses existing managers
2. Stores real insights to memory/knowledge_graph
3. Can verify facts via web search
4. Checks ethics via EthicsFramework
5. Checks contradictions via ContradictionManager

**Disadvantages:**
1. Still has fallback strings when model unavailable
2. No actual model fine-tuning
3. Analysis only, not learning in ML sense

---

## PART 3: Training Loops Analysis

### Training Loop 1: `SelfDialogLearningSystem._learning_loop()`

**File:** `cogniflex/learning/self_dialog_learning.py` (lines 180-195)

**When Runs:**
- On `start()` if `enabled=True` in config
- Interval: `cycle_interval` (default 300 seconds)

**What It Does:**
1. Calls `run_self_dialog_cycle()` which:
   - Generates question via model
   - Generates response via model
   - Checks ethics
   - Detects contradictions
   - Verifies facts via web search
   - Analyzes quality
   - Stores insights

**Config Check:**
```python
self.enabled = cfg.get("enabled", True)  # Default True
```

**Disabled by:** Nothing (enabled by default)

**Fake/Synthetic Data:** NO - uses real managers

---

### Training Loop 2: `EnhancedSelfLearningSystem._learning_loop()`

**File:** `cogniflex/core/enhanced_self_learning.py` (lines 369-389)

**When Runs:**
- On `start()` if `training_disabled=False`
- Interval: 60 seconds check

**What It Does:**
1. Calls `auto_collect_training_data()`
2. Calls `_perform_training_session()`
3. Each epoch calls `_train_epoch()` - **FAKE METRICS**

**Config Check:**
```python
if self.config.get('training_disabled', True):  # Default True (DISABLED)
    logger.info("EnhancedSelfLearningSystem отключена через конфигурацию")
    return True
```

**Disabled by:** `training_disabled=True` in brain_config.json

**Fake/Synthetic Data:** YES - `_train_epoch()` generates fake loss/accuracy

---

### Training Loop 3: `MemoryGraphTrainer._train_worker()`

**File:** `cogniflex/learning/memory_graph_trainer.py` (lines 529-597)

**When Runs:**
- On `train_async()` call
- Default: 10 epochs

**What It Does:**
1. Calls `prepare_training_data()` which may return synthetic data
2. Loops through epochs calling `_train_epoch()`
3. Saves best model checkpoint

**Config Check:**
```python
if training_disabled:
    self.disabled = True
    self.model = None
    return
```

**Disabled by:** 
- `learning.enable_training: false`
- `learning.training_disabled: true`
- `system.disable_learning_threads: true`
- `system.disable_background_training: true`

**Fake/Synthetic Data:** YES - if knowledge_graph unavailable, creates synthetic random tensors

---

### Training Loop 4: `IntegratedLearningManager._learning_worker()`

**File:** `cogniflex/learning/integrated_learning_manager.py` (lines 190-209)

**When Runs:**
- On `start_learning()` call

**What It Does:**
1. Pulls tasks from queue
2. Calls `_process_learning_task()`:
   - `_train_from_document()` - delegates to TrainingOrchestrator
   - `_train_knowledge_graph()` - saves data to storage
   - `_fine_tune_model()` - **SAVES CHECKPOINT ONLY, NO TRAINING**

**Config Check:** None (always runs when called)

**Fake/Synthetic Data:** NO

---

### Training Loop 5: `LearningProcessor._main_processing_loop()`

**File:** `cogniflex/learning/learning_processor.py` (lines 150-184)

**When Runs:**
- On `initialize()`
- Interval: `cycle_interval` (default 300 seconds)

**What It Does:**
1. Calls `_execute_learning_cycle()`
2. Calls `_process_tasks()` which calls `_simulate_learning_process()` - **FAKE**

**Config Check:** None

**Fake/Synthetic Data:** YES - `_simulate_learning_process()` returns fake metrics

---

### Training Loop 6: `TrainingOrchestrator.train_from_document()`

**File:** `cogniflex/mlearning/training_orchestrator.py` (lines 654-974)

**When Runs:**
- On manual training request
- Can resume from checkpoint

**What It Does:**
1. Tokenizes document segments
2. Processes in batches
3. Updates knowledge graph with extracted entities
4. Supports fractal training path

**Config Check:**
```python
if not self._can_train_now():
    return {"status": "deferred", "reason": "models_not_ready_or_cache_unavailable"}
```

**Fake/Synthetic Data:** NO - real tokenization and knowledge extraction

---

## PART 4: Stub Implementations Summary

### Methods Returning "not_implemented"

| File | Method | Return |
|------|--------|--------|
| `self_analyzer.py` | `analyze_knowledge_gaps()` | `{"status": "not_implemented"}` |
| `self_analyzer.py` | `analyze_contradictions()` | `{"status": "not_implemented"}` |

### Methods Returning Fake/Hardcoded Data

| File | Method | Fake Data |
|------|--------|-----------|
| `learning_manager.py` | `evaluate_model()` | `accuracy: 0.0, loss: 0.0` |
| `learning_manager.py` | `get_model_status()` | `status: "ready", progress: 1.0` |
| `learning_manager.py` | `stop_training()` | Always `True` |
| `enhanced_self_learning.py` | `_train_epoch()` | Fake loss/accuracy with numpy |
| `enhanced_self_learning.py` | `_validate_model()` | Fake accuracy ~0.7-0.98 |
| `self_learning_system.py` | `_train_model_on_data()` | `time.sleep(2)` then delete file |
| `self_learning_system.py` | `_save_updated_model()` | Logs only, no save |
| `learning_processor.py` | `_simulate_learning_process()` | Fake weights/metrics |

### Methods with Synthetic Fallback Data

| File | Method | Fallback Behavior |
|------|--------|-------------------|
| `memory_graph_trainer.py` | `prepare_training_data()` | Creates 100 nodes, 200 edges of random tensors if no graph |
| `memory_graph_ml.py` | `_compute_fallback_embedding()` | Deterministic random based on hash |
| `self_dialog_learning.py` | `generate_question()` | Returns `"What is your understanding of {topic}?"` |
| `self_dialog_learning.py` | `generate_response()` | Returns `"I need to develop my understanding..."` |

---

## SUMMARY: Real vs Fake

### ACTIVE (Real Components)
1. **SelfDialogLearningSystem** - Analysis only, no model training
2. **AnalyzerCore** - SQLite storage for learning opportunities
3. **PerformanceAnalyzer** - Real component iteration
4. **TrainingOrchestrator** - Real document processing and knowledge extraction
5. **MemoryGraphML** - Real embeddings (with sentence-transformers fallback)

### DISABLED (Not Running)
1. **EnhancedSelfLearningSystem** - training_disabled=True
2. **MemoryGraphTrainer** - training_disabled=True, enable_training=False

### FAKE/STUB (Don't Do Real ML)
1. **LearningManager** - evaluate_model, get_model_status, stop_training
2. **EnhancedSelfLearningSystem._train_epoch()** - Fake loss/accuracy
3. **SelfLearningSystem** - Sleep 2 seconds, delete file
4. **LearningProcessor._simulate_learning_process()** - Returns hardcoded metrics

### Configuration Status (brain_config.json)
```json
{
  "learning": {
    "enable_training": false,        // DISABLED
    "training_disabled": true,      // DISABLED
    "disable_background_training": true  // DISABLED
  },
  "self_dialog_learning": {
    "enabled": true                 // ACTIVE
  },
  "system": {
    "disable_learning_threads": true,   // DISABLED
    "disable_background_training": true  // DISABLED
  }
}
`
