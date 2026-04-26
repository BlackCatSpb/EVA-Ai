# EVA-Ai - План интеграции FCP компонентов

**Дата:** 2026-04-26
**Версия:** 1.0
**Статус:** В работе

---

## Сводка

После анализа трёх проектов (EVA-Ai, FCP, FMF_EVA) выявлено **18 отсутствующих компонентов** из FCP, которые необходимы для полноценной работы гибридной архитектуры.

### Распределение по приоритетам

| Приоритет | Количество | Описание |
|-----------|-----------|----------|
| 🔴 Критический | 7 | Без них FCP не работает |
| 🔴 Высокий | 2 | Основа для интеграции |
| 🟡 Средний | 5 | Важные компоненты |
| 🟢 Низкий | 4 | Дополнительные улучшения |

---

## 🔴 КРИТИЧЕСКИЕ (Этап 1)

### 1. fcp_core/types.py
**Назначение:** Базовые типы данных для всей системы FCP.

**Содержит:**
```python
@dataclass Subgraph          # Извлечённый подграф
@dataclass MemorySegment      # Сегмент в TCM
@dataclass Concept            # Концепт для графа
@dataclass Fact              # Факт (триплет)
@dataclass Contradiction      # Обнаруженное противоречие
@dataclass ResolutionResult   # Результат разрешения
```

**Что сделать:**
1. Создать файл `eva_ai/fcp_core/types.py`
2. Скопировать все dataclass из `FCP/src/fcp_core/types.py`
3. Адаптировать импорты для EVA

**Файл-источник:** `C:\Users\black\OneDrive\Desktop\FCP\src\fcp_core\types.py`

---

### 2. fcp_core/config.py
**Назначение:** Модульная конфигурация FCP - параметры модели, слоёв, TCM, LoRA.

**Содержит:**
```python
@dataclass FCPConfig:
    - vocab_size, embedding_dim, num_layers
    - num_heads, kv_heads, head_dim, intermediate_size
    - graph_retrieval_k, master_tokens, gnn_iterations
    - stop_threshold, early_exit_threshold
    - tcm_max_segments, tcm_top_k
    - lora_rank_base, lora_rank_domain, lora_rank_reasoning
    - model_path, graph_db_path
    - device, num_threads, performance_hint
```

**Что сделать:**
1. Создать файл `eva_ai/fcp_core/config.py`
2. Скопировать FCPConfig из `FCP/src/fcp_core/config.py`
3. Добавить методы `from_model()` и `minimal()`
4. Интегрировать с существующей EVA конфигурацией (brain_config.json)

**Файл-источник:** `C:\Users\black\OneDrive\Desktop\FCP\src\fcp_core\config.py`

---

### 3. fcp_core/hybrid_layer.py
**Назначение:** Гибридный слой FCP с 5 этапами обработки (SPEC section 3.2).

**Содержит:**
```python
class FractalGatedHybridLayer:
    # Этап 1: extract_subgraph() - k-NN поиск в HNSW
    # Этап 2: node_aware_routing() - определение LLM vs GNN
    # Этап 3: transformer_block() - attention + FFN
    # Этап 4: activation_gate() - halt probability
    # Этап 5: stream_merging() - cross-attention или gated add
```

**Что сделать:**
1. Создать файл `eva_ai/fcp_core/hybrid_layer.py`
2. Скопировать FractalGatedHybridLayer из `FCP/src/fcp_core/hybrid_layer.py`
3. Адаптировать для работы с EVA FractalGraphV2
4. Добавить импорт KCA и SRG из EVA fcp_core
5. Добавить логирование через EVA logger

**Связи:**
- Использует: `fcp_core/types.py` (Subgraph, LayerState, HaltDecision)
- Использует: `fcp_gnn/graph_encoder.py` (для извлечения подграфа)
- Использует: EVA KCA, SRG

**Файл-источник:** `C:\Users\black\OneDrive\Desktop\FCP\src\fcp_core\hybrid_layer.py` (646 строк)

---

### 4. fcp_core/hybrid_stack.py
**Назначение:** Модульный стек гибридных слоёв с динамическим управлением.

**Содержит:**
```python
@dataclass StackConfig
class HybridStack:
    - _build_layers() - создание слоёв
    - add_layers() / remove_layers() - динамическое управление
    - process() - forward pass через все слои
    - get_statistics() - статистика early exits
```

**Что сделать:**
1. Создать файл `eva_ai/fcp_core/hybrid_stack.py`
2. Скопировать HybridStack и StackConfig из `FCP/src/fcp_core/hybrid_stack.py`
3. Интегрировать с FractalGatedHybridLayer
4. Добавить поддержку EVA event_bus для событий слоёв

**Связи:**
- Использует: `fcp_core/hybrid_layer.py`
- Использует: `fcp_core/config.py`

**Файл-источник:** `C:\Users\black\OneDrive\Desktop\FCP\src\fcp_core\hybrid_stack.py` (203 строки)

---

### 5. fcp_gnn/graph_encoder.py
**Назначение:** GNN энкодер с SAGEConv и HNSW для извлечения graph_vec.

**Содержит:**
```python
class FractalGraphEncoder(nn.Module):
    # SAGEConv слои: conv1 (input→hidden), conv2 (hidden→output)
    # forward() - извлечение graph_vec и gate_weights
    # build_hnsw_index() - построение HNSW
    # retrieve_subgraph() - k-NN поиск
    # encode_node() - кодирование одного узла
```

**Что сделать:**
1. Создать файл `eva_ai/fcp_gnn/graph_encoder.py`
2. Скопировать FractalGraphEncoder из `FCP/src/fcp_gnn/graph_encoder.py`
3. Добавить поддержку EVA FractalGraphV2 (sqlite) вместо PyG Data
4. Подготовить для конвертации в ONNX (для обучения в Colab)

**Важно:** Этот компонент требует обучения в Colab. После обучения нужно:
- Конвертировать в ONNX через `convert_gnn_to_ov.py`
- Сохранить в `EVA-Ai/models/`

**Файл-источник:** `C:\Users\black\OneDrive\Desktop\FCP\src\fcp_gnn\graph_encoder.py` (346 строк)

---

### 6. fcp_gnn/hybrid_transformer_layer.py
**Назначение:** Гибридный трансформер-слой с GNN injection на всех 32 слоях.

**Содержит:**
```python
class HybridTransformerLayer:
    # Полная гибридизация: LLM + GNN + LoRA + KCA + SRG
    # inject_gnn() - инъекция graph_vec
    # forward() - combined forward
```

**Что сделать:**
1. Создать файл `eva_ai/fcp_gnn/hybrid_transformer_layer.py`
2. Скопировать из `FCP/src/fcp_gnn/hybrid_transformer_layer.py`
3. Интегрировать с EVA AdaptiveFusionInjector (уже есть в `fcp_gnn/hybrid_integration.py`)

**Связи:**
- Использует: `fcp_gnn/injector.py` (AdaptiveFusionInjector)
- Использует: `fcp_gnn/graph_encoder.py` (FractalGraphEncoder)
- Использует: EVA LoRA (ada_lora.py)

**Файл-источник:** `C:\Users\black\OneDrive\Desktop\FCP\src\fcp_gnn\hybrid_transformer_layer.py`

---

### 7. fcp_gnn/convert_gnn_to_ov.py
**Назначение:** Конвертация PyTorch GNN → ONNX → OpenVINO.

**Содержит:**
```python
# PyTorch → ONNX конвертация
# ONNX → OpenVINO оптимизация
# Сохранение в models/
```

**Что сделать:**
1. Создать файл `eva_ai/fcp_gnn/convert_gnn_to_ov.py`
2. Скопировать из `FCP/src/fcp_gnn/convert_gnn_to_ov.py`
3. Добавить параметры для EVA (output_path, device)

**Использование:**
```bash
# После обучения GNN в Colab
python -m eva_ai.fcp_gnn.convert_gnn_to_ov --input model.pt --output models/
```

**Файл-источник:** `C:\Users\black\OneDrive\Desktop\FCP\src\fcp_gnn\convert_gnn_to_ov.py`

---

## 🔴 ВЫСОКИЙ ПРИОРИТЕТ (Этап 2)

### 8. fcp_core/input_layer.py
**Назначение:** Входной слой - токенизация, embeddings, RoPE.

**Что сделать:**
1. Создать файл `eva_ai/fcp_core/input_layer.py`
2. Скопировать InputLayer из `FCP/src/fcp_core/input_layer.py`
3. Интегрировать с EVA токенизатором (openvino_tokenizer.bin)

---

### 9. fcp_core/output_layer.py
**Назначение:** Выходной слой - RMS-norm, lm_head, sampling.

**Что сделать:**
1. Создать файл `eva_ai/fcp_core/output_layer.py`
2. Скопировать OutputLayer из `FCP/src/fcp_core/output_layer.py`
3. Интегрировать с EVA дектренизатором

---

## 🟡 СРЕДНИЙ ПРИОРИТЕТ (Этап 3)

### 10. fcp_knowledge/graph_curator.py
**Назначение:** Управление жизненным циклом узлов графа.

**Что сделать:**
1. Создать файл `eva_ai/fcp_knowledge/graph_curator.py`
2. Скопировать из `FCP/src/fcp_knowledge/graph_curator.py`
3. Адаптировать для EVA FractalGraphV2

**Примечание:** В EVA уже есть类似功能 в `eva_ai/knowledge/` и `eva_ai/curation/`

---

### 11. fcp_knowledge/learning_manager.py
**Назначение:** Менеджер обучения LoRA + GNN совместно.

**Что сделать:**
1. Создать файл `eva_ai/fcp_knowledge/learning_manager.py`
2. Скопировать из `FCP/src/fcp_knowledge/learning_manager.py`
3. Интегрировать с EVA SelfDialogLearning

---

### 12. fcp_tools/scenario_tcm.py
**Назначение:** Scenario-based TCM для эпизодической памяти.

**Что сделать:**
1. Скопировать в `eva_ai/tools/fcp/scenario_tcm.py`
2. Адаптировать для EVA

---

### 13. fcp_tools/semantic_cache_evictor.py
**Назначение:** Вытеснение из семантического кеша.

**Что сделать:**
1. Скопировать в `eva_ai/tools/fcp/semantic_cache_evictor.py`
2. Интегрировать с EVA memory system

---

### 14. fcp_core/adaptive_lora.py
**Назначение:** AdaLoRA слой с динамическим рангом.

**Статус:** Уже есть в `fcp_migration/fcp_core/adaptive_lora.py`

**Что сделать:**
1. Проверить работоспособность
2. Перенести в `eva_ai/fcp_core/adaptive_lora.py`
3. Обновить импорты в EVA

---

## 🟢 НИЗКИЙ ПРИОРИТЕТ (Этап 4)

### 15-18. Обновить EVA fcp_core __init__.py
**Назначение:** Экспортировать все новые компоненты.

**Что сделать:**
1. Обновить `eva_ai/fcp_core/__init__.py`:
   ```python
   from eva_ai.fcp_core.types import (
       Subgraph, MemorySegment, Concept, Fact, 
       Contradiction, ResolutionResult
   )
   from eva_ai.fcp_core.config import FCPConfig, StackConfig
   from eva_ai.fcp_core.hybrid_layer import FractalGatedHybridLayer
   from eva_ai.fcp_core.hybrid_stack import HybridStack
   from eva_ai.fcp_core.input_layer import InputLayer
   from eva_ai.fcp_core.output_layer import OutputLayer
   ```

2. Обновить `eva_ai/fcp_gnn/__init__.py`

3. Обновить `eva_ai/fcp_knowledge/__init__.py` (создать папку)

4. Обновить `eva_ai/tools/fcp/__init__.py`

---

## 📋 Чеклист выполнения

### Этап 1: Критические ✅
- [x] 1. fcp_core/types.py
- [x] 2. fcp_core/config.py
- [x] 3. fcp_core/hybrid_layer.py
- [x] 4. fcp_core/hybrid_stack.py
- [x] 5. fcp_gnn/graph_encoder.py
- [x] 6. fcp_gnn/hybrid_transformer_layer.py
- [x] 7. fcp_gnn/convert_gnn_to_ov.py

### Этап 2: Высокий ✅
- [x] 8. fcp_core/input_layer.py
- [x] 9. fcp_core/output_layer.py

### Этап 3: Средний ✅
- [x] 10. fcp_knowledge/graph_curator.py
- [x] 11. fcp_knowledge/learning_manager.py
- [x] 12. fcp_tools/scenario_tcm.py
- [x] 13. fcp_tools/semantic_cache_evictor.py
- [x] 14. fcp_core/adaptive_lora.py (скопирован)

### Этап 4: Низкий ✅
- [x] 15-18. Обновить все __init__.py

---

## 🔗 Зависимости между компонентами

```
config.py
    ↓
types.py ← ← ← ← ← ← ← ←
    ↓                        │
hybrid_layer.py             │
    ↓                        │
hybrid_stack.py ← ← ← ← ← ←
    ↓
input_layer.py
    ↓
output_layer.py
    ↓
graph_encoder.py
    ↓
hybrid_transformer_layer.py
    ↓
convert_gnn_to_ov.py
```

---

## 📝 Примечания

1. **Colab для GNN:** После создания graph_encoder.py и convert_gnn_to_ov.py:
   - Обучить GNN в Colab (данные в `FCP/data/gnn_training_data.json`)
   - Конвертировать в ONNX
   - Скопировать в `EVA-Ai/models/graph_encoder.onnx`

2. **Существующие EVA компоненты:**
   - KCA, SRG, ConvergenceController - уже есть в `fcp_core/fractal_graph.py`
   - FractalGraphV2 - уже есть в `fcp_core/fractal_graph.py`
   - AdaptiveFusionInjector - уже есть в `fcp_gnn/hybrid_integration.py`
   - FractalGraphEncoderLocal - уже есть в `fcp_gnn/hybrid_integration.py`

3. **Интеграция с EVA:**
   - Новые компоненты должны использовать EVA logger
   - Использовать EVA event_bus для событий
   - Интегрировать с существующим brain_query.py

---

## 🟣 ЭТАП 5: ИНТЕГРАЦИЯ С CORE BRAIN (Новый)

### 19. Обновить brain_config.json
**Назначение:** Единая конфигурация для всех FCP компонентов.

**Что сделать:**
1. Добавить секцию `fcp` с путями к модулям
2. Указать настройки для HybridStack
3. Конфликтовать нечего - будет единый источник истины

```json
"fcp": {
    "enabled": true,
    "hybrid_stack": {
        "num_layers": 32,
        "hidden_dim": 2560,
        "use_gnn": true,
        "use_lora": true,
        "use_kca": true,
        "use_srg": true
    },
"memory_snapshot": {
        "enabled": true,
        "snapshot_interval": 1,
        "save_to_graph": true,
        "snapshot_all_layers": true,
        "num_layers": 32
      },
    "modules": {
        "graph_encoder": "models/graph_encoder.onnx",
        "gnn_runtime": "models/graph_encoder.xml"
    }
}
```

---

### 20. Интегрировать HybridStack в init_factories.py
**Назначение:** Подключить новые модули к CoreBrain.

**Что сделать:**
1. Добавить функцию `create_fcp_hybrid_stack()` в `init_factories.py`
2. Подключить к `core_brain.components['hybrid_stack']`
3. Убрать дублирование - использовать наш `HybridStack`

**Файл:** `eva_ai/core/init_factories.py`

---

### 21. Создать MemorySnapshotIntegration
**Назначение:** Паттерн сохранения состояний LLM слоёв в граф.

**Concept:**
```
LLM слой N → [snapshot] → FractalGraphV2 (как узел)
                              ↓
                        retrieval
                              ↓
LLM слой N → [коррекция] ← GNN + KCA
```

**Что сделать:**
1. Создать `eva_ai/core/memory_snapshot_integration.py`
2. Реализовать сохранение hidden_states как узлов графа
3. Связать с ConceptMiner/ContradictionMiner
4. Инъекция коррекции обратно в генерацию

```python
class MemorySnapshotIntegration:
    def __init__(self, brain, fractal_graph, gnn_encoder):
        self.brain = brain
        self.graph = fractal_graph
        self.gnn = gnn_encoder
        
    def on_layer_output(self, layer_idx, hidden_states):
        # 1. Сохранить состояние как узел
        state_node = self.save_layer_state(layer_idx, hidden_states)
        
        # 2. Извлечь концепты и противоречия
        concepts = self.brain.concept_miner.extract_from_state(state_node)
        contradictions = self.brain.contradiction_miner.detect_from_state(state_node)
        
        # 3. Сформировать коррекцию
        correction = self.compute_correction(concepts, contradictions)
        
        return correction
    
    def save_layer_state(self, layer_idx, hidden_states):
        # Сохранить эмбеддинг слоя как узел графа
        pass
```

---

### 22. Обновить FCPPipelineV15
**Назначение:** Использовать HybridStack вместо текущего подхода.

**Что сделать:**
1. Заменить `HybridLayerProcessor` на наш `HybridStack`
2. Подключить `MemorySnapshotIntegration`
3. Убрать дублирование кода

---

### 23. Связать ConceptMiner с потоком состояний
**Назначение:** Концепты формируются из snapshot'ов, а не только из текста.

**Concept:**
- Layer snapshot → Concept → Contradiction → Graph
- Graph → Retrieval → Layer correction → Generation

**Архитектура Layer Interception:**

Для перехвата `hidden_states` на всех 32 слоях есть 2 варианта:

#### Вариант A: LayerCaptureModel (transformers)
- Использует `transformers.AutoModel` с `output_hidden_states=True`
- Даёт доступ к `hidden_states` после каждого слоя
- Требует PyTorch (~8GB RAM для Qwen-4B)
- Медленнее чем OpenVINO

#### Вариант B: OpenVINO Model Export (per-layer outputs)
- Экспорт модели в ONNX/OpenVINO с exposed layer outputs
- Требует реэкспорта модели
- Быстрый инференс, но сложная подготовка

**Текущий статус:**
- `eva_ai/core/layer_capture_model.py` - создан LayerCaptureModel
- `eva_ai/core/layerwise_openvino_model.py` - заготовка для низкоуровневого OpenVINO
- FCPPipelineV15 использует high-level OpenVINO API (НЕ поддерживает layer interception)

**Для полного решения:**
1. Добавить `LayerCaptureModel` в `brain_components.py`
2. Создать режим "layer_capture" в `brain_config.json`
3. Интегрировать с `HybridLayerBridge`

---

### 24. Убрать дублирование FractalGraphEncoder
**Статус:** НЕ ТРЕБУЕТСЯ - разные назначения

**Анализ:**
- `FractalGraphEncoderLocal` в `hybrid_integration.py` - numpy-only runtime (для продакшена)
- `GraphEncoderRuntime` в `graph_encoder.py` - numpy-only fallback (то же самое)
- `FractalGraphEncoder` в `graph_encoder.py` - PyTorch версия для обучения в Colab

**Вывод:**
- `FractalGraphEncoderLocal` используется HybridLayerProcessor/Manager (EVA runtime)
- `GraphEncoderRuntime` - то же самое, можно использовать как замену
- Оставим как есть, чтобы не сломать текущую систему

---

## 📋 Чеклист выполнения Этапа 5

- [x] 19. Обновить brain_config.json
- [x] 20. Интегрировать HybridStack в init_factories.py
- [x] 21. Создать MemorySnapshotIntegration
- [ ] 22. Обновить FCPPipelineV15 (architecture gap - см. секцию 23)
- [x] 23. Связать ConceptMiner с потоком состояний (ДОКУМЕНТИРОВАНО)
- [x] 24. Убрать дублирование FractalGraphEncoder (НЕ ТРЕБУЕТСЯ)

---

## 🔗 Архитектура после интеграции

```
┌─────────────────────────────────────────────────────────────┐
│                        CoreBrain                            │
├─────────────────────────────────────────────────────────────┤
│  EventBus ← все компоненты публикуют/подписываются          │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐ │
│  │ConceptMiner│    │Contradiction │    │ SelfDialog    │ │
│  │            │    │Miner         │    │ Learning      │ │
│  └──────┬─────┘    └──────┬───────┘    └───────┬───────┘ │
│         │                  │                    │          │
│         └──────────────────┼────────────────────┘          │
│                            ↓                               │
│                   ┌────────────────┐                       │
│                   │ FractalGraphV2 │ (sqlite + HNSW)       │
│                   └────────┬───────┘                       │
│                            ↓                               │
│         ┌─────────────────────────────────┐               │
│         │    HybridStack (32 слоя)        │               │
│         │  ┌─────────────────────────┐    │               │
│         │  │ FractalGatedHybridLayer│    │               │
│         │  │ 1. extract_subgraph    │    │               │
│         │  │ 2. node_aware_routing  │    │               │
│         │  │ 3. transformer_block    │    │               │
│         │  │ 4. activation_gate     │    │               │
│         │  │ 5. stream_merging     │    │               │
│         │  └─────────────────────────┘    │               │
│         │  MemorySnapshot на КАЖДОМ слое  │               │
│         └─────────────────────────────────┘               │
│                            ↓                               │
│                   ┌────────────────┐                       │
│                   │  GNN Encoder   │ (GraphEncoderRuntime)│
│                   └────────────────┘                       │
│                            ↓                               │
│                   ┌────────────────┐                       │
│                   │   KCA/SRG      │ (коррекция)          │
│                   └────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

---

**Следующий шаг:** Начать с пункта 19 (обновить brain_config.json)
