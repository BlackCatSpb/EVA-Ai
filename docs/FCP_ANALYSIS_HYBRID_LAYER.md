# FCP Analysis: HybridLayerProcessor and HybridLayerManager

**Дата:** 2026-05-02  

---

## Executive Summary

Данный отчёт анализирует проблемы интеграции компонентов HybridLayerProcessor и HybridLayerManager в FCP Pipeline EVA-Ai. Выявлены критические проблемы изоляции компонентов и отсутствия передачи real hidden states от OpenVINO.

---

## 1. Как работает каждый компонент

### 1.1 HybridLayerProcessor (hybrid_integration.py:332-509)

**Назначение:** Обработчик гибридного слоя, интегрирующий LLM + GNN + KCA + SRG + LoRA.

**Компоненты:**
- KCA - Knowledge Conscious Attention (fcp_core/__init__.py:104)
- SRG - Semantic Relevance Gate (fcp_core/__init__.py:199)
- GNN Encoder - FractalGraphEncoderLocal (локальная numpy версия)
- Injectors - TextFusionInjectorLocal, AdaptiveFusionInjectorLocal

**Основной метод - process():**
`python
def process(self, query_text, hidden_states, knowledge_nodes):
    # 1. Build GNN graph from knowledge nodes
    # 2. Retrieve subgraph via HNSW
    # 3. SRG evaluation - mode (direct/reasoning/variational)
    # 4. KCA correction (if reasoning mode)
    # 5. Text injection for prompt enrichment
    # Return: (enriched_prompt, corrected_hidden_states, metadata)
`

**Ключевая проблема:** Метод требует hidden_states как обязательный параметр, но в FCPPipelineV15 эти states никогда не передаются от OpenVINO.

---

### 1.2 HybridLayerManager (hybrid_integration.py:584-655)

**Назначение:** Менеджер гибридных слоёв для всей модели - управление состоянием и инъекциями на каждом слое.

**Архитектура:**
- processors: Dict[int, HybridLayerProcessor] - один процессор на слой
- global_graph_encoder - шарится между слоями
- _state_cache - кеш состояний

**Проблема:** Не инициализируется как отдельный компонент в системе EVA. Создаётся только внутри FCPPipelineV15.

---

### 1.3 FractalGraphEncoderLocal vs FractalGraphEncoder

| Характеристика | FractalGraphEncoderLocal | FractalGraphEncoder |
|---------------|------------------------|---------------------|
| Framework | NumPy | PyTorch |
| GNN Layers | Ручные матричные операции | SAGEConv (torch-geometric) |
| LoRA | Встроен | Separate adapter |
| Обучение | Нет (random init) | Да (checkpoint) |
| Использование | HybridLayerProcessor | - |

---

## 2. Почему компоненты изолированы друг от друга

### 2.1 HybridLayerProcessor не в COMPONENT_LIST

COMPONENT_LIST (init_core.py:44-67) НЕ содержит:
- hybrid_layer_processor
- hybrid_layer_manager

Компоненты создаются внутри FCPPipelineV15._init_hybrid_layers(), изолированы от EventBus.

### 2.2 HybridLayerManager не в COMPONENT_LIST

Также создаётся внутри FCPPipelineV15 (строка 221), не имеет доступа к системным сервисам.

---

## 3. Связь с OpenVINO Pipeline

### 3.1 Текущий поток данных

`
User Query
    - _build_prompt() - chat_prompt
    - _process_with_hybrid_layers() (ДО генерации!)
        - _get_query_embedding() - random/synthetic
        - srg.evaluate() - mode (logits=zeros(100))
        - kca.forward() - synthetic states
    - _generate() - OpenVINO (NO hidden states available!)
    - _save_layer_snapshot() - после генерации
`

**Критическая проблема:** _process_with_hybrid_layers() вызывается ДО генерации!

### 3.2 Почему hidden states не передаются

1. _process_with_hybrid_layers вызывается до _generate()
2. OpenVINO LLMPipeline.generate() НЕ возвращает hidden_states
3. HybridLayerProcessor.process() получает только текст, не states
4. KCA работает с synthetic states:
   `python
   initial_states = np.tile(query_embedding, (seq_len, 1)).astype(np.float32)
   `

---

## 4. Проблемы с graph_encoder.pt

### 4.1 Где требуется

fcp_pipeline.py:224-238 - _init_hybrid_layers()

### 4.2 Текущий статус

- Файл НЕ существует - поиск по EVA-Ai/models/*graph_encoder*.pt - None
- GNN Encoder работает с random init - np.random.seed(42) в FractalGraphEncoderLocal

### 4.3 Как получить graph_encoder.pt

1. Запустить FCP/colab_upload/train_gnn.ipynb
2. Dataset: FCP/colab_upload/lora_dataset.json (146 примеров)
3. Output: graph_encoder.pt

---

## 5. Что нужно исправить для полной интеграции

### 5.1 Критические исправления

#### 5.1.1 Передача real hidden states

Два варианта:

**Вариант A (рекомендуемый):** Обработка после генерации
- Минимальная обработка перед генерацией (только retrieval)
- Генерация через OpenVINO
- Получение hidden states через LayerCaptureModel или StateInjector
- Обработка после генерации (KCA + SRG)

**Вариант B:** Интеграция LayerCaptureModel
- Создать адаптер между OpenVINO и HybridLayerProcessor
- Использовать существующую архитектуру LayerCaptureModel

#### 5.1.2 Добавить в COMPONENT_LIST

init_core.py:44-67:
- hybrid_layer_processor
- hybrid_layer_manager

#### 5.1.3 Интеграция с EventBus

HybridLayerProcessor должен подписываться на:
- pipeline.complete - для обработки после генерации
- memory.graph_updated - для обновления локального графа
- system.state_changed - для синхронизации состояния

### 5.2 Высокие исправления

#### 5.2.1 Исправить SRG evaluation

Текущий код (НЕПРАВИЛЬНО):
`python
logits=np.zeros(100)  # ЗАГЛУШКА!
`

Исправить: после генерации получаем logits из модели.

#### 5.2.2 Исправить KCA input

Текущий код работает с synthetic states. Исправить: передавать real_hidden_states.

#### 5.2.3 Исправить _get_query_embedding

Текущий код использует input_ids (не embeddings) или случайный вектор.
Исправить: использовать sentence-transformers или усреднение эмбеддингов токенизатора.

### 5.3 Средние исправления

#### 5.3.1 Обучить GNN encoder

1. Запустить FCP/colab_upload/train_gnn.ipynb
2. Скачать graph_encoder.pt
3. Поместить в C:\Users\black\OneDrive\Desktop\EVA-Ai\models\graph_encoder.pt

---

## 6. Выводы

1. **HybridLayerProcessor создаётся, но изолирован** - нет связи с EventBus, создаётся вне COMPONENT_LIST

2. **Hidden states не передаются** - _process_with_hybrid_layers вызывается до генерации, работает с synthetic states

3. **HybridLayerManager не инициализируется как компонент** - создаётся внутри FCPPipelineV15, изолирован от системы

4. **FractalGraphEncoderLocal vs FractalGraphEncoder** - дублирование функциональности, разные реализации

5. **graph_encoder.pt не существует** - GNN работает с random init, качество низкое

---

## 7. Следующие шаги

1. Добавить hybrid_layer_processor в COMPONENT_LIST - для интеграции в систему
2. Модифицировать generate() - для обработки после генерации с real hidden states
3. Обучить GNN encoder - создать graph_encoder.pt
4. Исправить SRG/KCA inputs - передавать реальные данные
5. Интегрировать LayerCaptureModel - для захвата состояний слоёв
