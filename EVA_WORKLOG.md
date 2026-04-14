# EVA AI - План исправлений

## Статус: В РАБОТЕ

## Логика архитектуры
```
memory/ → reasoning/ → generation/ → core brain
     ↓           ↓            ↓           ↓
   fractal    entity      pipeline    event_bus
  graph_v2   extractor   coordinator   model_access
```

## Приоритет 1: Core Brain и EventBus

### 1.1 EventBus - Исправление подписки обработчиков
- **Проблема**: `_handle_component_error()` missing 1 required positional argument: 'event'
- **Статус**: ✅ ГОТОВО
- **Файл**: `eva_ai/core/event_bus.py`
- **Изменения**: Исправлены 3-элементные кортежи (subscription_id, weak_handler, priority)

### 1.2 MemoryManager - Отсутствующие методы
- **Проблема**: `'MemoryManager' object has no attribute 'clear_cache'` и `'optimize'`
- **Статус**: ✅ ГОТОВО
- **Файл**: `eva_ai/memory/manager_core.py`
- **Решение**: Добавлена проверка `hasattr` перед вызовом методов в `_deferred_optimize` и `_deferred_cleanup`

### 1.3 Signal Handler Signatures - EventBus subscribers
- **Проблема**: `EventSubscriptionMixin._on_component_ready()` missing 1 required positional argument: 'event'
- **Статус**: ✅ ГОТОВО
- **Файлы**: 
  - `eva_ai/core/brain_coordination.py` - `_on_component_ready` (event=None → event)
  - `eva_ai/core/system_state.py` - `_handle_component_initialized`, `_handle_component_started`, `_handle_component_stopped` (event=None → event)
  - `eva_ai/core/event_bus_bridge.py` - `_on_new_event` (event=None → event)

---

## Приоритет 2: Model & Pipeline

### 2.1 Model A File Not Found
- **Проблема**: `Model A file not found: .../qwen2.5-3b-instruct-q4_k_m.gguf`
- **Статус**: ⏳ В ОЧЕРЕДИ
- **Файл**: `eva_ai/core/brain_query.py` - исправить путь к Model A

### 2.2 Sentence Transformers Loading
- **Проблема**: `Invalid argument` при загрузке intfloat/multilingual-e5-base
- **Статус**: ⏳ В ОЧЕРЕДИ
- **Файлы**: 
  - `eva_ai/unified_text_processor.py`
  - `eva_ai/fractal_graph_v2/embeddings.py`

---

## Приоритет 3: FractalGraph v2

### 3.1 FractalGraphV2.get_clusters() - не существует
- **Проблема**: ConceptMiner делает O(n²) на лету
- **Статус**: ⏳ В ОЧЕРЕДИ
- **Файл**: `eva_ai/memory/fractal_graph_v2/` - реализовать метод get_clusters()

---

## Приоритет 4: Knowledge & Concepts

### 4.1 ConceptExtractor - не сохраняет концепты
- **Проблема**: Только возвращает список, не сохраняет в FGv2
- **Статус**: ⏳ В ОЧЕРЕДИ
- **Файл**: `eva_ai/knowledge/concept_extractor.py`

### 4.2 ContradictionGenerator - интеграция
- **Проблема**: Создан, но не интегрирован в brain_query
- **Статус**: ⏳ В ОЧЕРЕДИ
- **Файл**: `eva_ai/contradiction/contradiction_generator.py`

---

## Приоритет 5: Security

### 5.1 SecurityFramework - HARDCODED backdoor
- **Проблема**: admin:admin backdoor (CVSS 9.8)
- **Статус**: ⏳ В ОЧЕРЕДИ
- **Файл**: `eva_ai/core/security_framework.py:141`

---

## Выполненные исправления

| # | Дата | Исправление | Файл | Статус |
|---|------|------------|------|--------|
| 1 | - | EventBus 3-element tuple fix | event_bus.py | ✅ DONE |
| 2 | 2026-04-14 | Signal handler signatures (4 метода) | brain_coordination.py, system_state.py, event_bus_bridge.py | ✅ DONE |
| 3 | 2026-04-14 | MemoryManager deferred commands with hasattr check | manager_core.py | ✅ DONE |
