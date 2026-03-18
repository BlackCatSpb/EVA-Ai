# CogniFlex — Детальный план рефакторинга

> Составлен: 2026-03-18
> Ветка: `dev/refactor-cleanup`

---

## Статус после первичной чистки (выполнено)

- [x] Исправлен merge conflict в `brain_config.json`
- [x] Удалено ~60 одноразовых скриптов из корня проекта
- [x] Все MD-документы перемещены в `docs/`
- [x] Исправлен `.gitignore` (тесты теперь не игнорируются)

---

## ФАЗА 0 — Критические баги (блокируют запуск)

### 0.1 QueryProcessor.process() не существует
- **Файл**: `cogniflex/core/query_processor.py`
- **Ошибка**: `'QueryProcessor' object has no attribute 'process'`
- **Фикс**: добавить алиас `process = process_query` или переименовать

### 0.2 MemoryManager.memory_locks отсутствует
- **Файл**: `cogniflex/memory/memory_manager.py`
- **Ошибка**: `'MemoryManager' object has no attribute 'memory_locks'`
- **Фикс**: инициализировать `self.memory_locks = {}` в `__init__`

### 0.3 BackgroundCoordinator — ошибка интеграции (код 10)
- **Файл**: `cogniflex/core/background_coordinator.py`
- **Ошибка**: `Ошибка интеграции с системой событий: 10`
- **Фикс**: привести API подписки к текущему интерфейсу EventSystem

### 0.4 Путь к модели захардкожен и неверный
- **Файл**: `cogniflex/core/core_brain.py:297`
- **Проблема**: путь к fractal модели относительный и не совпадает с реальной структурой
- **Фикс**: путь к модели читать из `brain_config.json → model.path`, с проверкой наличия файла

---

## ФАЗА 1 — Чистка модульной структуры

### 1.1 Удалить дубликаты в `cogniflex/memory/`

| Удалить (дубликат) | Оставить (canonical) |
|---|---|
| `memory_long_term.py` | `long_term_memory.py` |
| `memory_working.py` | `working_memory.py` |
| `memory_cache.py` | `memory_core.py` |

**Перед удалением**: проверить импорты через `grep -r "memory_long_term\|memory_working\|memory_cache" cogniflex/`
**После**: обновить все импорты

### 1.2 Удалить дубликаты в `cogniflex/gui/`

| Удалить | Оставить |
|---|---|
| `gui_util.py` | `gui_utils.py` |
| `widgets.py` | `gui_widgets.py` |
| `settings.py` | `settings_module.py` |

### 1.3 Убрать дубликат `generation_coordinator`

- `cogniflex/core/generation_coordinator.py` — новый файл (последний коммит)
- `cogniflex/generation/generation_coordinator.py` — старый

**Решение**: сравнить содержимое, объединить, оставить в `cogniflex/generation/`, удалить из `core/`, добавить импорт в `core/__init__.py`

### 1.4 Навести порядок в `cogniflex/mlearning/` — 12 менеджеров моделей

Текущий зоопарк файлов:
```
model_manager.py              ← единая точка входа (canonical)
hybrid_model_manager.py       ← вариант с несколькими моделями
universal_model_manager.py    ← ещё один вариант
current_manager.py            ← неизвестно что
unified_fractal_manager.py    ← fractal-специфичный
fractal_model_manager.py      ← дубликат fractal
fractal_rugpt3_manager.py     ← ещё один fractal rugpt3
enhanced_rugpt3_manager.py    ← "улучшенный" rugpt3
rugpt3_model_manager.py       ← базовый rugpt3
local_rugpt3_loader.py        ← загрузчик
qwen_model_manager.py         ← qwen (отключён)
bitnet_model_manager.py       ← bitnet (задел на будущее)
```

**Целевая структура**:
```
mlearning/
  model_manager.py            ← единый фасад (ModelManager)
  backends/
    rugpt3_backend.py         ← весь rugpt3-специфичный код
    bitnet_backend.py         ← bitnet (из bitnet_model_manager.py)
    qwen_backend.py           ← qwen (из qwen_model_manager.py)
  training/
    training_orchestrator.py  ← объединить simple + full
    fractal_trainer.py
```

### 1.5 Убрать дубликаты `*_integrated.py` паттерна

В каждом модуле есть: `*_core.py`, `*_manager.py`, `*_integrated.py`, `*_integration.py`.
Это артефакт вайбкодинга — каждый раз создавался новый файл вместо редактирования старого.

**Модули для ревью**:
- `cogniflex/adaptation/` — 6 файлов, реально нужно 2-3
- `cogniflex/contradiction/` — 10 файлов, реально нужно 3-4
- `cogniflex/ethics/` — 7 файлов (`cogniflex_ethics.py` + `ethics_framework.py` + `ethics_core.py` — скорее всего одно и то же)
- `cogniflex/knowledge/` — 13 файлов
- `cogniflex/learning/` — 14 файлов

### 1.6 Переместить файлы не на своих местах

```
# Переместить:
cogniflex/nlp_fallbacks.py          → cogniflex/nlp/fallbacks.py
cogniflex/system_selftest.py        → tools/system_selftest.py
cogniflex/setup.py                  → удалить или поднять в корень
cogniflex/config/apply_optimal_config.py  → scripts/
cogniflex/scripts/complete_fractal_solution.py  → удалить (копия)

# Создать недостающие __init__.py:
cogniflex/monitoring/__init__.py
cogniflex/neuromorphic/__init__.py
cogniflex/runtime/__init__.py
cogniflex/storage/__init__.py
cogniflex/system/__init__.py
```

---

## ФАЗА 2 — Архитектурный рефакторинг

### 2.1 Разбить CoreBrain

`core_brain.py` — 2238 строк, инициализирует 58+ компонентов.

**Целевая структура**:
```python
# core_brain.py — только оркестрация, ~300 строк
class CoreBrain:
    def __init__(self, config):
        self.config = config
        self._boot()

    def _boot(self):
        ComponentRegistry.initialize(self)  # ← вынести сюда

# core/registry.py — регистрация и lifecycle компонентов
class ComponentRegistry:
    REQUIRED = [EventSystem, ConfigManager, ResourceManager, MemoryManager]
    OPTIONAL = [WebSearchEngine, EthicsManager, DistributedSystem, ...]
```

### 2.2 Явный список обязательных vs опциональных компонентов

Сейчас всё инициализируется в try/except — непонятно что критично.

```python
# core/component_spec.py — создать
REQUIRED_COMPONENTS = {
    "event_system": EventSystem,
    "config_manager": ConfigManager,
    "resource_manager": ResourceManager,
    "memory_manager": MemoryManager,
    "query_processor": QueryProcessor,
    "response_generator": ResponseGenerator,
}

OPTIONAL_COMPONENTS = {
    "knowledge_graph": (KnowledgeGraph, "semantic search disabled"),
    "ethics_manager": (EthicsManager, "ethics checks disabled"),
    "web_search": (WebSearchEngine, "web search disabled"),
    "distributed": (DistributedSystem, "clustering disabled"),
}
```

### 2.3 Разбить knowledge_graph.py (308KB)

```
knowledge/
  knowledge_graph.py        ← фасад + NetworkX граф, ~300 строк
  nodes.py                  ← NodeType, RelationType, Node классы
  storage.py                ← SQLite persistence
  search.py                 ← поиск по графу
  analytics.py              ← метрики графа
  visualization.py          ← визуализация
```

### 2.4 Унифицировать конфигурацию

Сейчас: `brain_config.json` + `cogniflex/config.py` + `config/universal_analyzer_config.ini` — три источника правды.

**Цель**: один `brain_config.json`, читается через `ConfigManager`, все модули получают конфиг через DI.

---

## ФАЗА 3 — Слой моделей (подробнее в MODEL_SELECTION.md)

### 3.1 Исправить TODO-заглушки

```python
# fractal_rugpt3_manager.py:337
# TODO: Реализовать применение фрактальных весов
# → либо реализовать, либо удалить fractal-слой и работать напрямую с HF

# unified_storage.py:110
# TODO: Implement text vectorization
# → подключить sentence-transformers или USE

# unified_fractal_store.py:149, 163
# TODO: Implement proper batching
# → реализовать простой батч через torch.stack
```

### 3.2 Добавить поддержку нескольких моделей через бэкенды

См. `docs/MODEL_SELECTION.md`

---

## ФАЗА 4 — Тесты и CI

### 4.1 Убрать дублирующиеся тесты

В `tests/` — 79 файлов, многие перекрываются по scope.
Сгруппировать:
```
tests/
  unit/           ← изолированные тесты компонентов
  integration/    ← тесты взаимодействия
  e2e/            ← полный pipeline
  benchmarks/     ← перформанс-тесты
```

### 4.2 Добавить минимальный CI

```yaml
# .github/workflows/ci.yml
- pytest tests/unit/ --no-model-weights
- python minimal_test.py
- python -m cogniflex.run --selftest
```

### 4.3 Покрыть непокрытые модули тестами

- `cogniflex/knowledge/knowledge_graph.py` — нет тестов
- `cogniflex/ethics/` — нет assertion-тестов
- `cogniflex/gui/` — только smoke tests

---

## ФАЗА 5 — Фичи и улучшения

### 5.1 Включить обучение

```json
// brain_config.json
"learning": {
  "enable_training": true,       ← включить
  "training_disabled": false
}
```

Требует исправления фаз 0-3.

### 5.2 Sentence-transformers для эмбеддингов

Вернуть SentenceTransformer для семантического поиска в Knowledge Graph.
Оптимальная модель: `intfloat/multilingual-e5-small` — 117MB, работает на CPU.

### 5.3 REST API

Добавить FastAPI обёртку поверх `CoreBrain`:
```
cogniflex/api/
  main.py         ← FastAPI app
  routes/
    chat.py       ← POST /chat
    memory.py     ← GET /memory
    knowledge.py  ← GET /knowledge/search
```

### 5.4 WebSocket для GUI

Заменить polling в GUI на WebSocket для real-time стриминга ответов.

---

## Приоритетность

```
Немедленно (неделя 1):
  Фаза 0 — критические баги
  Фаза 1.1–1.3 — очевидные дубликаты

Краткосрочно (недели 2-3):
  Фаза 1.4–1.6 — mlearning cleanup
  Фаза 2.1–2.2 — CoreBrain + registry

Среднесрочно (месяц 1-2):
  Фаза 2.3–2.4 — knowledge graph, конфиг
  Фаза 3 — модельный слой
  Фаза 4 — тесты и CI

Долгосрочно (месяц 2+):
  Фаза 5 — новые фичи, API, обучение
```
