# Deep Dive Analysis Summary

## Overview

Проведены углублённые исследования всех критических и высоких проблем из `cross_analysis_final.md`.

---

## C1: FCP Isolated from WebSearch/Ethics

**Файл:** `deep_dive_C1_FCP_INTEGRATION.md`

### Суть проблемы:
FCPPipelineV15 генерирует ответы изолированно:
- Нет WebSearch обогащения запроса
- Нет этической проверки ответа
- Нет контекста от Self-Dialog Learning

### Точки интеграции:
1. **До генерации:** Добавить `needs_web_search()` → `web_search_engine.search()`
2. **После генерации:** Добавить `ethics_framework.check_with_context()`

### Код изменений (brain_query.py):
```python
# BEFORE FCP generate (~line 340):
need_search, _ = needs_web_search(query)
if need_search:
    result = web_search.search(query, max_results=5)
    enhanced_query = f"{query}\n\nContext: {result}"

response = pipeline.generate(enhanced_query, ...)

# AFTER generate:
ethics_result = ethics_fw.check_with_context(response, query)
```

---

## C2: 3x Duplicate /api/chat

**Файл:** `deep_dive_C2_DUPLICATE_CHAT.md`

### Суть проблемы:
Три определения в разных файлах создают непредсказуемое поведение.

| # | Файл | Строк | Статус |
|---|------|-------|--------|
| 1 | `server_routes.py` | ~151 | ❌ УДАЛИТЬ |
| 2 | `gui/web_gui/server_routes.py` | ~399 | ❌ УДАЛИТЬ |
| 3 | `gui/web_gui/server_routes_chat.py` | ~18 | ✅ ОСТАВИТЬ |

### Почему #3 правильный:
- Подключён через `register_chat_routes(app, web_gui_instance)`
- Исправляет одинарные кавычки в JSON
- Есть timeout для длинных запросов (120-360 сек)
- Детальное логирование
- Есть версионированный `/v1/chat`

---

## C3: Three Contradiction Systems

**Файл:** `deep_dive_C3_CONTRADICTION_SYSTEMS.md`

### Суть проблемы:
Три системы выполняют похожие функции:

| Система | Метод | Триггер | Статус |
|---------|-------|---------|--------|
| Generator | Template-based | concept extracted | Оставить для стимуляции |
| Miner | Cosine + NLI | system.idle | Оставить как основную |
| Legacy detect | semantic/logical/temporal | manual | Интегрировать как layer |

### Рекомендуемая архитектура:
```
UnifiedContradictionManager (единая точка входа)
    ↓
    ├── Fast Layer: Generator (шаблоны)
    ├── Core Layer: Miner (FGv2 анализ)
    └── Fact Layer: Legacy (consistency check)
```

---

## H1: KGAdapter Not Created

**Файл:** `deep_dive_H1_KGADAPTER.md`

### Суть проблемы:
При рефакторинге функция `create_knowledge_graph()` была переименована в `create_knowledge_components()`, но создание KGAdapter было удалено.

**485 обращений** к `brain.knowledge_graph` ожидают его наличия.

### Решение:
Добавить в `create_knowledge_components()` (строка ~493):
```python
kg_adapter = KnowledgeGraphAdapter(fractal_graph)
brain.knowledge_graph = kg_adapter
```

---

## H2: DialogConceptsMixin - ЛОЖНАЯ ПРОБЛЕМА

**Файл:** `deep_dive_H2_DIALOG_CONCEPTS.md`

### Важное открытие:
DialogConceptsMixin **ПРАВИЛЬНО инициализирован** в `dialog_core.py:89`.

Система работает:
- Очередь концептов → обрабатывается
- Очередь противоречий → обрабатывается
- EventBus интеграция → работает

**Единственный реальный баг:** H3 (summary_parts)

---

## H3: summary_parts Not Defined

**Файл:** `deep_dive_H3_H4.md`

### Суть проблемы:
В `_run_graph_curator_after_cycle()` (dialog_core.py:1049):
```python
return " | ".join(summary_parts)  # summary_parts НЕ определён!
```

### Исправление:
```python
def _run_graph_curator_after_cycle(self):
    # ... code ...
    # summary_parts = []  # Добавить если нужен return
    return  # Убрать бессмысленный return
```

---

## H4: SystemMonitor Isolated

**Файл:** `deep_dive_H3_H4.md`

### Суть проблемы:
SystemMonitor работает автономно, не подключён к EventBus CoreBrain.

### План интеграции:

1. **Добавить обработчики в SystemMonitor:**
```python
def on_system_ready(self, data):
    """Событие: система готова"""
    pass

def on_component_error(self, data):
    """Событие: ошибка компонента"""
    self.log_error(f"Component error: {data}")
```

2. **В init_factories.py подписать:**
```python
event_bus.subscribe('system.ready', system_monitor.on_system_ready)
event_bus.subscribe('component.error', system_monitor.on_component_error)
```

3. **SystemMonitor публикует:**
```python
if problem_detected:
    event_bus.publish('system.error', {'component': name, 'error': msg})
```

---

## Итоговая таблица

| ID | Проблема | Суть | Статус |
|----|----------|------|--------|
| C1 | FCP изоляция | Нет WebSearch/Ethics | Требует исправления |
| C2 | Дублирование /api/chat | 3 определения | Требует удаления |
| C3 | Три системы противоречий | Дублирование | Требует объединения |
| H1 | KGAdapter не создан | missing | Требует создания |
| H2 | DialogConceptsMixin | **ЛОЖНАЯ** | Не требует действий |
| H3 | summary_parts | undefined | Требует исправления |
| H4 | SystemMonitor изолирован | no EventBus | Требует интеграции |

---

## References

- `cross_analysis_fcp_ethics.md` - исходный анализ C1
- `cross_analysis_server_monitoring.md` - исходный анализ C2, H4
- `cross_analysis_dialog_miners.md` - исходный анализ C3, H2
- `cross_analysis_core_memory.md` - исходный анализ H1
- `self_dialog_system.md` - анализ dialog_core