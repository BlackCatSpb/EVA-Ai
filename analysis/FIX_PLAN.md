# ПЛАН ИСПРАВЛЕНИЙ EVA AI

**Дата:** 27.04.2026  
**Порядок:** Фундамент первым  
**Обоснование:** Без базовой инфраструктуры (EventBus, Security) исправление функционала бессмысленно

---

## Порядок работ

### Фаза 0: Фундамент (дни 1-3)

#### K6: EventBus PriorityQueue (ДЕНЬ 1) ⚠️ КРИТИЧЕСКИ
**Почему первый:** Вся коммуникация компонентов сломана - приоритеты не работают!

**Файлы:**
- `eva_ai/core/event_bus.py` или `event_system.py`

**Что делать:**
1. Найти PriorityQueue использование
2. Исправить чтобы работали приоритеты (CRITICAL > HIGH > NORMAL > LOW)
3. Проверить подписки компонентов

**Признак успеха:** `subscribe(..., priority=CRITICAL)` выполняется раньше `subscribe(..., priority=LOW)`

---

#### K8: Pickle Security (ДЕНЬ 2) ⚠️ КРИТИЧЕСКИ
**Почему второй:** Уязвимость безопасности в 15+ местах

**Файлы:**
- `cache_disk.py`
- `disk_cache.py`
- `fractal_torch_storage.py`
- `storage/*.py`

**Что делать:**
1. Заменить `pickle.load()` на `json.load()` или `msgpack`
2. Добавить валидацию данных
3. Удалить `pickle.dump()` где возможно

**Риск:** Высокий - может сломать загрузку моделей

---

#### K1: Distributed System (ДЕНЬ 3) ⚠️ КРИТИЧЕСКИ
**Почему третий:** Мёртвый код, ошибки инициализации

**Файлы:**
- `distributed/` модуль
- `core/init_*` - где вызывается `_init_distributed_system()`

**Что делать:**
1. Найти вызов `_init_distributed_system()`
2. Либо реализовать, либо удалить модуль distributed
3. Проверить нет ли других зависимостей

---

### Фаза 1: Критический функционал (дни 4-7)

#### C2: /api/chat дублирование (ДЕНЬ 4) ⚠️ КРИТИЧЕСКИ
**Почему:** Непредсказуемое поведение сервера

**Файлы:**
- `server_routes.py` (удалить строку ~151)
- `gui/web_gui/server_routes.py` (удалить строку ~399)
- `gui/web_gui/server_routes_chat.py` (ОСТАВИТЬ)

**Что делать:**
1. Удалить определения из первых двух файлов
2. Оставить только в server_routes_chat.py
3. Проверить что сервер запускается

---

#### C1: FCP + WebSearch + Ethics (ДЕНЬ 5-6) ⚠️ КРИТИЧЕСКИ
**Почему:** FCP генерирует ответы без обогащения

**Файлы:**
- `brain_query.py` (~340-360)

**Что делать:**
1. Добавить WebSearch вызов перед FCP генерацией
2. Добавить Ethics проверку после генерации
3. Проверить TwoModelPipeline как reference

---

#### C3: Contradiction Systems (ДЕНЬ 7) ⚠️ КРИТИЧЕСКИ
**Почему:** Три системы делают одно и то же

**Файлы:**
- `contradiction/contradiction_generator.py`
- `contradiction/contradiction_miner.py`
- `contradiction/detect_*.py`

**Что делать:**
1. Создать `UnifiedContradictionManager`
2. Оставить Miner как основу
3. Объединить Generator и Legacy как слои

---

### Фаза 2: Высокий приоритет (дни 8-12)

#### H3: summary_parts (ДЕНЬ 8) ⚡ БЫСТРО
**Файлы:** `dialog_core.py:1049`

**Исправление:**
```python
# Убрать бессмысленный return или добавить:
summary_parts = []
return " | ".join(summary_parts)
```

---

#### H1: KGAdapter (ДЕНЬ 8-9)
**Файлы:** `init_factories.py` (~493)

**Исправление:**
```python
kg_adapter = KnowledgeGraphAdapter(fractal_graph)
brain.knowledge_graph = kg_adapter
```

---

#### H4: SystemMonitor (ДЕНЬ 9-10)
**Файлы:** `system_monitor.py`, `init_factories.py`

**Исправление:**
1. Добавить обработчики событий в SystemMonitor
2. Подписать на EventBus в init_factories

---

#### V1: hot_deployment мёртвый код (ДЕНЬ 10-11)
**Файлы:** `hot_deployment/` (10 из 12 файлов)

**Что делать:**
1. Проверить какие файлы используются (grep import)
2. Удалить неиспользуемые

---

#### V2: OpenVINOGenerator дубли (ДЕНЬ 11-12)
**Файлы:** 3 версии в разных местах

**Решение:** Объединить в один класс

---

#### V4: LRUCache дубли (ДЕНЬ 12) ⚡ БЫСТРО
**Файлы:** `cache_ram.py` vs `memory_cache.py`

**Решение:** Удалить один, использовать другой

---

### Фаза 3: Тестирование (дни 13-14)

| # | Тест | День |
|---|------|------|
| 3.1 | Запуск EVA с нуля | 13 |
| 3.2 | /api/chat работает | 13 |
| 3.3 | FCP + WebSearch + Ethics | 14 |
| 3.4 | EventBus приоритеты | 14 |
| 3.5 | Self-dialog работает | 14 |

---

## Статус

| ID | Задача | Статус |
|----|--------|--------|
| K6 | EventBus PriorityQueue | ⏳ |
| K8 | Pickle Security | ⏳ |
| K1 | Distributed System | ⏳ |
| C2 | /api/chat дубли | ⏳ |
| C1 | FCP + WebSearch | ⏳ |
| C3 | Contradiction объединение | ⏳ |
| H3 | summary_parts | ⏳ |
| H1 | KGAdapter | ⏳ |
| H4 | SystemMonitor | ⏳ |
| V1 | hot_deployment | ⏳ |
| V2 | OpenVINOGenerator | ⏳ |
| V4 | LRUCache | ⏳ |

---

## Ресурсы

- Deep Dive отчёты: `analysis/deep_dive_*.md`
- PR#3 анализ: `analysis/pr3_external_audit_analysis.md`
- Финальный отчёт: `analysis/cross_analysis_final_v2.md`