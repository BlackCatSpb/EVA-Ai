# EVA System Architecture Audit Report

**Date:** April 14, 2026  
**Auditors:** AI Architect Agents (43 parallel agents across 8 cycles)  
**Document reviewed:** `system_flow_v2.md`

---

## Executive Summary

Полный аудит системы EVA выявил критические проблемы: избыточные компоненты, заглушки, сломанные миграции и отсутствие интеграции.

### Общая оценка системы: 5.2/10 (снижена с 5.4)

**КРИТИЧЕСКИЕ ПРОБЛЕМЫ ВЫЯВЛЕНЫ В ЦИКЛЕ 8:**
- BackgroundCoordinator: Detectors НЕ зарегистрированы (автопилот сломан)
- SnapshotManager: Per-request instantiation ломает персистентность
- EmbeddingsManager: CPU fallback по умолчанию, hash() нестабилен
- ResponseGenerator: extract_ambiguous_terms() критический баг подтверждён
- component_managers.py: Все 8 классов - ЗАГЛУШКИ, рекомендуется удаление

---

## 1. КРИТИЧЕСКИЕ ПРОБЛЕМЫ (6 ЦИКЛ)

### 1.0 NEW: ModelAccessManager НЕ ИНТЕГРИРОВАН (5/10)

**Проблема:** brain_query вызывает `pipeline.process_query()` напрямую без ModelAccessManager.

| Аспект | Оценка |
|--------|--------|
| brain_query | Не использует MAM |
| SelfDialogLearning | Использует MAM |
| Приоритеты | Не работают для пользовательских запросов |

### 1.1 Generation Coordinator - ДУБЛИРОВАНИЕ (4/10)

**Проблема:** GenerationCoordinator избыточен — UnifiedGenerator и HybridPipelineAdapter уже обеспечивают единую точку входа.

| Аспект | Оценка |
|--------|--------|
| Интеграция | Не использует EventBus/DCS |
| Legacy код | 70% провайдеров не используются |
| Приоритеты | Двойная система (GC priorities + MAM) |

**Рекомендация:** Удалить как избыточный слой.

---

### 1.2 NLP & Preprocessing - ДУБЛИРОВАНИЕ (4/10)

**ТРИ разных EntityExtractor:**

| Модуль | Реализация |
|--------|-----------|
| `preprocess/` | GGUF-based |
| `reasoning/` | Pattern-based (regex) |
| `gui/` | GUI extraction |

**Проблемы:**
- Нет unified preprocessing → reasoning flow
- Результаты игнорируются после clarification
- Нет EventBus интеграции

---

### 1.3 System Health - ИЗОЛИРОВАН (5/10)

| Проблема | Описание |
|----------|----------|
| Дублирование | `monitoring/` и `system/` пересекаются |
| Нет EventBus | Изолирован от системы событий |
| Нет auto-start | Только ручные вызовы |

**Fault Tolerance:** Всего 92 строки, `recovery_strategies` — пустой словарь.

---

### 1.4 NEW: EventBus Priority НЕ РАБОТАЕТ (3/10)

**Проблема:** Параметр priority в subscribe()/publish() игнорируется, FIFO вместо приоритетов.

| Аспект | Оценка |
|--------|--------|
| EventBus.subscribe() | priority параметр игнорируется |
| EventBus.publish() | Нет приоритетной обработки |
| Критические события | system.idle, pipeline.complete идут в FIFO |

### 1.5 NEW: GraphCurator ПОЛНОСТЬЮ ИЗОЛИРОВАН (2/10)

**Проблема:** GraphCurator не использует EventBus или DeferredCommandSystem.

| Аспект | Оценка |
|--------|--------|
| EventBus | ❌ Не подписан ни на что |
| DeferredCommandSystem | Присвоен но НЕ ИСПОЛЬЗУЕТСЯ |
| threading.Timer | Фиксированный интервал 600 сек |
| is_running() | Метод отсутствует |

### 1.6 Config & Scripts - СЛОМАНЫ (3/10)

**Config:**
- `optimal_config.json` и `fractal_model_config.json` — дублирование
- `apply_optimal_config.py` только печатает, не применяет
- Хардкод `os.getcwd() + "eva"` ломается при запуске из другой директории

**Scripts — СЛОМАНЫ:**

| Скрипт | Проблема |
|--------|----------|
| migrate_kg_to_fg.py | Вызывает несуществующий kg_to_fg_migration |
| migrate_to_optimized.py | Вызывает несуществующий optimized_fractal_model_manager |
| migrate_events.py | Только документация |
| load_gguf_to_fg.py | Зависит от fg_gguf_architecture_mapper |
| activate_max_cache.py | Зависит от unified_fractal_manager |
| export_qwen.py | Неверный путь eva\mlearning\ |

**Работает только:** `simple_test.py`

---

## 2. ПОЛНЫЙ РЕЙТИНГ КОМПОНЕНТОВ (25)

| # | Компонент | Оценка | Главная проблема |
|---|----------|--------|-----------------|
| 1 | SelfDialogLearning | 8.0/10 | FIFO queue |
| 2 | Web GUI | 8.0/10 | Thread leak |
| 3 | Concept system | 7.5/10 | Нет автосохранения |
| 4 | Contradiction | 7.2/10 | Могут дублировать |
| 5 | CoreBrain init | 7.0/10 | Mixed style |
| 6 | Backends | 7.0/10 | Transformers/ONNX stubs |
| 7 | FractalGraphV2 | 7.0/10 | Нет get_clusters() |
| 8 | Monitoring | 7.0/10 | Изолирован |
| 9 | PyTorch Adapter | 7.0/10 | Не интегрирован |
| 10 | Storage/Cache | 6.5/10 | Pickle, нет TTL |
| 11 | Reasoning | 6.5/10 | Дублирование SRE |
| 12 | Tools/Security | 6.5/10 | Слабое SHA256 |
| 13 | Generation | 6.5/10 | 3+ coordinator |
| 14 | Coordination | 6.5/10 | Priority не работает |
| 15 | Wikipedia KB | 6.2/10 | CPU-only |
| 16 | NLP | 4.0/10 | Legacy, не интегрирован |
| 17 | Preprocess | 4.0/10 | 3 EntityExtractor |
| 18 | System Health | 5.0/10 | Изолирован |
| 19 | WebSearch | 5.0/10 | Подмена поисковиков |
| 20 | brain_query | 5.5/10 | Нет интеграции |
| 21 | Adaptation | 4.5/10 | 4 версии класса |
| 22 | GraphCurator | 4.2/10 | Нет EventBus/DCS |
| 23 | Config | 3.0/10 | Дублирование, не работает |
| 24 | Scripts | 3.0/10 | Сломанные миграции |
| 25 | Distributed | 3.0/10 | Заглушки |
| 26 | Recovery | 3.0/10 | Сирота |
| 27 | Neuromorphic | 2.0/10 | Fallback-only |
| 28 | Fractal standalone | 3.0/10 | Не используется |

---

## 3. ДУБЛИРОВАНИЯ (ИТОГ)

### 3.1 Четырёхкратное дублирование

| Функция | Версии |
|---------|--------|
| **EntityExtractor** | 3 (preprocess, reasoning, gui) |
| **AdaptationManager** | 4 |
| **RecoveryManager** | 3 |
| **Fractal*Store** | 3 |
| **SearchResult** | 2 |

### 3.2 Избыточные системы

- **GenerationCoordinator** — дублирует UnifiedGenerator
- **System Health** — дублирует Monitoring
- **Config файлы** — дублируют друг друга

---

## 4. СЛОМАННЫЕ КОМПОНЕНТЫ

### Scripts (6 из 8 сломаны):
- `migrate_kg_to_fg.py` → несуществующий импорт
- `migrate_to_optimized.py` → несуществующий импорт
- `export_qwen.py` → неверный путь

### Config (2 из 2 проблемы):
- `optimal_config.json` = `fractal_model_config.json`
- `apply_optimal_config.py` не применяет настройки

---

## 5. НЕИНТЕГРИРОВАННЫЕ СИСТЕМЫ

| Система | EventBus | DeferredCommandSystem |
|---------|---------|---------------------|
| GenerationCoordinator | ❌ | ❌ |
| NLP/Preprocess | ❌ | ❌ |
| System Health | ❌ | ❌ |
| Fault Tolerance | ❌ | ❌ |
| Config scripts | ❌ | ❌ |
| Neuromorphic | ❌ | ❌ |
| Distributed | ❌ | ❌ |

---

## 6. РЕКОМЕНДАЦИИ

### НЕМЕДЛЕННО:
1. **Удалить GenerationCoordinator** — избыточен
2. **Починить или удалить сломанные миграции**
3. **Удалить дублирующие config файлы**
4. **Интегрировать или удалить** NLP/Preprocess

### ВЫСОКИЙ ПРИОРИТЕТ:
5. Объединить 3 EntityExtractor в 1
6. Интегрировать System Health в EventBus
7. Реализовать Fault Tolerance или удалить
8. Интегрировать PyTorch Adapter в pipeline

---

## 7. ФАЙЛЫ ОТЧЁТОВ (6 ЦИКЛОВ)

**Цикл 1-4:** (22 компонента)
- audit_summary.md
- audit_*.md (22 файла)

**Цикл 5:** (новые находки)
- audit_generation_coordinator.md (4/10)
- audit_nlp_preprocess.md (4/10)
- audit_system_runtime.md (5/10)
- audit_config_scripts.md (3/10)

**Цикл 6:** (интеграционные проблемы)
- audit_model_access_system.md (5/10) - ModelAccessManager не интегрирован в brain_query
- audit_eventbus_priorities.md (3/10) - Priority игнорируется, FIFO вместо приоритетов
- audit_graph_curator_integration.md (2/10) - Нет EventBus/DCS интеграции
- audit_concept_extractor_autosave.md (4.5/10) - extract_concepts() не сохраняет автоматически
- audit_websearch_providers.md (5/10) - Google→DuckDuckGo, Yandex→Brave подмена
- audit_duplicate_systems.md (3.5/10) - 4 AdaptationManager, 3 RecoveryManager, 3 FractalStore

**Цикл 7:** (новые компоненты)
- audit_deferred_command_system.md (7.5/10) - DCS работает, PriorityQueue используется, но EventBus синглтон
- audit_hybrid_cache.md (5.3/10) - Pickle небезопасен, TTL частично, интеграция с memory_manager
- audit_pipeline_adapter.md (6/10) - max_tokens завышены x16-32, DualGenerator не использует MAM
- audit_entity_extractor.md (3/10) - 4 версии, extract_ambiguous_terms() НЕ СУЩЕСТВУЕТ (критический баг!)
- audit_dual_generator.md (8/10) - dual_generator_pie.py мёртвый код, дублирование _clean_response
- audit_fractal_memory_graph.md (5.6/10) - НЕТ get_clusters(), нет WAL, embedding fallback на случайные векторы

**Цикл 8:** (фоновые системы и критические компоненты)
- audit_background_coordinator.md (6/10) - Detectors НЕ зарегистрированы, автопилот сломан
- audit_snapshot_manager.md (5/10) - Per-request instantiation, нет deep copy, коллизия имён
- audit_embeddings_manager.md (3.7/10) - CPU fallback по умолчанию, hash() нестабилен, EmbeddingCache не подключен
- audit_knowledge_curator.md (2/10) - KnowledgeCurator НЕ СУЩЕСТВУЕТ, GraphCurator изолирован
- audit_response_generator.md (3/10) - extract_ambiguous_terms() критический баг подтверждён
- audit_component_managers.md (3/10) - Все 8 классов ЗАГЛУШКИ, рекомендуется удаление

---

## 8. КРИТИЧЕСКИЕ ПРОБЛЕМЫ (НОВЫЕ)

### 8.1 ModelAccessManager - НЕ ИНТЕГРИРОВАН В brain_query (5/10)

| Проблема | Описание |
|----------|----------|
| brain_query вызывает напрямую | `pipeline.process_query()` без координации |
| ModelAccessManager создан | В `unified_generator.py` но не используется в `brain_query.py` |
| Приоритеты не работают | CRITICAL запросы идут в FIFO очереди |
| Конфликты доступа | Самодиалог и запросы конкурируют без приоритетов |

**Рекомендация:** Интегрировать ModelAccessManager.request_access() в `_handle_gguf_pipeline()`

### 8.2 EventBus Priority System - НЕ РАБОТАЕТ (3/10)

| Проблема | Описание |
|----------|----------|
| Priority игнорируется | Все подписчики получают события в FIFO порядке |
| Нет приоритетной очереди | `queue.PriorityQueue` не используется в EventBus |
| Критические события | `system.idle`, `pipeline.complete` не имеют приоритетов |

**Рекомендация:** Переписать EventBus.subscribe() для поддержки приоритетов

### 8.3 GraphCurator - ПОЛНОСТЬЮ ИЗОЛИРОВАН (2/10)

| Проблема | Описание |
|----------|----------|
| Нет EventBus | Не подписывается ни на какие события |
| Нет DCS | `deferred_system` присвоен но никогда не используется |
| threading.Timer | Фиксированный интервал 600 сек вместо адаптивного |
| Нет is_running() | Ломает проверку в brain_init.py |

**Рекомендация:** Переписать с использованием EventBus и DeferredCommandSystem

### 8.4 ConceptExtractor - НАРУШАЕТ SRP (4.5/10)

| Проблема | Описание |
|----------|----------|
| extract_concepts() не сохраняет | Только возвращает список, не вызывает save_concept_to_graph() |
| Сохранение в brain_query | Нарушение инкапсуляции |
| Нет EventBus событий | Другие компоненты не могут реагировать |

**Рекомендация:** Добавить автосохранение в `extract_concepts()` + EventBus.publish()

### 8.5 WebSearch - ПОДМЕНА ПОИСКОВИКОВ (5/10)

| Заявлено | Реальность | Оценка |
|----------|------------|--------|
| search_google() | DuckDuckGo HTML | 2/10 |
| search_yandex() | Brave Search | 2/10 |
| Tavily API | Основной движок, нет fallback | 7/10 |
| Wikipedia | Официальный API | 9/10 |

**Рекомендация:** Убрать подмену или документировать ограничения

---

## 9. НОВЫЕ КРИТИЧЕСКИЕ ПРОБЛЕМЫ (ЦИКЛ 7)

### 9.1 EntityExtractor - КРИТИЧЕСКИЙ БАГ (3/10)

| Проблема | Описание |
|----------|----------|
| `extract_ambiguous_terms()` НЕ СУЩЕСТВУЕТ | Вызывается в 7 местах, но метода нет! |
| 4 версии EntityExtractor | Массовое дублирование |
| Нет EventBus интеграции | Изоляция от системы событий |

**Вызывается в:**
```
contradiction/core_detection.py: Lines 321, 322, 362, 363, 655
memory/manager_operations.py: Line 436
core/response_generator.py: Line 191
```

### 9.2 FractalMemoryGraph - НЕТ get_clusters() (5.6/10)

| Проблема | Описание |
|----------|----------|
| Метод `get_clusters()` отсутствует | ConceptMiner делает O(n^2) кластеризацию на лету |
| SQLite без WAL | Блокировка записи при чтении |
| Embedding fallback | Случайные векторы ломают семантический поиск |

### 9.3 HybridCache - Pickle НЕБЕЗОПАСЕН (5.3/10)

| Проблема | Описание |
|----------|----------|
| Pickle используется | `cache_disk.py`, `disk_cache.py` - уязвимость |
| TTL частично | Для `_get_token_impl` не проверяется |
| Нет персистентности | embedding кэш теряется при перезапуске |

### 9.4 DualGenerator - МЁРТВЫЙ КОД (8/10)

| Проблема | Описание |
|----------|----------|
| `dual_generator_pie.py` не используется | Можно удалить |
| `_clean_response()` дублируется | В CondensedGenerator и ExtendedGenerator |

---

## 10. НОВЫЕ КРИТИЧЕСКИЕ ПРОБЛЕМЫ (ЦИКЛ 8)

### 10.1 BackgroundCoordinator - АВТОПИЛОТ СЛОМАН (6/10)

| Проблема | Описание |
|----------|----------|
| Detectors НЕ зарегистрированы | `_detectors` пуст, автопилот не работает |
| Jobs зарегистрированы | TrainingJob, WebIndexJob, ModuleRecoveryJob |
| EventBus интеграция | Двойная (legacy + new) |

**brain_components.py:486-512** - Detectors не зарегистрированы:
```python
# НЕ зарегистрированы:
# brain.background.register_detector(LearningOpportunityDetector())
# brain.background.register_detector(WebDiscoveryDetector())
# brain.background.register_detector(ModuleRecoveryDetector())
```

### 10.2 SnapshotManager - PER-REQUEST INSTANTIATION (5/10)

| Проблема | Описание |
|----------|----------|
| Создаётся каждый запрос | `HybridPipelineAdapter` создаёт новый инстанс |
| Нет deep copy | Данные сохраняются по ссылке |
| Два класса с одним именем | `snapshot_manager.py` и `graph_learning.py` |

### 10.3 EmbeddingsManager - CPU ПО УМОЛЧАНИЮ (3.7/10)

| Проблема | Описание |
|----------|----------|
| init_factories.py:469 | `embedding_device="cpu"` по умолчанию |
| hash() нестабилен | `str(hash(text))` меняется между запусками |
| EmbeddingCache не подключен | SQLite cache не используется |
| Нет retry на CPU | При ошибке GPU - случайные векторы |

### 10.4 KnowledgeCurator - НЕ СУЩЕСТВУЕТ (2/10)

| Проблема | Описание |
|----------|----------|
| KnowledgeCurator не найден | Есть только GraphCurator |
| SelfDialogLearning подписан | На события `curator.*` которых нет |
| GraphCurator изолирован | Не публикует события |

### 10.5 ResponseGenerator - КРИТИЧЕСКИЙ БАГ (3/10)

| Проблема | Описание |
|----------|----------|
| extract_ambiguous_terms() | НЕ СУЩЕСТВУЕТ, вызывается в 7+ местах |
| brain_query не использует | ResponseGenerator вне fallback chain |
| Fallback сломан | `_fallback_generation()` никогда не возвращается |

### 10.6 component_managers.py - 8 ЗАГЛУШЕК (3/10)

| Класс | Реализация |
|-------|------------|
| SecurityManager | Проксирует вызовы |
| AuthManager | Принимает любой пароль |
| AlertManager | Пустой класс |
| MonitoringManager | Всегда "healthy" |
| HealthChecker | Только обёртка |
| MetricsCollector | Без истории |
| RecoveryManager | Ничего не восстанавливает |
| StateManager | Нет персистентности |

**Рекомендация: УДАЛИТЬ component_managers.py полностью**

---

## 11. ВЫВОДЫ

Система EVA имеет хороший фундамент, но страдает от:
- **Избыточности** — 8 заглушек в component_managers.py
- **Сломанности** — extract_ambiguous_terms() не существует, автопилот сломан
- **Изоляции** — компоненты не интегрированы в EventBus
- **Персистентности** — SnapshotManager создаётся на каждый запрос

**Для production необходимо:**
1. Удалить избыточные компоненты (component_managers.py)
2. Починить сломанные миграции
3. Интегрировать разрозненные системы
4. Объединить дублирующиеся модули
5. **ИСПРАВИТЬ КРИТИЧЕСКИЙ БАГ: extract_ambiguous_terms()**
6. **ЗАРЕГИСТРИРОВАТЬ Detectors в BackgroundCoordinator**

---

*Отчёт подготовлен AI Architect Agents*
*43 специализированных агентов проверили 43+ компонентов*
*April 14, 2026*
