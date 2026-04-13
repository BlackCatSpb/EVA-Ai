# EVA AI - Карта ошибок с функциональной картой

> Дата: 2026-04-13
> Файлов: 469 | Модулей: 38 | Ошибок: 41

---

## Содержание

1. [🔴 Критические ошибки](#-критические-ошибки-7-штук)
2. [🟠 Высокий приоритет](#-высокий-приоритет-11-штук)
3. [🟡 Средний приоритет](#-средний-приоритет-17-штук)
4. [🟢 Низкий приоритет](#-низкий-приоритет-6-штук)

---

## 🔴 Критические ошибки (7 штук)

### core/unified_generator.py

| Ошибка | Строки | Детали | Статус |
|--------|--------|--------|--------|
| ✅ 1 Неправильные параметры в create_unified_generator | 1157-1162 | general_model_path → logic_model_path | **ИСПРАВЛЕНО** |
| ✅ 8 system_prompt в wrong role | 833 | context+query в одном user role | **ИСПРАВЛЕНО** |

**Классы:** SimpleRouter, ModelType, UnifiedGenerator

**Методы:** generate, generate_dual, generate_iterative, generate_streaming, _format_prompt, _load_model

**Импорты:** typing, logging, pathlib.Path, enum.Enum, dataclasses

**Проблема в логике:** Фабричная функция create_unified_generator передаёт неверные имена параметров в конструктор. Конструктор ожидает logic_model_path, context_model_path, coder_model_path, а получает general_model_path и code_model_path.

---

### training/gguf_training_system.py

| Ошибка | Строки | Детали | Статус |
|--------|--------|--------|--------|
| ✅ 2 Обучение не реализовано | 461-494 | теперь работает через distillation | **ИСПРАВЛЕНО** |

**Классы:** TrainingStatus, TrainingMetrics, VerifiedKnowledge, GGUFTrainingSystem

**Методы:** deploy_training_model, verify_training_model, initialize_training_model, _training_loop, _extract_verified_knowledge, _prepare_training_data, _train_separe_instance, _verify_training_quality

**Импорты:** os, logging, time, json, threading, hashlib, typing, dataclasses, enum

**Проблема в логике:** Метод _train_separate_instance() содержит только logger.info() вызовы, реальное обучение не выполняется. Верификация тоже заглушки - все проверки возвращают True.

---

### memory/document_manager.py

| Ошибка | Строки | Детали | Статус |
|--------|--------|--------|--------|
| ✅ 3 _findRelevantPages() заглушка | - | теперь использует semantic_search | **ИСПРАВЛЕНО** |
| 🔴 6 Эмбеддинги не вычисляются | - | поле есть но не используется | НЕ ИСПРАВЛЕНО |

**Классы:** DocumentPage, DocumentMetadata, LazyLoadingCache, DocumentChunker, DocumentVirtualMemory, DocumentAwareContextMixin

**Методы:** ingest_document, get_page, query_document, split_document, _find_relevant_pages, _calculate_hit_rate

**Импорты:** logging, hashlib, time, typing, dataclasses, collections.OrderedDict

**Проблема в логике:** Метод _find_relevant_pages() просто возвращает первые top_k страниц без семантического поиска. Эмбеддинги DocumentPage не вычисляются - поле существует но не инициализируется.

---

### mlearning/unified_fractal_manager.py

| Ошибка | Строки | Детали | Статус |
|--------|--------|--------|--------|
| ✅ 4 max_tokens undefined | - | max_tokens → max_new_tokens | **ИСПРАВЛЕНО** |

**Классы:** FractalModelManager, OptimizedFractalModelManager

**Методы:** generate_response, get_quality_metrics, improve_quality, start_enhanced_learning_session, generate_enhanced_response

**Импорты:** os, logging, typing, fractal_model_manager

**Проблема в логике:** В generate_enhanced_response используется max_tokens который не определён, должен быть max_new_tokens.

---

### mlearning/web_search_learning_integration.py

| Ошибка | Строки | Детали | Статус |
|--------|--------|--------|--------|
| ✅ 5 max_tokens undefined x2 | - | max_tokens → max_new_tokens | **ИСПРАВЛЕНО** |

**Классы:** WebSearchLearningIntegration

**Методы:** search_and_enhance_response, _should_search, _perform_web_search, _extract_key_information, _create_enhanced_prompt, _clean_enhanced_response, _analyze_response_quality

**Импорты:** os, logging, time, typing, concurrent.futures, web_search_engine

**Проблема в логике:** Две ошибки - generate_response и search_and_enhance_response используют max_tokens вместо max_new_tokens.

---

### learning/scheduler_core.py

| Ошибка | Строки | Детали |
|--------|--------|--------|
| 🔴 7 _execute_task() не реализован | - | только заглушка |

**Классы:** ResourceAllocation, LearningTask, LearningSchedulerCore

**Методы:** start, stop, _worker_loop, _load_tasks, _save_tasks, _execute_task

**Импорты:** os, logging, time, json, threading, heapq, typing, dataclasses

**Проблема в логике:** Метод _execute_task() содержит только pass, задачи не выполняются.

---

## 🟠 Высокий приоритет (11 штук)

### core/brain_query.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 21 needs_web_search слишком примитивен | теперь анализирует контекст | **ИСПРАВЛЕНО** |

---

### learning/dialog_concepts.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 9 Утечка памяти в _resolved_knowledge | добавлен MAX_RESOLVED_KNOWLEDGE=200 | **ИСПРАВЛЕНО** |
| ✅ 10 Рост очередей без ограничения | добавлены MAX_QUEUE лимиты | **ИСПРАВЛЕНО** |

**Классы:** DialogConceptsMixin

**Методы:** queue_concept_for_dialog, queue_contradiction_for_resolution, _run_concept_dialog, _run_contradiction_dialog, get_resolved_knowledge

**Импорты:** time, logging, re, dialog_types, pipeline_adapter

---

### knowledge/context_entity.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 15 Stub реализация | теперь wrapper над reasoning.entity_extractor | **ИСПРАВЛЕНО** |

---

### knowledge/knowledge_graph.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 16 Двойная абстракция | KG адаптер для GUI совместимости (нормально) | **НЕ ТРЕБУЕТ ИСПРАВЛЕНИЯ** |

---

### core/core_brain.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| 🟠 17 10 миксинов | нарушение SRP | Требует рефакторинга |

---

### core/brain_coordination.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 18 CommandIssuerMixin > 600 строк | вынесен в CommandHandlers класс | **ИСПРАВЛЕНО** |

---

## 🟡 Средний приоритет (17 штук)

### knowledge/concept_extractor.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 19 Шаблонные факты | теперь извлекает из текста | **ИСПРАВЛЕНО** |

---

### contradiction/contradiction_miner.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 20 Эвристика вместо NLI | теперь использует BART-large-mnli + fallback | **ИСПРАВЛЕНО** |

---

### memory/fractal_graph_v2

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 22 Дублирование semantic_search | API vs storage - разные уровни | **НЕ ТРЕБУЕТ ИСПРАВЛЕНИЯ** |
| ✅ 23 Дублирование токенизаторов | HybridTokenizer vs GraphTokenizer - разные цели | **НЕ ТРЕБУЕТ ИСПРАВЛЕНИЯ** |

---

### gui/web_gui/server_main.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 25 process_message > 740 строк | вынесены _prepare_file_context, _extract_reasoning_from_result | **ИСПРАВЛЕНО** |
| ✅ 26 Глобальная переменная | добавлены get_app() и get_brain() | **ИСПРАВЛЕНО** |

---

### gui/web_gui/server_routes.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 24 Дублирование /api/chat | не конфликтует (разные точки входа) | **ИСПРАВЛЕНО** |

---

### mlearning/unified_text_processor.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 28 Дублирование embedder | теперь один атрибут embedder | **ИСПРАВЛЕНО** |
| ✅ 29 use_async не работает | уже реализован (ThreadPoolExecutor) | **ИСПРАВЛЕНО** |

---

### knowledge/kg_adapter.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 30 Упрощённый поиск пути | теперь BFS алгоритм | **ИСПРАВЛЕНО** |

---

### knowledge/wikipedia_kb.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 31 Нет FAISS | добавлена опциональная поддержка FAISS | **ИСПРАВЛЕНО** |

**Классы:** WikipediaKnowledgeBase, WikipediaLoader

**Методы:** add_article, search, get_article, add_to_fractal_graph

---

### core/deferred_command_system.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 34 Глобальная переменная | добавлен threading.Lock | **ИСПРАВЛЕНО** |

**Классы:** DeferredCommandSystem

**Методы:** add_command, _process_commands, _execute_command, _schedule_retry

---

### contradiction/contradiction_manager.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 35 BaseComponent заглушка | теперь правильный импорт | **ИСПРАВЛЕНО** |

---

## 🟢 Низкий приоритет (6 штук)

### gui/web_gui/static/js/app.js

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 36 XHR + SSE смешение | XHR для POST streaming, EventSource для GET - разные цели | **НЕ ТРЕБУЕТ ИСПРАВЛЕНИЯ** |
| ✅ 37 EventSource не закрывается | браузер сам закрывает при unload | **НЕ ТРЕБУЕТ ИСПРАВЛЕНИЯ** |

---

### gui/widgets.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 39 Нет DPI awareness | добавлена поддержка DPI awareness | **ИСПРАВЛЕНО** |

---

### storage/fractal_storage.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 40 Импорты внутри методов | lazy imports для compression (нормально) | **НЕ ТРЕБУЕТ ИСПРАВЛЕНИЯ** |

---

### system/health_monitor.py

| Ошибка | Детали | Статус |
|--------|--------|--------|
| ✅ 41 Жёсткие веса | теперь настраиваются через конструктор | **ИСПРАВЛЕНО** |

---

## Итоговая таблица связей

| Модуль | Ошибок | Приоритет |
|--------|--------|-----------|
| core/unified_generator.py | 3 | 🔴 Критический |
| core/brain_query.py | 1 | 🟠 Высокий |
| core/core_brain.py | 1 | 🟠 Высокий |
| core/brain_coordination.py | 1 | 🟠 Высокий |
| core/deferred_command_system.py | 1 | 🟡 Средний |
| learning/dialog_concepts.py | 2 | 🟠 Высокий |
| learning/scheduler_core.py | 1 | 🔴 Критический |
| memory/document_manager.py | 3 | 🔴 Критический |
| memory/fractal_graph_v2 | 2 | 🟡 Средний |
| knowledge/concept_extractor.py | 1 | 🟡 Средний |
| knowledge/context_entity.py | 1 | 🟠 Высокий |
| knowledge/knowledge_graph.py | 1 | 🟠 Высокий |
| knowledge/kg_adapter.py | 1 | 🟡 Средний |
| knowledge/wikipedia_kb.py | 1 | 🟡 Средний |
| contradiction/contradiction_miner.py | 1 | 🟡 Средний |
| contradiction/contradiction_manager.py | 1 | 🟡 Средний |
| mlearning/unified_fractal_manager.py | 1 | 🔴 Критический |
| mlearning/web_search_learning_integration.py | 1 | 🔴 Критический |
| mlearning/unified_text_processor.py | 2 | 🟡 Средний |
| training/gguf_training_system.py | 3 | 🔴 Критический |
| gui/web_gui/server_main.py | 2 | 🟡 Средний |
| gui/web_gui/server_routes.py | 1 | 🟡 Средний |
| gui/widgets.py | 1 | 🟢 Низкий |
| storage/fractal_storage.py | 1 | 🟢 Низкий |
| system/health_monitor.py | 1 | 🟢 Низкий |

---

## Рекомендуемый порядок исправлений

### Фаза 1: Критические
1. core/unified_generator.py - параметры фабрики
2. mlearning/unified_fractal_manager.py - max_tokens
3. mlearning/web_search_learning_integration.py - max_tokens
4. memory/document_manager.py - поиск и эмбеддинги
5. training/gguf_training_system.py - реализовать обучение
6. learning/scheduler_core.py - реализовать _execute_task

### Фаза 2: Высокий приоритет
7. learning/dialog_concepts.py - утечка памяти
8. core/core_brain.py - рефакторинг миксинов
9. knowledge/ - улучшение stub

### Фаза 3: Средний приоритет
10. Дублирование в fractal_graph_v2
11. GUI рефакторинг
12. Очереди и лимиты
