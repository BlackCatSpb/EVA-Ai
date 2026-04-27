# Анализ Core Generation EVA

## 1. Компоненты системы генерации

### 1.1 brain_query.py (QueryMixin)
**Назначение:** Обработка пользовательских запросов с многоуровневым fallback

**Основные атрибуты:**
- two_model_pipeline - основной генератор (GGUF)
- fcp_pipeline - FCP Pipeline V15
- self_dialog_learning - самодиалог
- concept_extractor - извлечение концептов
- web_search_engine - веб-поиск (Tavily)
- fractal_graph_v2 - граф знаний

**Ключевые константы:**
- FG_ONLY_MODE = False - режим только FractalGraph
- FALLBACK_RESPONSES - запасные ответы
- GREETING_RESPONSES - быстрые ответы на приветствия

---

### 1.2 hybrid_pipeline_adapter.py (HybridPipelineAdapter)
**Назначение:** Гибридный адаптер с переключением между режимами генерации

**Режимы работы:**
- MODE_FRACTAL = fractal - Только FractalPipeline
- MODE_DUAL = dual - DualGenerator (2 модели)
- MODE_RECURSIVE = recursive - RecursiveModelPipeline
- MODE_HYBRID = hybrid - Fractal + fallback
- MODE_FMF_ONLY = fmf_only - Только FMF

**Основные атрибуты:**
- fractal_pipeline - FractalPipeline
- dual_generator - DualGenerator
- recursive_pipeline - RecursiveModelPipeline
- _model_access - ModelAccessManager

---

### 1.3 fcp_pipeline.py (FCPPipelineV15)
**Назначение:** Основной FCP Pipeline с KCA (Knowledge Conscious Attention) и SRG (Semantic Relevance Gate)

**Основные компоненты:**
- fcp_config - FCPConfig
- fractal_graph - FractalGraphV2
- kca - KnowledgeConsciousAttention
- srg - SemanticRelevanceGate
- convergence_controller - ConvergenceController
- hybrid_layer_manager - HybridLayerManager
- hybrid_processor - HybridLayerProcessor
- memory_snapshot - MemorySnapshotIntegration

**Параметры генерации по умолчанию:**
- max_new_tokens: 4096
- temperature: 0.2
- top_p: 0.9
- top_k: 40
- repetition_penalty: 1.1

---

### 1.4 model_access_manager.py (ModelAccessManager)
**Назначение:** Координация доступа к модели с приоритизацией

**Приоритеты:**
- CRITICAL = 0 - Пользовательские запросы
- HIGH = 1 - Самодиалог (концепты/противоречия)
- NORMAL = 2 - Фоновые задачи
- LOW = 3 - Долгосрочное обучение

**Основные атрибуты:**
- request_queue - PriorityQueue запросов
- active_requests - активные запросы
- _access_lock - блокировка доступа
- _model_busy - флаг занятости модели

---

## 2. Поток данных (Data Flow)

### 2.1 Основной поток генерации

User Query
  |
  v
process_query() [brain_query.py:177]
  |
  +-> Проверка приветствия (GREETING_RESPONSES)
  +-> Проверка кэша (_query_cache)
  |
  v
_execute_query_strategy() [brain_query.py:308]
  |
  +-> FG_ONLY_MODE = False (отключено)
  +-> qwen_only_mode = False
  +-> disable_pytorch = True (GGUF mode)
  |
  v
_handle_gguf_pipeline() [brain_query.py:324]
  |
  +-> Проверка FCPPipelineV15 (fcp_pipeline)
  |     |
  |     v
  |     pipeline.generate() [fcp_pipeline.py:458]
  |       |
  |       +-> _build_prompt() - форматирование с историей
  |       +-> _generate() - генерация через OpenVINO
  |       +-> conversation_history - сохранение в историю
  |
  +-> Fallback: two_model_pipeline.process_query()
        |
        +-> ModelAccessManager.request_access()
        +-> Веб-поиск (Tavily) если нужен
        +-> Контекст концептов (self_dialog_learning)
        +-> generation_tracker - отслеживание прогресса

### 2.2 Поток извлечения концептов

process_query() result
  |
  v
_extract_key_concepts() [brain_query.py:1118]
  |
  +-> concept_extractor.extract_concepts(query, response)
  |     |
  |     +-> save_concept_to_graph() -> FGv2
  |
  +-> self_dialog_learning.queue_concept_for_dialog()
  |     |
  |     +-> trigger_self_dialog() - запуск самодиалога
  |
  +-> ClosedCognitiveLoop.update() - обновление цикла

### 2.3 Поток через HybridPipelineAdapter

process_query() [hybrid_pipeline_adapter.py:388]
  |
  +-> MODE_DUAL -> _process_dual()
  |     +-> dual_generator.generate() + _model_access
  |
  +-> MODE_FRACTAL -> _process_fractal()
  |     +-> fractal_pipeline.process_query()
  |
  +-> MODE_RECURSIVE -> _process_recursive()
  |     +-> recursive_pipeline.process_query()
  |
  +-> MODE_FMF_ONLY -> _process_fmf()
        +-> fmf_pipeline.process_query()

---

## 3. Методы

### 3.1 brain_query.py - Методы QueryMixin

| Метод | Строка | Назначение | Используется |
|-------|--------|------------|--------------|
| process_query | 177 | Основная точка входа | YES Основной |
| _execute_query_strategy | 308 | Диспетчер стратегий | YES Вызывается из process_query |
| _handle_gguf_pipeline | 324 | GGUF генерация | YES Основной |
| _handle_fg_only | 462 | FractalGraph only | NO FG_ONLY_MODE=False |
| _handle_qwen_mode | 612 | Qwen-only режим | NO qwen_only_mode=False |
| _handle_llama_cpp | 805 | LlamaCpp генерация | RARE Fallback |
| _handle_fallback | 954 | Общий fallback | YES Fallback |
| _check_proactive_fallback | 155 | Проверка fallback | YES Используется |
| _update_fallback_state | 166 | Обновление состояния | YES Используется |
| _extract_key_concepts | 1118 | Извлечение концептов | YES После генерации |
| _save_to_fractal_graph | 1097 | Сохранение в FG | YES Используется |
| _generate_template_response | 570 | Шаблонные ответы | NO Только для FG_ONLY_MODE |
| _generate_with_timeout | 1034 | Генерация с таймаутом | YES Используется |
| _format_reasoning_for_gui | 1055 | Форматирование для GUI | UNKNOWN |

### 3.2 hybrid_pipeline_adapter.py - Методы

| Метод | Строка | Назначение | Используется |
|-------|--------|------------|--------------|
| process_query | 388 | Основной API | YES Вызывается из brain_query |
| _process_fractal | 427 | FractalPipeline обработка | YES MODE_FRACTAL |
| _process_dual | 474 | DualGenerator обработка | YES MODE_DUAL |
| _process_recursive | 606 | Recursive обработка | YES MODE_RECURSIVE |
| _process_fmf | 452 | FMF обработка | YES MODE_FMF_ONLY |
| _process_hybrid | 627 | Гибридная обработка | YES MODE_HYBRID |
| generate_with_virtual_tokens | 716 | Виртуальные токены | UNKNOWN |
| load_models | 168 | Загрузка моделей | YES При инициализации |
| unload_models | 209 | Выгрузка моделей | RARE |
| set_mode | 666 | Смена режима | RARE |
| get_stats | 686 | Статистика | NO |

### 3.3 fcp_pipeline.py - Методы FCPPipelineV15

| Метод | Строка | Назначение | Используется |
|-------|--------|------------|--------------|
| generate | 458 | Основная генерация | YES Вызывается из brain_query |
| generate_streaming | 315 | Streaming генерация | RARE |
| _build_prompt | 494 | Формирование промпта | YES generate() |
| _generate | 503 | Вызов OpenVINO | YES generate() |
| load_lora_adapter | 521 | Загрузка LoRA | NO Не вызывается |
| enrich_with_kca | 568 | KCA обогащение | NO Не используется |
| get_fcp_status | 535 | Статус FCP | UNKNOWN API |
| get_statistics | 532 | Статистика | UNKNOWN API |

### 3.4 model_access_manager.py - Методы

| Метод | Строка | Назначение | Используется |
|-------|--------|------------|--------------|
| request_access | 160 | Запрос доступа | YES GGUF pipeline |
| get_result | 219 | Получить результат | YES request_access |
| _process_loop | 257 | Обработка очереди | YES В фоновом потоке |
| _on_model_request | 118 | Обработка события | YES EventBus |
| _on_model_release | 131 | Освобождение модели | YES EventBus |
| _on_model_status | 139 | Статус (пустой) | NO Заглушка |
| get_status | 341 | Статус менеджера | UNKNOWN API |
| start | 143 | Запуск менеджера | YES Инициализация |
| stop | 153 | Остановка менеджера | RARE |

---

## 4. Заглушки (Stubs)

### 4.1 brain_query.py

| Строка | Код | Описание |
|--------|-----|----------|
| 940 | pass | Пустой except в _handle_llama_cpp при запуске self-dialog |

### 4.2 model_access_manager.py

| Строка | Код | Описание |
|--------|-----|----------|
| 141 | pass | Пустой метод _on_model_status - не реализована обработка статуса |

---

## 5. Мертвые методы

### 5.1 Методы с постоянным отключением

| Метод | Файл | Причина |
|-------|------|---------|
| _handle_fg_only | brain_query.py | FG_ONLY_MODE = False - постоянно отключено |
| _generate_template_response | brain_query.py | Вызывается только из _handle_fg_only |
| _handle_qwen_mode | brain_query.py | Используется только при qwen_only_mode=True |
| _on_model_status | model_access_manager.py | Пустой метод (заглушка) |

### 5.2 Методы которые потенциально мертвые

| Метод | Файл | Причина |
|-------|------|---------|
| load_lora_adapter | fcp_pipeline.py | Не вызывается в основном потоке |
| enrich_with_kca | fcp_pipeline.py | Не вызывается в generate(), только для внешнего API |
| generate_with_virtual_tokens | hybrid_pipeline_adapter.py | Не вызывается, сложная функциональность |
| ModelAccessContext | model_access_manager.py | Определен но не используется в системе |
| get_stats | hybrid_pipeline_adapter.py | Не используется внешним кодом |

### 5.3 Неиспользуемые константы

| Константа | Файл | Причина |
|-----------|------|---------|
| FALLBACK_RESPONSES | brain_query.py | Определены но не используются напрямую |
| FALLBACK_RESPONSE_DEFAULT | brain_query.py | Не используется |
| FG_ONLY_MODE | brain_query.py | Всегда False |

---

## 6. Выводы

### 6.1 Архитектура

Система генерации построена на следующих принципах:
1. Многоуровневый fallback: process_query -> _execute_query_strategy -> _handle_gguf_pipeline -> _handle_fallback
2. Приоритет FCPPipelineV15: Если доступен - используется напрямую
3. Координация через ModelAccessManager: Предотвращает конфликты доступа
4. Извлечение концептов после генерации: Автоматическое обучение

### 6.2 Активные компоненты

| Компонент | Роль | Статус |
|-----------|------|--------|
| process_query | Main entry | ACTIVE |
| fcp_pipeline.generate | Основная генерация | ACTIVE |
| two_model_pipeline | Fallback генератор | ACTIVE |
| _extract_key_concepts | Обучение | ACTIVE |
| ModelAccessManager | Координация | ACTIVE |

### 6.3 Проблемные области

1. Сложная запутанность: Много fallback-ов делает код сложным для отладки
2. Неиспользуемые режимы: FG_ONLY_MODE, qwen_only_mode постоянно отключены
3. Дублирование функций: Несколько методов делают одно и то же
4. Недореализованные методы: load_lora_adapter, enrich_with_kca не интегрированы
5. Заглушки: _on_model_status пустой, ModelAccessContext не используется

### 6.4 Рекомендации

1. Упростить fallback логику: Убрать неиспользуемые режимы
2. Интегрировать LoRA: Использовать load_lora_adapter в основном потоке
3. Удалить мертвый код: _handle_fg_only, _generate_template_response, ModelAccessContext
4. Документировать поток: Добавить комментарии для понимания fallback-ов
5. Вынести веб-поиск: Извлечь Tavily логику в отдельный компонент

---

*Анализ проведен: 2026-04-27*
*Версия EVA: Current*
