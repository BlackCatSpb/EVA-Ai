# CogniFlex AI - Детальное Описание Системы

## Дата: 2026-03-29
Версия: 1.4 (исправления критических ошибок - AI Архитектор + 3 Девелопера + Тестировщик)

---

## 1. Поток инициализации в CoreBrain

Процесс инициализации системы CogniFlex представляет собой многоэтапный процесс последовательной активации компонентов, каждый из которых отвечает за определённую функциональность. Центральным классом системы является `CoreBrain`, расположенный в файле `cogniflex/core/core_brain.py`. Этот класс координирует работу всех остальных модулей и обеспечивает их взаимодействие друг с другом.

### 1.1 Конструктор CoreBrain.__init__

При создании экземпляра класса `CoreBrain` выполняется начальная настройка базовых параметров системы. Конструктор принимает опциональный параметр `config` — словарь конфигурации, который при отсутствии загружается из файла `brain_config.json` методом `_load_brain_config()`.

**ВАЖНО: Архитектура системы использует паттерн "ленивой загрузки" компонентов через ComponentInitializer.**

После загрузки конфигурации выполняется инициализация событийной системы и создаётся система отложенных команд через класс `DeferredCommandSystem`.

**Базовые компоненты инициализируемые напрямую в __init__:**

1. **ConfigManager** из `cogniflex/core/config_manager.py` — управление конфигурацией.

2. **SystemStateManager** из `cogniflex/core/system_state.py` — состояния системы.

3. **ResourceManager** из `cogniflex/core/resource_manager.py` — мониторинг ресурсов.

4. **SelfAnalyzer** из `cogniflex/learning/self_analyzer.py` — обнаружение возможностей для обучения.

5. **SystemMetricsManager** из `cogniflex/core/system_metrics.py` — сбор метрик.

6. **EnhancedSelfLearningSystem** из `cogniflex/core/enhanced_self_learning.py` — расширенное самообучение. Вызывается `start()` в конструкторе.

7. **MemoryGraphML** из `cogniflex/core/memory_graph_ml.py` — обучение на графе памяти.

8. **SelfDialogLearningSystem** из `cogniflex/learning/self_dialog_learning.py` — самообучение через диалог (ЗАПУСКАЕТСЯ ПОЗЖЕ в методе initialize()).

9. **QueryProcessor** из `cogniflex/core/query_processor.py`.

10. **ComponentInitializer** из `cogniflex/core/component_initializer.py` — **фабрика для ленивой загрузки компонентов**.

11. **HybridTokenCache** — синглтон через `get_shared_cache()`.

12. **FractalModelManager** — lazy loading.

13. **QwenModelManager** — lazy loading.

14. **BackgroundCoordinator** — фоновые задачи.

**Компоненты загружаемые через ComponentInitializer (в методе initialize()):**
- ml_unit — основной ML движок
- knowledge_graph — граф знаний  
- text_processor — обработка текста
- response_generator — генерация ответов
- ethics_framework — этическая проверка
- adaptation_manager — адаптация
- contradiction_manager — проверка противоречий
- web_search_engine — веб-поиск
- reasoning_engine — движок рассуждений

### 1.2 Метод CoreBrain.initialize()

Метод `initialize()` выполняет полную инициализацию всех компонентов системы и вызывается после создания экземпляра `CoreBrain`. Этот метод возвращает булево значение, indicating об успешности или неудаче процесса инициализации.

**Этапы инициализации:**

1. **`_initialize_detailed_logging()`** — логирование информации о системе: версия Python, платформа, количество процессоров, объём оперативной памяти, информация о доступности CUDA.

2. **`state_manager.set_state(SystemState.INITIALIZING, "Инициализация компонентов")`** — обновление состояния системы.

3. **`resource_manager.start_monitoring()`** — запуск мониторинга ресурсов.

4. **`metrics_manager.start_tracking()`** — начало отслеживания метрик.

5. **`component_initializer.initialize_components()`** — загрузка и активация всех зарегистрированных компонентов (модельный менеджер, текстовый процессор, генератор ответов, фреймворк этики).

6. **`_initialize_memory_manager()`** — обеспечение полной активации менеджера памяти.

7. **Установка ссылок на компоненты** — подключение model_manager, text_processor, response_generator к ядру системы.

8. **События `*_ready`** — через `events.trigger(f'{component_name}_ready', ...)` для memory_manager, text_processor, response_generator, ethics_framework.

9. **Инициализация FractalModelManager** — проверка наличия директории модели и необходимых файлов, установка флага `fractal_ready = True`.

10. **Инициализация GenerationCoordinator** — через фабричную функцию `initialize_generation_coordinator()`.

11. **Интеграция Self-Reasoning Engine** — через класс `ReasoningIntegration` и метод `integrate_with_brain()`. Результат сохраняется в `self.reasoning_integration` и `self.components['reasoning_integration']`.

12. **Запуск SelfDialogLearningSystem** — вызов метода `start()` для SelfDialogLearningSystem (ЭТО ДРУГАЯ СИСТЕМА, не EnhancedSelfLearningSystem!).

12. **`state_manager.set_state(SystemState.READY, "Инициализация завершена успешно")`** — переключение в состояние READY.

13. **`metrics_manager.record_system_startup(total_time)`** — фиксирование времени инициализации.

14. **Выполнение отложенных команд**, накопленных за время инициализации.

15. **Запуск SelfDialogLearningSystem** — вызов метода `start()` для фоновых процессов самообучения.

---

## 2. Поток обработки запроса

Обработка пользовательского запроса в системе CogniFlex представляет собой конвейерную архитектуру с несколькими этапами обработки, кэшированием промежуточных результатов и параллельным выполнением независимых задач. Центральным компонентом этого процесса является класс `QueryProcessor` из модуля `cogniflex/core/query_processor.py`.

### 2.1 Входная точка process_query()

Метод `process_query(query: str, user_context: Optional[Dict] = None)` принимает текстовый запрос пользователя и опциональный контекст. Возвращаемое значение — словарь со структурированным ответом, включающим текст ответа, источник данных, доказательства, метрики обработки и результаты проверок.

**При получении пустого запроса** система возвращает стандартное сообщение с запросом ввода текста.

### 2.2 Этап предварительной обработки NLP

Метод `_process_nlp(query: str)` выполняет лингвистический анализ входного текста:

1. **Проверка кэша** — используется хэш запроса в качестве ключа
2. **При отсутствии кэша** — запрос направляется в `ml_unit.process_text(query)`
3. **Результат** — извлечение ключевых слов, именованных сущностей, интента пользователя, сентиментальный анализ
4. **Сохранение в гибридный кэш** для последующего использования

**Это ОБЩИЙ анализ текста (морфология, сущности, интент).**

### 2.3 Извлечение концепта

Метод `_extract_concept(query: str)` идентифицирует ключевую тему или концепт запроса. При наличии `AdaptationManager` используется его метод `_extract_concept_from_query()`.

**Это ВЫДЕЛЕНИЕ ТЕМЫ для поиска в графе знаний. Это ДРУГАЯ функция, не дублирование!**

Оба метода используются последовательно:
- Строка 108: `nlp_info = self._process_nlp(query)` — анализ текста
- Строка 111: `concept = self._extract_concept(query)` — выделение темы

### 2.4 Поиск в графе знаний

Метод `_search_knowledge_graph(query: str, limit: int = 3)`:

1. Проверка кэша результатов поиска
2. Вызов `knowledge_graph.search_nodes()` / `search()` / `find_nodes()` / `query_nodes()`
3. При обнаружении результатов — переход к генерации ответа
4. При отсутствии — проверка `augment_with_web_on_kg` для веб-поиска

### 2.5 Параллельный поиск

Метод `_parallel_search(query: str)` использует `ThreadPoolExecutor` для одновременного выполнения:

- **Поиск в памяти** — `memory_manager.search_memories_by_entity()`
- **Веб-поиск** — `web_search_engine.search()`

### 2.6 Генерация ответа

Метод `_generate_response(query, evidence, nlp_info, concept, user_context)`:

1. Создание контекста генерации
2. Вызов `ml_unit.generate()` с передачей запроса и контекста
3. Извлечение результата по ключам `'text'` или `'generated_text'`

### 2.7 Проверка этичности

Метод `_check_ethics(response, nlp_info, user_context)`:

- При наличии `ethics_framework` — вызов `analyze_content()`
- Возвращает оценку этичности, список нарушений, рекомендации

### 2.8 Проверка на противоречия

Метод `_check_contradictions(query, response)`:

- При наличии `contradiction_resolver` — вызов `check_response_contradictions()` или `get_active_contradictions()`
- Добавление флага `contradiction_detected = True` при обнаружении

### 2.9 Завершающие операции

- Обновление метрик обработки
- Вызов `_detect_ambiguity()` для определения неоднозначностей
- Добавление рассуждений в результат через `_add_reasoning_to_result()`

---

## 3. Поток обучения TrainingOrchestrator

Система обучения в CogniFlex реализована через класс `TrainingOrchestrator` из модуля `cogniflex/mlearning/training_orchestrator.py`. Этот компонент координирует процесс обучения графа знаний из документов.

### 3.1 Инициализация TrainingOrchestrator

**Конструктор принимает параметры:**
- `brain` — ссылка на ядро системы
- `cache_dir` — директория кэша
- `batch_size` — размер батча (по умолчанию 16)
- `overlap_tokens` — перекрытие токенов (64)
- `max_retries` — максимум попыток (3)
- `backoff_sec` — задержка между попытками (2.0)
- `pipeline_version` — версия конвейера

**Метод `_try_init_components()`:**
- Извлечение `ml_unit` и `knowledge_graph` из brain
- Поиск токенизатора в: `brain.fractal_tokenizer`, `ml_unit.token_streamer`, `text_processor.tokenizer`, `model_manager.tokenizer`

### 3.2 Проверка готовности _can_train_now()

Проверяются следующие условия:
1. **Наличие токенизатора** — fractal_tokenizer OR ml_unit.token_streamer OR text_processor.tokenizer
2. **Доступность гибридного кэша** — hybrid_cache OR ml_unit.hybrid_cache OR brain.memory_manager.hybrid_cache
3. **Готовность моделей** — models_ready OR fractal_ready OR model_manager.models

### 3.3 Метод train_from_document()

```
train_from_document(imported_doc, model_id, use_fractal, fractal_config)
    │
    ├─> brain.clear_all_contradictions()
    │
    ├─> Загрузка/создание TrainingProgress
    │       ├─> Если новый документ: total_chunks = len(segments)
    │       └─> Если resume: загрузка из progress файла
    │
    ├─> Если use_fractal=True:
    │       └─> _train_with_fractal()
    │
    └─> Иначе: Стандартный конвейер
            │
            ├─> _enter_training_mode()
            │
            ├─> Цикл по батчам:
            │       │
            │       ├─> Создание батча (batch_size сегментов)
            │       │
            │       ├─> _process_batch() - обработка
            │       │       ├─> token_streamer - токенизация
            │       │       ├─> ml_unit - генерация эмбеддингов
            │       │       └─> knowledge_graph.add_node()/add_edge()
            │       │
            │       ├─> Обновление progress
            │       │       - processed_chunks
            │       │       - last_batch_end
            │       │       - last_success_ts
            │       │
            │       └─> progress.save() - сохранение чекпоинта
            │
            └─> _exit_training_mode()
```

### 3.4 Класс TrainingProgress

```python
TrainingProgress:
    - document_id: str
    - total_chunks: int
    - processed_chunks: int
    - last_batch_end: int
    - last_success_ts: float
    - model_id: str
    - pipeline_version: str
    - save(path) -> сохраняет в JSON
```

### 3.5 Управление ресурсами

```
При высокой памяти (>85%):
    batch_size = batch_size / 2

При критической памяти (>95%):
    Обучение приостанавливается
```

### 3.6 SelfDialogLearningSystem (Улучшения)

Самообучение через диалог с улучшениями:

```
SelfDialogLearningSystem:
    │
    ├─> min_quality_threshold: 0.9 (повышён с 0.6)
    │       Цель: достижение более высокого качества
    │
    ├─> max_dialog_turns: 15 (увеличено с 10)
    │       Больше итераций для глубокого анализа
    │
    ├─> auto_dialog_interval: 300 сек (5 минут)
    │       Интервал между автоматическими диалогами
    │
    ├─> _recently_processed_topics: Dict
    │       Хеш-таблица для исключения дубликатов
    │       TTL: 600 сек (10 минут)
    │
    └─> _get_conversation_context()
            Получение истории диалогов для контекста
```

**Интеграция с Knowledge Graph:**
```
_simulate_assistant_response(prompt, context)
    │
    ├─> Поиск релевантного контекста в knowledge_graph
    │
    ├─> Формирование расширенного промпта с контекстом
    │
    └─> Генерация ответа с учётом найденных знаний
```

**Триггер самодиалога:**
- Запускается после каждого запроса пользователя через веб-интерфейс
- Результат сохраняется в сессии и отображается в чате

---

## 4. Self-Reasoning система

Механизм самостоятельного рассуждения реализован в классе `SelfReasoningEngine` из модуля `cogniflex/reasoning/self_reasoning_engine.py`.

### 4.1 Архитектура SelfReasoningEngine

```
SelfReasoningEngine:
    - max_iterations: int = 5
    - confidence_threshold: float = 0.75
    - max_new_tokens: int = 2048
    - max_recursion_depth: int = 3
    
    Компоненты:
    - ClarificationGenerator - уточняющие вопросы
    - FractalStorage - хранение цепочек рассуждений
    - FractalEmbedder / FractalRetriever - семантический поиск
```

### 4.2 Основной цикл process_query()

```
process_query(query, user_context)
    │
    ├─> Проверка доступности Qwen модели
    │
    ├─> _is_complex_query() - определение сложности
    │       (содержит "и"/"или"/"но", >15 слов, причинно-следственные)
    │
    ├─> Если сложный запрос:
    │       └─> _recursive_process_query() - декомпозиция
    │
    └─> Иначе: основной цикл
            │
            └─> for iteration in range(max_iterations):
                    │
                    ├─> 1. _generate_with_qwen() - генерация ответа
                    │       │
                    │       └─> Fallback каскад:
                    │           - Qwen singleton
                    │           - FractalModelManager
                    │           - ResponseGenerator
                    │           - GenerationCoordinator
                    │
                    ├─> 2. _analyze_response() - анализ
                    │       │
                    │       ├─> ethics_framework.analyze_response()
                    │       ├─> contradiction_manager.detect_contradictions()
                    │       └─> knowledge_graph.search()
                    │
                    ├─> 3. calculate_overall_confidence() - расчёт уверенности
                    │
                    ├─> 4. should_terminate() - проверка завершения
                    │       │
                    │       └─> Если confidence >= 0.75: завершить
                    │
    └─> 5. Если НЕ завершён (confidence < 0.75) И есть итерации:
                └─> _generate_clarification() - уточняющий вопрос
                
**Логика:** Если should_terminate() возвращает True → break (выход из цикла). Иначе → проверка iteration < max_iterations → генерация уточнений.
```

### 4.3 Рекурсивная обработка _recursive_process_query()

```
_recursive_process_query(query, depth)
    │
    ├─> Если depth > max_recursion_depth:
    │       └─> Линейная обработка
    │
    ├─> _decompose_query() - декомпозиция на подзадачи
    │       │
    │       └─> Промпт модели: "Разбей запрос на 2-4 простых"
    │
    ├─> Для каждого подзапроса:
    │       └─> Рекурсивный вызов с depth+1
    │
    ├─> retrieve_similar_reasoning() - поиск похожих рассуждений
    │       │
    │       └─> FractalRetriever - семантический поиск
    │
    └─> _synthesize_recursive_results() - синтез результатов
            │
            └─> Промпт модели с объединением подответов
```

### 4.4 Обратная связь и самообучение

```
process_user_feedback(query, feedback, rating)
    │
    └─> Если rating < 0.3:
            └─> _trigger_self_learning()

self_correct(query, correction)
    │
    ├─> Формирование промпта с коррекцией
    ├─> Генерация исправленного ответа
    └─> Сохранение в FractalStorage

learn_from_outcome(rating)
    │
    ├─> Если rating < 0.3: понизить порог уверенности
    └─> Если rating > 0.7: повысить порог уверенности
```

### 4.5 Логические факторы рассуждения

Система анализирует каждый ответ по 5 логическим факторам:

```
LOGICAL_FACTORS:
    │
    ├─> ethics (Этика)
    │       weight: 0.2
    │       Оценка: соответствие этическим нормам
    │       Источник: ethics_checker
    │
    ├─> knowledge (Знания)
    │       weight: 0.25
    │       Оценка: фактическая точность информации
    │       Источник: knowledge_graph.search()
    │
    ├─> contradiction (Противоречия)
    │       weight: 0.2
    │       Оценка: отсутствие внутренних противоречий
    │       Источник: contradiction_manager
    │
    ├─> context (Контекст)
    │       weight: 0.15
    │       Оценка: учёт контекста запроса
    │       Источник: анализ ключевых слов, история диалога
    │
    └─> logic (Логика)
            weight: 0.2
            Оценка: логическая согласованность
            Источник: анализ структуры ответа
```

**Методы анализа:**
- `_analyze_logical_factors()` - общий анализ всех факторов
- `_evaluate_ethics_factor()` - оценка этического фактора
- `_evaluate_knowledge_factor()` - оценка фактора знаний
- `_evaluate_contradiction_factor()` - оценка противоречий
- `_evaluate_context_factor()` - оценка контекста
- `_evaluate_logic_factor()` - оценка логики

### 4.6 Ветвление рассуждений

При низких оценках факторов система генерирует альтернативные ветви рассуждения:

```
_should_use_alternative_branch(factors_result)
    │
    └─> Условие: ЕСЛИ (оценка фактора < 0.7) ИЛИ (общая оценка < 0.8)
            │
            └─> Возвращает: True → создать альтернативную ветвь
```

```
_find_alternative_reasoning_branches()
    │
    ├─> Найти слабые факторы (оценка < 0.7)
    │
    ├─> Для каждого слабого фактора:
    │       │
    │       └─> Сгенерировать промпт для альтернативы:
    │               │
    │               ├─> ethics → "Переформулируй этически корректно"
    │               ├─> knowledge → "Проверь факты и дополни"
    │               ├─> contradiction → "Устрани противоречия"
    │               ├─> context → "Учти контекст запроса"
    │               └─> logic → "Исправь логические ошибки"
    │
    └─> Вернуть список альтернатив (максимум 3)
```

```
_merge_reasoning_branches()
    │
    └─> Объединить основной ответ с альтернативными вариантами
```

### 4.7 Контекст диалога в рассуждении

```
_build_contextual_query(query, conversation_history)
    │
    ├─> Получить последние 10 сообщений истории
    │
    ├─> Формировать расширенный промпт:
    │   "Предыдущий контекст разговора:
    │    Пользователь: [сообщение 1]
    │    Ассистент: [сообщение 2]
    │    ...
    │    Текущий вопрос: [query]"
    │
    └─> Вернуть расширенный промпт для генерации
```

---

## 5. Работа с памятью

Система управления памятью в CogniFlex состоит из двух основных компонентов: `MemoryManager` и `KnowledgeGraph`.

### 5.1 Архитектура MemoryManager

```
MemoryManager:
    │
    ├─> working_memory: Dict - текущие операции
    │       (временное хранилище, быстрый доступ)
    │
    ├─> semantic_memory: Dict - структурированные знания
    │       (факты, концепции, связи)
    │
    ├─> episodic_memory: List - история событий
    │       (хронологический порядок)
    │
    ├─> user_profiles: Dict - профили пользователей
    │
    └─> hybrid_cache: HybridTokenCache - кэш токенов
```

### 5.2 Операции

```
add_memory(memory_type, content, metadata, user_id)
    │
    ├─> Валидация типа памяти
    ├─> Валидация размера (max 100000 символов)
    ├─> Генерация ID: mem_{timestamp}_{random_hex}
    ├─> Сохранение в память
    └─> _save_memory() - запись на диск

get_memory(memory_id)
    │
    └─> Поиск по всем типам памяти

get_conversation_history(user_id, limit)
    │
    └─> episodic_memory[-limit:] с фильтрацией по user_id

search_memories_by_entity(entity_term)
    │
    └─> Поиск по семантической памяти
```

### 5.3 KnowledgeGraph структура

**Основные классы:**

- **KnowledgeNode** — узел знаний с идентификатором, именем, описанием, типом, доменом, силой связи, временными метками, метаданными
- **KnowledgeEdge** — связь между узлами с исходным и целевым узлом, типом отношения, силой связи
- **NodeType** — перечисление: концепт, сущность, факт, событие, процесс
- **RelationType** — перечисление: is_a, part_of, has_property, causes, supports, contradicts, similar_to

### 5.4 Инициализация графа знаний

1. Инициализация интеграционных компонентов (гибридный кэш, текстовый процессор, MLUnit, экстрактор сущностей)
2. Создание базы данных SQLite для персистентного хранения
3. Загрузка существующих узлов и связей из БД
4. Инициализация индексов: по доменам, типам узлов, типам связей, временным меткам
5. Запуск фоновых служб мониторинга целостности

### 5.5 Операции с узлами и связями

- **add_node()** — создание нового узла с уникальным ID на основе хэша содержимого
- **add_edge()** — создание связи между существующими узлами
- **update_node() / update_edge()** — обновление с сохранением истории изменений
- **search_nodes(query, limit)** — поиск по совпадению в имени, описании, ключевым словам

### 5.6 Проверка целостности и противоречий

- **_check_graph_integrity()** — проверка отсутствующих узлов в связях, обнаружение дубликатов
- **_check_for_contradictions()** — анализ связей типа contradicts

---

## 6. Graphical User Interface (GUI)

### 6.1 Структура (Tkinter)

```
CogniFlexGUI (Tkinter)
    │
    ├─> ChatModule - основной чат
    │       ├─> input_text - поле ввода
    │       ├─> chat_display - отображение сообщений
    │       ├─> send_button - кнопка отправки
    │       └─> Events: message_sent, response_received
    │
    ├─> MemoryTab - просмотр памяти
    │
    ├─> KnowledgeGraphModule - визуализация графа
    │
    └─> SystemTab - системная информация
```

### 6.2 Поток обработки сообщения

```
_send_message()
    │
    ├─> Получение текста из input_text
    ├─> Добавление в чат (роль: user)
    ├─> Очистка input_text
    │
    └─> В фоне:
            │
            ├─> brain.process_query(query, context)
            │
            └─> _display_response(response)
                    ├─> Добавление в чат (роль: assistant)
                    ├─> Обновление метрик
                    └─> Отображение рассуждений если есть
```

### 6.3 Status Bar

```
Метрики:
- CPU: cpu_usage_percent
- RAM: memory_usage_percent  
- HitRate: cache_hit_rate
- CacheUtil: cache_utilization_percent
- DiskEntries: disk_entries_count
- IOtokens: io_tokens_formatted
- Противоречия: contradiction_count
```

---

## 6A. Веб-интерфейс (Web GUI)

Веб-интерфейс CogniFlex реализован на Flask и доступен по адресу `http://127.0.0.1:5555`.

### 6A.1 Структура

```
WebGUI (Flask)
    │
    ├─> server.py - Flask сервер
    │       ├─> / - главная страница (index.html)
    │       ├─> /api/login - аутентификация
    │       ├─> /api/sessions - управление сессиями
    │       ├─> /api/session/<id> - загрузка сессии
    │       ├─> /api/chat - обработка сообщений
    │       ├─> /api/upload - загрузка файлов
    │       ├─> /api/status - статус системы
    │       ├─> /api/metrics - метрики
    │       ├─> /api/memory-graph - данные графа памяти
    │       ├─> /api/analytics - аналитика
    │       └─> /api/learning - обучение
    │
    ├─> templates/index.html - HTML шаблон
    │
    ├─> static/js/app.js - клиентский JavaScript
    │
    └─> static/css/style.css - стили
```

### 6A.2 Аутентификация

- Логин: `admin`
- Пароль: `cogniflex`
- Сессии сохраняются в файл `sessions.json` в папке `cogniflex_gui_cache/`

### 6A.3 Загрузка файлов

Поддерживаемые форматы:
- PDF (через PyMuPDF, pdfplumber, PyPDF2)
- DOCX (через python-docx)
- TXT и текстовые файлы
- Изображения (jpg, png, gif, bmp)
- Кодовые файлы (py, js, ts, java, cpp, c, h, html, css, json, xml, yaml, md)

**OCR (Tesseract):**
- Путь: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Языки: English (eng), Russian (rus)
- Автоматическое распознавание текста с изображений и сканов PDF

### 6A.4 QWEN-style popup меню

При выделении текста в чате появляется контекстное меню с 5 кнопками:
- **Копировать** - копирует выделенный текст в буфер обмена
- **Спросить** - добавляет выделенный текст в поле ввода с вопросом
- **Объяснить** - добавляет запрос на объяснение
- **Перевести** - добавляет запрос на перевод
- **Переписать** - добавляет запрос на перефразирование

### 6A.5 Страницы интерфейса

1. **Чат (Chat)** - основной интерфейс общения с AI
2. **Память (Memory)** - просмотр узлов и связей графа знаний
3. **Аналитика (Analytics)** - метрики системы:
   - Производительность (запросы, время, успешность)
   - Система (CPU, RAM, VRAM)
   - Обучение (диалоги, пробелы, изучено)
   - История активности
4. **Обучение (Learning)** - возможности самообучения:
   - Список возможностей с приоритетами
   - Статистика обучения
   - Недавние диалоги
5. **Настройки (Settings)**:
   - Автообучение (вкл/выкл)
   - SRE (Reasoning) (вкл/выкл)
   - Контекстная память (вкл/выкл)
   - Тёмная тема (вкл/выкл)
   - Звуковые уведомления (вкл/выкл)

### 6A.6 Отображение рассуждений

При отправке сообщения отображается блок "🤔 Рассуждение" с:
- Анимированной индикацией "думания"
- Списком шагов рассуждения:
  - 💭 generation - генерация ответа
  - 🔍 analysis - анализ
  - ⚖️ logical_analysis - анализ логических факторов
  - ❓ clarification - уточнения
  - 🔄 alternative_branch - альтернативные ветви
  - ✅ final_synthesis - финальный синтез
- Индикацией уверенности (цветовая: зелёный >0.8, жёлтый >0.5, красный <0.5)

### 6A.7 Отображение самодиалога

После каждого запроса запускается самодиалог. Результат показывается в свернутом блоке:
- Тема диалога
- Исход (successful/needs_improvement)
- Выявленные пробелы знаний
- Выполненные действия

---

## 7. Событийная система (EventBus)

### 7.1 Типы событий

```
Компонентные события:
- memory_manager_ready
- text_processor_ready
- response_generator_ready
- ethics_framework_ready
- fractal_model_ready

Системные события:
- system_startup
- system_shutdown
- component_initialized
- training_start
- training_progress
- training_complete
- query_received
- query_processed

Обучение:
- learning_opportunity_detected
- self_dialog_created
- self_dialog_completed
```

### 7.2 Подписка и публикация

```
events.subscribe(event_type, handler)
    └─> Добавление в список обработчиков

events.trigger(event_type, data)
    └─> Вызов всех обработчиков с data

events.unsubscribe(event_type, handler)
    └─> Удаление из списка
```

---

## 8. Фоновые задачи (BackgroundCoordinator)

### 8.1 Типы задач

```
TrainingJob - обучение на документах
WebIndexJob - индексация веб-контента
ModuleRecoveryJob - восстановление компонентов
```

### 8.2 Адаптивное планирование

```
autopilot_idle_threshold_s: 300 (5 минут)
autopilot_cpu_soft: 70%
autopilot_cpu_hard: 90%

При простое > threshold:
    - Постепенное замедление (exponential backoff)
    
При CPU > hard:
    - Приостановка задач
    
При CPU > soft:
    - Уменьшение приоритета
```

---

## 9. Конфигурация

### 9.1 brain_config.json

```json
{
  "model": {
    "name": "qwen3.5-0.8b",
    "temperature": 0.7,
    "max_new_tokens": 2048,
    "top_p": 0.9,
    "repetition_penalty": 1.1
  },
  "cache": {
    "max_vram_mb": 12000,
    "max_ram_mb": 8000,
    "disk_path": "cogniflex_cache"
  },
  "learning": {
    "enabled": true,
    "self_dialog_interval": 300,
    "min_quality_threshold": 0.9,
    "max_dialog_turns": 15,
    "auto_learning_interval": 60
  },
  "ethics": {
    "enabled": true,
    "threshold": 0.8
  }
}
```

---

## 10. Диагностика и метрики

### 10.1 Ключевые метрики

```
Системные:
- system.startup_time - время запуска
- system.memory_usage - использование памяти
- system.cpu_usage - использование CPU

Запросы:
- query.latency - время обработки запроса
- query.success_rate - процент успешных запросов
- query.cache_hit_rate - попадание в кэш

Обучение:
- training.documents_started - начато документов
- training.total_chunks - всего чанков
- training.processed_chunks - обработано чанков
- training.failures - ошибки обучения
```

### 10.2 Состояния системы

```
SystemState:
- INITIALIZING - инициализация
- READY - готова к работе
- ERROR - ошибка
- OFFLINE - офлайн
- SHUTTING_DOWN - выключение
- MAINTENANCE - обслуживание
- DEGRADED - ограниченная функциональность
```

---

## 11. Интеграция компонентов

### 11.1 Взаимодействие MemoryManager и KnowledgeGraph

MemoryManager и KnowledgeGraph интегрированы через brain. При инициализации MemoryManager получает ссылку на knowledge_graph и может выполнять взаимный поиск:

- Результаты поиска в графе знаний используются как доказательства при генерации ответов
- Записи из памяти могут добавляться в граф как новые узлы знаний

### 11.2 Поток данных при обработке запроса

```
Пользовательский запрос
        │
        ▼
QueryProcessor.process_query()
        │
        ├─> MemoryManager.get_conversation_history()
        │
        ├─> KnowledgeGraph.search_nodes()
        │
        ├─> SelfReasoningEngine.process_query()
        │        │
        │        ├─> EthicsFramework.analyze_content()
        │        ├─> ContradictionManager.detect_contradictions()
        │        └─> KnowledgeGraph.search()
        │
        ├─> MLUnit.generate()
        │
        └─> MemoryManager.add_interaction()
```

---

## 12. Исправления v1.4 (2026-03-29)

Система прошла аудит AI Архитектором, 3 AI Девелоперами и AI Тестировщиком.

### 12.1 Исправленные ошибки HIGH

| # | Файл | Строка | Проблема | Статус |
|---|------|--------|----------|--------|
| 1 | memory_manager.py | 272 | semantic_memory итерируется как dict вместо list | ✅ |
| 2 | memory_manager.py | 397 | remove_node() сломан для dict-типов памяти | ✅ |
| 3 | core_brain.py | 1033 | Отсутствует get_session_context() | ✅ |
| 4 | server.py | 139 | UUID меняется каждый логин → сессии не персистентны | ✅ |
| 5 | knowledge_graph.py | - | Отсутствует метод search() | ✅ |
| 6 | self_reasoning_engine.py | 412 | Циклическая структура данных в _analyze_logical_factors | ✅ |
| 7 | self_reasoning_engine.py | 529 | Пустое тело проверки в _evaluate_logic_factor | ✅ |

### 12.2 Добавленные методы

```
MemoryManager:
    ├─> get_session_context(session_id) - контекст сессии из эпизодической памяти
    ├─> get_graph_data() - данные графа для визуализации GUI
    └─> remove_node() - исправлен для dict-типов

KnowledgeGraph:
    └─> search(query, limit=5) - алиас для search_nodes()

AuthManager (server.py):
    └─> user_id = hashlib.md5(username.encode()).hexdigest()
```

### 12.3 Архитектурные улучшения

1. **Fallback для reasoning**: process_query() теперь проверяет reasoning_integration если self_reasoning_engine=None
2. **Greeting handler**: пропускает обработку при наличии "прикрепил файл" в запросе
3. **Персистентные сессии**: user_id теперь детерминированный на основе имени пользователя

---

## 13. Заключение

Система CogniFlex AI представляет собой комплексную архитектуру с чётким разделением ответственности между компонентами:

1. **Поток инициализации** обеспечивает последовательную активацию всех модулей с обработкой зависимостей и отложенной загрузкой тяжёлых компонентов.

2. **Обработка запросов** реализует конвейерную архитектуру с параллельным поиском, кэшированием и многоуровневыми проверками качества.

3. **Система обучения** обеспечивает устойчивость к сбоям через контрольные точки и адаптивное управление ресурсами.

4. **Механизм самостоятельного рассуждения** позволяет системе итеративно улучшать качество ответов через анализ и уточнение.

5. **Управление памятью** реализует многоуровневую архитектуру с персистентным хранением и эффективным кэшированием.

---

## Версии документа

| Версия | Дата | Описание |
|--------|------|----------|
| 1.0 | 2026-03-27 | Начальная версия |
| 1.1 | 2026-03-27 | Расширенная документация |
| 1.2 | 2026-03-29 | Исправления багов (greeting handler, self-dialog loop) |
| 1.3 | 2026-03-29 | Веб-интерфейс (Flask), загрузка файлов, OCR, логические факторы рассуждения |
| 1.4 | 2026-03-29 | Аудит AI Архитектора: 7 критических ошибок, 4 новых метода, архитектурные улучшения |
