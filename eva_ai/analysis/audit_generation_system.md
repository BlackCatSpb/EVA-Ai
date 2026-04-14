# Отчёт: Система генерации

## 1. Проверка импортов

### Статус: ВСЕ ИМПОРТЫ ВАЛИДНЫ

Проверены критические импорты:
- eva_ai.core.context_chunking - существует
- eva_ai.core.model_access_manager - существует
- eva_ai.core.openvino_generator - существует
- eva_ai.core.pie_model_paths - существует
- eva_ai.memory.fractal_graph_v2.dual_generator - существует

### Циклические зависимости
**Не обнаружены** в основном потоке импортов.

---

## 2. Соответствие методов документации

### ModelAccessManager.request_access()
**Документация:** Единственная точка входа для генерации текста, приоритизация CRITICAL для запросов пользователя

**Реальность:**
- Метод существует и использует PriorityQueue
- Но реализация УПРОЩЕНА: это просто обёртка над queue + callback execution
- **Нет вытеснения (preemption)** - запросы с более высоким приоритетом НЕ прерывают текущие
- Фактически работает как FIFO очередь с приоритетами, а не как единая точка входа

### OpenVINOGeneratorRegistry
**Документация:** Синглтон-реестр для шаринга модели на GPU

**Реализация:**
- get_or_create() работает но config_hash НЕ используется в ключе!
- Ключ = (model_path, device) без config_hash

### Lazy Loading (Ленивая загрузка)
**Документация:** Модели загружаются только при первом использовании

**Проблема:**
- В конструкторе с use_registry=True сразу вызывается _load_model()
- Фактически lazy loading РАБОТАЕТ только при use_registry=False

---

## 3. Детальный анализ ключевых компонентов

### 3.1 ModelAccessManager._process_loop()

**Выводы:**
1. Очередь с приоритетами - но НЕТ вытеснения текущей задачи
2. Если модель занята, запрос просто ждёт в очереди
3. Callback выполняется СИНХРОННО в _process_loop

### 3.2 SimpleRouter

**Логика:**
- CODER - если >= 2 кодовых ключевых слова
- CONTEXT - если >= 1 контекстных слов ИЛИ длина >= 25
- LOGIC - по умолчанию

**CONTEXT_KEYWORDS включает:** 'что такое', 'кто такой', 'как работает', 'объясни', 'расскажи' и т.д.

**Выводы:**
- Почти ЛЮБОЙ вопросительный запрос маршрутизируется в CONTEXT
- LOGIC модель будет использоваться редко

### 3.3 UnifiedGenerator._load_model()

**Проблемы:**
- n_gpu_layers=0 - ВСЕГДА CPU, даже если device указан GPU
- Это llama.cpp загрузка, НЕ OpenVINO

---

## 4. Выявленные проблемы и несоответствия

### Проблема 1: Дублирование генераторов
Существует 4+ генератора в системе:
1. UnifiedGenerator (unified_generator.py)
2. HybridPipelineAdapter (hybrid_pipeline_adapter.py)
3. DualGenerator (fractal_graph_v2/dual_generator.py)
4. RecursiveModelPipeline (recursive_model_pipeline.py)

### Проблема 2: OpenVINO lazy loading - непоследовательно
При use_registry=True модель загружается сразу в конструкторе.

### Проблема 3: GPU routing - жёсткий и упрощённый
Все задачи кроме coder/self_dialog идут на CPU.

### Проблема 4: ModelAccessManager без вытеснения
Высокоприоритетный запрос не прерывает низкоприоритетный.

### Проблема 5: Конфликт llama.cpp vs OpenVINO
- _load_model() использует llama.cpp
- _init_openvino_devices() использует OpenVINO GenAI

---

## 5. Оценка: что реально работает vs что упрощено

### РЕАЛЬНО РАБОТАЕТ
1. PriorityQueue для запросов - базовая сортировка по приоритету
2. Синглтон паттерн для ModelAccessManager и OpenVINOGeneratorRegistry
3. Базовый роутинг по ключевым словам
4. Lazy loading флаг
5. Multiple generation methods

### УПРОЩЕНО / ИМИТАЦИЯ
1. ModelAccessManager - это НЕ единая точка входа, а просто очередь с callback
2. GPU Model Sharing - реестр существует, но работает только если передать правильный creator_fn
3. Вытеснение (preemption) - НЕ реализовано
4. L2 Routing - заявлен как L2, но по факту просто keyword matching
5. OpenVINO Continuous Batching - упомянут, но не реализован

### НЕ СООТВЕТСТВУЕТ ДОКУМЕНТАЦИИ
1. Единая точка входа - ModelAccessManager не контролирует модели напрямую
2. Приоритизация с вытеснением - НЕ реализовано
3. L2 Роутинг - просто keyword matching

---

## 6. Итоговые рекомендации

### Критические проблемы:
1. Убрать дублирование генераторов или чётко разделить ответственность
2. Реализовать вытеснение в ModelAccessManager или убрать упоминание о нём
3. Исправить SimpleRouter или назвать его SimpleKeywordRouter

### Желательные улучшения:
1. Добавить config_hash в ключ реестра OpenVINOGeneratorRegistry
2. Унифицировать загрузку моделей
3. Реализовать динамический device selection

