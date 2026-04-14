# Отчёт: Reasoning

## 1. Структура

### Директория eva_ai/reasoning/

`
eva_ai/reasoning/
|-- __init__.py                      # Экспорты модулей
|-- self_reasoning_engine.py          # Основной движок (SelfReasoningEngine)
|-- enhanced_reasoning_engine.py      # Расширенный движок с регенерацией
|-- reasoning_types.py                # Типы данных (ReasoningStep, ReasoningResult)
|-- integration.py                    # Интеграция с CoreBrain
|
|-- # Вспомогательные модули
|-- confidence_scorer.py              # Оценка уверенности
|-- clarification_generator.py        # Генерация уточняющих вопросов
|-- analytics_module.py               # Анализ текста (сущности, логические блоки)
|-- prompt_composer.py               # Компоновка промптов для регенерации
|-- semantic_stability.py             # Проверка стабильности ответов
|-- combined_metric.py                # Метрика улучшения
|-- entity_extractor.py               # Извлечение сущностей для самообучения
|-- correlation_calculator.py         # Проверка корреляции ответов
|
|-- # SRE подмодули (методы движка)
|-- sre_context.py                    # Контекст, Wikipedia, параметры генерации
|-- sre_quality.py                    # Проверка качества ответа
|-- sre_recursive.py                  # Рекурсивная обработка сложных запросов
|-- sre_feedback.py                   # Feedback система
|
+-- fractal_ml/                       # Фрактальное хранилище
    |-- __init__.py
    |-- fractal_base.py               # Базовые классы (FractalNode, FractalEdge)
    |-- fractal_storage.py             # Хранилище с иерархической структурой
    |-- fractal_retriever.py           # Поиск по хранилищу
    |-- fractal_embedder.py            # Эмбеддинги
    +-- fractal_tokenizer.py           # Токенизатор
`

### Ключевые классы

| Класс | Назначение |
|-------|-----------|
| SelfReasoningEngine | Основной движок с циклом Generate -> Analyze -> Clarify -> Repeat |
| EnhancedReasoningEngine | Расширенный движок с регенерацией и 3 модулями |
| ReasoningIntegration | Интеграция SelfReasoningEngine с CoreBrain |
| FractalStorage | Иерархическое хранилище (L0-L3) |
| AnalyticsModule | Анализ текста: сущности, логические блоки |
| PromptComposer | Объединение промптов от модулей |
| SemanticStabilityChecker | Проверка стабильности (similarity > 0.95) |
| CorrelationCalculator | Корреляция между ответами |
| ConfidenceScorer | Расчёт уверенности с адаптивными весами |

---

## 2. Реализация

### 2.1 SelfReasoningEngine

**Принцип работы:**
- Цикл: Generate -> Analyze -> Clarify -> Repeat until confidence >= 0.75
- Максимум 5 итераций
- Рекурсивная обработка сложных запросов (до глубины 3)

**Логические факторы:**
- ethics: weight 0.2
- knowledge: weight 0.25
- contradiction: weight 0.2
- context: weight 0.15
- logic: weight 0.2

**Flow:**
1. Анализ типа запроса (кратко/подробно)
2. Генерация через Two-Model Pipeline (Model A/B)
3. Проверка качества Model A и Model B
4. Анализ логических факторов
5. Альтернативные ветвления при низких оценках факторов
6. Проверка критериев остановки (адаптивный порог)
7. Сохранение в FractalStorage

### 2.2 EnhancedReasoningEngine

**Принцип работы:**
- Query -> Qwen -> Analytics -> 3 Modules (parallel) -> Composer -> Qwen -> Stability check

**Три параллельных модуля:**
1. Contradiction module - проверка противоречий
2. Ethics module - проверка этичности
3. Web search module - обогащение контекста

**Критерии остановки:**
- Semantic stability: similarity > 0.95
- Combined improvement: < threshold (0.05)
- Max iterations: 5

### 2.3 FractalStorage

**Иерархическая структура:**
- L0 (Root) -> L1 (Category) -> L2 (Entity) -> L3 (Detail)
- BRANCHING_FACTOR = 16
- EMBEDDING_DIM = 384

**Типы узлов:**
- reasoning_step - шаг рассуждения
- clarification - вопрос уточнения
- query - запрос пользователя
- response - ответ системы

### 2.4 Confidence Scoring

**Адаптивные веса (coarse-to-fine):**
- Iteration 1: ethics 0.40, contradiction 0.25, knowledge 0.35
- Iteration 2: ethics 0.30, contradiction 0.35, knowledge 0.35
- Iteration 3: ethics 0.20, contradiction 0.40, knowledge 0.40
- Iteration 4: ethics 0.15, contradiction 0.40, knowledge 0.45
- Iteration 5: ethics 0.10, contradiction 0.35, knowledge 0.55

**Адаптивные пороги:**
- Iteration 1: 0.80
- Iteration 2: 0.75
- Iteration 3: 0.70
- Iteration 4: 0.65
- Iteration 5: 0.60

---

## 3. Интеграция

### 3.1 Интеграция в CoreBrain

Через init_factories.py:
- self_reasoning_engine
- enhanced_reasoning_engine

### 3.2 Интеграция в brain_query.py

Fallback цепочка:
1. SelfReasoningEngine.process_query() - используется в _handle_fallback()
2. EnhancedReasoningEngine.process_query() - используется если SRE недоступен
3. QwenModelManager - финальный fallback

### 3.3 Two-Model Pipeline в SelfReasoningEngine

SRE использует two_model_pipeline для генерации:
- Model A - для кратких ответов (256 tokens)
- Model B - для развёрнутых (512 tokens)

### 3.4 Модульная интеграция

EnhancedReasoningEngine интегрируется с:
- brain.contradiction_manager - проверка противоречий
- brain.ethics_framework - проверка этики
- brain.web_search_engine - обогащение контекста
- brain.knowledge_graph - сохранение сущностей

---

## 4. Оценка

### Сильные стороны

1. Двухуровневая архитектура
   - SelfReasoningEngine - базовый цикл
   - EnhancedReasoningEngine - регенерация с модулями

2. Адаптивность
   - Адаптивные веса и пороги итераций
   - coarse-to-fine рассуждение

3. Модульность
   - FractalStorage для иерархического хранения
   - Разделение на подмодули (context, quality, recursive)

4. Интеграция с генерацией
   - Two-Model Pipeline
   - Fallback цепочка

### Слабые стороны / Проблемы

1. Сложность кода
   - Много дублирования между SelfReasoningEngine и EnhancedReasoningEngine
   - SRE подмодули (sre_*.py) - это методы-копии, а не отдельные классы

2. Качество анализа
   - AnalyticsModule использует простые regex-паттерны
   - ClarificationGenerator - шаблонные вопросы

3. Интеграция
   - FractalStorage отдельно от FractalGraphV2
   - Нет единой точки входа для reasoning

4. Статус
   - EnhancedReasoningEngine: Fractal Qwen отключён (_use_fractal_for_prompts = False)
   - Модули используют CPU для промтов, GPU для генерации

### Рекомендации

1. Унифицировать SelfReasoningEngine и EnhancedReasoningEngine
2. Выделить SRE подмодули в отдельные классы (не методы)
3. Интегрировать FractalStorage с FractalGraphV2
4. Улучшить NLP в AnalyticsModule (использовать NLP библиотеки)
5. Активировать Fractal Qwen для prompt generation
