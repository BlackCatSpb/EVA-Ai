# Анализ NLP & Preprocess & Reasoning EVA

## Часть 1: NLP система

### Файлы
- `nlp/text_processor.py` - обработка текста
- `nlp/__init__.py` - экспорт

### TextProcessor

**Класс:** EVA текстовый процессор с поддержкой:
- Токенизация через EATtokenizer или HuggingFace
- Очистка текста (special chars, URLs, emails)
- Sentence splitting для русского и английского

**Основные методы:**
- `tokenize(text)` - токенизация
- `clean_text(text)` - очистка
- `split_sentences(text)` - разбиение на предложения
- `get_stats()` - статистика

**Статус: АКТИВЕН**

---

## Часть 2: Preprocess система

### Файлы
- `preprocess/preprocessing_pipeline.py` - пайплайн предобработки
- `preprocess/__init__.py` - экспорт

### PreprocessingPipeline

**Класс:** Пайплайн предобработки данных для обучения

**Методы:**
- `run(text)` - полный прогон
- `normalize_text()` - нормализация
- `filter_quality()` - фильтрация качества

**Статус: ИСПОЛЬЗУЕТСЯ** в обучении

---

## Часть 3: Reasoning система

### Файлы (12 файлов)
- `self_reasoning_engine.py` - ядро саморассуждения
- `sre_recursive.py` - рекурсивное рассуждение
- `sre_quality.py` - оценка качества
- `sre_context.py` - контекстуальное рассуждение
- `reasoning_nodes.py` - узлы рассуждения
- `confidence_scorer.py` - оценка уверенности
- `clarification_generator.py` - генерация уточнений
- и др.

### SelfReasoningEngine

**Класс:** Движок саморассуждения для EVA

**Функции:**
- Генерировать рассуждения для ответов
- Оценивать качество через SRE-Quality
- Контекстуальная обработка
- Рекурсивное уточнение

**Методы:**
- `generate_reasoning(context, query)` - генерация
- `evaluate_quality(reasoning)` - оценка
- `refine_reasoning(reasoning)` - уточнение

### ReasoningNodes

**Узлы:**
- `ReasoningStep` - шаг рассуждения
- `ReasoningChain` - цепочка
- `ReasoningGraph` - граф рассуждений

### ConfidenceScorer

**Оценка:**
- Внутриответная согласованность
- Фактическая точность
- Уверенность модели

### ClarificationGenerator

**Генерация уточняющих вопросов** при неопределённости

---

## Выводы

| Система | Статус | Использование |
|---------|--------|---------------|
| NLP | ✅ Активен | Токенизация, очистка |
| Preprocess | ✅ Используется | Обучение |
| Reasoning | ✅ Активен | Саморассуждение |

Все три системы являются частью основного пайплайна EVA.