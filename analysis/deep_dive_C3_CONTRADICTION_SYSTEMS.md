# C3 Deep Dive: Системы детекции противоречий - дублирование функционала

## Содержание

1. [Введение](#введение)
2. [Все три системы (анализ)](#все-три-системы-анализ)
   - [1.1 ContradictionGenerator](#11-contradictiongenerator)
   - [1.2 ContradictionMiner](#12-contradictionminer)
   - [1.3 Legacy Detection System](#13-legacy-detection-system)
   - [1.4 ContradictionManager - центральный хаб](#14-contradictionmanager---центральный-хаб)
3. [Функции и их пересечения](#функции-и-их-пересечения)
   - [2.1 Сравнительная таблица](#21-сравнительная-таблица)
   - [2.2 Дублирующиеся функции](#22-дублирующиеся-функции)
   - [2.3 Потоки данных](#23-потоки-данных)
4. [Рекомендуемая архитектура](#рекомендуемая-архитектура)
   - [3.1 Принципы объединения](#31-принципы-объединения)
   - [3.2 Новая архитектура](#32-новая-архитектура)
   - [3.3 Обязанности компонентов](#33-обязанности-компонентов)
5. [План объединения](#план-объединения)
   - [4.1 Фаза 1: Инфраструктура (неделя 1)](#41-фаза-1-инфраструктура-неделя-1)
   - [4.2 Фаза 2: Миграция ContradictionGenerator (неделя 2)](#42-фаза-2-миграция-contradictiongenerator-неделя-2)
   - [4.3 Фаза 3: Миграция Legacy (неделя 3)](#43-фаза-3-миграция-legacy-неделя-3)
   - [4.4 Фаза 4: Тестирование и оптимизация (неделя 4)](#44-фаза-4-тестирование-и-оптимизация-неделя-4)
6. [Риски и митигации](#риски-и-митигации)
7. [Итоговые рекомендации](#итоговые-рекомендации)

---

## Введение

Проблема C3 формулируется следующим образом: в системе EVA AI существует три независимых системы детекции противоречий, которые выполняют схожие задачи, но работают изолированно друг от друга. Это приводит к:

- Дублированию кода и логики
- Несогласованности результатов
- Путанице в том, какую систему использовать
- Сложности поддержки и развития
- Потенциальным конфликтам при обнаружении противоречий

Цель данного документа — провести детальный анализ каждой системы, определить зоны пересечения и предложить архитектурное решение для объединения.

---

## Все три системы (анализ)

### 1.1 ContradictionGenerator

**Расположение:** eva_ai/contradiction/contradiction_generator.py

#### Общее описание

ContradictionGenerator — это система генерации противоречий на основе шаблонов. Она не анализирует существующие знания системы, а создаёт искусственные противоречивые точки зрения для заданных концептов. Основная цель — стимулировать самодиалог и исследование концептов с разных сторон.

#### Ключевые характеристики

- **Метод работы:** Шаблонный (template-based)
- **Триггер:** При извлечении концепта из запроса/ответа (ConceptExtractor вызывает генерацию)
- **Источник данных:** Внешние шаблоны (positive/negative viewpoints)
- **Выход:** GeneratedContradiction объекты с двумя точками зрения

#### Архитектура

`
ContradictionGenerator
├── _viewpoint_templates - Словарь шаблонов по доменам
│   ├── general - Общие шаблоны
│   ├── technology - Технологические
│   ├── science - Научные
│   └── philosophy - Философские
├── generate_contradiction() - Генерация одного противоречия
├── _generate_reasoning() - Генерация обоснования
├── save_contradiction() - Сохранение в ContradictionManager
├── format_for_dialog() - Форматирование для самодиалога
├── generate_batch() - Пакетная генерация
├── get_contradictions_for_prompt() - Получение контекста для промпта
└── save_resolution() - Сохранение разрешения
`

#### Пример работы

`python
# Входные данные
concept_name = "искусственный интеллект"
domain = "technology"

# Результат (GeneratedContradiction)
{
    "concept": "искусственный интеллект",
    "viewpoint_a": "ИИ делает жизнь лучше",
    "viewpoint_b": "ИИ создаёт новые проблемы",
    "divergence_level": 0.78,
    "reasoning_a": "Потому что ИИ создаёт возможности для развития",
    "reasoning_b": "Потому что ИИ создаёт риски и неопределённость"
}
`

#### Поток вызовов

1. **brain_query.py** -> извлекает концепты через ConceptExtractor
2. **ConceptExtractor** -> вызывает ContradictionGenerator.generate_contradiction()
3. **ContradictionGenerator** -> создаёт противоречие из шаблонов
4. **ContradictionGenerator.save_contradiction()** -> сохраняет в ContradictionManager
5. **ContradictionManager** -> добавляет в список противоречий и hybrid_cache
6. Позже: **SelfDialogLearning** -> обрабатывает противоречие в самодиалоге

#### Плюсы и минусы

| Плюсы | Минусы |
|-------|--------|
| Быстрая генерация (не требует LLM) | Искусственные противоречия, не отражают реальные конфликты в знаниях |
| Простая архитектура | Не анализирует структуру графа знаний |
| Разнообразные домены | Шаблоны могут быть нерелевантны |
| Не требует NLI модели | Ограниченная глубина анализа |

---

### 1.2 ContradictionMiner

**Расположение:** eva_ai/contradiction/contradiction_miner.py

#### Общее описание

ContradictionMiner — это система проактивного обнаружения реальных логических противоречий в структуре FractalGraph v2. В отличие от ContradictionGenerator, Miner анализирует существующие узлы графа и выявляет меж��у ними отношения семантической близости и логического противоречия. Это аналитическая система, работающая в фоновом режиме.

#### Ключевые характеристики

- **Метод работы:** Математический анализ (cosine similarity + NLI classification)
- **Триггер:** Системные события (memory.node_created, memory.graph_updated, system.idle) + интервал проверки
- **Источник данных:** FractalGraph v2 (узлы и их эмбеддинги)
- **Выход:** ContradictionNode в FGv2 + очередь в SelfDialogLearning

#### Архитектура

`
ContradictionMiner
├── DEFAULT_CONFIG - Конфигурация по умолчанию
│   ├── sim_threshold: 0.75 - Порог семантической близости
│   ├── contra_threshold: 0.65 - Порог противоречия (NLI)
│   ├── max_candidates_per_cycle: 5
│   ├── check_interval_seconds: 3600
│   └── priority_coefficients (alpha, beta, gamma)
├── _detection_cycle() - Основной цикл обнаружения
│   ├── _detect_candidate_pairs() - Этап 1: Поиск пар (sim >= 0.75, contra >= 0.65)
│   ├── _cluster_pairs() - Этап 2: Транзитивное замыкание
│   ├── _filter_and_prioritize() - Этап 3: Приоритизация
│   ├── _generate_formulation() - Этап 4: Генерация через LLM
│   ├── _validate_candidate() - Этап 5: Валидация
│   └── _create_contradiction_node() - Этап 6: Создание узла в FGv2
├── _compute_similarity() - Косинусное сходство эмбеддингов
├── _compute_contradiction() - NLI оценка (Bart-large-mnli)
├── _compute_contradiction_heuristic() - Fallback эвристика
├── _has_excluding_relation() - Проверка исключающих отношений
├── get_metrics() - Получение метрик работы
├── resolve_contradiction() - Разрешение противоречия
└── force_check() - Принудительный запуск проверки
`

#### Алгоритм работы

1. **Получение узлов из FGv2** — извлекаются все узлы с эмбеддингами
2. **Попарное сравнение** — для каждой пары вычисляется:
   - Cosine similarity: sim(u,v) = cos(emb(u), emb(v))
   - NLI contradiction score: contra(u,v) через BART-large-mnli
3. **Фильтрация** — пара становится кандидатом если:
   - sim >= 0.75 (высокая семантическая близость)
   - contra >= 0.65 (логическое противоречие)
4. **Кластеризация** — транзитивное замыкание объединяет связанные пары
5. **Приоритизация** — формула: priority = α·|C| + β·avg_confidence + γ·max_contra
6. **Генерация формулировки** — через LLM с SYSTEM_PROMPT
7. **Создание ContradictionNode** — узел типа 'contradiction' в FGv2
8. **Очередь самодиалога** — добавление в SelfDialogLearning для разрешения

#### Пример работы

`
# Входные данные (пара узлов из FGv2)
node_A: "AI полезен для медицины"
node_B: "AI опасен и неприменим в медицине"

# Метрики
similarity = 0.82  # Высокая семантическая близость
contradiction = 0.73  # Логическое противоречие (NLI)

# Выход (ContradictionNode)
{
    "node_type": "contradiction",
    "content": {
        "title": "Противоречие в оценке применения AI в медицине",
        "description": "С одной стороны, AI демонстрирует высокую эффективность...",
        "resolution_question": "Какие условия определяют безопасность AI в медицине?"
    },
    "metadata": {
        "status": "active",
        "cluster_size": 2,
        "max_contra_score": 0.73,
        "priority": 0.68,
        "source_nodes": [node_A_id, node_B_id]
    }
}
`

#### Плюсы и минусы

| Плюсы | Минусы |
|-------|--------|
| Обнаруживает реальные противоречия в знаниях | Требует NLI модель (дополнительная память) |
| Математически обоснованный подход | Зависит от качества эмбеддингов |
| Автоматическая генерация формулировок | Высокая вычислительная сложность (O(n²)) |
| Интеграция с FGv2 и SelfDialog | Интервал проверки 1 час — может пропустить противоречия |
| Кэширование для оптимизации | Требует много узлов в графе для эффективной работы |

---

### 1.3 Legacy Detection System

**Расположение:** eva_ai/contradiction/detect_core.py, detect_semantic.py, detect_logical.py, detect_temporal.py

#### Общее описание

Legacy Detection System — это наиболее зрелая и сложная система, существовавшая до текущей рефакторизации. Она использует множество методов детекции противоречий, включая анализ фактов в knowledge_graph, проверку иерархий, эксклюзивность и темпоральные зависимости. Система состоит из основного класса ContradictionDetector и нескольких миксинов.

#### Ключевые характеристики

- **Метод работы:** Многоуровневый анализ (семантический, логический, темпоральный)
- **Триггер:** Ручной вызов (detect_contradictions()) или фоновая проверка
- **Источник данных:** Knowledge Graph (факты, концепты, отношения)
- **Выход:** Список словарей с противоречиями

#### Архитектура

`
ContradictionDetector (detect_core.py)
├── Наследует от:
│   ├── SemanticDetectionMixin (detect_semantic.py)
│   │   ├── _calculate_text_divergence() - Семантическое расхождение
│   │   └── _calculate_lexical_divergence() - Лексическое расхождение
│   ├── LogicalDetectionMixin (detect_logical.py)
│   │   ├── detect_hierarchy_contradictions() - Иерархические противоречия
│   │   ├── detect_exclusivity_contradictions() - Эксклюзивность
│   │   └── _has_cyclic_dependency() - Циклические зависимости
│   └── TemporalDetectionMixin (detect_temporal.py)
│       └── detect_temporal_contradictions() - Временные противоречия
├── detect_contradictions() - Основной метод детекции
├── _detect_contradictions_for_concept() - Детекция для концепта
├── _find_potential_contradictions() - Поиск потенциальных ��ар
├── _calculate_divergence() - Расчёт уровня расхождения
├── _create_contradiction() - Создание объекта противоречия
├── detect_contradictions_in_new_fact() - Проверка нового факта
├── analyze_fact_consistency() - Анализ согласованности
├── detect_all_specialized_contradictions() - Все типы
├── get_detection_statistics() - Статистика
├── generate_contradiction_report() - Генерация отчёта
├── integrate_contradiction_resolution() - Интеграция решения
└── start_background_detection() / stop_background_detection() - Фоновая работа
`

#### Типы противоречий

1. **Numeric contradictions** — числовые значения, выходящие за пределы 2σ от среднего
2. **Boolean contradictions** — противоречивые булевы значения
3. **Text contradictions** — семантически противоположные тексты
4. **Hierarchy contradictions** — концепт принадлежит взаимоисключающим родителям
5. **Exclusivity contradictions** — only_in vs not_only_in
6. **Temporal contradictions** — противоречия во временных отношениях

#### Поток вызовов (упрощённый)

1. **Внешний вызов** -> ContradictionManager (или напрямую)
2. **ContradictionDetector** -> получает факты из knowledge_graph
3. **Группировка по концептам и отношениям**
4. **Поиск потенциальных пар** -> различные методы (numeric, boolean, text)
5. **Расчёт divergence** -> порог 0.65
6. **Формирование объекта противоречия** -> добавление метаданных
7. **Сохранение** -> в detected_contradictions

#### Пример работы

`python
# Входные данные (факты в knowledge_graph)
fact_A = {
    "concept": "вода",
    "relation_type": "has_property",
    "value": 100,  # температура кипения
    "source": "encyclopedia"
}
fact_B = {
    "concept": "вода",
    "relation_type": "has_property", 
    "value": 0,  # температура замерзания
    "source": "user_input"
}

# Результат
{
    "contradiction_id": "contradiction_a1b2c3d4",
    "concept": "вода",
    "conflicting_facts": [fact_A, fact_B],
    "divergence_level": 1.0,  # max для numeric
    "timestamp": 1714060800.0,
    "status": "detected",
    "metadata": {
        "relation_type": "has_property",
        "detection_method": "automatic"
    }
}
`

#### Плюсы и минусы

| Плюсы | Минусы |
|-------|--------|
| Много типов противоречий (6+) | Сложная архитектура (4 файла, миксины) |
| Работает с фактами, не только с узлами | Частично перекрывается с ContradictionMiner |
| Зрелый и протестированный код | Использует устаревший knowledge_graph |
| Гибкая настройка порогов | Требует sentence-transformers модель |
| Интеграция с системой репутации источников | Не интегрирован с FGv2 |
| Фоновая работа | Изолирован от основного потока |

---

### 1.4 ContradictionManager — центральный хаб

**Расположение:** eva_ai/contradiction/contradiction_manager.py

#### Общее описание

ContradictionManager выступает центральным хранилищем и координатором для всех противоречий в системе. Он получает противоречия из всех трёх систем, хранит их в памяти и hybrid_cache, и предоставляет API для работы с ними.

#### Ключевые методы

- dd_contradiction() — добавление противоречия
- _save_contradiction_to_cache() — сохранение в hybrid_cache
- detect_contradictions() — запуск Legacy детектора
- esolve_contradiction() — разрешение противоречия
- check_with_context() — проверка с контекстом
- check_fractal_graph_contradictions() — проверка через FGv2
- generate_refinement_prompt() — генерация промпта для регенерации

#### Где используется

1. **init_factories.py** — создаётся в CoreBrain
2. **brain_query.py** — проверка противоречий при генерации ответа
3. **ContradictionGenerator** — сохраняет сгенерированные противоречия
4. **SelfDialogLearning** — обрабатывает очередь противоречий

---

## Функции и их пересечения

### 2.1 Сравнительная таблица

| Характеристика | ContradictionGenerator | ContradictionMiner | Legacy Detection |
|----------------|------------------------|-------------------|-------------------|
| **Тип** | Генерация (synthetic) | Обнаружение (analytic) | Обнаружение (holistic) |
| **Источник** | Шаблоны | FGv2 узлы | Knowledge Graph факты |
| **Метод** | Template-based | Cosine + NLI | Multi-strategy |
| **Триггер** | Concept extraction | System events | Manual/Periodic |
| **Выход** | GeneratedContradiction | ContradictionNode | List[Dict] |
| **NLI модель** | Нет | BART-large-mnli | Sentence-transformers |
| **Интеграция FGv2** | Indirect (через CM) | Direct | Нет |
| **Интеграция KG** | Indirect | Indirect | Direct |
| **Сложность** | O(1) | O(n²) | O(n·m) |
| **Скорость** | Мгновенно | Медленно | Средне |

#### Обозначения:
- **CM** — ContradictionManager
- **FGv2** — FractalGraph v2
- **KG** — Knowledge Graph (legacy)
- **n** — количество узлов/фактов
- **m** — количество типов проверки

---

### 2.2 Дублирующиеся функции

#### Функция 1: Сохранение противоречий

| Система | Метод | Назначение |
|---------|-------|------------|
| ContradictionGenerator | save_contradiction() | Сохраняет в ContradictionManager |
| ContradictionMiner | _create_contradiction_node() | Создаёт узел в FGv2 |
| Legacy | (через ContradictionManager) | Добавляет в detected_contradictions |

**Пересечение:** Все три системы в конечном счёте хранят противоречия. Generator и Miner явно вызывают ContradictionManager.add_contradiction(). Legacy также использует этот метод.

#### Функция 2: Получение контекста для промпта

| Система | Метод | Назначение |
|---------|-------|------------|
| ContradictionGenerator | get_contradictions_for_prompt() | Форматирует противоречия для LLM |
| ContradictionMiner | _generate_formulation() | Генерирует формулировку через LLM |
| Legacy | (отсутствует) | Н�� используется для генерации |

**Пересечение:** Generator и Miner занимаются формированием текстового представления противоречий для последующей обработки.

#### Функция 3: Разрешение противоречий

| Система | Метод | Назначение |
|---------|-------|------------|
| ContradictionGenerator | save_resolution() | Сохраняет результат самодиалога |
| ContradictionMiner | resolve_contradiction() | Отмечает как resolved |
| Legacy | integrate_contradiction_resolution() | Интегрирует в KG |

**Пересечение:** Все три системы имеют методы для финальной стадии жизненного цикла противоречия — его разрешения.

#### Функция 4: Проверка на дубликаты

| Система | Метод | Назначение |
|---------|-------|------------|
| ContradictionMiner | _validate_candidate() | Проверяет node_ids |
| Legacy | _are_facts_equivalent() | Проверяет факты |

**Пересечение:** Проверка на существование аналогичного противоречия перед добавлением.

---

### 2.3 Потоки данных

#### Поток 1: Генерация противоречий (Generator)

`
User Query -> brain_query.process_query()
                    |
                    v
        ConceptExtractor.extract_concepts()
                    |
                    v
        ContradictionGenerator.generate_contradiction()
                    |
                    v
        ContradictionManager.add_contradiction()
                    |
                    v
        hybrid_cache.add_contradiction()
                    |
                    v
        SelfDialogLearning.queue_contradiction_for_resolution()
                    |
                    v
        SelfDialogLearning._run_contradiction_dialog()
                    |
                    v
        ContradictionGenerator.save_resolution()
`

#### Поток 2: Обнаружение противоречий (Miner)

`
System Idle / Graph Updated / Node Created
            |
            v
    ContradictionMiner._detection_cycle()
            |
            +---> _detect_candidate_pairs() [O(n²) анализ]
            |         |
            |         +---> FGv2.get_nodes()
            |                   |
            |                   +---> cosine_similarity()
            |                   +---> NLI.classify()
            |
            +---> _cluster_pairs() [Транзитивное замыкание]
            |
            +---> _generate_formulation() [LLM]
            |
            +---> _create_contradiction_node() [FGv2.add_node()]
            |
            v
    SelfDialogLearning.queue_contradiction_for_resolution()
            |
            v
        SelfDialogLearning._run_contradiction_dialog()
            |
            v
        ContradictionMiner.resolve_contradiction()
`

#### Поток 3: Legacy Detection

`
External call / Background detection
            |
            v
    ContradictionDetector.detect_contradictions()
            |
            +---> KG.get_all_concepts()
            |
            for each concept:
                |
                +---> _detect_contradictions_for_concept()
                |         |
                |         +---> _find_potential_contradictions()
                |         |         |
                |         |         +---> Numeric check
                |         |         +---> Boolean check
                |         |         +---> Text similarity check
                |         |
                |         +---> _calculate_divergence()
                |         |
                |         +---> _create_contradiction()
                |
                v
            ContradictionManager.add_contradiction()
`

---

## Рекомендуемая архитектура

### 3.1 Принципы объединения

Для решения проблемы C3 предлагается придерживаться следующих принципов:

1. **Единая точка входа** — ContradictionManager остаётся центральным хабом, но получает унифицированное API
2. **Разделение ответственности** — каждая система отвечает за свой уровень:
   - Generator — быстрая генерация для стимуляции диалога
   - Miner — глубокий анализ графа (реальные противоречия)
   - Legacy — анализ фактов и согласованности
3. **Общий формат данных** — унифицированная структура ContradictionData
4. **Конвейерная обработка** — противоречия проходят единый жизненный цикл
5. **Приоритизация** — система решает, какую стратегию применять

### 3.2 Новая архитектура

`
                          ┌─────────────────────────────────────────────┐
                          │           CONTRADICTION MANAGER              │
                          │              (Единая точка входа)            │
                          │                                             │
                          │  - unified_detect()                         │
                          │  - unified_resolve()                        │
                          │  - get_context_for_prompt()                 │
                          │  - get_unified_stats()                      │
                          └─────────────────┬───────────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    v                       v                       v
        ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
        │   GENERATOR       │   │   MINER           │   │   LEGACY          │
        │   (Fast Layer)   │   │   (Deep Layer)    │   │   (Fact Layer)   │
        │                   │   │                   │   │                   │
        │ Template-based    │   │ Graph analysis    │   │ Fact consistency │
        │ Generation        │   │ Real contradictions│   │ Multi-type check │
        │                   │   │                   │   │                   │
        │ Trigger:          │   │ Trigger:          │   │ Trigger:          │
        │ Concept extract   │   │ System idle       │   │ Manual/Fallback   │
        │                   │   │ Graph updated     │   │                   │
        └───────────────────┘   └───────────────────┘   └───────────────────┘
                    │                       │                       │
                    └───────────────────────┼───────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    v                       v                       v
        ┌──────────────────────────────────────────────────────────────────┐
        │                    UNIFIED STORAGE LAYER                        │
        │                                                                 │
        │  - ContradictionRegistry (primary storage)                     │
        │  - HybridCache (fast access)                                   │
        │  - FGv2 (persistent graph nodes for conflicts)                 │
        └──────────────────────────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    v                       v                       v
        ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
        │   SelfDialog      │   │   KCA Module      │   │   Brain Query     │
        │   Learning        │   │   (Attention)     │   │   (Generation)    │
        └───────────────────┘   └───────────────────┘   └───────────────────┘
`

### 3.3 Обязанности компонентов

#### UnifiedContradictionManager (рефакторинг)

Переименовывается из ContradictionManager для отражения новой роли.

**Основные методы:**

1. unified_detect(source=None, target=None, options=None) — единый метод детекции
   - source: 'generator' | 'miner' | 'legacy' | 'auto' (по умолчанию)
   - target: концепт или текст для проверки
   - options: дополнительные параметры

2. unified_resolve(contradiction_id, resolution_data) — единое разрешение

3. get_context_for_prompt(concept_name, max_count=3) — контекст для генерации

4. get_unified_stats() — статистика по всем системам

5. oute_to_appropriate_service(concept) — роутинг на нужную систему

#### ContradictionGenerator (Fast Layer)

Сохраняется как есть, но интегрируется через UnifiedContradictionManager:

- Вызов через manager.unified_detect(source='generator', target=concept)
- Возвращает унифицированный формат

#### ContradictionMiner (Deep Layer)

Сохраняется как есть:

- Работает в фоне на события
- Создаёт ContradictionNode в FGv2
- Через UnifiedContradictionManager добавляет в общую очередь

#### LegacyDetector (Fact Layer)

Переносится в submodule:

- Используется как fallback ��ля проверки согласованности
- Вызывается при обнаружении новых фактов
- Интегрирован в nalyze_fact_consistency()

---

## План объединения

### 4.1 Фаза 1: Инфраструктура (Неделя 1)

**Задачи:**

- [ ] Создать унифицированный формат данных UnifiedContradiction
- [ ] Добавить UnifiedStorage слой
- [ ] Рефакторить ContradictionManager в UnifiedContradictionManager
- [ ] Добавить метод oute_to_appropriate_service()

**Формат UnifiedContradiction:**

`python
@dataclass
class UnifiedContradiction:
    id: str
    source_layer: str  # 'generator' | 'miner' | 'legacy'
    concept: str
    conflicting_statements: List[str]
    divergence_level: float
    status: str  # 'detected' | 'analyzing' | 'resolved'
    resolution: Optional[str]
    source_metadata: Dict  # Оригинальные данные от системы-источника
    created_at: float
    resolved_at: Optional[float]
`

**Файлы для изменения:**

- eva_ai/contradiction/contradiction_manager.py — рефакторинг
- eva_ai/contradiction/unified_types.py — новый файл

**Критерии приёмки:**

- UnifiedManager имеет единый API
- Все три системы могут использовать единый формат

---

### 4.2 Фаза 2: Миграция ContradictionGenerator (Неделя 2)

**Задачи:**

- [ ] Добавить адаптер для Generator в UnifiedManager
- [ ] Обновить вызовы в ConceptExtractor
- [ ] Проверить сохранение в hybrid_cache

**Изменения в коде:**

1. В concept_extractor.py:
`python
# Было
self.contr_generator.generate_contradiction(concept_name)

# Стало
self.contradiction_manager.unified_detect(
    source='generator',
    target=concept_name
)
`

2. Добавить метод-адаптер в UnifiedManager:
`python
def _handle_generator_request(self, target):
    generator = getattr(self.brain, 'contradiction_generator')
    result = generator.generate_contradiction(target)
    return self._convert_to_unified(result, source='generator')
`

**Критерии приёмки:**

- Generator работает через UnifiedManager
- Совместимость с существующим кодом

---

### 4.3 Фаза 3: Миграция Legacy (Неделя 3)

**Задачи:**

- [ ] Интегрировать LegacyDetector как Fact Layer
- [ ] Добавить проверку согласованности фактов
- [ ] Удалить избыточные методы в ContradictionManager

**Изменения в архитектуре:**

1. LegacyDetector становится "Fact Consistency Checker"
2. Вызывается при:
   - Добавлении новых фактов в KG
   - Запросе анализа согласованности
   - Fallback, когда Miner не нашёл противоречий

**Критерии приёмки:**

- Legacy работает как Fact Layer
- Нет дублирующихся проверок

---

### 4.4 Фаза 4: Тестирование и оптимизация (Неделя 4)

**Задачи:**

- [ ] Написать интеграционные тесты
- [ ] Измерить производительность
- [ ] Добавить метрики для мониторинга
- [ ] Документировать новую архитектуру

**Метрики:**

- Время отклика каждого слоя
- Количество обнаруженных противоречий по типам
- Процент разрешённых противоречий
- Нагрузка на NLI модель

**Критерии приёмки:**

- Все тесты проходят
- Производительность не ухудшена
- Documentationupdated

---

## Риски и митигации

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| Поломка существующего кода при рефакторинге | Высокая | Высокое | Фазовая миграция, обратная совместимость |
| Потеря функциональности Legacy | Средняя | Высокое | Полное тестирование каждой функции |
| Miner перестанет работать из-за изменений | Средняя | Среднее | Сохранить оригинальные методы |
| Увеличение сложности | Высокая | Низкое | Документирование, clear API |
| Конфликты при сохранении | Низкая | Среднее | Idempotent операции |

---

## Итоговые рекомендации

После проведённого анализа рекомендуется следующая стратегия:

1. **Оставить ContradictionMiner как основную систему** — наиболее технологически продвинутая, работает с FGv2, создаёт реальные узлы

2. **保留 ContradictionGenerator как Fast Layer** — полезна для стимуляции самодиалога, работает независимо

3. **Интегрировать Legacy как Fact Consistency Layer** — для проверки входящих фактов и согласованности

4. **UnifiedManager как единая точка входа** — скрывает сложность от вызывающего кода

5. **Приоритет выполнения:**
   - При извлечении концепта → Generator (Fast)
   - При обновлении графа → Miner (Deep)
   - При добавлении факта → Legacy (Consistency)

---

## Приложение: Список файлов

| Файл | Роль | Статус |
|------|------|--------|
| contradiction_generator.py | Fast Layer | Сохранить |
| contradiction_miner.py | Deep Layer | Сохранить |
| contradiction_manager.py | Central Hub | Рефакторить |
| detect_core.py | Legacy | Интегрировать |
| detect_semantic.py | Legacy | Интегрировать |
| detect_logical.py | Legacy | Интегрировать |
| detect_temporal.py | Legacy | Интегрировать |

---

*Дата создания: 2026-04-27*
*Автор: AI Architect Analysis*
*Версия: 1.0*