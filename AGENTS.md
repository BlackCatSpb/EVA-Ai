# AGENTS.md - EVA AI System Context

## Goal

Восстановление и отладка EVA AI системы. Реализация системы концептов и противоречий через самодиалоги.

## Instructions

- Всегда запускать EVA в отдельном терминале: `cd C:\Users\black\OneDrive\Desktop\CogniFlex && python -m eva_ai`
- Перед запуском всегда чистить логи: `Remove-Item "C:\Users\black\OneDrive\Desktop\CogniFlex\*.log" -Force`
- Слушать логи через Tee-Object для понимания ошибок
- Использовать русский язык для общения
- Работать только с DualGenerator/FractalGraph v2

## Цикл тестирования

1. **Таймаут проверки загрузки**: 15 секунд
2. **Логи анализирует агент**: выделить WARNING и ERROR, сформировать отчёт
3. **Агент передаёт отчёт** в чат для продолжения работы
4. **Продолжать по текущему плану ISSUES_TRACKER**

## New Architecture - Concepts & Contradictions System

### Двухуровневая система извлечения концептов:

#### 1A. ConceptExtractor (`eva_ai/knowledge/concept_extractor.py`) - БЫСТРЫЙ УРОВЕНЬ
Извлекает концепты из каждого запроса/ответа в реальном времени:
- **Частотный анализ** терминов из текста
- Создаёт узлы типа 'concept' в FGv2
- Генерирует базовые факты (is_a, has_property, can, related_to)
- Немедленно добавляет в очередь самодиалога
- **Работает синхронно** при каждом запросе

**Key Methods:**
- `extract_concepts(query, response)` - извлечение из текста
- `save_concept_to_graph(concept)` - сохранение в FGv2
- `format_concept_for_dialog(concept)` - форматирование для диалога

#### 1B. ConceptMiner (`eva_ai/knowledge/concept_miner.py`) - ГЛУБОКИЙ УРОВЕНЬ
Обнаруживает семантические лакуны в структуре графа в фоновом режиме:
- **Математический анализ** кластеров: центроиды, косинусное сходство, дисперсия
- Детекция семантических лакун: `ΔC = min(1 - cos(μC, v))`
- Адаптивный порог: `τ = τ_base * (1 + variance_k * σC)`
- Генерация гипотез через EVAGenerator (основная система генерации)
- **Многоуровневая валидация**: NLI, Ontology, Ethics, Web
- Жизненный цикл: `PROVISIONAL → CONFIRMED → STABLE → ARCHIVED`
- **Работает асинхронно** при простое системы

**Key Methods:**
- `_detect_semantic_gaps(clusters)` - детекция лакун
- `_generate_hypothesis(candidate)` - генерация через LLM
- `_validate_candidate(candidate)` - NLI/Ontology/Ethics/Web
- `_integrate_candidate(candidate)` - интеграция в FGv2

**Сравнение подходов:**

| Характеристика | ConceptExtractor | ConceptMiner |
|----------------|------------------|--------------|
| **Триггер** | Каждый запрос | Простой системы |
| **Источник** | Текст запроса/ответа | Кластеры узлов FGv2 |
| **Скорость** | Мгновенно | Медленно (анализ) |
| **Глубина** | Поверхностная | Глубокая (семантика) |
| **Валидация** | Нет | NLI + Ontology + Ethics |
| **Цель** | Быстрое пополнение | Качественные концепты |

### 2. Двухуровневая система противоречий:

#### 2A. ContradictionGenerator (`eva_ai/contradiction/contradiction_generator.py`) - ШАБЛОННЫЙ УРОВЕНЬ
Генерирует противоречия для концептов через шаблоны:
- Создаёт противоположные точки зрения (positive vs negative)
- Использует шаблоны по доменам (general, technology, science, philosophy)
- **Работает с концептами** извлечёнными из текста
- Быстрая генерация для самодиалога

**Key Methods:**
- `generate_contradiction(concept_name, domain)` - генерация через шаблоны
- `save_contradiction(contradiction)` - сохранение в систему
- `auto_generate_for_unknown_concepts(min_concepts)` - авто-генерация

#### 2B. ContradictionMiner (`eva_ai/contradiction/contradiction_miner.py`) - АНАЛИТИЧЕСКИЙ УРОВЕНЬ
Обнаруживает реальные противоречия в структуре графа:
- **Математическая модель**: sim(u,v) ≥ 0.75, contra(u,v) ≥ 0.65
- Анализ пар узлов с косинусным сходством эмбеддингов
- **NLI-оценка** логического противоречия (entailment/neutral/contradiction)
- Кластеризация конфликтных пар через транзитивное замыкание
- Создание ContradictionNode в FGv2 со связями `contradicts`
- **Работает в фоне** при простое системы

**Key Methods:**
- `_detect_candidate_pairs()` - поиск пар (sim ≥ 0.75, contra ≥ 0.65)
- `_cluster_pairs()` - кластеризация через транзитивное замыкание
- `_generate_formulation()` - генерация описания через LLM
- `_create_contradiction_node()` - создание узла в FGv2

**Сравнение подходов:**

| Характеристика | ContradictionGenerator | ContradictionMiner |
|----------------|------------------------|-------------------|
| **Триггер** | Извлечение концепта | Простой системы / обновление графа |
| **Источник** | Концепты из текста | Пары узлов в FGv2 |
| **Метод** | Шаблоны (positive/negative) | Математика + NLI |
| **Пороги** | Нет (всегда генерирует) | sim ≥ 0.75, contra ≥ 0.65 |
| **Выход** | Conflicting facts | ContradictionNode в графе |
| **Связи** | Через ContradictionManager | Рёбра `contradicts` в FGv2 |
| **Цель** | Обсуждение в самодиалоге | Разрешение неконсистентности |

### 3. DialogConceptsMixin (`eva_ai/learning/dialog_concepts.py`)
Интегрирует концепты и противоречия в самодиалог:
- Очередь концептов для обсуждения
- Очередь противоречий для разрешения
- Специализированные диалоги для концептов и противоречий
- Сохранение результатов в кеш контекста

**Key Methods:**
- `queue_concept_for_dialog(concept_name, priority)` - добавление в очередь
- `queue_contradiction_for_resolution(contr_id, concept, priority)` - добавление противоречия
- `_run_concept_dialog(dialog, concept_data)` - диалог о концепте
- `_run_contradiction_dialog(dialog, contradiction_data)` - разрешение противоречия
- `extract_knowledge_from_cache(concept)` - извлечение знаний из кеша

### 4. Modified SelfDialogLearning (`eva_ai/learning/dialog_core.py`)
Обновлён для работы с концептами:
- Использует DialogConceptsMixin
- Переопределён `_generate_dialog_from_conversations()`
- Приоритет: противоречия > концепты > история разговоров

### 5. Integration in CoreBrain (`eva_ai/core/init_factories.py`)
Инициализация в `create_knowledge_graph()`:
- Создаёт **ConceptExtractor** (быстрый уровень)
- Создаёт **ConceptMiner** (глубокий уровень) - запускается в фоне
- Создаёт **ContradictionGenerator**
- Добавляет все в brain.components

### 6. Integration in brain_query.py (ЦИКЛ ГЕНЕРАЦИИ)

Обработка в текущем цикле генерации ответа:

**A. Перед генерацией** (`_execute_with_web_search`):
- Получаем контекст из концептов: `get_concepts_for_prompt(query)`
- Получаем контекст противоречий: `get_contradictions_for_prompt(term)`
- Добавляем разрешённые знания из кеша
- Контекст добавляется к запросу перед отправкой в pipeline

**B. После генерации** (`process_query`):
- Извлекаем концепты: `_extract_key_concepts(query, response)`
- Сохраняем в FGv2
- Добавляем в очередь самодиалога

**C. Методы для цикла генерации**:
- `concept_extractor.get_concepts_for_prompt(query)` → строка с фактами
- `contradiction_generator.get_contradictions_for_prompt(concept)` → строка с точками зрения
- `dialog_concepts_mixin.get_context_for_generation(query)` → полный контекст

## Flow - How It Works

```
1. User Query → brain_query.process_query()
   ↓
2. БЫСТРЫЙ УРОВЕНЬ: ConceptExtractor
   - extract_concepts(query, response)
   - Save to FGv2 as 'concept' nodes
   - Queue to self_dialog_learning
   ↓
3. ГЛУБОКИЙ УРОВЕНЬ: ConceptMiner (фоново)
   - Подписка на system.idle / memory.graph_updated
   - _detect_semantic_gaps(clusters)
   - _generate_hypothesis() via GGUF
   - _validate_candidate() NLI/Ontology/Ethics
   - _integrate_candidate() → FGv2
   - Queue to self_dialog_learning
   ↓
4. SelfDialogLearning._worker_loop()
   - _generate_dialog_from_conversations()
   - _get_next_dialog_topic() 
   - Priority: contradiction > concept > conversation
   ↓
5. For Concept:
   _run_concept_dialog()
   - ASSISTANT: представляет концепт
   - CRITIC: ищет противоречия
   - LEARNER: предлагает направления
   - TEACHER: даёт рекомендации
   ↓
6. For Contradiction (от Generator - шаблоны):
   _run_contradiction_dialog()
   - ASSISTANT: представляет обе точки зрения
   - CRITIC: анализирует стороны
   - LEARNER: ищет синтез
   - TEACHER: формулирует разрешение
   ↓
7. For ContradictionNode (от Miner - анализ):
   - ContradictionMiner._detect_candidate_pairs() → sim/contra
   - _cluster_pairs() → кластеры через транзитивное замыкание
   - _generate_formulation() → title/description/resolution_question
   - _create_contradiction_node() → FGv2 с рёбрами contradicts
   - Queue to self_dialog_learning с resolution_question
   ↓
8. Save Results:
   - _save_concept_dialog_results() → cache
   - _save_contradiction_resolution() → cache
   - Update contradiction status to 'resolved'
   ↓
8. Knowledge Extraction:
   extract_knowledge_from_cache()
   - Convert to facts
   - Use in future responses
```

## Contradiction Generation Flow

```
1. ContradictionGenerator.generate_contradiction(concept, domain)
   - Select template pair (positive vs negative)
   - Format with concept name
   - Generate reasoning for both sides
   ↓
2. GeneratedContradiction object:
   - concept: str
   - viewpoint_a: str
   - viewpoint_b: str
   - divergence_level: float
   - reasoning_a: str
   - reasoning_b: str
   ↓
3. Save via ContradictionManager:
   - Create conflicting_facts list
   - Add to cm.contradictions
   ↓
4. Queue for resolution:
   - self_dialog_learning.queue_contradiction_for_resolution()
   ↓
5. Self-dialog resolves it
   - _run_contradiction_dialog()
   - Save resolution
   - Update status
```

## Contradiction Detection Flow (ContradictionMiner)

```
1. ContradictionMiner._detection_cycle() (при system.idle)
   ↓
2. _detect_candidate_pairs()
   - Получаем все узлы из FGv2
   - Для каждой пары: sim(u,v) = cos(emb(u), emb(v))
   - Если sim ≥ 0.75: проверяем contra(u,v) через NLI
   - Если contra ≥ 0.65: добавляем в список кандидатов
   ↓
3. _cluster_pairs()
   - Строим граф конфликтов из пар
   - Транзитивное замыкание: если (A,B) и (B,C) → {A,B,C}
   - Формируем ContradictionCandidate для каждого кластера
   ↓
4. _filter_and_prioritize()
   - priority = α·|C| + β·avg_confidence + γ·max_contra
   - Сортируем по приоритету
   ↓
5. _generate_formulation()
   - Генерация через LLM (temperature=0.25)
   - Выход: CONTRADICTION_TITLE, DESCRIPTION, RESOLUTION_QUESTION
   ↓
6. _validate_candidate()
   - Проверка качества формулировки
   - Проверка на дубликаты
   ↓
7. _create_contradiction_node()
   - Создание узла типа 'contradiction' в FGv2
   - Связи 'contradicts' к узлам кластера
   - Queue to self_dialog_learning
   ↓
8. Resolution:
   - Самодиалог с Resolution Question
   - Web-поиск для уточнения
   - Новый узел с контекстом
   - Статус: resolved
```

## File Structure

```
eva_ai/
├── knowledge/
│   ├── concept_extractor.py          # Concept extraction (быстрый уровень)
│   ├── concept_miner.py              # Concept mining (глубокий уровень)
│   ├── __init__.py                   # Exports ConceptExtractor, ConceptMiner
│   └── ...
├── contradiction/
│   ├── contradiction_generator.py    # Contradiction generation (шаблоны)
│   ├── contradiction_miner.py        # Contradiction mining (анализ графа)
│   ├── __init__.py                   # Exports ContradictionGenerator, ContradictionMiner
│   └── ...
├── learning/
│   ├── dialog_concepts.py            # Concepts in dialogs
│   ├── dialog_core.py                # SelfDialogLearning with DeferredCommandSystem
│   ├── __init__.py                   # Exports DialogConceptsMixin
│   └── ...
└── core/
    ├── init_factories.py             # Creates all components
    ├── brain_query.py                # Uses ConceptExtractor
    ├── model_access_manager.py       # Model access coordination (NEW)
    ├── openvino_generator.py         # OpenVINO GenAI with Registry (MODIFIED)
    ├── deferred_command_system.py     # Deferred command system
    └── event_bus.py                  # EventBus
```

## API Endpoints for Concepts/Contradictions

```python
# Concepts
/api/concepts                     # GET - список концептов
/api/concepts/<id>/facts          # GET - факты о концепте
/api/concepts/extract             # POST - извлечь из текста

# Contradictions  
/api/contradictions               # GET - список противоречий
/api/contradictions/<id>/resolve  # POST - разрешить
/api/contradictions/generate      # POST - сгенерировать

# Self-Dialog
/api/self_dialog/queue            # GET - очередь диалогов
/api/self_dialog/resolved         # GET - разрешённые знания
```

## Testing

```bash
# Test concept extraction
curl -X POST http://localhost:7860/api/concepts/extract \
  -H "Content-Type: application/json" \
  -d '{"query": "Что такое искусственный интеллект?", "response": "ИИ это технология..."}'

# Generate contradictions
curl -X POST http://localhost:7860/api/contradictions/generate \
  -H "Content-Type: application/json" \
  -d '{"concepts": ["искусственный интеллект", "автоматизация"]}'

# Check resolved knowledge
 curl http://localhost:7860/api/self_dialog/resolved
```

## Discoveries

### Архитектура EVA:
- **HybridPipelineAdapter** - центральный компонент генерации
  - Model A (qwen2.5-3b) - для кратких ответов (1024 tokens)
  - Model B (qwen2.5-3b model_b) - для развёрнутых (4096 tokens)
  - FractalPipeline - с виртуальными токенами
- **DualGenerator** - выбирает между Model A и B

### Новая система концептов:
- Концепты извлекаются из каждого запроса/ответа
- Сохраняются в FGv2 как узлы типа 'concept'
- Автоматически добавляются в очередь самодиалога
- Самодиалог исследует концепт и ищет противоречия

### Новая система противоречий:
- Противоречия генерируются автоматически для концептов
- Две противоположные точки зрения на каждый концепт
- Самодиалог разрешает противоречие через синтез
- Результаты сохраняются в кеш контекста

### Wikipedia KB интеграция:
- Доступен для обогащения концептов
- Может использоваться в самодиалоге для проверки фактов
- Интеграция в ConceptExtractor для получения определений

## Next Steps

1. ✅ ConceptExtractor - извлечение и сохранение концептов
2. ✅ ContradictionGenerator - генерация противоречий
3. ✅ DialogConceptsMixin - интеграция в самодиалог
4. ✅ Integration - связка всех компонентов
5. ✅ ModelAccessManager - координация доступа к модели
6. ✅ OpenVINOGeneratorRegistry - шаринг GPU модели
7. ✅ DeferredCommandSystem integration - SelfDialogLearning, ConceptMiner, ContradictionMiner
8. ⏳ Wikipedia KB - обогащение концептов внешними знаниями
9. ⏳ Testing - проверка работы всей системы
10. ⏳ Routes - добавление API endpoints для концептов/противоречий

## Relevant files / directories

### Core Generation:
- `eva_ai/core/hybrid_pipeline_adapter.py` - HybridPipelineAdapter
- `eva_ai/core/brain_query.py` - обработка запросов, извлечение концептов
- `eva_ai/memory/fractal_graph_v2/` - FractalGraph v2

### New Concept System:
- `eva_ai/knowledge/concept_extractor.py` - **NEW**
- `eva_ai/contradiction/contradiction_generator.py` - **NEW**
- `eva_ai/learning/dialog_concepts.py` - **NEW**
- `eva_ai/learning/dialog_core.py` - интеграция

### Routes:
- `eva_ai/gui/web_gui/server_routes.py` - нужно добавить endpoints
- `eva_ai/gui/web_gui/server_routes_graph.py` - endpoints для графа

### Knowledge:
- `eva_ai/knowledge/__init__.py` - экспорты
- `eva_ai/knowledge/kg_adapter.py` - адаптер для FGv2
- `eva_ai/knowledge/wikipedia_kb.py` - Wikipedia KB

### Config:
- `brain_config.json` - конфигурация системы
- `C:\Users\black\OneDrive\Desktop\CogniFlex` - корень проекта

## System Integration Architecture

### CoreBrain - Главный координатор

CoreBrain инициализирует и координирует все компоненты:

```
CoreBrain
├── event_bus (EventBus)                    # Центральная шина событий
├── deferred_system (DeferredCommandSystem)  # Система отложенных команд
├── two_model_pipeline (PipelineAdapter)    # UnifiedGenerator
│   ├── _openvino_cpu (OpenVINOGenerator)  # CPU для Logic/Context
│   ├── _openvino_coder (OpenVINOGenerator) # GPU для Coder/Self-dialog
│   └── _model_access (ModelAccessManager) # Координация доступа к модели
└── components
    ├── concept_miner (ConceptMiner)        # Глубокий анализ кластеров
    ├── contradiction_miner (ContradictionMiner) # Анализ противоречий
    └── self_dialog_learning (SelfDialogLearning) # Самодиалоги
```

### DeferredCommandSystem - Система отложенных команд

Централизованная система для асинхронного выполнения задач:

- **CommandPriority**: CRITICAL > HIGH > NORMAL > LOW
- **EventBus интеграция**: публикует события команд
- **Load Shedding**: сброс нагрузки при перегрузке
- **Recovery**: автоматический перезапуск упавших команд

**Используется компонентами:**
- ConceptMiner: планирование майнинг-циклов
- ContradictionMiner: проверка противоречий
- SelfDialogLearning: обработка диалогов

### ModelAccessManager - Координация доступа к модели

Предотвращает конфликты при одновременном доступе к модели:

- **Priority Queue**: CRITICAL (запросы) > HIGH (самодиалог) > NORMAL (фоновые)
- **EventBus события**: model.request, model.completed, model.failed
- **Блокировка**: предотвращает одновременный доступ к GPU/CPU

### OpenVINOGeneratorRegistry - Шаринг GPU модели

Синглтон-реестр для эффективного использования VRAM:

```python
# Если модель уже загружена на устройство - возвращается существующий инстанс
registry = get_openvino_registry()
gen = registry.get_or_create(model_path, device, config)
# gen2 получит тот же инстанс если model_path и device совпадают
gen2 = registry.get_or_create(model_path, device, config2)  # Returns gen!
```

### EventBus - Шина событий

Все компоненты публикуют и подписываются на события:

**ConceptMiner подписан на:**
- memory.graph_updated
- memory.clustering_complete
- pipeline.complete
- system.ready
- system.idle

**ContradictionMiner подписан на:**
- memory.node_created
- memory.graph_updated
- system.idle

**SelfDialogLearning подписан на:**
- curator.knowledge_extracted
- curator.graph_optimized
- curator.cleanup_done
- system.idle
- system.state_changed
- concept.confirmed
- contradiction.detected

**ModelAccessManager публикует:**
- model.request
- model.completed
- model.failed

## Правило принятия решений

**ВСЕГДА спрашивать у пользователя:**
1. Что исправлять следующим?
2. Запускать ли систему для тестирования?
3. Какой приоритет у найденных ошибок?

**Не принимать самостоятельных решений** о запуске системы или выборе задач.

## Текущий план работы

Основной план: `eva_ai/analysis/ISSUES_TRACKER.md`
- 68 проблем, ~21 исправлено
- Следующие приоритеты: 1.1.3, 1.2.2, 1.3.1
