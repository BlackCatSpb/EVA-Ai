# Перекрёстный анализ: Core + Memory + Knowledge

## Дата: 2026-04-27

---

## 1. Обзор интеграций

### 1.1 Компоненты Knowledge System

| Компонент | Файл | Назначение | Статус |
|----------|------|------------|--------|
| ConceptExtractor | knowledge/concept_extractor.py | Быстрое извлечение концептов из текста | INTEGRATED |
| ConceptMiner | knowledge/concept_miner.py | Глубокий анализ кластеров FGv2 | INTEGRATED |
| ContradictionGenerator | contradiction/contradiction_generator.py | Генерация шаблонных противоречий | PARTIAL |
| ContradictionMiner | contradiction/contradiction_miner.py | Анализ противоречий в графе | INTEGRATED |

### 1.2 Интеграция в CoreBrain (init_factories.py)

| Компонент | Строка | Интеграция |
|-----------|--------|-----------|
| ConceptExtractor | 509-520 | Создаётся в create_knowledge_graph() |
| ConceptMiner | 565-593 | Создаётся, запускается в фоне |
| ContradictionGenerator | 522-533 | Создаётся в create_knowledge_graph() |
| ContradictionMiner | 535-563 | Создаётся, запускается |
| WikipediaKB | 595-603 | Создаётся для enrichment |

---

## 2. Анализ потоков данных

### 2.1 ConceptExtractor Flow

`
Query + Response
    |
    v
.extract_concepts() [concept_extractor.py:78]
    |
    v
.save_concept_to_graph() -> FGv2 [concept_extractor.py:296]
    |
    v
queue_concept_for_dialog() -> SelfDialogLearning
`

**Вызывается из:**
- brain_query.py:1136 (_extract_key_concepts)
- hybrid_dialog_manager.py:677, 736
- unified_generator.py:1494

### 2.2 ConceptMiner Flow

`
system.idle / memory.graph_updated
    |
    v
_concept_mining_cycle() [concept_miner.py:450]
    |
    v
_detect_semantic_gaps() -> _integrate_candidate() -> FGv2
    |
    v
queue_concept_for_dialog() -> SelfDialogLearning
`

**Подписка на события:**
- system.idle
- memory.graph_updated

---

## 3. Найденные проблемы интеграции

### 3.1 ПРОБЛЕМА 1: KGAdapter не создаётся в CoreBrain

| Характеристика | Описание |
|--------------|---------|
| Проблема | KGAdapter (knowledge_graph) не инициализируется в init_factories.py |
| Файл | init_factories.py:500-609 |
| Связанные отчёты | knowledge_system.md, memory_system.md |

**Детали:**
- В create_knowledge_graph() компоненты создаются напрямую без KGAdapter
- knowledge_graph.py существует но не используется
- ConceptExtractor работает напрямую с FGv2

---

### 3.2 ПРОБЛЕМА 2: Не все методы ConceptExtractor используются

| Характеристика | Описание |
|--------------|---------|
| Проблема | save_learned_concept() не вызывается из CoreBrain |
| Файл | concept_extractor.py:487 |
| Связанные отчёты | core_generation.md |

**Методы НЕ используются:**
- save_learned_concept() - для сохранения изученных концептов
- get_concept_facts() - получение фактов о концепте
- format_concept_for_dialog() - форматирование для диалога

---

### 3.3 ПРОБЛЕМА 3: ContradictionGenerator слабо интегрирован

| Харак��еристика | Описание |
|--------------|---------|
| Проблема | ContradictionGenerator создаётся но почти не используется |
| Файл | init_factories.py:522-533 |
| Связанные отчёты | knowledge_system.md |

**Вызывается только из:**
- unified_generator.py:1509 (get_contradictions_for_prompt)
- dialog_concepts.py:838 (get_contradictions_for_prompt)

**Не используется в:**
- brain_query.py (только ConceptExtractor)
- hybrid_dialog_manager.py (частично)

---

### 3.4 ПРОБЛЕМА 4: DialogConceptsMixin не инициализирован в CoreBrain

| Характеристика | Описание |
|--------------|---------|
| Проблема | DialogConceptsMixin не подключён к SelfDialogLearning |
| Файл | learning/dialog_concepts.py, learning/dialog_core.py |
| Связанные отчёты | self_dialog_system.md |

**Детали:**
- DialogConceptsMixin миксин должен добавляться в SelfDialogLearning
- Методы queue_concept_for_dialog(), queue_contradiction_for_resolution() не работают
- Проверить инициализацию в start_webgui.py

---

### 3.5 ПРОБЛЕМА 5: Заглушки в KGAdapter

| Характеристика | Описание |
|--------------|---------|
| Проблема | kg_adapter.py:166-170 - __getattr__ возвращает None-функцию |
| Файл | knowledge/kg_adapter.py:166-170 |
| Связанные отчёты | knowledge_system.md |

---

## 4. Используемые vs Неиспользуемые методы

### 4.1 ConceptExtractor (из 30 методов)

| Метод | Строка | Используется |
|-------|--------|------------|
| extract_concepts() | 78 | YES |
| save_concept_to_graph() | 296 | YES |
| get_concepts_for_prompt() | 415 | YES |
| save_learned_concept() | 487 | NO |
| get_concept_facts() | 391 | NO |
| format_concept_for_dialog() | 543 | NO |
| _extract_terms() | 110 | YES (внутри) |
| _create_concept() | 133 | YES (внутри) |
| _generate_facts() | 183 | YES (внутри) |

### 4.2 ContradictionGenerator

| Метод | Используется |
|-------|------------|
| generate_contradiction() | NO |
| get_contradictions_for_prompt() | PARTIAL |

---

## 5. Связи между системами

### 5.1 Knowledge -> Memory

`
ConceptExtractor ---> FGv2 (add_node)
ConceptMiner -----> FGv2 (get_clusters)
WikipediaKB ----> FGv2 (add_node)
`

### 5.2 Knowledge -> Self-Dialog

`
ConceptExtractor ---> queue_concept_for_dialog()
ConceptMiner ----> queue_concept_for_dialog()
ContradictionGenerator -> queue_contradiction_for_resolution()
`

### 5.3 Core -> Knowledge

`
brain_query.py
    ├── .extract_concepts() -> ConceptExtractor
    └── .queue_concept_for_dialog() -> SelfDialogLearning
`

---

## 6. Рекомендации по интеграции

### 6.1 Высокий приоритет

1. **Подключить DialogConceptsMixin**
   - Файл: learning/dialog_core.py
   - Действие: Проверить наследование SelfDialogLearning
   - Проверить start_webgui.py

2. **Интегрировать KGAdapter**
   - Файл: init_factories.py:500-510
   - Действие: Создать KnowledgeGraphAdapter в create_knowledge_graph()

3. **Активировать ContradictionGenerator**
   - Файл: brain_query.py
   - Действие: Использовать для генерации противоречий после извлечения концепта

### 6.2 Средний приоритет

1. **Использовать save_learned_concept()**
   - Вызывать после self_dialog learning

2. **Интегрирова��ь WikipediaKB в ConceptExtractor**
   - Обогащать концепты определениями из Wikipedia

3. **Исправить _update_lifecycle() в ConceptMiner**
   - Реализовать полный жизненный цикл

---

## 7. Связанные отчёты

| Отчёт | Описание |
|-------|----------|
| core_generation.md | HybridPipeline, DualGenerator |
| memory_system.md | FractalGraphV2 |
| knowledge_system.md | ConceptExtractor/Miner |
| fcp_system.md | FCP |
| self_dialog_system.md | Самодиалог |

---

## 8. Выводы

### Интеграция: ЧАСТИЧНАЯ

**Работает:**
- ConceptExtractor -> FGv2 (сохранение концептов)
- ConceptExtractor -> SelfDialogLearning (очередь концептов)
- ConceptMiner -> запуск в фоне
- brain_query -> ConceptExtractor

**Не работает / Частично:**
- ContradictionGenerator (создаётся но не используется)
- DialogConceptsMixin (не подключён)
- KGAdapter (не создаётся)
- save_learned_concept() (не вызывается)

### Следующие шаги

1. Проверить инициализацию DialogConceptsMixin в start_webgui.py
2. Подключить ContradictionGenerator в brain_query
3. Создать KGAdapter в init_factories

---

*Анализ проведён: 2026-04-27*
*Версия EVA: Current*
