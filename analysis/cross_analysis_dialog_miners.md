# Перекрёстный анализ: Self-Dialog + Miners

**Дата:** 2026-04-27
**Аналитик:** AI Архитектор EVA
**Версия:** EVA AI v2
---

## Содержание

1. [Обзор системы](#1-обзор-системы)
2. [Дублирование функционала](#2-дублирование-функционала)
3. [Проблемы интеграции](#3-проблемы-интеграции)
4. [Связанные отчёты](#4-связанные-отчёты)
5. [Выводы и рекомендации](#5-выводы-и-рекомендации)

---

## 1. Обзор системы

### 1.1 Поток данных Miners -> SelfDialogLearning

`
ConceptMiner                              SelfDialogLearning
     │                                          │
     ├──queue_concept_for_dialog() ───────────>│ (очередь концептов)
     │                                          │
     └──trigger_self_dialog(reason=) ─────────>│ (триггер обработки)
     │
ContradictionMiner                         
     │                                          
     ├──queue_contradiction_for_resolution() ->│ (очередь противоречий)
     │                                          
     └──trigger_self_dialog(reason=) ─────────>│ (триггер обработки)
`

### 1.2 Вызовы методов Integration

| Источник | Метод | Назначение | Статус |
|----------|-------|------------|--------|
| **ConceptMiner** | queue_concept_for_dialog() | Добавляет подтверждённый концепт | ✅ Вызывается |
| **ConceptMiner** | 	rigger_self_dialog() | Триггер после концепта | ✅ Вызывается |
| **ContradictionMiner** | queue_contradiction_for_resolution() | Добавляет противоречие | ✅ Вызывается |
| **ContradictionMiner** | 	rigger_self_dialog() | Триггер после детекции | ✅ Вызывается |
| **brain_query** | queue_concept_for_dialog() | Очередь из запроса | ✅ Вызывается |
| **brain_query** | 	rigger_self_dialog() | Триггер | ✅ Вызывается |

---

## 2. Дублирование функционала

### 2.1 ContradictionGenerator vs ContradictionMiner vs Legacy Detect

| Функция | ContradictionGenerator | ContradictionMiner | Legacy Detect | SelfDialogLearning |
|--------|---------------------|-------------------|-------------|-------------------|
| **Детекция** | ❌ (шаблоны) | ✅ (математика+NLI) | ✅ (semantic/logical/temporal) | ❌ |
| **Генерация** | ✅ (шаблоны) | ❌ | ❌ | ❌ |
| **Разрешение** | ❌ | ❌ | ✅ (core_resolution) | ✅ |
| **Сохранение** | ✅ (кеш) | ✅ (FGv2) | ✅ | ✅ (FGv2+кеш) |

### 2.2 Критическое дублирование

**Проблема: Три отдельные системы детекции противоречий**

1. **ContradictionGenerator** (быстрый уровень)
   - Файл: contradiction/contradiction_generator.py
   - Метод: шаблоны positive/negative
   - Выход: GeneratedContradiction объект

2. **ContradictionMiner** (аналитический уровень)
   - Файл: contradiction/contradiction_miner.py
   - Метод: sim ≥ 0.75, contra ≥ 0.65 + NLI
   - Выход: ContradictionNode в FGv2

3. **Legacy Detect System** (дополнительные модули)
   - detect_semantic.py - семантическое сравнение
   - detect_logical.py - логический анализ
   - detect_temporal.py - временные противоречия

### 2.3 Мёртвый код

**ContradictionGenerator строки 401-433:**
`python
def _format_contradiction_prompt(self, concept_name: str, contradictions: list) -> str:
    # ... основная функция ...
    
    try:  # ← ЭТОТ БЛОК НИКОГДА НЕ ВЫЗЫВАЕТСЯ!
        cm = getattr(self.brain, 'contradiction_manager', None)
        # ...
`
**Статус:** Мёртвый код - никогда не выполняется

---

## 3. Проблемы интеграции

### 3.1 DialogConceptsMixin - Интегрированные методы

| Метод | Интеграция | Использование |
|-------|------------|--------------|
| queue_concept_for_dialog() | ✅ | ConceptMiner, brain_query |
| queue_contradiction_for_resolution() | ✅ | ContradictionMiner |
| _get_next_dialog_topic() | ✅ | _worker_loop |
| _run_concept_dialog() | ✅ | _process_with_dual_circuit_batch |
| _run_contradiction_dialog() | ✅ | _process_with_dual_circuit_batch |
| _save_concept_dialog_results() | ✅ | _run_concept_dialog |
| _save_contradiction_resolution() | ✅ | _run_contradiction_dialog |
| _save_learned_facts_to_fg() | ✅ | _save_contradiction_resolution |
| get_resolved_knowledge() | ⚠️ | Не используется в brain_query |
| extract_knowledge_from_cache() | ⚠️ | Не используется в цикле генерации |
| get_context_for_generation() | ⚠️ | Вызывается, но есть баг |

### 3.2 DialogConceptsMixin - НЕ Интегрированные методы

**Метод:** get_resolved_knowledge(limit) (строка 705-729)
- **Проблема:** Нигде не вызывается явно
- **Использование:** Не используется в brain_query
- **Рекомендация:** Добавить в rain_query._execute_with_web_search()

**Метод:** extract_knowledge_from_cache(concept) (строка 731-783)
- **Проблема:** Вызывается косвенно через get_context_for_generation(), но есть баг
- **Баг:** Строки 715-722 - get_recent метод может отсутствовать
- **Рекомендация:** Добавить fallback для get_recent

**Метод:** get_context_for_generation(query) (строка 811-858)
- **Используется в:** brain_query:401
- **Проблема:** Зависит от concept_extractor и contradiction_generator
- **Рекомендация:** Добавить проверку наличия атрибутов

### 3.3 Интеграция ConceptMiner -> SelfDialogLearning

**Файл:** knowledge/concept_miner.py:938-947

`python
# Добавляем в очередь самодиалога и триггерим
if hasattr(self.brain, 'self_dialog_learning') and self.brain.self_dialog_learning:
    self.brain.self_dialog_learning.queue_concept_for_dialog(
        candidate.title,
        priority=candidate.confidence
    )
    # Триггер для запуска self-learning по требованию
    try:
        if hasattr(self.brain.self_dialog_learning, 'trigger_self_dialog'):
            self.brain.self_dialog_learning.trigger_self_dialog(reason='concept_mined')
    except Exception as e:
        logger.debug(f"Trigger error: {e}")
`

**Статус:** ✅ РАБОТАЕТ

### 3.4 Интеграция ContradictionMiner -> SelfDialogLearning

**Файл:** contradiction/contradiction_miner.py:850-860

`python
# Добавляем в очередь самодиалога и триггерим
if hasattr(self.brain, 'self_dialog_learning') and self.brain.self_dialog_learning:
    self.brain.self_dialog_learning.queue_contradiction_for_resolution(
        contr_id=node_id,
        concept=candidate.title,
        priority=candidate.priority
    )
    # Триггер для запуска self-learning по требованию
    try:
        if hasattr(self.brain.self_dialog_learning, 'trigger_self_dialog'):
            self.brain.self_dialog_learning.trigger_self_dialog(reason='contradiction_detected')
    except Exception as e:
        logger.debug(f"Trigger error: {e}")
`

**Статус:** ✅ РАБОТАЕТ

### 3.5 Интеграция brain_query -> SelfDialogLearning

**Файл:** core/brain_query.py:1145-1152

`python
if hasattr(self, 'self_dialog_learning') and self.self_dialog_learning:
    self.self_dialog_learning.queue_concept_for_dialog(
        concept_name, priority=data.get('confidence', 0.5)
    )
    if hasattr(self.self_dialog_learning, 'trigger_self_dialog'):
        self.self_dialog_learning.trigger_self_dialog(reason='query_concept')
`

**Статус:** ✅ РАБОТАЕТ

---

## 4. Связанные отчёты

### 4.1 self_dialog_system.md

**Ключевые выводы:**
- Ошибка в dialog_core.py:1049 (summary_parts не определена)
- 2 заглушки с pass
- Нет DialogOrchestrator

### 4.2 knowledge_system.md

**Ключевые выводы:**
- ConceptMiner работает ✅
- ContradictionGenerator: мёртвый код строки 401-433
- kg_adapter: __getattr__ возвращает None-функцию

### 4.3 contradiction_legacy_system.md

**Ключевые выводы:**
- Detect System: semantic, logical, temporal - все используются
- Learn System: используется
- Tracking & Resolution: используется

---

## 5. Выводы и рекомендации

### 5.1 Итоговая таблица

| Параметр | Оценка |
|----------|--------|
| Интеграция Miners -> SDL | ✅ 8/10 |
| Интеграция brain_query -> SDL | ✅ 9/10 |
| Дублирование функционала | ⚠️ КРИТИЧЕСКОЕ |
| Использование кеша | ❌ 4/10 |
| Общая оценка | ⚠️ 6/10 |

### 5.2 Критические проблемы

| # | Проблема | Файл | Строка | Приоритет |
|---|----------|------|--------|-----------|
| 1 | Мёртвый код ContradictionGenerator | contradiction_generator.py | 401-433 | HIGH |
| 2 | Extract knowledge не используется | dialog_concepts.py | 731-783 | MEDIUM |
| 3 | get_resolved_knowledge не используется | dialog_concepts.py | 705-729 | MEDIUM |
| 4 | Три системы детекции противоречий | contradict_*/ + detect_*/ | - | HIGH |

### 5.3 Рекомендации

1. **Удалить мёртвый код** в ContradictionGenerator (строки 401-433)

2. **Добавить использование** extract_knowledge_from_cache() в brain_query для обогащения контекста

3. **Объединить или удалить** одну из систем детекции противоречий:
   - Вариант A: Оставить ContradictionMiner + Legacy Detect
   - Вариант B: Оставить ContradictionGenerator + Legacy Detect

4. **Добавить fallback** для get_recent метода в кеше

5. **Исправить ошибку** в dialog_core.py:1049 (summary_parts)

---

## Связанные файлы

| Файл | Назначение |
|------|-------------|
| eva_ai/learning/dialog_core.py | SelfDialogLearning |
| eva_ai/learning/dialog_concepts.py | DialogConceptsMixin (858 строк) |
| eva_ai/knowledge/concept_miner.py | ConceptMiner (1036 строк) |
| eva_ai/contradiction/contradiction_miner.py | ContradictionMiner (928 строк) |
| eva_ai/contradiction/contradiction_generator.py | ContradictionGenerator (479 строк) |
| eva_ai/contradiction/detect_*.py | Legacy Detect системы |
| eva_ai/core/brain_query.py | Обработка запросов |

---

*Анализ: AI Архитектор EVA*
