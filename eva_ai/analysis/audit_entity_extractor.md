# Аудит EntityExtractor в EVA AI

**Дата:** 2026-04-14
**Аудитор:** EVA AI Analysis System

---

## Резюме

| Параметр | Значение |
|----------|----------|
| Найдено версий | 4 |
| Критических ошибок | 1 |
| Дублирование | ДА |
| EventBus | НЕТ |
| Оценка | 3/10 |
---

## 1. Найденные версии EntityExtractor

### 1.1 reasoning/entity_extractor.py (ОСНОВНОЙ)
**398 строк** - Полноценный извлекатель на regex-паттернах

**Методы:**
- extract_from_query(query) -> ExtractionResult
- extract_from_response(response) -> ExtractionResult
- extract_from_contradiction(contradiction, weight, is_rejected)
- extract_all(query, response, contradictions, weights)
- save_to_knowledge_graph(entities, knowledge_graph)
- format_for_self_learning(entities)

### 1.2 knowledge/context_entity.py (ОБЁРТКА)
**165 строк** - Обёртка над reasoning.EntityExtractor

**Методы:**
- extract_entities(text)
- extract_from_query(query)
- extract_from_response(response)
- extract_all(query, response, contradictions, weights)
- resolve_ambiguity(entity, context)
- find_related_entities(entity, limit)
- save_to_knowledge_graph(entities)

**Вывод:** Просто обёртка + AmbiguousEntity классы

### 1.3 gui/web_gui/server_auth.py:EntityExtractor (ПРИМИТИВНЫЙ)
**~41 строка** - Примитивный keyword-based извлекатель

**Методы:**
- extract_entities(text) - простой поиск ключевых слов
- is_personal_info(text)

### 1.4 preprocess/preprocessing_pipeline.py:GGUFEntityExtractor
**~188 строк** - LLM-базированный извлекатель

**Методы:**
- extract_entities(query, session_context)
- check_clarification_needed(query, entities, session_context)
---

## 2. Использование в системе

| Компонент | Импорт | Версия |
|----------|--------|--------|
| reasoning/enhanced_reasoning_engine.py | eva_ai.reasoning.entity_extractor | reasoning.EntityExtractor |
| core/processor_core.py | eva_ai.knowledge.context_entity | context_entity.EntityExtractor |
| core/response_generator.py | eva_ai.knowledge.context_entity | context_entity.EntityExtractor |
| contradiction/core_detection.py | eva_ai.knowledge.context_entity | context_entity.EntityExtractor |
| memory/manager_core.py | eva_ai.knowledge.context_entity | context_entity.EntityExtractor |
| gui/web_gui/server_main.py | eva_ai.gui.web_gui.server_auth | server_auth.EntityExtractor |
| preprocess/preprocessing_pipeline.py | (локальный) | GGUFEntityExtractor |

---

## 3. Дублирование логики

**ПРОБЛЕМА:** knowledge/context_entity.py полностью дублирует reasoning/entity_extractor.py

knowledge/context_entity.py просто обёртывает reasoning.EntityExtractor и не добавляет
новой функциональности, кроме fallback на FGv2 semantic_search.

---

## 4. КРИТИЧЕСКАЯ ОШИБКА

### 4.1 extract_ambiguous_terms() НЕ СУЩЕСТВУЕТ

**Где вызывается (7 мест):**

- contradiction/core_detection.py: Lines 321, 322, 362, 363, 655
- memory/manager_operations.py: Line 436
- core/response_generator.py: Line 191

**Проверка:** Метод НЕ определён ни в одном классе EntityExtractor!

| Класс | extract_ambiguous_terms? |
|-------|--------------------------|
| reasoning.entity_extractor.EntityExtractor | **НЕТ** |
| knowledge.context_entity.EntityExtractor | **НЕТ** |
| gui.server_auth.EntityExtractor | **НЕТ** |
| GGUFEntityExtractor | **НЕТ** |

**При вызове будет AttributeError!**
---

## 5. EventBus Интеграция

**НЕТ** - Все 4 класса полностью изолированы от EventBus.

EntityExtractor не публикует и не подписывается на события.

---

## 6. Использование результатов

**Только enhanced_reasoning_engine реально сохраняет сущности.**

Остальные компоненты:
- core/response_generator.py: вызывает несуществующий метод
- contradiction/core_detection.py: вызывает несуществующий метод
- memory/manager_operations.py: вызывает несуществующий метод
- preprocess/preprocessing_pipeline.py: извлекает но НЕ сохраняет

---

## 7. Оценка: 3/10

| Критерий | Оценка |
|----------|--------|
| Функциональность | 4/10 |
| Дублирование | 2/10 |
| Интеграция | 2/10 |
| Архитектура | 3/10 |
| Поддержка | 1/10 |

---

## 8. Рекомендации

### КРИТИЧЕСКИЕ:

1. **Исправить или удалить вызовы extract_ambiguous_terms()** в 7 местах
   - contradiction/core_detection.py: удалить строки 321-322, 362-363, 655
   - core/response_generator.py: удалить _detect_ambiguity_before_response
   - memory/manager_operations.py: удалить extract_entities_from_text

2. **Удалить knowledge/context_entity.py** (дублирует reasoning.entity_extractor)

3. **Обновить импорты в 4 файлах:**
   - core/processor_core.py:25
   - core/response_generator.py:50
   - contradiction/core_detection.py:20
   - memory/manager_core.py:16

### ВЫСОКИЙ ПРИОРИТЕТ:

4. **Интегрировать EventBus** в reasoning/entity_extractor.py

5. **Использовать результаты GGUFEntityExtractor** (сохранять в KG)

### СРЕДНИЙ ПРИОРИТЕТ:

6. **Заменить GUI EntityExtractor** на полноценный из reasoning

---

## 9. План действий

| # | Действие | Приоритет |
|---|----------|-----------|
| 1 | Исправить вызовы extract_ambiguous_terms() | КРИТИЧЕСКИЙ |
| 2 | Удалить knowledge/context_entity.py | ВЫСОКИЙ |
| 3 | Обновить импорты в 4 файлах | ВЫСОКИЙ |
| 4 | Интегрировать EventBus | ВЫСОКИЙ |
| 5 | Использовать результаты GGUFEntityExtractor | СРЕДНИЙ |
| 6 | Заменить GUI EntityExtractor | СРЕДНИЙ |

---

## 10. Выводы

1. **4 версии EntityExtractor** - массовое дублирование
2. **Критический баг** - вызов несуществующего метода в 7 местах
3. **Нет EventBus интеграции** - изоляция от системы событий
4. **Результаты не используются** - извлечение есть, сохранение нет
5. **Обёртка над обёрткой** - knowledge/context_entity обёртывает reasoning

**Общая оценка: 3/10** - требуется срочная рефакторизация

---

*Отчёт сгенерирован EVA AI Analysis System*