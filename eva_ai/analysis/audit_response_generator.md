# Аудит ResponseGenerator

**Дата:** 2026-04-14
**Общая оценка: 3/10**

## Критический баг: extract_ambiguous_terms()

Метод ВЫЗЫВАЕТСЯ но НЕ СУЩЕСТВУЕТ!

### Места вызова:
- core/response_generator.py:191
- contradiction/core_detection.py: 321, 322, 362, 363, 655
- memory/manager_operations.py: 436

### Проверка наличия:
- knowledge/context_entity.py - НЕТ метода
- reasoning/entity_extractor.py - НЕТ метода

## Интеграция brain_query

brain_query.py НЕ использует ResponseGenerator напрямую!
Fallback chain имеет 7 уровней, RG не входит.

## Рекомендации

1. Удалить вызовы extract_ambiguous_terms() или реализовать метод
2. Интегрировать RG в brain_query как primary generator

## Файлы
- eva_ai/core/response_generator.py
- eva_ai/core/brain_query.py
- eva_ai/knowledge/context_entity.py