# Анализ Knowledge System EVA

## Компоненты

| Модуль | Файл | Назначение |
|--------|------|------------|
| knowledge | concept_extractor.py | Быстрое извлечение концептов |
| knowledge | concept_miner.py | Глубокий майнинг лакун |
| contradiction | contradiction_generator.py | Генерация шаблонных противоречий |
| contradiction | contradiction_miner.py | Майнинг противоречий в графе |

## ConceptExtractor (Быстрый уровень)

**Методы:**
- `extract_concepts(query, response)` - извлечение из текста
- `save_concept_to_graph(concept)` - сохранение в FGv2
- `get_concepts_for_prompt(query)` - контекст для генерации

## ConceptMiner (Глубокий уровень)

**Триггеры:** system.idle, memory.graph_updated

**Методы:**
- `_detect_semantic_gaps(clusters)` - детекция лакун
- `_generate_hypothesis(candidate)` - генерация через LLM
- `_validate_candidate(candidate)` - NLI + Ethics + Web
- `_integrate_candidate(candidate)` - сохранение в FGv2

## ContradictionGenerator

**Методы:**
- `generate_contradiction(concept_name, domain)` - шаблоны
- `get_contradictions_for_prompt(concept)` - контекст

## ContradictionMiner

**Методы:**
- `_detect_candidate_pairs()` - sim ≥ 0.75, contra ≥ 0.65
- `_cluster_pairs(pairs)` - транзитивное замыкание
- `_create_contradiction_node(candidate)` - узел в FGv2

## Заглушки и проблемы

1. **contradiction_generator.py:401-433** - мёртвый код (дублирование)
2. **kg_adapter.py:166-170** - `__getattr__` возвращает None-функцию
3. **concept_miner.py** - `_update_lifecycle()` частично мёртвый

## Выводы

Система работает. Нужно удалить дублирующий код в ContradictionGenerator.