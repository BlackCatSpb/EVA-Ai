# Аудит KnowledgeGraphCore системы EVA AI

**Дата аудита:** 2026-04-14  
**Аудитор:** EVA AI System Analyzer  
**Версия EVA:** Current  

---

## Резюме

| Компонент | Статус | Оценка |
|-----------|--------|--------|
| KnowledgeGraph (обёртка) | Используется для совместимости | 7/10 |
| FractalGraphV2 (реализация) | Основное хранилище | 8/10 |
| KnowledgeGraphAdapter | Прослойка совместимости | 6/10 |
| EventBus интеграция | Частичная (через компоненты) | 5/10 |
| Использование в системе | Широко распространено | 8/10 |

---

## 1. Найденная реализация KnowledgeGraphCore

### 1.1 Архитектура: Три уровня

- **KnowledgeGraph** (eva_ai/knowledge/knowledge_graph.py) - Обёртка для совместимости со старым API
- **KnowledgeGraphAdapter** (eva_ai/knowledge/kg_adapter.py) - Адаптер для API-совместимости
- **FractalMemoryGraph** (eva_ai/memory/fractal_graph_v2/__init__.py) - ОСНОВНАЯ реализация

### 1.2 Файловая структура

| Файл | Назначение | Строк |
|------|------------|-------|
| eva_ai/knowledge/knowledge_graph.py | Обёртка-совместимость | 71 |
| eva_ai/knowledge/kg_adapter.py | Адаптер вызовов | 171 |
| eva_ai/memory/fractal_graph_v2/__init__.py | FractalMemoryGraph | 1262 |
| eva_ai/memory/fractal_graph_v2/storage.py | FractalGraphV2 (SQLite) | 1294 |

---

## 2. Отличие от FractalGraphV2

### 2.1 Ключевые различия

| Характеристика | KnowledgeGraph | FractalGraphV2 |
|----------------|----------------|-----------------|
| **Тип** | Обёртка (wrapper) | Реальная реализация |
| **API** | Старое KG API | Новое FGv2 API |
| **Хранилище** | Через адаптер | SQLite + векторные индексы |
| **Семантический поиск** | Перенаправление | Встроенный (cosine similarity) |
| **Кластеризация** | Перенаправление | Встроенная (3 метода) |
| **Противоречия** | Перенаправление | Встроенная детекция |

### 2.2 KnowledgeGraphAdapter - детали

**Проблемы:**
1. Метод add_edge() вызывает edge_type=relation_type - НЕВЕРНОЕ ИМЯ ПАРАМЕТРА (kg_adapter.py:74)
2. Метод find_path_between_concepts() имеет баг с итерацией по edges (kg_adapter.py:132-140)
3. Метод __getattr__ возвращает заглушку - скрывает ошибки

### 2.3 FractalGraphV2 Storage - детали

**Сильные стороны:**
- SQLite с полной схемой (nodes, edges, semantic_groups)
- Векторные индексы с кэшированием нормализованных эмбеддингов
- Три метода кластеризации: agglomerative, dbscan, simple
- Встроенная детекция противоречий по косинусному расстоянию
- Сериализация в .eva контейнеры

---

## 3. EventBus интеграция

### 3.1 Текущее состояние

**НАПРЯМУЮ EventBus НЕ ИСПОЛЬЗУЕТСЯ** в FractalGraphV2. Интеграция реализована **на уровне компонентов**, использующих граф.

### 3.2 Компоненты с EventBus подпиской

#### ConceptMiner
- memory.graph_updated
- memory.clustering_complete
- pipeline.complete
- system.ready
- system.idle

#### ContradictionMiner
- memory.node_created
- memory.graph_updated
- system.idle

#### WebGUI Bridge
- KNOWLEDGE_GRAPH_UPDATED
- MEMORY_GRAPH_UPDATED

### 3.3 Проблемы EventBus интеграции

| Проблема | Серьёзность | Описание |
|----------|-------------|----------|
| FGv2 не публикует события | Высокая | При добавлении узлов/связей нет publish |
| Нет стандартных событий | Средняя | memory.node_created, memory.graph_updated не генерируются FGv2 |
| Ручная синхронизация | Средняя | Компоненты должны вручную подписываться |

---

## 4. Использование в системе

### 4.1 Карта использования

- brain_components.py - knowledge_graph property
- init_factories.py - create_knowledge_graph()
- concept_miner.py - для майнинга концептов
- contradiction_miner.py - для майнинга противоречий
- unified_cache_bridge.py - объединение cache и graph
- gui/knowledge_graph_module.py - визуализация
- reasoning engines - контекст для рассуждений
- HealthMonitor - мониторинг здоровья

### 4.2 Статистика использования

- Всего файлов с ссылками на knowledge_graph: ~40+
- brain.knowledge_graph прямых ссылок: ~15
- Статические импорты: ~8
- Динамические getattr: ~20

---

## 5. Оценка архитектуры

### 5.1 Плюсы

1. **Совместимость**: Обёртка KG позволяет старому коду работать с FGv2
2. **Разделение хранилища**: FractalGraphV2 отдельно от бизнес-логики
3. **Богатый API**: semantic_search, clustering, contradiction detection
4. **Производительность**: LRU кэш для поиска, batch операции
5. **Интеграция с EventBus**: через компоненты-подписчики

### 5.2 Минусы

1. **Скрытая сложность**: KnowledgeGraph это обёртка - неочевидно
2. **Дублирование функциональности**: KGAdapter добавляет уровень косвенности
3. **Неполная EventBus интеграция**: FGv2 не генерирует события
4. **Баги в адаптере**: неверные имена параметров
5. **Нет стандарт событий**: для node_created, edge_created и т.д.

### 5.3 Итоговая оценка: 7/10

| Критерий | Балл | Максимум | Комментарий |
|----------|------|----------|-------------|
| Функциональность | 8 | 10 | Полный набор операций с графом |
| Архитектура | 6 | 10 | Слишком много слоёв абстракции |
| Производительность | 8 | 10 | Кэширование, batch операции |
| Интеграция EventBus | 5 | 10 | Частичная, нет событий от FGv2 |
| Документация | 6 | 10 | Обёртка не документирована |
| Удобство использования | 7 | 10 | Хороший API, но запутанная структура |

---

## 6. Рекомендации

### 6.1 Критические исправления

1. **Исправить баг в kg_adapter.py:74**
   - Текущий код: self._fg.add_edge(source, target, edge_type=relation_type)
   - Должно быть: self._fg.add_edge(source_id=source, target_id=target, relation_type=relation_type)

2. **Добавить EventBus события в FractalGraphV2**
   - После добавления узла публиковать memory.node_created
   - После добавления связи публиковать memory.edge_created

### 6.2 Улучшения архитектуры

1. Документировать обёртку: добавить docstring о том что KnowledgeGraph это wrapper
2. Упростить структуру: возможно убрать один слой абстракции
3. Стандартизировать Events: все компоненты должны публиковать стандартные события

### 6.3 Долгосрочные улучшения

1. Рассмотреть прямое использование FractalMemoryGraph вместо KnowledgeGraph
2. Добавить больше метрик и мониторинга в графовую подсистему
3. Реализовать более эффективный batch processing

---

## 7. Заключение

**KnowledgeGraphCore** представляет собой многослойную систему с обёрткой для обратной совместимости. Основная реализация - **FractalGraphV2** - является хорошо продуманной системой хранения знаний с семантическим поиском и кластеризацией.

**Проблемы:**
- Слишком много слоёв абстракции (KnowledgeGraph -> KGAdapter -> FractalMemoryGraph -> FractalGraphV2)
- Неполная интеграция с EventBus на уровне хранилища
- Баги в коде адаптера

**Рекомендация:** Использовать напрямую FractalMemoryGraph/FractalGraphV2 где возможно, чтобы избежать накладных расходов от обёрток.

---

*Отчёт сгенерирован EVA AI System Analyzer*
