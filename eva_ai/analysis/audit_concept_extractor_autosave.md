# ConceptExtractor Auto-Save Audit

## Дата аудита
2026-04-14

## Агент
ConceptExtractor Auto-Save Audit

## Оценка
**4.5/10**

---

## Основные находки

### 1. ConceptExtractor.extract_concepts() ТОЛЬКО возвращает список - НЕ сохраняет

**Проблема:**
Метод `extract_concepts()` в `ConceptExtractor` только извлекает концепты и возвращает их как список, но **не выполняет сохранение** в FractalGraph v2.

**Текущее поведение:**
```python
def extract_concepts(self, query: str, response: str) -> List[ConceptResult]:
    # ... анализ текста ...
    return concepts  # Только возврат, без сохранения
```

**Ожидаемое поведение:**
Метод должен сам вызывать `save_concept_to_graph()` для каждого извлечённого концепта.

---

### 2. Сохранение делается вручную в brain_query._extract_key_concepts()

**Проблема:**
Сохранение концептов вынесено в `brain_query.py`, что создаёт дублирование логики и нарушает инкапсуляцию.

**Текущее расположение:**
```python
# brain_query.py - process_query()
concepts = concept_extractor.extract_concepts(query, response)
for concept in concepts:
    concept_extractor.save_concept_to_graph(concept)  # Вручную!
    self.dialog_concepts_mixin.queue_concept_for_dialog(...)
```

**Проблема:** Это должно быть внутри `ConceptExtractor`, а не внешний код.

---

### 3. Нарушение Single Responsibility Principle (SRP)

**Проблема:**
`brain_query.py` отвечает за обработку запросов AND за сохранение концептов. Это смешение обязанностей.

**Последствия:**
- Дублирование кода сохранения если `extract_concepts()` вызывается из другого места
- Сложность поддержки - изменения в логике сохранения нужно вносить в нескольких местах
- Риск того, что `extract_concepts()` будет использоваться без сохранения

---

### 4. Нет EventBus событий при извлечении концептов

**Проблема:**
При извлечении концептов не публикуются события в EventBus, что делает невозможным:
- Отслеживание активности извлечения концептов
- Реакцию других компонентов на новые концепты
- Логирование и мониторинг

**Рекомендуемые события:**
```python
self.event_bus.publish("concept.extracted", {
    "concepts": concepts,
    "query": query,
    "timestamp": datetime.now()
})
```

---

## Рекомендации по исправлению

### 1. Интегрировать сохранение в extract_concepts()

```python
def extract_concepts(self, query: str, response: str) -> List[ConceptResult]:
    # ... анализ текста ...
    
    # Автосохранение
    for concept in concepts:
        self.save_concept_to_graph(concept)
        
    # Публикация события
    if self.event_bus:
        self.event_bus.publish("concept.extracted", {
            "concepts": concepts,
            "count": len(concepts),
            "query": query
        })
    
    return concepts
```

### 2. Убрать ручное сохранение из brain_query.py

После интеграции автосохранения, убрать цикл сохранения из `brain_query._extract_key_concepts()`.

### 3. Добавить EventBus.publish() при извлечении

Использовать существующий EventBus для публикации событий `concept.extracted`.

---

## Приоритет исправления
**HIGH** - нарушение SRP и потенциальные проблемы с консистентностью данных

---

## Файлы для изменения
1. `eva_ai/knowledge/concept_extractor.py` - добавить автосохранение и события
2. `eva_ai/core/brain_query.py` - убрать ручное сохранение
