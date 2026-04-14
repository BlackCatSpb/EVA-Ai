# Отчёт: brain_query обработка
## 1. Импорты

**Файл**: va_ai/core/brain_query.py

### Используемые импорты:
`python
import re
import time
import logging
import random
import threading
from typing import Dict, Any, Optional, List
`

### Прямые импорты concept_extractor и contradiction_generator: **НЕТ**

- ConceptExtractor импортируется в init_factories.py (строка 512)
- ContradictionGenerator импортируется в init_factories.py (строка 525)
- В brain_query.py эти компоненты используются **косвенно** через self_dialog_learning

---

## 2. Предобработка

### Что定义了 как предобработка:
1. **Greeting check** (строка 180-192): Быстрый ответ на приветствия через кэш GREETING_RESPONSES
2. **Query cache check** (строка 195-200): Проверка кэшированных ответов
3. **Web search decision** (строка 349): 
eeds_web_search(query) - определяет нужен ли поиск

### Реальная предобработка (ПОСЛЕ генерации):
Метод _extract_key_concepts (строка 1399-1459) вызывается **ПОСЛЕ** получения ответа:
`python
# Извлекаем концепты из запроса и ответа для обучения
try:
    self._extract_key_concepts(query, result.get('response', ''))
except Exception as e:
    query_logger.debug(f"Concept extraction error: {e}")
`

### Проблема:
**Нет предобработки ДО генерации!** Концепты извлекаются после ответа, а не до.

---

## 3. Построение контекста

### Место: _handle_gguf_pipeline (строки 366-386)

`python
# Добавляем контекст из концептов и противоречий
concepts_context = ""
if hasattr(self, 'self_dialog_learning') and self.self_dialog_learning:
    try:
        concepts_context = self.self_dialog_learning.get_context_for_generation(query)
        if concepts_context:
            query_logger.debug(f"Added concepts/contradictions context: {len(concepts_context)} chars}")
    except Exception as e:
        query_logger.debug(f"Error getting concepts context: {e}")

# Добавляем контекст от Tavily к запросу
enhanced_query = query
if concepts_context:
    enhanced_query = concepts_context + "\n\n" + query

if search_results:
    web_context = "\n\nДополнительная информация из интернета:\n"
    # ... форматирование результатов поиска
    enhanced_query = enhanced_query + web_context
`

### Формирование context_parts в get_context_for_generation (dialog_concepts.py):
`python
# 1. Получаем концепты из запроса
if self.brain and hasattr(self.brain, 'concept_extractor'):
    concepts_context = self.brain.concept_extractor.get_concepts_for_prompt(query)

# 2. Проверяем противоречия для извлечённых концептов
if self.brain and hasattr(self.brain, 'contradiction_generator'):
    # Извлекаем термины из запроса
    terms = re.findall(r'\b[а-яёa-z]{4,}\b', query.lower())
    for term in terms[:3]:
        contr_context = self.brain.contradiction_generator.get_contradictions_for_prompt(term)

# 3. Добавляем разрешённые знания из кеша
resolved = self.extract_knowledge_from_cache()
`

### Вывод:
**Контекст строится правильно**, но:
1. Только в _handle_gguf_pipeline (не во всех стратегиях)
2. ContradictionGenerator **используется косвенно** через DialogConceptsMixin

---

## 4. Интеграция

### Инициализация в init_factories.py:create_knowledge_graph():

`python
# ConceptExtractor
from eva_ai.knowledge.concept_extractor import create_concept_extractor
concept_extractor = create_concept_extractor(
    fractal_graph=fg,
    brain=initializer.core_brain
)
initializer.core_brain.concept_extractor = concept_extractor

# ContradictionGenerator
from eva_ai.contradiction.contradiction_generator import create_contradiction_generator
contr_generator = create_contradiction_generator(
    brain=initializer.core_brain,
    fractal_graph=fg
)
initializer.core_brain.contradiction_generator = contr_generator
`

### Интеграция в brain_query:

| Компонент | Используется напрямую? | Как используется |
|-----------|------------------------|------------------|
| concept_extractor | Частично | Через _extract_key_concepts() после генерации |
| contradiction_generator | **НЕТ** | Только через self_dialog_learning.get_context_for_generation() |
| ModelAccessManager | **НЕТ** | Создан в model_access_manager.py, но **не используется** |

### ModelAccessManager - ЗАГЛУШКА:
`python
# model_access_manager.py - полноценный класс с приоритетами и очередями
class ModelAccessManager:
    def request_access(self, priority, task_type, callback, ...): ...
    def get_result(self, request_id, timeout): ...
`

**Но в brain_query.py:**
- ModelAccessManager **не импортируется**
- **не создаётся**
- **не используется** для координации доступа к модели

---

## 5. Проблемы

### Критические:

1. **ModelAccessManager не интегрирован**
   - Класс существует, но не используется
   - Нет координации доступа к модели между запросами и самодиалогом
   - Запросы и фоновые задачи конкурируют за модель без приоритетов

2. **Предобработка после генерации**
   - Концепты извлекаются после ответа (строка 267)
   - Нет извлечения сущностей или определения намерений ДО генерации
   - Контекст концептов не влияет на текущий ответ

3. **ContradictionGenerator не используется напрямую**
   - Только косвенно через self_dialog_learning
   - Метод get_contradictions_for_prompt() существует, но не вызывается в brain_query

### Существующие:

4. **Нет entity extraction**
   - Только greeting detection через словари
   - Нет NLP-обработки для извлечения сущностей

5. **Нет intent detection**
   - 
eeds_web_search() - простая эвристика по ключевым словам
   - Нет классификации намерений

6. **Web search результаты не всегда используются**
   - В _handle_gguf_pipeline результаты добавляются в enhanced_query
   - В других стратегиях (_handle_fallback) web search может не вызываться

---

## 6. Оценка

### Согласно документации AGENTS.md:

| Требование из документации | Реализация |
|---------------------------|------------|
| Предобработка: извлечение сущностей | **НЕТ** - только greeting check |
| Предобработка: определение намерений | **ЧАСТИЧНО** - 
eeds_web_search() |
| Построение контекста из concept_extractor | **ДА** - через self_dialog_learning |
| Построение контекста из contradiction_generator | **КОСВЕННО** - через DialogConceptsMixin |
| Построение контекста из cache | **ДА** - в get_context_for_generation() |
| Передача в ModelAccessManager | **НЕТ** - ModelAccessManager не используется |

### Итоговая оценка: 4/7 (57%)

**Реализовано:**
- Контекст из concept_extractor работает
- Кэширование работает
- Web search интеграция работает

**Не реализовано:**
- ModelAccessManager координация
- Предобработка до генерации
- Прямое использование ContradictionGenerator
- Извлечение сущностей и намерений

### Рекомендации:

1. **Интегрировать ModelAccessManager** в brain_query для координации доступа
2. **Добавить предобработку ДО генерации** - извлечение concept_extractor.get_concepts_for_prompt()
3. **Использовать contradiction_generator напрямую** или убедиться что DialogConceptsMixin работает
4. **Добавить entity extraction** на основе concept_extractor перед генерацией
