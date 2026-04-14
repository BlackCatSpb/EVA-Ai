# Отчёт: Система концептов

## 1. Проверка импортов

### ConceptExtractor (concept_extractor.py)
**Статус: ВАЛИДНЫ**

Импорты:
```python
import re
import time
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
```

- Все импорты стандартные библиотеки
- Дополнительных внешних зависимостей нет
- Типизация используется корректно

### ConceptMiner (concept_miner.py)
**Статус: ВАЛИДНЫ**

Импорты:
```python
import os, time, json, logging, threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
```

- Все импорты стандартные библиотеки или уже присутствуют в проекте
- numpy используется для вычислений (косинусное сходство, нормализация)
- ThreadPoolExecutor для асинхронного выполнения

### Knowledge __init__.py
**Статус: ВАЛИДНЫ**

Экспорты согласованы с исходными файлами.

---

## 2. Соответствие методов документации

### ConceptExtractor

| Метод | В документации | В коде | Статус |
|-------|---------------|--------|--------|
| `extract_concepts(query, response)` | Да | Да (строка 55) | СООТВЕТСТВУЕТ |
| `save_concept_to_graph(concept)` | Да | Да (строка 273) | СООТВЕТСТВУЕТ |
| `format_concept_for_dialog(concept)` | Да | Да (строка 451) | СООТВЕТСТВУЕТ |
| `get_concepts_for_prompt(query)` | Да | Да (строка 353) | СООТВЕТСТВУЕТ |

### ConceptMiner

| Метод | В документации | В коде | Статус |
|-------|---------------|--------|--------|
| `_detect_semantic_gaps(clusters)` | Да | Да (строка 509) | СООТВЕТСТВУЕТ |
| `_validate_candidate(candidate)` | Да | Да (строка 758) | СООТВЕТСТВУЕТ |
| `_generate_hypothesis(candidate)` | Да | Да (строка 624) | СООТВЕТСТВУЕТ |
| `_integrate_candidate(candidate)` | Да | Да (строка 911) | СООТВЕТСТВУЕТ |
| `_get_clusters()` | Да | Да (строка 442) | СООТВЕТСТВУЕТ |

### Жизненный цикл

| Статус | В документации | В коде | Статус |
|--------|---------------|--------|--------|
| PROVISIONAL | Да | ConceptStatus.PROVISIONAL (строка 30) | СООТВЕТСТВУЕТ |
| CONFIRMED | Да | ConceptStatus.CONFIRMED (строка 31) | СООТВЕТСТВУЕТ |
| STABLE | Да | ConceptStatus.STABLE (строка 32) | СООТВЕТСТВУЕТ |
| ARCHIVED | Да | ConceptStatus.ARCHIVED (строка 33) | СООТВЕТСТВУЕТ |

---

## 3. Детальный анализ

### 3.1 ConceptExtractor - Частотный анализ

**Реализация (строки 87-108):**

```python
def _extract_terms(self, text: str) -> List[str]:
    # Находим слова (русские и английские)
    words = re.findall(r'\b[а-яёa-z]{4,}\b', text.lower())
    
    # Подсчитываем частоту
    freq = {}
    for word in words:
        if word not in self._stop_words:
            freq[word] = freq.get(word, 0) + 1
    
    # Сортируем по частоте и возвращаем топ-15
    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in sorted_terms[:15]]
```

**Анализ:**
- Используется регулярное выражение для извлечения слов 4+ символов
- Частотный подсчёт реализован корректно
- Стоп-слова фильтруются
- Ограничение: максимум 5 новых концептов за раз (строка 78)

**Проблема:** Метод `extract_concepts` вызывается без сохранения концептов в граф! В строке 85 возвращается список концептов, но не вызывается `save_concept_to_graph`. Это **НЕСООТВЕТСТВИЕ** документации, где сказано что концепты сохраняются синхронно.

### 3.2 ConceptMiner - Семантические лакуны

**Формула из документации:**
- ΔC = min(1 - cos(μC, v)) - семантический разрыв
- μC = (1/|C|)Σv - центроид кластера
- σ²C = (1/|C|)Σ cos_dist(μC, v) - дисперсия
- τ = τ_base * (1 + variance_k * σC) - адаптивный порог

**Реализация (строки 509-580):**

```python
def _detect_semantic_gaps(self, clusters: Dict) -> List[PhantomCandidate]:
    # ...
    for cluster_id, node_ids in clusters.items():
        # ...
        centroid = np.mean(embeddings, axis=0)  # μC = (1/|C|)Σv
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-8:
            centroid = centroid / centroid_norm

        gaps = []
        for emb in embeddings:
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 1e-8:
                cos_sim = np.dot(centroid, emb) / emb_norm
            else:
                cos_sim = 0.0
            gap = 1.0 - cos_sim  # ΔC = min(1 - cos(μC, v))
            gaps.append(gap)

        variance = float(np.std(gaps))  # σ²C
        semantic_gap = float(min(gaps))  # min(ΔC)

        threshold = base_threshold * (1 + variance_k * variance)  # τ = τ_base * (1 + variance_k * σC)
```

**Анализ:**
- Математическая формула реализована **КОРРЕКТНО**
- Центроид вычисляется как среднее векторов
- Косинусное сходство вычисляется правильно
- Адаптивный порог соответствует документации

### 3.3 Валидация ConceptMiner

**NLI (строки 811-839):**
- Проверяет логическую согласованность концепта с контекстом
- Использует `_call_pipeline()` для вызова LLM
- Возвращает статус: entailment/neutral/contradiction
- При entailment увеличивает confidence на 0.25

**Ontology (строки 866-876):**
```python
def _check_ontology_compliance(self, candidate: PhantomCandidate) -> Dict:
    min_connections = 3
    if len(candidate.nodes) < min_connections:
        return {"compliant": False, "reason": "..."}
    return {"compliant": True}
```
- **УПРОЩЕНО**: Только проверка количества связей
- Нет реальной онтологической валидации

**Ethics (строки 841-864):**
- Проверка на вредоносный, дискриминационный контент
- Использует `_call_pipeline()` с специальным промптом
- Возвращает risk_level: low/high

**Web (строки 878-909):**
```python
def _verify_web(self, candidate: PhantomCandidate) -> Dict:
    # Генерирует поисковый запрос
    # Вызывает brain.web_search.search()
    # Возвращает verified, results_count, source_quality
```
- Опционально (controlled by `enable_web_search_validation`)
- По умолчанию **ВЫКЛЮЧЕНА** в init_factories.py (строка 582: `'enable_web_search_validation': False`)

### 3.4 EventBus подписка

**ConceptMiner подписывается на (строки 260-266):**
```python
events_to_subscribe = [
    ("memory.graph_updated", "_on_memory_graph_updated"),
    ("memory.clustering_complete", "_on_memory_clustering_complete"),
    ("pipeline.complete", "_on_pipeline_complete"),
    ("system.ready", "_on_system_ready"),
    ("system.idle", "_on_system_idle"),
]
```

**Анализ:**
- Подписка реализована корректно через `event_bus.subscribe()`
- Обработчики существуют и вызываются
- При idle вызывается `_schedule_mining_if_needed()`

### 3.5 DeferredCommandSystem интеграция

**Реализация (строки 323-347):**
```python
if self.deferred_system and self.config.get("priority_queue") != "DISABLED":
    from eva_ai.core.deferred_command_system import CommandPriority
    
    priority_map = {
        "CRITICAL": CommandPriority.CRITICAL,
        "HIGH": CommandPriority.HIGH,
        "NORMAL": CommandPriority.NORMAL,
        "LOW": CommandPriority.LOW
    }
    priority = priority_map.get(self.config.get("priority_queue", "NORMAL"), CommandPriority.NORMAL)
    
    self.deferred_system.add_command(
        command=self._mining_cycle,
        priority=priority,
        max_retries=2,
        retry_delay=5.0,
        command_id=f"concept_mining_{int(time.time())}"
    )
```

**Анализ:**
- Интеграция с DeferredCommandSystem **КОРРЕКТНА**
- Fallback на ThreadPoolExecutor если DeferredCommandSystem недоступен (строка 347)

---

## 4. Проблемы и несоответствия

### КРИТИЧЕСКИЕ

1. **ConceptExtractor не сохраняет концепты в граф автоматически**
   - Метод `extract_concepts()` возвращает список концептов
   - Но **НЕ вызывает** `save_concept_to_graph()`
   - В документации сказано: "Немедленно добавляет в очередь самодиалога" и "Создаёт узлы типа 'concept' в FGv2"
   - В реальности концепты только возвращаются, но не сохраняются
   - **Это 需要 исправление**

2. **Ontology валидация упрощена**
   - Документация: "Онтологическая совместимость" через формальную проверку
   - Реализация: только проверка `len(candidate.nodes) < 3`
   - Нет реальной онтологической проверки

3. **Web-валидация выключена по умолчанию**
   - В `init_factories.py` строка 582: `'enable_web_search_validation': False`
   - Документация: "Веб-верификация (опционально)"
   - Это допустимо, но нужно понимать что валидация неполная

### СУЩЕСТВЕННЫЕ

4. **Несоответствие имени события**
   - В EventBus определён тип: `CONCEPT_MINING_START = "concept.mining.start"` (строка 115)
   - ConceptMiner публикует: `"concept.mining.start"` (строка 628)
   - Совпадает, но есть и другие:
     - `CONCEPT_MINING_COMPLETE = "concept.mining.complete"` (строка 116)
     - `CONCEPT_CANDIDATE_GENERATED = "concept.candidate.generated"` (строка 118)
     - `CONCEPT_VALIDATION_COMPLETE = "concept.validation.complete"` (строка 119)
   - Всё используется правильно

5. **ConceptExtractor не интегрирован в EventBus**
   - Нет подписки на события
   - Работает только синхронно при вызове
   - В документации не указано что он подписан, но логично было бы ожидать события при извлечении

### МЕХАНИЧЕСКИЕ

6. **Ограничение на 5 новых концептов за раз (строка 78)**
   - `for term in new_terms[:5]`
   - Документация не оговаривает лимит
   - Может привести к потере концептов при большом потоке

7. **Отсутствие обработки `system.idle` для ConceptExtractor**
   - Только ConceptMiner реагирует на idle
   - ConceptExtractor работает только по запросу

---

## 5. Оценка реализации

### ConceptExtractor: 7/10

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Частотный анализ | 9/10 | Корректная реализация |
| Сохранение в граф | 3/10 | **НЕ работает** - нужно вызывать save_concept_to_graph |
| Факты is_a, has_property, can, related_to | 8/10 | Реализовано через паттерны |
| Интеграция с самодиалогом | 5/10 | Возвращает концепты, но не добавляет в очередь |

### ConceptMiner: 8/10

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Математика семантических лакун | 10/10 | Формулы реализованы точно |
| EventBus интеграция | 9/10 | Корректная подписка и обработка |
| DeferredCommandSystem | 9/10 | Интеграция корректна |
| NLI валидация | 8/10 | Работает через LLM |
| Ontology валидация | 4/10 | **Упрощена** до подсчёта связей |
| Ethics валидация | 7/10 | Базовый скрининг |
| Web валидация | 6/10 | Опциональна, выключена по умолчанию |
| Жизненный цикл | 9/10 | PROVISIONAL→CONFIRMED→STABLE→ARCHIVED |

### Общая оценка: 7.5/10

---

## 6. Рекомендации

### Немедленные исправления:

1. **ConceptExtractor.extract_concepts()** - добавить вызов `save_concept_to_graph()` для каждого концепта
2. **Ontology валидация** - расширить до реальной проверки онтологических правил или задокументировать упрощение

### Улучшения:

3. Добавить события EventBus при извлечении концептов в ConceptExtractor
4. Включить web-валидацию когда ready
5. Рассмотреть возможность фоновой работы ConceptExtractor при idle

### Документация:

6. Зафиксировать что Ontology валидация упрощена
7. Уточнить лимит в 5 концептов за раз
