# Анализ Legacy Contradiction Detection EVA

## Detect System (дополнительные модули)

### detect_core.py

**Базовый класс детекции:**
- Abstract Detector
- Методы: detect(), validate(), refine()
- Интерфейс для всех детекторов

**Статус: ОСНОВА**

---

### detect_semantic.py

**Семантическая детекция:**
- Semantic similarity analysis
- Embedding-based сравнение
- Threshold-based detection

**Методы:**
- `detect_semantic_contradiction(text1, text2)`
- `_compute_embedding_similarity()`

**Статус: ИСПОЛЬЗУЕТСЯ**

---

### detect_logical.py

**Логическая детекция:**
- Logical structure analysis
- Проверка consistency
- Rule-based validation

**Методы:**
- `detect_logical_contradiction(statement1, statement2)`
- `_analyze_logical_structure()`

**Статус: ИСПОЛЬЗУЕТСЯ**

---

### detect_temporal.py

**Временная детекция:**
- Temporal consistency
- Time-based противоречия
- Timeline analysis

**Методы:**
- `detect_temporal_contradiction(event1, event2)`
- `_parse_temporal_relations()`

**Статус: ИСПОЛЬЗУЕТСЯ**

---

## Learn System (обучение противоречий)

### learn_core.py

**Ядро обучения:**
- Обучение на примерах противоречий
- Pattern extraction
- Feedback integration

---

### learn_patterns.py

**Изучение паттернов:**
- Паттерны противоречий
- Category classification
- Template generation

---

### learn_feedback.py

**Feedback обучение:**
- User feedback integration
- Quality assessment
- Continuous improvement

---

## Tracking & Resolution (отслеживание и разрешение)

### core_tracking.py

**Отслеживание:**
- Track contradictions over time
- History management
- Status updates

---

### core_resolution.py

**Разрешение:**
- Resolution strategies
- Synthesis generation
- Solution validation

---

## Выводы

| Модуль | Статус |
|--------|--------|
| detect_core | ✅ Основа |
| detect_semantic | ✅ Используется |
| detect_logical | ✅ Используется |
| detect_temporal | ✅ Используется |
| learn_* | ✅ Используется |
| core_tracking | ✅ Используется |
| core_resolution | ✅ Используется |

Система детекции противоречий - развитая подсистема с множественными методами обнаружения.