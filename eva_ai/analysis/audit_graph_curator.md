# Отчёт: GraphCurator

## 1. Импорты

```python
import logging
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from collections import defaultdict
```

**Проблема**: Отсутствует импорт EventBus. Нет импорта DeferredCommandSystem.

---

## 2. Методы оптимизации

### 2.1 cleanup (`_cleanup_garbage`)
**Цель**: Удаление мусора

**Условия удаления узла**:
- Не защищённый тип (not in PROTECTED_TYPES)
- Нет связей (orphan)
- effective_confidence < 0.15
- Создан более 7 дней назад

**Лимит**: Максимум 50 узлов за цикл

**Защищённые типы**:
```python
PROTECTED_TYPES = {
    'concept', 'contradiction', 'model_a', 'model_b', 'model_c', 
    'model_root', 'semantic_group', 'domain_profile'
}
```

---

### 2.2 consolidate (`_consolidate_nodes`)
**Цель**: Создание семантических групп

**Логика**:
1. Кластеризация узлов по уровням 1 и 2 (threshold=0.6, method="simple")
2. Для кластеров с 3+ узлами:
   - Создание новой группы через `storage.create_semantic_group()`
   - Или обновление существующей через `_update_group()`
3. Объединение пересекающихся групп (>70% пересечение成员)

**Проблема**: Требует `storage.cluster_nodes()` который может отсутствовать.

---

### 2.3 promote/demote (`_process_level_promotions`)
**Цель**: Перемещение узлов между фрактальными уровнями

**Промоут** (повышение):
- confidence > 0.8 AND access_count > 5
- level < 3 → level + 1

**Демоут** (понижение):
- confidence < 0.2 AND access_count < 2
- level > 0 → level - 1

---

## 3. EventBus интеграция

### 3.1 Документация (system_design_v2.md)
```markdown
- Event bus integration: Subscribes to system events
- Adaptive intervals: 60-600 seconds based on activity
- Deferred commands: Uses priority queue for background tasks
```

### 3.2 Реальность
**GraphCurator НЕ использует EventBus!**

- Нет подписки на события
- Работает автономно в `threading.Thread`
- Фиксированный интервал `check_interval = 600` секунд
- `DeferredCommandSystem` передаётся в `brain_init.py` но НЕ ИСПОЛЬЗУЕТСЯ в коде

```python
# brain_init.py - передаётся но не используется:
brain.graph_curator._deferred_system = brain.deferred_system

# graph_curator.py - не применяется:
# self._deferred_system нигде не используется в коде
```

---

## 4. Проблемы

### Критические

1. **EventBus не интегрирован**
   - Нет подписки на `system.idle`, `memory.graph_updated` и т.д.
   - Не реагирует на события графа

2. **DeferredCommandSystem не используется**
   - Передаётся в инициализации
   - Не применяется для приоритизации задач

3. **Адаптивные интервалы не реализованы**
   - Документация: 60-600 секунд based on activity
   - Реальность: Фиксированные 600 секунд

### Существенные

4. **Кластеризация может не работать**
   - `storage.cluster_nodes()` может отсутствовать
   - Нет fallback логики

5. **Метрики неполные**
   - Нет `nodes_curated`, `links_created/removed`
   - Нет `orphan_nodes`, `avg_cycle_time`

---

## 5. Оценка

| Критерий | Документация | Реальность | Статус |
|----------|--------------|------------|--------|
| EventBus подписка | Да | Нет | FAIL |
| Deferred commands | Да | Нет | FAIL |
| Adaptive intervals | 60-600s | 600s fixed | PARTIAL |
| Cleanup | Да | Да | OK |
| Promote/Demote | Да | Да | OK |
| Consolidate | Да | Частично | PARTIAL |
| Protected types | concept, contradiction, model_* | Да | OK |

### Итог: 3/7 (42%)

**Основные недостатки**:
- Полностью отсутствует EventBus интеграция
- Не используется DeferredCommandSystem
- Нет адаптивных интервалов

**Рекомендации**:
1. Добавить EventBus подписку в `__init__`
2. Интегрировать DeferredCommandSystem для фоновых задач
3. Реализовать адаптивные интервалы
