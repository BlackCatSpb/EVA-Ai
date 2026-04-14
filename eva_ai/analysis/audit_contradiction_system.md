# Отчёт: Система противоречий

## 1. Проверка импортов

### ContradictionGenerator (contradiction_generator.py)
**Импорты:**
`python
import time, logging, random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
`
**Статус:** ✅ Все импорты доступны, стандартные библиотеки

### ContradictionMiner (contradiction_miner.py)
**Импорты:**
`python
import os, time, json, logging, threading
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
`
**Дополнительно:**
- 	ransformers.pipeline для NLI модели (facebook/bart-large-mnli)
- Fallback если модель недоступна

**Статус:** ⚠️ Требует 	ransformers если NLI модель нужна

### __init__.py
**Lazy loading механизм:** ✅ Корректно реализован через __getattr__
**Экспорты:** ✅ Все ключевые классы экспортируются

---

## 2. Соответствие документации

### ContradictionGenerator — ШАБЛОННЫЙ УРОВЕНЬ

| Требование | Реализация | Статус |
|------------|-----------|--------|
| Шаблоны positive/negative | _viewpoint_templates с 4 доменами (general, technology, science, philosophy) | ✅ |
| Генерация противоречивых фактов | generate_contradiction() формирует viewpoint_a и viewpoint_b | ✅ |
| generate_contradiction(concept_name, domain) | Метод существует, параметры match | ✅ |
| save_contradiction(contradiction) | Метод существует, сохраняет в ContradictionManager | ✅ |
| Быстрая генерация | Использует шаблоны + случайный выбор, O(1) | ✅ |
| Интеграция с самодиалогом | ormat_for_dialog() → очередь через brain.self_dialog_learning | ✅ |

### ContradictionMiner — АНАЛИТИЧЕСКИЙ УРОВЕНЬ

| Требование | Реализация | Статус |
|------------|-----------|--------|
| sim(u,v) >= 0.75 | _compute_similarity() — косинусное сходство, порог 0.75 | ✅ |
| contra(u,v) >= 0.65 | _compute_contradiction() — NLI или эвристика, порог 0.65 | ✅ |
| _detect_candidate_pairs() | Метод существует, итерирует пары узлов | ✅ |
| _cluster_pairs() | Метод существует, BFS для связных компонент | ✅ |
| Транзитивное замыкание | Реализовано через BFS по графу конфликтов | ✅ |
| LLM генерация (temperature=0.25) | _generate_formulation() через EVAGenerator | ✅ |
| ContradictionNode в FGv2 | _create_contradiction_node() с рёбрами contradicts | ✅ |
| Кластеризация пар | BFS по связным компонентам | ✅ |

---

## 3. Детальный анализ

### 3.1 ContradictionGenerator — Шаблоны

**Механизм работы:**

1. **Выбор шаблона:** andom.choice(templates) — случайный выбор пары утверждений
2. **Форматирование:** 	emplate_pair[0].format(concept=concept_name) — подстановка концепта
3. **Генерация обоснования:** _generate_reasoning() — случайный шаблон из списка
4. **Divergence:** andom.uniform(0.6, 0.95) — случайное расхождение 0.6-0.95

**Шаблоны по доменам:**
- general: 4 пары (позитив/негатив)
- 	echnology: 4 пары
- science: 3 пары
- philosophy: 3 пары

**Обоснования (reasoning):**
- Positive: "создаёт возможности", "подтверждается опытом", "анализ показывает", "исторически вёл к прогрессу"
- Negative: "создаёт риски", "негативное влияние", "проблемы", "вело к кризисам"

**Вывод:** ✅ Шаблоны работают корректно, но оченьgeneric. Фактически это "болванки" а не настоящие противоречия.

### 3.2 ContradictionMiner — Косинусное сходство

**Реализация (_compute_similarity):**
`python
v1 = np.array(emb1)
v2 = np.array(emb2)
norm1 = np.linalg.norm(v1)
norm2 = np.linalg.norm(v2)
sim = float(np.dot(v1, v2) / (norm1 * norm2))
`
**Это классическое косинусное сходство:** 
sim(u,v) = \frac{u \cdot v}{\|u\| \|v\|}

**Кэширование:** Результаты кэшируются в _similarity_cache

**Вывод:** ✅ Реализация корректна

### 3.3 ContradictionMiner — NLI валидация

**Реализация (_compute_contradiction):**

1. **Основной путь:** BART-large-mnli через 	ransformers.pipeline
   - Формирует последовательность: content1 [SEP] content2
   - Классифицирует как entailment/contradiction/neutral
   - Возвращает score для contradiction

2. **Fallback:** Эвристика (_compute_contradiction_heuristic)
   - Словарь антонимов (27 пар слов)
   - Проверка на наличие антонимов в тексте → score = 0.7
   - Проверка на отрицания (не, нет, без, отсутствует, никогда) → score = 0.65

**Проблема:** NLI модель загружается при первом вызове через singleton. Если 	ransformers не установлен — будет fallback. Но эвристика очень слабая.

**Вывод:** ⚠️ NLI модель используется реально, но есть fallback. Качество эвристики низкое.

### 3.4 Транзитивное замыкание для кластеризации

**Реализация (_cluster_pairs):**

`python
# Строим граф конфликтов
graph = defaultdict(set)
for id1, id2, sim, contra in pairs:
    graph[id1].add(id2)
    graph[id2].add(id1)

# BFS для нахождения связных компонент
visited = set()
clusters = []
for node in graph:
    if node in visited:
        continue
    cluster = []
    queue = [node]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        cluster.append(current)
        for neighbor in graph[current]:
            if neighbor not in visited:
                queue.append(neighbor)
`

**Механизм:**
- Если найдена пара (A,B) и (B,C), обе добавляются в граф
- BFS собирает все связанные узлы в один кластер {A,B,C}
- Это и есть **транзитивное замыкание**

**Метрики кластера:**
`python
avg_sim = np.mean([pair_contra.get(tuple(sorted([cluster[j], cluster[k]])), 0) 
                  for j in range(len(cluster)) for k in range(j+1, len(cluster))])
max_contra = max([...])
`

**Вывод:** ✅ Транзитивное замыкание реализовано корректно через BFS

---

## 4. Проблемы

### 4.1 ContradictionGenerator

| # | Проблема | Серьёзность | Описание |
|---|----------|------------|----------|
| 1 | **Слабые шаблоны** | Высокая | Шаблоны оченьgeneric: "X является положительным явлением" vs "X несёт негативные последствия". Нет реального анализа концепта |
| 2 | **Случайное divergence** | Средняя | andom.uniform(0.6, 0.95) не отражает реальную степень противоречия |
| 3 | **Обоснования — болванки** | Средняя | "Анализ показывает преимущества X" — не содержит реальных аргументов |
| 4 | **Нет проверки дубликатов** | Низкая | uto_generate_for_unknown_concepts() не проверяет, были ли уже сгенерированы похожие противоречия |
| 5 | **Синхронная работа** | Низкая | generate_batch() обрабатывает концепты последовательно, нет параллелизации |

### 4.2 ContradictionMiner

| # | Проблема | Серьёзность | Описание |
|---|----------|------------|----------|
| 1 | **Зависимость от эмбеддингов** | Критическая | Если узлы не имеют embedding — они пропускаются. В текущей реализации FGv2 не все узлы имеют эмбеддинги |
| 2 | **O(n²) сложность** | Высокая | _detect_candidate_pairs() проверяет все пары. При большом графе — медленно |
| 3 | **NLI модель — singleton** | Средняя | _NLI_MODEL глобальный, нет возможности выбора модели или отключения |
| 4 | **Эвристика слишком простая** | Средняя | 27 пар антонимов + проверка на "не" — недостаточно для сложных текстов |
| 5 | **Кэш based на hash(str(emb))** | Низкая | Строковое представление эмбеддинга может быть неуникальным |
| 6 | **LLM вызов без retry** | Средняя | _call_pipeline() не обрабатывает случаи when LLM недоступен |
| 7 | **check_interval_seconds = 3600** | Средняя | Проверка раз в час — может быть слишком редко при активном пополнении графа |

### 4.3 Интеграция

| # | Проблема | Серьёзность | Описание |
|---|----------|------------|----------|
| 1 | **ContradictionGenerator и Miner работают независимо** | Средняя | Нет координации: Generator создаёт шаблонные противоречия, Miner — аналитические. Могут дублироваться |
| 2 | **ContradictionManager может отсутствовать** | Высокая | В save_contradiction() проверяется getattr(brain, 'contradiction_manager', None) — если нет, противоречие не сохраняется |
| 3 | **Очередь в самодиалог** | Низкая | ContradictionMiner добавляет в очередь, но _run_contradiction_dialog() не реализован в текущем DialogConceptsMixin |

---

## 5. Оценка

### ContradictionGenerator

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Соответствие спецификации | 8/10 | Методы match, но качество шаблонов низкое |
| Полнота реализации | 7/10 | Есть основные методы, нет авто-дедупликации |
| Утилитарность | 6/10 | Генерирует "болванки" противоречий, не анализируя концепт |
| **Итоговая** | **7/10** | Работает, но генерирует generic противоречия |

### ContradictionMiner

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Соответствие спецификации | 9/10 | Все ключевые методы реализованы по спецификации |
| Математическая корректность | 9/10 | Косинусное сходство, транзитивное замыкание, NLI — корректны |
| Устойчивость | 6/10 | Fallback на эвристику есть, но качество низкое |
| **Итоговая** | **8/10** | Хорошая реализация аналитического уровня |

### Общая оценка системы

| Компонент | Вес | Оценка |
|-----------|-----|--------|
| ContradictionGenerator | 40% | 7/10 |
| ContradictionMiner | 40% | 8/10 |
| Интеграция | 20% | 6/10 |
| **Итого** | **100%** | **7.2/10** |

---

## Рекомендации

### Высокий приоритет:
1. **Улучшить шаблоны ContradictionGenerator** — добавить доменно-специфичные шаблоны или использовать LLM для генерации точек зрения
2. **Добавить проверку embedding** — в ContradictionMiner узлы без эмбеддингов пропускаются, но это не логируется

### Средний приоритет:
3. **Координация Generator/Miner** — избегать дублирования противоречий
4. **Увеличить словарь антонимов** — для улучшения fallback эвристики
5. **Добавить deduplication** — проверка на похожие уже существующие противоречия

### Низкий приоритет:
6. **Параллелизация batch генерации** — для Generator
7. **Настройка интервалов** — сделать check_interval настраиваемым
