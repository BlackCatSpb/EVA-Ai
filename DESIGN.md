# CogniFlex Архитектура: Фрактальное Хранилище + Self-Reasoning

## Дата: 2026-03-23
Версия: 1.0

---

## 1. Цель Системы

Создать когнитивную систему с:
- **Фрактальным хранилищем** - иерархическая ML-структура с рекурсивной адресацией
- **Self-Reasoning Engine** - цикл рассуждения с контекстными вопросами
- **Единой моделью Qwen** - только один экземпляр, никаких fallback
- **Полной аналитикой** - ethics, contradiction, knowledge gaps в цикле рассуждения

---

## 2. Фрактальное Хранилище

### 2.1 Структура Директорий

```
cogniflex/reasoning/
├── __init__.py
├── fractal_ml/
│   ├── __init__.py
│   ├── fractal_base.py       # Базовые классы (FractalNode, FractalEdge)
│   ├── fractal_tokenizer.py  # Собственная токенизация
│   ├── fractal_embedder.py  # Эмбеддинги для адресации
│   ├── fractal_address.py   # Рекурсивная адресация L0→L1→L2→L3
│   ├── fractal_storage.py   # Главное хранилище
│   ├── fractal_retriever.py # Извлечение с нужной глубиной
│   └── fractal_index.py     # Индексация и поиск
├── self_reasoning_engine.py # Главный движок рассуждения
├── confidence_scorer.py     # Оценка уверенности
├── clarification_generator.py # Генерация контекстных вопросов
├── reasoning_types.py       # Типы узлов для KG
└── integration.py           # Интеграция с CoreBrain
```

### 2.2 Параметры Фрактальной Структуры

| Параметр | Значение | Описание |
|----------|----------|----------|
| MAX_LEVELS | 4 | Глубина иерархии |
| BRANCHING_FACTOR | 16 | Ветвей на уровень |
| BASE_SIZE_KB | 1 | Размер L0 в KB |
| EMBEDDING_DIM | 384 | Размер эмбеддингов |
| MAX_NODES_PER_LEVEL | 16384 | Макс. узлов на уровень |

### 2.3 Формула Размеров

```
L0 = 1 KB
L1 = 16 KB   (L0 × 16)
L2 = 256 KB (L1 × 16)
L3 = 4 MB   (L2 × 16)
```

### 2.4 Рекурсивная Адресация

```
Запрос: "собака"
  ↓ L0: "Собака" (общность)
     ↓ L1: "Порода: Хаски"
        ↓ L2: "Особенности: цвет, характер, размер"
           ↓ L3: "Конкретное описание: кличка, возраст, хозяин"
```

### 2.5 Типы Узлов Хранения

```python
class FractalNodeType(Enum):
    ROOT = "root"           # L0 - максимально общий
    CATEGORY = "category"   # L1 - категория
    ENTITY = "entity"       # L2 - конкретный объект
    DETAIL = "detail"       # L3 - детальное описание
    REASONING_STEP = "reasoning_step"  # Шаг рассуждения
    CLARIFICATION = "clarification"    # Вопрос уточнения
```

---

## 3. Self-Reasoning Engine

### 3.1 Цикл Работы

```
┌─────────────────────────────────────────────────────────────┐
│                    SELF-REASONING LOOP                      │
├─────────────────────────────────────────────────────────────┤
│  1. INITIAL_QUERY: "По дороге ехала машина"                │
│                           ↓                                  │
│  2. QWEN_GENERATE: Первичный ответ от Qwen (singleton)     │
│                           ↓                                  │
│  3. ANALYZE:                                                 │
│     ├─ EthicsFramework.analyze_response()                   │
│     ├─ ContradictionDetector.detect_contradictions()       │
│     └─ KnowledgeAnalyzer.analyze_knowledge_gaps()          │
│                           ↓                                  │
│  4. CONFIDENCE_SCORE: Расчёт уверенности (0.0-1.0)          │
│                           ↓                                  │
│  5. IF confidence >= 0.75:                                  │
│        → FINAL_RESPONSE: Выдать в GUI                       │
│     ELSE:                                                    │
│        → CLARIFICATION_QUESTIONS: Генерация вопросов       │
│        → STORE_IN_FRAKTAL: Сохранить цепочку                │
│        → LOOP: Вернуться к шагу 2                           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| MAX_ITERATIONS | 5 | Макс. циклов уточнения |
| CONFIDENCE_THRESHOLD | 0.75 | Минимальная уверенность |
| ETHICS_WEIGHT | 0.30 | Вес в формуле уверенности |
| CONTRADICTION_WEIGHT | 0.30 | Вес в формуле уверенности |
| KNOWLEDGE_WEIGHT | 0.40 | Вес в формуле уверенности |

### 3.3 Формула Уверенности

```
Confidence = (ethics_score × 0.30) + 
              (contradiction_score × 0.30) + 
              (knowledge_score × 0.40)
```

### 3.4 Пример Генерации Вопросов

**Запрос:** "По дороге ехала машина"

**Генерируемые вопросы (контекстные, не рандомные):**
- "По какой дороге ехала машина?"
- "Какая именно машина?"
- "Куда она ехала?"
- "Что такое дорога в данном контексте?"
- "Что такое машина и какие её характеристики?"

---

## 4. Интеграция с CoreBrain

### 4.1 Изменения brain_config.json

```json
{
  "reasoning": {
    "enabled": true,
    "max_iterations": 5,
    "confidence_threshold": 0.75,
    "store_reasoning_chains": true,
    "fractal_storage_path": "cogniflex/core/cogniflex_cache/reasoning/fractal/"
  },
  "model": {
    "name": "qwen3.5-0.8b",
    "type": "qwen",
    "qwen_only_mode": true,
    "disable_fallback": true
  }
}
```

### 4.2 Интеграционные точки

1. **CoreBrain.process_query()** → вызов SelfReasoningEngine
2. **Qwen singleton** → передаётся в SelfReasoningEngine
3. **Knowledge Graph** → расширение типами REASONING
4. **Fractal Storage** → создаётся при инициализации brain

---

## 5. Ожидаемые Результаты

| # | Результат | Описание |
|---|-----------|----------|
| 1 | Фрактальное хранилище работает | Иерархическая структура с адресацией |
| 2 | Self-Reasoning Engine работает | Цикл "вопрос-анализ-уточнение" |
| 3 | Контекстные вопросы | Связаны с запросом, не рандомные |
| 4 | Хранение рассуждений | Вся цепочка в fractal storage |
| 5 | Никаких fallback | Только Qwen + аналитика |
| 6 | Confidence-based termination | Остановка при уверенности ≥0.75 |

---

## 6. Критерии Успеха

- ✅ Система задаёт вопросы при низкой уверенности
- ✅ Вопросы контекстно связаны с запросом
- ✅ Единый экземпляр Qwen (singleton)
- ✅ Все рассуждения сохраняются
- ✅ Нет случайных/бессмысленных ответов
- ✅ Фрактальная структура масштабируема

---

## 7. История Изменений

| Версия | Дата | Описание |
|--------|------|----------|
| 1.0 | 2026-03-23 | Начальная версия плана |
| 1.1 | 2026-03-23 | Реализация: fractal_base.py, confidence_scorer.py, clarification_generator.py, reasoning_types.py, self_reasoning_engine.py, integration.py |

## 8. Текущий Статус Реализации

### Созданные файлы:

**cogniflex/reasoning/fractal_ml/:**
- [x] fractal_base.py - FractalNode, FractalEdge, FractalAddress, FractalIndex
- [x] fractal_tokenizer.py - FractalTokenizer

**cogniflex/reasoning/:**
- [x] confidence_scorer.py - Расчёт уверенности (формула реализована)
- [x] clarification_generator.py - Генерация контекстных вопросов
- [x] reasoning_types.py - ReasoningStep, ReasoningResult
- [x] self_reasoning_engine.py - SelfReasoningEngine класс
- [x] integration.py - Интеграция с CoreBrain

### Ожидают создания (от агентов):
- fractal_storage.py
- fractal_retriever.py  
- fractal_embedder.py
- fractal_address.py
- __init__.py файлы

### Интеграция с CoreBrain:
- Требуется добавить в brain_config.json секцию "reasoning"
- Требуется вызов ReasoningIntegration в ComponentInitializer
