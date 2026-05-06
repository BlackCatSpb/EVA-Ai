# EVAGenerator - Архитектура системы генерации

## Содержание

1. [Обзор решения](#обзор-решения)
2. [Архитектура](#архитектура)
3. [Компоненты системы](#компоненты-системы)
4. [Технические параметры](#технические-параметры)
5. [Reasoning System](#reasoning-system)
6. [API](#api)
7. [Ключевые достижения](#ключевые-достижения)

---

## Обзор решения

**EVAGenerator** — модульная система генерации текста на основе двух физических моделей GGUF, обеспечивающая оптимальный баланс между скоростью и качеством ответов.

### Ключевые характеристики

| Характеристика | Значение |
|---------------|----------|
| Модели | 2 × Qwen 2.5 3B (GGUF, 4-bit) |
| Режимы генерации | 2 (condensed, extended) |
| Автоматическое определение | Да |
| Контекст | FractalGraphV2 |
| Reasoning | 7-шаговая детализация |

---

## Архитектура

### Высокоуровневая схема

```
┌─────────────────────────────────────────────────────────────┐
│                    HybridPipelineAdapter                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  DualGenerator                        │   │
│  │  ┌─────────────────┐     ┌─────────────────────────┐  │   │
│  │  │ CondensedModel  │     │    ExtendedModel        │  │   │
│  │  │ (Model A)       │     │    (Model B)            │  │   │
│  │  │ max_tokens:512  │     │    max_tokens:2048      │  │   │
│  │  │ temp:0.1        │     │    temp:0.4             │  │   │
│  │  └─────────────────┘     └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                              ↓                               │
│                    FractalGraphV2 (контекст)                │
└─────────────────────────────────────────────────────────────┘
```

### Поток обработки

```
1. ЗАПРОС → HybridPipelineAdapter
           │
           ↓
2. АНАЛИЗ → Определение типа запроса
           │
           ↓
3. ВЫБОР  → Condensed или Extended генератор
           │
           ↓
4. ГЕНЕРАЦИЯ → llama_cpp inference
           │
           ↓
5. ОЧИСТКА → Удаление артефактов, повторений
           │
           ↓
6. КОНТЕКСТ → Включение знаний из FGv2
           │
           ↓
7. ОТВЕТ   → reasoning_steps + response
```

---

## Компоненты системы

### 1. HybridPipelineAdapter (`eva_ai/core/hybrid_pipeline_adapter.py`)

Центральный координирующий компонент.

**Функции:**
- Управление жизненным циклом моделей
- Маршрутизация запросов
- Сбор reasoning steps
- Fallback на альтернативные пайплайны

**Режимы работы:**
- `dual` — DualGenerator (основной)
- `fractal` — FractalPipeline
- `recursive` — RecursiveModelPipeline
- `hybrid` — комбинированный

### 2. DualGenerator (`eva_ai/memory/fractal_graph_v2/dual_generator.py`)

Основной генератор с двумя специализированными моделями.

#### CondensedGenerator (Model A)

```
┌────────────────────────────────────┐
│      CondensedGenerator            │
├────────────────────────────────────┤
│ max_tokens:   512                  │
│ temperature:  0.1                  │
│ repeat_penalty: 1.3                │
│                                    │
│ Промт:                            │
│ "Ты — краткий ассистент.          │
│  Дай ответ в 1-2 предложениях."    │
└────────────────────────────────────
```

#### ExtendedGenerator (Model B)

```
┌────────────────────────────────────┐
│       ExtendedGenerator            │
├────────────────────────────────────┤
│ max_tokens:   2048                 │
│ temperature:  0.4                  │
│ repeat_penalty: 1.3                │
│                                    │
│ Промт:                            │
│ "Дай развёрнутый и подробный      │
│  ответ. НЕ повторяй уже написанное.│
│  Контекст: {graph_context}"        │
└────────────────────────────────────
```

### 3. Автоматическое определение типа

```python
auto_keywords = ['кратко', 'вкратце', 'суть', 'кто такой', 'перечисли', 'назови']

if any(kw in query.lower() for kw in auto_keywords):
    → condensed mode
else:
    → extended mode
```

---

## Технические параметры

### Параметры моделей

| Параметр | Condensed | Extended |
|----------|-----------|----------|
| max_tokens | 512 | 2048 |
| temperature | 0.1 | 0.4 |
| repeat_penalty | 1.3 | 1.3 |
| top_p | 0.9 | 0.9 |
| n_ctx | 4096 | 4096 |

### Stop sequences

```python
stop_tokens = [
    '</s>',           # Конец последовательности
    'User:', 'user:', # Начало нового запроса
    'Human:', 'human:',
    'Вопрос:',        # Предотвращение зацикливания
    'Контекст:'
]
```

### Контекст из FractalGraphV2

```python
def get_context(query):
    # Извлечение релевантных узлов
    for node in graph.nodes:
        if any(keyword in node.content):
            return node.content[:200]
    return "Нет контекста"
```

### Статистика генерации

```python
@dataclass
class GeneratorStats:
    total_calls: int      # Всего вызовов
    total_time: float    # Общее время
    avg_time: float      # Среднее время
    total_tokens: int    # Всего токенов
```

---

## Reasoning System

### Структура рассуждений

Каждый запрос сопровождается 7-шаговым reasoning:

```
┌─────────────────────────────────────────────────────────────┐
│ 🔍 query_analysis      - Анализ запроса                    │
│ ⚙️  model_selection    - Выбор режима                      │
│ 📚 context_retrieval   - Извлечение контекста               │
│ 🤖 generation          - Генерация ответа                   │
│ 📖/📝 extended/condensed - Результат с параметрами          │
│ ✅ quality_check       - Проверка качества                  │
│ ✨ final_synthesis     - Финальный синтез                   │
└─────────────────────────────────────────────────────────────┘
```

### Формат данных

```python
{
    'step': 1,                    # Номер шага
    'phase': 'query_analysis',    # Фаза
    'thought': 'Анализ запроса...',  # Описание
    'confidence': 1.0,            # Уверенность (0-1)
    'icon': '🔍'                  # Иконка
}
```

### Иконки по фазам

| Фаза | Иконка |
|------|--------|
| query_analysis | 🔍 |
| model_selection | ⚙️ |
| context_retrieval | 📚 |
| generation | 🤖 |
| condensed | 📝 |
| extended | 📖 |
| quality_check | ✅ |
| final_synthesis | ✨ |
| ethics_check | 🛡️ |
| web_search | 🌐 |

---

## API

### Использование DualGenerator

```python
from eva_ai.memory.fractal_graph_v2.dual_generator import DualGenerator

# Инициализация
dual = DualGenerator(
    llama_condensed=llama_model_a,   # Model A
    llama_extended=llama_model_b,    # Model B
    graph=fractal_graph
)

# Автоматический режим (рекомендуется)
response = dual.generate("Что такое Python?", mode="auto")

# Принудительно краткий
result = dual.generate_condensed("Что такое?")

# Принудительно развёрнутый
result = dual.generate_extended("Расскажи подробнее")

# С деталями
details = dual.generate("Вопрос?", mode="auto", return_details=True)
# {response, mode, time, length, tokens_estimate}
```

### Использование HybridPipelineAdapter

```python
from eva_ai.core.hybrid_pipeline_adapter import HybridPipelineAdapter

# Полная инициализация
adapter = HybridPipelineAdapter(
    fractal_graph=graph,
    mode='dual',
    model_a_path='path/to/model_a.gguf',
    model_b_path='path/to/model_b.gguf',
    n_ctx=4096,
    n_threads=8,
    load_models=True
)

# Обработка запроса
result = adapter.process_query(
    query="Ваш вопрос",
    generation_mode="auto"  # auto | condensed | extended
)

# Результат содержит:
# - response: текст ответа
# - reasoning_steps: массив шагов
# - quality: оценка качества
# - processing_time: время выполнения
```

### Ответ API

```python
{
    'response': str,                    # Основной ответ
    'final_response': str,              # Финальный ответ
    'natural_response': str,            # Натуральный ответ
    'confidence': 0.9,                  # Уверенность
    'quality': {
        'score': 0.9,                   # Оценка качества
        'is_gibberish': False,
        'reasons': ['OK']
    },
    'reasoning_steps': [                # 7 шагов
        {
            'step': 1,
            'phase': 'query_analysis',
            'thought': '...',
            'confidence': 1.0,
            'icon': '🔍'
        },
        ...
    ],
    'processing_time': 15.3,            # Секунды
    'source': 'gguf_pipeline'
}
```

---

## Ключевые достижения

### 1. Скорость

| Тип ответа | Время |
|-------------|-------|
| Краткий (condensed) | 5-10 сек |
| Развёрнутый (extended) | 15-25 сек |

### 2. Качество

- **Стабильный score**: 0.90
- **Без повторений**: repeat_penalty=1.3
- **Без мусора**: vowel-based проверка

### 3. Интеллектуальность

- Автоматическое определение типа запроса
- Контекст из FractalGraphV2 (507 узлов)
- Детальные reasoning steps в UI

### 4. Модульность

- 4 режима работы адаптера
- Fallback на альтернативные пайплайны
- Статистика и мониторинг

### 5. Интеграция

- Web GUI с визуализацией reasoning
- Real-time обновление статуса
- Цветовая индикация уверенности

### 6. Конфигурация

```json
{
  "model": {
    "pipeline_mode": "dual",
    "llama_cpp_n_ctx": 4096,
    "llama_cpp_threads": 8
  }
}
```

---

## Файловая структура

```
eva_ai/core/
├── hybrid_pipeline_adapter.py    # Координатор
└── fractal_pipeline.py           # Альтернативный пайплайн

eva_ai/memory/fractal_graph_v2/
├── dual_generator.py             # ★ Основной генератор
├── eva_generator.py             # Совместимость
├── prompt_templates.py           # Промты
└── storage.py                    # FractalGraphV2

eva_ai/gui/web_gui/
├── server_main.py                # Reasoning на сервере
└── static/
    ├── js/app.js                 # Отображение в UI
    └── css/style.css             # Стили
```

---

## Запуск

```bash
# Консольный режим
python -m eva_ai

# Web GUI
python start_webgui.py
# http://127.0.0.1:5555
```

---

*Документация: EVAGenerator Architecture*
*Версия: 1.0*