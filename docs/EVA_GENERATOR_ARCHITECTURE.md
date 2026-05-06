# EVAGenerator и гибридная архитектура генерации

## Содержание

1. [Обзор](#обзор)
2. [Принцип работы](#принцип-работы)
3. [Архитектура](#архитектура)
4. [Компоненты](#компоненты)
5. [Технические параметры](#технические-параметры)
6. [Reasoning System](#reasoning-system)
7. [Web GUI интеграция](#web-gui-интеграция)
8. [API](#api)
9. [Конфигурация](#конфигурация)
10. [Оптимизации](#оптимизации)
11. [Troubleshooting](#troubleshooting)

---

## Обзор

### Цель

Создание единой архитектуры EVA AI на основе FractalGraphV2 (FGv2) с гибридной интеграцией GGUF модели Qwen 2.5 3B, заменяющей устаревший KnowledgeGraph (KG) и медленный RecursiveModelPipeline.

### Ключевые особенности

- **2 физических модели** - отдельные инстансы для разных типов ответов
- **Автоматическое определение** типа запроса
- **Детальные рассуждения** - визуализация процесса генерации в Web GUI
- **Контекст из графа** - извлечение знаний из FractalGraphV2
- **Контроль качества** - проверка и очистка ответов

### Проблемы старой архитектуры

| Проблема | Описание |
|----------|---------|
| **RecursiveModelPipeline** | ~50-180 секунд на запрос (множественные вызовы Model A + Model B + ethics + quality checks) |
| **KnowledgeGraph** | Удалён из системы |
| **Качество** | Нестабильная оценка качества (0.20-0.80) |
| **Длина ответов** | Неконтролируемая |

### Результат оптимизации

| Метрика | Было | Стало | Улучшение |
|---------|------|-------|-----------|
| Время краткого ответа | ~58s | ~5-10s | **в 6 раз** |
| Время развёрнутого ответа | ~180s | ~15-25s | **в 7 раз** |
| Total time (4 запроса) | ~190s | ~40s | **в 4.5 раз** |
| Quality score | 0.20-0.80 | 0.90 | Стабильно |
| Ответов без повторений | Нет | Да | **100%** |

---

## Принцип работы

### Полный цикл обработки запроса

```
Пользователь → Web GUI → CoreBrain → HybridPipelineAdapter → DualGenerator
                                                                  │
                                                                  ↓
                                                    ┌─────────────┴─────────────┐
                                                    ↓                           ↓
                                            CondensedGenerator        ExtendedGenerator
                                            (Model A, краткий)        (Model B, развёрнутый)
                                                    │                           │
                                                    └───────────┬───────────────┘
                                                                ↓
                                                          ┌─────┴─────┐
                                                          ↓           ↓
                                                   Очистка    Проверка качества
                                                          │           │
                                                          └───────────┴───────────┐
                                                                                  ↓
                                                          Web GUI ← Reasoning Steps
```

### Автоматическое определение режима

Система автоматически выбирает тип генерации на основе ключевых слов:

| Режим | Ключевые слова | max_tokens | temperature |
|-------|---------------|------------|-------------|
| **condensed** | кратко, вкратце, суть, кто такой, перечисли, назови | 512 | 0.1 |
| **extended** | все остальные запросы | 2048 | 0.4 |

### Reasoning Steps (Рассуждения)

При каждом запросе система фиксирует этапы обработки:

```
1. 🔍 query_analysis      - Анализ запроса пользователя
2. ⚙️ model_selection    - Выбор режима генерации
3. 📚 context_retrieval  - Извлечение контекста из FractalGraphV2
4. 🤖 generation          - Генерация ответа через выбранную модель
5. 📝/📖 condensed/extended - Результат с параметрами
6. ✅ quality_check       - Проверка качества
7. ✨ final_synthesis    - Формирование финального ответа
```

---

## Архитектура

### Иерархия компонентов

```
CoreBrain
    │
    ├── HybridPipelineAdapter (mode='dual')
    │       │
    │       ├── DualGenerator
    │       │       ├── CondensedGenerator (Model A)
    │       │       │       └── llama_cpp: qwen2.5-3b-instruct-q4_k_m.gguf
    │       │       │
    │       │       └── ExtendedGenerator (Model B)
    │       │               └── llama_cpp: qwen2.5-3b-instruct-q4_k_m_model_b.gguf
    │       │
    │       └── FractalGraphV2 (контекст)
    │               └── 507 nodes (опыт и знания)
    │
    └── FractalGraphV2
            └── Семантический поиск
```

### Режимы работы HybridPipelineAdapter

| Режим | Описание | Когда использовать |
|-------|---------|-------------------|
| `dual` | DualGenerator с 2 физическими моделями | **Рекомендуется** - оптимальная скорость |
| `fractal` | FractalPipeline | Тестирование, сравнение |
| `recursive` | RecursiveModelPipeline (старый) | Fallback при проблемах |
| `hybrid` | FractalPipeline + fallback | Постепенный переход |

---

## Компоненты

### 1. HybridPipelineAdapter (`eva_ai/core/hybrid_pipeline_adapter.py`)

Центральный адаптер для управления генерацией.

#### Инициализация

```python
from eva_ai.core.hybrid_pipeline_adapter import HybridPipelineAdapter

adapter = HybridPipelineAdapter(
    fractal_graph=graph,           # FractalGraphV2 для контекста
    mode='dual',                  # Режим работы
    model_a_path=MODEL_A_PATH,    # Путь к Model A
    model_b_path=MODEL_B_PATH,    # Путь к Model B
    n_ctx=4096,                   # Размер контекста
    n_threads=8,                 # Потоки CPU
    load_models=True              # Загрузить модели сразу
)
```

#### Загрузка моделей

```python
# Автоматическая загрузка при инициализации
# или вручную:
adapter.load_models()

# Выгрузка:
adapter.unload_models()
```

#### Обработка запроса

```python
result = adapter.process_query(
    query="Что такое Python?",
    generation_mode="auto"  # auto, condensed, extended
)
```

### 2. DualGenerator (`eva_ai/memory/fractal_graph_v2/dual_generator.py`)

Физически разделённые генераторы для разных задач.

#### Структура класса

```python
class DualGenerator:
    """Объединённый генератор с 2 физическими моделями."""
    
    def __init__(self, llama_condensed, llama_extended, graph=None):
        self.condensed = CondensedGenerator(...)  # Model A
        self.extended = ExtendedGenerator(...)      # Model B
        self.graph = graph
    
    def generate(self, query, mode="auto", return_details=False):
        # Возвращает dict с деталями при return_details=True
        # Или только response при return_details=False
    
    def generate_condensed(self, query):
        # Краткий ответ
    
    def generate_extended(self, query):
        # Развёрнутый ответ
```

#### CondensedGenerator

Для быстрых кратких ответов.

| Параметр | Значение | Описание |
|----------|---------|----------|
| `max_tokens` | 512 | Максимум токенов |
| `temperature` | 0.1 | Низкая температура для стабильности |
| `repeat_penalty` | 1.3 | Штраф за повторения |

**Промт:**
```
Ты — краткий ассистент. Дай ответ в 1-3 предложениях.

Вопрос: {query}

Ответ:
```

#### ExtendedGenerator

Для развёрнутых ответов с контекстом.

| Параметр | Значение | Описание |
|----------|---------|----------|
| `max_tokens` | 2048 | Больше токенов для развёрнутости |
| `temperature` | 0.4 | Баланс между точностью и креативностью |
| `repeat_penalty` | 1.3 | Предотвращение повторений |

**Промт:**
```
Дай развёрнутый и подробный ответ. НЕ повторяй уже написанное.

Вопрос: {query}
Контекст: {graph_context}

Подробный ответ (без повторений):
```

### 3. EVAGenerator (`eva_ai/memory/fractal_graph_v2/eva_generator.py`)

Оригинальный генератор с гибридными токенами (сохранён для совместимости).

#### Типы запросов и ключевые слова

```python
QUERY_TYPE_KEYWORDS = {
    'кратко': ['кратко', 'вкратце', 'суть', 'кто такой', 
               'дай определение', 'назови', 'перечисли'],
    'подробно': ['подробно', 'детально', 'развернуто', 
                 'расскажи', 'объясни', 'опиши', 'проанализируй'],
    'technical': ['технич', 'как работает', 'механизм', 
                  'algorithm', 'how does'],
    'comparison': ['сравни', 'различия', 'vs', 'versus', 
                   'difference', 'compared']
}
```

#### Параметры генерации по типам

```python
GENERATION_PARAMS = {
    'кратко': {
        'max_tokens': 64,
        'temperature': 0.1,
        'use_chain': False
    },
    'подробно': {
        'max_tokens': 512,
        'temperature': 0.5,
        'use_chain': True
    },
    'technical': {
        'max_tokens': 512,
        'temperature': 0.4,
        'use_chain': False
    },
    'default': {
        'max_tokens': 256,
        'temperature': 0.3,
        'use_chain': True
    }
}
```

---

## Технические параметры

### Параметры моделей llama_cpp

```python
# Конфигурация загрузки
model_config = {
    'chat_format': 'qwen',           # Формат чата Qwen
    'n_ctx': 4096,                   # Размер контекста
    'n_threads': 8,                  # Потоки CPU
    'verbose': False,                # Без логов
    'n_gpu_layers': 0,               # CPU-only
}

# Параметры генерации
generation_config = {
    'max_tokens': 512-2048,          # Зависит от режима
    'temperature': 0.1-0.4,          # Зависит от режима
    'repeat_penalty': 1.3,           # Штраф за повторения
    'top_p': 0.9,                    # Ядерность выборки
    'stop': [
        '</s>',                       # Конец токена
        'User:', 'user:',            # Начало нового запроса
        'Human:', 'human:',          # Альтернатива
        'Вопрос:', 'Контекст:'       # Предотвращение зацикливания
    ],
    'echo': False                    # Не возвращать промт
}
```

### Размеры моделей

| Модель | Файл | Размер | Назначение |
|--------|------|--------|-----------|
| Model A | `qwen2.5-3b-instruct-q4_k_m.gguf` | ~2GB | Краткие ответы |
| Model B | `qwen2.5-3b-instruct-q4_k_m_model_b.gguf` | ~2GB | Развёрнутые ответы |

### Лимиты генерации

| Режим | max_tokens | Целевая длина | Время (примерное) |
|-------|------------|---------------|-------------------|
| condensed | 512 | 100-300 символов | 5-10 секунд |
| extended | 2048 | 500-2000 символов | 15-25 секунд |

---

## Reasoning System

### Структура reasoning_steps

```python
{
    'step': int,           # Номер шага (1, 2, 3...)
    'phase': str,         # Фаза (query_analysis, generation...)
    'thought': str,        # Описание мысли процесса
    'confidence': float,   # Уверенность (0.0-1.0)
    'icon': str           # Emoji иконка
}
```

### Иконки по фазам

```javascript
const icons = {
    'generation': '💭',              // Генерация
    'model_a_generation': '🧠',     // Model A
    'model_b_generation': '💡',     // Model B
    'query_analysis': '🔍',          // Анализ запроса
    'model_selection': '⚙️',         // Выбор модели
    'context_retrieval': '📚',      // Извлечение контекста
    'condensed': '📝',              // Краткий режим
    'extended': '📖',                // Развёрнутый режим
    'quality_check': '✅',          // Проверка качества
    'final_synthesis': '✨',        // Финальный синтез
    'contradiction_check': '⚖️',   // Проверка противоречий
    'ethics_check': '🛡️',          // Этическая проверка
    'web_search': '🌐',             // Веб-поиск
    'refinement': '🔄',            // Уточнение
    'self_dialog': '💬'             // Самодиалог
};
```

### Цвета индикаторов уверенности

```css
.step-conf.high {     /* confidence >= 0.7 */
    background: rgba(34, 197, 94, 0.2);  /* зелёный */
    color: #22c55e;
}

.step-conf.medium {   /* 0.4 <= confidence < 0.7 */
    background: rgba(234, 179, 8, 0.2); /* жёлтый */
    color: #eab308;
}

.step-conf.low {      /* confidence < 0.4 */
    background: rgba(239, 68, 68, 0.2); /* красный */
    color: #ef4444;
}
```

---

## Web GUI интеграция

### Отображение в интерфейсе

```
┌─────────────────────────────────────────────────────────────┐
│ Рассуждения (7 шагов)  ▼                                   │
├─────────────────────────────────────────────────────────────┤
│ 🔍 1  query_analysis     Анализ запроса...            100% │
│ ⚙️ 2  model_selection   Выбор режима...              90%  │
│ 📚 3  context_retrieval  Извлечение контекста...      85%  │
│ 🤖 4  generation         Генерация ответа...         80%  │
│ 📖 5  extended           Результат (18.3s, 890 chars) 90%  │
│ ✅ 6  quality_check      Проверка качества...        90%  │
│ ✨ 7  final_synthesis    Формирование ответа...      95%  │
└─────────────────────────────────────────────────────────────┘
```

### Файлы интеграции

| Файл | Назначение |
|------|------------|
| `server_main.py` | Формирование reasoning_steps |
| `app.js` | Отображение шагов рассуждения |
| `style.css` | Стили reasoning блока |

### Серверная часть (server_main.py)

```python
# Формирование reasoning_steps
reasoning_steps = []

reasoning_steps.append({
    'step': 1,
    'phase': 'query_analysis',
    'thought': f'Анализ запроса: "{query[:50]}..."',
    'confidence': 1.0,
    'icon': '🔍'
})

# ... другие шаги

return_data = {
    'response': response_text,
    'reasoning_steps': reasoning_steps,
    # ...
}
```

### Клиентская часть (app.js)

```javascript
// Отрисовка шагов
const stepsHtml = reasoning.map((step, idx) => {
    const phase = step.phase || 'unknown';
    const thought = step.thought || '';
    const conf = step.confidence || 0;
    const icon = step.icon || icons[phase] || '🔹';
    
    return `
        <div class="reasoning-step">
            <span class="step-icon">${icon}</span>
            <span class="step-num">${idx + 1}</span>
            <span class="step-phase">${phase}</span>
            <span class="step-thought">${thought}</span>
            <span class="step-conf ${confClass}">${(conf * 100).toFixed(0)}%</span>
        </div>
    `;
}).join('');
```

---

## API

### HybridPipelineAdapter

```python
from eva_ai.core.hybrid_pipeline_adapter import HybridPipelineAdapter

# Создание
adapter = HybridPipelineAdapter(
    fractal_graph=fractal_graph,
    mode='dual',
    model_a_path='path/to/model_a.gguf',
    model_b_path='path/to/model_b.gguf'
)

# Обработка запроса
result = adapter.process_query(
    query="Что такое Python?",
    generation_mode="auto"  # auto | condensed | extended
)

# Переключение режима
adapter.set_mode('fractal')  # dual | fractal | recursive | hybrid

# Статистика
stats = adapter.get_stats()
# {
#     'mode': 'dual',
#     'fractal_pipeline_ready': False,
#     'dual_generator_ready': True,
#     'dual_stats': {
#         'condensed': {'calls': 10, 'avg_time': 5.2},
#         'extended': {'calls': 5, 'avg_time': 18.3}
#     }
# }

# Управление моделями
adapter.load_models()
adapter.unload_models()
```

### DualGenerator

```python
from eva_ai.memory.fractal_graph_v2.dual_generator import DualGenerator

# Создание
dual = DualGenerator(
    llama_condensed=llama_model_a,
    llama_extended=llama_model_b,
    graph=fractal_graph
)

# Генерация с деталями
result = dual.generate("Что такое Python?", mode="auto", return_details=True)
# {
#     'response': 'Python — это язык программирования...',
#     'mode': 'condensed',
#     'time': 5.2,
#     'length': 156,
#     'tokens_estimate': 42
# }

# Простая генерация
response = dual.generate("Объясни разницу...", mode="extended")
# Возвращает только строку

# Прямой вызов генераторов
result = dual.generate_condensed("Что такое?")
result = dual.generate_extended("Расскажи подробнее...")

# Статистика
stats = dual.get_stats()
```

### Формат ответа

```python
{
    # Основной контент
    'response': str,                  # Основной ответ
    'final_response': str,             # Финальный ответ
    'natural_response': str,           # Натуральный ответ
    
    # Метаданные
    'confidence': float,               # Уверенность (0.0-1.0)
    'quality': {
        'score': float,               # Оценка качества
        'is_gibberish': bool,         # Мусор ли ответ
        'reasons': list               # Причины оценки
    },
    'query_type': str,                # Тип запроса
    
    # Reasoning
    'reasoning_steps': [              # Шаги рассуждения
        {
            'step': 1,
            'phase': 'query_analysis',
            'thought': '...',
            'confidence': 0.9,
            'icon': '🔍'
        },
        # ...
    ],
    
    # Производительность
    'processing_time': float,         # Общее время обработки
    'model_a_result': dict,           # Результат Model A
    'model_b_result': dict,           # Результат Model B
    'model_c_result': None,           # Model C не используется
    
    # Контекст
    'has_code': bool,                 # Содержит ли код
    'fractal_context': str,           # Контекст из графа
    'source': str                     # Источник (gguf_pipeline)
}
```

---

## Конфигурация

### brain_config.json

```json
{
  "model": {
    "use_two_model_pipeline": true,
    "pipeline_mode": "dual",
    
    "model_a_gguf_path": "eva_ai/memory/fractal_torch_storage/gguf_models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf",
    "model_b_gguf_path": "eva_ai/memory/fractal_torch_storage/gguf_models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m_model_b.gguf",
    "model_c_gguf_path": "eva_ai/memory/fractal_torch_storage/gguf_models/qwen2.5-coder-1.5b-instruct/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    
    "llama_cpp_n_ctx": 4096,
    "llama_cpp_threads": 8
  }
}
```

### brain_query.py

```python
# Важно: FG_ONLY_MODE должен быть False
FG_ONLY_MODE = False  # Отключено - используем HybridPipelineAdapter
```

### Инициализация в brain_components.py

```python
def _init_two_model_pipeline(brain):
    model_config = brain.config.get('model', {})
    pipeline_mode = model_config.get('pipeline_mode', 'dual')
    
    brain.two_model_pipeline = HybridPipelineAdapter(
        fractal_graph=brain.fractal_memory,
        mode=pipeline_mode,
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        n_ctx=model_config.get('llama_cpp_n_ctx', 4096),
        n_threads=model_config.get('llama_cpp_threads', 8),
        load_models=True
    )
    
    brain.two_model_pipeline_ready = True
```

---

## Оптимизации

### 1. Качество генерации

```python
def _check_quality(self, response: str, query: str) -> Dict:
    score = 0.8
    
    # Проверка повторений
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.2:
        score = 0.3  # Сильные повторения
    
    # Проверка гласных
    vowel_ratio = vowel_count / len(response)
    if vowel_ratio < 0.1:
        score = 0.2  # Мусор
    
    # Длина
    if len(response) < 30:
        score = 0.5
    elif len(response) > 50:
        score = 0.9
    
    return {'score': score, 'is_gibberish': score < 0.5}
```

### 2. Предотвращение повторений

```python
# repeat_penalty в llama_cpp
output = llama(
    prompt,
    max_tokens=2048,
    temperature=0.4,
    repeat_penalty=1.3,  # Штраф за повторения
    stop=["</s>", "User:", "Вопрос:", "Контекст:"]
)
```

### 3. Очистка ответов

```python
def _clean_response(self, text: str) -> str:
    # Удаление заполнителей
    fillers = ['хорошо,', 'конечно,', 'вот,', 'отлично,']
    for f in fillers:
        if text.lower().startswith(f):
            text = text[len(f):].strip()
    
    # Ограничение предложений для краткого режима
    if self.max_tokens < 200:
        sentences = text.split('.')[:2]
        text = '. '.join(sentences)
    
    return text
```

### 4. Контекст из графа

```python
def _get_context(self, query: str) -> str:
    """Извлечение релевантного контекста из FractalGraphV2."""
    relevant = []
    query_lower = query.lower()
    
    for node_id, node in self.graph.nodes.items():
        content = getattr(node, 'content', '')
        if content and any(kw in content.lower() for kw in query_lower.split()[:3]):
            relevant.append(content[:200])
    
    return ' | '.join(relevant[:3]) if relevant else "Нет контекста"
```

---

## Troubleshooting

### Pipeline не запускается

**Симптом:** Запросы обрабатываются через `fractal_graph_v2_random`, а не через `gguf_pipeline`.

**Решение:**
```python
# brain_query.py должен содержать:
FG_ONLY_MODE = False  # Не True!
```

### Модели не загружаются

**Симптом:** `DualGenerator ready: False`

**Решение:** Проверить пути в `brain_config.json`:
```json
"model_a_gguf_path": "полный/путь/к/model.gguf"
```

### Повторяющиеся ответы

**Симптом:** Модель повторяет один и тот же текст.

**Решение:**
```python
# Убедиться что repeat_penalty установлен
output = llama(..., repeat_penalty=1.3)
```

### Ответы слишком короткие

**Симптом:** Max tokens не достигается, ответы обрезаются.

**Решение:** Проверить `max_tokens` в `DualGenerator.__init__`:
- condensed: 512
- extended: 2048

### Нет reasoning steps в UI

**Симптом:** Блок "Рассуждения" пустой или не отображается.

**Решение:**
1. Проверить что `reasoning_steps` есть в ответе
2. Проверить что `app.js` обновлён
3. Очистить кэш браузера

---

## Файловая структура

```
eva_ai/
├── core/
│   ├── hybrid_pipeline_adapter.py    # Гибридный адаптер
│   ├── fractal_pipeline.py           # FractalPipeline
│   ├── recursive_model_pipeline.py  # Реэкспорт старого
│   ├── brain_components.py          # Инициализация
│   └── brain_query.py               # Обработка запросов
│
├── memory/
│   └── fractal_graph_v2/
│       ├── dual_generator.py         # Основной генератор ★
│       ├── eva_generator.py          # Гибридный генератор
│       ├── prompt_templates.py       # Системные промты
│       ├── hybrid_tokenizer.py       # Токенизация
│       ├── gguf_shadow.py            # Профилирование
│       ├── semantic_context_cache.py # Кэш контекста
│       └── storage.py                # FractalGraphV2
│
└── gui/
    └── web_gui/
        ├── server_main.py            # Сервер (reasoning_steps)
        └── static/
            ├── js/app.js              # Клиент (отрисовка)
            └── css/style.css          # Стили reasoning
```

---

## Запуск

```bash
# Запуск системы
cd C:\Users\black\OneDrive\Desktop\CogniFlex
python -m eva_ai

# Или через Web GUI
python start_webgui.py
```

**Web GUI:** http://127.0.0.1:5555

---

*Документация создана: 2026-04-09*
*Обновлено: 2026-04-09*
*Версия: 2.0*
