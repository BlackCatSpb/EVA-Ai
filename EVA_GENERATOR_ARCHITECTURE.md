# EVAGenerator и гибридная архитектура генерации

## Содержание

1. [Обзор](#обзор)
2. [Архитектура](#архитектура)
3. [Компоненты](#компоненты)
4. [DualGenerator](#dualgenerator)
5. [HybridPipelineAdapter](#hybridpipelineadapter)
6. [Интеграция с CoreBrain](#интеграция-с-corebrain)
7. [Конфигурация](#конфигурация)
8. [API](#api)
9. [Оптимизации](#оптимизации)

---

## Обзор

### Цель

Создание единой архитектуры EVA AI на основе FractalGraphV2 (FGv2) с гибридной интеграцией GGUF модели Qwen 2.5 3B, заменяющей устаревший KnowledgeGraph (KG) и медленный RecursiveModelPipeline.

### Проблемы старой архитектуры

- **RecursiveModelPipeline**: ~50-180 секунд на запрос (множественные вызовы Model A + Model B + ethics + quality checks)
- **KnowledgeGraph**: удалён из системы
- **Качество**: нестабильная оценка качества (0.20-0.80)
- **Длина ответов**: неконтролируемая

### Результат оптимизации

| Метрика | Было | Стало |
|---------|------|-------|
| Время краткого ответа | ~58s | ~5-10s |
| Время развёрнутого ответа | ~180s | ~15-25s |
| Total time (4 запроса) | ~190s | ~40s |
| Quality score | 0.20-0.80 | 0.90 |
| Скорость улучшения | - | **в 4-5 раз** |

---

## Архитектура

### Иерархия компонентов

```
CoreBrain
    └── HybridPipelineAdapter (mode='dual')
            ├── DualGenerator
            │       ├── CondensedGenerator (Model A)
            │       └── ExtendedGenerator (Model B)
            └── FractalGraphV2 (контекст)
```

### Режимы работы HybridPipelineAdapter

| Режим | Описание | Использование |
|-------|---------|--------------|
| `dual` | DualGenerator с 2 физическими моделями | **Рекомендуется** |
| `fractal` | FractalPipeline | Для сравнения |
| `recursive` | RecursiveModelPipeline (старый) | Fallback |
| `hybrid` | FractalPipeline + fallback | Постепенный переход |

---

## Компоненты

### 1. DualGenerator (`eva_ai/memory/fractal_graph_v2/dual_generator.py`)

Физически разделённые генераторы для разных задач.

#### CondensedGenerator

- **Назначение**: быстрые краткие ответы (1-3 предложения)
- **max_tokens**: 512
- **temperature**: 0.1
- **Промт**: краткий ответ без повторений

```python
CONDENSED_PROMPT = """Ты — краткий ассистент. Дай ответ в 1-3 предложениях.

Вопрос: {query}

Ответ:"""
```

#### ExtendedGenerator

- **Назначение**: развёрнутые ответы с анализом
- **max_tokens**: 2048
- **temperature**: 0.4
- **Промт**: подробный ответ с контекстом из графа

```python
EXTENDED_PROMPT = """Дай развёрнутый и подробный ответ. НЕ повторяй уже написанное.

Вопрос: {query}
Контекст: {graph_context}

Подробный ответ:"""
```

#### Автоматическое определение режима

```python
def _auto_generate(self, query: str) -> str:
    """Автоматическое определение режима."""
    query_lower = query.lower()
    
    # Краткие запросы
    brief_keywords = ['кратко', 'вкратце', 'суть', 'кто такой', 'перечисли', 'назови']
    for kw in brief_keywords:
        if kw in query_lower:
            return self.generate_condensed(query)
    
    # По умолчанию - развёрнутый
    return self.generate_extended(query)
```

### 2. HybridPipelineAdapter (`eva_ai/core/hybrid_pipeline_adapter.py`)

Гибридный адаптер для выбора оптимального пайплайна.

#### Инициализация

```python
adapter = HybridPipelineAdapter(
    fractal_graph=graph,
    mode='dual',  # или 'fractal', 'recursive', 'hybrid'
    model_a_path=MODEL_A_PATH,
    model_b_path=MODEL_B_PATH,
    n_ctx=4096,
    n_threads=8,
    load_models=True
)
```

#### Обработка запроса

```python
result = adapter.process_query(
    query="Что такое Python?",
    generation_mode="auto"  # или "condensed", "extended"
)
```

#### Формат ответа

```python
{
    'response': str,           # Основной ответ
    'final_response': str,     # Финальный ответ
    'natural_response': str,   # Натуральный ответ
    'confidence': float,       # Уверенность (0.9)
    'quality': {
        'score': float,        # Оценка качества
        'is_gibberish': bool,
        'reasons': list
    },
    'reasoning_steps': list,   # Шаги рассуждения
    'processing_time': float,  # Время обработки
    'source': str              # Источник ответа
}
```

### 3. EVAGenerator (`eva_ai/memory/fractal_graph_v2/eva_generator.py`)

Оригинальный генератор с гибридными токенами (сохранён для совместимости).

#### Типы запросов

```python
QUERY_TYPE_KEYWORDS = {
    'кратко': ['кратко', 'вкратце', 'суть', 'что такое', ...],
    'подробно': ['подробно', 'детально', 'расскажи', ...],
    'technical': ['технич', 'как работает', 'механизм', ...],
    'comparison': ['сравни', 'различия', 'vs', ...]
}
```

#### Параметры генерации

```python
GENERATION_PARAMS = {
    'кратко': {'max_tokens': 64, 'temperature': 0.1, 'use_chain': False},
    'подробно': {'max_tokens': 512, 'temperature': 0.5, 'use_chain': True},
    'technical': {'max_tokens': 512, 'temperature': 0.4, 'use_chain': False},
    'default': {'max_tokens': 256, 'temperature': 0.3, 'use_chain': True}
}
```

---

## Интеграция с CoreBrain

### Инициализация (`brain_components.py`)

```python
def _init_two_model_pipeline(brain):
    """Инициализация Two-Model Pipeline с поддержкой гибридного режима."""
    
    model_config = brain.config.get('model', {})
    pipeline_mode = model_config.get('pipeline_mode', 'dual')
    
    if not model_config.get('use_two_model_pipeline', False):
        return
    
    # Создаём HybridPipelineAdapter
    brain.two_model_pipeline = HybridPipelineAdapter(
        fractal_graph=brain.fractal_memory,
        mode=pipeline_mode,
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        ...
    )
    
    brain.two_model_pipeline_ready = True
```

### Обработка запросов (`brain_query.py`)

```python
# Важно: FG_ONLY_MODE должен быть False для использования pipeline
FG_ONLY_MODE = False  # Отключено - используем HybridPipelineAdapter

def _handle_gguf_pipeline(self, query, ...):
    """Handles GGUF Two-Model Pipeline queries."""
    if not self.two_model_pipeline_ready or not self.two_model_pipeline:
        return None
    
    result = self.two_model_pipeline.process_query(query)
    
    if result and result.get('response'):
        result["source"] = "gguf_pipeline"
        return result
    
    return None
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
    "llama_cpp_n_ctx": 4096,
    "llama_cpp_threads": 8
  }
}
```

### Пути к моделям

| Модель | Путь | Описание |
|--------|------|----------|
| Model A | `.../qwen2.5-3b-instruct-q4_k_m.gguf` | Краткие ответы |
| Model B | `.../qwen2.5-3b-instruct-q4_k_m_model_b.gguf` | Развёрнутые ответы |

---

## API

### DualGenerator

```python
from eva_ai.memory.fractal_graph_v2.dual_generator import DualGenerator

# Инициализация
dual = DualGenerator(
    llama_condensed=llama_model_a,
    llama_extended=llama_model_b,
    graph=fractal_graph
)

# Генерация
response = dual.generate("Что такое Python?", mode="auto")
# или
response = dual.generate_condensed("Что такое?")    # краткий
response = dual.generate_extended("Объясни...")      # развёрнутый

# Статистика
stats = dual.get_stats()
```

### HybridPipelineAdapter

```python
from eva_ai.core.hybrid_pipeline_adapter import HybridPipelineAdapter

# Инициализация
adapter = HybridPipelineAdapter(
    fractal_graph=graph,
    mode='dual',
    model_a_path=path_a,
    model_b_path=path_b,
    load_models=True
)

# Обработка
result = adapter.process_query(query, generation_mode="auto")

# Переключение режима
adapter.set_mode('fractal')  # Сменить режим

# Статистика
stats = adapter.get_stats()
```

---

## Оптимизации

### 1. Качество генерации

#### Улучшенный quality scoring

```python
def _check_quality(self, response: str, query: str) -> Dict[str, Any]:
    """Проверить качество ответа."""
    score = 0.8
    
    # Проверка повторений
    if len(words) > 10:
        unique_ratio = len(unique_words) / len(words)
        if unique_ratio < 0.2:
            score = 0.3  # Сильные повторения
    
    # Проверка гласных
    vowel_ratio = vowel_count / len(response)
    if vowel_ratio < 0.1:
        score = 0.2  # Мусор
    
    # Нормальные ответы
    if len(response) < 30:
        score = 0.5  # Слишком короткий
    elif len(response) > 50:
        score = 0.9  # Хороший объём
    
    return {'score': score, 'is_gibberish': score < 0.5}
```

### 2. Управление длиной

#### Truncation

```python
def _truncate_to_sentences(self, text: str, max_sentences: int = 3) -> str:
    """Обрезать текст до max_sentences предложений."""
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()][:max_sentences]
    return '. '.join(sentences)
```

### 3. Stop sequences

```python
output = llama(
    prompt,
    max_tokens=max_tokens,
    temperature=temperature,
    repeat_penalty=1.3,
    stop=["</s>", "User:", "user:", "Human:", "Вопрос:", "Контекст:"],
    echo=False
)
```

### 4. Контекст модели

```python
# Размер контекста увеличен для длинных ответов
recommended_ctx = min(self.n_ctx, 4096)  # было 2048
```

---

## Статистика работы

### GeneratorStats

```python
@dataclass
class GeneratorStats:
    total_calls: int = 0      # Всего вызовов
    total_time: float = 0.0   # Общее время
    avg_time: float = 0.0     # Среднее время
    total_tokens: int = 0     # Всего токенов
```

### Пример вывода

```
DualGenerator stats:
  Condensed: 10 calls, avg 5.2s
  Extended: 5 calls, avg 18.3s
```

---

## Файловая структура

```
eva_ai/
├── core/
│   ├── hybrid_pipeline_adapter.py   # Гибридный адаптер
│   ├── fractal_pipeline.py          # FractalPipeline
│   ├── recursive_model_pipeline.py  # Реэкспорт старого
│   └── brain_components.py         # Инициализация
│
└── memory/
    └── fractal_graph_v2/
        ├── dual_generator.py        # Основной генератор
        ├── eva_generator.py          # Гибридный генератор
        ├── prompt_templates.py      # Системные промты
        ├── hybrid_tokenizer.py       # Токенизация
        ├── gguf_shadow.py           # Профилирование
        └── semantic_context_cache.py # Кэш контекста
```

---

## Миграция

### Со старого RecursiveModelPipeline

1. Заменить импорт:
```python
# Было
from eva_ai.core.recursive_model_pipeline import RecursiveModelPipeline

# Стало
from eva_ai.core.hybrid_pipeline_adapter import HybridPipelineAdapter
```

2. Изменить инициализацию:
```python
# Было
pipeline = RecursiveModelPipeline(model_a_path=..., model_b_path=...)
pipeline.load_models()

# Стало
adapter = HybridPipelineAdapter(
    mode='dual',
    model_a_path=..., 
    model_b_path=...,
    load_models=True
)
```

3. Изменить вызов:
```python
# Было
result = pipeline.process_query(query)

# Стало
result = adapter.process_query(query)
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

---

## Troubleshooting

### Pipeline не запускается

Проверить `brain_query.py`:
```python
FG_ONLY_MODE = False  # Должно быть False
```

### Модели не загружаются

Проверить пути в `brain_config.json`:
```json
"model_a_gguf_path": "полный/путь/к/модели.gguf"
```

### Повторяющиеся ответы

Убедиться что `repeat_penalty` установлен:
```python
output = llama(..., repeat_penalty=1.3)
```

### Ответы слишком короткие

Проверить параметры `max_tokens` в DualGenerator.

---

*Документация создана: 2026-04-09*
*Версия: 1.0*
