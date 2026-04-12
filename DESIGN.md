# EVA-Ai — Архитектура и Принципы Работы

**Версия:** 3.1  
**Дата:** 2026-04-12  
**Репозиторий:** https://github.com/BlackCatSpb/EVA-Ai

---

## 1. Quick Reference

### 1.1 Основные точки входа

| Файл | Назначение | Команда запуска |
|------|------------|-----------------|
| `eva/run.py` | Запуск Core + Web GUI | `python -m eva` |
| `wsgi.py` | Production (Gunicorn) | `gunicorn -c gunicorn_config.py wsgi:app` |

### 1.2 Ключевые компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| UnifiedGenerator | `core/unified_generator.py` | Трёхэтапная итеративная генерация |
| PipelineAdapter | `core/pipeline_adapter.py` | Адаптер для совместимости |
| FractalGraphV2 | `memory/fractal_graph_v2/` | Фрактальная память |
| SimpleRouter | `core/unified_generator.py` | L2 роутинг моделей |

---

## 2. Архитектура Системы

### 2.1 Философия системы

EVA-Ai — когнитивная система с **итеративным рассуждением**. Ключевое отличие от традиционных LLM:

```
Запрос → LOGIC (краткий ответ) → CONTEXT (расширение) → LOGIC (рефлексия) → Ответ
```

Каждый ответ проходит три этапа:
1. **LOGIC** — формирует краткий, логичный ответ
2. **CONTEXT** — расширяет ответ с учётом контекста
3. **LOGIC (рефлексия)** — проверяет противоречия, извлекает концепты

### 2.2 Pie UnifiedGenerator Архитектура

UnifiedGenerator заменяет старый Two-Model Pipeline и использует три специализированные модели:

| Модель | Тип | Назначение | Контекст |
|--------|-----|------------|----------|
| **LOGIC** | RuadaptQwen3-4B | Логика, рассуждения, краткие ответы | 4096 токенов |
| **CONTEXT** | RuadaptQwen3-4B | Развёрнутые объяснения, анализ | 32768 токенов |
| **CODER** | Qwen Coder 1.5B | Генерация кода | 4096 токенов |

**L2 Роутинг (SimpleRouter):**
- CODED_KEYWORDS (2+ совпадений) → CODER модель
- CONTEXT_KEYWORDS (1+ совпадений) или длина ≥ 25 → CONTEXT модель
- По умолчанию → LOGIC модель

### 2.3 Chat Template

Система использует правильный Qwen chat template:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
```

**Stop tokens:** `<|im_end|>`, `<|im_start|>`, `<|endoftext|>`

---

## 3. Компоненты Системы

### 3.1 UnifiedGenerator

**Класс GenerationResult:**
```python
@dataclass
class GenerationResult:
    text: str                          # Сгенерированный текст
    model_used: str                    # Какая модель использовалась
    generation_time: float            # Время генерации
    tokens_generated: int              # Количество токенов
    confidence: float                  # Уверенность (0-1)
    metadata: Optional[Dict]          # Дополнительные данные
```

**Методы:**
- `generate()` — одиночная генерация через роутинг
- `generate_dual()` — двухэтапная LOGIC → CONTEXT
- `generate_iterative()` — трёхэтапная с рефлексией (текущая)

### 3.2 PipelineAdapter

Адаптер обеспечивает совместимость UnifiedGenerator с интерфейсом TwoModelPipeline. Все запросы направляются через `generate_iterative()`.

**Интеграция в CoreBrain:**
- Создаётся в `init_factories.py`
- Привязывается к `brain.two_model_pipeline`
- Используется в `brain_query.py` при `disable_pytorch=True`

### 3.3 FractalGraphV2

**Структура узлов:**
- `concept` — концепты и идеи
- `conversation` — история диалогов
- `contradiction` — обнаруженные противоречия
- `entity` — именованные сущности

**Связи:**
- `related_to` — семантическая связь
- `contradicts` — противоречие
- `supports` — поддержка

### 3.4 WebGUI

**Endpoints:**
- `/api/chat` — основной endpoint для сообщений
- `/api/status` — статус системы
- `/api/shutdown` — корректное завершение
- `/api/stream` — streaming генерация

**Особенности:**
- Markdown рендеринг (regex-based)
- GPT-подобные блоки кода
- Отображение reasoning_steps
- Сессионное управление

---

## 4. Поток Обработки Запроса

```
1. WebGUI → Приём сообщения
2. EthicsFramework → Проверка контента
3. Cache Check → Быстрый кэш (приветствия)
4. UnifiedGenerator.generate_iterative()
   ├── Этап 1: LOGIC → краткий ответ
   ├── Этап 2: CONTEXT → расширение
   └── Этап 3: LOGIC → рефлексия + концепты
5. FractalGraphV2 → Сохранение
6. Ответ WebGUI
```

---

## 5. Конфигурация

### 5.1 brain_config.json

```json
{
  "model": {
    "use_unified_generator": true,
    "logic_model_path": ".../ruadapt_qwen3_4b_q4_k_m.gguf",
    "context_model_path": ".../ruadapt_qwen3_4b_q4_k_m.gguf",
    "coder_model_path": ".../qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "llama_cpp_n_ctx": 16384,
    "llama_cpp_threads": 8,
    "disable_pytorch": true
  }
}
```

### 5.2 Environment Variables

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — для CUDA
- TF32 ускорение активируется автоматически

---

## 6. Запуск и Остановка

### 6.1 Запуск

```bash
cd C:\Users\black\OneDrive\Desktop\CogniFlex
python -m eva_ai
```

### 6.2 Остановка

- **Ctrl+C** — корректное завершение через signal handler
- **WebGUI** → Settings → "Остановить EVA" — вызов `/api/shutdown`

---

## 7. Различия v3.0 → v3.1

| Компонент | v3.0 | v3.1 |
|-----------|------|------|
| Генерация | Two-Model (A → B) | Трёхэтапная (LOGIC → CONTEXT → LOGIC) |
| Рефлексия | Нет | Проверка противоречий и концептов |
| Модели | Qwen2.5-3B × 2 | RuadaptQwen3-4B + Coder 1.5B |
| Routing | По ключевым словам | По ключевым словам + длине |

---

## 8. Ключевые файлы

```
eva_ai/
├── core/
│   ├── unified_generator.py    # Ядро генерации (Pie)
│   ├── pipeline_adapter.py     # Адаптер совместимости
│   ├── brain_query.py          # Обработка запросов
│   └── init_factories.py       # Инициализация компонентов
├── gui/web_gui/
│   ├── server_main.py          # Flask приложение
│   ├── server_routes_chat.py  # Chat endpoints
│   └── static/js/app.js        # Frontend
├── memory/fractal_graph_v2/    # FractalGraph система
└── run.py                      # Точка входа
```
