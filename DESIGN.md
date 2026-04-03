# EVA-Ai — Архитектура и Принципы Работы

**Версия:** 2.0  
**Дата:** 2026-04-03  
**Репозиторий:** https://github.com/BlackCatSpb/EVA-Ai

---

## 1. Обзор Системы

EVA-Ai — когнитивная нейросетевая система с рекурсивным рассуждением, фрактальной памятью и самообучением.

### Ключевые особенности

- **Three-GGUF Pipeline** — три модели работают последовательно: логика → развитие концепций → генерация кода
- **Адаптивная генерация** — параметры автоматически подстраиваются при ошибках (зацикливание, фразы-паразиты, семантическая застрялость)
- **Фрактальная память** — иерархический граф знаний с горячими/тёплыми/холодными узлами
- **Самообучение** — система анализирует диалоги, выявляет пробелы знаний, ведёт самодиалоги
- **GPU-ускорение** — embedding-модель на CUDA, KV-кэш q8_0
- **Веб-интерфейс** — чат с историей, рассуждениями, управлением сессиями

### Аппаратные требования

| Компонент | Требование |
|-----------|------------|
| CPU | 6+ ядер (AVX2) |
| RAM | 16 ГБ минимум |
| GPU | CUDA с 2+ ГБ VRAM (опционально, для эмбеддингов) |
| SSD | 10+ ГБ свободно |

---

## 2. Архитектура Компонентов

### 2.1 Ядро (Core)

| Компонент | Файл | Описание |
|-----------|------|----------|
| **CoreBrain** | `core/core_brain.py` | Центральный координатор, инициализация, обработка запросов |
| **ComponentInitializer** | `core/component_initializer.py` | Фабрика компонентов, управление зависимостями |
| **RecursiveModelPipeline** | `core/recursive_model_pipeline.py` | Three-GGUF pipeline с адаптивными параметрами |
| **AdaptiveParameterController** | `core/recursive_model_pipeline.py` | Контроллер адаптивных параметров генерации |
| **EventBus** | `core/event_bus.py` | Pub/sub событийная шина |
| **ResourceManager** | `core/resource_manager.py` | Мониторинг CPU/RAM/GPU |
| **SystemMonitor** | `core/system_monitor.py` | Health checks, метрики |
| **DeferredCommandSystem** | `core/deferred_command_system.py` | Отложенные команды |

### 2.2 Рассуждение (Reasoning)

| Компонент | Файл | Описание |
|-----------|------|----------|
| **SelfReasoningEngine** | `reasoning/self_reasoning_engine.py` | Цикл рассуждения с контекстными вопросами |
| **ReasoningIntegration** | `reasoning/integration.py` | Интеграция SRE с CoreBrain |
| **EnhancedReasoningEngine** | `reasoning/enhanced_reasoning_engine.py` | Расширенное рассуждение с аналитикой |
| **FractalStorage** | `reasoning/fractal_ml/fractal_storage.py` | Фрактальное хранилище рассуждений |

### 2.3 Память (Memory)

| Компонент | Файл | Описание |
|-----------|------|----------|
| **UnifiedFractalMemory** | `memory/unified_fractal_memory.py` | Единая фрактальная память (граф) |
| **GraphLearning** | `memory/graph_learning.py` | Обучение на графе памяти |
| **HybridTokenCache** | `memory/hybrid_token_cache.py` | Гибридный кэш токенов (RAM/SSD) |
| **EmbeddingCache** | `memory/embedding_cache.py` | Persistent LRU кэш эмбеддингов (SQLite) |

### 2.4 Машинное обучение (MLearning)

| Компонент | Файл | Описание |
|-----------|------|----------|
| **FractalModelManager** | `mlearning/fractal_model_manager.py` | Управление GGUF моделями |
| **HybridModelManager** | `mlearning/hybrid_model_manager.py` | Гибридное управление моделями |
| **QwenModelManager** | `mlearning/qwen_model_manager.py` | Управление PyTorch моделями Qwen |
| **UnifiedTextProcessor** | `mlearning/unified_text_processor.py` | Обработка текста + эмбеддинги |
| **SentenceTransformersCache** | `mlearning/sentence_transformers_cache.py` | Кэш embedding-моделей |
| **MLUnit** | `mlearning/ml_unit.py` | ML ядро системы |
| **TokenizerRegistry** | `mlearning/tokenizer_registry.py` | Реестр токенизаторов |

### 2.5 Генерация (Generation)

| Компонент | Файл | Описание |
|-----------|------|----------|
| **GenerationCoordinator** | `generation/generation_coordinator.py` | Координация генерации ответов |
| **ResponseGenerator** | `response_generator.py` | Генерация финальных ответов |

### 2.6 Обучение (Learning)

| Компонент | Файл | Описание |
|-----------|------|----------|
| **SelfDialogLearning** | `learning/self_dialog_learning.py` | Самодиалоги для обучения |
| **LearningManager** | `learning/learning_manager.py` | Управление обучением |
| **LearningOpportunityManager** | `learning_opportunity_manager.py` | Выявление возможностей обучения |

### 2.7 Аналитика и Контроль

| Компонент | Файл | Описание |
|-----------|------|----------|
| **ContradictionManager** | `contradiction/contradiction_detection.py` | Обнаружение противоречий |
| **EthicsFramework** | `ethics/ethics_framework.py` | Этическая проверка |
| **AnalyticsManager** | `analytics/analytics_manager.py` | Аналитика системы |
| **AnalyzerCore** | `analyzer_core.py` | Ядро анализа |
| **PerformanceAnalyzer** | `performance_analyzer.py` | Анализ производительности |

### 2.8 Знания и Поиск

| Компонент | Файл | Описание |
|-----------|------|----------|
| **KnowledgeGraph** | `knowledge/knowledge_graph.py` | Граф знаний |
| **WebSearchEngine** | `web_search.py` | Поиск в интернете |
| **QwenAPIEnhancer** | `knowledge/qwen_api_enhancer.py` | Обогащение через Wikipedia/Web |

### 2.9 Интерфейс (GUI)

| Компонент | Файл | Описание |
|-----------|------|----------|
| **WebGUI Server** | `gui/web_gui/server.py` | Flask сервер + API |
| **WebGUI Frontend** | `gui/web_gui/static/` | HTML/CSS/JS интерфейс |
| **SessionManager** | `gui/web_gui/server.py` | Управление сессиями и историей чата |

---

## 3. Поток Данных

### 3.1 Обработка запроса

```
User Query (Web GUI)
        ↓
┌───────────────────────────────────────────────┐
│  WebGUI Server (server.py)                    │
│  ├─ Ethics check                              │
│  ├─ Entity extraction                         │
│  ├─ Chat history → chat_history[]             │
│  └─ Context nodes → context_nodes[]           │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│  CoreBrain.process_query()                    │
│  ├─ Query analysis (кратко/подробно/код)      │
│  ├─ Chat history из сессии                    │
│  └─ Выбор пайплайна генерации                 │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│  SelfReasoningEngine.process_query()          │
│  ├─ Build contextual query (история + запрос) │
│  ├─ Determine query type                      │
│  └─ Запуск Two-Model Pipeline                 │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│  RecursiveModelPipeline.process_query()       │
│  ├─ Fractal context из графа памяти           │
│  ├─ Model A: логический ответ                 │
│  │   └─ Adaptive params + quality check       │
│  ├─ Model B: расширенный ответ                │
│  │   └─ Adaptive params + quality check       │
│  ├─ Model C: код (если нужен)                 │
│  └─ Final response                            │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│  Post-processing                              │
│  ├─ Сохранение в chat_history[]               │
│  ├─ Сохранение в context_nodes[]              │
│  ├─ Graph learning (опыт)                     │
│  ├─ Self-dialog learning                      │
│  └─ Return response → Web GUI                │
└───────────────────────────────────────────────┘
```

### 3.2 Two-Model Pipeline (детально)

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Model A (Логическое Ядро)                         │
│  GGUF: qwen2.5-3b-instruct-q4_k_m.gguf                     │
│  Задача: Извлечь факты, дать точный ответ                  │
│  Params: temp=0.3, top_p=0.9, top_k=40, rep_pen=1.5        │
│  Quality check: зацикливание, китайский, фразы-паразиты    │
│  ↓ Если провал → AdaptiveParameterController:              │
│    - Зацикливание: ↑temp +0.25, ↑rep_pen +0.4, ↓top_k -10  │
│    - Фразы-паразиты: ↑temp +0.2, ↑top_k +20                │
│    - Семантическая застрялость: радикальный сдвиг всех     │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Model B (Развитие Концепций)                      │
│  GGUF: qwen2.5-3b-instruct-q4_k_m_model_b.gguf             │
│  Задача: Развить мысль, добавить детали и примеры          │
│  Params: temp=0.3, top_p=0.9, top_k=40, rep_pen=2.0        │
│  Quality check: те же проверки                             │
│  ↓ Если провал → те же адаптивные стратегии                │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Model C (Кодер) — ленивая загрузка                │
│  GGUF: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf             │
│  Задача: Генерация кода (только при запросах с кодом)      │
│  Загружается только при обнаружении code-ключевых слов     │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Адаптивные параметры

**Диапазоны:**

| Параметр | Мин | База | Макс |
|----------|-----|------|------|
| temperature | 0.05 | 0.3 | 1.5 |
| top_p | 0.1 | 0.9 | 1.0 |
| top_k | 10 | 40 | 100 |
| repeat_penalty | 0.5 | 1.5 | 3.0 |
| max_tokens | 64 | 1024 | 4096 |

**Стратегии адаптации:**

| Причина провала | Действие |
|-----------------|----------|
| Зацикливание | ↑temperature +0.25, ↑repeat_penalty +0.4, ↓top_k -10 |
| Китайские символы | ↓temperature -0.15, ↓top_p -0.2, ↑repeat_penalty +0.3 |
| Фразы-паразиты | ↑temperature +0.2, ↑top_k +20, ↑repeat_penalty +0.2 |
| Пустой ответ | ↑max_tokens +256, ↑temperature +0.15 |
| Семантическая застрялость (>0.85 схожесть) | Радикальный сдвиг: ↑↑temperature, ↑↑repeat_penalty |

---

## 4. Память и Обучение

### 4.1 UnifiedFractalMemory

Иерархический граф с 1046 узлами и 1042 связями:

```
Уровень 0: model_root (6 узлов)
Уровень 1: component (12 узлов)
Уровень 2: chunk (28 узлов)
Уровень 3: layer (100 узлов)
Уровень 4: tensor (900 узлов)
```

**Уровни доступа:**
- **Hot** (18 узлов) — часто используемые, в RAM
- **Cold** (1028 узлов) — редко используемые, на SSD

### 4.2 Graph Learning

Система сохраняет опыт после каждой генерации:
- Model A ответ → опыт с quality score
- Model B ответ → опыт с quality score
- Кластеризация опытов для поиска паттернов

### 4.3 Self-Dialog Learning

Автоматические самодиалоги:
1. После каждого запроса пользователя создаётся тема для самодиалога
2. Система ведёт внутренний диалог по теме
3. Результаты сохраняются в граф памяти
4. Выявляются пробелы знаний для дальнейшего обучения

### 4.4 Embedding Cache

SQLite-based LRU кэш (100,000 записей):
- Ключ: SHA256 хеш текста
- Значение: embedding vector (768-dim, multilingual-e5-base)
- Автоматическая eviction старых записей
- Ускоряет повторные запросы в 10-100x

---

## 5. Веб-интерфейс

### 5.1 Архитектура

```
┌─────────────────────────────────────────────┐
│  Browser (HTML/CSS/JS)                      │
│  ├─ Chat view с историей                    │
│  ├─ Reasoning panel (этапы рассуждения)     │
│  ├─ Session management                      │
│  └─ File attachment                         │
└──────────────────┬──────────────────────────┘
                   ↓ HTTP/JSON
┌─────────────────────────────────────────────┐
│  Flask Server (server.py)                   │
│  ├─ POST /api/chat → brain.process_query()  │
│  ├─ GET  /api/status → system health        │
│  ├─ GET  /api/system → system info          │
│  ├─ POST /api/login → authentication        │
│  └─ Session management                      │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  CoreBrain → SelfReasoningEngine → Pipeline │
└─────────────────────────────────────────────┘
```

### 5.2 Сессии и История

- **chat_history[]** — сырая история чата (role/content/timestamp), используется для контекста генерации
- **context_nodes[]** — структурированные узлы (entities, reasoning, file_data), используются для анализа
- При логине всегда создаётся новая чистая сессия
- Последние 20 сообщений передаются как контекст

### 5.3 Кнопки действий

Каждое сообщение имеет:
- **Копировать** — копировать текст в буфер
- **Полезно** — положительная оценка
- **Неверно** — отрицательная оценка
- **Перегенерировать** — повторная генерация (только для ответов ЕВА)

---

## 6. Конфигурация

### 6.1 brain_config.json (ключевые секции)

```json
{
  "model": {
    "use_two_model_pipeline": true,
    "model_a_gguf_path": "eva/memory/fractal_torch_storage/gguf_models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf",
    "model_b_gguf_path": "eva/memory/fractal_torch_storage/gguf_models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m_model_b.gguf",
    "model_c_gguf_path": "eva/memory/fractal_torch_storage/gguf_models/qwen2.5-coder-1.5b-instruct/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "n_ctx": 8192,
    "n_threads": 8
  },
  "reasoning": {
    "enabled": true,
    "max_iterations": 5,
    "confidence_threshold": 0.75
  },
  "embedding": {
    "model_name": "intfloat/multilingual-e5-base",
    "device": "auto"
  }
}
```

### 6.2 Системные промпты

**Model A (Логическое Ядро):**
```
Ты — Модуль Логического Ядра EVA.
Задача: Извлекать точные факты из запроса без расширений.
Спецификации:
1. Отвечай только подтверждёнными фактами.
2. Максимум 3 предложения.
3. Не используй слова «возможно», «вероятно», «может быть».
4. Избегай оценочных суждений.
5. Если информации недостаточно — сообщи об этом прямо.
Ограничения:
- Не добавляй вступления или заключения.
- Не повторяй вопрос пользователя.
- Отвечай строго на русском языке.
Формат вывода: Русский. Ответ начинается сразу с факта.
Конец инструкции.
```

**Model B (Развитие Концепций):**
```
Ты — Модуль Развития Концепций EVA.
Задача: Развивай мысль, добавляй детали и примеры.
Спецификации:
1. Расширяй факты примерами и пояснениями.
2. Используй структурированный формат (списки, абзацы).
3. Отвечай строго на русском языке.
4. Добавляй контекст и смежные темы.
5. Максимум 10 предложений.
Ограничения:
- Не повторяй факты дословно.
- Не используй английские или китайские вставки.
Формат вывода: Русский, развёрнутый ответ.
Конец инструкции.
```

---

## 7. Защита и Безопасность

### 7.1 Singleton Protection

PID-файл (`.eva_instance.pid`) предотвращает запуск нескольких экземпляров:
- При старте создаётся PID-файл
- При повторном запуске проверяется жив ли процесс
- При завершении PID-файл удаляется (atexit)
- Stale PID-файлы автоматически очищаются

### 7.2 Этическая рамка

7 этических принципов проверяют каждый запрос и ответ.

### 7.3 Обнаружение противоречий

Система обнаруживает противоречия между текущим ответом и сохранёнными знаниями.

---

## 8. Запуск

```bash
python -m eva
```

Система запускает:
1. CoreBrain с инициализацией всех компонентов
2. Two-Model Pipeline (Model A + B + C lazy)
3. Embedding модель на GPU (если доступна)
4. Web GUI на http://127.0.0.1:5555
5. Фоновые задачи (self-dialog, graph curator, monitoring)

---

## 9. Структура Директорий

```
CogniFlex/
├── eva/
│   ├── core/                    # Ядро системы
│   │   ├── core_brain.py        # Главный координатор
│   │   ├── recursive_model_pipeline.py  # Three-GGUF pipeline
│   │   ├── component_initializer.py     # Фабрика компонентов
│   │   ├── event_bus.py         # Событийная шина
│   │   └── ...
│   ├── reasoning/               # Рассуждение
│   │   ├── self_reasoning_engine.py
│   │   ├── integration.py
│   │   ├── enhanced_reasoning_engine.py
│   │   └── fractal_ml/          # Фрактальное хранилище
│   ├── memory/                  # Память
│   │   ├── unified_fractal_memory.py
│   │   ├── graph_learning.py
│   │   ├── hybrid_token_cache.py
│   │   └── embedding_cache.py
│   ├── mlearning/               # ML компоненты
│   │   ├── fractal_model_manager.py
│   │   ├── unified_text_processor.py
│   │   ├── sentence_transformers_cache.py
│   │   └── ...
│   ├── gui/web_gui/             # Веб-интерфейс
│   │   ├── server.py
│   │   ├── templates/index.html
│   │   └── static/
│   │       ├── css/style.css
│   │       └── js/app.js
│   ├── learning/                # Обучение
│   ├── knowledge/               # Знания
│   ├── contradiction/           # Противоречия
│   ├── ethics/                  # Этика
│   ├── analytics/               # Аналитика
│   └── generation/              # Генерация
├── brain_config.json            # Конфигурация
├── logs/                        # Логи
├── eva/memory/                  # Фрактальная память
└── eva/mlearning/eva_models/    # PyTorch модели
```

---

## 10. История Версий

| Версия | Дата | Описание |
|--------|------|----------|
| 1.0 | 2026-03-23 | Начальная архитектура |
| 1.5 | 2026-03-26 | Qwen-only модель, исправления конфигурации |
| 2.0 | 2026-04-03 | Three-GGUF Pipeline, адаптивные параметры, Web GUI, GPU embeddings, singleton protection, embedding cache, KV-cache q8_0, structured prompts, semantic stuck detection |
