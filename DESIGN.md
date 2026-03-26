# CogniFlex Архитектура: Фрактальное Хранилище + Self-Reasoning

## Дата: 2026-03-26
Версия: 1.5

---

## 1. Цель Системы

Создать когнитивную систему с:
- **Фрактальным хранилищем** - иерархическая ML-структура с рекурсивной адресацией
- **Self-Reasoning Engine** - цикл рассуждения с контекстными вопросами
- **Единой моделью Qwen3.5-0.8b** - только один экземпляр, никаких fallback
- **Полной аналитикой** - ethics, contradiction, knowledge gaps в цикле рассуждения
- **Гибридным кэшем** - VRAM → RAM → SSD tiered caching

---

## 2. Текущая Архитектура (2026-03-26)

### 2.1 Компоненты Системы (28 компонентов)

| Компонент | Статус | Описание |
|-----------|--------|----------|
| CoreBrain | ✅ | Центральный координатор |
| ComponentInitializer | ✅ | Управление зависимостями при инициализации |
| EventBus | ✅ | Pub/sub событийная шина |
| ResourceManager | ✅ | Управление ресурсами |
| ConfigManager | ✅ | Загрузка brain_config.json |
| MemoryManager | ✅ | Управление памятью |
| HybridTokenCache | ✅ | Гибридный кэш токенов (VRAM/RAM/SSD) |
| KnowledgeGraph | ✅ | Граф знаний (4 уровня абстракции) |
| TextProcessor | ✅ | Обработка текста |
| MLUnit | ✅ | ML ядро системы |
| ModelManager (Qwen) | ✅ | qwen3.5-0.8b |
| QueryProcessor | ✅ | Обработка запросов |
| ResponseGenerator | ✅ | Генерация ответов |
| ReasoningEngine | ✅ | Логика рассуждений |
| SelfReasoningEngine | ✅ | Self-reasoning цикл |
| TrainingOrchestrator | ✅ | Оркестрация обучения |
| LearningManager | ✅ | Управление обучением |
| LearningScheduler | ✅ | Планировщик обучения |
| SystemMonitor | ✅ | Мониторинг системы |
| MetricsCollector | ✅ | Сбор метрик |
| AnalyticsManager | ✅ | Аналитика |
| ContradictionManager | ✅ | Обнаружение противоречий |
| EthicsFramework | ✅ | Этическая проверка |
| WebSearchEngine | ✅ | Поиск в интернете |
| QwenAPIEnhancer | ✅ | Qwen API + Wikipedia fallback |
| BackgroundCoordinator | ✅ | Фоновые задачи |
| AdaptationManager | ✅ | Адаптация системы |
| GUI (CogniFlexGUI) | ✅ | Tkinter интерфейс |

### 2.2 Конфигурация Модели (brain_config.json)

```json
{
  "model": {
    "name": "qwen3.5-0.8b",
    "type": "qwen",
    "path": "cogniflex/mlearning/cogniflex_models/qwen3.5-0.8b",
    "max_length": 32768,
    "max_new_tokens": 2048,
    "device": "cuda"
  },
  "generation": {
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "max_new_tokens": 2048
  },
  "hybrid_cache": {
    "max_hot_tokens": 8192,
    "device": "cuda",
    "max_context_length": 32768
  }
}
```

---

## 3. Исправленные Ошибки (AI Agent Fixes)

### 3.1 Критические Исправления

| # | Проблема | Файл:Линия | Исправление |
|---|----------|------------|-------------|
| 1 | Tokenizer is None | response_generator.py | Добавлен fallback на self.tokenizer |
| 2 | Windows path в tokenizer | text_processor.py | Добавлен fallback на относительный путь |
| 3 | GUI "ML недоступна" | chat_module.py | Добавлена проверка ml_unit |
| 4 | FractalStorage параметр | component_initializer.py | storage_path → storage_dir |
| 5 | Config import shadowing | config.py vs config/ | Перенесено в config/__init__.py |
| 6 | max_new_tokens = 256 | qwen_model_manager.py | Изменено на 2048 |
| 7 | max_length = 2048 | text_processor.py | Изменено на 32768 |
| 8 | Empty pass statements | multiple files | Заменены на logger.warning |
| 9 | Hardcoded device | model_manager.py | Используется из config |
| 10 | Error masking | ml_unit.py:709,713 | Возвращает False при ошибке |

### 3.2 Конфигурационные Исправления

| # | Проблема | Исправление |
|---|----------|-------------|
| 1 | RUGPT3 в config | Изменено на Qwen3.5-0.8b |
| 2 | ALLOWED_MODELS | ['qwen', 'qwen3.5-0.8b', 'qwen3.5-2b'] |
| 3 | DEFAULT_MODEL | qwen3.5-0.8b |
| 4 | MODEL_CONFIGS | qwen3.5-0.8b первой в списке |

---

## 4. Фрактальное Хранилище

### 4.1 Структура Директорий

```
cogniflex/reasoning/
├── __init__.py
├── fractal_ml/
│   ├── __init__.py
│   ├── fractal_base.py       # Базовые классы (FractalNode, FractalEdge)
│   ├── fractal_tokenizer.py  # Собственная токенизация
│   ├── fractal_embedder.py   # Эмбеддинги для адресации
│   ├── fractal_address.py    # Рекурсивная адресация L0→L1→L2→L3
│   ├── fractal_storage.py    # Главное хранилище
│   ├── fractal_retriever.py  # Извлечение с нужной глубиной
│   └── fractal_index.py      # Индексация и поиск
├── self_reasoning_engine.py  # Главный движок рассуждения
├── confidence_scorer.py       # Оценка уверенности
├── clarification_generator.py # Генерация контекстных вопросов
├── reasoning_types.py        # Типы узлов для KG
└── integration.py            # Интеграция с CoreBrain
```

### 4.2 Параметры Фрактальной Структуры

| Параметр | Значение | Описание |
|----------|----------|----------|
| MAX_LEVELS | 4 | Глубина иерархии |
| BRANCHING_FACTOR | 16 | Ветвей на уровень |
| BASE_SIZE_KB | 1 | Размер L0 в KB |
| EMBEDDING_DIM | 384 | Размер эмбеддингов |
| MAX_NODES_PER_LEVEL | 16384 | Макс. узлов на уровень |

### 4.3 Формула Размеров

```
L0 = 1 KB
L1 = 16 KB   (L0 × 16)
L2 = 256 KB (L1 × 16)
L3 = 4 MB   (L2 × 16)
```

---

## 5. Self-Reasoning Engine

### 5.1 Цикл Работы

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

### 5.2 Параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| MAX_ITERATIONS | 5 | Макс. циклов уточнения |
| CONFIDENCE_THRESHOLD | 0.75 | Минимальная уверенность |
| ETHICS_WEIGHT | 0.30 | Вес в формуле уверенности |
| CONTRADICTION_WEIGHT | 0.30 | Вес в формуле уверенности |
| KNOWLEDGE_WEIGHT | 0.40 | Вес в формуле уверенности |

### 5.3 Формула Уверенности

```
Confidence = (ethics_score × 0.30) + 
              (contradiction_score × 0.30) + 
              (knowledge_score × 0.40)
```

---

## 6. Гибридный Кэш (HybridTokenCache)

### 6.1 Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID CACHE TIERS                       │
├─────────────────────────────────────────────────────────────┤
│  VRAM (cuda:0)     →  2883685 токенов (~11GB)              │
│       ↓                                                     │
│  RAM              →  буфер                                  │
│       ↓                                                     │
│  SSD              →  персистентный кэш                      │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| max_hot_tokens | 8192 | Макс. токенов в hot window |
| device | cuda | Устройство по умолчанию |
| max_context_length | 32768 | Макс. длина контекста |

---

## 7. Интеграция с CoreBrain

### 7.1 brain_config.json секция reasoning

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

### 7.2 Интеграционные точки

1. **CoreBrain.process_query()** → вызов SelfReasoningEngine
2. **Qwen singleton** → передаётся в SelfReasoningEngine
3. **Knowledge Graph** → расширение типами REASONING
4. **Fractal Storage** → создаётся при инициализации brain

---

## 8. Ожидаемые Результаты

| # | Результат | Описание |
|---|-----------|----------|
| 1 | Фрактальное хранилище работает | Иерархическая структура с адресацией |
| 2 | Self-Reasoning Engine работает | Цикл "вопрос-анализ-уточнение" |
| 3 | Контекстные вопросы | Связаны с запросом, не рандомные |
| 4 | Хранение рассуждений | Вся цепочка в fractal storage |
| 5 | Никаких fallback | Только Qwen + аналитика |
| 6 | Confidence-based termination | Остановка при уверенности ≥0.75 |

---

## 9. Критерии Успеха

- ✅ Система задаёт вопросы при низкой уверенности
- ✅ Вопросы контекстно связаны с запросом
- ✅ Единый экземпляр Qwen (singleton)
- ✅ Все рассуждения сохраняются
- ✅ Нет случайных/бессмысленных ответов
- ✅ Фрактальная структура масштабируема

---

## 10. История Изменений

| Версия | Дата | Описание |
|--------|------|----------|
| 1.0 | 2026-03-23 | Начальная версия плана |
| 1.1 | 2026-03-23 | Реализация: fractal_base.py, confidence_scorer.py, clarification_generator.py, reasoning_types.py, self_reasoning_engine.py, integration.py |
| 1.2 | 2026-03-25 | AI Agent исправления: Tokenizer, GUI, FractalStorage, Config imports |
| 1.3 | 2026-03-25 | max_new_tokens/max_length фиксы, pass statement исправления |
| 1.4 | 2026-03-25 | Config alignment, device management, hardcoded values |
| 1.5 | 2026-03-26 | Qwen-only модель (убраны RUGPT3 ссылки), обновлён CLAUDE.md |

---

## 11. Текущий Статус Реализации

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

### Интеграция с CoreBrain:
- [x] Добавлена в brain_config.json секция "reasoning"
- [x] Вызов ReasoningIntegration в ComponentInitializer
- [x] 28 компонентов инициализируются корректно

### Тестирование:
- [x] python -m cogniflex.run - успешный запуск
- [x] GUI (1280x800) - запускается
- [x] Все 28 компонентов - инициализированы

---

## 12. Git Структура

- **Main branch**: `C:/Users/black/OneDrive/Desktop/CogniFlex` - активная разработка
- **Worktrees**: `C:/Users/black/.windsurf/worktrees/CogniFlex/` - старые снимки, не используются