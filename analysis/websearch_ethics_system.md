# Анализ WebSearch & Ethics System EVA

## Содержание

1. [WebSearch System](#websearch-system)
   - [1.1. Архитектура](#11-архитектура)
   - [1.2. Ключевые классы и компоненты](#12-ключевые-классы-и-компоненты)
   - [1.3. Интеграция с EVA](#13-интеграция-с-eva)
   - [1.4. Методы и их назначение](#14-методы-и-их-назначение)
   - [1.5. Проблемы и заглушки](#15-проблемы-и-заглушки)

2. [Ethics System](#ethics-system)
   - [2.1. Архитектура](#21-архитектура)
   - [2.2. Ключевые классы и компоненты](#22-ключевые-классы-и-компоненты)
   - [2.3. Интеграция с EVA](#23-интеграция-с-eva)
   - [2.4. Методы и их назначение](#24-методы-и-их-назначение)
   - [2.5. Проблемы и заглушки](#25-проблемы-и-заглушки)

---

# WebSearch System

## 1.1. Архитектура

Система веб-поиска EVA состоит из трёх основных компонентов, образующих многоуровневую архитектуру:

`
WebSearch System Architecture
├── WebSearchEngine (web_search_engine.py)
│   ├── SQLite Database (history, stats)
│   ├── ThreadPoolExecutor (concurrent requests)
│   ├── CacheManager (result caching)
│   └── FractalGraph v2 Integration
│
├── IntegratedWebSearchEngine (web_search_integrated.py)
│   ├── BaseComponent (EventBus integration)
│   ├── AsyncWebSearchClient (aiohttp, connection pooling)
│   ├── Tavily API integration
│   └── Singleton pattern
│
└── SearchEngines (search_engines.py)
    ├── DuckDuckGo HTML
    ├── Searx Metasearch
    ├── Brave Search
    └── Wikipedia API
`

### Уровни системы:

**Уровень 1 (Базовый):** SearchEngines - низкоуровневые адаптеры для различных поисковых систем с fallback-механизмами.

**Уровень 2 (Основной):** WebSearchEngine - основной движок с кэшированием, статистикой и интеграцией с FractalGraph.

**Уровень 3 (Интегрированный):** IntegratedWebSearchEngine - компонент с поддержкой BaseComponent и EventBus, асинхронными операциями.

---

## 1.2. Ключевые классы и компоненты

### 1.2.1. WebSearchEngine (web_search_engine.py, 697 строк)

**Основные характеристики:**

- Синхронный поисковый движок с поддержкой SQLite
- Фоновые потоки для обработки задач и очистки кэша
- Интеграция с FractalGraph v2

**Конструктор:**
`python
def __init__(self, brain=None, cache_dir: Optional[str] = None)
`

**Ключевые атрибуты:**

- search_settings - настройки поиска (max_results=10, timeout=15s, cache_ttl=86400s)
- ctive_search_engines - активные движки: tavily, wikipedia
- _db_manager - менеджер базы данных
- _cache_manager - менеджер кэша
- search_queue - очередь асинхронных задач

### 1.2.2. IntegratedWebSearchEngine (web_search_integrated.py, 1101 строка)

**Основные характеристики:**

- Наследует BaseComponent (EventBus integration)
- Singleton паттерн
- Асинхронный поиск через aiohttp с connection pooling

**Ключевые компоненты:**

- AsyncWebSearchClient - асинхронный клиент с пулом соединений
- 	avily_search() - синхронная функция Tavily API
- 	avily_search_async() - асинхронная функция Tavily API

### 1.2.3. SearchEngines (search_engines.py, 532 строки)

**Основные характеристики:**

- Адаптеры для DuckDuckGo, Searx, Brave, Wikipedia
- Ротация User-Agent для обхода блокировок
- Локальная база знаний как fallback

**User-Agent ротация:**
`python
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...',
    # Всего 7 различных User-Agent
]
`

---

## 1.3. Интеграция с EVA

### 1.3.1. Интеграция с FractalGraph v2

**Метод:** dd_to_fractal_graph(knowledge: List[Dict])

`python
# Добавление результатов веб-поиска в граф
node = fractal_graph.add_node(
    content=item.get(\"content\", \"\"),
    node_type=\"web_knowledge\",
    level=2,
    confidence=item.get(\"relevance\", 0.5),
    metadata={
        \"source\": item.get(\"source\", \"web\"),
        \"url\": item.get(\"metadata\", {}).get(\"url\"),
        \"domain\": item.get(\"domain\", \"general\"),
        \"concept\": item.get(\"concept\", \"\")
    }
)
`

### 1.3.2. Интеграция с EventBus

**События в IntegratedWebSearchEngine:**

- web_search_engine.initialized - при инициализации
- web_search_engine.started - при запуске
- web_search_engine.stopped - при остановке
- web_search_engine.cache_hit - при использовании кэша
- web_search_engine.search_performed - после выполнения поиска

### 1.3.3. Интеграция с Concept System

**Метод:** web_search_and_learn(concept: str, num_results: int = 3)

Преобразует результаты поиска в формат знаний для концептов:
`python
{
    \"concept\": str,
    \"content\": str,          # сниппет
    \"domain\": \"general\",
    \"source\": \"web:engine\",
    \"relevance\": float,
    \"metadata\": {
        \"url\": str,
        \"engine\": str,
        \"timestamp\": float
    }
}
`

---

## 1.4. Методы и их назначение

### 1.4.1. WebSearchEngine Методы

| Метод | Назначение | Параметры | Возвращаемое значение |
|-------|------------|-----------|----------------------|
| search() | Синхронный поиск с кэшированием | query, max_results, use_cache | Dict с результатами |
| search_async() | Асинхронный поиск | query, max_results | task_id |
| _perform_search() | Выполнение поиска через движки | query, max_results | List[SearchResult] |
| web_search_and_learn() | Поиск + конвертация в знания | concept, 
um_results | List[Dict] |
| dd_to_fractal_graph() | Добавление в FGv2 | knowledge | Dict с added_count, node_ids |
| search_and_add_to_graph() | Поиск + добавление в граф | concept, 
um_results | Dict |
| get_stats() | Получение статистики | - | Dict |
| get_recent_queries() | История запросов | limit | List[Dict] |
| clear_cache() | Очистка кэша | - | - |

### 1.4.2. IntegratedWebSearchEngine Методы

| Метод | Назначение | Параметры | Возвращаемое значение |
|-------|------------|-----------|----------------------|
| search() | Синхронный поиск Tavily | query, search_config, max_results | Dict с результатами |
| search_async() | Асинхронный поиск Tavily | query, search_config, max_results | Dict |
| search_batch_async() | Пакетный поиск | queries, search_config, max_results | List[Dict] |
| search_with_filters() | Поиск с фильтрами | query, ilters | Dict |
| enrich_with_context() | Обогащение контекста | query, esponse, max_results | Dict |
| ormat_enrichment_prompt() | Форматирование для Qwen | enrichment_result | str |
| get_search_suggestions() | Автодополнение | partial_query | List[str] |
| get_search_statistics() | Статистика | - | Dict |

### 1.4.3. SearchEngines Методы

| Метод | Назначение | Параметры | Возвращаемое значение |
|-------|------------|-----------|----------------------|
| search_google() | Поиск через Google (fallback DDG) | query, max_results | List[SearchResult] |
| search_yandex() | Поиск через Yandex (fallback Brave) | query, max_results | List[SearchResult] |
| search_bing() | Поиск через Bing | query, max_results | List[SearchResult] |
| search_duckduckgo() | Поиск через DuckDuckGo | query, max_results | List[SearchResult] |
| search_wikipedia() | Поиск через Wikipedia API | query, max_results | List[SearchResult] |
| search_searx() | Поиск через Searx | query, max_results | List[SearchResult] |
| _create_local_results() | Fallback локальные результаты | query, max_results | List[SearchResult] |

---

## 1.5. Проблемы и заглушки

### 1.5.1. Заглушки (Stubs)

**Проблема 1: Заглушка при недоступности Tavily API**

В web_search_integrated.py строки 527-550 используется fallback с симулированными результатами:

`python
# Fallback: симулированные результаты при ошибке API
logger.warning(f\"Tavily API недоступен, используем fallback...\")
results = []
for i in range(min(max_results, 5)):
    result = {
        \"title\": f\"Результат поиска #{i+1} для: {query}\",
        \"url\": f\"https://example.com/result{i+1}\",
        \"snippet\": f\"Это фрагмент текста о {query}...\",
        ...
    }
    results.append(result)
`

**Статус:** ЗАГЛУШКА - не выполняет реальный поиск

**Проблема 2: Локальная база знаний как fallback**

В search_engines.py строки 272-371 _create_local_results() возвращает статические данные:

`python
knowledge_base = {
    'python': [...],           # 2 статических результата
    'машинное обучение': [...], # 2 статических результата
    'искусственный интеллект': [...], # 2 статических результата
    'нейронные сети': [...],
    'глубокое обучение': [...]
}
`

**Статус:** ЗАГЛУШКА - ограниченная тематика, статический контент

### 1.5.2. Технические проблемы

**Проблема 3: Зависимость от внешнего API Tavily**

- Нет fallback на другие движки (DuckDuckGo, Searx)
- При отсутствии API ключа - полная неработоспособность

**Проблема 4: Кэширование в двух местах**

- WebSearchEngine использует _cache_manager
- IntegratedWebSearchEngine использует собственный search_cache
- Дублирование функциональности

**Проблема 5: Нет повторных попыток (retry)**

- При таймауте запроса - мгновенный fail
- Нет exponential backoff

---

# Ethics System

## 2.1. Архитектура

Система этики EVA построена на принципе множественного наследования через Mixins:

`
Ethics System Architecture
├── EthicsFramework (framework_core.py)
│   └── Наследует:
│       ├── EthicsPrinciplesMixin
│       │   ├── _load_configuration()
│       │   ├── add_ethical_principle()
│       │   └── _init_default_principles()
│       │
│       ├── EthicsChecksMixin
│       │   ├── analyze_request()
│       │   ├── analyze_response()
│       │   └── _evaluate_principle() для 7 категорий
│       │
│       └── EthicsViolationsMixin
│           ├── get_violation_history()
│           ├── resolve_violation()
│           └── _analyze_ethical_trends()
│
├── SituationsEvaluationMixin (situations_evaluation.py)
│   ├── get_situation_dashboard_data()
│   └── generate_situation_visualization()
│
└── PrinciplesManager (principles_manager.py)
    ├── SQLite database
    ├── Dashboard data
    └── Visualization generation
`

### Архитектура этических категорий:

Система оценивает запросы по 7 категориям принципов:

1. **privacy** - приватность и конфиденциальность
2. **safety** - безопасность (насилие, оружие, наркотики)
3. **fairness** - справедливость (дискриминация)
4. **transparency** - прозрачность
5. **autonomy** - автономия
6. **beneficence** - благотворительность/польза
7. **accountability** - подотчетность

---

## 2.2. Ключевые классы и компоненты

### 2.2.1. EthicsFramework (framework_core.py, 177 строк)

**Основные характеристики:**

- Главный класс системы этики
- Наследует 3 миксина для функциональности
- Фоновые потоки для мониторинга

**Жизненный цикл:**
`python
def start(self):
    self.running = True
    # Запускает мониторинг нарушений

def stop(self):
    self.running = False
    # Останавливает фоновые процессы
`

**Фоновые службы:**
- _violation_monitor_thread - проверяет старые нарушения каждые 60 сек
- _principle_check_thread - обновляет статистику принципов каждый час

### 2.2.2. EthicalPrinciple (framework_principles.py)

**Датакласс для этического принципа:**

`python
@dataclass
class EthicalPrinciple:
    name: str           # Идентификатор (no_violence, honesty)
    description: str    # Описание
    phrase: str         # Краткая фраза
    weight: float       # Вес (1.0-1.5)
    threshold: float    # Порог срабатывания (0.6-0.8)
    category: str       # Категория (safety, integrity, accuracy...)
    priority: int       # Приоритет (8-10)
    last_updated: float # Время последнего обновления
    active: bool        # Активен ли принцип
`

**Стандартные принципы по умолчанию:**

| Принцип | Фраза | Вес | Порог | Категория |
|---------|-------|-----|-------|-----------|
| no_violence | Без насилия | 1.5 | 0.6 | safety |
| honesty | Честность | 1.2 | 0.7 | integrity |
| fact_verification | Проверка фактов | 1.2 | 0.7 | accuracy |
| safe_code | Безопасный код | 1.5 | 0.6 | security |
| risk_blocking | Блокировка рисков | 1.3 | 0.65 | safety |
| output_control | Контроль вывода | 1.0 | 0.75 | quality |

### 2.2.3. EthicalAssessment (framework_checks.py)

**Датакласс для результата оценки:**

`python
@dataclass
class EthicalAssessment:
    principle_scores: Dict[str, float]  # Оценки по принципам
    violations: List[Dict[str, Any]]     # Нарушения
    recommendations: List[str]           # Рекомендации
    confidence: float                    # Уверенность (0-1)
    timestamp: float                     # Время оценки
    principle_name: Optional[str]        # Имя принципа (для risk_assessment)
    score: float                         # Оценка
    explanation: Optional[str]           # Объяснение
    context: Optional[Dict]              # Контекст
    violation_detected: bool             # Обнаружено нарушение
    severity: str                       # severity: low/medium/high
`

### 2.2.4. EthicalDecision (framework_violations.py)

**Датакласс для решения по этике:**

`python
@dataclass
class EthicalDecision:
    approved: bool              # Одобрено или нет
    principle: str             # Принцип
    severity: float            # Серьезность (0-1)
    description: str           # Описание
    context: Dict[str, Any]    # Контекст
    timestamp: float           # Время
    resolved: bool             # Разрешено ли
    resolution: Optional[str]  # Решение
    resolution_timestamp: Optional[float]
    source: str                # Источник (system/user)
    violation_id: str          # Уникальный ID (автогенерация)
`

### 2.2.5. PrinciplesManager (principles_manager.py, 669 строк)

**Основные характеристики:**

- SQLite-база данных для хранения принципов
- История изменений и оценок
- Дашборд данные и визуализация

**Таблицы БД:**

1. principles - сами принципы
2. principles_history - история изменений
3. principles_assessments - оценки применения принципов

---

## 2.3. Интеграция с EVA

### 2.3.1. Интеграция с EventBus

**Публикуемые события:**

`python
# При высокой серьёзности (> 0.9)
event_type = \"ethics.violation\"
priority = EventPriority.HIGH

# При низкой серьёзности
event_type = \"ethics.warning\"
priority = EventPriority.NORMAL
`

**Данные события:**
`python
{
    'principle': violation.principle,
    'severity': violation.severity,
    'description': violation.description,
    'context': context
}
`

### 2.3.2. Интеграция с генерацией ответов

**Метод:** generate_regeneration_prompt()

Генерирует промпт для перегенерации ответа при этических нарушениях:

`python
prompt = 'Обнаружены этические нарушения в ответе:\n'
prompt += \"КРИТИЧЕСКИЕ НАРУШЕНИЯ:\n\"
prompt += \"1. [no_violence] Описание...\n\"
prompt += \"\nПРЕДУПРЕЖДЕНИЯ:\n\"
prompt += \"1. [honesty] Описание...\n\"
prompt += \"\nПереформулируй ответ, устранив нарушения.\"
`

---

## 2.4. Методы и их назначение

### 2.4.1. EthicsChecksMixin Методы

| Метод | Назначение | Параметры | Возвращаемое значение |
|-------|------------|-----------|----------------------|
| nalyze_request() | Анализ запроса пользователя | equest, context | Dict с approved, violations, recommendations |
| nalyze_response() | Анализ ответа модели | query, esponse | Dict |
| nalyze_content() | Универсальный анализ | content, context | EthicsAnalysisResult |
| check_with_context() | Проверка с контекстом | 	ext, query, context | Dict с violation_count, overall_score |
| generate_regeneration_prompt() | Генерация промпта для перегенерации | ethics_result, query, esponse | str |

### 2.4.2. Методы оценки принципов

| Метод | Назначение |
|-------|------------|
| _evaluate_privacy() | Оценка приватности (имя, адрес, телефон, документы) |
| _evaluate_safety() | Оценка безопасности (насилие, оружие, наркотики, взлом) |
| _evaluate_fairness() | Оценка справедливости (расизм, сексизм, дискриминация) |
| _evaluate_transparency() | Оценка прозрачности (секреты, обман, манипуляции) |
| _evaluate_autonomy() | Оценка автономии (контроль мыслей, манипуляции выбором) |
| _evaluate_beneficence() | Оценка пользы (вред пользователю, дезинформация) |
| _evaluate_accountability() | Оценка подотчетности (анонимность, уход от ответственности) |

### 2.4.3. EthicsViolationsMixin Методы

| Метод | Назначение | Параметры | Возвращаемое значение |
|-------|------------|-----------|----------------------|
| get_violation_history() | История нарушений | limit, principle | List[Dict] |
| esolve_violation() | Разрешение нарушения | iolation_id, esolution | bool |
| get_active_violations() | Активные нарушения | - | List[Dict] |
| get_ethics_statistics() | Статистика | - | Dict |
| export_ethics_data() | Экспорт данных | ile_path | bool |
| import_ethics_data() | Импорт данных | ile_path | bool |
| generate_ethics_report() | Генерация отчета | - | Dict |

### 2.4.4. SituationsEvaluationMixin Методы

| Метод | Назначение | Параметры | Возвращаемое значение |
|-------|------------|-----------|----------------------|
| get_situation_dashboard_data() | Данные для дашборда | - | Dict |
| generate_situation_visualization() | Визуализация (matplotlib) | iew_type | base64 строка |
| get_system_health() | Оценка здоровья системы | - | Dict |

### 2.4.5. PrinciplesManager Методы

| Метод | Назначение | Параметры | Возвращаемое значение |
|-------|------------|-----------|----------------------|
| dd_principle() | Добавление принципа | principle, user_id | principle_id |
| update_principle() | Обновление принципа | principle_id, updates | bool |
| get_principle() | Получение принципа | principle_id | EthicalPrinciple |
| get_principle_by_name() | Получение по имени | 
ame | EthicalPrinciple |
| get_principles_by_category() | Поиск по категории | category | List[Tuple] |
| ecord_assessment() | Запись оценки | principle_id, score, confidence, context | - |
| get_assessment_history() | История оценок | principle_id, days | List[Dict] |
| get_principles_dashboard_data() | Дашборд данные | - | Dict |
| generate_ethical_visualization() | Визуализация | iew_type | base64 строка |

---

## 2.5. Проблемы и заглушки

### 2.5.1. Технические проблемы

**Проблема 1: Дублирование метода close()**

В principles_manager.py строки 659-663 и 665-669:

`python
def close(self):
    \"\"\"Закрывает менеджер принципов и освобождает ресурсы.\"\"\"
    logger.info(\"Закрытие менеджера этических принципов...\")
    # Здесь можно добавить дополнительные действия при закрытии
    logger.info(\"Менеджер этических принципов закрыт\")

def close(self):  # ДУБЛИКАТ!
    \"\"\"Закрывает менеджер принципов и освобождает ресурсы.\"\"\"
    logger.info(\"Закрытие менеджера этических принципов...\")
    logger.info(\"Менеджер этических принципов закрыт\")
`

**Статус:** БАГ - дублирование метода, второй переопределит первый

**Проблема 2: Зависимость от matplotlib**

В situations_evaluation.py и principles_manager.py:

`python
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ...
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
`

**Влияние:**
- generate_situation_visualization() возвращает пустую строку
- generate_ethical_visualization() возвращает пустую строку

**Статус:** ЧАСТИЧНАЯ ЗАГЛУШКА - метод работает, но без визуализации

**Проблема 3: Mixin без инициализации**

В situations_evaluation.py Mixin использует атрибуты, которые не инициализируются:

`python
# Используется в get_situation_dashboard_data():
recent_solutions = self.solutions_cache.get(\"solutions\", [])  # НЕ инициализирован!
open_issues = [i for i in self.ethical_issues if not i.resolved]  # НЕ инициализирован!

# Используется в export_ethics_data():
self.solutions_cache.get(\"solutions\", [])  # НЕ инициализирован!
self.review_cache.get(\"reviews\", [])       # НЕ инициализирован!
self.ethical_issues                         # НЕ инициализирован!
self.principles_manager                     # НЕ инициализирован!
`

**Статус:** БАГ - Mixin требует инициализации от реализующего класса

### 2.5.2. Архитектурные ограничения

**Проблема 4: Простая ключевая оценка**

Методы _evaluate_privacy(), _evaluate_safety() используют простой keyword matching:

`python
# _evaluate_safety()
dangerous_keywords = [
    \"убить\", \"навредить\", \"повредить\", \"опасно\", \"опасность\", \"вред\", \"взрыв\",
    \"оружие\", \"наркотик\", \"наркотики\", \"взлом\", \"взломать\"
]
for keyword in dangerous_keywords:
    if keyword in request.lower():
        score += 0.2
`

**Ограничения:**
- Легко обойти (синонимы, замена букв)
- Нет понимания контекста
- Нет NLI/классификации

**Статус:** БАЗОВАЯ РЕАЛИЗАЦИЯ - требует улучшения (возможно использование NLI из Concepts System)

**Проблема 5: Нетpersistence для всех данных**

- PrinciplesManager сохраняет в SQLite
- EthicsFramework сохраняет в JSON файлы
- Два разных подхода к персистентности

**Проблема 6: Проверка только запросов**

Текущая реализация проверяет:
- nalyze_request() - входящие запросы
- nalyze_response() - исходящие ответы

Но нет проверки:
- Контекста (предыдущие сообщения)
- Внешних источников (веб-поиск)
- Knowledge Graph

---

# Сравнение систем

## WebSearch System

| Аспект | Оценка | Комментарий |
|--------|--------|-------------|
| Архитектура | Хорошо | Три уровня, четкое разделение |
| Async поддержка | Хорошо | aiohttp с connection pooling |
| Кэширование | Хорошно | SQLite + JSON |
| Fallback | Плохо | Заглушки вместо реального поиска |
| Интеграция FGv2 | Хорошо | Метод add_to_fractal_graph() |
| EventBus | Хорошо | Интегрирован |

## Ethics System

| Аспект | Оценка | Комментарий |
|--------|--------|-------------|
| Архитектура | Хорошо | Mixins + Composition |
| Принципы | Хорошо | 6 стандартных принципов |
| Оценка | Средне | Keyword-based, легко обойти |
| Визуализация | Средне | Зависит от matplotlib |
| Персистентность | Средне | Два подхода (SQLite + JSON) |
| Интеграция | Хорошо | EventBus + BaseComponent |

---

# Рекомендации по улучшению

## WebSearch

1. **Добавить fallback на DuckDuckGo/Searx** - не полагаться только на Tavily
2. **Реализовать retry с exponential backoff** - для надежности
3. **Унифицировать кэширование** - один менеджер для всех компонентов

## Ethics

1. **Устранить дублирование close()** - удалить дубликат
2. **Добавить инициализацию атрибутов** - в Mixin или конструктор
3. **Улучшить оценку принципов** - использовать NLI из Concepts System
4. **Унифицировать персистентность** - выбрать SQLite или JSON

---

*Дата анализа: 2026-04-27*
*Система: EVA AI*
