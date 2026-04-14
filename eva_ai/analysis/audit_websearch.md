# WebSearch System Audit

## 1. Overview

Система WebSearch в EVA AI представляет собой многоуровневую архитектуру для выполнения веб-поиска с поддержкой кэширования, событийной шины и фоновых процессов.

### Структура файлов:
```
eva_ai/websearch/
├── __init__.py                    # Экспорты модулей
├── web_search_engine.py          # Основной движок (707 строк)
├── web_search_integrated.py      # Интегрированный движок с EventBus (1101 строка)
├── search_engines.py             # Интеграция с поисковиками (522 строки)
├── database_manager.py           # SQLite история поиска (220 строк)
├── cache_manager.py              # JSON кэширование (74 строки)
├── search_models.py              # Модели SearchResult/SearchQuery (26 строк)
├── search_types.py               # Enums и типы (67 строк)
└── eva_web_search_cache/        # Директория кэша
    ├── search_history.db         # SQLite база
    └── search_cache.json         # JSON кэш
```

### Архитектура:
1. **WebSearchEngine** - базовый движок с SQLite и JSON кэшем
2. **IntegratedWebSearchEngine** - обёртка с BaseComponent и EventBus
3. **SearchEngines** - абстракция над поисковиками (Google, Yandex, Bing, Wikipedia)
4. **AsyncWebSearchClient** - асинхронный клиент с connection pooling

---

## 2. Search Engines Supported

### 2.1 Основные поисковые системы:

| Поисковик | Поддержка | Метод | Статус |
|-----------|-----------|-------|--------|
| **Google** | Да | duckduckgo_html (заглушка) | Зависит от внешнего сервиса |
| **Yandex** | Да | brave_search (заглушка) | Зависит от внешнего сервиса |
| **Bing** | Да | HTML parsing | Работает через scrape |
| **Wikipedia** | Да | Wikipedia API | Работает стабильно |
| **DuckDuckGo** | Да | HTML версия | Работает |
| **Brave Search** | Да | HTML parsing | Работает |
| **Searx** | Да | JSON API | Fallback система |
| **Tavily API** | Да | REST API | **Основной движок** |

### 2.2 Детальный анализ реализации:

#### **search_engines.py** - Класс SearchEngines:

```python
# Методы поиска:
- search_google()      # -> _search_duckduckgo_html() [ЗАГЛУШКА]
- search_yandex()      # -> _search_brave() [ЗАГЛУШКА]
- search_bing()        # Реальный HTML parsing
- search_wikipedia()   # Wikipedia API (en + ru)
- search_duckduckgo()  # Многоуровневый fallback
- search_searx()       # Searx метапоисковик
```

#### **Проблемы с поисковиками:**

1. **Google** - реально использует DuckDuckGo HTML:
   ```python
   def search_google(self, query: str, max_results: int):
       return self._search_duckduckgo_html(query, max_results)  # НЕ Google!
   ```

2. **Yandex** - реально использует Brave Search:
   ```python
   def search_yandex(self, query: str, max_results: int):
       return self._search_brave(query, max_results)  # НЕ Yandex!
   ```

3. **Bing** - единственный реальный scraping:
   ```python
   # Парсит li.b_algo из HTML ответа bing.com
   ```

4. **DuckDuckGo/Searx/Brave** - работают через HTML parsing

5. **Wikipedia** - стабильный API:
   ```python
   # Использует en.wikipedia.org/w/api.php и ru.wikipedia.org/w/api.php
   # Правильный User-Agent: '"ЕВАAI/1.0 ..."'
   ```

### 2.3 Tavily API - Основной движок:

**Интегрированный поисковый движок использует Tavily как primary:**

```python
# web_search_integrated.py
def tavily_search(query: str, api_key: str = None, max_results: int = 5):
    # Загружает ключ из brain_config.json или env TAVILY_API_KEY
    # POST https://api.tavily.com/search
    # Returns: {"results": [...], "error": "..."}
```

**API Endpoints:**
- Single: `POST https://api.tavily.com/search`
- Async client: `AsyncWebSearchClient` с `aiohttp.TCPConnector`

**Требует:**
- Tavily API Key в `brain_config.json` или `TAVILY_API_KEY` env var
- Формат: `{"query": "...", "max_results": N}`

---

## 3. Integration Analysis

### 3.1 Инициализация в системе:

**Файл:** `eva_ai/core/init_factories.py` (строка 435)

```python
def create_web_search_engine(initializer):
    from eva_ai.websearch.web_search_integrated import IntegratedWebSearchEngine
    web_search = IntegratedWebSearchEngine(brain=initializer.core_brain)
    # Инициализирует как BaseComponent
    # Подписывается на EventBus
    return web_search
```

### 3.2 Использование в brain_query.py:

**Функция определения необходимости поиска:**
```python
def needs_web_search(query: str) -> tuple[bool, str]:
    # Приветствия -> False
    # Запросы о себе -> False
    # Математика/код -> False
    # Текущие события (2025, 2026, новости) -> True
    # Технические запросы -> True
```

**Точки вызова в process_query():**
- Строка 348: `web_search.search(query, max_results=5)`
- Строка 394: Фиксирует `web_search_info` в результатах

### 3.3 EventBus события:

**IntegratedWebSearchEngine публикует:**
- `web_search_engine.initialized` - при инициализации
- `web_search_engine.started` - при запуске
- `web_search_engine.stopped` - при остановке
- `web_search_engine.cache_hit` - при кэш-хите
- `web_search_engine.search_performed` - после поиска

**Подписки:**
- Не зарегистрированы явные подписки на внешние события

### 3.4 Интеграция с компонентами:

| Компонент | Использование WebSearch |
|-----------|------------------------|
| `brain_query.py` | Основной потребитель - поиск для ответов |
| `contradiction_manager.py` | verify_fact_with_web_search() |
| `contradiction_miner.py` | enable_web_search_for_resolution |
| `unified_generator.py` | _get_web_search_context() |
| `server_routes_chat.py` | Поиск в GUI чате |
| `WebSearchLearningIntegration` | Интеграция с обучением |

---

## 4. Problems and Issues

### 4.1 Критические проблемы:

#### **Проблема 1: Подмена поисковиков**
```python
# search_engines.py, строки 59-67
def search_google(self, query: str, max_results: int):
    return self._search_duckduckgo_html(query, max_results)  # Это НЕ Google!

def search_yandex(self, query: str, max_results: int):
    return self._search_brave(query, max_results)  # Это НЕ Yandex!
```
**Последствие:** При выборе "google" или "yandex" в `active_search_engines`, используются совершенно другие движки.

#### **Проблема 2: Tavily API как единственный надёжный источник**
```python
# web_search_integrated.py, строка 464
# БЕЗ FALLBACK - только Tavily
result = {"status": "error", "error": tavily_result.get('error', 'Tavily failed'), "results": []}
```
**Проблема:** Если Tavily API key отсутствует или API недоступен - поиск не работает.

#### **Проблема 3: Отсутствие реального Google/Yandex API**
- Нет API ключей для Google Custom Search
- Нет API ключей для Yandex
- Bing/Google/Yandex работают через HTML scraping, что:
  - Медленно
  - Может быть заблокировано
  - Нарушает ToS поисковиков

### 4.2 Средние проблемы:

#### **Проблема 4: Дублирование моделей**
```python
# search_models.py - один формат
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str  # строка!
    
# search_types.py - ДРУГОЙ формат  
@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    source: SearchEngine  # Enum!
```
**Проблема:** Две разные модели SearchResult с разными полями и типами.

#### **Проблема 5: Сложный fallback в DuckDuckGo**
```python
# search_engines.py, строки 124-143
search_methods = [
    self._search_duckduckgo_html,
    self._search_searx,
    self._search_brave
]
# Пытается последовательно, если все fail - локальные результаты
```
**Проблема:** Многоуровневый fallback создаёт неконсистентность результатов.

#### **Проблема 6: Очередь поиска не используется**
```python
# web_search_engine.py
self.search_queue = queue.Queue()  # Создаётся
self.search_thread = threading.Thread(...)  # Запускается
# НО: метод search() не кладёт задачи в очередь!
# search_async() кладёт, но возвращает task_id, который нигде не отслеживается
```

### 4.3 Мелкие проблемы:

1. **Кэш-директории:**
   - `eva_web_search_cache/` - для WebSearchEngine
   - `cogniflex_web_search_cache/` - создаётся, но не используется

2. **Статистика:**
   - `_update_query_stats()` вызывается только в `__init__`
   - stats в памяти не синхронизируются с DB постоянно

3. **Searx instances:**
   ```python
   self.searx_instances = [
       "https://searx.be/search",
       "https://searx.org/search",  # Домен продан!
       "https://search.bus-sc.com/search",
       "https://searx.fmac.xyz/search"
   ]
   ```
   searx.org может быть нерабочим.

### 4.4 Заглушки (Stubs):

#### **local_knowledge база:**
```python
# search_engines.py, _create_local_results()
# Жёстко заданные темы: python, машинное обучение, ии, нейронные сети, глубокое обучение
```
**Использование:** Только как последний fallback.

#### **simulated_search fallback:**
```python
# web_search_integrated.py, _basic_web_search()
for i in range(min(max_results, 5)):
    result = {
        "title": f"Результат поиска #{i+1} для: {query}",
        "url": f"https://example.com/result{i+1}",
        # ...
        "source": "simulated_search"
    }
```
**Проблема:** Генерирует фейковые результаты когда Tavily падает.

---

## 5. Overall Assessment

### 5.1 Архитектура: 6/10

**Плюсы:**
- Многоуровневая система с кэшированием
- EventBus интеграция
- Async клиент для производительности
- SQLite + JSON dual caching

**Минусы:**
- Дублирование SearchResult моделей
- Неиспользуемая очередь поиска
- Сложный fallback мешает отладке

### 5.2 Поисковики: 4/10

**Работающие:**
- Tavily API (если есть ключ) - **ОСНОВНОЙ**
- Wikipedia API - стабильно
- Bing HTML parsing - работает частично

**Не работающие как заявлено:**
- Google -> DuckDuckGo
- Yandex -> Brave
- Searx -> ненадёжно

### 5.3 Кэширование: 7/10

- TTL-based cache (24 часа)
- SQLite для истории
- JSON для быстрого доступа
- Размер лимитирован (1000 записей в памяти)

### 5.4 Интеграция: 6/10

- Хорошо интегрирован в brain_query
- EventBus события публикуются
- Но подписки на внешние события отсутствуют

### 5.5 Рекомендации:

1. **Критично:** Добавить fallback на Wikipedia/Searx если Tavily недоступен
2. **Критично:** Исправить подмену Google->DuckDuckGo, Yandex->Brave
3. **Важно:** Убрать дублирование SearchResult моделей
4. **Важно:** Удалить неиспользуемую очередь search_queue или документировать её purpose
5. **Желательно:** Добавить реальные API ключи для Google/Yandex/Bing

### 5.6 Итоговая оценка: **5/10**

Система имеет хорошую архитектуру но страдает от:
1. Отсутствия реальных API для топовых поисковиков
2. Зависимости от одного сервиса (Tavily)
3. Подмены поисковиков (маркетинговое обещание vs реальность)
4. Фейковых fallback результатов

**Для production рекомендуется:** Либо купить Tavily API key, либо реализовать реальные API интеграции.

