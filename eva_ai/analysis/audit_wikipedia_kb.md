# Отчёт: Wikipedia KB

## 1. Импорты

### Стандартные библиотеки:
- os - пути к файлам
- json - сериализация эмбеддингов
- sqlite3 - хранение статей и чанков
- hashlib - генерация ID статей (SHA256)
- logging - логирование
- 	hreading - блокировки (RLock)
- 	yping - типы
- datetime - временные метки

### Опциональные зависимости:
`python
try:
    import faiss        # Быстрый векторный поиск
    import numpy as np  # Массивы эмбеддингов
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS не установлен, используется fallback поиск")
`

### Внешние зависимости:
`python
from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer
# Модель: 'intfloat/multilingual-e5-base', устройство: 'cpu'
`

---

## 2. Поиск и API

### WikipediaKnowledgeBase

#### Методы:

| Метод | Описание |
|-------|----------|
| search(query, limit=5, min_similarity=0.3) | Семантический поиск по статьям |
| dd_article(title, text, url, category, chunk_size=500) | Добавить статью |
| get_article(article_id) | Получить статью по ID |
| get_stats() | Статистика БД |
| clear() | Очистить базу |
| dd_to_fractal_graph(fractal_graph, top_k=100, node_type='wikipedia') | Экспорт всех статей в FGv2 |
| search_and_add_to_graph(query, fractal_graph, top_k=10) | Поиск + добавление в FGv2 |

#### Структура SQLite:

`sql
-- articles: полные статьи
CREATE TABLE articles (
    id TEXT PRIMARY KEY,        -- SHA256 от title (первые 16 символов)
    title TEXT NOT NULL,
    url TEXT,
    text TEXT,                   -- Ограничено 10000 символами!
    chunk_count INTEGER,
    created_at TEXT,
    category TEXT
)

-- chunks: разбитые на чанки статьи с эмбеддингами
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,         -- {article_id}_{index}
    article_id TEXT,
    title TEXT,
    text TEXT NOT NULL,
    embedding TEXT,              -- JSON список float
    chunk_index INTEGER,
    created_at TEXT
)
`

#### Алгоритм поиска:

1. **FAISS (быстрый)**: IndexIVFFlat или IndexFlatIP для больших баз
2. **Fallback (медленный)**: косинусное сходство через dot product

`python
def _compute_embedding(text):
    # intfloat/multilingual-e5-base на CPU
    emb = embedder.encode([text.strip()])[0]
    return emb.tolist()
`

#### WikipediaLoader

`python
class WikipediaLoader:
    def search_articles(query, limit=10)   # Wikipedia API search
    def load_article(title, chunk_size=500) # Загрузка статьи по названию
    def search_and_add(query, limit=5)     # Поиск + добавление
`

---

## 3. Интеграция с графом

### add_to_fractal_graph()

`python
def add_to_fractal_graph(fractal_graph, top_k=100, node_type="wikipedia"):
    # Извлекает top_k статей из SQLite
    # Для каждой статьи:
    node = fractal_graph.add_node(
        content=f"""{title}

{text[:1000]}""",        # Только первые 1000 символов!
        node_type="wikipedia",
        level=1,
        confidence=0.7,
        metadata={
            "title": title,
            "url": url,
            "category": category,
            "source": "wikipedia"
        }
    )
`

### search_and_add_to_graph()

`python
def search_and_add_to_graph(query, fractal_graph, top_k=10):
    # 1. Семантический поиск в KB
    results = self.search(query, limit=top_k)
    
    # 2. Добавление каждого результата в FGv2
    for item in results:
        node = fractal_graph.add_node(
            content=item["text"],
            node_type="wikipedia",
            level=2,
            confidence=item.get("similarity", 0.5),
            metadata={
                "title": item.get("title"),
                "url": item.get("url"),
                "source": "wikipedia_search",
                "query": query
            }
        )
`

### FractalMemoryGraph.add_node() интерфейс:

`python
def add_node(
    self,
    content: str,
    node_type: str = "concept",
    level: int = 1,
    confidence: float = 0.5,
    metadata: Optional[Dict] = None,
    auto_vectorize: bool = True,
    auto_cluster: bool = True,
    cluster_threshold: float = 0.6
) -> FractalNode
`

---

## 4. Кэширование

### SQLite:
- PRAGMA journal_mode=WAL - режим Write-Ahead Logging
- PRAGMA synchronous=NORMAL - безопасный, но быстрый
- Индексы: idx_chunks_article, idx_chunks_title

### FAISS:
- Строится один раз при первом поиске (lazy)
- _build_faiss_index() вызывается в _search_with_faiss()
- Хранится в памяти: self._faiss_index
- Маппинг индексов: self._faiss_ids

### Singleton:
`python
_wikipedia_kb = None
_kb_lock = threading.Lock()

def get_wikipedia_kb(data_dir=None) -> WikipediaKnowledgeBase:
    global _wikipedia_kb
    with _kb_lock:
        if _wikipedia_kb is None:
            _wikipedia_kb = WikipediaKnowledgeBase(data_dir=data_dir)
        return _wikipedia_kb
`

---

## 5. Проблемы

### Критические:

1. **Текст статей обрезается до 10000 символов**
   `python
   conn.execute(..., (article_id, title, url, text[:10000], len(chunks), now, category))
   `
   Длинные статьи теряют информацию

2. **При экспорте в FGv2 берётся только 1000 символов**
   `python
   content=f"""{title}\n\n{text[:1000]}"""
   `
   Для dd_to_fractal_graph() - потеря 90% контента

3. **Нет инкрементального обновления**
   - dd_article() проверяет SELECT id FROM articles WHERE id = ?
   - Если статья существует - возврат без обновления
   -_embeddings не пересчитываются

### Существенные:

4. **CPU-only эмбеддер**
   - device='cpu' - медленная векторизация
   - Нет GPU ускорения даже если доступно

5. **FAISS optional**
   - Без FAISS поиск O(n) по всем чанкам
   - Для 500k чанков это очень медленно

6. **Нет автоматической синхронизации**
   - WikipediaLoader требует ручного вызова
   - Нет расписания обновлений

7. **Chunking по предложениям с потерями**
   `python
   sentences = re.split(r'(?<=[.!?])\s+', text)
   `
   Исключения, кавычки, сноски могут нарушить логику

8. **WikipediaLoader - отдельный класс**
   - Не интегрирован в основной поток
   - Нет автоматической загрузки статей

### Мелкие:

9. **Нет TTL или размера очереди**
   - max_chunks: int = 500000 не проверяется

10. **Embedding lazy loading**
    - Если sentence_transformers_cache не работает - будет retry каждый раз

---

## 6. Оценка

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| **Функциональность** | 7/10 | Работает, но есть существенные ограничения |
| **Производительность** | 6/10 | FAISS помогает, но CPU узкое место |
| **Интеграция с FGv2** | 7/10 | Методы есть, но truncating контента |
| **Удобство использования** | 5/10 | Требует ручной настройки WikipediaLoader |
| **Надёжность** | 6/10 | Singleton, WAL, fallback есть |

### Итог:

Wikipedia KB реализована как **автономная база знаний** с семантическим поиском. Архитектура правильная:
- FAISS для быстрого поиска
- SQLite для хранения
- Интеграция с FGv2

**Однако** ключевые проблемы:
1. Truncation контента (10000 и 1000 символов)
2. CPU-only векторизация
3. Ручная загрузка статей

### Рекомендации:
1. Увеличить лимиты или сделать их параметрами
2. Добавить GPU support для эмбеддера
3. Интегрировать WikipediaLoader в автозагрузку
4. Добавить инкрементальное обновление статей
