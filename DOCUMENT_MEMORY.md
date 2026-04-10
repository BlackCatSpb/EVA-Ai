# Document Virtual Memory System - Интеграция

## Что реализовано

Система превращает большие документы в "книгу со страницами" с индексацией:

### 1. Разбиение на страницы (`DocumentChunker`)
- Документы автоматически разбиваются на страницы по ~512 токенов
- Страницы имеют перекрытие (50 токенов) для сохранения контекста
- Каждая страница получает уникальный ID: `{document_id}_page_{number}`

### 2. Хранение в графе (`FractalGraph`)
- Страницы сохраняются как узлы в фрактальном графе
- Связи между страницами: `next_page`, `prev_page`
- Корневой узел документа с метаданными

### 3. Кэширование (`LazyLoadingCache`)
- LRU-кэш для часто используемых страниц
- Автоматическая выгрузка редко используемых страниц
- Ограничение по размеру: max 20 страниц / 10000 токенов

### 4. Интеграция с веб-интерфейсом
- Большие документы (>1000 токенов) автоматически загружаются в виртуальную память
- При запросе загружаются только релевантные страницы
- В UI отображается индикатор "📖 В виртуальной памяти"

## API Endpoints

```
GET /api/documents/memory?session_id={id}
    - Получить список документов в виртуальной памяти

GET /api/documents/memory/{document_id}?session_id={id}
    - Получить статистику по документу

DELETE /api/documents/memory/{document_id}?session_id={id}
    - Очистить документы сессии
```

## Использование

### Автоматическое (через веб-интерфейс)
1. Загрузите файл через интерфейс
2. Если файл >1000 токенов, он автоматически попадет в виртуальную память
3. Задавайте вопросы - система будет искать релевантные страницы

### Программное (через код)
```python
from eva_ai.memory.document_manager import DocumentVirtualMemory

# Создать менеджер документов
doc_memory = DocumentVirtualMemory(brain=brain)

# Загрузить документ
doc_id = doc_memory.ingest_document(
    content=large_text,
    title="Мой документ",
    document_id="doc_001"
)

# Запрос к документу
result = doc_memory.query_document(
    document_id=doc_id,
    query="что написано о Python?",
    top_k=3  # Количество страниц для загрузки
)

print(result['context'])  # Контекст из релевантных страниц
```

### Через DualGenerator
```python
# Получить доступ к dual_generator
dg = brain.two_model_pipeline.dual_generator

# Загрузить документ
doc_id = dg.load_document(content, "Название")

# Сгенерировать ответ с учетом документа
result = dg.generate_with_document(
    query="вопрос по документу",
    document_id=doc_id,
    mode="extended"
)
```

## Преимущества

1. **Экономия памяти** - загружаются только нужные страницы
2. **Быстрота** - кэширование часто используемых страниц
3. **Масштабируемость** - работа с документами любого размера
4. **Контекст** - перекрытие страниц сохраняет связность текста

## Тестирование

```bash
python test_document_integration.py
```

## Файлы

- `eva_ai/memory/document_manager.py` - Основная реализация
- `eva_ai/gui/web_gui/server_main.py` - Интеграция с веб-интерфейсом
- `eva_ai/gui/web_gui/server_routes.py` - API endpoints
- `eva_ai/gui/web_gui/static/js/app.js` - UI отображение
- `eva_ai/gui/web_gui/static/css/style.css` - Стили
