# Анализ системы EVA AI - Полный отчёт

> Дата анализа: 2026-04-13
> Версия: 2.0 (обновлённый)

---

## ЦЕЛЬ ПРОЕКТА

Создать **автономную самообучаемую систему** со следующими принципами:

1. **ML модели (Qwen) = интерпретатор/переводчик** с натурального языка на язык системы
2. **Постоянное обучение** - запоминание ВСЕХ диалогов
3. **Извлечение знаний** - концепты, противоречия, факты из каждого разговора
4. **Заполнение графа памяти** - структурированная база знаний (не синтетическое обогащение)
5. **Фрактальная структура** - многоуровневая система хранения
6. **Автономность** - саморегулируемость и адаптивность к изменениям
7. **Основа для будущей модели** - токенизатор и модель на основе структуры графа

---

## Содержание

1. [UnifiedGenerator](#1-unifiedgenerator--генерация)
2. [FractalGraph V2](#2-fractalgraph-v2)
3. [Concept/Contradiction система](#3-conceptcontradiction-система)
4. [Self-Dialog система](#4-self-dialog-система)
5. [Web Search модуль](#5-web-search-модуль)
6. [HybridCache](#6-hybridcache)
7. [Brain Query](#7-brain-query)
8. [Web GUI / Server](#8-web-gui--server)
9. [Сводка проблем](#9-сводка-проблем)

---

## 1. UnifiedGenerator / Генерация

### 1.1 Карта импортов

```python
import time                        # Замер времени генерации
import logging                     # Логирование работы системы
from typing import Dict, List, Optional, Any, Tuple, Generator  # Типизация
from dataclasses import dataclass  # Датакласс GenerationResult
from pathlib import Path           # Работы с путями к моделям
from enum import Enum              # Enum ModelType

from .context_chunking import ChunkedContextProcessor, StreamingGenerator, ContextChunk  # Обработка больших контекстов
```

### 1.2 Логика импортов

| Импорт | Назначение |
|--------|------------|
| **time** | Замер времени генерации (`generation_time`) |
| **logging** | Логирование этапов генерации |
| **typing** | Типизация для maintainability |
| **dataclass** | Структура `GenerationResult` |
| **pathlib.Path** | Кроссплатформенная работа с путями |
| **enum.Enum** | Типы моделей (LOGIC, CONTEXT, CODER) |
| **ChunkedContextProcessor** | Обработка контекстов >8K токенов |

### 1.3 Карта методов

```python
class ModelType(Enum):
    LOGIC = "logic"      # Краткие ответы
    CONTEXT = "context" # Развёрнутые ответы
    CODER = "coder"     # Генерация кода

class SimpleRouter:
    # CODER_KEYWORDS, CONTEXT_KEYWORDS - словари для маршрутизации
    def route(self, query: str) -> ModelType:
        # Определение типа модели по ключевым словам

class UnifiedGenerator:
    def __init__(self, logic_model_path, context_model_path, coder_model_path, ...)
    def _load_model(self, model_type: ModelType) -> bool
    def generate(self, query, context, max_tokens, temperature, system_prompt) -> GenerationResult
    def generate_dual(self, query, context, max_tokens_logic, max_tokens_context, temperature, system_prompt) -> GenerationResult
    def generate_iterative(self, query, context, max_tokens_logic, max_tokens_context, temperature, system_prompt, check_contradictions, check_concepts) -> GenerationResult
    def _get_concepts_context(self, query: str) -> str
    def _get_contradictions_context(self, query: str) -> str
    def _get_web_search_context(self, query: str) -> str
    def _build_context(self, query, provided_context) -> str
    def _format_prompt(self, query, context, system_prompt, model_type) -> str
    def _split_text_chunks(self, text, chunk_size) -> List[str]  # НЕ ИСПОЛЬЗУЕТСЯ!
    def generate_streaming(self, query, context, max_tokens, temperature, chunk_size) -> Generator
```

### 1.4 Критические проблемы

#### Проблема 1: Несоответствие параметров фабрики

**Файл:** `eva_ai/core/unified_generator.py:1157-1162`

```python
# Фабричная функция передаёт НЕВЕРНЫЕ параметры:
return UnifiedGenerator(
    general_model_path=Path(general_path) if general_path else None,  # ❌ НЕВЕРНО!
    code_model_path=Path(code_path) if code_path else None,             # ❌ НЕВЕРНО!
    n_ctx=n_ctx,
    n_threads=n_threads,
    fractal_graph=fractal_graph,
    brain=brain
)

# А конструктор ожидает:
def __init__(
    self,
    logic_model_path: Optional[Path] = None,      # ✅
    context_model_path: Optional[Path] = None,    # ✅
    coder_model_path: Optional[Path] = None,       # ✅
    ...
)
```

**Влияние:** Фабрика передаёт неправильные имена параметров → модели не загрузятся корректно.

---

#### Проблема 2: Неправильный prompt format

**Файл:** `eva_ai/core/unified_generator.py:833-866`

```python
def _format_prompt(self, query, context, system_prompt, model_type):
    # Текущий код (НЕПРАВИЛЬНО):
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Контекст: {context}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
```

**Проблемы:**
1. `system_prompt` добавляется в role `system`, но формат неправильный для Qwen
2. Для модели используется message с `Контекст:`, что может путать модель

**Правильный формат для Qwen:**
```python
prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
Контекст: {context}
Запрос: {query}
<|im_end|>
<|im_start|>assistant
"""
```

---

#### Проблема 3: `_split_text_chunks` не используется

**Файл:** `eva_ai/core/unified_generator.py:1094-1128`

Метод определён, но НИГДЕ не вызывается. В `generate_streaming` используется собственная логика буфера.

---

### 1.5 Избыточные методы

1. **`_split_text_chunks`** - не используется, дублирует логику буфера
2. **`generate_code`** - просто враппер над `generate()` с фиксированным system_prompt
3. **`StreamingGenerator` импорт** - импортируется но не используется

---

## 2. FractalGraph V2

### 2.1 Структура модулей

```
eva_ai/memory/fractal_graph_v2/
├── __init__.py              # FractalMemoryGraph (main)
├── storage.py               # FractalGraphV2 (SQLite)
├── types.py                 # FractalNode, FractalEdge, SemanticGroup
├── embeddings.py            # EmbeddingsManager
├── gguf_parser.py           # Парсинг GGUF
├── gguf_extractor.py        # Извлечение знаний из GGUF
├── gguf_shadow.py           # Когнитивная тень модели
├── hybrid_tokenizer.py      # Aho-Corasick + BPE токенизация
├── semantic_context_cache.py # FAISS кэш
├── prompt_templates.py      # Шаблоны промтов
├── virtual_token_handler.py # Обработка виртуальных токенов
├── eva_container.py         # .eva формат
├── tokenizer.py             # Альтернативный токенизатор
└── dual_generator.py        # DualGenerator (Condensed + Extended)
```

### 2.2 Ключевые методы

| Метод | Назначение |
|-------|------------|
| `semantic_search()` | Семантический поиск по косинусному сходству |
| `keyword_search()` | Поиск по ключевым словам |
| `add_node()` | Добавить узел |
| `add_edge()` | Добавить связь |
| `create_group()` | Создать семантическую группу |
| `get_context()` | Получить контекст узла |
| `load_gguf_knowledge()` | Извлечь знания из GGUF |

### 2.3 Проблемы

1. **Дублирование токенизаторов:**
   - `HybridTokenizer` (hybrid_tokenizer.py) - для виртуальных токенов
   - `GraphTokenizer` (tokenizer.py) - для контекста генерации

2. **Дублирование semantic_search:**
   - `semantic_search` (FractalMemoryGraph) - через EmbeddingsManager
   - `SemanticContextCache.search` - CPU-based FAISS

3. **Недостающие методы:**
   - `delete_node` - нет метода удаления
   - `update_node` - нет метода обновления
   - `merge_groups` - нет метода слияния групп

---

## 3. Concept/Contradiction система

### 3.1 ConceptExtractor (быстрый уровень)

```python
def extract_concepts(query, response, context):
    # Извлекает концепты из текста
    # Частотный анализ терминов

def _generate_facts(term, query, response):
    # ⚠️ ПРОБЛЕМА: Генерирует шаблоны, а не реальные факты
    facts = [
        {'relation_type': 'is_a', 'value': f'{term} - это понятие, которое...'},  # ❌
        {'relation_type': 'has_property', 'value': f'{term} имеет свойство...'},  # ❌
        {'relation_type': 'can', 'value': f'{term} может...'},  # ❌
        {'relation_type': 'related_to', 'value': f'{term} связан с...'},  # ❌
    ]
```

**Проблема:** Факты имеют placeholder форму `"{term} - это понятие, которое..."` - бесполезные заглушки. Параметры `query` и `response` не используются.

---

### 3.2 ContradictionMiner (аналитический уровень)

```python
def _compute_contradiction(self, node1: Dict, node2: Dict) -> float:
    # ⚠️ ПРОБЛЕМА: Эвристика вместо NLI
    contradictions_map = {
        'быстрый': ['медленный', 'медленнее'],
        'хороший': ['плохой', 'худший'],
        'да': ['нет', 'не'],
        # ... всего 12 пар
    }
    
    # Простая проверка наличия антонима
    # НЕ использует NLI модель!
```

**Проблема:**
- Словарь только 12 пар антонимов
- Не понимает контекст, синонимы
- Русский язык - английские узлы игнорируются
- Должна использоваться NLI-модель (deberta-v3-xsmall-mnli), но она не реализована

---

### 3.3 Проблемы архитектуры

| Проблема | Описание |
|----------|----------|
| Шаблонные факты | ConceptExtractor генерирует "заглушки" вместо реальных фактов |
| Эвристика вместо NLI | ContradictionMiner использует словарь антонимов вместо NLI-модели |
| Дублирование кода | Оба майнера имеют похожую структуру - нужен базовый класс MinerBase |

---

## 4. Self-Dialog система

### 4.1 Определение переменных

```python
# dialog_concepts.py:28
self._resolved_knowledge = []  # ❌ Растёт бесконечно!

# dialog_concepts.py
self._concept_queue = []       # ❌ Только append, нет limit
self._contradiction_topics = [] # ❌ Только append
```

### 4.2 Утечка памяти

**Проблема:** `_resolved_knowledge.append(summary)` в:
- `_save_concept_dialog_results()` (строка 512)
- `_save_contradiction_resolution()` (строка 530)

Нет ограничения размера списка.

---

### 4.3 Бесполезные fallback методы

```python
# dialog_concepts.py - эти методы бесполезны:

def _generate_concept_intro(self, concept: str, info: Dict) -> str:
    return f"""Изучаем концепт: {concept}
    Базовое определение: {concept} - это ключевое понятие..."""

def _generate_concept_criticism(self, concept: str, info: Dict) -> str:
    return f"""Критический анализ концепта '{concept}':
    Возможные проблемы: ..."""

# Аналогичные методы:
# _generate_learning_directions
# _generate_teaching_recommendations  
# _present_contradiction
# _analyze_contradiction_sides
# _synthesize_contradiction
# _formulate_resolution
```

**Проблема:** Генерируют статический текст без реальной логики. UnifiedGenerator практически всегда доступен.

---

## 5. Web Search модуль

### 5.1 Интеграция

```python
# unified_generator.py
def _get_web_search_context(self, query: str) -> str:
    need_search, _ = needs_web_search(query)  # Эвристика
    if not need_search:
        return ""
    
    web_search = get_web_search_engine()
    results = web_search.search(query, max_results=3)  # ⚠️ 3 результата
    return formatted_results

# brain_query.py
need_search, search_reason = needs_web_search(query)
if need_search:
    web_result = web_search.search(query, max_results=5)  # ⚠️ 5 результатов!
```

**Проблема:** Несогласованность - max_results=3 в UnifiedGenerator, =5 в brain_query

---

### 5.2 needs_web_search

```python
# brain_query.py:18
def needs_web_search(query: str) -> tuple[bool, str]:
    # Приветствия → False
    if any(word in query.lower() for word in ['привет', 'здравствуй', ...]):
        return False, "приветствие"
    
    # Запросы о себе → False
    if any(word in query.lower() for word in ['кто ты', 'что ты', ...]):
        return False, "запрос о себе"
    
    # Математика/код → False
    if any(word in query.lower() for word in ['посчитай', 'напиши код', ...]):
        return False, "математика/код"
    
    # Всё остальное → True
    return True, "обогащение контекста"
```

**Проблема:** Примитивная эвристика - только стоп-слова, нет анализа вопросительных слов (кто/что/где/когда/почему)

---

### 5.3 _rank_by_relevance

```python
# web_search_integrated.py:970
def _rank_by_relevance(results, query):
    query_words = set(re.findall(r'\w+', query.lower()))
    
    for result in results:
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        
        # Простое совпадение слов
        title_matches = sum(1 for word in query_words if word in title)
        snippet_matches = sum(1 for word in query_words if word in snippet)
        
        relevance = (title_matches * 0.6 + snippet_matches * 0.4) / max(len(query_words), 1)
```

**Проблемы:**
- Не использует эмбеддинги
- Только прямое совпадение слов
- Не учитывает синонимы, морфологию

---

## 6. HybridCache

### 6.1 Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    HybridTokenCache                         │
├─────────────────────────────────────────────────────────────┤
│  Уровень 1: VRAM (если GPU доступен)                      │
│    - vram_cache: LRUCache (~1.5GB)                         │
├─────────────────────────────────────────────────────────────┤
│  Уровень 2: RAM                                            │
│    - ram_cache: LRUCache (~1GB)                            │
├─────────────────────────────────────────────────────────────┤
│  Уровень 3: Disk                                           │
│    - TokenDiskCache: до 50GB                               │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Ключевые методы

| Метод | Назначение |
|-------|------------|
| `get(key)` | Получить из кэша (VRAM → RAM → Disk) |
| `put(key, value)` | Добавить в кэш |
| `get_recent(limit)` | Получить последние записи |
| `add_context()` | Сохранить контекст сессии |
| `add_search_results()` | Кэшировать результаты поиска |

### 6.3 Особенности

- **LRU в памяти**: использует OrderedDict
- **Персистентность**: метаданные в JSON, данные в бинарных файлах
- **Динамическое управление памятью**: адаптируется под доступную RAM

---

## 7. Brain Query

### 7.1 Обработка запроса

```
process_query(query)
    │
    ├─→ greeting_cache (быстрые ответы)
    │
    ├─→ _get_cached_response (TTL=300s)
    │
    ├─→ _handle_gguf_pipeline() [primary]
    │       │
    │       ├─→ needs_web_search()
    │       ├─→ Self-dialog контекст
    │       └─→ pipeline.process_query()
    │
    ├─→ _handle_qwen_mode()
    │
    ├─→ _handle_fg_only()
    │
    └─→ _handle_fallback() [11 уровней!]
```

### 7.2 Проблемы

1. **Fallback chain слишком длинная** - 11 уровней
2. **Несколько систем кэширования** - дублирование функционала

---

## 8. Web GUI / Server

### 8.1 server_main.py (Основной сервер)

**Проблемы:**
- Очень длинный `process_message` (740+ строк)
- Глобальная переменная `web_gui_instance` - нет thread safety
- Ошибка в строке 331: `active_doc_id = max(..., key=...)` без проверки на пустой словарь

### 8.2 server_routes.py

**Проблемы:**
- Дублирование `/api/chat` роутов (с server_routes_chat.py)
- Слишком много try/except блоков в analytics
- pytesseract загружается при импорте

### 8.3 server_routes_chat.py

**Проблемы:**
- Дублирование `/api/chat` - конфликт при регистрации
- Нет fallback для streaming

### 8.4 app.js (Клиент)

**Проблемы:**
- XHR + SSE смешение подходов
- EventSource не закрывается при logout
- Глобальные переменные в IIFE
- Нет retry логики при ошибках сети

---

## 9. Reasoning модули

### 9.1 SelfReasoningEngine

**Методы:**
- `process_query()` - главный метод
- `analyze_response()` - анализ по 5 факторам
- `_analyze_logical_factors()` - ethics, knowledge, contradiction, context, logic
- `_evaluate_*_factor()` - 5 методов оценки

**Проблемы:**
- Сложная архитектура с множеством fallback
- Рекурсивный цикл (max 5 итераций) может быть медленным

### 9.2 EnhancedReasoningEngine

Аналогичная структура с дополнительными проверками качества.

---

## 10. Memory системы

### 10.1 MemoryManager

**Методы:**
- `add_memory()`, `get_memory()`, `delete_memory()` - CRUD операции
- `search_knowledge()` - семантический поиск через FGv2
- `add_fact()` - добавление фактов в граф

**Проблемы:**
- Персистентность через 4 JSON файла (медленно)
- Нет индексации для entity search
- Thread safety неполная

### 10.2 DocumentManager

**Методы:**
- `ingest_document()` - загрузка документа
- `query_document()` - поиск по документу
- `get_page()` - lazy loading страниц

**Проблемы:**
- `_findRelevantPages()` - заглушка (возвращает первые top_k)
- Hit rate всегда 0.85 (заглушка)
- Embeddings не вычисляются

---

## 11. Training система

### 11.1 GGUFTrainingSystem

**Проблемы:**
- **Обучение не реализовано** - `_train_separate_instance()` только логирует
- Нет PEFT/LoRA библиотек
- Верификация - заглушки (всегда True)
- Не интегрирован в brain

---

## 11. Training система

### 11.1 GGUFTrainingSystem

**Проблемы:**
- **Обучение не реализовано** - `_train_separate_instance()` только логирует
- Нет PEFT/LoRA библиотек
- Верификация - заглушки (всегда True)
- Не интегрирован в brain

---

## 12. MLearning модули

### 12.1 unified_text_processor.py

**Проблемы:**
- Дублирование `embedder` и `embedding_model`
- Неполный cleanup (executor не очищается)
- use_async=True но асинхронная обработка не реализована
- Утечка памяти: sentiment_analyzer не удаляется

### 12.2 universal_model_manager.py

**Проблемы:**
- Дублирование `get_recommendation()` и `_auto_select()`
- Нет валидации в _auto_select
- Пустой fallback в generate()

### 12.3 unified_fractal_manager.py

**Проблемы:**
- **max_tokens undefined** (критическая ошибка!)
- Неверный путь конфига

### 12.4 web_search_learning_integration.py

**Проблемы:**
- **max_tokens undefined x2** (критическая ошибка!)
- search_cache race condition
- Низкий порог auto_search_threshold = 0.5

---

## 13. Knowledge модули

### 13.1 kg_adapter.py

**Проблемы:**
- Устаревший API
- Упрощённый поиск пути

### 13.2 wikipedia_kb.py

**Проблемы:**
- Производительность (полная загрузка в память)
- Зависимость от эмбеддера
- Нет FAISS/index

### 13.3 context_entity.py

**Проблемы:**
- Stub-реализация (проксирует на FGv2)
- Нет реального NLP (NER, POS-tagging)

### 13.4 knowledge_graph.py

**Проблемы:**
- Двойная абстракция
- Fallback создаёт новый FG без предупреждения

---

## 14. GUI модули

### 14.1 gui_main.py

**Проблемы:**
- Жёсткая привязка к TkAgg
- Нет обработки ошибок инициализации analytics_module

### 14.2 widgets.py

**Проблемы:**
- Глобальное состояние _toast_tk_root
- Нет DPI awareness

### 14.3 chat_history.py

**Проблемы:**
- Нет индексации (O(n) при 1000+ сообщений)
- Потокобезопасность

### 14.4 analytics_module.py

**Проблемы:**
- Нет валидации brain при инициализации
- matplotlib Figure объекты не закрываются
- threading в GUI без after()

---

## 15. Storage и System

### 15.1 fractal_storage.py

**Проблемы:**
- Импорты внутри методов
- Нет валидации данных
- get_model() не загружает модель

### 15.2 health_monitor.py

**Проблемы:**
- Жёсткие веса компонентов
- Незакрытое соединение SQLite
- analysis_queue без проверки

### 15.3 system_monitor.py

**Проблемы:**
- Глобальный экземпляр с автозапуском
- Нет лимита на алерты
- Race conditions

---

## 16. Ethics и Distributed

### 16.1 framework_core.py (Ethics)

**Проблемы:**
- Потокобезопасность
- Хардкодированные интервалы

### 16.2 distributed_system.py

**Проблемы:**
- HTTP без обработки таймаутов
- SQLite в многопоточной среде
- Несуществующие методы вызываются

---

## Сводка проблем

### 9.1 Критические (block system)

| # | Проблема | Файл | Строки |
|---|----------|------|--------|
| 1 | Неправильные параметры в `create_unified_generator` | unified_generator.py | 1157-1162 |
| 2 | system_prompt в wrong role | unified_generator.py | 833-866 |

### 9.2 Высокий приоритет

| # | Проблема | Файл |
|---|----------|------|
| 3 | `_split_text_chunks` не используется | unified_generator.py:1094 |
| 4 | Утечка памяти в `_resolved_knowledge` | dialog_concepts.py:28 |
| 5 | Рост очередей без ограничения | dialog_concepts.py |
| 6 | Несогласованность max_results (3 vs 5) | unified_generator.py vs brain_query.py |

### 9.3 Средний приоритет

| # | Проблема | Файл |
|---|----------|------|
| 7 | Шаблонные факты в ConceptExtractor | concept_extractor.py:160 |
| 8 | Эвристика вместо NLI | contradiction_miner.py:446 |
| 9 | needs_web_search слишком примитивен | brain_query.py:18 |
| 10 | Дублирование semantic_search | fractal_graph_v2 |
| 11 | Дублирование токенизаторов | fractal_graph_v2 |

### 9.4 Избыточные методы

- `_split_text_chunks` (UnifiedGenerator)
- Fallback методы в dialog_concepts.py (8 штук)
- `_basic_web_search` в web_search_integrated.py

### 9.5 Отсутствующие методы

- `delete_node`, `update_node` (FractalGraph)
- `retry` логика в генерации
- NLI в ContradictionMiner
- Реальный семантический поиск в DocumentManager

---

## 17. Core модули

### 17.1 init_factories.py

**Проблемы:**
- Дублирование try/except логики
- Hardcoded параметры (max_vram_gb=0.5)
- Нет валидации зависимостей
- Риск ImportError

### 17.2 core_brain.py

**Проблемы:**
- 10 миксинов - нарушение SRP
- Дублирование EventBus (self.events и self._new_event_bus)
- Нет явного порядка инициализации
- Глобальное состояние (singleton)

### 17.3 event_bus.py

**Проблемы:**
- Избыточное логирование (10+ раз на каждый handler)
- EventPriority не используется при обработке
- Нет dead letter queue
- Ограниченная история (max_history=10000)

### 17.4 brain_coordination.py

**Проблемы:**
- CommandIssuerMixin > 600 строк
- Хардкод retry логики (max_retries=3)
- Нет circuit breaker
- Duplicate _on_pipeline_complete и _track_query_success

---

## 18. Memory модули

### 18.1 hybrid_token_cache.py

**Проблемы:**
- Сложная архитектура 3 уровней (VRAM/RAM/Disk)
- Динамические импорты (циклические зависимости)
- Потенциальные утечки памяти

### 18.2 memory_manager.py

**Проблемы:**
- Дублирование инициализации кэша
- Сложная система блокировок
- EventBus зависимость

### 18.3 document_manager.py

**Проблемы:**
- Неполная реализация поиска
- Hit rate всегда 0.85
- Memory leak в _page_index

### 18.4 memory_types.py

**Проблемы:**
- Неполный набор типов
- Ограниченная функциональность
- Нет валидации

---

## 19. Learning и Commands

### 19.1 scheduler_core.py

**Проблемы:**
- `_execute_task()` не реализован (заглушка)
- Hardcoded values (task_timeout=300, max_concurrent_tasks=8)
- Race conditions

### 19.2 deferred_command_system.py

**Проблемы:**
- Глобальная переменная _global_event_bus
- Избыточное логирование
- Нет обработки зависших команд

### 19.3 contradiction_manager.py

**Проблемы:**
- BaseComponent - заглушка
- self.detector может быть None
- Примитивный расчёт весов

---

## 20. Полная сводка проблем (ИТОГ)

### 🔴 КРИТИЧЕСКИЕ (система не работает)

| # | Проблема | Файл | Строки |
|---|----------|------|--------|
| 1 | Неправильные параметры в `create_unified_generator` | unified_generator.py | 1157-1162 |
| 2 | Обучение не реализовано (симуляция) | gguf_training_system.py | 350+ |
| 3 | `_findRelevantPages()` заглушка | document_manager.py | - |
| 4 | max_tokens undefined | unified_fractal_manager.py | - |
| 5 | max_tokens undefined x2 | web_search_learning_integration.py | - |
| 6 | DocumentManager эмбеддинги не вычисляются | document_manager.py | - |
| 7 | `_execute_task()` не реализован | scheduler_core.py | - |

### 🟠 ВЫСОКИЙ ПРИОРИТЕТ

| # | Проблема | Файл |
|---|----------|------|
| 8 | system_prompt в wrong role | unified_generator.py:833 |
| 9 | Утечка памяти в `_resolved_knowledge` | dialog_concepts.py:28 |
| 10 | Рост очередей без ограничения | dialog_concepts.py |
| 11 | Несогласованность max_results (3 vs 5) | unified_generator.py vs brain_query.py |
| 12 | Персистентность через JSON файлы | memory_manager.py |
| 13 | Нет интеграции Training в brain | gguf_training_system.py |
| 14 | Верификация в Training заглушки | gguf_training_system.py |
| 15 | Stub в context_entity.py | context_entity.py |
| 16 | Двойная абстракция knowledge_graph | knowledge_graph.py |
| 17 | 10 миксинов в CoreBrain | core_brain.py |
| 18 | CommandIssuerMixin > 600 строк | brain_coordination.py |

### 🟡 СРЕДНИЙ ПРИОРИТЕТ

| # | Проблема | Файл |
|---|----------|------|
| 19 | Шаблонные факты в ConceptExtractor | concept_extractor.py:160 |
| 20 | Эвристика вместо NLI | contradiction_miner.py:446 |
| 21 | needs_web_search слишком примитивен | brain_query.py:18 |
| 22 | Дублирование semantic_search | fractal_graph_v2 |
| 23 | Дублирование токенизаторов | fractal_graph_v2 |
| 24 | Дублирование `/api/chat` роутов | server_routes.py vs server_routes_chat.py |
| 25 | Очень длинный process_message | server_main.py:203 |
| 26 | Глобальная переменная web_gui_instance | server_main.py |
| 27 | Hit rate всегда 0.85 | document_manager.py |
| 28 | Дублирование embedder/embedding_model | unified_text_processor.py |
| 29 | use_async не работает | unified_text_processor.py |
| 30 | Упрощённый поиск пути в kg_adapter | kg_adapter.py |
| 31 | Нет FAISS в wikipedia_kb | wikipedia_kb.py |
| 32 | Дублирование EventBus | core_brain.py |
| 33 | Избыточное логирование | event_bus.py |
| 34 | Глобальная переменная _global_event_bus | deferred_command_system.py |
| 35 | BaseComponent заглушка | contradiction_manager.py |

### 🟢 НИЗКИЙ ПРИОРИТЕТ

| # | Проблема | Файл |
|---|----------|------|
| 36 | XHR + SSE смешение подходов | app.js |
| 37 | EventSource не закрывается при logout | app.js |
| 38 | `_split_text_chunks` не используется | unified_generator.py:1094 |
| 39 | Нет DPI awareness | widgets.py |
| 40 | Импорты внутри методов | fractal_storage.py |
| 41 | Жёсткие веса в health_monitor | health_monitor.py |

---

## Итоговый подсчёт

- **Критических**: 7
- **Высокий приоритет**: 11
- **Средний приоритет**: 17
- **Низкий приоритет**: 6
- **Всего**: 41 проблема

---

## Рекомендуемый порядок исправлений

### Фаза 1: Критические
1. `create_unified_generator` параметры
2. max_tokens в ML модулях
3. `_execute_task()` в scheduler

### Фаза 2: Высокий приоритет
4. Очереди и память в Self-Dialog
5. Интеграция Training
6. CoreBrain миксины

### Фаза 3: Рефакторинг
7. Дублирование и абстракции
8. Конфигурация (hardcoded values)
9. Логирование

---

*Анализ завершён: 20 областей, 41 проблема*
*ЦЕЛЬ: Автономная самообучаемая система*
| 29 | Глобальное состояние _toast_tk_root | widgets.py |
| 30 | analytics_module matplotlib утечки | analytics_module.py |
| 31 | Distributed HTTP без таймаутов | distributed_system.py |
| 32 | Health monitor SQLite не закрывается | health_monitor.py |
| 33 | System monitor автозапуск | system_monitor.py |

### 🟢 НИЗКИЙ ПРИОРИТЕТ

| # | Проблема | Файл |
|---|----------|------|
| 34 | XHR + SSE смешение подходов | app.js |
| 35 | EventSource не закрывается при logout | app.js |
| 36 | `_split_text_chunks` не используется | unified_generator.py:1094 |
| 37 | Нет DPI awareness | widgets.py |
| 38 | Импорты внутри методов | fractal_storage.py |
| 39 | Жёсткие веса в health_monitor | health_monitor.py |

---

## Рекомендации по исправлению

### Фаза 1: Критические (срочно)

1. ✅ Исправить параметры в `create_unified_generator`
2. ✅ Исправить prompt format для Qwen
3. ⏳ Реализовать обучение или удалить модуль Training
4. ⏳ Исправить `_findRelevantPages()` в DocumentManager
5. ⏳ Исправить max_tokens в unified_fractal_manager и web_search_learning_integration

### Фаза 2: Важно

6. Удалить/исправить неиспользуемые методы
7. Добавить ограничение очередей в Self-Dialog
8. Унифицировать max_results в Web Search
9. Интегрировать Training в brain
10. Исправить Stub в context_entity.py

### Фаза 3: Опционально

11. Интегрировать NLI в ContradictionMiner
12. Улучшить факты в ConceptExtractor
13. Рефакторить FractalGraph (токенизаторы, semantic_search)
14. Исправить дублирование роутов
15. Добавить DPI awareness

---

*Документ создан для планирования работы системы*
*ЦЕЛЬ: Автономная самообучаемая система*
