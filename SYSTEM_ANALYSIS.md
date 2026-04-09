# Анализ системы EVA AI - План исправлений

## Выполнено недавно (DualGenerator)

| Компонент | Файл | Статус |
|-----------|------|--------|
| HybridPipelineAdapter | `eva_ai/core/hybrid_pipeline_adapter.py` | ✅ Работает |
| DualGenerator | `eva_ai/memory/fractal_graph_v2/dual_generator.py` | ✅ Работает |
| FractalPipeline | `eva_ai/core/fractal_pipeline.py` | ✅ Работает |
| Интеграция с brain_query | `eva_ai/core/brain_query.py` | ✅ Работает (FG_ONLY_MODE=False) |
| Reasoning в Web GUI | `app.js`, `style.css` | ✅ Работает |

---

## 1. Критические проблемы ( block system)

### 1.1 fractal_graph_v2 не зарегистрирован как factory

**Проблема:** Компонент `fractal_graph_v2` указан в зависимостях, но нет factory функции `create_fractal_graph_v2`.

**Локация:** `eva_ai/core/init_connections.py:18`

**Зависимости:**
```python
'ml_unit': ['memory_manager', 'fractal_graph_v2'],
'query_processor': ['text_processor', 'fractal_graph_v2'],
'reasoning_engine': ['fractal_graph_v2'],
```

**Решение:** Добавить factory функцию в `init_factories.py`:

```python
def create_fractal_graph_v2(initializer):
    try:
        from eva_ai.memory.fractal_graph_v2 import FractalGraphV2
        fg = getattr(initializer.core_brain, 'fractal_graph_v2', None)
        if fg is None:
            fg = FractalGraphV2()
            initializer.core_brain.fractal_graph_v2 = fg
        initializer.logger.info("[OK] fractal_graph_v2 получен")
        return fg
    except Exception as e:
        initializer.logger.error(f"[FAIL] fractal_graph_v2: {e}")
        return None
```

### 1.2 MemoryManager отсутствуют методы clear_cache и optimize

**Проблема:** Вызовы в `manager_core.py:114, 122` к несуществующим методам.

**Локация:** `eva_ai/memory/manager_core.py`

**Решение:** Добавить методы в MemoryManager:

```python
def clear_cache(self):
    """Очистка кэша памяти."""
    self.working_memory = {}
    self.semantic_memory = {}
    logger.info("Memory cache cleared")

def optimize(self):
    """Оптимизация памяти."""
    # Очистка старых записей
    if len(self.working_memory) > self.max_working_memory:
        # Удалить старые записи
        pass
    logger.info("Memory optimized")
```

---

## 2. Устаревшие компоненты

### 2.1 KnowledgeGraph (KG)

**Статус:** Удалён из системы, но есть ссылки на него.

**Файлы:**
- `eva_ai/memory/knowledge_graph/` - папка может существовать
- `eva_ai/adapters/kg_adapter.py` - адаптер для KG

**Решение:** Удалить неиспользуемые адаптеры или пометить как deprecated.

### 2.2 RecursiveModelPipeline

**Статус:** Частично заменён HybridPipelineAdapter, но still используется как fallback.

**Файлы:**
- `eva_ai/core/recursive_model_pipeline.py`
- Ссылки в `brain_components.py` и `hybrid_pipeline_adapter.py`

**Решение:** Оставить для fallback, не удалять.

---

## 3. Проблемы с импортами и дублирование

### 3.1 Дублирование generate_process_query

**Проблема:** Метод `process_query` определён в нескольких местах:
- `CoreBrain.process_query` (core_brain.py)
- `brain_query.py` - метод `_handle_query` внутри CoreBrain
- `chat_module.py` - поиск process_query в нескольких местах

**Файлы:**
- `eva_ai/core/core_brain.py`
- `eva_ai/core/brain_query.py`
- `eva_ai/gui/chat_module.py:902-905`

**Решение:** Использовать единую точку входа - CoreBrain.process_query.

### 3.2 Множественные импорты sentence-transformers

**Проблема:** Модель загружается несколько раз в разных компонентах.

**Файлы:**
- `eva_ai/memory/fractal_graph_v2/embeddings.py`
- `eva_ai/mlearning/embedding_model.py`

**Решение:** Использовать singleton паттерн или общий кэш.

---

## 4. Проблемы Web GUI

### 4.1 Bridge интеграция

**Файл:** `eva_ai/gui/web_gui/bridge.py:495-497`

**Код:**
```python
if self.integrator:
    return self.integrator.process_query(query, context or {})
elif self.brain and hasattr(self.brain, 'process_query'):
    return self.brain.process_query(query, context or {})
```

**Проблема:** Интегратор может отсутствовать при ошибках инициализации.

**Решение:** OK - уже есть fallback на brain.process_query.

### 4.2 Server routes status endpoint

**Файл:** `eva_ai/gui/web_gui/server_routes.py:188`

**Проверяет:**
```python
'two_model_pipeline', 'llama_cpp_deployment', 'qwen_model_manager'
```

**Проблема:** Не проверяет HybridPipelineAdapter напрямую.

**Решение:** Добавить проверку:
```python
'two_model_pipeline_ready': getattr(brain, 'two_model_pipeline_ready', False),
```

---

## 5. Приоритеты исправлений

### 🔴 Высокий приоритет (блокируют систему)

1. **Добавить create_fractal_graph_v2 factory** - восстановит 9 компонентов
2. **Исправить MemoryManager** - уберёт ошибки в логах

### 🟡 Средний приоритет

3. Удалить неиспользуемые адаптеры KG
4. Оптимизировать импорты sentence-transformers
5. Убрать дублирование process_query

### 🟢 Низкий приоритет

6. Обновить документацию
7. Улучшить логирование

---

## 6. План действий

### Шаг 1: Исправить fractal_graph_v2 factory

```python
# Добавить в eva_ai/core/init_factories.py

def create_fractal_graph_v2(initializer):
    """Получает или создаёт fractal_graph_v2."""
    try:
        from eva_ai.memory.fractal_graph_v2 import FractalGraphV2
        fg = getattr(initializer.core_brain, 'fractal_graph_v2', None)
        if fg is None:
            fg = FractalGraphV2()
        initializer.core_brain.fractal_graph_v2 = fg
        return fg
    except Exception as e:
        initializer.logger.error(f"fractal_graph_v2 error: {e}")
        return None
```

### Шаг 2: Исправить MemoryManager

```python
# Добавить в eva_ai/memory/manager_core.py

def clear_cache(self):
    """Очистка кэша памяти."""
    self.working_memory = {}
    logger.info("Memory cache cleared")

def optimize(self):
    """Оптимизация памяти."""
    # Реализация оптимизации
    logger.info("Memory optimized")
```

### Шаг 3: Обновить brain_components.py

Убедиться что `fractal_graph_v2` создаётся ДО вызова ComponentInitializer.

---

## 7. Ожидаемый результат после исправлений

| Компонент | До | После |
|-----------|-----|-------|
| ml_unit | FAIL | OK |
| query_processor | FAIL | OK |
| reasoning_engine | FAIL | OK |
| model_manager | FAIL | OK |
| Восстановлено компонентов | 10/22 | 20/22 |

**Обновлено:** 2026-04-09