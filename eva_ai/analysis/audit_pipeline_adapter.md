# Аудит PipelineAdapter и HybridPipelineAdapter

**Дата:** 2026-04-14  
**Аудитор:** EVA AI System  
**Файлы:** eva_ai/core/pipeline_adapter.py, eva_ai/core/hybrid_pipeline_adapter.py, eva_ai/core/unified_generator.py, eva_ai/core/model_access_manager.py, eva_ai/memory/fractal_graph_v2/dual_generator.py

---

## 1. Архитектура Pipeline

### 1.1 Иерархия компонентов

```
brain.two_model_pipeline
|
+- PipelineAdapter (обёртка UnifiedGenerator)
|   +- UnifiedGenerator
|   |   +- ModelAccessManager
|   |   +- _openvino_cpu (OpenVINOGenerator)
|   |   +- _openvino_gpu (OpenVINOGenerator)
|   |   +- models[LOGIC/CONTEXT/CODER] (llama.cpp)
|   |   +- SimpleRouter
|   |
|   +- HybridPipelineAdapter (альтернативная реализация)
|       +- fractal_pipeline (FractalPipeline)
|       +- dual_generator (DualGenerator)
|       +- recursive_pipeline (RecursiveModelPipeline)
```

### 1.2 Ключевые компоненты

| Компонент | Файл | Назначение |
|----------|------|------------|
| PipelineAdapter | pipeline_adapter.py | Адаптер UnifiedGenerator для совместимости с TwoModelPipeline |
| HybridPipelineAdapter | hybrid_pipeline_adapter.py | Гибридный адаптер с поддержкой 3 режимов: fractal, dual, recursive |
| UnifiedGenerator | unified_generator.py | Центральный генератор с 3 моделями (LOGIC/CONTEXT/CODER) |
| ModelAccessManager | model_access_manager.py | Менеджер очереди доступа к модели с приоритетами |
| DualGenerator | fractal_graph_v2/dual_generator.py | Физически разделённые CondensedGenerator и ExtendedGenerator |

---

## 2. Интеграция PipelineAdapter с UnifiedGenerator

### 2.1 Инициализация

PipelineAdapter создаётся через create_pipeline_adapter() в brain_components.py:_init_unified_generator():

```python
adapter = create_pipeline_adapter(
    logic_model_path=logic_model_path,
    context_model_path=context_model_path,
    coder_model_path=coder_model_path,
    n_ctx=n_ctx,
    n_threads=n_threads,
    fractal_graph=getattr(brain, fractal_graph_v2, None),
    brain=brain,
    use_openvino=use_openvino,
    cpu_device=cpu_device,
    gpu_device=gpu_device
)
```

### 2.2 Связь с UnifiedGenerator

- PipelineAdapter принимает UnifiedGenerator в конструкторе (self._generator = unified_generator)
- Проксирует вызовы process_query(), generate(), generate_streaming(), generate_with_context() -> UnifiedGenerator
- Держит ссылки model_a, model_b, model_c для совместимости со старым TwoModelPipeline

### 2.3 generate_iterative() в UnifiedGenerator

```python
def generate_iterative(self, query, context, max_tokens_logic, max_tokens_context, ...):
    # Этап 1: LOGIC - краткий ответ
    logic_text = logic_model(logic_prompt, max_tokens=max_tokens_logic)
    
    # Этап 2: CONTEXT - расширение с концептами и противоречиями
    enriched_context = full_context
    if check_concepts:
        concepts_context = self._get_concepts_context(query)
        enriched_context += f"\n\nСвязанные концепты: {concepts_context}"
    if check_contradictions:
        contradictions_context = self._get_contradictions_context(query)
        enriched_context += f"\n\nИзвестные противоречия: {contradictions_context}"
    
    final_text = context_model(combined_prompt, max_tokens=max_tokens_context)
```

**Вызовы:**  
- PipelineAdapter.process_query() -> UnifiedGenerator.generate_iterative() (строка 77)
- DialogConceptsMixin -> прямое использование generate_iterative()

---

## 3. Интеграция с ModelAccessManager (MAM)

### 3.1 Архитектура MAM

```python
class ModelAccessManager:
    # Singleton с thread-safe инициализацией
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, event_bus=None, max_workers=4):
        self.request_queue = queue.PriorityQueue()  # Приоритетная очередь
        self._access_lock = threading.RLock()       # Блокировка модели
        self._model_busy = False
```

### 3.2 Приоритеты доступа

```python
class AccessPriority(Enum):
    CRITICAL = 0   # Пользовательские запросы
    HIGH = 1       # Самодиалог, концепты, противоречия
    NORMAL = 2     # Фоновые задачи
    LOW = 3        # Долгосрочное обучение
```

### 3.3 Интеграция в UnifiedGenerator

**Инициализация:**
```python
def _init_model_access_manager(self):
    self._model_access = ModelAccessManager(event_bus=self.event_bus, max_workers=4)
    self._model_access.start()
```

**Использование в generate():**
```python
if self._model_access is not None:
    priority = self._get_priority_for_task(task_type)
    request_id = self._model_access.request_access(
        priority=priority,
        task_type=task_type,
        callback=self._do_generate,
        query=query, context=context, ...
        timeout=60.0
    )
    return self._model_access.get_result(request_id, timeout=60.0)
```

**Приоритеты задач:**
```python
def _get_priority_for_task(self, task_type):
    priority_map = {
        query: AccessPriority.CRITICAL,
        self_dialog: AccessPriority.HIGH,
        concept_mining: AccessPriority.HIGH,
        contradiction_mining: AccessPriority.HIGH,
        coder: AccessPriority.HIGH,
        default: AccessPriority.NORMAL
    }
```

### 3.4 EventBus интеграция

```python
def _subscribe_to_events(self):
    self.event_bus.subscribe("model.request", ...)
    self.event_bus.subscribe("model.release", ...)
    self.event_bus.subscribe("model.status", ...)
```

---

## 4. HybridPipelineAdapter vs PipelineAdapter

### 4.1 HybridPipelineAdapter режимы

| Режим | Описание | Pipeline |
|-------|----------|----------|
| MODE_FRACTAL | FractalPipeline с виртуальными токенами | FractalPipeline |
| MODE_DUAL | DualGenerator с 2 физическими моделями | DualGenerator |
| MODE_RECURSIVE | RecursiveModelPipeline (старый) | RecursiveModelPipeline |
| MODE_HYBRID | FractalPipeline + fallback | FractalPipeline -> RecursiveModelPipeline |

### 4.2 Ключевые проблемы интеграции

#### Проблема 1: Две разные архитектуры

HybridPipelineAdapter использует llama.cpp модели напрямую:
```python
self.model_a = Llama(model_path=self.model_a_path, ...)
```

PipelineAdapter + UnifiedGenerator использует OpenVINO или llama.cpp через MAM:
```python
self._openvino_cpu = OpenVINOGenerator(model_path=logic_model, device=self.cpu_device, ...)
# или
self.models[model_type] = Llama(model_path=str(path), ...)
```

#### Проблема 2: DualGenerator - это FractalGraph V2 компонент

DualGenerator находится в eva_ai/memory/fractal_graph_v2/dual_generator.py и НЕ использует MAM.

#### Проблема 3: Нет унифицированного доступа к MAM

- UnifiedGenerator использует MAM
- HybridPipelineAdapter.MODE_DUAL -> DualGenerator - НЕ использует MAM
- HybridPipelineAdapter.MODE_FRACTAL -> FractalPipeline - НЕ использует MAM

---

## 5. generate_iterative() - Детальный анализ

### 5.1 Процесс выполнения

```
generate_iterative(query)
|
+-1-> _build_context() - собирает контекст из:
|    |  * FractalGraph (semantic_search)
|    |  * HybridTokenCache (ram_cache)
|    |  * Session history
|    |  * _build_concept_contradiction_context() <- концепты и противоречия
|    |
|    +-> Mam.request_access(CRITICAL, default, callback=_do_generate_iterative)
|        |
|        +-> _do_generate_iterative()
|            |
|            +-2-> LOGIC model -> краткий ответ
|            |   |  prompt = _format_prompt(query, full_context, system, LOGIC)
|            |   |  max_tokens = max_tokens_logic (default 4096!)
|            |   |  temperature = 0.7
|            |   |
|            |   +-> result = logic_model(prompt, ...)
|            |
|            +-3-> _get_concepts_context() - поиск концептов в FGv2
|            |   +-> search_nodes(node_type=concept, limit=5)
|            |
|            +-4-> _get_contradictions_context() - поиск противоречий в FGv2
|            |   +-> search_nodes(node_type=contradiction, limit=3)
|            |
|            +-5-> _get_web_search_context() - веб-поиск если нужен
|            |
|            +-6-> CONTEXT model -> развёрнутый ответ
|                |  prompt = "Запрос: {query}\n\nКраткий ответ: {logic_text}"
|                |  context = enriched_context (концепты + противоречия)
|                |  max_tokens = max_tokens_context (default 4096!)
|                |
|                +-> result = context_model(combined_prompt, ...)
```

### 5.2 Проблемы generate_iterative()

| Проблема | Серьёзность | Описание |
|----------|-------------|----------|
| Огромные max_tokens по умолчанию | Критическая | max_tokens_logic=4096, max_tokens_context=4096 - это 8192 токенов суммарно! |
| MAM не используется в _do_generate_iterative | Высокая | Callback _do_generate_iterative выполняется синхронно, без отдельного приоритета |
| Контекст собирается 2 раза | Средняя | _build_context() вызывается ДО MAM.request_access, затем снова в _do_generate_iterative |
| Нет кэширования результатов LOGIC | Средняя | Краткий ответ LOGIC не кэшируется для повторного использования |

---

## 6. Оценка интеграции по 10-балльной шкале

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Унификация интерфейса | 6/10 | PipelineAdapter и HybridPipelineAdapter имеют разные API и архитектуры |
| Интеграция с MAM | 5/10 | Только UnifiedGenerator использует MAM; HybridPipelineAdapter - нет |
| Управление приоритетами | 7/10 | MAM реализован хорошо, но используется только в UnifiedGenerator |
| Обработка ошибок | 6/10 | Есть fallback-цепочки, но нет чёткой стратегии восстановления |
| Производительность | 5/10 | generate_iterative с 8192 max_tokens - потенциальная проблема |
| Прозрачность для brain | 7/10 | brain.two_model_pipeline скрывает различия реализаций |
| Согласованность кэширования | 6/10 | Контекст собирается повторно, нет unified cache |

**ИТОГО: 6/10**

---

## 7. Конкретные рекомендации

### 7.1 Критические исправления

#### 1. Исправить max_tokens в generate_iterative()

**Проблема:** max_tokens_logic=4096 и max_tokens_context=4096 по умолчанию.

**Решение:**
```python
def generate_iterative(
    self,
    query: str,
    context: Optional[str] = None,
    max_tokens_logic: int = 256,   # Было: 4096
    max_tokens_context: int = 1024, # Было: 4096
    ...
)
```

#### 2. Интегрировать MAM в HybridPipelineAdapter.MODE_DUAL

**Проблема:** DualGenerator не использует MAM.

**Решение:** Создать обёртку DualGeneratorMAM, которая использует MAM для доступа к модели.

### 7.2 Высокоприоритетные исправления

#### 3. Устранить дублирование контекста

**Проблема:** _build_context() вызывается 2 раза.

**Решение:** Передавать предварительно собранный контекст в MAM callback.

#### 4. Кэшировать ответ LOGIC

**Проблема:** Краткий ответ не сохраняется для повторного использования.

**Решение:** Сохранять в hybrid_cache.

### 7.3 Среднеприоритетные исправления

#### 5. Унифицировать интерфейс HybridPipelineAdapter и PipelineAdapter

Оба адаптера должны реализовывать общий интерфейс.

#### 6. Добавить метрики в MAM

Расширить get_stats() для мониторинга производительности.

---

## 8. Выводы

### 8.1 Что хорошо

1. ModelAccessManager - хорошо спроектированный singleton с приоритетной очередью
2. UnifiedGenerator - централизованная точка входа для всех типов генерации
3. SimpleRouter - гибкий роутинг между LOGIC/CONTEXT/CODER моделями
4. Fallback цепочки - есть запасные варианты при сбоях

### 8.2 Что нужно исправить

1. max_tokens в generate_iterative() - завышены в 16-32 раза
2. MAM не используется в HybridPipelineAdapter - DualGenerator и FractalPipeline вне системы приоритетов
3. Двойной сбор контекста - неэффективно
4. Нет unified cache - концепты и противоречия извлекаются повторно

### 8.3 Общая оценка

> Интеграция PipelineAdapter с UnifiedGenerator работает, но есть существенные проблемы с производительностью и консистентностью. HybridPipelineAdapter представляет собой альтернативную реализацию, которая не использует MAM и потому может конфликтовать с UnifiedGenerator при одновременном доступе к модели.

---

*Отчёт сгенерирован EVA AI System Audit Module*
