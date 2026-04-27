# Перекрёстный анализ: FCP + Ethics + WebSearch

## Содержание

1. [Обзор интеграции](#1-обзор-интеграции)
2. [FCP интеграция с CoreBrain](#2-fcp-интеграция-с-corebrain)
3. [Ethics Framework интеграция](#3-ethics-framework-интеграция)
4. [WebSearch интеграция](#4-websearch-интеграция)
5. [Проблемы и конфликты](#5-проблемы-и-конфликты)
6. [Связанные отчёты](#6-связанные-отчёты)
7. [Рекомендации по интеграции](#7-рекомендации-по-интеграции)

---

## 1. Обзор интеграции

Данный отчёт анализирует перекрёстное взаимодействие трёх ключевых подсистем EVA:

- **FCP (Fractal Cognitive Processor)** — гибридная модель GNN + Transformer с KCA и SRG
- **Ethics Framework** — система этической оценки запросов и ответов
- **WebSearch** — система веб-поиска с интеграцией Tavily и FractalGraphV2

### Архитектура потоков данных

```
User Query → brain_query.process_query()
    ↓
┌─────────────────────────────────────────────┐
│  FCP Pipeline V15 (PRIMARY)                 │
│  - generate() → FCPPipelineV15.generate()  │
│  - НЕТ WebSearch                            │
│  - НЕТ Ethics check                        │
└─────────────────────────────────────────────┘
    ↓ Fallback если FCP недоступен
┌─────────────────────────────────────────────┐
│  Two-Model Pipeline (FALLBACK)              │
│  - needs_web_search() → WebSearch           │
│  - _check_ethics() → EthicsFramework        │
└─────────────────────────────────────────────┘
    ↓
Response
```

### Ключевые файлы

| Компонент | Файл инициализации | Файл использования |
|-----------|-------------------|-------------------|
| FCP | init_factories.py:775 create_fcp_pipeline | brain_query.py:327 _handle_gguf_pipeline |
| Ethics | init_factories.py:377 create_ethics_framework | pipeline_core.py:141 _check_ethics |
| WebSearch | init_factories.py:391 create_web_search_engine | brain_query.py:376 needs_web_search |

---

## 2. FCP интеграция с CoreBrain

### 2.1. Инициализация FCPPipelineV15

Источник: eva_ai/core/init_factories.py строки 775-827

```python
def create_fcp_pipeline(initializer):
    from eva_ai.core.fcp_pipeline import FCPPipelineV15
    
    config = initializer.core_brain.config.get(fcp_pipeline, {})
    model_path = config.get(model_path, ...)
    
    pipeline = FCPPipelineV15(
        model_path=model_path,
        graph_path=graph_path,
        lora_dir=lora_dir,
        enable_kca=enable_kca,
        enable_srg=enable_srg,
        enable_memory_snapshot=enable_memory
    )
    
    initializer.core_brain.fcp_pipeline = pipeline
    initializer.core_brain.components[fcp_pipeline] = pipeline
```

**Компоненты FCPPipelineV15:**

| Компонент | Описание | Файл |
|-----------|----------|------|
| **FCPConfig** | Конфигурация FCP (36 слоёв, 2560 hidden) | eva_ai/fcp_core/config.py |
| **SemanticRelevanceGate (SRG)** | Определение режима генерации | eva_ai/fcp_core/__init__.py |
| **KnowledgeConsciousAttention (KCA)** | Коррекция hidden states | eva_ai/fcp_core/__init__.py |
| **ConvergenceController** | Проверка сходимости KCA | eva_ai/fcp_core/__init__.py |
| **FractalGraphV2** | Векторный граф знаний | eva_ai/memory/fractal_graph_v2.py |
| **HybridLayerProcessor** | Обработка гибридных слоёв | eva_ai/fcp_gnn/hybrid_integration.py |
| **HybridLayerManager** | Менеджер слоёв | eva_ai/fcp_gnn/hybrid_integration.py |
| **MemorySnapshotIntegration** | Интеграция snapshotов | eva_ai/fcp_gnn/memory_snapshot_integration.py |
| **OpenVINO LLMPipeline** | Base LLM (Qwen3-4B) | openvino-genai |

### 2.2. Использование в brain_query

Источник: eva_ai/core/brain_query.py строки 327-361

```python
def _handle_gguf_pipeline(self, query, user_context, ...):
    # Проверяем FCP Pipeline V15 (основной)
    pipeline = getattr(self, fcp_pipeline, None)
    if pipeline and hasattr(pipeline, generate):
        response = pipeline.generate(
            query,
            max_new_tokens=max_new_tokens,
            enable_thinking=False,
            enable_injection=True,
            use_lora=True,
            conversation_history=conversation_history
        )
        
        return {
            response: response,
            status: success,
            source: fcp_pipeline_v15,
            ...
        }
    
    # Fallback на старый pipeline если FCP не доступен
    pipeline = getattr(self, two_model_pipeline, None)
```

### 2.3. Методы FCPPipelineV15

| Метод | Параметры | Возвращает | Назначение |
|-------|-----------|------------|------------|
| generate() | query, max_tokens, enable_thinking, enable_injection, use_lora, conversation_history | str | Основная генерация |
| generate_stream() | query, stream_callback, ... | Generator | Потоковая генерация |
| get_status() | - | Dict | Статус компонентов |
| get_fcp_components() | - | Dict | Компоненты FCP |

### 2.4. FCP в CoreBrain компонентах

Согласно eva_ai/core/init_core.py:
- fcp_pipeline добавляется в список компонентов
- Зависимости: closed_cognitive_loop → fractal_graph_v2, fcp_pipeline

---

## 3. Ethics Framework интеграция

### 3.1. Инициализация EthicsFramework

Источник: eva_ai/core/init_factories.py строки 377-388

```python
def create_ethics_framework(initializer):
    from eva_ai.ethics.ethics_framework import EthicsFramework
    
    event_bus = getattr(initializer.core_brain, event_bus, None)
    ethics_framework = EthicsFramework(
        brain=initializer.core_brain,
        event_bus=event_bus
    )
    
    initializer.core_brain.ethics_framework = ethics_framework
    return ethics_framework
```

### 3.2. Использование в RecursiveModelPipeline

Источник: eva_ai/core/pipeline_core.py строки 129-156

```python
class RecursiveModelPipeline:
    def __init__(self, ...):
        self.ethics_framework = None
        self._init_ethics_framework()
    
    def _check_ethics(self, query: str) -> Dict[str, Any]:
        if not self.ethics_framework:
            return {allowed: True, reason: Ethics not available}
        
        try:
            result = self.ethics_framework.assess_ethics({query: query})
            decision = result.decision if hasattr(result, decision) else allow
            is_high_risk = hasattr(result, requires_human_review) and result.requires_human_review
            return {
                allowed: decision != block,
                reason: result.justification if hasattr(result, justification) else ,
                risk_level: high if is_high_risk else low
            }
        except Exception as e:
            logger.error(fОшибка ethical check: {e})
            return {allowed: True, reason: fCheck error: {e}}
```

### 3.3. Ethics в других компонентах

| Компонент | Метод | Использование |
|-----------|-------|----------------|
| SelfReasoningEngine | self_reasoning_engine.py:882 | ethics.analyze_response(query, response) |
| ConceptMiner | concept_miner.py:779 | Проверка candidate.validation_ethics.get(risk_level) == high |
| IntegrationManager | integration_manager.py:426 | ethics.validate_ethics() |
| GUI Server | server_main.py:318 | Отображение этической оценки в UI |

### 3.4. Проблема: Ethics НЕ используется в FCP Pipeline

**КРИТИЧЕСКАЯ ПРОБЛЕМА:**

В brain_query.py метод _handle_gguf_pipeline:
- FCP Pipeline V15 вызывается напрямую (строки 340-347)
- НЕТ вызова _check_ethics() перед генерацией
- Ethics проверяется ТОЛЬКО в fallback pipeline (two_model_pipeline)

Это означает:
- Запросы через FCP не проходят этическую проверку
- Потенциально опасные запросы могут пройти без модерации
- Несоответствие поведения между primary и fallback pipeline

---

## 4. WebSearch интеграция

### 4.1. Инициализация WebSearch

Источник: eva_ai/core/init_factories.py строки 391-410

```python
def create_web_search_engine(initializer):
    from eva_ai.websearch.web_search_integrated import IntegratedWebSearchEngine
    
    web_search = IntegratedWebSearchEngine(brain=initializer.core_brain)
    
    if hasattr(web_search, initialize) and not web_search.is_initialized:
        web_search.initialize()
    
    initializer.core_brain.web_search_engine = web_search
    return web_search
```

### 4.2. Определение необходимости поиска

Источник: eva_ai/core/brain_query.py строки 21-82

```python
def needs_web_search(query: str) -> tuple[bool, str]:
    query_clean = re.sub(r[^\w\s], , query.lower().strip())
    query_lower = query.lower().strip()
    words = query_clean.split()
    
    # Приветствия - не нужен поиск
    greetings = [привет, здравствуй, hi, hello]
    if query_clean in greetings:
        return False, приветствие
    
    # Запросы о себе - не нужен поиск
    self_patterns = [кто ты, что ты, твоё имя]
    if any(p in query_lower for p in self_patterns):
        return False, запрос о себе
    
    # Математика/код - не нужен поиск
    math_patterns = [посчитай, вычисли, напиши код]
    if any(p in query_lower for p in math_patterns):
        return False, математика/код
    
    # Вопросы о текущих событиях - нужен поиск
    current_events_keywords = [2025, 2026, сейчас, недавно]
    if any(kw in query_lower for kw in current_events_keywords):
        return True, текущие события
    
    # Фактические вопросы - нужен поиск
    factual_patterns = [кто изобрел, кто создал, когда произошло]
    if any(p in query_lower for p in factual_patterns):
        return True, фактический вопрос
    
    # По умолчанию - поиск для обогащения
    return True, обогащение контекста
```

### 4.3. Использование в brain_query (FALLBACK ONLY)

Источник: eva_ai/core/brain_query.py строки 375-417

```python
# === WEB SEARCH для Two-Model Pipeline ===
web_search = getattr(self, web_search_engine, None)

search_results = []
if web_search and hasattr(web_search, search):
    need_search, search_reason = needs_web_search(query)
    
    if need_search:
        try:
            web_result = web_search.search(query, max_results=5)
            search_results = web_result.get(results, []) if web_result else []
        except Exception as e:
            query_logger.warning(fTavily search error: {e})

# Добавляем контекст от Tavily к запросу
enhanced_query = query

if search_results:
    web_context = \n\nДополнительная информация из интернета:\n
    for i, r in enumerate(search_results[:3], 1):
        web_context += f{i}. {r.get(title, )}: {r.get(content, )[:200]}...\n
    enhanced_query = enhanced_query + web_context
```

### 4.4. Интеграция с FractalGraphV2

WebSearchEngine имеет метод add_to_fractal_graph():
- Добавляет результаты поиска как узлы типа web_knowledge
- Интегрируется с Concept System через web_search_and_learn()

### 4.5. КРИТИЧЕСКАЯ ПРОБЛЕМА: WebSearch НЕ используется в FCP Pipeline

**Проблема:** В _handle_gguf_pipeline строки 327-361:
- FCP Pipeline V15 вызывается напрямую
- WebSearch вызывается ТОЛЬКО в fallback branch (two_model_pipeline)
- FCP Pipeline не имеет доступа к веб-контексту

Это приводит к:
- Запросы через FCP не обогащаются веб-информацией
- Устаревшие данные если нет в графе
- Неоптимальные ответы на фактические вопросы

---

## 5. Проблемы и конфликты

### 5.1. Критические проблемы

| # | Проблема | Файл | Строки | Влияние |
|---|----------|------|--------|---------|
| 5.1.1 | FCP Pipeline НЕ использует WebSearch | brain_query.py | 327-361 | Запросы без веб-контекста |
| 5.1.2 | FCP Pipeline НЕ использует Ethics | brain_query.py | 327-361 | Нет этической проверки |
| 5.1.3 | Два разных flow для primary/fallback | brain_query.py | 327-460 | Несоответствие поведения |
| 5.1.4 | Нет unified обработки в FCP | fcp_pipeline.py | - | Изолированная архитектура |

### 5.2. Средние проблемы

| # | Проблема | Файл | Описание |
|---|----------|------|----------|
| 5.2.1 | WebSearch fallback - заглушки | web_search_integrated.py:527-550 | Симулированные результаты при недоступности API |
| 5.2.2 | Ethics keyword-based оценка | framework_checks.py | Легко обойти, нет понимания контекста |
| 5.2.3 | Нет retry для WebSearch | - | Нет exponential backoff при таймауте |
| 5.2.4 | Дублирование кэширования | - | WebSearchEngine и IntegratedWebSearchEngine используют разные кэши |

### 5.3. Конфликты архитектуры

#### Конфликт 1: FCP vs Standard Pipeline

```
Primary Flow (FCP):
  Query → FCPPipelineV15.generate() → Response
            ↑
         НЕТ WebSearch
         НЕТ Ethics

Fallback Flow (Two-Model):
  Query → needs_web_search() → WebSearch → process_query() → Response
                                      ↓
                               _check_ethics() → Ethics
```

**Проблема:** Разные уровни функциональности для одного типа запроса.

#### Конфликт 2: WebSearch в FCP Pipeline

FCPPipelineV15 имеет:
- FractalGraphV2 для знаний
- KCA для коррекции
- LoRA адаптеры

НО:
- Нет метода для обогащения из веба
- Нет интеграции с needs_web_search()
- Нет вызова web_search_and_learn()

#### Конфликт 3: Ethics в FCP Pipeline

FCPPipelineV15 имеет:
- Self-dialog для концептов
- ContradictionMiner для противоречий

НО:
- Нет вызова ethics_framework.analyze_request()
- Нет вызова ethics_framework.analyze_response()
- Нет EventBus событий для этики

### 5.4. Пробелы в интеграции

| Компонент | Есть в FCP | Есть в Fallback | Пробел |
|-----------|------------|-----------------|--------|
| Concept Extraction | + (ConceptExtractor) | + | - |
| Graph Retrieval | + (FractalGraphV2) | + | - |
| WebSearch | - | + | FCP не использует веб |
| Ethics Check | - | + | FCP не проверяет этику |
| LoRA Adaptation | + | + | - |
| KCA Correction | + | - | Fallback не использует |
| SRG Mode Selection | + | - | Fallback не использует |

---

## 6. Связанные отчёты

### 6.1. fcp_system.md

Основные выводы:
- FCP представляет гибридную модель GNN + Transformer с 36 слоями
- KCA выполняет итеративную коррекцию hidden states
- SRG определяет режим генерации (direct/reasoning/variational)
- Множество заглушек и упрощённых реализаций
- **Неполная интеграция с EVA Core**

Проблемы из отчёта, релевантные этому анализу:
- FCP Training Pipeline не реализован
- GNN Encoder не обучен
- LoRA адаптеры не загружены в FCPPipelineV15

### 6.2. websearch_ethics_system.md

Основные выводы:
- WebSearch имеет 3 уровня архитектуры (SearchEngines → WebSearchEngine → IntegratedWebSearch)
- Ethics использует Mixins паттерн с 7 категориями принципов
- Обе системы имеют проблемы с заглушками

Проблемы из отчёта, релевантные этому анализу:
- WebSearch: заглушки при недоступности Tavily API
- Ethics: простая keyword-based оценка, легко обойти
- Интеграция: Ethics не проверяет внешние источники (веб-поиск)

---

## 7. Рекомендации по интеграции

### 7.1. Приоритет 1 — Исправление критических проблем

#### 7.1.1. Добавить WebSearch в FCP Pipeline

**Вариант A: В FCPPipelineV15.generate()**
```python
def generate(self, query, ...):
    # Проверяем нужен ли веб-поиск
    from eva_ai.core.brain_query import needs_web_search
    need_search, reason = needs_web_search(query)
    
    search_results = []
    if need_search and self.web_search_engine:
        search_results = self.web_search_engine.search(query, max_results=5)
    
    # Обогащаем запрос
    enhanced_query = query
    if search_results:
        enhanced_query = self._enrich_with_web_context(query, search_results)
    
    # Генерируем
    response = self.pipeline.generate(enhanced_query, ...)
    return response
```

**Вариант B: В brain_query перед вызовом FCP**
```python
# В _handle_gguf_pipeline
pipeline = getattr(self, fcp_pipeline, None)

# Добавляем веб-контекст для FCP
if pipeline and hasattr(pipeline, web_search_engine):
    need_search, _ = needs_web_search(query)
    if need_search:
        search_results = self.web_search_engine.search(query, max_results=3)
        query = self._enrich_query_with_web(query, search_results)
```

#### 7.1.2. Добавить Ethics в FCP Pipeline

**Вариант A: Pre-generation check**
```python
# В _handle_gguf_pipeline перед вызовом FCP
if self.ethics_framework:
    ethics_result = self.ethics_framework.analyze_request(query)
    if not ethics_result.get(approved, True):
        # Блокируем или модифицируем запрос
        query = self._modify_query_for_ethics(query, ethics_result)
```

**Вариант B: Post-generation check**
```python
# После генерации FCP
response = pipeline.generate(query, ...)

if self.ethics_framework:
    ethics_result = self.ethics_framework.analyze_response(query, response)
    if ethics_result.get(violations):
        # Перегенерация или предупреждение
        response = self._regenerate_without_violations(query, ethics_result)
```

#### 7.1.3. Унифицировать flow

Создать единый метод обработки для обоих pipeline:
```python
def _process_with_enrichments(self, query, pipeline_type=fcp):
    # 1. Ethics check
    if self.ethics_framework:
        ethics_result = self._check_ethics(query)
        if not ethics_result[allowed]:
            return self._handle_blocked_query(query, ethics_result)
    
    # 2. WebSearch
    if pipeline_type == fcp and hasattr(self, fcp_pipeline):
        search_results = self._get_web_context(query)
        query = self._enrich_query(query, search_results)
    
    # 3. Generation
    if pipeline_type == fcp:
        return self.fcp_pipeline.generate(query, ...)
    else:
        return self.two_model_pipeline.process_query(query, ...)
```

### 7.2. Приоритет 2 — Улучшение интеграции

#### 7.2.1. EventBus интеграция для FCP

Добавить публикацию событий в FCPPipelineV15:
```python
def generate(self, query, ...):
    # Публикуем событие начала генерации
    self.event_bus.publish(fcp.generation.start, {
        query: query,
        timestamp: time.time()
    })
    
    response = self.pipeline.generate(...)
    
    # Публикуем событие завершения
    self.event_bus.publish(fcp.generation.complete, {
        response: response,
        source: fcp_pipeline_v15
    })
    
    return response
```

#### 7.2.2. Улучшить WebSearch fallback

Заменить заглушки на реальные fallback движки:
```python
def _basic_web_search(self, query, search_config):
    # Пробуем DuckDuckGo
    results = self._search_duckduckgo(query)
    if results:
        return results
    
    # Пробуем Searx
    results = self._search_searx(query)
    if results:
        return results
    
    # Пробуем Wikipedia
    results = self._search_wikipedia(query)
    if results:
        return results
    
    return []  # Не симулированные результаты!
```

#### 7.2.3. Улучшить Ethics оценку

Интегрировать NLI из Concepts System:
```python
def _evaluate_safety(self, request):
    # Текущее: keyword matching
    dangerous_keywords = [убить, навредить, ...]
    
    # Улучшенное: NLI проверка
    if self.nli_model:
        result = self.nli_model.classify(
            premise=request,
            hypothesis=Это безопасный запрос
        )
        if result[contradiction] > 0.8:
            score += 0.4
    
    return score
```

### 7.3. Приоритет 3 — Архитектурные изменения

#### 7.3.1. Создать UnifiedPipelineInterface

```python
class UnifiedPipelineInterface:
    def __init__(self, fcp_pipeline, two_model_pipeline, 
                 web_search_engine, ethics_framework):
        self.fcp = fcp_pipeline
        self.legacy = two_model_pipeline
        self.web_search = web_search_engine
        self.ethics = ethics_framework
    
    def generate(self, query, ...):
        # Всегда: Ethics check
        if not self._check_ethics(query):
            return self._blocked_response()
        
        # Всегда: WebSearch если нужно
        query = self._enrich_with_web(query)
        
        # Выбор pipeline
        if self.fcp:
            return self.fcp.generate(query, ...)
        else:
            return self.legacy.process_query(query, ...)
```

#### 7.3.2. Добавить Pipeline Router

```python
def _select_pipeline(self, query):
    # FCP для сложных запросов с концептами
    if self._has_complex_concepts(query) or self._needs_reasoning(query):
        return fcp
    
    # Two-model для простых запросов с веб-поиском
    if needs_web_search(query):
        return legacy
    
    # По умолчанию FCP
    return fcp
```

---

## Заключение

Анализ выявил **критические пробелы в интеграции** FCP, Ethics и WebSearch с CoreBrain:

1. **FCP Pipeline изолирован** от WebSearch и Ethics — работает как независимая система
2. **Fallback pipeline (two_model_pipeline)** имеет полную функциональность, но является устаревшим
3. **Два разных flow** для обработки запросов создают несоответствие в поведении системы
4. **Нет unified интерфейса** для выбора и использования pipeline

Для достижения консистентности системы необходимо:
1. Интегрировать WebSearch в FCP Pipeline
2. Интегрировать Ethics check в FCP Pipeline  
3. Создать единый интерфейс для обоих pipeline
4. Улучшить fallback механизмы в WebSearch
5. Заменить keyword-based Ethics на NLI-based

---

*Дата анализа: 2026-04-27*
*Связанные отчёты: fcp_system.md, websearch_ethics_system.md*
*Основные файлы: brain_query.py, init_factories.py, fcp_pipeline.py, pipeline_core.py*