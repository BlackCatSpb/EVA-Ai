# EVA AI System - Полный Анализ Компонентов и Проблем

**Дата:** 2026-03-30  
**Версия:** 2.0  
**Цель:** Аудит 29 компонентов системы, их связей, импортов и инициализации

---

## 1. СПИСОК 29 КОМПОНЕНТОВ СИСТЕМЫ

### 1.1 Компоненты из ComponentInitializer (26)

| # | Компонент | Фабрика | Файл | Статус |
|---|-----------|---------|------|--------|
| 1 | event_bus | create_event_bus | eva/core/event_bus.py | ✅ |
| 2 | resource_manager | create_resource_manager | eva/core/resource_manager.py | ✅ |
| 3 | config_manager | create_config_manager | eva/core/config_manager.py | ✅ |
| 4 | memory_manager | create_memory_manager | eva/memory/memory_manager.py | ✅ |
| 5 | hybrid_cache | create_hybrid_cache | eva/memory/hybrid_token_cache.py | ✅ |
| 6 | knowledge_graph | create_knowledge_graph | eva/knowledge/knowledge_graph_integrated.py | ✅ |
| 7 | qwen_api_enhancer | create_qwen_api_enhancer | eva/knowledge/qwen_api_enhancer.py | ✅ |
| 8 | text_processor | create_text_processor | eva/mlearning/unified_text_processor.py | ✅ |
| 9 | ml_unit | create_ml_unit | eva/mlearning/ml_unit.py | ✅ |
| 10 | model_manager | create_model_manager | eva/mlearning/hybrid_model_manager.py | ✅ |
| 11 | query_processor | create_query_processor | eva/core/query_processor.py | ✅ |
| 12 | response_generator | create_response_generator | eva/core/response_generator.py | ✅ |
| 13 | reasoning_engine | create_reasoning_engine | eva/core/reasoning_engine.py | ✅ |
| 14 | training_orchestrator | create_training_orchestrator | eva/mlearning/training_orchestrator.py | ✅ |
| 15 | learning_manager | create_learning_manager | eva/learning/learning_manager.py | ✅ |
| 16 | learning_scheduler | create_learning_scheduler | eva/core/learning_scheduler.py | ⚠️ |
| 17 | analytics_manager | create_analytics_manager | eva/analytics/analytics_manager.py | ✅ |
| 18 | system_monitor | create_system_monitor | eva/monitoring/system_monitor.py | ✅ |
| 19 | metrics_collector | create_metrics_collector | eva/core/metrics_collector.py | ✅ |
| 20 | contradiction_manager | create_contradiction_manager | eva/contradiction/contradiction_manager.py | ✅ |
| 21 | adaptation_manager | create_adaptation_manager | eva/adaptation/adaptation_core.py | ✅ |
| 22 | ethics_framework | create_ethics_framework | eva/ethics/ethics_framework.py | ✅ |
| 23 | web_search_engine | create_web_search_engine | eva/websearch/web_search_engine.py | ✅ |
| 24 | gui | create_gui | eva/gui/core_gui.py | ✅ |
| 25 | fractal_storage | create_fractal_storage | eva/reasoning/fractal_ml.py | ✅ |
| 26 | self_reasoning_engine | create_self_reasoning_engine | eva/reasoning/__init__.py | ✅ |

### 1.2 Компоненты, инициализируемые отдельно в CoreBrain (2)

| # | Компонент | Метод | Файл | Статус |
|---|-----------|-------|------|--------|
| 27 | background_coordinator | _initialize_background_coordinator | eva/core/background_coordinator.py | ✅ |
| 28 | generation_coordinator | _initialize_generation_coordinator | eva/core/generation_coordinator.py | ✅ |

### 1.3 Создаваемые динамически (1)

| # | Компонент | Где создается | Статус |
|---|-----------|--------------|--------|
| 29 | reasoning_integration | Внутри create_self_reasoning_engine | ✅ |

**ИТОГО: 29 компонентов**

---

## 2. КРИТИЧЕСКИЕ ПРОБЛЕМЫ

### 2.1 FractalAttentionSystem - Не подключен к CoreBrain

**Файл:** `eva/core/integration_layer.py:117`
```python
self.fractal_attention = FractalAttentionSystem(self.core_brain)
```

**Проблема:** FractalAttentionSystem создается в integration_layer, но НЕ регистрируется в brain.attention_system.

**Файл:** `eva/core/component_initializer.py:531-537`
```python
attention_system = getattr(self.core_brain, 'attention_system', None)
if attention_system is None:
    # ВСЕГДА попадает сюда!
    self.logger.warning("[WARN] attention_system не найден - используется DummyAttentionSystem")
```

**Влияние:** learning_scheduler использует DummyAttentionSystem вместо реального FractalAttentionSystem.

---

### 2.2 SystemOptimizer - Неверный параметр

**Файл:** `eva/core/fractal_attention_system.py:78`
```python
self.system_optimizer = SystemOptimizer(self)  # ОШИБКА: передает self (FractalAttentionSystem)
```

**Ожидается:** `SystemOptimizer(self.core_brain)`

**Проверка:** `eva/core/system_optimizer.py:21`
```python
def __init__(self, core_brain):  # Ожидает CoreBrain!
```

---

### 2.3 Event API - Неправильные методы

**Проблема:** GUI файлы используют `.on()` который не существует в EventSystem.

**Существующие методы в EventSystem:**
- `.subscribe()` - вместо `.on()`
- `.trigger()` - вместо `.emit()`
- `.unsubscribe()`

**Файлы с ошибками:**

| Файл | Линия | Проблемный код |
|------|-------|----------------|
| bridge.py | 43-47 | `self.brain.events.on('query_received', ...)` |
| core_gui.py | 635 | `self.brain.events.on('model_load', ...)` |
| core_gui.py | 639 | `self.brain.events.on('models_ready', ...)` |
| core_gui.py | 645 | `self.brain.events.on('request_gui_reload', ...)` |
| chat_module.py | 170 | `brain.events.on('model_load', ...)` |
| chat_module.py | 172 | `brain.events.on('models_ready', ...)` |

---

### 2.4 DummyAttentionSystem - Отсутствует атрибут

**Файл:** `eva/core/component_initializer.py:534-537`
```python
class DummyAttentionSystem:
    def __init__(self):
        self.pending_opportunities = []
    # ОТСУТСТВУЕТ: self.core_brain = None
```

**Влияние:** LearningScheduler обращается к `self.attention_system.core_brain`, что вызывает AttributeError.

---

## 3. ПРОВЕРКА ИМПОРТОВ

### 3.1 Проверенные и работающие

| Модуль | Импорт | Статус |
|--------|--------|--------|
| EventBus | `from eva.core.event_bus import EventBus` | ✅ |
| MemoryManager | `from eva.memory.memory_manager import MemoryManager` | ✅ |
| KnowledgeGraph | `from eva.knowledge.knowledge_graph_integrated import IntegratedKnowledgeGraph` | ✅ |
| HybridTokenCache | `from eva.memory.hybrid_token_cache import get_shared_cache` | ✅ |
| QwenAPIEnhancer | `from eva.knowledge.qwen_api_enhancer import QwenAPIEnhancer` | ✅ |
| TextProcessor | `from eva.mlearning.unified_text_processor import UnifiedTextProcessor` | ✅ |
| MLUnit | `from eva.mlearning.ml_unit import MLUnit` | ✅ |
| ModelManager | `from eva.mlearning.hybrid_model_manager import HybridModelManager` | ✅ |
| QueryProcessor | `from eva.core.query_processor import QueryProcessor` | ✅ |
| ResponseGenerator | `from eva.core.response_generator import ResponseGenerator` | ✅ |
| ReasoningEngine | `from eva.core.reasoning_engine import ReasoningEngine` | ✅ |
| TrainingOrchestrator | `from eva.mlearning.training_orchestrator import TrainingOrchestrator` | ✅ |
| LearningManager | `from eva.learning.learning_manager import LearningManager` | ✅ |
| LearningScheduler | `from eva.core.learning_scheduler import LearningScheduler` | ✅ |
| AnalyticsManager | `from eva.analytics.analytics_manager import AnalyticsManager` | ✅ |
| SystemMonitor | `from eva.monitoring.system_monitor import SystemMonitor` | ✅ |
| EthicsFramework | `from eva.ethics.ethics_framework import EthicsFramework` | ✅ |
| ЕВАGUI | `from eva.gui.core_gui import ЕВАGUI` | ✅ |
| WebSearchEngine | `from eva.websearch.web_search_engine import WebSearchEngine` | ✅ |
| FractalStorage | `from eva.reasoning.fractal_ml import FractalStorage` | ✅ |
| SelfReasoningEngine | `from eva.reasoning import SelfReasoningEngine` | ✅ |
| BackgroundCoordinator | `from eva.core.background_coordinator import BackgroundCoordinator` | ✅ |
| GenerationCoordinator | `from eva.core.generation_coordinator import initialize_generation_coordinator` | ✅ |

### 3.2 Потенциальные проблемы с импортами

| Модуль | Импорт | Проблема |
|--------|--------|-----------|
| MetricsCollector | `from eva.core.metrics_collector import MetricsCollector` | Fallback на AnalyticsManager |
| SystemOptimizer | `from eva.core.system_optimizer import SystemOptimizer` | Неправильный параметр при создании |

---

## 4. ЛОГИКА ИНИЦИАЛИЗАЦИИ

### 4.1 Порядок инициализации

```
CoreBrain.__init__()
    ↓
Initialize ComponentInitializer
    ↓
component_factories = {  # 26 фабрик зарегистрированы }
    ↓
initialize_components()
    ├─ event_bus (1)
    ├─ resource_manager (2)
    ├─ config_manager (3)
    ├─ memory_manager (4)
    ├─ hybrid_cache (5)
    ├─ knowledge_graph (6)
    ├─ qwen_api_enhancer (7)
    ├─ text_processor (8)
    ├─ ml_unit (9)
    ├─ model_manager (10)
    ├─ query_processor (11)
    ├─ response_generator (12)
    ├─ reasoning_engine (13)
    ├─ training_orchestrator (14)
    ├─ learning_manager (15)
    ├─ learning_scheduler (16) ← ПРОБЛЕМА: attention_system = None
    ├─ analytics_manager (17)
    ├─ system_monitor (18)
    ├─ metrics_collector (19)
    ├─ contradiction_manager (20)
    ├─ adaptation_manager (21)
    ├─ ethics_framework (22)
    ├─ web_search_engine (23)
    ├─ gui (24)
    ├─ fractal_storage (25)
    └─ self_reasoning_engine (26)  ← Также создает reasoning_integration
    ↓
_post_initialize_connections()
    ↓
background_coordinator инициализируется отдельно в core_brain._initialize_background_coordinator()
generation_coordinator инициализируется отдельно в core_brain._initialize_generation_coordinator()
    ↓
brain.start() - запуск компонентов
```

### 4.2 Зависимости компонентов

| Компонент | Зависит от | Статус |
|-----------|-----------|--------|
| memory_manager | event_bus | ✅ |
| hybrid_cache | - | ✅ |
| knowledge_graph | event_bus | ✅ |
| text_processor | hybrid_cache, memory_manager | ✅ |
| ml_unit | hybrid_cache | ✅ |
| model_manager | - | ✅ |
| query_processor | knowledge_graph, text_processor | ✅ |
| response_generator | model_manager, tokenizer | ✅ |
| reasoning_engine | knowledge_graph | ✅ |
| training_orchestrator | ml_unit, hybrid_cache | ⚠️ |
| learning_manager | memory_manager | ✅ |
| learning_scheduler | attention_system | ❌ ВСЕГДА Dummy |
| analytics_manager | memory_manager | ✅ |
| system_monitor | - | ✅ |
| contradiction_manager | knowledge_graph | ✅ |
| ethics_framework | knowledge_graph | ✅ |
| gui | brain, integrator | ⚠️ integrator = None |

---

## 5. ИНТЕГРАЦИОННЫЙ СЛОЙ

### 5.1 Интегратор (ЕВАIntegrator)

**Файл:** `eva/core/integration_layer.py`

Создает:
- FractalAttentionSystem (стр. 117) - НО НЕ регистрирует в brain!
- ResponseGenerator
- GenerationCoordinator
- MemoryManager
- KnowledgeGraph
- ReasoningEngine

**Проблема:** Интегратор создает FractalAttentionSystem, но:
1. Не присваивает его brain.attention_system
2. ComponentInitializer ищет attention_system в brain, но не находит

---

## 6. ИСПРАВЛЕНИЯ ПО ПРИОРИТЕТАМ

### ЭТАП 1 - Критические (Блокируют работу)

| # | Проблема | Файл | Линия | Исправление |
|---|----------|------|-------|-------------|
| 1 | SystemOptimizer неверный параметр | fractal_attention_system.py | 78 | `SystemOptimizer(self)` → `SystemOptimizer(self.core_brain)` |
| 2 | FractalAttentionSystem не в brain | integration_layer.py | 117 | Добавить: `self.core_brain.attention_system = self.fractal_attention` |
| 3 | .on() → .subscribe() | bridge.py | 43-47 | Заменить все `.on()` на `.subscribe()` |
| 4 | .on() → .subscribe() | core_gui.py | 635,639,645 | Заменить все `.on()` на `.subscribe()` |
| 5 | .on() → .subscribe() | chat_module.py | 170,172 | Заменить все `.on()` на `.subscribe()` |

### ЭТАП 2 - Важные (Влияют на функционал)

| # | Проблема | Файл | Линия | Исправление |
|---|----------|------|-------|-------------|
| 6 | DummyAttentionSystem без brain | component_initializer.py | 534-537 | Добавить `self.core_brain = None` в класс |
| 7 | Integrator не передается в GUI | start_webgui.py | 68 | Создать и передать integrator |
| 8 | Integrator не передается в GUI | eva/run.py | 48 | Создать и передать integrator |

### ЭТАП 3 - Оптимизация

| # | Проблема | Файл | Исправление |
|---|----------|------|-------------|
| 9 | LearningScheduler отсутствуют методы | learning_scheduler.py | Добавить: identify_learning_opportunities, schedule_learning_session, get_high_priority_opportunities |

---

## 7. ПРОВЕРКА ПОДКЛЮЧЕНИЙ МОДУЛЕЙ

### 7.1 CoreBrain → Components

```python
# component_initializer.py создает и регистрирует:
self.core_brain.components[component_name] = component
```

### 7.2 Component → Component

```
EventBus (публикует/подписывается)
    ↓
    ├─→ MemoryManager
    ├─→ KnowledgeGraph  
    ├─→ QueryProcessor
    └─→ ResponseGenerator

FractalAttentionSystem (должен быть в brain.attention_system)
    ↓
    ├─→ SelfDialogManager
    ├─→ ContradictionResolver
    ├─→ LearningScheduler (сейчас использует Dummy!)
    └─→ SystemOptimizer (получает неверный параметр!)
```

---

## 8. РЕЗЮМЕ

- **29 компонентов** в системе (26 фабрик + 2 из core_brain + 1 динамический)
- **5 критических проблем** блокируют полноценную работу
- **3 важных проблемы** влияют на функционал
- Все импорты существуют и работают
- Проблема в **логике инициализации и связях между компонентами**

---

**Конец отчета**
