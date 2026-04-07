# Архитектура системы CogniFlex (EVA AI)

Документация описывает архитектуру, зависимости и порядок инициализации компонентов системы CogniFlex (EVA AI).

## Содержание

1. [Общее описание](#общее-описание)
2. [Карта импортов](#карта-импортов)
3. [Карта инициализаций](#карта-инициализаций)
4. [Карта зависимостей](#карта-зависимостей)
5. [Используемые компоненты](#используемые-компоненты)
6. [Отключённые компоненты](#отключённые-компоненты)

---

## Общее описание

Система CogniFlex (EVA AI) представляет собой комплексную систему искусственного интеллекта с следующими основными модулями:

- **eva.core** - ядро системы, координация компонентов
- **eva.memory** - управление памятью и кэширование  
- **eva.mlearning** - машинное обучение и модели
- **eva.knowledge** - граф знаний и куратор
- **eva.generation** - генерация ответов

---

## Карта импортов

Ниже представлена схема импортов между основными модулями системы.

### eva.core -> eva.memory

- brain_components.py: from eva.memory.hybrid_token_cache import get_shared_cache
- brain_components.py: from eva.memory.unified_fractal_memory import UnifiedFractalMemory
- brain_components.py: from eva.memory.memory_manager import MemoryManager
- init_factories.py: from eva.memory.memory_manager import MemoryManager
- init_factories.py: from eva.memory.hybrid_token_cache import get_shared_cache
- brain_init.py: from eva.memory.fractal_graph_v2 import FractalMemoryGraph

### eva.core -> eva.mlearning

- brain_components.py: from eva.mlearning.fractal_model_manager import FractalModelManager
- brain_components.py: from eva.mlearning.language_filter import ModelModeController
- brain_components.py: from eva.mlearning.hot_deployment.llama_cpp_hot import LlamaCppHotDeployment
- init_factories.py: from eva.mlearning.unified_text_processor import UnifiedTextProcessor
- init_factories.py: from eva.mlearning.ml_unit import MLUnit
- init_factories.py: from eva.mlearning.hybrid_model_manager import HybridModelManager

### eva.core -> eva.knowledge

- brain_components.py: from eva.knowledge.online_knowledge import OnlineKnowledgeAccess
- brain_init.py: from eva.knowledge.wikipedia_kb import get_wikipedia_kb
- brain_init.py: from eva.knowledge.wikipedia_loader import get_wikipedia_loader
- brain_init.py: from eva.knowledge.graph_curator import GraphCurator
- init_factories.py: from eva.knowledge.knowledge_graph_integrated import IntegratedKnowledgeGraph

### eva.generation -> eva.memory

- eva/generation/__init__.py: from eva.memory.hybrid_token_cache import HybridTokenCache
- eva/generation/__init__.py: from eva.memory.disk_cache import DiskCache

### eva.mlearning -> eva.memory

- current_manager.py: from eva.memory.hybrid_token_cache import HybridTokenCache
- storage/opt_models.py: from eva.memory.hybrid_token_cache import HybridTokenCache
- async_text_generator.py: from eva.memory.disk_cache import DiskCache

### eva.knowledge -> eva.memory

- knowledge_hybrid_index.py: from eva.memory.disk_cache import DiskCache

### eva.core -> eva.core

- core_brain.py: from eva.core.event_system import EventSystem
- core_brain.py: from eva.core.event_bus import EventBus, get_event_bus
- core_brain.py: from eva.core.deferred_command_system import DeferredCommandSystem
- core_brain.py: from eva.core.fractal_attention_system import FractalAttentionSystem
- brain_components.py: from eva.core.recursive_model_pipeline import RecursiveModelPipeline
- brain_init.py: from eva.generation.generation_coordinator import initialize_generation_coordinator
- brain_init.py: from eva.reasoning.integration import ReasoningIntegration

### Основная схема импортов

eva.core
    |-- eva.memory (MemoryManager, HybridTokenCache, FractalGraphV2)
    |-- eva.mlearning (FractalModelManager, ModelManager, TextProcessor)
    |-- eva.knowledge (KnowledgeGraph, GraphCurator)
    +-- eva.core (EventBus, DeferredSystem)

eva.generation  
    +-- eva.memory (HybridTokenCache, DiskCache)

eva.mlearning
    +-- eva.memory (HybridTokenCache, DiskCache)

eva.knowledge
    +-- eva.memory (DiskCache)

---

## Карта инициализаций

Порядок инициализации компонентов системы.

### Этап 1: Базовые системы (CoreBrain.__init__)

1. Загрузка конфигурации: config = self._load_brain_config()
2. Инициализация EventSystem и EventBus
3. Инициализация DeferredCommandSystem
4. Инициализация менеджеров (_init_managers):
   - ConfigManager
   - SystemStateManager  
   - ResourceManager
   - SelfAnalyzer
   - SystemMetricsManager
5. Инициализация fractal модели (_init_fractal_model): FractalModelManager
6. Инициализация LlamaCpp (_init_llama_cpp): LlamaCppHotDeployment
7. Инициализация Two-Model Pipeline (_init_two_model_pipeline): RecursiveModelPipeline
8. Инициализация препроцессинга (_init_preprocessing): PreprocessingPipeline
9. Инициализация Qwen конфига (_init_qwen_config)
10. Инициализация фона (_init_background): BackgroundCoordinator
11. Инициализация контроллера режимов (_init_mode_controller): ModelModeController
12. Инициализация MemoryManager: self._init_memory_manager()

### Этап 2: CoreBrain.initialize()

1. Подписка на системные события
2. Запуск мониторинга ресурсов
3. Инициализация FractalAttentionSystem
4. Инициализация компонентов через ComponentInitializer
5. Инициализация MemoryManager
6. Соединение компонентов (_connect_components)
7. Финализация fractal модели (_init_fractal_final)
8. Инициализация генерационного координатора (_init_gen_coord)
9. Инициализация Wikipedia (_init_wikipedia)
10. Инициализация reasoning (_init_reasoning)

### Этап 3: Post-Init Services (_start_post_init_services)

1. Запуск SelfDialogLearningSystem
2. Инициализация FractalGraphV2
3. Запуск GraphCurator
4. Инициализация GGUFTrainingSystem

### Этап 4: Запуск компонентов (CoreBrain.start())

1. Запуск всех компонентов с методом start()
2. Запуск GUI
3. Запуск BackgroundCoordinator

### Порядок создания компонентов в ComponentInitializer

1. event_bus - Шина событий
2. resource_manager - Управление ресурсами
3. config_manager - Управление конфигурацией
4. memory_manager - Менеджер памяти
5. hybrid_cache - Гибридный кэш
6. knowledge_graph - Граф знаний
7. qwen_api_enhancer - Qwen API обогащение
8. text_processor - Обработка текста
9. ml_unit - ML юнит
10. model_manager - Менеджер моделей
11. query_processor - Процессор запросов
12. response_generator - Генератор ответов
13. reasoning_engine - Движок рассуждений
14. analytics_manager - Менеджер аналитики
15. system_monitor - Системный монитор
16. metrics_collector - Сборщик метрик
17. contradiction_manager - Менеджер противоречий
18. adaptation_manager - Менеджер адаптации
19. ethics_framework - Этический фреймворк
20. web_search_engine - Поисковый движок
21. fractal_storage - Фрактальное хранилище
22. self_reasoning_engine - Движок саморассуждений
23. enhanced_reasoning_engine - Улучшенный движок рассуждений

---

## Карта зависимостей

Зависимости между компонентами системы.

### Основные зависимости

CoreBrain (КООРДИНАТОР)
  - event_bus, deferred_system, resource_manager, metrics_manager
         |
         +-------------------+-------------------+--------------------+
         |                   |                   |                    |
         v                   v                   v                    v
   MemoryManager      KnowledgeGraph       ModelManager       TextProcessor
   requires:          requires:           requires:          requires:
   - event_bus        - event_bus         - ml_unit          - hybrid_cache
   - deferred_sys     - memory_manager    - hybrid_cache     - model_manager
   - hybrid_cache    - hybrid_cache
                      |
                 +----+----+
                 v         v
           HybridTokenCache  GraphCurator
           requires:         requires:
           - memory_manager  - event_bus
                            - deferred_system
                            - knowledge_graph

### Детальные зависимости

| Компонент | Требует | Обеспечивает |
|-----------|---------|---------------|
| memory_manager | event_bus, deferred_system, hybrid_cache | Управление памятью |
| hybrid_cache | memory_manager | Кэширование токенов |
| text_processor | hybrid_cache, model_manager | Обработка текста |
| query_processor | text_processor, model_manager, knowledge_graph, hybrid_cache | Обработка запросов |
| response_generator | query_processor, model_manager, text_processor | Генерация ответов |
| knowledge_graph | event_bus, memory_manager | Граф знаний |
| graph_curator | event_bus, deferred_system, knowledge_graph/fractal_graph_v2 | Куратор графа |
| model_manager | ml_unit, hybrid_cache | Управление моделями |
| ml_unit | hybrid_cache | ML операции |
| reasoning_engine | knowledge_graph, memory_manager | Рассуждения |

### Обратные зависимости

HybridTokenCache обеспечивает кэширование для:
    - text_processor
    - query_processor  
    - ml_unit
    - model_manager

KnowledgeGraph обеспечивает знания для:
    - query_processor
    - reasoning_engine

MemoryManager обеспечивает память для:
    - KnowledgeGraph
    - ReasoningEngine

---

## Используемые компоненты

Следующие компоненты активно используются в текущей версии системы.

### 1. FractalGraphV2 (eva.memory.fractal_graph_v2)

Описание: Новый фрактальный граф памяти с поддержкой:
- Векторного поиска и эмбеддингов
- Семантической кластеризации
- Проверки противоречий
- Самодиалога для верификации знаний

Основные классы:
- FractalMemoryGraph - главный API
- FractalGraphV2 - хранилище
- EmbeddingsManager - управление эмбеддингами

Использование:
from eva.memory.fractal_graph_v2 import FractalMemoryGraph

fg = FractalMemoryGraph(storage_dir=..., embedding_device=cuda)
node = fg.add_node(content=Test, node_type=concept)
results = fg.semantic_search(query, top_k=5)
fg.save_experience(query, response, model_used, quality_score)

### 2. GraphCurator (eva.knowledge.graph_curator)

Описание: Координатор графа знаний с функциями:
- Фоновая работа с адаптивным интервалом
- Подключение к event bus и deferred commands
- Метрики качества графа
- Извлечение знаний из GGUF моделей
- Чистка и ре-кластеризация

Основные классы:
- GraphCurator - главный класс
- CuratorState - состояния куратора
- CuratorMetrics - метрики

Зависимости:
- event_bus - для подписки/публикации событий
- deferred_system - для отложенных команд
- fractal_graph_v2 или knowledge_graph - граф для курации

### 3. KnowledgeGraph (eva.knowledge)

Описание: Граф знаний с возможностями:
- Хранение сущностей и связей
- Семантический поиск
- Интеграция с Wikipedia
- Онтологии и таксономии

Реализации:
- IntegratedKnowledgeGraph - основная реализация
- KnowledgeGraph - классическая реализация

### 4. HybridTokenCache (eva.memory.hybrid_token_cache)

Описание: Гибридный кэш токенов с:
- LRU кэшированием в памяти
- Redis/дисковое кэширование
- Системой приоритетов
- Синхронизацией с event bus

Основные классы:
- HybridTokenCache - основной кэш
- get_shared_cache() - получение синглтона

### 5. MemoryManager (eva.memory.memory_manager)

Описание: Менеджер памяти с типами:
- Working memory (оперативная)
- Semantic memory (семантическая)  
- Episodic memory (эпизодическая)
- User profiles (профили пользователей)

Основные классы:
- MemoryManager - главный класс

Зависимости:
- event_bus - для событий
- deferred_system - для отложенных команд
- hybrid_cache - для кэширования

### 6. ComponentInitializer (eva.core.component_initializer)

Описание: Единый инициализатор всех компонентов с:
- Фабриками для создания компонентов
- Управлением зависимостями
- Валидацией инициализации
- Метриками здоровья

Модули:
- init_core.py - основной класс
- init_factories.py - фабрики компонентов
- init_connections.py - связи зависимостей
- init_validation.py - валидация

### 7. Дополнительные активные компоненты

| Компонент | Модуль | Описание |
|-----------|--------|----------|
| EventBus | eva.core.event_bus | Шина событий |
| DeferredCommandSystem | eva.core.deferred_command_system | Отложенные команды |
| ResourceManager | eva.core.resource_manager | Управление ресурсами |
| FractalModelManager | eva.mlearning.fractal_model_manager | Управление fractal моделью |
| HybridModelManager | eva.mlearning.hybrid_model_manager | Гибридный менеджер моделей |
| UnifiedTextProcessor | eva.mlearning.unified_text_processor | Обработка текста |
| QueryProcessor | eva.core.query_processor | Обработка запросов |
| ResponseGenerator | eva.core.response_generator | Генерация ответов |
| BackgroundCoordinator | eva.core.background_coordinator | Фоновая координация |
| GGUFTrainingSystem | eva.training.gguf_training_system | Обучение GGUF |

---

## Отключённые компоненты

Следующие компоненты были отключены и заменены более новыми реализациями.

### 1. MemoryGraphML (ОТКЛЮЧЁН)

Статус: Отключён (заменён на FractalGraphV2)

Причина: Используем fractal_graph_v2 вместо memory_graph_ml

Код отключения (brain_components.py, строки 86-95):
# ОТКЛЮЧЕНО - используем fractal_graph_v2 вместо MemoryGraphML
brain.memory_graph_ml = None  # Используем fractal_graph_v2

Рекомендация: Используйте FractalMemoryGraph из eva.memory.fractal_graph_v2

### 2. MemoryGraphTrainer (ОТКЛЮЧЁН)

Статус: Отключён

Причина: Заменён системами:
- SelfDialogLearningSystem - для самообучения
- FractalStorage - для хранения рассуждений

Альтернативы:
from eva.reasoning.fractal_ml import FractalStorage
from eva.learning.self_dialog_learning import SelfDialogLearningSystem

### 3. Другие отключённые компоненты

| Компонент | Статус | Замена |
|-----------|--------|--------|
| memory_graph_ml | Отключён | FractalGraphV2 |
| MemoryGraphTrainer | Отключён | SelfDialogLearningSystem |
| old_knowledge_graph | Заменён | IntegratedKnowledgeGraph |
| LegacyCache | Заменён | HybridTokenCache |

---

## Граф архитектуры

+-----------------------------------------------------------------+
|                           COGNIFLEX (EVA AI)                    |
+-----------------------------------------------------------------+

  +-------------------------------------------------------------+
  |                      EVACORE (ЯДРО)                         |
  |  +-------------+  +-------------+  +--------------------+   |
  |  |  CoreBrain  |  |  EventBus   |  | DeferredSystem    |   |
  |  |(координатор)|  |  (события)  |  | (отлож. команды)  |   |
  |  +------+------+  +-------------+  +--------------------+   |
  |         |                                                    |
  |  +------+------------------------------------------------+  |
  |  |            ComponentInitializer (21 компонент)        |  |
  |  |  +---------+ +---------+ +---------+ +-----------+     |  |
  |  |  |Resource | | Config  | |Metrics  | |  Ethics   |     |  |
  |  |  | Manager | | Manager | |Collector| | Framework |     |  |
  |  |  +---------+ +---------+ +---------+ +-----------+     |  |
  |  +-------------------------------------------------------+  |
  +-------------------------------------------------------------+

  +-------------+  +-------------+  +-------------+
  |   EVAMEMORY |  |  EVAKNOWLEDGE |  |  EVAMLEARNING |
  |             |  |              |  |              |
  | +---------+ |  | +---------+ |  | +---------+ |
  | |MemoryMgr| |  | |Knowledge | |  | |ModelMgr | |
  | |         | |  | |Graph     | |  | |         | |
  | |- Working| |  | |         | |  | |- Fractal| |
  | |- Semantic| |  | +----+----+ |  | |- Qwen   | |
  | |- Episodic| |  |      |       |  | +----+----+ |
  | +----+----+ |  | +-----+-----+ |  |      |      |
  |      |       |  | |Graph     | |  | +----+----+ |
  | +----+----+ |  | |Curator    | |  | |Text    | |
  | |Hybrid    | |  | |          | |  | |Processor| |
  | |TokenCache| |  | |- Curation| |  | |        | |
  | |         | |  | |- Extract | |  | |- Tokenize| |
  | |- LRU    | |  | |- Clean   | |  | |- Quality| |
  | |- Disk   | |  | +-----------+ |  | +---------+ |
  | +----+----+ |  +-------------+  +-------------+
  |      |       |                     |
  | +----+----+ |           +---------+--------+
  | |Fractal    |           |    Query        |
  | |GraphV2    |           |    Processor    |
  | |           |           +--------+--------+
  | |- Nodes    |           |                 |
  | |- Edges    |           +---------+--------+
  | |- Groups   |           |   Response      |
  | |- Embeddings|          |   Generator    |
  | +-----------+          +-----------------+
  +-------------+          +-----------------+
                              |
  +-------------+          +-----------------+
  |   EVAGENERATION         |   Reasoning     |
  |               |         |   Engine        |
  | Generation    |         +-----------------+
  | Coordinator   |
  +-------------+

  +-------------+          +-----------------+
  |  EVATRAINING |          |   EVAMONITORING |
  |             |          |                 |
  |GGUFTraining |          | SystemMonitor   |
  |System       |          | MetricsManager  |
  +-------------+          +-----------------+

  +-------------------------------------------------------------+
  |                    BACKGROUND SERVICES                       |
  |  - BackgroundCoordinator (автопилот, idle detection)        |
  |  - SelfDialogLearningSystem (самообучение)                 |
  |  - GGUFTrainingSystem (дообучение моделей)                 |
  +-------------------------------------------------------------+

---

## Версии и история изменений

| Версия | Дата | Изменения |
|--------|------|-----------|
| 2.0.0 | 2026-04 | FractalGraphV2, GraphCurator, ComponentInitializer |
| 1.x | 2025 | Legacy система с MemoryGraphML |

---

*Документация создана: 2026-04-07*
*Система: CogniFlex (EVA AI)*
