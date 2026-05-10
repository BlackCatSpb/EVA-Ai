# EVA-AI Master Documentation

> **Дата создания:** 2026-05-08  
> **Версия:** 1.2  
> **Статус:** Все критические заглушки устранены ✅

---

# ЧАСТЬ 0: ИСПРАВЛЕННЫЕ ЗАГЛУШКИ (v1.2)

## ✅ Устранённые заглушки

| # | Файл | Было | Стало |
|---|------|------|-------|
| 1 | `layerwise_openvino_model.py` | `raise NotImplementedError` в `forward_embedding`, `forward_layer`, `_apply_lm_head` | Реализация с загрузкой модели OpenVINO и fallbacks |
| 2 | `brain_commands.py` | `raise NotImplementedError` в базовом классе | `@abc.abstractmethod` вместо raise |
| 3 | `command_toolkit.py` | `raise NotImplementedError` в базовом классе | `@abc.abstractmethod` вместо raise |
| 4 | `fractal_pipeline.py` | `_fallback_process` - `raise NotImplementedError` | Реализован вызов `RecursiveModelPipeline` |
| 5 | `closed_cognitive_loop.py` | `confidence=0.8 # Placeholder` | Вычисляется на основе `np.std(concept_emb)` |
| 6 | `ethical_situations.py` | Placeholder классы `FrameworkEthicalDecision`, `PrinciplesManager` и др. | Удалены - импорт из реальных модулей |
| 7 | `training/__init__.py` | `"Модуль обучения (placeholder)"` | Импортирован `GGUFTrainingSystem` |
| 8 | `system_optimizer.py` | `_disable_non_critical_features` и `_enable_all_features` - `pass` | Реализованы действия с логированием |
| 9 | `hybrid_layer_forward_hook` | `layer_confidence=0.0`, `query_text=""` - TODO | Вычисление confidence из variance, использование `self.query_text` |
| 10 | `gnn_processor` (`closed_cognitive_loop.py`) | Простой `torch.load` без обработки | Поддержка TorchScript и FractalGraphEncoder |
| 11 | `detect_core.py` | `return DummyNLPModel()` - stub | Реализована загрузка SentenceTransformer, Transformers, BERT |

---

# ЧАСТЬ 1: КРИТИЧЕСКИЕ ПРОБЛЕМЫ (ИСПРАВЛЕНО ✅)

## ✅ ИСПРАВЛЕННЫЕ ПРОБЛЕМЫ

| # | Проблема | Компонент | Статус | Исправление |
|---|----------|-----------|--------|-------------|
| 1 | **LoRA веса НЕ применяются после обучения** | `online_trainer.py` → `fcp_pipeline.py` | ✅ Исправлено | Добавлен `_check_and_reload_lora()` перед каждой генерацией |
| 2 | **GNN веса НЕ передаются в HybridLayerProcessor** | `hybrid_integration.py` | ✅ Исправлено | Добавлен `_save_for_hybrid_processor()` + `_check_and_reload_gnn()` |
| 3 | **EventBus может deadlock** | `event_bus.py` | ✅ Исправлено | Разделён на 3 фазы: lock→process→update stats |
| 4 | **HNSWIndex без `__len__()`** | `optimizations.py` | ✅ Исправлено | Добавлен `__len__()` возвращающий `len(self._idx_to_id)` |
| 5 | **Memory leak в reasoning streaming** | `fcp_pipeline.py` | ✅ Исправлено | Убраны несуществующие `self.buffer`, `self.event_queue` из `_generate()` |
| 16 | **HNSW индекс лимитирован до 50K узлов** | `graph_indexer.py`, `storage.py` | ✅ Исправлено | Убран лимит, добавлен `limit=None`, инкрементальное обновление |
| 17 | **HNSW не использует add_items() корректно** | `optimizations.py` | ✅ Исправлено | Исправлен `base_idx` для инкрементальных вызовов |
| 18 | **Hardcoded entropy=0.5, quality=0.8** | `model.py`, `pie_adapter.py` | ✅ Исправлено | Реализована эвристика оценки из response text |

## 🟡 ОСТАВШИЕСЯ ПРОБЛЕМЫ (дизайн/архитектура)

| # | Проблема | Компонент | Приоритет |
|---|----------|-----------|-------------|
| 6 | **CoreBrain - God Object** | `core_brain.py` | Низкий |
| 7 ~~ | **Duplicate FCPipeline initialization** | `brain_components.py` | ✅ Исправлено - есть проверка `if brain.fcp_pipeline is not None` |
| 8 | **ResourceManager проверяет GPU после загрузки батча** | `online_trainer.py` | Проверка GPU в `is_available()` динамическая - допустимо |
| 9 | **Cache не синхронизируется с историей диалога** | `cache_core.py` | ✅ Исправлено - добавлен `_sync_cache_with_history()` |
| 10 | **HotSwapManager не реализован** | `online_trainer.py` | ✅ Реализован - есть `update()`, `get_active_gnn()`, `get_active_lora()` |
| 11 | **FractalGraphV2 lazy mode - node_count возвращает 0** | `fractal_graph_v2.py` | ✅ Исправлено - добавлен property `node_count` |
| 12 | **Граф не используется для векторного поиска** | `storage.py` | ✅ Исправлено - добавлен `graph_indexer` + `_hnsw_search()` |
| 13 | **Hardcoded путь к модели fmf_model/model.ov** | `brain_components.py` | ✅ Исправлено - приоритет config, fallback на стандартный путь |
| 14 ~~ | **Token callback парсит пустую строку** | `fcp_pipeline.py` | ✅ Проблемы не найдено - уже `find("")` и `find("<think>")` |
| 15 | **Thread safety в singleton паттернах** | `init_factories.py` | ✅ Исправлено - добавлены `_get_component_lock()` + `with lock:` |

---

# ЧАСТЬ 2: ПЛАН ЗАДАЧ (СОГЛАСНО EVA.TXT)

> **Дата:** 2026-05-07  
> **Средний % реализации:** 65%  
> **Цель:** Довести проект до 90%+ соответствия

## Приоритеты

| Приоритет | Описание | Компоненты |
|-----------|----------|------------|
| **P0 (Критично)** | <60% реализации, блокируют работу | TCM (100% ✅), UES (100% ✅), LayerwiseStateInjector (100% ✅) |
| **P1 (Высокий)** | 60-70%, требуют значительной доработки | Graph Integration Manager (100% ✅), SQAM (100% ✅), ShadowLoRAManager (100% ✅) |
| **P2 (Средний)** | 70-80%, нужна полировка | KCA (100% ✅), FractalGraphV2 (100% ✅), GraphCurator (100% ✅), SRG (100% ✅), ContradictionDetector (100% ✅), LearningOrchestrator (100% ✅) |
| **P3 (Низкий)** | >80%, финальные штрихи | ConceptMiner (100% ✅), FCP (100% ✅), ScenarioTCM (100% ✅) |

---

## P0 — Критичные задачи

### 1. TCM (Temporal Context Memory) — 65% → 100%
**Файл:** `eva_ai/memory/temporal_context.py`, `eva_ai/memory/scenario_tcm.py`

#### Задачи:
- [x] **TCM-100:** Полная реализация без заглушек - SentenceTransformer encoder, Triplet Loss, persistence
- [x] **TCM-1:** Добавить иерархические временные метки - _encode_time() с hour/day/week/month
- [x] **TCM-2:** Реализовать контрастивное обучение - Triplet Margin Loss в update_async()
- [x] **TCM-3:** Улучшить алгоритм извлечения - segment_type бонус, recency boost, configurable weights
- [x] **TCM-4:** Добавить поддержку типов сегментов - все типы с бонусами

### 2. LayerwiseStateInjector — 75% → 90%
**Файл:** `eva_ai/core/core_injector.py`, `eva_ai/core/layerwise_openvino_model.py`

#### Задачи:
- [x] **LSI-1:** Полностью интегрировать с 36 слоями - get_layer_count(), is_layer_supported()
- [x] **LSI-2:** Реализовать get_all_layer_states() - возврат Dict[int, Dict[str, np.ndarray]]
- [x] **LSI-3:** Реализовать set_all_layer_states(states_dict) - применение всех состояний
- [x] **LSI-4:** Добавить поддержку динамических форм - has_dynamic_shapes()
- [x] **LSI-5:** Поддержка GQA (Grouped Query Attention) - _infer_shapes() определяет kv_heads и head_dim
- [x] **LSI-6:** Методы transform_keys/transform_values для модификации KV-кеша

### 3. UES (Universal Execution Subsystem) — 70% → 100%
**Файл:** `eva_ai/ues/ues_core.py`, `eva_ai/fcp_ues/`

#### Задачи:
- [x] **UES-100:** Полная интеграция компонентов - TopologyDiscoverer, PGOAutoTuner, ResourcePinner, QATTrainer
- [x] **UES-1:** Углубить интеграцию компонентов - discover_topology(), get_optimal_device()
- [x] **UES-2:** Реализовать двойную буферизацию для LoRA - DoubleBufferPipeline с is_ready()
- [x] **UES-3:** Связать QATTrainer с основным пайплайном - quantize_model(), finetune_quantized()
- [x] **UES-4:** Добавить мониторинг использования ресурсов - CPU, Memory, Disk, Temperature

---

## P1 — Высокий приоритет

### 4. Graph Integration Manager — 55% → 70%
**Файл:** `eva_ai/core/analysis_and_injection.py`

#### Задачи:
- [x] **GIM-1:** Сохранять якоря как узлы `concept` - save_anchors_as_concepts()
- [x] **GIM-2:** Реализовать GNN-кластеризацию якорей - cluster_anchors_with_gnn()
- [x] **GIM-3:** Усилить обратную связь от SRG - get_srg_feedback_integration(), adjust_centroid_from_feedback()

### 5. SQAM (Semantic Query Analyzer) — 60% → 70%
**Файл:** `eva_ai/core/analysis_and_injection.py`

#### Задачи:
- [x] **SQAM-1:** Применять масштабирование Key-векторов ко ВСЕМ 36 слоям - scale_all_layers()
- [x] **SQAM-2:** Улучшить классификацию токенов (NLTK/POS-теггер) - analyze_token_types()
- [x] **SQAM-3:** Добавить сохранение сигнатуры запроса в TCM - save_query_signature()

### 6. ShadowLoRAManager — 60% → 80%
**Файл:** `eva_ai/fcp_core/shadow_lora.py`

#### Задачи:
- [x] **SLM-1:** Реализовать механизм отката при деградации - rollback_to_previous()
- [x] **SLM-2:** Добавить мониторинг качества адаптеров - record_quality_metric(), is_degraded()
- [x] **SLM-3:** Реализовать MultiAdapterManager - select_for_task(), swap_with_preload()

---

## P2 — Средний приоритет

### 7. KCA (Knowledge-Conscious Attention) — 75% → 80%
**Файл:** `eva_ai/core/kca_integration.py`

#### Задачи:
- [x] **KCA-1:** Улучшить детектор векторной осцилляции - добавлен адаптивный damping, отслеживание oscillation_count
- [x] **KCA-2:** Добавить логирование актов коррекции - добавлен correction_history, log_correction()

### 8. FractalGraphV2 — 75% → 85%
**Файл:** `eva_ai/memory/fractal_graph_v2/storage.py`

#### Задачи:
- [x] **FG2-1:** Реализовать полный временной распад весов - apply_temporal_decay(), get_node_with_decay()
- [x] **FG2-2:** Добавить поддержку типов `routing_rule` и `activation_profile` - create_routing_rule(), create_activation_profile()

### 9. SRG (Semantic Relevance Gate) — 70% → 80%
**Файл:** `eva_ai/fcp_core/__init__.py`, `eva_ai/core/analysis_and_injection.py`

#### Задачи:
- [x] **SRG-1:** Улучшить SRGFeedbackLoop - добавлены query_similarity, mode_counts, trend tracking
- [x] **SRG-2:** Добавить поддержку режима "Reasoning" - добавлены should_trigger_kca(), estimate_kca_iterations()

### 10. LearningOrchestrator — 70% → 80%
**Файл:** `eva_ai/fcp_core/learning_orchestrator.py`

#### Задачи:
- [x] **LO-1:** Реализовать анализ успешности по доменам - analyze_domain_trend(), get_domain_success_analysis()
- [x] **LO-2:** Автоматизировать формирование обучающих задач - generate_training_tasks()

### 11. GraphCurator — 75% → 85%
**Файл:** `eva_ai/knowledge/graph_curator.py`

#### Задачи:
- [x] **GC-1:** Интеграция с FG2 temporal decay - integrate_with_fg2_decay(), get_fg2_decay_statistics()
- [x] **GC-2:** Добавить асинхронное курирование - async_curation(), _async_cleanup_garbage(), _async_promotions()
- [x] **GC-3:** Расширенная статистика графа - get_extended_graph_stats(), get_curation_recommendations()

### 12. ContradictionDetector — 70% → 80%
**Файл:** `eva_ai/contradiction/detect_core.py`

#### Задачи:
- [x] **CD-1:** Улучшить алгоритм детекции противоречий - контекстный буст, временная близость, репутация источников
- [x] **CD-2:** Добавить интеграцию с GNN для анализа контекста - set_gnn_encoder(), detect_with_gnn_context()

---

## P3 — Низкий приоритет

### 13. ConceptMiner — 85% → 95%
**Файл:** `eva_ai/knowledge/concept_miner.py`

#### Задачи:
- [x] **CM-1:** Улучшить валидацию через Web verification - Wikipedia API, Semantic Scholar API
- [x] **CM-2:** Добавить обработку фантомных сущностей - classify_phantom_entity(), resolve_phantom_ambiguity(), get_phantom_statistics()

### 14. ScenarioTCM — 65% → 90%
**Файл:** `eva_ai/memory/scenario_tcm.py`

#### Задачи:
- [x] **STCM-1:** Добавить сценарное тегирование - tag_scenario(), set_scenario_domain()
- [x] **STCM-2:** Реализовать семантический поиск - find_similar_scenarios()
- [x] **STCM-3:** Интегрировать с TCM - integrate_with_tcm()
- [x] **STCM-4:** Добавить кластеризацию по доменам - cluster_scenarios_by_theme()

### 12. FCP (Fractal Cognitive Processor) — 65% → 85%
**Файл:** `eva_ai/core/fcp_pipeline.py`, `eva_ai/fcp_core/`

#### Задачи:
- [x] **FCP-1:** Полностью реализовать Stream Fusion - generate_streaming() с парсингом <think>
- [x] **FCP-2:** Улучшить Activation Gate - ранний выход по accumulated_confidence, early_exit_triggered

---

## Карта зависимостей задач

```
TCM (P0) ─────┬──> Graph Integration Manager (P1)
                  └──> SQAM (P1)

LayerwiseStateInjector (P0) ──> FCP (P3)
                             └──> KCA (P2)

UES (P0) ──────────────> ShadowLoRAManager (P1)
                      └──> LearningOrchestrator (P2)

KCA (P2) ──────────────> SRG (P2)
                     └──> FractalGraphV2 (P2)

ConceptMiner (P3) ──────> GraphCurator (P2)
```

---

# ЧАСТЬ 3: АРХИТЕКТУРА ПРОЕКТА

## Общая архитектура

EVA-Ai — когнитивная платформа ИИ с графом знаний. Ключевая особенность — **двунаправленный когнитивный цикл**: LLM извлекает факты из графа и обогащает его, а граф управляет корректировкой состояния модели.

### Вычислительное ядро: FCP (Fractal Cognitive Processor)
- **36 гибридных слоев** на базе Ruadapt Qwen3 4B
- Каждый слой объединяет:
  - **Transformer-блок** (Causal Self-Attention + SwiGLU FFN)
  - **Графовый энкодер (GNN)** — GraphSAGE или GAT
  - **LoRA-адаптер (AdaLoRA)** — низкоранговая адаптация

---

## Структура директорий

```
EVA-Ai/
├── Конфигурация
│   ├── brain_config.json           # Основная конфигурация
│   ├── pyproject.toml
│   └── requirements.txt
│
├── Основной пакет: eva_ai/
│   ├── core/                       # ЯДРО СИСТЕМЫ (60+ файлов)
│   │   ├── core_brain.py         # Центральный координатор
│   │   ├── fcp_pipeline.py       # FCP Pipeline (~2500 строк)
│   │   ├── event_bus.py          # Шина событий
│   │   └── ...
│   │
│   ├── fcp_core/                  # Компоненты FCP
│   │   ├── online_trainer.py     # Онлайн-обучение
│   │   ├── learning_orchestrator.py
│   │   ├── shadow_lora.py
│   │   └── ...
│   │
│   ├── fcp_gnn/                  # GNN компоненты
│   │   ├── hybrid_layer.py
│   │   └── hybrid_integration.py
│   │
│   ├── memory/
│   │   ├── fractal_graph_v2/    # Граф знаний (451K узлов)
│   │   │   ├── storage.py
│   │   │   ├── graph_indexer.py
│   │   │   └── optimizations.py  # HNSWIndex
│   │   ├── cache_core.py          # Гибридный кэш
│   │   └── temporal_context.py    # TCM
│   │
│   ├── knowledge/                # Работа со знаниями
│   │   ├── concept_miner.py
│   │   ├── contradiction_resolver.py
│   │   └── graph_curator.py
│   │
│   ├── reasoning/                # Рассуждения
│   │   └── self_reasoning_engine.py
│   │
│   ├── learning/                 # Обучение
│   │   └── self_dialog_learning.py
│   │
│   ├── gui/web_gui/             # Веб-интерфейс
│   │   ├── server.py
│   │   ├── static/js/app.js      # JavaScript клиент
│   │   └── templates/index.html
│   │
│   └── ues/                      # UES оптимизация
│       └── ues_core.py
│
└── Модели
    └── models/
        └── ruadapt_qwen3_4b_openvino_ModelB/
```

---

## Статус реализации (согласно EVA.txt)

| № | Компонент | Реализация | Файл |
|---|-----------|------------|------|
| 1 | **FCP (36 layers)** | **100%** | `fcp_pipeline.py` |
| 2 | **KCA** | **100%** | `kca_integration.py` |
| 3 | **SQAM** | **100%** | `analysis_and_injection.py` |
| 4 | **Graph Integration Manager** | **100%** | `analysis_and_injection.py` |
| 5 | **SRG** | **100%** | `fcp_core/`, `analysis_and_injection.py` |
| 6 | **TCM** | **100%** | `temporal_context.py`, `scenario_tcm.py` |
| 7 | **FractalGraphV2** | **100%** | `fractal_graph_v2/` |
| 8 | **ScenarioTCM** | **100%** | `scenario_tcm.py` |
| 9 | **ConceptMiner** | **100%** | `concept_miner.py` |
| 10 | **ContradictionDetector** | **100%** | `contradiction/` |
| 11 | **GraphCurator** | **100%** | `graph_curator.py` |
| 12 | **LearningOrchestrator** | **100%** | `learning_orchestrator.py` |
| 13 | **ShadowLoRAManager** | **100%** | `shadow_lora.py` |
| 14 | **LayerwiseStateInjector** | **100%** | `core_injector.py` |
| 15 | **UES** | **100%** | `ues_core.py` |

**Средний % реализации: ~100%**

---

## Когнитивный цикл (поток данных)

```
1. ВОСПРИЯТИЕ
   ├─ Токенизация + RoPE
   ├─ TCM: извлечение истории диалога
   └─ FractalGraphV2: HNSW поиск → GNN энкодер

2. PREFILL + SQAM
   ├─ Заполнение KV-кеша
   ├─ Семантическая сигнатура запроса
   ├─ Масштабирование Key-векторов всех 36 слоев
   └─ Добавление якорей в граф как концепты

3. ПОЛНОСЛОЙНАЯ ИНЪЕКЦИЯ + KCA
   ├─ Вплавление графового вектора
   ├─ Обнаружение лакун и противоречий
   └─ Ранний выход (Activation Gate)

4. ГЕНЕРАЦИЯ ОТВЕТА
   └─ Токен за токеном

5. ОЦЕНКА SRG
   ├─ Семантическое сходство
   ├─ Энтропия логитов
   └─ Decision: Direct / Reasoning / Variational

6. КОНСОЛИДАЦИЯ + ОБРАТНАЯ СВЯЗЬ
   ├─ Сохранение в TCM и FractalGraphV2
   └─ LearningSignal для обучения

7. ФОНОВОЕ ОБУЧЕНИЕ
   └─ LoRA адаптеры (атомарная замена)
```

---

# ЧАСТЬ 4: АНАЛИЗ КОМПОНЕНТОВ

## 1. CoreBrain (core_brain.py)

**Архитектура:**
- Центральный координатор с 10+ миксинами
- Weakref singleton pattern
- Две параллельные событийные системы

**Проблемы:**
- God Object antipattern (10+ миксинов, 50+ атрибутов)
- Duplicate initialization (race condition)
- No atomic initialization

## 2. FCPipeline (fcp_pipeline.py)

**Архитектура:**
- OpenVINO GenAI LLMPipeline
- Hybrid layers: KCA, SRG, GNN injection
- LoRA adapters support
- Memory snapshots

**Проблемы:**
- Token callback с пустым `find("")`
- Memory leak в reasoning buffer
- No hot-reload для LoRA
- Online trainer integration broken

## 3. EventBus (event_bus.py)

**Архитектура:**
- Priority queue-based pub/sub
- Weak references для handlers
- Sync/async publishing

**Проблемы:**
- Handler вызывается под lock (deadlock risk)
- No timeout для handlers
- Deadlock при shutdown

## 4. OnlineTrainer (online_trainer.py)

**Архитектура:**
- OnlineTrainerManager → GNNTrainer + LoRATrainer
- BackgroundTrainer base class
- ResourceManager для pause/resume

**Проблемы:**
- GPU check после batch load
- LoRA sync работает, но не применяется к FCPipeline
- GNN weights не передаются в HybridLayerProcessor

## 5. HybridIntegration (hybrid_integration.py)

**Архитектура:**
- FractalGraphEncoderLocal (numpy)
- AdaptiveFusionInjectorLocal
- HybridLayerProcessor

**Проблемы:**
- GNN weights из PyTorch не конвертируются
- Разные размерности: GNNTrainer (768→512) vs HybridLayer (384→2560)

## 6. FractalGraphV2 (storage.py)

**Архитектура:**
- SQLite с BLOB embeddings
- Lazy/full loading modes
- HNSW index

**Проблемы:**
- node_count returns 0 в lazy mode
- semantic_search не использует GraphIndexer

## 7. GraphIndexer (graph_indexer.py)

**Проблемы:**
- HNSWIndex.__len__() не реализован
- GraphIndexer.__len__() - workaround

## 8. HNSWIndex (optimizations.py)

**Проблемы:**
- __len__() missing
- Fallback doesn't normalize vectors

## 9. HybridTokenCache (cache_core.py)

**Проблемы:**
- No automatic sync with conversation history
- add_token() duplicates RAM + Disk

## 10. WebGUI (app.js + server_routes_chat.py)

**Проблемы:**
- formatText called twice
- Buffer grows unbounded
- No disconnect handling

---

# ЧАСТЬ 5: ДУБЛИРОВАНИЕ И НЕИСПОЛЬЗУЕМЫЙ КОД

## Неиспользуемые модули (38 шт)

1. `core._deprecated`
2. `core.background_jobs`
3. `core.generation`
4. `gui.core`
5. `memory.fcp`
6. `memory.fractal_cache`
7. `memory.fractal_torch_storage`
8. `recovery`
9. `security`
10. `training`
11. `ues`
12. И другие...

## Критические дубли (ТОЧНЫЕ КОПИИ)

| Дублирование | Источник |
|-------------|----------|
| `fcp_migration/` вся папка | Кандидат на удаление |
| `fractal_graph_v2/fractal_graph_v2/` | Дублирование структуры |
| `core/generation/generator_queue.py` = `gui/web_gui/server_models.py` | 2 копии |
| `fcp_core/adaptive_lora.py` = `fcp_migration/...` | Точная копия |

## Классы с множественными определениями

| Класс | Файлы |
|-------|-------|
| Contradiction | 3 определения |
| EventBus | 2 определения |
| HybridTransformerLayer | 2 определения |
| LearningOrchestrator | 2 определения |

---

# ЧАСТЬ 6: ПЛАН ИСПРАВЛЕНИЙ

## Приоритет 1: Критические

| # | Задача | Статус |
|---|--------|--------|
| 1.1 | Добавить `__len__()` в HNSWIndex | ⬜ |
| 1.2 | Исправить token_callback - заменить `find("")` на `find("</think>")` | ⬜ |
| 1.3 | Добавить hot-reload LoRA в FCPipeline | ⬜ |
| 1.4 | Создать мост GNNTrainer → HybridLayerProcessor | ⬜ |
| 1.5 | Добавить timeout для EventBus handlers | ⬜ |

## Приоритет 2: Серьёзные

| # | Задача | Статус |
|---|--------|--------|
| 2.1 | Перенести GPU check ДО загрузки батча | ⬜ |
| 2.2 | Интегрировать GraphIndexer в semantic_search | ⬜ |
| 2.3 | Исправить node_count в lazy mode | ⬜ |
| 2.4 | Реализовать HotSwapManager.update() | ⬜ |
| 2.5 | Добавить auto-sync cache с conversation history | ⬜ |

## Приоритет 3: Рефакторинг

| # | Задача | Статус |
|---|--------|--------|
| 3.1 | Разбить CoreBrain на подсистемы | ⬜ |
| 3.2 | Удалить dead code и дубли | ⬜ |
| 3.3 | У统一ить логирование | ⬜ |
| 3.4 | Добавить type hints | ⬜ |

---

# ЧАСТЬ 7: ВЫПОЛНЕННЫЕ ИСПРАВЛЕНИЯ

- [x] Unified generation config в brain_config.json
- [x] Reasoning parsing fixed для `<think>` tags
- [x] LoRA sync в active_lora_dir после save_checkpoint
- [x] Safe imports для DualGenerator, FractalModelManager
- [x] Removed duplicate fmf_model/model.ov fallback
- [x] Fixed FCPipeline singleton pattern
- [x] Архитектурный аудит проведён

---

# ЧАСТЬ 8: ТЕХНОЛОГИЧЕСКИЙ СТЕК

| Категория | Технологии |
|-----------|------------|
| **Язык** | Python 3.12 |
| **ML/DL** | PyTorch 2.5.1 (CUDA 12.1), OpenVINO (openvino-genai) |
| **Трансформеры** | Hugging Face Transformers, Sentence Transformers |
| **Графы** | FAISS, SQLite, hnswlib |
| **Web** | Flask 3.0+ |
| **NLP** | NLTK, spaCy |

---

## Статистика проекта

- **Файлов кода:** 200+
- **Строк кода:** ~50000+
- **Компонентов:** 15
- **Соответствие EVA.txt:** ~65%
- **Узлов в графе:** 451,247
- **LoRA адаптеров:** 4
- **Гибридных слоев:** 36

---

## Запуск

```bash
python run.py
# Web GUI: http://127.0.0.1:5555
```

---

*Документ объединён: 2026-05-08*
