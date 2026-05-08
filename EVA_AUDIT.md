# EVA AI - Архитектурный аудит и план исправлений

**Дата:** 2026-05-08  
**Версия:** 1.0  
**Статус:** В работе

---

## Краткое содержание

Анализ выявил **5 критических**, **10 серьёзных** и множество мелких проблем в архитектуре EVA AI. Основные недоработки связаны с отсутствием реальной интеграции между системой обучения (LoRA/GNN) и инференсом (FCPipeline), а также с thread safety и утечками памяти.

---

## 🔴 КРИТИЧЕСКИЕ ПРОБЛЕМЫ

| # | Проблема | Компонент | Последствие | Файлы |
|---|----------|-----------|-------------|-------|
| 1 | **LoRA веса НЕ применяются после обучения** | `online_trainer.py` → `fcp_pipeline.py` | Обученные веса сохраняются, но FCPipeline не загружает их автоматически. Нет механизма hot-reload | `eva_ai/fcp_core/online_trainer.py`, `eva_ai/core/fcp_pipeline.py` |
| 2 | **GNN веса НЕ передаются в HybridLayerProcessor** | `hybrid_integration.py` | PyTorch веса из GNNTrainer не конвертируются в numpy для локального энкодера. Разные размерности (768→512 vs 384→2560) | `eva_ai/fcp_core/online_trainer.py`, `eva_ai/fcp_gnn/hybrid_integration.py` |
| 3 | **EventBus может deadlock** | `event_bus.py` | Вызов handler под lock - блокирующий handler замораживает всю шину событий. Риск при shutdown | `eva_ai/core/event_bus.py` |
| 4 | **HNSWIndex без `__len__()`** | `optimizations.py` | `len(index)` вызывает AttributeError. GraphIndexer использует обходной хак через `_vector_count` | `eva_ai/memory/fractal_graph_v2/optimizations.py` |
| 5 | **Memory leak в reasoning streaming** | `fcp_pipeline.py` | буфер растёт бесконечно при длинных рассуждениях. `reasoning_buffer` не ограничен | `eva_ai/core/fcp_pipeline.py` |

---

## 🟠 СЕРЬЁЗНЫЕ ПРОБЛЕМЫ

| # | Проблема | Компонент | Последствие | Файлы |
|---|----------|-----------|-------------|-------|
| 6 | **CoreBrain - God Object** | `core_brain.py` | 10+ миксинов, 50+ атрибутов. Нарушение SRP. Сложно тестировать и поддерживать | `eva_ai/core/core_brain.py` |
| 7 | **Duplicate FCPipeline initialization** | `brain_components.py` | `_init_fcp_pipeline()` вызывается 2жды (в `__init__` и в `_init_unified_generator`). Race condition возможна | `eva_ai/core/core_brain.py`, `eva_ai/core/brain_components.py` |
| 8 | **ResourceManager проверяет GPU после загрузки батча** | `online_trainer.py` | Трата ресурсов - батч загружается, потом проверяется GPU. Нужно наоборот | `eva_ai/fcp_core/online_trainer.py` |
| 9 | **Cache не синхронизируется с историей диалога** | `cache_core.py` | `add_context()` существует, но не вызывается автоматически после сообщений | `eva_ai/memory/cache_core.py` |
| 10 | **HotSwapManager не реализован** | `online_trainer.py` | Создаётся, но `update()` не выполняет реальную замену весов | `eva_ai/fcp_core/online_trainer.py` |
| 11 | **FractalGraphV2 lazy mode - node_count возвращает 0** | `fractal_graph_v2.py` | Property node_count использует `len(nodes)` вместо `get_node_count_lazy()` | `eva_ai/memory/fractal_graph_v2/__init__.py` |
| 12 | **Граф не используется для векторного поиска** | `storage.py` | Основной `semantic_search()` НЕ использует GraphIndexer. Прямой перебор O(n) | `eva_ai/memory/fractal_graph_v2/storage.py` |
| 13 | **Hardcoded путь к модели fmf_model/model.ov** | `brain_components.py` | Fallback на несуществующий путь вызывает ошибку при отсутствии config | `eva_ai/core/brain_components.py` |
| 14 | **Token callback парсит пустую строку как end tag** | `fcp_pipeline.py` | `idx = buffer.find("")` всегда возвращает 0, логика сломана | `eva_ai/core/fcp_pipeline.py:806` |
| 15 | **Thread safety в singleton паттернах** | `init_factories.py` | Проверка `existing is not None` не атомарна - race condition | `eva_ai/core/init_factories.py` |

---

## 🟡 МЕЛКИЕ ПРОБЛЕМЫ

### Код Quality
- [ ] Dead code: `_init_two_model_pipeline()` в `brain_components.py`
- [ ] Dead code: `create_enhanced_reasoning_engine()` в `init_factories.py`
- [ ] Dead code: Строки 356-386 в `server_routes_chat.py`
- [ ] Unused imports: `json` в `fcp_pipeline.py`, `queue` в `core_brain.py`
- [ ] `print()` вместо `logger.info()` в нескольких местах
- [ ] jQuery подключён но не используется в `index.html`

### Hardcoded values
- [ ] `'C:/Users/black/OneDrive/Desktop/FCP/lora_adapters'` в `fcp_pipeline.py`
- [ ] `num_layers = 36` в `online_trainer.py` (предполагаемое)
- [ ] `chunk_size=5` в `hybrid_dialog_manager.py`
- [ ] Timeout по длине сообщения в `server_routes_chat.py`

### Logging
- [ ] Verbose EventBus subscribe/unsubscribe логирование
- [ ] Mix `query_logger` и `logger` - нет统一ного подхода
- [ ] console.log в production JS коде

### Типизация
- [ ] Missing type hints в `event_bus.py`
- [ ] No type hints в `init_factories.py`

---

## 📋 DETAILED ANALYSIS BY COMPONENT

### 1. Core Brain (core_brain.py)

**Architecture:**
- Central coordinator с 10+ миксинами
- Weakref singleton pattern
- Две параллельные событийные системы (EventSystem + EventBus)

**Problems:**
- God Object antipattern
- Duplicate initialization calls
- No atomic initialization

**File:** `eva_ai/core/core_brain.py`

---

### 2. FCPipeline (fcp_pipeline.py)

**Architecture:**
- OpenVINO GenAI LLMPipeline
- Hybrid layers: KCA, SRG, GNN injection
- LoRA adapters support
- Memory snapshots

**Problems:**
- Token callback с пустым `find("")`
- Memory leak в reasoning buffer
- No hot-reload для LoRA
- Online trainer integration broken

**File:** `eva_ai/core/fcp_pipeline.py`

---

### 3. EventBus (event_bus.py)

**Architecture:**
- Priority queue-based pub/sub
- Weak references для handlers
- Sync/async publishing

**Problems:**
- Handler вызывается под lock - deadlock risk
- No timeout для handlers
- Deadlock при shutdown если handler заблокирован

**File:** `eva_ai/core/event_bus.py`

---

### 4. Online Trainer (online_trainer.py)

**Architecture:**
- OnlineTrainerManager → GNNTrainer + LoRATrainer
- BackgroundTrainer base class
- ResourceManager для pause/resume

**Problems:**
- GPU check после batch load
- LoRA sync работает, но не применяется к FCPipeline
- GNN weights не передаются в HybridLayerProcessor
- Synthetic fallback при пустом графе
- No early stopping

**File:** `eva_ai/fcp_core/online_trainer.py`

---

### 5. Hybrid Integration (hybrid_integration.py)

**Architecture:**
- FractalGraphEncoderLocal (numpy)
- AdaptiveFusionInjectorLocal
- HybridLayerProcessor

**Problems:**
- GNN weights из PyTorch не конвертируются
- Разные размерности: GNNTrainer (768→512) vs HybridLayer (384→2560)
- LoRA в энкодере не обучается
- HNSW index rebuilds при каждом запросе

**File:** `eva_ai/fcp_gnn/hybrid_integration.py`

---

### 6. FractalGraphV2 (storage.py + __init__.py)

**Architecture:**
- SQLite с BLOB embeddings
- Lazy/full loading modes
- HNSW index for search

**Problems:**
- node_count returns 0 в lazy mode
- semantic_search не использует GraphIndexer
- Singleton создаёт новый instance при другой storage_dir

**Files:** 
- `eva_ai/memory/fractal_graph_v2/storage.py`
- `eva_ai/memory/fractal_graph_v2/__init__.py`

---

### 7. GraphIndexer (graph_indexer.py)

**Architecture:**
- SQLite → HNSW index
- Fallback на SQL LIKE

**Problems:**
- HNSWIndex.__len__() не реализован
- GraphIndexer.__len__() - workaround, не solution

**File:** `eva_ai/memory/fractal_graph_v2/graph_indexer.py`

---

### 8. HNSWIndex (optimizations.py)

**Architecture:**
- 3 backends: nmslib > faiss > fallback brute force
- Cosine similarity

**Problems:**
- __len__() missing
- addPoint for nmslib may have wrong API
- Fallback doesn't normalize vectors

**File:** `eva_ai/memory/fractal_graph_v2/optimizations.py`

---

### 9. HybridTokenCache (cache_core.py)

**Architecture:**
- Multi-level: VRAM (1.5GB) > RAM (1GB) > Disk (50GB)
- LRU eviction

**Problems:**
- No automatic sync with conversation history
- add_token() duplicates RAM + Disk

**File:** `eva_ai/memory/cache_core.py`

---

### 10. WebGUI (app.js + server_routes_chat.py)

**Architecture:**
- XHR streaming
- SSE event parsing
- Reasoning UI with collapsible panel

**Problems:**
- formatText called twice (wastes CPU)
- Buffer grows unbounded
- No disconnect handling during generation
- Partial JSON silently ignored

**Files:**
- `eva_ai/gui/web_gui/static/js/app.js`
- `eva_ai/gui/web_gui/server_routes_chat.py`

---

## ✅ ВЫПОЛНЕННЫЕ ИСПРАВЛЕНИЯ

- [x] Unified generation config в brain_config.json
- [x] Reasoning parsing fixed для `<think>` tags
- [x] LoRA sync в active_lora_dir после save_checkpoint
- [x] Safe imports для DualGenerator, FractalModelManager
- [x] Removed duplicate fmf_model/model.ov fallback
- [x] Fixed FCPipeline singleton pattern

---

## 📌 ПЛАН ИСПРАВЛЕНИЙ

### Приоритет 1: Критические (блокируют работу)

| # | Задача | Статус |
|---|--------|--------|
| 1.1 | Добавить `__len__()` в HNSWIndex | ⬜ |
| 1.2 | Исправить token_callback - заменить `find("")` на `find("</think>")` | ⬜ |
| 1.3 | Добавить hot-reload LoRA в FCPipeline | ⬜ |
| 1.4 | Создать мост GNNTrainer → HybridLayerProcessor | ⬜ |
| 1.5 | Добавить timeout для EventBus handlers | ⬜ |

### Приоритет 2: Серьёзные (влияют на производительность)

| # | Задача | Статус |
|---|--------|--------|
| 2.1 | Перенести GPU check ДО загрузки батча | ⬜ |
| 2.2 | Интегрировать GraphIndexer в semantic_search | ⬜ |
| 2.3 | Исправить node_count в lazy mode | ⬜ |
| 2.4 | Реализовать HotSwapManager.update() | ⬜ |
| 2.5 | Добавить auto-sync cache с conversation history | ⬜ |

### Приоритет 3: Рефакторинг

| # | Задача | Статус |
|---|--------|--------|
| 3.1 | Разбить CoreBrain на подсистемы | ⬜ |
| 3.2 | Удалить dead code | ⬜ |
| 3.3 | У统一ить логирование | ⬜ |
| 3.4 | Добавить type hints | ⬜ |

---

## 📊 МЕТРИКИ

| Компонент | Строк кода | Проблем критических | Проблем серьёзных |
|-----------|-----------|---------------------|-------------------|
| core_brain.py | ~1500 | 2 | 3 |
| fcp_pipeline.py | ~2500 | 2 | 4 |
| event_bus.py | ~600 | 1 | 2 |
| online_trainer.py | ~1400 | 2 | 4 |
| hybrid_integration.py | ~700 | 2 | 3 |
| storage.py | ~1000 | 1 | 2 |
| optimizations.py | ~500 | 1 | 2 |
| cache_core.py | ~800 | 1 | 2 |
| **ИТОГО** | **~9500** | **12** | **22** |

---

## 🔗 СВЯЗИ МЕЖДУ КОМПОНЕНТАМИ

```
┌─────────────────────────────────────────────────────────────────┐
│                        FCPipeline                                │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐ │
│  │ LoRA Trainer│  │ GNN Trainer │  │ HybridLayerProcessor    │ │
│  │             │  │             │  │                          │ │
│  │ save() ─────┼─→│ save() ─────┼─→│ load_trained_encoder()   │ │
│  │ _sync_to_   │  │             │  │                          │ │
│  │ active_lora │  │ graph_encoder│ ←(НЕ РАБОТАЕТ)            │ │
│  └─────────────┘  └─────────────┘  └──────────────────────────┘ │
│         ↓                ↓                                        │
│  eva_ai/models/lora  checkpoint/gnn                              │
│  lora_model.safetensors gnn_model.pt                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │   ResourceManager   │
                    │                     │
                    │ IDLE ──generation──►│
                    │  BUSY ◄───end────────│
                    └─────────────────────┘
```

---

## 📝 NOTES

- GNN Trainer использует input_dim=768, output=512
- HybridLayerProcessor ожидает input_dim=384, hidden=2560
- Разрыв в размерностях需要 конвертер
- LoRA sync уже работает (добавлен `set_active_lora_dir` и `_sync_to_active_lora_dir`)
- Но FCPipeline не вызывает hot-reload после обновления весов
