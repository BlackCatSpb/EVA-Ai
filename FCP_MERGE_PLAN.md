# EVA-Ai + FCP + FMF Слияние - Детальный План

**Дата:** 25.04.2026  
**Статус:** Планирование

---

## 1. Анализ текущих систем

### 1.1 EVA-Ai - Основные компоненты

| Модуль | Функция | Статус |
|--------|---------|--------|
| `core/hybrid_pipeline_adapter.py` | Гибридный адаптер с DualGenerator | ✅ Active |
| `core/openvino_generator.py` | OpenVINO GenAI с Registry | ✅ Active |
| `core/core_brain.py` | Центральный координатор | ✅ Active |
| `core/brain_query.py` | Обработка запросов | ✅ Active |
| `core/model_access_manager.py` | Координация доступа | ✅ Active |
| `knowledge/concept_extractor.py` | Извлечение концептов | ✅ Active |
| `knowledge/concept_miner.py` | Глубокий анализ | ✅ Active |
| `memory/fractal_graph_v2/` | Фрактальный граф памяти | ✅ Core |
| `contradiction/` | Система противоречий | ✅ Active |
| `learning/dialog_core.py` | SelfDialogLearning | ✅ Active |

### 1.2 FCP - Модули для миграции

| Модуль | Функция | Назначение |
|--------|---------|-----------|
| `pipelines/mvp_pipeline_v15.py` | Полный FCP Pipeline v15 | ЯДРО |
| `fcp_gnn/hybrid_transformer_layer.py` | GNN инъекция на 32 слоях | GNN Integration |
| `fcp_gnn/graph_encoder.py` | SAGEConv + HNSW | Graph Encoder |
| `fcp_gnn/injector.py` | AdaptiveFusionInjector | Инъекция |
| `fcp_lora/shadow_lora_ov.py` | ShadowLoRAManagerOV | LoRA Management |
| `fcp_tools/orchestrator.py` | ToolOrchestrator | Tools |
| `fcp_tools/thinking_controller.py` | ThinkingController | Thinking Mode |
| `fcp_tools/expert_system.py` | ExpertSystem | Multi-agent |
| `fcp_knowledge/learning_manager.py` | LearningGraphManager | Learning |
| `fcp_knowledge/graph_curator.py` | GraphCurator | Curation |

### 1.3 FMF

| Компонент | Назначение |
|-----------|-----------|
| `model.ov` | OpenVINO модель |
| `src/` | Исходный код FMF |

---

## 2. План миграции

### Phase 1: Core Architecture (Приоритет HIGH)

| # | Действие | Источник | Назначение | Статус |
|---|----------|----------|------------|--------|
| 1.1 | REPLACE | FCP/pipelines/mvp_pipeline_v15.py | eva_ai/core/fcp_pipeline.py | [P] |
| 1.2 | REPLACE | FCP/fcp_gnn/hybrid_transformer_layer.py | eva_ai/core/hybrid_transformer_layer.py | [P] |
| 1.3 | REPLACE | FCP/fcp_core/adaptive_lora.py | eva_ai/core/adaptive_lora.py | [P] |

### Phase 2: Knowledge Systems (Приоритет HIGH)

| # | Действие | Источник | Назначение | Статус |
|---|----------|----------|------------|--------|
| 2.1 | MOVE | FCP/fcp_gnn/graph_encoder.py | eva_ai/knowledge/graph_encoder.py | [P] |
| 2.2 | MOVE | FCP/fcp_gnn/convert_gnn_to_ov.py | eva_ai/knowledge/gnn_converter.py | [P] |
| 2.3 | MOVE | FCP/fcp_knowledge/learning_manager.py | eva_ai/knowledge/learning_manager.py | [P] |
| 2.4 | MOVE | FCP/fcp_knowledge/graph_curator.py | eva_ai/knowledge/graph_curator.py | [P] |
| 2.5 | MOVE | FCP/fcp_data/ru_data_loaders.py | eva_ai/knowledge/data_loaders.py | [P] |

### Phase 3: Core Tools & Agents (Приоритет MEDIUM)

| # | Действие | Источник | Назначение | Статус |
|---|----------|----------|------------|--------|
| 3.1 | MOVE | FCP/fcp_tools/orchestrator.py | eva_ai/core/tool_orchestrator.py | [P] |
| 3.2 | MOVE | FCP/fcp_tools/thinking_controller.py | eva_ai/core/thinking_controller.py | [P] |
| 3.3 | MOVE | FCP/fcp_tools/expert_system.py | eva_ai/core/expert_system.py | [P] |
| 3.4 | MOVE | FCP/fcp_tools/attribution.py | eva_ai/core/attribution_report.py | [P] |
| 3.5 | MOVE | FCP/fcp_tools/clarification.py | eva_ai/core/clarification_generator.py | [P] |

### Phase 4: Memory Integration (Приоритет MEDIUM)

| # | Действие | Источник | Назначение | Статус |
|---|----------|----------|------------|--------|
| 4.1 | MOVE | FCP/fcp_tools/scenario_tcm.py | eva_ai/memory/scenario_tcm.py | [P] |
| 4.2 | MOVE | FCP/fcp_tools/semantic_cache_evictor.py | eva_ai/memory/semantic_evictor.py | [P] |
| 4.3 | MOVE | FCP/src/memory/graph_search.py | eva_ai/memory/fcp_graph_search.py | [P] |
| 4.4 | MERGE | FCP/src/memory/hybrid_cache.py + eva_ai/memory/ | eva_ai/memory/hybrid_cache.py | [P] |

### Phase 5: LoRA Management (Приоритет MEDIUM)

| # | Действие | Источник | Назначение | Статус |
|---|----------|----------|------------|--------|
| 5.1 | MOVE | FCP/fcp_lora/shadow_lora_ov.py | eva_ai/core/shadow_lora_manager.py | [P] |
| 5.2 | MOVE | FCP/fcp_gnn/injector.py | eva_ai/core/fusion_injector.py | [P] |

### Phase 6: Backward Compatibility (Приоритет LOW)

| # | Действие | Источник | Назначение | Статус |
|---|----------|----------|------------|--------|
| 6.1 | KEEP | eva_ai/core/hybrid_pipeline_adapter.py | eva_ai/core/hybrid_pipeline_adapter.py | [P] |
| 6.2 | KEEP | eva_ai/core/openvino_generator.py | eva_ai/core/openvino_generator.py | [P] |
| 6.3 | KEEP | eva_ai/knowledge/concept_extractor.py | eva_ai/knowledge/concept_extractor.py | [P] |

---

## 3. Структура после слияния

```
eva_ai/
├── core/                          # [NEW + OLD]
│   ├── fcp_pipeline.py            # NEW - FCPPipelineV15
│   ├── hybrid_transformer_layer.py # NEW - GNN на 32 слоях
│   ├── adaptive_lora.py           # NEW - AdaLoRA
│   ├── shadow_lora_manager.py    # NEW - LoRA management
│   ├── thinking_controller.py   # NEW - Thinking mode
│   ├── tool_orchestrator.py      # NEW - Tools
│   ├── expert_system.py           # NEW - Multi-agent
│   ├── fusion_injector.py         # NEW - Graph vector injection
│   ├── attribution_report.py     # NEW - Attribution
│   ├── clarification_generator.py # NEW - Clarifications
│   ├── hybrid_pipeline_adapter.py  # KEEP - Backup
│   ├── openvino_generator.py     # KEEP - Generator
│   ├── core_brain.py            # KEEP - Coordinator
│   └── ...
│
├── knowledge/                     # [EXPANDED]
│   ├── graph_encoder.py          # NEW - SAGEConv + HNSW
│   ├── gnn_converter.py          # NEW - PyTorch -> OpenVINO
│   ├── learning_manager.py       # NEW - LearningGraphManager
│   ├── graph_curator.py         # NEW - Graph curation
│   ├── data_loaders.py           # NEW - Data loaders
│   ├── concept_extractor.py     # KEEP - Fast extraction
│   ├── concept_miner.py          # KEEP - Deep mining
│   └── ...
│
├── memory/                        # [EXPANDED]
│   ├── scenario_tcm.py            # NEW - Episodic memory
│   ├── semantic_evictor.py       # NEW - Cache eviction
│   ├── fcp_graph_search.py      # NEW - FCP search
│   ├── hybrid_cache.py          # MERGE - Combined cache
│   ├── fractal_graph_v2/        # KEEP - Core graph
│   └── ...
│
├── fcp/                          # [DEPRECATED - Reference only]
│   └── (moved to core/knowledge/memory)
│
└── ...
```

---

## 4. Импорты после миграции

```python
# New imports pattern:
from eva_ai.core.fcp_pipeline import FCPPipelineV15
from eva_ai.core.hybrid_transformer_layer import HybridTransformerLayer
from eva_ai.core.adaptive_lora import AdaLoRALayer
from eva_ai.core.shadow_lora_manager import ShadowLoRAManagerOV
from eva_ai.knowledge.graph_encoder import FractalGraphEncoder
from eva_ai.knowledge.learning_manager import LearningGraphManager
from eva_ai.core.tool_orchestrator import ToolOrchestrator
from eva_ai.core.expert_system import ExpertSystem
```

---

## 5. Тестирование после каждой фазы

### Phase 1 Тесты:
- [ ] Import FCPPipelineV15
- [ ] Test generation with pipeline
- [ ] Test LoRA adapter loading

### Phase 2 Тесты:
- [ ] Test GraphEncoder encoding
- [ ] Test LearningManager signals
- [ ] Test GraphCurator cycles

### Phase 3 Тесты:
- [ ] Test ToolOrchestrator tools
- [ ] Test ThinkingController
- [ ] Test ExpertSystem multi-agent

### Phase 4 Тесты:
- [ ] Test ScenarioTCM save/load
- [ ] Test Cache eviction
- [ ] Test Graph search

### Phase 5 Тесты:
- [ ] Test LoRA swap
- [ ] Test Fusion injection

### Integration Tests:
- [ ] Full pipeline end-to-end
- [ ] EVA startup with new components
- [ ] Memory graph integration

---

## 6. Коммиты

| # | Коммит | Содержание |
|---|-------|------------|
| 1 | "Add FCPPipelineV15 to core" | Phase 1 |
| 2 | "Add GNN/learning to knowledge" | Phase 2 |
| 3 | "Add tools to core" | Phase 3 |
| 4 | "Add memory extensions" | Phase 4 |
| 5 | "Add LoRA management" | Phase 5 |
| 6 | "Final integration and tests" | Integration |

---

## 7. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Import conflicts | HIGH | Use `__as_import__` aliases |
| Broken imports | HIGH | Test after each phase |
| Performance regression | MEDIUM | Benchmark before/after |
| Missing dependencies | MEDIUM | Add to requirements.txt |

---

## 8. Backward Compatibility

Старые компоненты будут сохранены но помечены как deprecated:
- `eva_ai/core/hybrid_pipeline_adapter.py` → alias to new
- `eva_ai/core/openvino_generator.py` → используется если FCPPipelineV15 не доступен

---

## Execution

Начинаем? Подтверди - запускаю Phase 1.