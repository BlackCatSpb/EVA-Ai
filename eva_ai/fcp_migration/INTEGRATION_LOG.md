# Журнал интеграции EVA-Ai + FCP

**Дата начала:** 25.04.2026  
**Статус:** В процессе

---

## Компонент 1: FCPPipelineV15 (Ядро генерации)

### Текущая реализация (EVA)
**Файл:** `eva_ai/core/hybrid_pipeline_adapter.py`
**Класс:** `HybridPipelineAdapter`
**Функции:**
- `process_query()` - обработка запроса
- `_init_pipelines()` - инициализация пайплайнов
- DualGenerator с 2 физическими моделями
- Режимы: 'fractal', 'dual', 'recursive', 'hybrid'

### Требуется интегрировать (FCP)
**Файл:** `eva_ai/fcp_migration/pipelines/mvp_pipeline_v15.py`
**Класс:** `FCPPipelineV15`
**Функции:**
- `generate()` - генерация с GNN инъекцией
- `_init_gnn()` - инициализация GNN
- `_init_lora_manager()` - LoRA management
- `_init_tools()` - ToolOrchestrator, ThinkingController
- `_init_knowledge()` - LearningGraphManager, GraphCurator
- LoRA адаптеры, атрибуция

### План интеграции:
1. Скопировать FCPPipelineV15 в `core/fcp_pipeline.py`
2. Исправить импорты: `from fcp_*` → `from eva_ai.*`
3. Адаптировать конструктор для использования config
4. Добавить factory в `init_factories.py`
5. Подключить к CoreBrain как `fcp_pipeline`

### Статус: [ ] Ожидает

---

## Компонент 2: GraphEncoder (GNN)

### Текущая реализация (EVA)
**Файл:** Нет прямого аналога
**Примечание:** Используется FractalGraphV2 для поиска, но без GNN энкодера

### Требуется интегрировать (FCP)
**Файл:** `eva_ai/fcp_migration/fcp_gnn/graph_encoder.py`
**Класс:** `GraphEncoderRuntime`, `FractalGraphEncoder`
**Функции:**
- `encode()` - получение graph vector
- `retrieve_subgraph()` - HNSW поиск
- SAGEConv архитектура

### План интеграции:
1. Скопировать в `knowledge/fcp_graph_encoder.py`
2. Исправить импорты
3. Подключить к FCPPipelineV15
4. Интегрировать с FractalGraphV2

### Статус: [ ] Ожидает

---

## Компонент 3: HybridTransformerLayer (GNN на всех слоях)

### Текущая реализация (EVA)
**Файл:** Нет
**Примечание:** Обычная генерация без послойной GNN инъекции

### Требуется интегрировать (FCP)
**Файл:** `eva_ai/fcp_migration/fcp_gnn/hybrid_transformer_layer.py`
**Класс:** `HybridTransformerLayer`, `HybridModelWithGNN`
**Функции:**
- GNN injection на всех 32 слоях
- AdaptiveFusionInjector
- LoRA adapter integration

### План интеграции:
1. Скопировать в `core/hybrid_transformer_layer.py`
2. Конвертировать GNN в OpenVINO формат
3. Исправить импорты
4. Подключить к FCPPipelineV15

### Статус: [ ] Ожидает

---

## Компонент 4: ShadowLoRAManagerOV

### Текущая реализация (EVA)
**Файл:** Нет прямого аналога
**Примечание:** Частичная поддержка LoRA через HybridPipelineAdapter

### Требуется интегрировать (FCP)
**Файл:** `eva_ai/fcp_migration/fcp_lora/shadow_lora_ov.py`
**Класс:** `ShadowLoRAManagerOV`
**Функции:**
- `register_adapter()` - регистрация адаптера
- `atomic_swap()` - атомарная замена
- `rollback()` - откат при деградации

### План интеграции:
1. Скопировать в `core/shadow_lora_manager.py`
2. Исправить импорты
3. Подключить к FCPPipelineV15
4. Интегрировать с LearningGraphManager

### Статус: [ ] Ожидает

---

## Компонент 5: LearningGraphManager

### Текущая реализация (EVA)
**Файл:** `eva_ai/knowledge/concept_miner.py`
**Класс:** `ConceptMiner`
**Функции:**
- Извлечение концептов из кластеров
- Сигналы обратной связи

### Требуется интегрировать (FCP)
**Файл:** `eva_ai/fcp_migration/fcp_knowledge/learning_manager.py`
**Класс:** `LearningGraphManager`, `LearningOrchestrator`
**Функции:**
- `add_signal()` - добавление сигнала
- `get_layer_sensitivity()` - статистика по слоям
- Планирование дообучения

### План интеграции:
1. Скопировать в `knowledge/fcp_learning_manager.py`
2. Исправить импорты
3. Интегрировать с ConceptMiner (объединить функциональность)
4. Подключить к CoreBrain

### Статус: [ ] Ожидает

---

## Компонент 6: ToolOrchestrator

### Текущая реализация (EVA)
**Файл:** `eva_ai/websearch/web_search_integrated.py`
**Класс:** `IntegratedWebSearchEngine`
**Функции:** Веб-поиск через Tavily

### Требуется интегрировать (FCP)
**Файл:** `eva_ai/fcp_migration/fcp_tools/orchestrator.py`
**Класс:** `ToolOrchestrator`
**Функции:**
- WebSearch (Exa + DuckDuckGo)
- Weather (Open-Meteo)
- Translator (MyMemory)
- Calculator

### План интеграции:
1. Скопировать в `tools/fcp_orchestrator.py`
2. Исправить импорты
3. Сохранить EVA веб-поиск как fallback
4. Подключить к FCPPipelineV15

### Статус: [ ] Ожидает

---

## Компонент 7: ThinkingController

### Текущая реализация (EVA)
**Файл:** Нет прямого аналога
**Примечание:** Qwen3 thinking mode не управляется динамически

### Требуется интегрировать (FCP)
**Файл:** `eva_ai/fcp_migration/fcp_tools/thinking_controller.py`
**Класс:** `ThinkingController`, `SimpleRoutingEngine`
**Функции:**
- `should_enable_thinking()` - решение на основе противоречий
- `build_chat_prompt()` - формирование промпта

### План интеграции:
1. Скопировать в `tools/fcp_thinking_controller.py`
2. Исправить импорты
3. Подключить к ContradictionDetector
4. Добавить в FCPPipelineV15

### Статус: [ ] Ожидает

---

## Компонент 8: ScenarioTCM

### Текущая реализация (EVA)
**Файл:** `eva_ai/memory/memory_working.py`
**Класс:** `WorkingMemory`
**Функции:** Краткосрочное хранение диалогов

### Требуется интегрировать (FCP)
**Файл:** `eva_ai/fcp_migration/fcp_tools/scenario_tcm.py`
**Класс:** `ScenarioTCM`
**Функции:**
- `add_turn()` - сохранение цепочек диалогов
- Семантическое извлечение по сценариям

### План интеграции:
1. Скопировать в `memory/fcp_scenario_tcm.py`
2. Исправить импорты
3. Интегрировать с WorkingMemory
4. Подключить к FCPPipelineV15

### Статус: [ ] Ожидает

---

## Компонент 9: ExpertSystem

### Текущая реализация (EVA)
**Файл:** Нет прямого аналога
**Примечание:** Multi-agent обсуждения через SelfDialogLearning

### Требуется интегрировать (FCP)
**Файл:** `eva_ai/fcp_migration/fcp_tools/expert_system.py`
**Класс:** `ExpertSystem`
**Функции:**
- Множественные эксперты с разными LoRA
- Critic для выявления противоречий
- Голосование/обсуждение

### План интеграции:
1. Скопировать в `tools/fcp_expert_system.py`
2. Исправить импорты
3. Интегрировать с SelfDialogLearning
4. Подключить к FCPPipelineV15

### Статус: [ ] Ожидает

---

## Компонент 10: SemanticCacheEvictor

### Текущая реализация (EVA)
**Файл:** `eva_ai/memory/cache_eviction.py`
**Класс:** `CacheEvictionPolicy`
**Функции:** LRU, LFU eviction стратегии

### Требуется интегрировать (FCP)
**Файл:** `eva_ai/fcp_migration/fcp_tools/semantic_cache_evictor.py`
**Класс:** `SemanticCacheEvictor`
**Функции:**
- Анализ важности токенов по близости к графу
- Продление жизни важных блоков кэша

### План инт��грации:
1. Скопировать в `memory/fcp_semantic_evictor.py`
2. Исправить импорты
3. Интегрировать с существующим cache_eviction
4. Подключить к HybridTokenCache

### Статус: [ ] Ожидает

---

## Итоговая структура после интеграции:

```
eva_ai/
├── core/
│   ├── fcp_pipeline.py              # NEW - FCPPipelineV15
│   ├── hybrid_transformer_layer.py  # NEW - GNN на всех слоях
│   ├── shadow_lora_manager.py       # NEW - LoRA management
│   └── hybrid_pipeline_adapter.py  # KEEP - backup
│
├── knowledge/
│   ├── fcp_graph_encoder.py         # NEW - GraphEncoder
│   ├── fcp_learning_manager.py      # NEW - LearningGraphManager
│   └── (EVA files remain)
│
├── memory/
│   ├── fcp_scenario_tcm.py          # NEW - ScenarioTCM
│   ├── fcp_semantic_evictor.py      # NEW - CacheEvictor
│   └── (EVA files remain)
│
├── tools/
│   ├── fcp_orchestrator.py          # NEW - ToolOrchestrator
│   ├── fcp_thinking_controller.py  # NEW - ThinkingController
│   ├── fcp_expert_system.py         # NEW - ExpertSystem
│   └── (EVA files remain)
│
└── models/
    └── ruadapt_qwen3_4b_openvino/   # Existing
```

---

## Конфигурация:

```json
{
  "fcp_pipeline": {
    "enabled": true,
    "model_path": "models/ruadapt_qwen3_4b_openvino",
    "gnn_ov_path": "models/gnn_encoder.xml",
    "lora_dir": "lora/fcp_finetuned",
    "use_injection": true,
    "use_lora": true
  }
}
```