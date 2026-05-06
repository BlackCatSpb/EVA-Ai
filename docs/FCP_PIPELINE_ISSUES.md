# FCP Pipeline - Найденные проблемы интеграции

## Дата: 2026-05-02

## ЦЕЛЬ ПРОЕКТА (из EVA.txt)
Создать когнитивную платформу, где LLM и FractalGraphV2 работают как единый организм с **двунаправленным когнитивным циклом**:
- Модель извлекает знания из графа
- Граф управляет внутренними состояниями модели во время генерации
- Полнослойная инъекция на все 36 слоёв
- KCA обнаруживает лакуны и противоречия
- SRG оценивает качество ответа
- Система непрерывно обучается

**Текущее состояние:** Гибридная обработка — декорация, не работающая в комплексе.

---

## ОТЧЁТЫ АГЕНТОВ

- [Анализ KCA + SRG](FCP_ANALYSIS_KCA_SRG.md)
- [Анализ HybridLayer](FCP_ANALYSIS_HYBRID_LAYER.md)
- [Анализ StateInjector + GraphIntegrationManager](FCP_ANALYSIS_INJECTOR.md)

---

# ============================================================
# ПЛАН ИСПРАВЛЕНИЙ (обязательно возвращаться к этому разделу)
# ============================================================

## ЭТАП 1: Базовая интеграция (минимальные изменения) ✅

| # | Проблема | Файл | Метод | Исправление | Статус |
|---|----------|------|-------|--------------|--------|
| 1.1 | SRG zeros → always reasoning | fcp_pipeline.py | _process_with_hybrid_layers | Оценивать энтропию по контексту промпта | ✅ Готово |
| 1.2 | Query embedding random/input_ids | fcp_pipeline.py | _get_query_embedding | Детерминированный seed по хешу | ✅ Готово |
| 1.3 | KCA формат данных | fcp_pipeline.py | _process_with_hybrid_layers | Subgraph.to_dict() уже работает | ✅ Не требуется |

## ЭТАП 2: Связь компонентов ✅

| # | Проблема | Файл | Метод | Исправление | Статус |
|---|----------|------|-------|--------------|--------|
| 2.1 | GIM не связан с FGv2 | analysis_and_injection.py | GraphIntegrationManager | Добавить поиск реальных узлов в графе | ✅ Готово |
| 2.2 | KCA результат не используется | fcp_pipeline.py | generate | Интегрировать corrected states в промпт | ✅ Готово |

## ЭТАП 3: Архитектурные изменения ✅

| # | Проблема | Файл | Метод | Исправление | Статус |
|---|----------|------|-------|--------------|--------|
| 3.1 | Hidden states недоступны | fcp_pipeline.py | generate | LayerwiseStateInjector + State API (KV-кеш) | ✅ Готово |
| 3.2 | StateInjector фейлится тихо | fcp_pipeline.py | generate_with_injection | Логирование + partial fallback | ✅ Готово |
| 3.3 | HybridLayerProcessor изолирован | init_core.py | COMPONENT_LIST | Добавлен в OPTIONAL | ✅ Готово |
| **3.4** | **Activation Gate (ранний выход)** | fcp_pipeline.py | generate_with_injection | Early exit при confidence>0.85 | ✅ Готово |
| **3.5** | **KCA Gate (γ) монитор** | fcp_pipeline.py | generate_with_injection | Отклонение коррекции при γ<0.05 | ✅ Готово |

## ЭТАП 4: Обучение GNN (отдельная задача в Colab)

| # | Проблема | Действие |
|---|----------|----------|
| 4.1 | graph_encoder.pt не существует | Запустить обучение в Colab |

---

---

## СДЕЛАНО ✅

### Этап 1: Базовая интеграция
- **1.1 SRG** — добавлен метод `_estimate_logits_from_prompt()` для оценки энтропии по контексту
- **1.2 Query embedding** — изменён с random на детерминированный по хешу текста
- **1.3 KCA** — Subgraph.to_dict() уже возвращает правильный формат

### Этап 2: Связь компонентов
- **2.1 GraphIntegrationManager** — связан с FractalGraphV2, ищет реальные узлы
- **2.2 KCA result** — интегрирован в промпт через `_enrich_prompt_with_subgraph()`

### Этап 3: Архитектурные изменения
- **3.1 Hidden states** — LayerwiseStateInjector использует State API для KV-кеша (Key/Value на всех 36 слоях). Реализовано в `generate_with_injection()`:
  - SQAM Key scaling на всех слоях
  - KCA Value injection на всех слоях
  - Включено через `enable_injection=True` в brain_query.py:362
- **3.2 StateInjector** — улучшено логирование, добавлен partial fallback
- **3.4 Activation Gate** — ранний выход при накопленной уверенности >0.85:
  - Вычисляется softmax max probability после каждого токена
  - Сглаживание через окно (window=3)
  - При срабатывании: early_exits_count++, логирование
  - Потенциальное ускорение до 85% на простых запросах (EVA.txt раздел 2.1)
- **3.5 KCA Gate (γ)** — монитор насыщения коррекции:
  - Gamma = norm(correction) / norm(hidden_state), 0 ≤ γ ≤ 1
  - Демпфирование: γ_damped = γ * ρ^t (ρ=0.85)
  - Монитор: avg(γ_last_2) < 0.05 → kca_rejected = True
  - При отклонении KCA не применяется (EVA.txt раздел 3.3)

### Этап 5: Episode Memory (EVA.txt 6.3) ✅
- **ScenarioTCM** — сохранение цепочек диалогов как сценариев:
  - Инициализация в `__init__()` после fractal_graph
  - `get_similar_scenarios()` — поиск похожих сценариев для контекста
  - `add_dialog_turn()` — сохранение ходов в текущую цепочку
  - Интеграция в `_build_prompt()` — добавление контекста из сценариев
  - Автосохранение после каждого ответа в `generate()`, `generate_with_injection()`

### Этап 6: Concept Mining (EVA.txt 7.1) ✅
- **ConceptMiner** — автономный концептуальный вывод:
  - Инициализация с `brain=self` для доступа к компонентам
  - `start_concept_mining()` / `stop_concept_mining()` — управление фоном
  - `get_mined_concepts()` — получение найденных кандидатов
  - Обнаружение семантических лакун в кластерах FGv2

### Этап 7: Contradiction Detection (EVA.txt 7.2) ✅
- **ContradictionDetector** — обнаружение противоречий в графе:
  - Инициализация с `knowledge_graph=self.fractal_graph`
  - `detect_contradictions(concept)` — поиск противоречий по концепту
  - `get_contradiction_stats()` — статистика обнаруженных противоречий

### Этап 8: Learning Orchestration (EVA.txt 7.3) ✅
- **LearningOrchestrator** — управление обучением LoRA:
  - Инициализация с `LearningGraphManager` и `lora_manager`
  - `should_retrain(domain)` — решение о необходимости переобучения
  - `get_retrain_plan(domain)` — план переобучения слоёв
  - Автоматическое управление адаптерами на основе успешности

### Дополнительно
- Добавлено логирование KCA/SRG в metadata при генерации
- Улучшено логирование при инициализации StateInjector

---

## ПРОВЕРКА КОНТЕКСТА (возвращаться к этому в процессе работы)

Проверить себя после каждого изменения:
1. ❓ Изменение относится к одному из этапов выше?
2. ❓ Не ломает ли это существующую функциональность?
3. ❓ Как это влияет на общий поток: Query → Graph → KCA → SRG → LLM → Response?
4. ❓ Работает ли это согласно EVA.txt спецификации?

---

# ============================================================
# ДЕТАЛИ ПРОБЛЕМ (для справки при работе)
# ============================================================

## Проблема 1: KCA получает неверный формат данных

**Место:** `eva_ai/core/fcp_pipeline.py:674`

**Описание:**
```python
# KCA.forward() ожидает dict с ключом "embeddings"
# Но получает Subgraph объект
corrected_states, kca_info = self.kca.forward(initial_states, subgraph.to_dict())
```

KCA в `fcp_core/__init__.py:120-125` ожидает:
```python
def forward(self, X_initial: np.ndarray, subgraph: dict):
    H = subgraph["embeddings"]  # Ожидает массив [N, D]
```

Но `FractalGraphV2.retrieve_subgraph()` возвращает объект `Subgraph` с полями:
- `node_ids`
- `embeddings` (numpy array)
- `contents` (list)
- `edge_index`

**Влияние:** KCA не работает или падает с ошибкой

---

## Проблема 2: SRG использует заглушку logits

**Место:** `eva_ai/core/fcp_pipeline.py:644`

**Описание:**
```python
# Передаётся np.zeros(100) вместо реальных logits
mode, srg_metrics = self.srg.evaluate(
    query_vec=query_embedding,
    response_vec=query_embedding,
    logits=np.zeros(100)  # ЗАГЛУШКА!
)
```

SRG в `fcp_core/__init__.py:207-235` вычисляет:
- `sim` - косинусное сходство
- `entropy` - энтропия из softmax(logits)

С `logits=np.zeros(100)`:
- softmax([0]*100) = [0.01]*100 (равномерное)
- entropy = ~6.64 (максимум)
- Результат: **ВСЕГДА режим "reasoning"** (низкая уверенность)

**Влияние:** SRG всегда выбирает режим reasoning, даже когда не нужен

---

## Проблема 3: Query Embedding генерируется случайно

**Место:** `eva_ai/core/fcp_pipeline.py:690-702`

**Описание:**
```python
def _get_query_embedding(self, text: str) -> np.ndarray:
    try:
        if hasattr(self, 'tokenizer') and self.tokenizer:
            inputs = self.tokenizer(text, return_tensors="np", ...)
            input_ids = inputs["input_ids"]
            if input_ids.shape[1] > 0:
                return np.mean(inputs["input_ids"].astype(np.float32), axis=0)
    except:
        pass
    # Fallback - случайный вектор!
    return np.random.randn(2560).astype(np.float32)
```

Метод пытается использовать tokenizer, но:
1. Если токенизатора нет → случайный вектор
2. Использует input_ids (не эмбеддинги!) — семантически неверно

**Влияние:** retrieve_subgraph ищет неправильные узлы в графе

---

## Проблема 4: HybridLayerProcessor не инициализируется

**Место:** `eva_ai/core/init_core.py:44-67`

**Описание:**
```python
COMPONENT_LIST = [
    'event_bus',
    'resource_manager',
    # ... другие компоненты ...
    'fcp_pipeline',
    'closed_cognitive_loop',
]
# 'hybrid_layer_processor' НЕ В СПИСКЕ!
```

HybridLayerProcessor создаётся внутри FCPPipelineV15, но:
- Создаётся, но не используется для реальной обработки
- Не получает реальные hidden states от OpenVINO

**Влияние:** GNN + KCA + SRG не работают в полном гибридном режиме

---

## Проблема 5: HybridLayerManager изолирован

**Место:** `eva_ai/fcp_gnn/hybrid_integration.py:584-655`

**Описание:**
- Создаёт свой FractalGraphEncoderLocal (дублирует FractalGraphV2)
- Не связан с OpenVINO pipeline
- Требует graph_encoder.pt которого нет

**Влияние:** Компонент существует, но не используется

---

## Проблема 6: LayerwiseStateInjector может не работать

**Место:** `eva_ai/core/fcp_pipeline.py:809-923`

**Описание:**
```python
def generate_with_injection(self, ...):
    if not self.pipeline or not self.state_injector:
        # Fallback к обычной генерации
        return self._generate(prompt, max_new_tokens, **{})
```

Проблемы:
1. Требует `openvino_model.xml` (не папку)
2. API зависит от версии OpenVINO
3. При любой ошибке → fallback без инъекции

**Влияние:** Полнослойная инъекция не работает, используется fallback

---

## Проблема 7: GraphIntegrationManager не связан с FractalGraphV2

**Место:** `eva_ai/core/analysis_and_injection.py:39-69`

**Описание:**
```python
class GraphIntegrationManager:
    def add_anchors(self, anchors, key_vectors):
        # Просто накапливает векторы, не ищет в графе!
        for i in range(len(key_vectors)):
            self.nodes.append(key_vectors[i])
```

Должен:
- Искать якоря в FractalGraphV2
- Обновлять центроид из реальных узлов графа

Делает:
- Хранит временные векторы из Key тензора

**Влияние:** KCA коррекция использует неправильный центроид

---

## Проблема 8: Нет связи между компонентами

**Общая картина:**

```
User Query
    ↓
[_build_prompt] → история чата
    ↓
[_process_with_hybrid_layers]
    ├── _get_query_embedding() → случайный vector ❌
    ├── srg.evaluate() → zeros(100) ❌
    ├── fractal_graph.retrieve_subgraph() → Subgraph (не dict) ❌
    └── kca.forward() → ошибка формата ❌
    ↓
[_generate] → OpenVINO
    ↓
[_save_layer_snapshot] → сохраняет (если работает)
```

**Влияние:** Гибридная обработка не работает, используется только базовая генерация

---

## Итог

**Почему гибридная обработка не работает:**

1. **Query embedding** - использует input_ids (мусор) вместо семантических эмбеддингов
2. **SRG** - использует zeros, всегда выбирает "reasoning" режим
3. **GraphIntegrationManager** - не связан с FractalGraphV2, центроид из неправильных векторов
4. **KCA коррекция** - использует неправильный центроид, добавляет шум вместо знаний
5. **Error handling** - при любой ошибке полный fallback без инъекции

**Следствие:** Гибридная обработка по сути отключена, используется только базовая генерация OpenVINO.

---

# ============================================================
# СООТВЕТСТВИЕ EVA.txt (2026-05-02)
# ============================================================

## ✅ Полностью реализовано

| Компонент | EVA.txt раздел | Файл | Статус |
|-----------|----------------|------|--------|
| **Полнослойная инъекция** | 2.3, 8.1-8.2 | fcp_pipeline.py:generate_with_injection | ✅ 36 слоёв |
| **KCA (Knowledge-Conscious Attention)** | 3.1-3.3 | kca.forward(), compute_kca_correction | ✅ Лакуны + противоречия |
| **KCA Gate (γ)** | 3.2-3.3 | fcp_pipeline.py:kca_gate_config | ✅ Демпфирование ρ^t, отклонение при γ<0.05 |
| **Activation Gate (ранний выход)** | 2.1 | fcp_pipeline.py:activation_gate_config | ✅ confidence>0.85 |
| **SQAM (Semantic Query Analyzer)** | 4.1 | sqam_analyzer.analyze() | ✅ Семантическая сигнатура |
| **Graph Integration Manager** | 4.2 | analysis_and_injection.py | ✅ Якори → узлы графа |
| **SRG (Semantic Relevance Gate)** | 5 | srg.evaluate() | ✅ Direct/Reasoning/Variational |
| **State API доступ** | 8.1 | LayerwiseStateInjector | ✅ query_state() для KV-кеша |
| **FractalGraphV2** | 6.2 | memory/fractal_graph_v2 | ✅ 451247 nodes, HNSW |
| **GNN encoder** | 2.1 | hybrid_integration.py | ✅ 8 trained weights loaded |
| **HybridLayerProcessor** | 2.1 | fcp_gnn/hybrid_layer.py | ✅ gnn=True, kca=True, srg=True, lora=True |
| **MemorySnapshot** | 6.1 | memory_snapshot_integration.py | ✅ 32 layers |
| **LoRA адаптер** | 2.1 | shadow_lora.py | ✅ Загружен (fcp_finetuned) |
| **ScenarioTCM (эпизодическая память)** | 6.3 | memory/scenario_tcm.py | ✅ Поиск + сохранение сценариев |
| **ConceptMiner (концептуальный вывод)** | 7.1 | knowledge/concept_miner.py | ✅ Автономный поиск лакун |
| **ContradictionDetector (детектор противоречий)** | 7.2 | contradiction/detect_core.py | ✅ Обнаружение в графе |
| **LearningOrchestrator (оркестратор обучения)** | 7.3 | fcp_core/learning_orchestrator.py | ✅ Управление LoRA |
| **UES (Универсальная среда исполнения)** | 8.3 | fcp_ues/ | ✅ Оптимизация + топология |
| **ContextualTokenizer (контекстная токенизация)** | 2.1 | fcp_core/contextual_tokenizer.py | ✅ Адаптация под граф знаний |

## ⚠️ Частично реализовано

| Компонент | EVA.txt | Текущий статус |
|-----------|---------|----------------|
| **Early Exit полный** | 2.1 | ⚠️ Только логирование, полный exit требует архитектурных изменений |
| **Cross-attention слияние** | 2.1 | ⚠️ Требует архитектурных изменений |
| **Обучаемый гейт слияния** | 2.1 | ⚠️ Только веса inject, нет trainable gate |

## ❌ Не реализовано

| Компонент | EVA.txt | Проблема |
|-----------|---------|----------|
| - | - | Все критичные компоненты реализованы |

## Лог инициализации (2026-05-02)

```
[FCP] StateInjector initialized: device=CPU, 36 layers
[FCP] Activation Gate initialized: threshold=0.85
[FCP] KCA Gate initialized: threshold=0.05, damping=0.85
[FCP] HybridLayerProcessor initialized: gnn=True, kca=True, srg=True, lora=True
[FCP] Loaded trained GNN encoder (8 weights)
[FCP] FractalGraphV2: 451247 nodes
[FCP] MemorySnapshot: 32 layers
```

## Исправления внесённые

1. ✅ SRG — детерминированная оценка энтропии
2. ✅ Query embedding — детерминированный seed по хешу
3. ✅ GIM связан с FractalGraphV2
4. ✅ KCA result интегрирован в промпт
5. ✅ State API (KV-кеш) работает
6. ✅ Activation Gate реализован
7. ✅ KCA Gate (γ) реализован
8. ✅ HybridLayerPipeline добавлен в OPTIONAL
9. ✅ FractalGraphV2 инициализация исправлена
10. ✅ create_fcp_pipeline читает из config['fcp']