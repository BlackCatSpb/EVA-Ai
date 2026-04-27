# Анализ FCP System EVA (Fractal Cognitive Processor)

## Содержание

1. [Обзор архитектуры FCP](#1-обзор-архитектуры-fcp)
2. [Гибридная модель GNN + Transformer](#2-гибридная-модель-gnn--transformer)
3. [Ключевые классы и компоненты](#3-ключевые-классы-и-компоненты)
4. [Интеграция с EVA через HybridPipelineAdapter](#4-интеграция-с-eva-через-hybridpipelineadapter)
5. [Проблемы и нереализованные функции](#5-проблемы-и-нереализованные-функции)
6. [Рекомендации по развитию](#6-рекомендации-по-развитию)

---

## 1. Обзор архитектуры FCP

### 1.1 Общая структура

Fractal Cognitive Processor (FCP) — это полная когнитивная система с гибридной архитектурой, объединяющая:

| Компонент | Описание | Файл |
|-----------|----------|------|
| **GNN (Graph Neural Network)** | Энкодер графовых структур с SAGEConv и HNSW | fcp_gnn/graph_encoder.py |
| **Transformer** | Базовый LLM (Qwen3-4B) с 36 слоями | fcp_gnn/hybrid_transformer_layer.py |
| **LoRA** | Адаптивная тонкая настройка с динамическим рангом | fcp_core/adaptive_lora.py |
| **KCA** | Knowledge Conscious Attention — коррекция через лакуны и противоречия | fcp_core/__init__.py |
| **SRG** | Semantic Relevance Gate — решение о режиме генерации | fcp_core/__init__.py |
| **FractalGraphV2** | Векторный граф знаний | fcp_core/fractal_graph.py |

### 1.2 Архитектура обработки

`
User Query -> FractalGraphV2 (retrieve) -> GNN Encoder -> graph_vec
                                                         |
                                                     SRG (gate)
                              +-------------+------------+
                              |             |            |
                        DIRECT MODE    REASONING   VARIATIONAL
                              |             |            |
                              |        KCA cycles        |
                              |        (correction)      |
                              +-------------------------+
                                           |
                              Hidden States (corrected)
                                           |
                              LoRA Adapter (optional)
                                           |
                              Base LLM Forward
                                           |
                              Response Generation
`

### 1.3 Ключевые параметры конфигурации

Из fcp_core/config.py:

- num_layers: 36
- embedding_dim: 2560
- num_heads: 32
- kv_heads: 8
- head_dim: 128
- intermediate_size: 9728
- max_seq_len: 262144

Graph параметры:
- graph_retrieval_k: 32
- master_tokens: 8
- gnn_iterations: 2

KCA параметры:
- kca_max_cycles: 5
- kca_rho: 0.85
- lambda_l: 0.5 (лакуны)
- lambda_c: 0.5 (противоречия)

SRG параметры:
- srg_cosine_threshold: 0.85
- srg_entropy_threshold: 2.0

---

## 2. Гибридная модель GNN + Transformer

### 2.1 Принцип работы

Гибридная модель FCP работает по принципу послойной инъекции — на каждом из 36 слоёв модели выполняется:

1. Извлечение подграфа — из FractalGraphV2 получаем K ближайших узлов
2. GNN кодирование — преобразуем подграф в graph_vec
3. SRG решение — определяем режим (direct/reasoning/variational)
4. KCA коррекция (если reasoning) — исправляем скрытые состояния
5. LoRA адаптация — применяем тонкую настройку
6. Слияние потоков — объединяем LM и GNN информацию

### 2.2 Режимы SRG (Semantic Relevance Gate)

| Режим | Условие | Описание |
|-------|---------|----------|
| direct | cos >= 0.85 AND ent <= 2.0 | Модель уверена, возвращаем ответ напрямую |
| reasoning | cos < 0.85 | Низкое сходство, нужна коррекция через KCA |
| variational | cos >= 0.85 AND ent > 2.0 | Высокая неопределённость, нужен вариативный подход |

### 2.3 KCA (Knowledge Conscious Attention)

KCA выполняет итеративную коррекцию скрытых состояний:

1. Attention: Токены -> Узлы графа
2. Вектор коррекции (лакуны + противоречия)
3. Адаптивное затухание (damping = rho ^ t)
4. Гейт для контроля инъекции
5. Проверка сходимости через ConvergenceController

Цикл повторяется max 5 раз (kca_max_cycles) или до сходимости.

### 2.4 LoRA с адаптивным рангом

AdaLoRA поддерживает динамический ранг:

- P и Q матрицы: [hidden_dim, max_rank]
- Диагональная матрица для масштабирования
- Метод adapt_rank() для динамического изменения

Распределение рангов по слоям:
- Слои 0-3: rank=4 (факты)
- Слои 4-11: rank=8 (рассуждения)
- Слои 12-35: rank=16 (творческие задачи)

---

## 3. Ключевые классы и компоненты

### 3.1 HybridTransformerLayer

Файл: eva_ai/fcp_gnn/hybrid_transformer_layer.py

Гибридный слой с GNN инъекцией на КАЖДОМ из 32+ слоёв:

- Standard Transformer: attention + FFN
- GNN Инъекция: на всех слоях (не только 4, 8, 16, 24)
- LoRA адаптер для тонкой настройки
- Адаптивный гейт для контроля инъекции

Особенности:
- GNN инъекция на всех слоях
- Адаптивный гейт для контроля силы инъекции
- LoRA с динамическим рангом

### 3.2 FractalGraphEncoder

Файл: eva_ai/fcp_gnn/graph_encoder.py

GNN энкодер с архитектурой SAGEConv + HNSW:

- SAGEConv: input -> hidden -> output
- Linear proj для output
- Gate projection для адаптивной инъекции
- HNSW индекс для поиска

Также есть numpy-only версия — GraphEncoderRuntime для OpenVINO без torch-geometric.

### 3.3 FractalGatedHybridLayer

Файл: eva_ai/fcp_core/hybrid_layer.py

Полный гибридный слой с 5 этапами обработки:

1. Контекстуальный токенизатор (graph retrieval + node routing)
2. Графовый кластеризатор (message passing + soft clustering)
3. Трансформерный блок (attention + FFN)
4. Активационный гейт (halt probability)
5. Слияние потоков (cross-attention или gated add)

### 3.4 AdaLoRA (Adaptive LoRA)

Файл: eva_ai/fcp_core/adaptive_lora.py

Адаптивный LoRA с динамическим рангом:

- Инициализация с max_rank (32)
- Динамическое изменение rank через adapt_rank()
- Диагональная матрица для масштабирования

MultiRankAdapter для разных типов задач:
- r=4 для первых слоёв (facts)
- r=8 для средних слоёв (reasoning)
- r=16 для последних слоёв (creative)

### 3.5 HybridLayerProcessor

Файл: eva_ai/fcp_gnn/hybrid_integration.py

Интегратор всех компонентов:

- LLM forward -> hidden_states
- GNN retrieval -> subgraph + graph_vec
- SRG evaluation -> mode (direct/reasoning/variational)
- KCA correction (if reasoning mode)
- LoRA injection (if enabled)
- Return corrected states + text

---

## 4. Интеграция с EVA через HybridPipelineAdapter

### 4.1 FCPPipelineV15

Файл: eva_ai/core/fcp_pipeline.py

Основной pipeline FCP:

- FCP Core Components (KCA, SRG, ConvergenceController, FractalGraphV2)
- Hybrid Layer Components (HybridLayerProcessor, HybridLayerManager, MemorySnapshot)
- OpenVINO LLMPipeline (base model)
- LoRA Manager (dynamic adapters)

### 4.2 Интеграция с EVA Core

Согласно AGENTS.md, FCPPipelineV15 интегрирован в EVA через HybridPipelineAdapter:

`
CoreBrain
├── event_bus (EventBus)
├── deferred_system (DeferredCommandSystem)
├── two_model_pipeline (PipelineAdapter)
│   ├── _openvino_cpu (OpenVINOGenerator)
│   ├── _openvino_coder (OpenVINOGenerator)
│   └── _model_access (ModelAccessManager)
└── components
    ├── concept_miner (ConceptMiner)
    ├── contradiction_miner (ContradictionMiner)
    └── self_dialog_learning (SelfDialogLearning)
`

### 4.3 SplitModelRunner (послойный доступ)

Файл: eva_ai/core/split_model_runner.py

Позволяет извлекать скрытые состояния на конкретных слоях:

- Split LLM into Part1 and Part2 for GNN injection
- Кеш моделей для каждого слоя
- Метод get_layer_output() для извлечения на конкретном слое
- Метод get_layer_outputs_analysis() для анализа множества слоёв

LayerAnalyzer — анализ слоёв для концептов:
- Метод find_concept_clusters() для кластеризации концептов по представлениям слоёв

---

## 5. Проблемы и нереализованные функции

### 5.1 Критические проблемы

#### 5.1.1 Упрощённые реализации компонентов

| Компонент | Проблема | Файл |
|-----------|----------|------|
| causal_self_attention | Просто копирует вход без реального attention | fcp_core/hybrid_layer.py:219-231 |
| swiglu_feed_forward | Просто копирует вход без FFN | fcp_core/hybrid_layer.py:233-240 |
| _get_query_embedding | Возвращает случайный вектор | fcp_gnn/hybrid_integration.py:346-349 |
| ConvergenceController | Частичная реализация | fcp_core/__init__.py:42-96 |

#### 5.1.2 SplitModelRunner — неполная реализация

- Part2 (вторая часть модели) — только заглушка
- Нет реальной инъекции GNN между частями
- Требует отдельной модели на каждый слой (потребление памяти)

#### 5.1.3 GNN Encoder не обучен

- Веса инициализированы случайно (np.random.seed(42))
- Нет обученного graph_encoder.pt
- Требуется обучение на данных ConceptNet/RuBQ/Saiga

### 5.2 Средние проблемы

#### 5.2.1 MemorySnapshotIntegration

- Зависимость от внешнего модуля memory_snapshot_integration.py
- Не проверена интеграция с ConceptMiner
- Требует доработки для реального использования

#### 5.2.2 KCA/SRG — упрощённые реализации

- KCA: Веса инициализированы как np.eye(d) * 0.5
- SRG: Требует реальные логиты для расчёта энтропии
- Нет обучения этих компонентов

#### 5.2.3 LoRA адаптеры

- fcp_finetuned LoRA не загружен в FCPPipelineV15
- Требуется обучение в Colab
- Нет механизма динамического переключения адаптеров

### 5.3 Мелкие проблемы

- Токенизатор: Использует Qwen2.5-3B токенизатор вместо Qwen3-4B
- HNSW: Требует hnswlib, нет fallback
- Системный промпт: Дублируется в FCPPipelineV15
- Кеширование: Нет персистентного кеша для подграфов
- Логирование: Неполное логирование KCA циклов

### 5.4 Нереализованные функции

1. FCP Training Pipeline — нет полного пайплайна обучения
2. GNN -> ONNX -> OpenVINO — конвертер есть, но не протестирован
3. Adaptive Layer Selection — динамическое переключение слоёв не работает
4. Multi-Domain LoRA — нет переключения между доменами
5. Real-time Graph Update — граф не обновляется в реальном времени

---

## 6. Рекомендации по развитию

### 6.1 Приоритет 1 — Исправление базовых проблем

1. Реализовать реальный attention и FFN
   - Заменить заглушки на полные реализации
   - Использовать torch/NumPy операции

2. Обучить GNN Encoder
   - Запустить train_gnn.ipynb в Colab
   - Создать dataset из ConceptNet/RuBQ

3. Интегрировать LoRA адаптеры
   - Загрузить fcp_finetuned адаптер
   - Реализовать динамическое переключение

### 6.2 Приоритет 2 — Улучшение компонентов

1. Улучшить KCA
   - Заменить случайные веса на обучаемые
   - Добавить больше проверок сходимости

2. Улучшить SRG
   - Интеграция с реальными логитами
   - Добавить больше режимов

3. MemorySnapshot
   - Отладить интеграцию
   - Добавить персистентность

### 6.3 Приоритет 3 — Новые функции

1. Real-time Graph Update
   - Автоматическое добавление новых узлов
   - Инкрементальное обновление HNSW

2. Adaptive Layer Selection
   - Ранняя остановка на основе confidence
   - Динамический выбор слоёв

3. Multi-Domain LoRA
   - Отдельные адаптеры для доменов
   - Динамическое переключение

---

## Заключение

FCP представляет собой амбициозную архитектуру гибридной модели с:

Преимуществами:
- Глубокая интеграция GNN в каждый слой
- Адаптивные механизмы (KCA, SRG, AdaLoRA)
- Модульная архитектура
- Готовый pipeline для OpenVINO

Недостатками:
- Множество заглушек и упрощённых реализаций
- Не обученные компоненты (GNN, KCA, SRG)
- Неполная интеграция с EVA Core

Для production использования необходимо:
1. Обучить GNN Encoder
2. Заменить заглушки на реальные реализации
3. Интегрировать LoRA адаптеры
4. Протестировать всю систему

---

Дата анализа: 2026-04-27
Версия EVA: FCP System v15
Основные файлы: 12 файлов в eva_ai/fcp_core, fcp_gnn, fcp_knowledge
