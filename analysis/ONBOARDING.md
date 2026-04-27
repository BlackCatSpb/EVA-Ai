# EVA AI System - Полное руководство разработчика

**Версия:** 1.0  
**Дата:** 27.04.2026  
**Статус:** В разработке

---

## Содержание

1. [Что такое EVA AI](#1-что-такое-eva-ai)
2. [Архитектура системы](#2-архитектура-системы)
3. [Основные компоненты](#3-основные-компоненты)
4. [Проблемы и ошибки](#4-проблемы-и-ошибки)
5. [План работ](#5-план-работ)
6. [Как начать работать](#6-как-начать-работать)
7. [Тестирование](#7-тестирование)
8. [Связанные документы](#8-связанные-документы)

---

## 1. Что такое EVA AI

EVA (Electronic Virtual Assistant) - это когнитивная AI система с гибридной архитектурой, сочетающей:
- **LLM (Large Language Model)** - Qwen3-4B для генерации текста
- **FractalGraph v2** - граф знаний для хранения информации
- **Concept System** - система концептов и их связей
- **Self-Dialog Learning** - самодиалог для обучения
- **FCP (Fractal Cognitive Processor)** - GNN + Transformer гибрид

### Основные возможности:
- Генерация ответов на русском/английском языках
- Ведение графа знаний с извлечением концептов
- Обнаружение и разрешение противоречий
- Фоновый майнинг знаний
- Web-поиск для обогащения ответов
- Этическая фильтрация

---

## 2. Архитектура системы

```
┌─────────────────────────────────────────────────────────────────┐
│                         CoreBrain                                │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ HybridPipe  │  │ FractalGraph │  │ SelfDialogLearning   │ │
│  │ (DualGen)   │  │     v2       │  │                      │ │
│  └─────────────┘  └──────────────┘  └──────────────────────┘ │
│         │                │                      │              │
│  ┌──────┴──────┐  ┌─────┴────┐  ┌────────┬────────┐          │
│  │ WebSearch   │  │Concept  │  │Concept │Contra  │          │
│  │ Engine      │  │Extractor│  │Miner   │Miner   │          │
│  └─────────────┘  └─────────┘  └────────┴────────┘          │
│         │                │                │                   │
└─────────│────────────────│────────────────│───────────────────┘
          │                │                │
    ┌─────┴─────┐    ┌─────┴─────┐   ┌────┴────┐
    │   Flask   │    │  KGAdapter│   │EventBus │
    │  Server   │    │           │   │         │
    └───────────┘    └───────────┘   └─────────┘
```

### Поток данных (Generation Flow):

```
User Query → API /api/chat → brain_query.py
                              ↓
                  HybridPipelineAdapter
                              ↓
            ┌─────────────────┼─────────────────┐
            ↓                 ↓                 ↓
      ConceptExtractor  WebSearchEngine   TwoModelPipeline
            ↓                 ↓
     FGv2 (concept)    Fallback Response
            ↓
    SelfDialogLearning ← ConceptMiner (async)
                              ↓
                        FGv2 (gaps)
                              ↓
                       ContradictionMiner (async)
                              ↓
                        FGv2 (contradictions)
```

---

## 3. Основные компоненты

### 3.1 Core Generation (core/)

| Файл | Назначение | Статус |
|------|------------|--------|
| `hybrid_pipeline_adapter.py` | Основной пайплайн генерации | ✅ Активен |
| `dual_generator.py` | Выбор между моделями A/B | ✅ Активен |
| `model_access_manager.py` | Координация доступа к GPU/CPU | ✅ Активен |
| `brain_query.py` | Обработка запросов | ✅ Активен |

**Главная точка входа:** `CoreBrain.run()` в `__main__.py`

### 3.2 Memory (memory/)

| Файл | Назначение | Статус |
|------|------------|--------|
| `fractal_graph_v2/` | Граф знаний | ✅ Активен |
| `hybrid_token_cache.py` | Гибридный кэш | ⚠️ Частично |

### 3.3 Knowledge System (knowledge/, contradiction/)

| Компонент | Файл | Назначение |
|-----------|------|-------------|
| **ConceptExtractor** | `concept_extractor.py` | Извлечение концептов из текста (быстрый) |
| **ConceptMiner** | `concept_miner.py` | Поиск семантических лакун (фоновый) |
| **ContradictionGenerator** | `contradiction_generator.py` | Генерация шаблонных противоречий |
| **ContradictionMiner** | `contradiction_miner.py` | Детекция реальных противоречий в графе |

### 3.4 Self-Dialog (learning/)

| Файл | Назначение | Статус |
|------|------------|--------|
| `dialog_core.py` | Ядро самодиалога | ⚠️ Есть баги |
| `dialog_concepts.py` | Интеграция концептов | ⚠️ Не инициализирован |

### 3.5 FCP System (fcp_core/, fcp_gnn/)

| Компонент | Назначение | Статус |
|-----------|------------|--------|
| `HybridTransformerLayer` | Гибридный слой GNN+Transformer | ⚠️ Заглушки |
| `FractalGraphEncoder` | GNN энкодер | ⚠️ Не обучен |
| `AdaLoRA` | Адаптивный LoRA | ⚠️ Не загружен |

### 3.6 Server/GUI (gui/, server)

| Файл | Назначение | Проблемы |
|------|------------|----------|
| `gui/server_routes.py` | API endpoints | ⚠️ 3x дублирование |
| `server_routes.py` | Старые endpoints | ❌ Дубликат |
| `gui/web_gui/server_routes.py` | Основные endpoints | ✅ Работает |

---

## 4. Проблемы и ошибки

### 4.1 Критические проблемы (исправить немедленно)

#### C1: FCP изолирован от Ethics и WebSearch
**Где:** `fcp_system.md`, `cross_analysis_fcp_ethics.md`  
**Описание:** FCP Pipeline не использует WebSearch и Ethics, работает изоли��ованно  
**Влияние:** FCP генерирует ответы без веб-обогащения и этической проверки

#### C2: Дублирование /api/chat (3 раза!)
**Где:** `cross_analysis_server_monitoring.md`  
**Файлы:**
- `eva_ai/server_routes.py:151`
- `eva_ai/gui/web_gui/server_routes.py:399`  
- `eva_ai/gui/web_gui/server_routes_chat.py:18`

**Описание:** Три определения одного endpoint - непредсказуемое поведение

#### C3: Три системы детекции противоречий
**Где:** `cross_analysis_dialog_miners.md`, `contradiction_legacy_system.md`
1. `ContradictionGenerator` - шаблоны
2. `ContradictionMiner` - математика + NLI
3. `detect_semantic.py`, `detect_logical.py`, `detect_temporal.py` - legacy

**Описание:** Дублирование функционала, трудно поддерживать

---

### 4.2 Высокий приоритет

#### H1: KGAdapter не создаётся
**Где:** `cross_analysis_core_memory.md`  
**Описание:** KGAdapter не инициализируется в `init_factories.py`  
**Следствие:** Не работают методы `find_path_between_concepts()`

#### H2: ContradictionGenerator слабо интегрирован
**Где:** `knowledge_system.md`  
**Описание:** Создаётся, но почти не используется в системе

#### H3: DialogConceptsMixin не инициализирован  
**Где:** `cross_analysis_core_memory.md`  
**Описание:** Миксин для интеграции концептов в самодиалог не подключён

#### H4: summary_parts не определён
**Где:** `self_dialog_system.md`, `dialog_core.py:1049`  
**Описание:** ReferenceError при формировании summary

#### H5: SystemMonitor изолирован
**Где:** `cross_analysis_server_monitoring.md`  
**Описание:** Не подключён к EventBus CoreBrain

---

### 4.3 Средний приоритет

#### M1: Мёртвый код в ContradictionGenerator
**Где:** `knowledge_system.md:401-433`  
**Описание:** Дублирующий try-блок после return

#### M2: FCP заглушки (attention, FFN)
**Где:** `fcp_system.md`  
**Описание:** `causal_self_attention` и `swiglu_feed_forward` просто копируют вход

#### M3: Background Jobs не протестированы
**Где:** `core_brain_background_system.md`

#### M4: Заглушки в KGAdapter.__getattr__
**Где:** `knowledge_system.md:166-170`  
**Описание:** Возвращает None-функцию вместо ошибки

---

## 5. План работ

### Фаза 1: Критические исправления (1-2 недели)

| # | Задача | Файлы | Ответственный |
|---|--------|-------|----------------|
| 1.1 | Удалить старые server_routes.py (дубликаты) | `server_routes.py` | - |
| 1.2 | Оставить один источник /api/chat | `gui/web_gui/server_routes.py` | - |
| 1.3 | Интегрировать WebSearch в FCP | `fcp_pipeline.py`, `brain_query.py` | - |
| 1.4 | Интегрировать Ethics в FCP | `fcp_pipeline.py` | - |
| 1.5 | Объединить системы детекции противоречий | `contradiction/*.py` | - |

### Фаза 2: Высокий приоритет (2-3 недели)

| # | Задача | Файлы | Ответственный |
|---|--------|-------|----------------|
| 2.1 | Создать KGAdapter в init_factories | `core/init_factories.py` | - |
| 2.2 | Исправить summary_parts в dialog_core | `dialog_core.py:1049` | - |
| 2.3 | Инициализировать DialogConceptsMixin | `dialog_core.py` | - |
| 2.4 | Подключить SystemMonitor к EventBus | `system_monitor.py` | - |
| 2.5 | Активировать использование ContradictionGenerator | `brain_query.py` | - |

### Фаза 3: Средний приоритет (3-4 недели)

| # | Задача | Файлы | Ответственный |
|---|--------|-------|----------------|
| 3.1 | Удалить мёртвый код (contradiction_generator:401-433) | - |
| 3.2 | Реализовать FCP attention и FFN | `hybrid_transformer_layer.py` | - |
| 3.3 | Обучить GNN Encoder | `graph_encoder.py` | - |
| 3.4 | Интегрировать LoRA адаптеры | `adaptive_lora.py` | - |

---

## 6. Как начать работать

### 6.1 Установка

```bash
# Клонировать репозиторий
cd C:\Users\black\OneDrive\Desktop\EVA-Ai

# Создать виртуальное окружение
python -m venv venv
venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt
```

### 6.2 Структура проекта

```
eva_ai/
├── analysis/           # Отчёты анализа (READ FIRST!)
├── core/              # Основная логика (CoreBrain, HybridPipeline)
├── memory/            # FractalGraphV2
├── knowledge/        # ConceptExtractor, ConceptMiner
├── contradiction/    # Система противоречий
├── learning/         # SelfDialogLearning
├── fcp_core/         # FCP ядро
├── fcp_gnn/          # GNN компоненты
├── websearch/        # Web-поиск
├── ethics/           # Этическая система
├── gui/              # GUI и web-интерфейс
├── server.py         # Flask сервер
└── __main__.py       # Точка входа
```

### 6.3 Запуск

```powershell
# Очистить логи
Remove-Item "C:\Users\black\OneDrive\Desktop\CogniFlex\*.log" -Force

# Запустить EVA
cd C:\Users\black\OneDrive\Desktop\CogniFlex
python -m eva_ai
```

### 6.4 Ключевые файлы для понимания

1. **`core/brain_query.py`** - Обработка запросов (точка входа)
2. **`core/hybrid_pipeline_adapter.py`** - Основной пайплайн
3. **`core/init_factories.py`** - Создание компонентов
4. **`knowledge/concept_extractor.py`** - Извлечение концептов
5. **`learning/dialog_core.py`** - Самодиалог
6. **`memory/fractal_graph_v2/`** - Граф знаний

---

## 7. Тестирование

### 7.1 Ручное тестирование

```bash
# Тест базовой генерации
curl -X POST http://localhost:5555/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Привет", "session_id": "test"}'

# Тест концептов
curl -X POST http://localhost:5555/api/concepts/extract \
  -H "Content-Type: application/json" \
  -d '{"query": "Что такое AI?", "response": "Это искусственный интеллект"}'

# Тест противоречий
curl -X POST http://localhost:5555/api/contradictions/generate \
  -H "Content-Type: application/json" \
  -d '{"concepts": ["искусственный интеллект"]}'
```

### 7.2 Автотесты

Тесты расположены в `fcp_migration/tests/`:
- `test_e2e.py` - end-to-end тесты
- `test_layer_depth.py` - тесты слоёв
- `test_hnsw.py` - тесты HNSW индекса

---

## 8. Связанные документы

### 8.1 Основные отчёты анализа

| Файл | Содержание |
|------|-------------|
| `core_generation.md` | Анализ Generation системы |
| `memory_system.md` | Анализ FractalGraphV2 |
| `knowledge_system.md` | Анализ Concept/Contradiction |
| `fcp_system.md` | Анализ FCP системы |
| `server_gui_system.md` | Анализ Server/GUI |
| `websearch_ethics_system.md` | Анализ WebSearch/Ethics |

### 8.2 Перекрёстные анализы

| Файл | Область |
|------|---------|
| `cross_analysis_core_memory.md` | Core + Memory + Knowledge |
| `cross_analysis_dialog_miners.md` | SelfDialog + Miners |
| `cross_analysis_fcp_ethics.md` | FCP + Ethics + WebSearch |
| `cross_analysis_server_monitoring.md` | Server + GUI + Monitoring |
| `cross_analysis_final.md` | Финальный отчёт |

### 8.3 Конфигурация

- `brain_config.json` - основная конфигурация
- `fractal_model_config.json` - конфигурация модели
- `gui_config.json` - конфигурация GUI

---

## 9. Часто задаваемые вопросы

### Q: Где главная точка входа?
**A:** `eva_ai/__main__.py` → `CoreBrain.run()`

### Q: Как добавить новый компонент?
**A:** Зарегистрировать в `core/init_factories.py` и добавить в `COMPONENT_LIST` в `core/init_core.py`

### Q: Как работает самодиалог?
**A:**Concept Miner → FGv2 → SelfDialogLearning → Concept resolution → Cache

### Q: Почему FCP не использует веб-поиск?
**A:** Это баг - нужно интегрировать WebSearch в FCP Pipeline (задача C1)

### Q: Где логи?
**A:** В папке `C:\Users\black\OneDrive\Desktop\CogniFlex\*.log`

---

## 10. Глоссарий

| Термин | Определение |
|--------|--------------|
| **CoreBrain** | Главный координатор всех компонентов |
| **HybridPipeline** | Пайплайн генерации с двумя моделями |
| **FractalGraphV2** | Граф знаний EVA |
| **Concept** | Извлечённая сущность из текста |
| **Contradiction** | Противоречие между фактами |
| **Self-Dialog** | Внутренний диалог для обучения |
| **FCP** | Fractal Cognitive Processor (GNN+Transformer) |
| **EventBus** | Шина событий для коммуникации компонентов |

---

## Контакты и поддержка

- Основной репозиторий: `C:\Users\black\OneDrive\Desktop\EVA-Ai\`
- Working directory: `C:\Users\black\OneDrive\Desktop\CogniFlex\`
- Телеграм бот: EvaAiTest (тестирование)

---

**Примечание:** Этот документ обновляется по мере исправления проблем. Дата последнего обновления: 27.04.2026