# Отчёт: Adaptation & Training

## 1. Структура

### eva_ai/adaptation/ (4 файла Python, ~2500 строк)

| Файл | Назначение | Строк |
|------|-----------|-------|
| __init__.py | Экспорты: AdaptationManager, UserFeedback, UserProfile | 15 |
| adaptation_types.py | Enum и dataclass: AdaptationLevel, LearningStyle, UserPreferences, AdaptationProfile | 64 |
| adaptation_profiles.py | Модели: UserFeedback, UserProfile | 94 |
| adaptation_manager.py | **ОСНОВНОЙ** AdaptationManager (fullfeatured) | 729 |
| adaptation_core.py | Базовая версия с SQLite | 604 |
| adaptation_integration.py | Monkey-patching методов | 763 |
| adaptation_integrated.py | IntegratedAdaptationManager (BaseComponent) | 311 |
| adaptation_analytics.py | Аналитические функции | 344 |

### eva_ai/training/ (1 файл)

| Файл | Назначение | Строк |
|------|-----------|-------|
| __init__.py | Экспорты: GGUFTrainingSystem, TrainingStatus, TrainingMetrics, VerifiedKnowledge | 5 |
| gguf_training_system.py | Система дообучения GGUF | 789 |

### eva_ai/runtime/ (2 файла)

| Файл | Назначение | Строк |
|------|-----------|-------|
| worker_pool.py | InferenceWorkerPool (multiprocessing) | 195 |
| simple_model.py | Example model function | 43 |

---

## 2. Реализация

### 2.1 Adaptation System

#### Архитектура (хаотичная)

- adaptation_manager.py (729 строк) - ОСНОВНОЙ
- adaptation_profiles.py - UserProfile, UserFeedback
- adaptation_types.py - enums, dataclasses
- adaptation_integration.py - monkey-patches methods onto AdaptationManager
- adaptation_core.py - alternative SQLite-based AdaptationManager
- adaptation_integrated.py - BaseComponent wrapper
- adaptation_analytics.py - добавляет методы через monkey-patching

**ПРОБЛЕМА**: 4 различных класса AdaptationManager с перекрывающейся функциональностью.

#### Ключевые компоненты:

**UserProfile** (adaptation_profiles.py):
- user_id, preferences, interaction_history
- adaptation_level (float, default 0.5)
- learning_style (visual, auditory, kinesthetic, reading, mixed)
- knowledge_level (float)
- response_preferences (formal/casual)
- cultural_profile

**AdaptationManager** (adaptation_manager.py):
- Хранилище профилей пользователей
- Feedback history (deque maxlen=10000)
- Concept cache + usage tracking
- Фоновый поток анализа (каждые 5 минут)
- SQLite persistence

**Интеграция через monkey-patching** (adaptation_integration.py):
- get_user_profile, update_user_profile, record_feedback
- analyze_user_patterns, export_adaptation_data, import_adaptation_data
- integrate_with_knowledge_graph, get_cultural_adaptation
- get_adaptation_progress, generate_adaptation_report, adapt_response

---

### 2.2 Training System

#### GGUFTrainingSystem (gguf_training_system.py)

**Архитектура**:
- TrainingGGUF (отдельный экземпляр для обучения)
- ProductionGGUF (НЕ обучается - основная модель)
- LoRA Adapters (domain-specific)
- VerificationSystem

**Paths**:
- base_model_path: ruadapt_qwen3_4b_q4_k_m.gguf
- training_model_path: eva_ai/models/training_qwen.gguf
- lora_path: eva_ai/models/lora_adapters

**Settings**:
- batch_size: 4, epochs: 3, learning_rate: 1e-4
- min_confidence: 0.7, min_knowledge_for_training: 5

**Flow обучения**:
1. _extract_verified_knowledge() - извлечение из KG
2. _prepare_training_data() - формирование Q/A пар
3. _train_separate_instance() - knowledge distillation
4. _verify_training_quality() - проверка качества
5. _save_lora_adapters() - сохранение адаптеров

**Knowledge Distillation**:
- Использует llama_cpp для генерации расширенных ответов
- LoRA адаптеры по доменам: programming, science, history, geography, general
- Сохраняет examples в JSON файлы

---

### 2.3 Runtime System

#### InferenceWorkerPool (worker_pool.py)

- Multiprocessing pool для model inference
- num_workers = max(1, mp.cpu_count() // 2)
- torch_threads: 2, interop_threads: 1
- Methods: submit(), recv(), infer_batches()

---

## 3. Интеграция

### 3.1 Adaptation Manager - Initialization

**init_factories.py:407-419**:
- create_adaptation_manager() создаёт AdaptationManager(brain=core_brain)
- Привязывается к core_brain.adaptation_manager

**brain_components.py:100-149**:
- SelfDialogLearningSystem инициализируется параллельно с AdaptationManager

### 3.2 Adaptation -> Self-Learning Integration

**DialogConceptsMixin** (dialog_concepts.py):
- queue_concept_for_dialog(concept_name, priority)
- queue_contradiction_for_resolution(contradiction_id, concept, priority)
- _get_next_dialog_topic() - Priority: contradiction > concept > conversation

**Integration point in brain_query.py:1415-1416**:
- self.self_dialog_learning.queue_concept_for_dialog()

### 3.3 Training System - Integration Points

**LOW/NONE DETECTED**:
- GGUFTrainingSystem НЕ интегрирован в основной цикл самообучения
- LoRA адаптеры не применяются к production model
- Нет связей между GGUFTrainingSystem и SelfDialogLearningSystem

---

## 4. Оценка

### Сильные стороны:

1. Развитая система профилей пользователей
2. Фоновый анализ паттернов
3. Аналитическая подсистема с dashboard
4. Separation of training and production models
5. Runtime multiprocessing с AMP support

### Проблемы и недостатки:

#### КРИТИЧЕСКИЕ:

1. **Дублирование кода AdaptationManager**
   - 4 версии: adaptation_manager.py, adaptation_core.py, adaptation_integrated.py + monkey-patching
   - Непонятно какая версия используется в production

2. **GGUFTrainingSystem изолирован**
   - Нет интеграции с SelfDialogLearning
   - LoRA adapters никогда не применяются к production model
   - min_knowledge_for_training=5 проверяется но KG не связан

3. **Неполная реализация training**
   - _check_model_integrity() = заглушка (return True)
   - _check_generation_quality() = заглушка (return True)

#### СУЩЕСТВЕННЫЕ:

4. No fallback when brain=None - методы падают
5. SQLite vs JSON persistence - нет синхронизации
6. Training model path hardcoded (Windows path)
7. Adaptation level calculation too simple (linear)

#### МЕДИУМ:

8. No validation on UserProfile updates
9. No rate limiting on adaptation decisions
10. Missing error handling in background threads

---

## Рекомендации

### Immediate:
1. Удалить дублирующие файлы (adaptation_core.py, adaptation_integrated.py)
2. Определить ОДИН canonical AdaptationManager
3. Интегрировать GGUFTrainingSystem в самообучение или удалить

### Short-term:
1. Реализовать _check_model_integrity() и _check_generation_quality()
2. Добавить применение LoRA adapters к production model
3. Унифицировать persistence
4. Добавить validation в update_user_profile

### Long-term:
1. Адаптивный adaptation_level на основе ML
2. Distributed training pipeline
3. A/B testing для adaptation strategies
