# CogniFlex - Список ошибок для исправления

## Дата: 17 марта 2026
## Статус системы: РАБОТАЕТ (23/23 компонентов)

---

## НАЙДЕННЫЕ ОШИБКИ (из логов)

### 1. QueryProcessor не имеет метода 'process'
- **Файл**: test_script.log
- **Ошибка**: `'QueryProcessor' object has no attribute 'process'`
- **Частота**: Много раз
- **Влияние**: Невозможно обрабатывать запросы напрямую

### 2. MemoryManager не имеет атрибута 'memory_locks'
- **Файл**: test_script.log
- **Ошибка**: `'MemoryManager' object has no attribute 'memory_locks'`
- **Влияние**: Проблемы с памятью

### 3. BackgroundCoordinator ошибка интеграции (код 10)
- **Файл**: multiple logs
- **Ошибка**: `Ошибка интеграции с системой событий: 10`
- **Частота**: При каждом запуске
- **Влияние**: Низкое (предупреждение)
- **Причина**: Несовместимость с events API

### 4. CoreBrain не имеет '_initialize_memory_manager'
- **Файл**: diagnostic_logs/cogniflex_app.log
- **Ошибка**: `'CoreBrain' object has no attribute '_initialize_memory_manager'`
- **Влияние**: HIGH

### 5. Invalid argument (errno 22)
- **Файлы**: component_initializer
- **Ошибка**: `[Errno 22] Invalid argument`
- **Частота**: Редко (зависит от состояния системы)

---

## ПРЕДУПРЕЖДЕНИЯ (не ошибки)

### 1. SentenceTransformer disabled
- **Сообщение**: `SentenceTransformer disabled by configuration`
- **Причина**: Мы отключили эмбеддинги намеренно
- **Статус**: OK

### 2. Load-shed: batch_size reduced
- **Сообщение**: `Load-shed: batch_size 32 -> 16 due to high memory`
- **Причина**: High RAM usage
- **Статус**: OK (нормальное поведение)

### 3. TF32 precision warning
- **Сообщение**: `Unknown attribute fp32_precision`
- **Причина**: Старая версия torch
- **Статус**: OK

---

## ЛОГИКА ИНИЦИАЛИЗАЦИИ (текущая)

```
1. CoreBrain.__init__()
   ├── Security → OK
   ├── DeferredCommandSystem → OK (6 workers)
   ├── EventBus → OK
   ├── SystemStateManager → OK
   ├── ResourceManager → OK
   ├── AnalyzerCore → OK
   ├── HealthMonitor → OK
   ├── LearningOpportunityManager → OK
   ├── PerformanceAnalyzer → OK
   ├── MemoryGraphTrainer → OK (cuda)
   ├── SelfAnalyzer → OK
   ├── EnhancedSelfLearning → OK
   ├── MemoryGraphML → OK
   ├── SelfLearningSystem → OK
   │
2. ComponentInitializer.initialize()
   ├── EventBus → OK
   ├── ResourceManager → OK
   ├── ConfigManager → OK
   ├── MemoryManager → OK
   ├── HybridTokenCache → OK (50GB disk, 11GB VRAM)
   ├── KnowledgeGraph → OK
   ├── TextProcessor → OK (SentenceTransformer disabled)
   │
3. MLUnit
   ├── MLCore → OK
   ├── FractalModelManager → OK (RUGPT3, cuda)
   ├── TextProcessor → OK
   ├── ResponseGenerator → OK
   ├── HybridTokenCache → OK
   ├── TrainingOrchestrator → OK
   │
4. ModelManager (HybridModelManager)
   ├── RUGPT3 tokenizer → OK (50257 vocab)
   └── RUGPT3 model → OK
```

---

## ЧТО РАБОТАЕТ

| Компонент | Статус |
|-----------|--------|
| RUGPT3 Model | ✅ OK |
| Tokenizer | ✅ OK (50257) |
| HybridTokenCache | ✅ OK |
| KnowledgeGraph | ✅ OK |
| TextProcessor | ✅ OK |
| ResponseGenerator | ✅ OK |
| TrainingOrchestrator | ✅ OK |
| GUI (23 modules) | ✅ OK |

---

## ПЛАН ИСПРАВЛЕНИЙ

### HIGH PRIORITY
1. Добавить метод `process` в QueryProcessor
2. Исправить атрибут `memory_locks` в MemoryManager  
3. Исправить `_initialize_memory_manager` в CoreBrain

### MEDIUM PRIORITY
1. BackgroundCoordinator - исправить интеграцию events (код 10)

---

## КАК ВОСПРОИЗВЕСТИ

```bash
# Запуск с тестом
python -m cogniflex.run

# Или отдельно
python cogniflex_test_script.py
```

---

*Документ создан: 2026-03-17*
