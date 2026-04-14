# Audit Report: GraphCurator Integration

**Date:** 2026-04-14  
**Auditor:** EVA AI Agent  
**Component:** GraphCurator  
**Integration Score:** 2/10

---

## Executive Summary

GraphCurator имеет критические проблемы интеграции с основной архитектурой EVA AI. Компонент не использует стандартные механизмы EventBus и DeferredCommandSystem, что нарушает архитектурные принципы системы.

---

## Основные находки

### 1. Отсутствие EventBus Integration (CRITICAL)

**Проблема:** GraphCurator НЕ использует EventBus - подписки нет.

**Ожидаемое поведение:** Компонент должен подписываться на события через EventBus для асинхронного взаимодействия с другими компонентами системы.

**Текущее поведение:** GraphCurator работает изолированно, без уведомления других компонентов о своих действиях.

**Влияние:** Другие компоненты (SelfDialogLearning) подписываются на события `curitor.*`, которые никогда не происходят.

---

### 2. Отсутствие DeferredCommandSystem (CRITICAL)

**Проблема:** GraphCurator НЕ использует DeferredCommandSystem (присвоен но не используется).

**Ожидаемое поведение:** Все фоновые задачи должны проходить через DeferredCommandSystem для централизованного управления, приоритизации и recovery.

**Текущее поведение:** Использует `threading.Timer` напрямую.

---

### 3. Прямое использование threading.Timer (HIGH)

**Проблема:** GraphCurator использует `threading.Timer` напрямую с фиксированным интервалом 600 сек.

**Локация:** Смотри исходный код GraphCurator

**Проблемы:**
- Фиксированный интервал 600 секунд не адаптируется к нагрузке системы
- Нет интеграции с системой приоритетов
- Нет механизма recovery при падении
- Нет координации с ModelAccessManager

---

### 4. Отсутствие метода is_running() (MEDIUM)

**Проблема:** Нет метода `is_running()` что ломает `brain_init.py`.

**Ожидаемое поведение:** Все управляемые компоненты должны иметь метод `is_running()` для проверки состояния.

**Текущее поведение:** brain_init.py вызывает is_running(), метод не существует.

---

### 5. Несуществующие события для SelfDialogLearning (MEDIUM)

**Проблема:** SelfDialogLearning подписывается на `curator.*` события которых никогда не происходит.

**Подписки в SelfDialogLearning:**
- `curator.knowledge_extracted` - НИКОГДА НЕ ПУБЛИКУЕТСЯ
- `curator.graph_optimized` - НИКОГДА НЕ ПУБЛИКУЕТСЯ
- `curator.cleanup_done` - НИКОГДА НЕ ПУБЛИКУЕТСЯ

**Влияние:** SelfDialogLearning никогда не получает уведомления от GraphCurator.

---

## Рекомендации

### Критические (должны быть исправлены)

1. **Добавить EventBus интеграцию:**
   - Подписаться на relevant events (system.idle, memory.graph_updated)
   - Публиковать события curator.* при выполнении операций

2. **Использовать DeferredCommandSystem:**
   - Заменить threading.Timer на deferred commands
   - Использовать CommandPriority для адаптивных интервалов

3. **Добавить метод is_running():**
   - Реализовать проверку состояния таймера/очереди

### Высокие (должны быть исправлены в спринт)

4. **Убрать hardcoded интервал 600 сек:**
   - Сделать интервал адаптивным
   - Добавить конфигурацию через brain_config.json

5. **Обеспечить публикацию событий для SelfDialogLearning:**
   - Публиковать curator.knowledge_extracted после извлечения
   - Публиковать curator.graph_optimized после оптимизации
   - Публиковать curator.cleanup_done после очистки

---

## Integration Checklist

| Компонент | Требуется | Реализовано | Статус |
|-----------|-----------|-------------|--------|
| EventBus subscription | Да | Нет | ❌ |
| EventBus publication | Да | Нет | ❌ |
| DeferredCommandSystem | Да | Нет | ❌ |
| is_running() method | Да | Нет | ❌ |
| Adaptive intervals | Да | Нет | ❌ |
| ModelAccessManager coordination | Да | Нет | ❌ |

---

## Заключение

GraphCurator требует полной переработки интеграции с основной архитектурой EVA AI. Текущая реализация нарушает ключевые архитектурные принципы и должна быть приведена в соответствие со стандартами системы.

**Общая оценка интеграции: 2/10**
