# EVA System Architecture Audit Report

**Date:** April 14, 2026  
**Auditors:** AI Architect Agents (11 parallel agents across 2 cycles)  
**Document reviewed:** `system_flow_v2.md`

---

## Executive Summary

Система EVA позиционируется как самопознающая когнитивная система с полной реализацией концептов, противоречий и самообучения. Аудит выявил **критические несоответствия** между документацией и реальной реализацией.

### Общая оценка системы: 7.0/10 (снижена с 7.2)

| Компонент | Оценка | Тренд |
|-----------|--------|-------|
| Система генерации | 6.5/10 | → |
| Система концептов | 7.5/10 | → |
| Система противоречий | 7.2/10 | → |
| Самодиалог | 8.0/10 | → |
| Координационная инфраструктура | 6.5/10 | → |
| FractalGraph V2 | 7.0/10 | → |
| **Web GUI** | **8.0/10** | 🆕 |
| **CoreBrain инициализация** | **7.0/10** | 🆕 |
| **brain_query обработка** | **5.5/10** | 🆕 |
| **GraphCurator** | **4.2/10** | 🆕 |
| **Wikipedia KB** | **6.2/10** | 🆕 |

---

## 1. Критические проблемы (требуют немедленного исправления)

### 1.1 EventBus Priority System НЕ РЕАЛИЗОВАНА
**Серьёзность:** КРИТИЧЕСКАЯ

Документация заявляет: "События имеют приоритеты (LOW, NORMAL, HIGH, CRITICAL). Подписчики получают события в порядке приоритета."

**Реальность:** Параметр `priority` в `subscribe()` полностью игнорируется. Все подписчики вызываются FIFO.

**Файл:** `eva_ai/core/event_bus.py`

---

### 1.2 GraphCurator НЕ использует EventBus и DeferredCommandSystem
**Серьёзность:** КРИТИЧЕСКАЯ

Документация заявляет: GraphCurator интегрирован в систему событий и использует DeferredCommandSystem.

**Реальность:** 
- EventBus не интегрирован вообще
- DeferredCommandSystem не используется
- Таймеры работают напрямую через `threading.Timer`

**Соответствие документации: 42%**

---

### 1.3 ConceptExtractor НЕ СОХРАНЯЕТ концепты автоматически
**Серьёзность:** КРИТИЧЕСКАЯ

Метод `extract_concepts()` только возвращает список, не вызывает `save_concept_to_graph()`.

---

### 1.4 ModelAccessManager НЕ интегрирован в brain_query
**Серьёзность:** КРИТИЧЕСКАЯ

brain_query создаёт ModelAccessManager, но **не использует его** для координации доступа к модели. Генерация идёт напрямую через pipeline.

---

### 1.5 FractalGraphV2.get_clusters() НЕ СУЩЕСТВУЕТ
**Серьёзность:** ВЫСОКАЯ

Метод `get_clusters()` отсутствует. ConceptMiner получает данные напрямую из `storage.nodes`.

---

## 2. Значительные упрощения

### 2.1 brain_query - предобработка после генерации
**Серьёзность:** ВЫСОКАЯ

`_extract_key_concepts()` вызывается ПОСЛЕ генерации (строка 267), не ДО. Концепты не влияют на текущий ответ.

### 2.2 brain_query - нет entity extraction и intent detection
**Серьёзность:** СРЕДНЯЯ

Только greeting check и `needs_web_search()` эвристика. Полноценная предобработка не реализована.

### 2.3 GraphCurator - адаптивные интервалы не реализованы
**Серьёзность:** СУЩЕСТВЕННАЯ

Интервалы зашиты в код, не адаптируются к нагрузке системы.

### 2.4 Wikipedia KB - CPU-only векторизация
**Серьёзность:** СРЕДНЯЯ

`device='cpu'` - нет GPU ускорения для эмбеддингов.

---

## 3. Web GUI (8/10)

### Сильные стороны:
- SSE streaming работает корректно
- XHR POST для bidirectional communication
- GUIBridge интегрирован с CoreBrain событиями
- 10/10 соответствие документации

### Проблемы:
- Worker thread leak при timeout в `/api/v1/chat`
- Нет heartbeat/keepalive для SSE
- Клиент игнорирует неполный JSON без логирования

---

## 4. CoreBrain Инициализация (7/10)

### Сильные стороны:
- 9 миксинов импортированы корректно
- Порядок инициализации соответствует документации
- DeferredCommandSystem связывается с компонентами

### Проблемы:
- FractalGraphV2 создаётся позже, чем нужен
- Смешанный стиль инициализации (CoreBrain + component_initializer)
- Множество fallback-механизмов

---

## 5. brain_query (5.5/10)

### Что работает:
- Контекст из concept_extractor (через self_dialog)
- Контекст из cache
- Greeting check

### Что НЕ работает:
| Требование документации | Реализация |
|----------------------|-----------|
| Предобработка: извлечение сущностей | ❌ НЕТ |
| Предобработка: определение намерений | ⚠️ Частично |
| Контекст из concept_extractor до генерации | ❌ После |
| Контекст из contradiction_generator | ⚠️ Косвенно |
| ModelAccessManager координация | ❌ НЕ ИНТЕГРИРОВАН |

---

## 6. GraphCurator (4.2/10 - самый слабый компонент)

### Соответствие документации: 42% (3/7 критериев)

| Критерий документации | Реализация |
|--------------------|-----------|
| Интеграция с EventBus | ❌ НЕТ |
| DeferredCommandSystem | ❌ НЕТ |
| OPERATIONS константы | ✅ Есть |
| Cleanup/Consolidate/Promote методы | ⚠️ Упрощено |
| CuratorState машина состояний | ⚠️ Частично |
| Интервалы адаптивные | ❌ НЕТ |
| Метрики | ✅ Есть |

---

## 7. Wikipedia KB (6.2/10)

### Сильные стороны:
- FAISS семантический поиск работает
- SQLite + FAISS гибридное хранение
- Интеграция с FractalGraphV2

### Проблемы:
- CPU-only векторизация
- Truncation контента (10000 → 1000 символов)
- Нет автообновления
- Статичные данные

---

## 8. Рекомендации по приоритету

### НЕМЕДЛЕННО:
1. **GraphCurator** - добавить EventBus и DeferredCommandSystem интеграцию
2. **brain_query** - интегрировать ModelAccessManager
3. **ConceptExtractor** - добавить автосохранение концептов

### ВЫСОКИЙ ПРИОРИТЕТ:
4. brain_query - перенести концепты ДО генерации
5. EventBus - реализовать priority system
6. FractalGraphV2 - добавить get_clusters()

### СРЕДНИЙ ПРИОРИТЕТ:
7. Web GUI - добавить heartbeat для SSE
8. Wikipedia KB - добавить GPU векторизацию опционально

---

## 9. Сводная таблица соответствия

| Компонент | Документация | Реализация | Оценка |
|----------|-------------|-----------|--------|
| ModelAccessManager | Единая точка входа | Не интегрирован | 5/10 |
| EventBus priority | Работает | Не реализовано | 2/10 |
| ConceptExtractor | Автосохранение | Не сохраняет | 3/10 |
| brain_query | Предобработка | После генерации | 5/10 |
| GraphCurator | EventBus + DCS | Ничего | 4/10 |
| FractalGraphV2 | get_clusters() | Нет метода | 0/10 |
| SelfDialogLearning | 4-этапный диалог | Работает | 9/10 |
| DeferredCommandSystem | Приоритеты | Работает | 9/10 |

---

## 10. Файлы отчётов

Все детальные отчёты доступны в `eva_ai/analysis/`:

**Цикл 1:**
- `audit_generation_system.md` (6.5/10)
- `audit_concept_system.md` (7.5/10)
- `audit_contradiction_system.md` (7.2/10)
- `audit_self_dialog.md` (8.0/10)
- `audit_coordination.md` (6.5/10)
- `audit_fractal_storage.md` (7.0/10)

**Цикл 2:**
- `audit_web_gui.md` (8.0/10)
- `audit_corebrain_init.md` (7.0/10)
- `audit_brain_query.md` (5.5/10)
- `audit_graph_curator.md` (4.2/10)
- `audit_wikipedia_kb.md` (6.2/10)

---

*Отчёт подготовлен AI Architect Agents*
*April 14, 2026*
