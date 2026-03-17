# CogniFlex Agent Swarm System

## Архитектура

```
                    ┌─────────────────────────────────────────────┐
                    │           COORDINATOR (Я)                    │
                    │        (Роль: CTO / Тимлид)                 │
                    └──────────────────┬──────────────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│   AGENT: UI/UX     │   │  AGENT: FRONTEND   │   │  AGENT: QA/TEST    │
│   (Research)       │   │   (Implementation)  │   │   (Verification)   │
└─────────┬───────────┘   └─────────┬───────────┘   └─────────┬───────────┘
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────────────┐
                    │         SHARED CONTEXT STORE               │
                    │    (Общая память агентов - docs/agents/)   │
                    └─────────────────────────────────────────────┘
```

## Агенты и их роли

### 1. UI/UX Agent (Research)
- **Назначение:** Анализ, исследования, рекомендации
- **Инструменты:** file read, grep, websearch, codesearch
- **Выход:** Отчёты, рекомендации, спецификации

### 2. Frontend Agent (Implementation)  
- **Назначение:** Реализация изменений в коде
- **Инструменты:** read, edit, write, grep
- **Выход:** Исправленный код, PR/коммиты

### 3. QA Agent (Verification)
- **Назначение:** Тестирование, верификация, проверка
- **Инструменты:** bash (тесты), grep, read
- **Выход:** Отчёты о проверке, баг-репорты

### 4. Documentation Agent
- **Назначение:** Документирование
- **Инструменты:** write, read
- **Выход:** Документация, README, API docs

## Протоколы коммуникации

### Протокол 1: Задача от Заказчика
```
Заказчик → Coordinator → Agent → Coordinator → Заказчик
```

### Протокол 2: Исследование + Реализация
```
Coordinator → UI/UX (анализ) → 
    → Coordinator (принимает рекомендации) → 
    → Frontend (реализация) →
    → QA (проверка) → 
    → Coordinator → Заказчик
```

### Протокол 3: Экстренная ситуация
```
Agent (error) → Coordinator → Заказчик → Решение → Agent
```

## Формат сообщений между агентами

### Запрос (Request)
```json
{
  "type": "task",
  "from": "coordinator",
  "to": "frontend_agent",
  "task_id": "task_001",
  "description": "Implement UI changes...",
  "context": {...},
  "deadline": "2026-03-17T18:00:00"
}
```

### Отчёт (Report)
```json
{
  "type": "report",
  "from": "frontend_agent", 
  "to": "coordinator",
  "task_id": "task_001",
  "status": "completed",
  "result": {...},
  "duration_minutes": 45
}
```

### Ошибка (Error)
```json
{
  "type": "error",
  "from": "qa_agent",
  "to": "coordinator", 
  "task_id": "task_001",
  "error": "Test failed...",
  "severity": "high"
}
```

## Общая память (Shared Context)

Агенты используют общую директорию для обмена данными:
- `docs/agents/shared_context.json` — текущее состояние
- `docs/agents/task_queue.json` — очередь задач
- `docs/agents/reports/` — отчёты агентов

## Команды координатора

### Запустить агента
```python
task(
    description="...", 
    prompt="...",
    subagent_type="explore"  # или general
)
```

### Параллельный запуск
```python
# Несколько агентов одновременно
task(agent1), task(agent2), task(agent3)
```

### Последовательный запуск (с зависимостями)
```python
# Agent1 → Agent2 → Agent3
result1 = task(agent1)
result2 = task(agent2, context=result1)
result3 = task(agent3, context=result2)
```

## Workflow Examples

### Пример 1: Исправление бага
1. Заказчик сообщает о проблеме
2. QA Agent воспроизводит баг
3. Frontend Agent исправляет
4. QA Agent проверяет
5. Coordinator докладывает Заказчику

### Пример 2: Новая функция
1. Заказчик описывает требования
2. UI/UX Agent анализирует и даёт рекомендации
3. Frontend Agent реализует
4. QA Agent тестирует
5. Documentation Agent документирует
6. Coordinator показывает результат Заказчику

---

*Система создана: 17 марта 2026*
