# Анализ Self-Dialog Learning System EVA

**Дата:** 2026-04-27
**Версия:** EVA AI v2
---

## 1. Архитектура системы самодиалога

### 1.1 Компонентная структура

Система самодиалога EVA использует множественное наследование через Python-миксины:

class SelfDialogLearning(DialogTopicsMixin, DialogGenerationMixin, DialogLearningMixin, DialogConceptsMixin)

**Основные компоненты:**
| Компонент | Файл | Строк |
|----------|------|-------|
| SelfDialogLearning | dialog_core.py | 1200 |
| DialogConceptsMixin | dialog_concepts.py | 858 |
| DialogGenerationMixin | dialog_generation.py | 322 |
| DialogTopicsMixin | dialog_topics.py | - |
| DialogLearningMixin | dialog_learning.py | - |

### 1.2 Зависимости

SelfDialogLearning зависит от:
- brain (CoreBrain)
- deferred_system (DeferredCommandSystem)
- event_bus (EventBus)
- hybrid_cache (HybridCache)

---

## 2. Жизненный цикл диалогов

### 2.1 Запуск

start() -> _worker_loop() (dialog_core.py:210-233)

### 2.2 Рабочий цикл

_worker_loop() (dialog_core.py:293-334):
1. Обработка задач из dialog_queue
2. Проверка learning opportunities
3. Генерация диалогов (каждые 300 сек)
4. Компактификация контекста (каждые 60 сек)

### 2.3 Приоритет тем

1. Противоречия (type: contradiction) - priority 0.7
2. Концепты (type: concept) - priority 0.5
3. История разговоров (fallback)

---

## 3. Методы интеграции концептов

### 3.1 queue_concept_for_dialog()

**Файл:** dialog_concepts.py:33-54

Добавляет концепт в очередь. Ограничение: MAX_CONCEPT_QUEUE = 100.

### 3.2 queue_contradiction_for_resolution()

**Файл:** dialog_concepts.py:56-78

Добавляет противоречие в очередь. Priority по умолчанию 0.7.

### 3.3 _run_concept_dialog()

**Файл:** dialog_concepts.py:148-274

4 роли: ASSISTANT -> CRITIC -> LEARNER -> TEACHER

### 3.4 _run_contradiction_dialog()

**Файл:** dialog_concepts.py:276-408

4 роли: ASSISTANT -> CRITIC -> LEARNER -> TEACHER

### 3.5 _get_next_dialog_topic()

**Файл:** dialog_concepts.py:80-107

Логика: противоречия -> концепты -> None

---

## 4. Сохранение результатов

### 4.1 _save_concept_dialog_results()

dialog_concepts.py:555-578

Сохраняет в: _resolved_knowledge, context cache, hybrid_cache

### 4.2 _save_contradiction_resolution()

dialog_concepts.py:611-639

Обновляет статус, сохраняет в FGv2

### 4.3 _save_learned_facts_to_fg()

dialog_concepts.py:641-672

Создает узел fact в FractalGraph v2

---

## 5. Подписка на события

dialog_core.py:117-141:
- system.idle -> _on_system_idle
- system.state_changed -> _on_system_state_changed
- concept.confirmed -> _on_concept_confirmed
- contradiction.detected -> _on_contradiction_detected

---

## 6. Выявленные проблемы

### 6.1 Критические

**Проблема 1: Синтаксическая ошибка**
Файл: dialog_core.py:1049
return join(summary_parts)  # summary_parts НЕ ОПРЕДЕЛЕНА
Влияние: NameError при выполнении

**Проблема 2: Рекурсия**
Файл: dialog_generation.py:115-117
brain.process_query() может вызвать самодиалог

### 6.2 Заглушки

dialog_core.py:249, :1003 - простые pass без логирования

### 6.3 Архитектурные

1. Смешение ответственностей
2. Жесткие зависимости от self.brain
3. Отсутствует DialogOrchestrator

---

## 7. Выводы

### Оценка: 7/10

**Сильные:** архитектура миксинов, EventBus/DeferredCommandSystem, приоритеты

**Слабые:** 1 ошибка (стр.1049), 2 заглушки, нет DialogOrchestrator

### Готовность: НЕТ

Требуется исправление критических проблем.

---

*Анализ AI-архитектор EVA*
