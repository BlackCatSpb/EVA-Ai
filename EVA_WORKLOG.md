# EVA AI - План исправлений

## ИНСТРУКЦИЯ ПО РАБОТЕ

### Цикл исправлений (автоматизированный):

1. **Перед запуском** - очищаю логи:
   ```powershell
   Remove-Item "C:\Users\black\OneDrive\Desktop\CogniFlex\*.log" -Force
   Remove-Item "C:\Users\black\OneDrive\Desktop\CogniFlex\logs\*.log" -Force
   ```

2. **Запуск системы** - в отдельном терминале:
   ```bash
   cd C:\Users\black\OneDrive\Desktop\CogniFlex && python -m eva_ai
   ```

3. **Мониторинг запуска**:
   - Ждать 3-4 минуты (таймаут запуска)
   - Каждые 15 секунд проверять лог `logs/eva_ai.log`
   - Критерий успешного запуска: строка `ЕВА успешно запущен` или `WebGUI сервер запущен`

4. **Остановка системы** - как только запуск подтверждён в логе

5. **Анализ логов** - через Task агента получить ВСЕ ошибки/warnings из лога

### Правила внесения ошибок:
- **Ошибки связанные напрямую** с текущим исправлением → исправляю сразу
- **Ошибки НЕ связанные** → вношу в соответствующий раздел плана
- **ВСЕ ошибки должны быть исправлены** - система должна работать без единой ошибки
- **После каждого этапа** → коммит и пуш в git

### Сверка с отчётом агента:
- После анализа логов - сверяю ВСЕ найденные ошибки с планом
- Если ошибка есть в логе но отсутствует в плане - добавляю в соответствующий раздел
- Если ошибка в плане не подтверждается логом - помечаю как N/A

---

## Статус: В РАБОТЕ

## Логика архитектуры
```
memory/ → reasoning/ → generation/ → core brain
     ↓           ↓            ↓           ↓
   fractal    entity      pipeline    event_bus
  graph_v2   extractor   coordinator   model_access
```

---

## Приоритет 1: Core Brain и EventBus

### 1.1 EventBus - Исправление подписки обработчиков
- **Проблема**: `_handle_component_error()` missing 1 required positional argument: 'event'
- **Статус**: ✅ ГОТОВО
- **Файл**: `eva_ai/core/event_bus.py`
- **Изменения**: Исправлены 3-элементные кортежи (subscription_id, weak_handler, priority)

### 1.2 MemoryManager - Отсутствующие методы
- **Проблема**: `'MemoryManager' object has no attribute 'clear_cache'` и `'optimize'`
- **Статус**: ✅ ГОТОВО
- **Файл**: `eva_ai/memory/manager_core.py`
- **Решение**: Добавлена проверка `hasattr` перед вызовом методов в `_deferred_optimize` и `_deferred_cleanup`

### 1.3 Signal Handler Signatures - EventBus subscribers
- **Проблема**: `EventSubscriptionMixin._on_component_ready()` missing 1 required positional argument: 'event'
- **Статус**: ✅ ГОТОВО
- **Файлы**: 
  - `eva_ai/core/brain_coordination.py` - `_on_component_ready` (event=None → event)
  - `eva_ai/core/system_state.py` - `_handle_component_initialized`, `_handle_component_started`, `_handle_component_stopped` (event=None → event)
  - `eva_ai/core/event_bus_bridge.py` - `_on_new_event` (event=None → event)

---

## Приоритет 2: Model & Pipeline

### 2.1 Model A File Not Found
- **Проблема**: `Model A file not found: .../qwen2.5-3b-instruct-q4_k_m.gguf`
- **Статус**: ⚠️ НЕАКТУАЛЬНО - Model A (qwen2.5-3b) не используется в текущей архитектуре (qwen3 4b + coder)

### 2.2 Tokenizer не загружается
- **Проблема**: `Tokenizer вернул False (опциональный компонент)`
- **Статус**: 🔨 В РАБОТЕ
- **Файлы**: 
  - `eva_ai/response_generator.py`
  - `eva_ai/mlearning/hybrid_model_manager.py`

### 2.3 System Memory Critical (95-99%)
- **Проблема**: `High memory_percent level: 95.4%`, `CRITICAL memory_percent level: 99.4%`
- **Статус**: 🔨 В РАБОТЕ
- **Файл**: `eva_ai/core/resource_manager.py` - оптимизация использования памяти

---

## Приоритет 3: FractalGraph v2

### 3.1 FractalGraphV2.get_clusters() - не существует
- **Проблема**: ConceptMiner делает O(n²) на лету
- **Статус**: ⏳ В ОЧЕРЕДИ
- **Файл**: `eva_ai/memory/fractal_graph_v2/` - реализовать метод get_clusters()

### 3.2 Knowledge Graph Factory Not Found
- **Проблема**: `[WARN] Factory for knowledge_graph not found - skipped`
- **Статус**: ⏳ В ОЧЕРЕДИ
- **Файл**: `eva_ai/component_initializer.py` - проверить регистрацию factory

---

## Приоритет 4: Knowledge & Concepts

### 4.1 ConceptExtractor - не сохраняет концепты
- **Проблема**: Только возвращает список, не сохраняет в FGv2
- **Статус**: ⏳ В ОЧЕРЕДИ
- **Файл**: `eva_ai/knowledge/concept_extractor.py`

### 4.2 ContradictionGenerator - интеграция
- **Проблема**: Создан, но не интегрирован в brain_query
- **Статус**: ⏳ В ОЧЕРЕДИ
- **Файл**: `eva_ai/contradiction/contradiction_generator.py`

### 4.3 Tavily API Key Not Found
- **Проблема**: `Tavily API key not found` - API key искался в корне конфига, а лежит в `web_search.tavily_api_key`
- **Статус**: ✅ ГОТОВО
- **Файлы**: 
  - `eva_ai/websearch/web_search_integrated.py` - исправлен путь к tavily_api_key
  - `eva_ai/gui/web_gui/server_api_wikipedia.py` - исправлен путь к tavily_api_key

---

## Приоритет 5: Security

### 5.1 SecurityFramework - HARDCODED backdoor
- **Проблема**: admin:admin backdoor (CVSS 9.8)
- **Статус**: ⏳ В ОЧЕРЕДИ
- **Файл**: `eva_ai/core/security_framework.py:141`

---

## Выполненные исправления

| # | Дата | Исправление | Файл | Статус |
|---|------|------------|------|--------|
| 1 | - | EventBus 3-element tuple fix | event_bus.py | ✅ DONE |
| 2 | 2026-04-14 | Signal handler signatures (4 метода) | brain_coordination.py, system_state.py, event_bus_bridge.py | ✅ DONE |
| 3 | 2026-04-14 | MemoryManager deferred commands with hasattr check | manager_core.py | ✅ DONE |
| 4 | 2026-04-14 | Tavily API Key path fix (config.web_search.tavily_api_key) | web_search_integrated.py, server_api_wikipedia.py | ✅ DONE |
