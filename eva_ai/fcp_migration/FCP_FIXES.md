# ПЛАН НЕДОРАБОТОК FCP Pipeline V15

**Дата:** 25.04.2026  
**Статус:** Требует доработки

---

## КРИТИЧНЫЕ (Блокируют работу)

### 1. Streaming отсутствует
**Проблема:** Ответ выдаётся сразу целиком, нет плавного стриминга
**Где:** `server_routes_chat.py` + `fcp_pipeline.py`
**Как должно быть:** Ответ долженAppear постепенно, токен за токеном
**Как сейчас:** `pipeline.generate()` возвращает полный ответ сразу

**Решение:**
- Использовать streamer API OpenVINO GenAI
- Или реализовать свой стриммер
- В GUI использовать SSE (Server-Sent Events) для передачи

**Файлы для 修改:**
- `eva_ai/gui/web_gui/server_routes_chat.py`
- `eva_ai/core/fcp_pipeline.py`

---

### 2. enable_thinking не работает
**Проблема:** В GUI передаётся `enable_thinking=False`, модель не использует Chain-of-Thought
**Где:** `server_routes_chat.py:204`
**Как было:** `enable_thinking=False`
**Как исправлено:** `enable_thinking=True` (уже исправлено)

**Статус:** ✅ Исправлено

---

## ВАЖНЫЕ (Улучшают функциональность)

### 3. ThinkingController - не реализован
**Спецификация (FCP.txt, стр. 48):** "динамически включает/отключает режим рассуждений Qwen3 в зависимости от наличия противоречий в графе и сложности запроса"
**Текущий статус:** Заглушка - просто добавляет токены `<|im_start|>assistant\n<think>\n`
**Как должно быть:**
- Анализировать запрос на сложность
- Проверять противоречия в графе
- Динамически включать/выключать thinking

**Файлы для разработки:**
- `eva_ai/tools/fcp/thinking_controller.py` - нужно расширить логику
- `eva_ai/core/fcp_pipeline.py` - интегрировать в generate()

### 4. ToolOrchestrator - не интегрирован
**Спецификация (FCP.txt, стр. 47):** "модель может генерировать специальные токены для вызова внешних инструментов (калькулятор, веб-поиск)"
**Текущий статус:** Класс существует, но не вызывается из пайплайна
**Как должно быть:**
- После генерации проверять есть ли tool_call токены
- Вызывать инструменты через ToolOrchestrator
- Добавлять результаты в контекст

**Файлы для разработки:**
- `eva_ai/tools/fcp/orchestrator.py` - интегрировать
- `eva_ai/core/fcp_pipeline.py` - добавить вызов после _generate()

### 5. AttributionReport - не интегрирован
**Спецификация (FCP.txt, стр. 49):** "собирает данные о том, какие слои, графовые узлы и LoRA-адаптеры повлияли на ответ"
**Текущий статус:** Класс существует, но не используется
**Как должно быть:**
- Отслеживать какие концепты использованы
- Записывать какой LoRA адаптер активен
- Предо��тавлять explanation по запросу

**Файлы для разработки:**
- `eva_ai/tools/fcp/attribution.py` - интегрировать
- `eva_ai/core/fcp_pipeline.py` - вызывать в generate()

---

## СРЕДНЕЙ ВАЖНОСТИ (Расширяют возможности)

### 6. GNN Инъекция - отключена
**Спецификация (FCP.txt, стр. 18-19):** "на определённых слоях выполняется инъекция графового вектора"
**Текущий статус:** `enable_injection=False` по умолчанию из-за ошибки размерностей
**Проблема:** GraphEncoder возвращает несовместимые размерности (384 vs ожидаемое)
**Как должно быть:**
- Получать подграф из FractalGraphV2
- Кодировать через GNN
- Впрыскивать в скрытые состояния

**Файлы для разработки:**
- `eva_ai/knowledge/fcp_gnn/graph_encoder.py` - исправить размерности
- `eva_ai/core/fcp_pipeline.py` - включить (после исправления)

### 7. ScenarioTCM - не интегрирован
**Спецификация (FCP.txt, стр. 25):** "сохраняющее цепочки диалогов как сценарии"
**Текущий статус:** Класс существует, но не используется
**Как должно быть:**
- Сохранять каждую пару (query, response)
- Извлекать по контексту для future use

**Файлы для разработки:**
- `eva_ai/memory/fcp/scenario_tcm.py` - интегрировать
- `eva_ai/core/fcp_pipeline.py` - вызывать в generate()

### 8. ExpertSystem - не интегрирован
**Спецификация (FCP.txt, стр. 46):** "несколько экспертов с разными LoRA, критик выявляет противоречия"
**Текущий статус:** Класс существует, не используется
**Как должно быть:**
- Запускать 2-3 экспертов с разными LoRA
- Critic анализирует противоречия
- Голосование за финальный ответ

**Файлы для разработки:**
- `eva_ai/tools/fcp/expert_system.py` - интегрировать
- `eva_ai/core/fcp_pipeline.py` - опционально для сложных случаев

### 9. SemanticCacheEvictor - не интегрирован
**Спецификация (FCP.txt, стр. 45):** "анализирует важность токенов в KV-кэше"
**Текущий статус:** Класс существует, не используется
**Как должно быть:**
- Оценивать важность блоков кэша
- Вытеснять менее важные

**Файлы для разработки:**
- `eva_ai/memory/fcp/semantic_cache_evictor.py` - интегрировать

---

## ФОНОВЫЕ СИСТЕМЫ (Запускаются отдельно)

### 10. GraphCurator - требует интеграции с EVA
**Спецификация (FCP.txt, стр. 31):** "фоновый процесс, запускающий дедупликацию, временной распад, поиск противоречий"
**Текущий статус:** Не запущен
**Как должно быть:**
- Запускаться раз в N минут
- Очищать orphaned узлы
- Искать противоречия
- Генерировать уточняющие вопросы через ClarificationGenerator

### 11. LearningGraphManager + LearningOrchestrator
**Спецификация (FCP.txt, стр. 35-36):** "ведёт учёт сигналов ��братной связи и статистики по слоям/доменам"
**Текущий статус:** Не интегрирован в generate loop
**Как должно быть:**
- После каждого запроса добавлять сигнал (query, response, success, confidence)
- Анализировать проблемные домены
- Запускать дообучение LoRA при деградации

### 12. ShadowLoRAManagerOV
**Спецификация (FCP.txt, стр. 38):** "атомарная замена LoRA в работающем пайплайне"
**Текущий статус:** Частично работает (пытается загрузить fcp_finetuned)
**Как проверить:**
```
pipeline.list_available_adapters()
pipeline.get_current_adapter()
```

---

## ИНТЕГРАЦИЯ С EVA

### 13. FractalGraphV2 интеграция
**Текущий статус:** graph_path=None
**Как должно быть:**
- Передавать fractal_graph_v2 в graph_path при инициализации
- Использовать для извлечения подграфа (HNSW)

### 14. ConceptExtractor / ConceptMiner интеграция
**Как должно быть:**
- После генерации извлекать концепты
- Добавлять в FractalGraphV2
- Добавлять в очередь самодиалога

### 15. SelfDialogLearning интеграция  
**Как должно быть:**
- Использовать FCPPipelineV15 для самодиалога
- Обрабатывать концепты и противоречия

---

## ИСПРАВЛЕНИЯ ДЛЯ ЭТАПА 2

### Приоритет 1 - Стриминг:
1. Добавить streamer в FCPPipelineV15.generate()
2. Обновить server_routes_chat.py для обработки стрима

### Приоритет 2 - Thinking:
1. ✅ enable_thinking=True в GUI (уже)
2. Проверить работоспособность

### Приоритет 3 - Интеграция компонентов:
1. ToolOrchestrator вызов после генерации
2. AttributionReport после генерации  
3. ScenarioTCM сохранение диалога

---

## Testing команды

```bash
# Тест стриминга
python test_fcp.py

# Проверка LoRA
python -c "from eva_ai.core.fcp_pipeline import FCPPipelineV15; p = FCPPipelineV15(...); print(p.list_available_adapters())"

# Проверка атрибуции
python -c "from eva_ai.tools.fcp.attribution import AttributionReport; print(AttributionReport().explain())"
```

---

## Статус by component

| Компонент | Статус | Файл |
|-----------|--------|-------|
| OpenVINO Pipeline | ✅ Working | fcp_pipeline.py |
| Tokenizer | ✅ Working | - |
| LoRA Manager | ⚠️ Частично | shadow_lora_manager.py |
| Thinking | ⚠️ Заглушка | fcp_pipeline.py:_build_prompt |
| ToolOrchestrator | ❌ Не интегрирован | tools/fcp/orchestrator.py |
| Attribution | ❌ Не интегрирован | tools/fcp/attribution.py |
| ScenarioTCM | ❌ Не интегрирован | memory/fcp/scenario_tcm.py |
| ExpertSystem | ❌ Не интегрирован | tools/fcp/expert_system.py |
| GraphEncoder | ❌ Ошибка размерностей | knowledge/fcp_gnn/graph_encoder.py |
| GraphCurator | ❌ Не запущен | knowledge/fcp_graph_curator.py |
| LearningManager | ❌ Не интегрирован | knowledge/fcp_learning_manager.py |
| Streaming | ❌ Нет | fcp_pipeline.py |