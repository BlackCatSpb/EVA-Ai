# EVA AI - Краткий справочник

## Быстрые команды

### Запуск
```powershell
cd C:\Users\black\OneDrive\Desktop\CogniFlex
Remove-Item "*.log" -Force
python -m eva_ai
```

### Тестирование API
```bash
# Чат
curl -X POST http://localhost:5555/api/chat -H "Content-Type: application/json" -d '{"message":"Привет","session_id":"test"}'

# Статус
curl http://localhost:5555/api/status
```

---

## Ключевые файлы

| Что искать | Файл |
|------------|------|
| Точка входа | `eva_ai/__main__.py` |
| Главный класс | `eva_ai/core/core_brain.py` |
| Обработка запросов | `eva_ai/core/brain_query.py` |
| Создание компонентов | `eva_ai/core/init_factories.py` |
| Graph знаний | `eva_ai/memory/fractal_graph_v2/` |
| Концепты | `eva_ai/knowledge/concept_extractor.py` |
| Майнинг | `eva_ai/knowledge/concept_miner.py` |
| Противоречия | `eva_ai/contradiction/contradiction_miner.py` |
| Самодиалог | `eva_ai/learning/dialog_core.py` |
| FCP | `eva_ai/fcp_core/`, `eva_ai/fcp_gnn/` |

---

## Приоритеты исправлений

### C1: FCP изолирован
- Добавить WebSearch и Ethics в FCP Pipeline
- Файлы: `fcp_pipeline.py`, `cross_analysis_fcp_ethics.md`

### C2: Дублирование /api/chat
- Удалить старые route из `server_routes.py`
- Оставить только в `gui/web_gui/server_routes.py`
- Файл: `cross_analysis_server_monitoring.md`

### C3: Три системы детекции
- Объединить: ContradictionGenerator + ContradictionMiner + detect_*.py
- Файл: `cross_analysis_dialog_miners.md`

### H1: KGAdapter
- Создать в `init_factories.py`
- Файл: `cross_analysis_core_memory.md`

### H2: summary_parts
- Исправить в `dialog_core.py:1049`
- Файл: `self_dialog_system.md`

---

## Структура логов

```
C:\Users\black\OneDrive\Desktop\CogniFlex\
├── eva_ai.log        # Основные логи
├── error.log         # Ошибки
└── web_gui.log       # GUI логи
```

---

## Частые проблемы

| Проблема | Решение |
|----------|---------|
| Модель не загружается | Проверить путь в `brain_config.json` |
| Ошибка в _extract_key_concepts | Проверить `brain_query.py` |
| Самодиалог не запускается | Проверить `dialog_core.py` |
| ConceptMiner не работает | Проверить EventBus подписку |

---

## Документация

- Полное руководство: `analysis/ONBOARDING.md`
- Анализы: `analysis/*.md`
- Перекрёстные анализы: `analysis/cross_analysis_*.md`