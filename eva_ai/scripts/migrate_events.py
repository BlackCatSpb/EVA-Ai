"""
Миграционный скрипт для логирования изменений в EventSystem.

Этот скрипт документирует изменения, связанные с переходом от старой
EventSystem к новой EventBus с обеспечением обратной совместимости.
"""

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eva_ai.migration")


def log_migration_event(message: str, level: str = "INFO"):
    """Логирование миграционных событий"""
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] [{level}] {message}"
    
    if level == "INFO":
        logger.info(log_entry)
    elif level == "WARNING":
        logger.warning(log_entry)
    elif level == "ERROR":
        logger.error(log_entry)


MIGRATION_NOTES = """
================================================================================
МИГРАЦИЯ EVENT SYSTEM -> EVENT BUS
================================================================================

ДАТА: 2026-04-05

ИЗМЕНЕНИЯ:
----------

1. eva/core/event_bus_bridge.py:
   - Добавлен метод _setup_bridges() для настройки двусторонней связи
   - Добавлен метод _bridge_old_to_new() для перенаправления событий из old EventSystem
   - Добавлен метод _bridge_new_to_old() для перенаправления событий из EventBus
   - Добавлен метод _on_new_event() для обработки новых событий
   - Добавлен метод _convert_to_old_format() для конвертации событий
   - Добавлен метод _register_bridge() для регистрации моста

2. eva/core/event_system.py:
   - Добавлен параметр event_bus в __init__ для поддержки внешнего EventBus
   - Добавлен атрибут external_event_bus для хранения внешней шины
   - Добавлен атрибут _compatibility_mode для управления режимом совместимости
   - Добавлен метод publish() для публикации событий в EventBus
   - Добавлен метод _old_publish() для обратной совместимости
   - Добавлена интеграция с EventBusBridge при наличии внешнего event_bus

СОВМЕСТИМОСТЬ:
--------------
- Старый код: event_system.trigger('event_name', data) - РАБОТАЕТ
- Новый код: event_bus.publish(Event(...)) - РАБОТАЕТ
- Двусторонняя связь: События автоматически транслируются между системами

События для моста:
- old -> new: ethical_check_request, pipeline.start, pipeline.complete, learning.started, memory.updated
- new -> old: pipeline.start, pipeline.complete, pipeline.failed, command.completed, command.failed

================================================================================
"""


def run_migration():
    """Запуск миграции"""
    log_migration_event("Начало миграции Event System -> Event Bus")
    log_migration_event(MIGRATION_NOTES)
    log_migration_event("Миграция завершена успешно")
    
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "notes": MIGRATION_NOTES
    }


if __name__ == "__main__":
    run_migration()
