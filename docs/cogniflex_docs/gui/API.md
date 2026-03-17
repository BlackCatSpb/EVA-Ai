# API документация GUI CogniFlex

Документ содержит справочную информацию по программному интерфейсу графического интерфейса CogniFlex.

## CogniFlexGUI

Файл: cogniflex/gui/core_gui.py

Основной класс полнофункционального графического интерфейса.

### Конструктор

def __init__(self, brain=None, integrator=None, cache_dir: Optional[str] = None)

Параметры:
- brain: Ссылка на CoreBrain
- integrator: Ссылка на CogniFlexIntegrator
- cache_dir: Путь к каталогу кэша

### Методы

- process_query_via_integrator(query, context) -> Dict
- get_system_status_via_integrator() -> Dict
- start_self_dialog_via_integrator()
- optimize_system_via_integrator()
- start()
- stop()
- reload()
- show_toast(message, level, duration, key)
- update_status(status, details)

## IntegratedCogniFlexGUI

Файл: cogniflex/gui/integrated_gui.py

Класс интегрированного GUI с поддержкой фрактальной архитектуры.

### Конструктор

def __init__(self, brain_or_integrator)

### Методы

- send_message(event=None)
- process_query(query)
- start_self_dialog()
- optimize_system()
- stop_system()

## ChatModule

Файл: cogniflex/gui/chat_module.py

Модуль чата для взаимодействия пользователя с системой.

### Конструктор

def __init__(self, gui)

### Методы

- activate()
- deactivate()
- _send_message()
- _toggle_self_dialog()
- _on_import_document()

## Функции модуля

- create_gui(brain=None, cache_dir=None) -> CogniFlexGUI
- create_integrated_gui(integrator) -> IntegratedCogniFlexGUI

## Настройки GUI

Файл: cogniflex/gui/settings.py

### Функции

- get_default_settings() -> Dict
- load_settings(settings_path) -> Dict
- save_settings(settings, settings_path)

### Параметры настроек

- theme: light или dark
- language: код языка
- font_size: размер шрифта
- show_reasoning: отображение панели рассуждений
- compact_mode: компактный режим
- show_notifications: показ уведомлений
- notification_duration: длительность уведомлений
- auto_update_interval: интервал обновления
