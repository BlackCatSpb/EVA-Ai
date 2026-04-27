# Анализ GUI & Core Files EVA

## Часть 1: GUI модули (дополнение)

### Widgets и Components (39 файлов в gui/)

| Категория | Файлы |
|-----------|-------|
| Chat | chat_module.py, chat_input.py, chat_messages.py, chat_history.py, chat_actions.py |
| Knowledge Graph | kg_visualization.py, kg_nodes.py, kg_actions.py, kg_search.py, kg_stats.py |
| Learning | learning_module.py, contradiction_module.py |
| Analytics | analytics_module.py, analytics_types.py |
| Memory | memory_module.py, neuromorphic_module.py |
| Core | gui_main.py, core_gui.py, base_gui.py, gui_modules.py |
| Themes/Layout | gui_themes.py, gui_tabs.py, gui_types.py |

### Основные компоненты

**Chat Interface:**
- Текстовый ввод с историей
- Отображение сообщений
- Действия (копирование, редактирование)

**Knowledge Graph Visualization:**
- Визуализация графа
- Узлы и связи
- Статистика

**Analytics Dashboard:**
- Графики использования
- Метрики системы

### Статус: ИСПОЛЬЗУЕТСЯ
Tkinter/PyQt GUI для десктоп приложения

---

## Часть 2: Core Files

### Файлы
- `run.py` - точка входа
- `__main__.py` - CLI интерфейс
- `server.py` - Flask сервер
- `server_routes.py` - маршруты API
- `server_handlers.py` - обработчики

### run.py

**Запуск EVA:**
```python
if __name__ == "__main__":
    CoreBrain().run()
```

### server.py

**Flask приложение:**
- Route: /, /api/chat, /api/graph, /api/concepts
- Template: chat.html
- Static: CSS/JS

### __main__.py

**CLI команды:**
- `python -m eva_ai` - запуск
- Параметры: --config, --profile, --debug

### Статус: ОСНОВА

---

## Адаптеры

### torch_adapter.py

**Функции:**
- Конвертация моделей
- Утилиты для PyTorch

### Статус: ИСПОЛЬЗУЕТСЯ

---

## Выводы

| Компонент | Статус |
|-----------|--------|
| GUI | ✅ Активен (39 файлов) |
| Core files | ✅ Основа системы |
| Adapters | ✅ Используется |

Все основные модули EVA проанализированы!