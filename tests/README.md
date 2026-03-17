# CogniFlex Test Suite

## Структура тестов

```
tests/
├── __init__.py
├── conftest.py           # Общие фикстуры
├── test_core/            # Тесты ядра
│   ├── __init__.py
│   └── test_*.py
├── test_gui/             # Тесты GUI
│   ├── __init__.py
│   └── test_*.py
├── test_ml/              # Тесты ML
│   ├── __init__.py
│   └── test_*.py
└── test_utils/           # Тесты утилит
    ├── __init__.py
    └── test_*.py
```

## Запуск тестов

```bash
# Все тесты
pytest tests/

# Конкретная директория
pytest tests/test_gui/

# С покрытием
pytest tests/ --cov=cogniflex --cov-report=html
```

## Требования

```
pytest>=7.3.1
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0
```

## Пример теста

```python
import pytest
from cogniflex.gui.core_gui import CogniFlexGUI

def test_gui_creation():
    """Тест создания GUI"""
    gui = CogniFlexGUI()
    assert gui is not None
    assert hasattr(gui, 'notebook')
```

---

*Created: 2026-03-17*
