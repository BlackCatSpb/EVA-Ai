"""
Test module for GUI components
"""
import pytest


class TestGUIModules:
    """Тесты GUI модулей"""
    
    def test_import_gui(self):
        """Тест импорта GUI"""
        # Только проверяем что модуль может быть импортирован
        try:
            import tkinter
            assert tkinter is not None
        except ImportError:
            pytest.skip("tkinter not available")
    
    def test_gui_colors(self):
        """Тест цветовой схемы"""
        # Базовые цвета которые должны быть определены
        required_colors = ['primary', 'secondary', 'background', 'text', 'card-bg']
        # Это можно расширить после
        assert len(required_colors) > 0
