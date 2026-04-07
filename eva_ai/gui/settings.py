"""
Модуль для работы с настройками графического интерфейса
"""
import logging
import json

logger = logging.getLogger(__name__)

import os
from typing import Dict, Any

def get_default_settings() -> Dict[str, Any]:
    """Возвращает настройки по умолчанию."""
    return {
        "gui": {
            "theme": "light",
            "language": "ru",
            "font_size": 10,
            "show_reasoning": True,
            "compact_mode": False,
            "show_notifications": True,
            "notification_duration": 5000,
            "auto_update_interval": 5000
        },
        "system": {
            "cache_dir": "eva_cache"
        }
    }

def load_settings(settings_path: str) -> Dict[str, Any]:
    """Загружает настройки из файла."""
    try:
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.info(f"Ошибка загрузки настроек: {e}")
    
    return get_default_settings()

def save_settings(settings: Dict[str, Any], settings_path: str):
    """Сохраняет настройки в файл."""
    try:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.info(f"Ошибка сохранения настроек: {e}")