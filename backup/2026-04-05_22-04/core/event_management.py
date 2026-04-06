# event_management.py
"""Система событий для ЕВА."""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable

logger = logging.getLogger("eva.events")

class SimpleEventSystem:
    """Простая событийная система для ЕВА."""

    def __init__(self):
        self.listeners = {}
        self.logger = logging.getLogger("eva.events")

    def on(self, event_name: str, callback: Callable):
        """Регистрирует обработчик события."""
        if event_name not in self.listeners:
            self.listeners[event_name] = []
        self.listeners[event_name].append(callback)
        self.logger.debug(f"Зарегистрирован обработчик для события: {event_name}")

    def trigger(self, event_name: str, *args, **kwargs):
        """Триггерит событие."""
        try:
            if event_name in self.listeners:
                for callback in self.listeners[event_name]:
                    try:
                        callback(*args, **kwargs)
                    except Exception as e:
                        self.logger.warning(f"Ошибка в callback для {event_name}: {e}")
        except Exception as e:
            self.logger.error(f"Ошибка при триггере события {event_name}: {e}")

    def off(self, event_name: str, callback: Optional[Callable] = None):
        """Удаляет обработчик события."""
        if event_name in self.listeners:
            if callback is None:
                del self.listeners[event_name]
                self.logger.debug(f"Удалены все обработчики для события: {event_name}")
            else:
                self.listeners[event_name] = [
                    cb for cb in self.listeners[event_name] if cb != callback
                ]
                self.logger.debug(f"Удален обработчик для события: {event_name}")

    def once(self, event_name: str, callback: Callable):
        """Регистрирует обработчик, который сработает один раз."""
        def wrapper(*args, **kwargs):
            self.off(event_name, wrapper)
            callback(*args, **kwargs)

        self.on(event_name, wrapper)
        self.logger.debug(f"Зарегистрирован одноразовый обработчик для события: {event_name}")
