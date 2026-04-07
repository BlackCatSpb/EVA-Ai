"""History loading, pagination, and scrolling for the chat module."""
import os
import json
import logging
from typing import Any, Dict

logger = logging.getLogger("eva_ai.gui.chat")


class ChatHistoryMixin:
    """Mixin providing history loading, pagination, and scrolling functionality."""

    def _load_chat_history(self):
        """Загружает историю чата из файла."""
        try:
            if self.gui.cache_dir:
                os.makedirs(self.gui.cache_dir, exist_ok=True)
                history_file = os.path.join(self.gui.cache_dir, "chat_history.json")

                if os.path.exists(history_file):
                    with open(history_file, "r", encoding="utf-8") as f:
                        loaded = json.load(f)

                    with self._history_lock:
                        self.message_history = list(loaded)

                    self._suppress_history_append = True
                    try:
                        for msg in loaded:
                            if not isinstance(msg, dict):
                                continue
                            sender = msg.get("sender", "Unknown")
                            message = msg.get("message", "")
                            msg_type = msg.get("type", "system")
                            timestamp = msg.get("timestamp")
                            extras = msg.get("extras")

                            if message:
                                self._add_message(
                                    sender,
                                    message,
                                    msg_type,
                                    timestamp=timestamp,
                                    process_formatting=False,
                                    extras=extras
                                )
                    finally:
                        self._suppress_history_append = False

        except (OSError, IOError, PermissionError, json.JSONDecodeError) as e:
            logger.error(f"Ошибка загрузки истории чата: {e}", exc_info=True)

    def _save_history_incremental(self):
        """Сохраняет историю чата после каждого сообщения."""
        try:
            if not getattr(self.gui, 'cache_dir', None):
                return

            os.makedirs(self.gui.cache_dir, exist_ok=True)
            history_file = os.path.join(self.gui.cache_dir, "chat_history.json")

            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(self.message_history, f, ensure_ascii=False, indent=2)

        except (OSError, IOError, PermissionError, TypeError, ValueError) as e:
            logger.debug(f"Не удалось инкрементально сохранить историю чата: {e}")

    def _save_chat_history(self):
        """Сохраняет историю сообщений."""
        if not hasattr(self, 'message_history'):
            return

        try:
            if self.gui.cache_dir:
                os.makedirs(self.gui.cache_dir, exist_ok=True)
                history_file = os.path.join(self.gui.cache_dir, "chat_history.json")
                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump(self.message_history, f, ensure_ascii=False, indent=2)
        except (OSError, IOError, PermissionError, TypeError, ValueError) as e:
            logger.error(f"Ошибка сохранения истории чата: {e}")
