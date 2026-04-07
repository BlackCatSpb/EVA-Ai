"""Message display, rendering, formatting, and timestamps for the chat module."""
import tkinter as tk
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from eva_ai.gui.chat_text_utils import _to_display_str, _fix_mojibake

logger = logging.getLogger("eva_ai.gui.chat")


class ChatMessagesMixin:
    """Mixin providing message display, rendering, formatting, and timestamp functionality."""

    def _add_message(self, sender: str, message: str, msg_type: str = "user",
                     timestamp: Optional[float] = None,
                     process_formatting: bool = True,
                     extras: Optional[Dict[str, Any]] = None):
        """Добавляет сообщение в чат."""
        import json
        if timestamp is None:
            timestamp = time.time()

        # Сохранение в историю
        if not getattr(self, '_suppress_history_append', False):
            entry = {
                "sender": sender,
                "message": message,
                "type": msg_type,
                "timestamp": timestamp
            }

            if extras and isinstance(extras, dict):
                try:
                    safe_extras = json.loads(json.dumps(extras, ensure_ascii=False, default=str))
                except Exception as e:
                    logger.debug(f"Error serializing extras: {e}")
                    safe_extras = {}
                for k, v in extras.items():
                    try:
                        safe_extras[k] = json.loads(json.dumps(v, ensure_ascii=False, default=str))
                    except Exception as e:
                        logger.debug(f"Error serializing extra key {k}: {e}")
                        safe_extras[k] = str(v)
                entry.update({"extras": safe_extras})

            with self._history_lock:
                self.message_history.append(entry)
                self._save_history_incremental()

                # Ограничение размера истории
                if len(self.message_history) > 500:
                    entry = self.message_history[0]
                    if isinstance(entry.get("extras"), dict):
                        large_keys = [k for k, v in entry["extras"].items()
                                      if isinstance(v, (list, dict)) and len(str(v)) > 1000]
                        for k in large_keys:
                            entry["extras"][k] = f"<truncated {len(entry['extras'][k])} items>"
                    self.message_history = self.message_history[-500:]

        # Отображение
        try:
            if not self.chat_display.winfo_exists():
                return

            self.chat_display.config(state=tk.NORMAL)

            # Новая строка
            try:
                last_char = self.chat_display.get("end-2c", "end-1c")
                if last_char not in ("\n", ""):
                    self.chat_display.insert(tk.END, "\n")
            except Exception as e:
                logger.debug(f"Error adding new line: {e}")

            # Временная метка
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            self.chat_display.insert(tk.END, f"[{time_str}] ", "timestamp")

            # Отправитель
            tag = "user" if msg_type == "user" else msg_type
            self.chat_display.insert(tk.END, f"{sender}: ", tag)

            # Сообщение
            if process_formatting:
                self._process_and_insert_formatted_message(_to_display_str(message))
            else:
                self.chat_display.insert(tk.END, _to_display_str(message))

            self.chat_display.insert(tk.END, "\n")
            self.chat_display.config(state=tk.DISABLED)

            # Force widget to update/refresh to prevent display artifacts
            self.chat_display.update_idletasks()
            self.chat_display.see(tk.END)

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка добавления сообщения в чат: {e}", exc_info=True)

    def _process_and_insert_formatted_message(self, message: str):
        """Обрабатывает и вставляет форматированное сообщение."""
        try:
            if not self.chat_display.winfo_exists():
                return

            start_index = self.chat_display.index(tk.END)
            self.chat_display.insert(tk.END, message)

            # Форматирование
            current_pos = 0
            for match in self.formatting_pattern.finditer(message):
                start_idx = match.start()
                end_idx = match.end()
                matched_text = match.group(0)

                if matched_text.startswith("**") or matched_text.startswith("__"):
                    format_type = "bold"
                    content = match.group(1) or match.group(2)
                elif matched_text.startswith("_"):
                    format_type = "italic"
                    content = match.group(3)
                elif matched_text.startswith("`"):
                    format_type = "code"
                    content = match.group(4)
                else:
                    continue

                if not content:
                    continue

                # Позиции
                start_tag_pos = f"{start_index}+{start_idx + current_pos}c"
                end_tag_pos = f"{start_index}+{start_idx + current_pos + len(matched_text)}c"
                content_start = f"{start_index}+{start_idx + current_pos + 2}c"
                content_end = f"{start_index}+{start_idx + current_pos + len(content) + 2}c"

                # Замена
                self.chat_display.delete(start_tag_pos, end_tag_pos)
                self.chat_display.insert(start_tag_pos, content)
                self.chat_display.tag_add(format_type, content_start, content_end)

                current_pos -= len(matched_text) - len(content)

            # Гиперссылки
            self._process_markdown_links(message, start_index)
            self._process_urls(message, start_index)
            self._process_emojis(message, start_index)
            self._process_images(message, start_index)

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки форматирования сообщения: {e}", exc_info=True)

    def _process_urls(self, message: str, start_index: str):
        """Обрабатывает URL в сообщении."""
        try:
            if not self.chat_display.winfo_exists():
                return

            for match in self.url_pattern.finditer(message):
                start_idx = match.start()
                end_idx = match.end()

                url_start = f"{start_index}+{start_idx}c"
                url_end = f"{start_index}+{end_idx}c"

                self.chat_display.tag_add("url", url_start, url_end)
                self.chat_display.tag_add(f"url_{match.group(0)}", url_start, url_end)

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки URL: {e}", exc_info=True)

    def _process_markdown_links(self, message: str, start_index: str):
        """Обрабатывает markdown-ссылки [текст](url)."""
        try:
            if not self.chat_display.winfo_exists():
                return

            offset_correction = 0
            for match in self.markdown_link_pattern.finditer(message):
                link_text = match.group(1)
                url = match.group(2)
                md_start = match.start()
                md_end = match.end()

                start_pos = f"{start_index}+{md_start + offset_correction}c"
                end_pos = f"{start_index}+{md_end + offset_correction}c"

                self.chat_display.delete(start_pos, end_pos)
                self.chat_display.insert(start_pos, link_text)

                link_end_pos = f"{start_pos}+{len(link_text)}c"
                self.chat_display.tag_add("url", start_pos, link_end_pos)
                self.chat_display.tag_add(f"url_{url}", start_pos, link_end_pos)

                offset_correction -= (md_end - md_start) - len(link_text)

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки markdown-ссылок: {e}", exc_info=True)

    def _process_emojis(self, message: str, start_index: str):
        """Обрабатывает эмодзи в сообщении."""
        try:
            if not self.chat_display.winfo_exists():
                return

            for match in self.emoji_pattern.finditer(message):
                start_idx = match.start()
                end_idx = match.end()

                emoji_start = f"{start_index}+{start_idx}c"
                emoji_end = f"{start_index}+{end_idx}c"

                self.chat_display.tag_add("emoji", emoji_start, emoji_end)

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки эмодзи: {e}", exc_info=True)

    def _process_images(self, message: str, start_index: str):
        """Обрабатывает изображения в сообщении."""
        try:
            if not self.chat_display.winfo_exists():
                return

            for match in self.image_pattern.finditer(message):
                url = match.group(1)
                start_idx = match.start(1)
                end_idx = match.end(1)

                url_start = f"{start_index}+{start_idx}c"
                url_end = f"{start_index}+{end_idx}c"

                self.chat_display.delete(url_start, url_end)
                self.chat_display.insert(url_start, "[Изображение]")

                self.chat_display.tag_add("url", url_start, f"{url_start}+10c")
                self.chat_display.tag_bind("url", "<Button-1>",
                    lambda e, u=url: self._open_image(u))

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки изображений: {e}", exc_info=True)

    def _configure_chat_tags(self):
        """Настраивает стили тегов для сообщений."""
        self.chat_display.tag_configure("user",
            foreground=self.gui.colors['primary'],
            font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_configure("system",
            foreground=self.gui.colors['text'],
            font=('Segoe UI', 10))
        self.chat_display.tag_configure("reasoning",
            foreground=self.gui.colors['text-muted'],
            font=('Segoe UI', 10, 'italic'))
        self.chat_display.tag_configure("timestamp",
            foreground=self.gui.colors['text-muted'],
            font=('Segoe UI', 8))
        self.chat_display.tag_configure("url",
            foreground=self.gui.colors['primary'],
            underline=True)
        self.chat_display.tag_configure("bold", font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_configure("italic", font=('Segoe UI', 10, 'italic'))
        self.chat_display.tag_configure("code",
            background=self.gui.colors['bg'],
            font=('Consolas', 9))
        self.chat_display.tag_configure("emoji", font=('Segoe UI Emoji', 10))

    def _remove_last_message(self):
        """Удаляет последнее сообщение из чата."""
        try:
            if not self.chat_display.winfo_exists():
                return

            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("end-2l", "end")
            self.chat_display.config(state=tk.DISABLED)

            with self._history_lock:
                if self.message_history:
                    self.message_history.pop()

        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка удаления последнего сообщения: {e}", exc_info=True)

    def _show_welcome_message(self):
        """Показывает приветственное сообщение."""
        welcome_msg = (
            "Добро пожаловать в ЕВА!\n"
            "Я - когнитивная система с поддержкой:\n"
            "• Этического анализа\n"
            "• Адаптации под пользователя\n"
            "• Распределенных вычислений\n"
            "• Управления знаниями\n"
            "Задайте ваш первый вопрос или нажмите F1 для просмотра справки."
        )
        self._add_message("ЕВА", welcome_msg, "system")

    def _redraw_chat(self):
        """Перерисовывает чат с новой темой."""
        if not self.message_history:
            return

        try:
            if not self.chat_display.winfo_exists():
                return

            current_pos = self.chat_display.yview()

            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)

            for msg in self.message_history:
                self._add_message(
                    msg["sender"],
                    msg["message"],
                    msg["type"],
                    timestamp=msg["timestamp"],
                    process_formatting=False
                )

            self.chat_display.yview_moveto(current_pos[0])
            self.chat_display.config(state=tk.DISABLED)

        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка перерисовки чата: {e}", exc_info=True)
