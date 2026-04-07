"""Reasoning steps display panel for the chat module."""
import tkinter as tk
from tkinter import ttk, scrolledtext
import logging
from typing import Optional

from eva_ai.gui.chat_text_utils import _fix_mojibake

logger = logging.getLogger("eva_ai.gui.chat")


class ChatReasoningMixin:
    """Mixin providing reasoning panel display and toggle functionality."""

    def _init_reasoning_panel(self):
        """Создает панель рассуждений с возможностью сворачивания."""
        try:
            self.reasoning_frame = ttk.Frame(self.chat_frame)
            self.reasoning_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=(0, 5))

            header = ttk.Frame(self.reasoning_frame)
            header.pack(fill=tk.X)
            self.reasoning_header = header

            self.reasoning_toggle_btn = ttk.Button(
                header,
                text="▶ Рассуждения",
                command=self._toggle_reasoning_panel
            )
            self.reasoning_toggle_btn.pack(side=tk.LEFT)

            self.reasoning_text = scrolledtext.ScrolledText(
                self.reasoning_frame,
                height=8,
                wrap=tk.WORD,
                state=tk.DISABLED,
                bg=self.gui.colors['card-bg'],
                fg=self.gui.colors['text-muted'],
                font=('Segoe UI', 10, 'italic'),
                padx=10,
                pady=8
            )

            self.reasoning_visible = False

        except (AttributeError, TypeError, RuntimeError, tk.TclError) as e:
            logger.debug(f"Error initializing reasoning panel: {e}")
            self.reasoning_frame = None
            self.reasoning_text = None
            self.reasoning_toggle_btn = None
            self.reasoning_visible = False

    def _toggle_reasoning_panel(self):
        """Переключает видимость панели рассуждений."""
        try:
            if not self.reasoning_frame or not self.reasoning_text or not self.reasoning_toggle_btn:
                return

            if self.reasoning_visible:
                if self.reasoning_text.winfo_ismapped():
                    self.reasoning_text.pack_forget()
                    self.reasoning_toggle_btn.config(text="▶ Рассуждения")
                    self.reasoning_visible = False
            else:
                if not self.reasoning_text.winfo_ismapped():
                    self.reasoning_text.pack(fill=tk.BOTH, expand=False, padx=0, pady=(4, 0))
                    self.reasoning_toggle_btn.config(text="▼ Рассуждения")
                    self.reasoning_visible = True

        except (AttributeError, TypeError, RuntimeError, tk.TclError):
            pass

    def _set_reasoning_content(self, text: Optional[str], auto_expand: bool = True):
        """Устанавливает текст рассуждений."""
        try:
            if not self.reasoning_text:
                return

            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            norm = _fix_mojibake(text or "").strip()
            self.reasoning_text.config(state=tk.NORMAL)
            self.reasoning_text.delete("1.0", tk.END)

            if norm:
                self.reasoning_text.insert(tk.END, norm)

            self.reasoning_text.config(state=tk.DISABLED)

            # Авторазворачивание
            if auto_expand and norm and getattr(self.gui, "reasoning_active", True):
                if not self.reasoning_visible:
                    self._toggle_reasoning_panel()

            # Автосворачивание при пустом тексте
            if not norm and self.reasoning_visible:
                self._toggle_reasoning_panel()

        except (AttributeError, TypeError, RuntimeError, tk.TclError, UnicodeError):
            pass
