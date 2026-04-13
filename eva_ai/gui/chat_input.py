"""Input area, send button, file attachment, and textarea for the chat module."""
import tkinter as tk
from tkinter import ttk, scrolledtext, Menu
import logging

logger = logging.getLogger("eva_ai.gui.chat")


class ChatInputMixin:
    """Mixin providing input area, send button, file attachment, and textarea functionality."""

    def _create_input_area(self):
        """Создает область ввода сообщения."""
        self.input_frame = ttk.Frame(self.chat_frame)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.input_text = scrolledtext.ScrolledText(
            self.input_frame,
            height=5,
            wrap=tk.WORD,
            bg=self.gui.colors['card-bg'],
            fg=self.gui.colors['text'],
            font=('Segoe UI', 11),
            insertbackground=self.gui.colors['primary'],
            highlightbackground=self.gui.colors['border'],
            highlightthickness=1
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Фокус на поле ввода
        try:
            self.input_text.focus_set()
        except (AttributeError, RuntimeError, TypeError):
            pass

        # Контекстное меню для ввода
        self._create_input_context_menu()

        # Кнопки
        self._create_input_buttons()

        # Привязка событий
        self._bind_input_events()

    def _create_input_context_menu(self):
        """Создает контекстное меню для поля ввода."""
        input_context_menu = Menu(self.input_text, tearoff=0)
        input_context_menu.add_command(label="Вырезать",
            command=lambda: self._cut_text(self.input_text))
        input_context_menu.add_command(label="Копировать",
            command=lambda: self._copy_text(self.input_text))
        input_context_menu.add_command(label="Вставить",
            command=lambda: self._paste_text(self.input_text))
        input_context_menu.add_separator()
        input_context_menu.add_command(label="Очистить",
            command=lambda: self.input_text.delete("1.0", tk.END))
        self.input_text.bind("<Button-3>",
            lambda event: input_context_menu.tk_popup(event.x_root, event.y_root))

    def _create_input_buttons(self):
        """Создает кнопки области ввода."""
        # Кнопка импорта файла
        self.import_button = ttk.Button(
            self.input_frame,
            text="Файл",
            command=self._on_import_document
        )
        self.import_button.pack(side=tk.RIGHT, padx=(3, 0))

        # Кнопка отправки
        self.send_button = ttk.Button(
            self.input_frame,
            text="Отправить",
            command=self._send_message
        )
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))

    def _bind_input_events(self):
        """Привязывает события поля ввода."""
        self.input_text.bind("<Return>", self._on_enter_pressed)
        self.input_text.bind("<Up>", self._on_history_up)
        self.input_text.bind("<Down>", self._on_history_down)

        # Горячие клавиши - используем bind_all() для перехвата на более высоком уровне
        self.input_text.bind_all("<Control-c>", self._on_copy_shortcut)
        self.input_text.bind_all("<Control-C>", self._on_copy_shortcut)
        self.input_text.bind_all("<Control-v>", self._on_paste_shortcut)
        self.input_text.bind_all("<Control-V>", self._on_paste_shortcut)
        self.input_text.bind_all("<Control-x>", self._on_cut_shortcut)
        self.input_text.bind_all("<Control-X>", self._on_cut_shortcut)
        # Ctrl+A для выделения всего
        self.input_text.bind_all("<Control-a>", self._on_select_all_shortcut)
        self.input_text.bind_all("<Control-A>", self._on_select_all_shortcut)
        self.input_text.bind_all("<Control-q>",
            lambda e: (self._quote_selection_to_input(), "break"))
        self.input_text.bind_all("<Control-Q>",
            lambda e: (self._quote_selection_to_input(), "break"))
        # Ctrl+Shift+V для вставки без форматирования
        self.input_text.bind_all("<Control-Shift-V>", self._on_paste_shortcut)
        self.input_text.bind_all("<Control-Shift-v>", self._on_paste_shortcut)

    def _create_status_bar(self):
        """Создает статус-бар с метриками."""
        self.status_frame = ttk.Frame(self.chat_frame)
        self.status_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.cpu_label = ttk.Label(self.status_frame, text="CPU: --%")
        self.cpu_label.pack(side=tk.LEFT)

        self.mem_label = ttk.Label(self.status_frame, text="RAM: --%")
        self.mem_label.pack(side=tk.LEFT, padx=(10, 0))

        # Индикатор готовности ML
        try:
            self.ml_status_canvas = tk.Canvas(
                self.status_frame,
                width=12,
                height=12,
                highlightthickness=0,
                bg=self.gui.colors['bg'])
            self.ml_status_canvas.pack(side=tk.LEFT, padx=(15, 4))
            self.ml_status_canvas.create_oval(
                1, 1, 11, 11,
                fill=self.gui.colors['success'],
                outline="",
                tags="ml_indicator")
            self.ml_status_label = ttk.Label(
                self.status_frame,
                text="Фрактальное хранилище: активно")
            self.ml_status_label.pack(side=tk.LEFT)
        except (AttributeError, TypeError, RuntimeError, tk.TclError):
            self.ml_status_canvas = None
            self.ml_status_label = None

    def _create_typing_indicator(self):
        """Создает индикатор набора текста с прогресс-баром."""
        try:
            self.typing_frame = ttk.Frame(self.chat_frame)
            
            self.typing_indicator = ttk.Label(
                self.typing_frame,
                text=self.typing_text,
                foreground=self.gui.colors['text-muted'])
            self.typing_indicator.pack(side=tk.LEFT, padx=(5, 10))
            
            self.generation_progress = ttk.Progressbar(
                self.typing_frame,
                mode='indeterminate',
                length=200
            )
            self.generation_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
            
            self.typing_frame.pack_forget()
        except (AttributeError, TypeError, RuntimeError, tk.TclError) as e:
            logger.debug(f"Error creating typing indicator: {e}")
            self.typing_indicator = None
            self.generation_progress = None
            self.typing_frame = None

    def _show_typing(self):
        """Показывает индикатор набора текста."""
        try:
            self.typing_active = True
            if self.typing_frame and self.chat_frame and self.chat_frame.winfo_exists():
                if not self.typing_frame.winfo_ismapped():
                    self.typing_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
                    if self.generation_progress:
                        self.generation_progress.start(10)
        except (AttributeError, TypeError, RuntimeError, tk.TclError):
            pass

    def _hide_typing(self):
        """Скрывает индикатор набора текста."""
        try:
            self.typing_active = False
            if self.generation_progress:
                self.generation_progress.stop()
            if self.typing_frame and self.typing_frame.winfo_ismapped():
                self.typing_frame.pack_forget()
        except (AttributeError, TypeError, RuntimeError, tk.TclError):
            pass

    def _update_generation_progress(self, value: int = None):
        """Обновляет прогресс генерации."""
        try:
            if self.generation_progress and value is not None:
                self.generation_progress['value'] = value
            elif self.generation_progress:
                if not self.generation_progress.cget('mode') == 'indeterminate':
                    self.generation_progress.configure(mode='indeterminate')
                if not self.generation_progress.winfo_ismapped():
                    self.generation_progress.start(10)
        except (AttributeError, TypeError, RuntimeError, tk.TclError):
            pass

    def _send_message(self):
        """Отправляет сообщение из поля ввода."""
        import time
        import random
        message = self.input_text.get("1.0", tk.END).strip()
        if not message:
            return

        # Добавляем контекст из выделенного текста если есть
        context_prefix = ""
        if hasattr(self, '_get_context_for_prompt') and callable(self._get_context_for_prompt):
            context_prefix = self._get_context_for_prompt()
            if context_prefix:
                message = context_prefix + message
                # Очищаем контекст после использования
                if hasattr(self, '_clear_selection_context') and callable(self._clear_selection_context):
                    self._clear_selection_context()

        # Проверка готовности - расширенная проверка
        try:
            brain = getattr(self.gui, 'brain', None)
            ml_ready = bool(getattr(brain, 'models_ready', False)) if brain else False
            fractal_ready = bool(getattr(brain, 'fractal_ready', False)) if brain else False
            has_ml_unit = hasattr(brain, 'ml_unit') and brain.ml_unit is not None if brain else False

            logger.debug(f"[DEBUG CHAT] brain={brain is not None}, models_ready={ml_ready}, fractal_ready={fractal_ready}, has_ml_unit={has_ml_unit}")

            # Если есть ml_unit - считаем что система может работать
            if has_ml_unit and not ml_ready:
                ml_ready = True
                fractal_ready = True
                logger.debug(f"[DEBUG CHAT] Enabled ml_ready=True because has_ml_unit={has_ml_unit}")

            if not ml_ready and not fractal_ready:
                info = "Модель ещё загружается. Пожалуйста, дождитесь готовности (" + \
                       self._current_ml_progress_text() + ")."
                self._add_message("ЕВА", info, "system")
                return
        except Exception as e:
            logger.debug(f"[DEBUG CHAT] Exception in check: {e}")

        # Добавление сообщения
        display_message = message
        if context_prefix:
            # Показываем пользователю только его вопрос, без технического контекста
            # Извлекаем вопрос из сообщения (после "Контекст для ответа:")
            if "Контекст для ответа:" in display_message:
                parts = display_message.split("Контекст для ответа:")
                context_part = parts[1].split("\n\n")[0] if len(parts) > 1 else ""
                question_part = parts[1].split("\n\n")[1] if len(parts) > 1 and len(parts[1].split("\n\n")) > 1 else parts[1] if len(parts) > 1 else ""
                display_message = f"📎 По контексту: {context_part[:50]}...\n\n{question_part}"
        
        self._add_message("Вы", display_message, "user")

        # Очистка поля
        self.input_text.delete("1.0", tk.END)

        # Очередь запросов
        request_id = f"req_{int(time.time())}_{random.randint(1000, 9999)}"
        self.pending_requests.add(request_id)
        self.request_queue.put({
            "message": message,
            "request_id": request_id
        })

    def _on_enter_pressed(self, event):
        """Обработка нажатия Enter."""
        try:
            # Shift=0x0001, Control=0x0004
            if (event.state & 0x0001) or (event.state & 0x0004):
                return None  # Новая строка
            self._send_message()
            return "break"
        except Exception as e:
            logger.debug(f"Error in _on_enter_pressed: {e}")
            return None

    def _on_history_up(self, event):
        """Обработка стрелки вверх (история)."""
        if self.message_history and self.history_index < len(self.message_history) - 1:
            self.history_index += 1
            msg = self.message_history[-(self.history_index + 1)]
            if isinstance(msg, dict) and msg.get("type") == "user":
                msg_text = msg.get("message", "")
                if msg_text:
                    self.input_text.delete("1.0", tk.END)
                    self.input_text.insert("1.0", msg_text)
                    return "break"

    def _on_history_down(self, event):
        """Обработка стрелки вниз (история)."""
        if self.history_index > 0:
            self.history_index -= 1
            if self.history_index >= 0:
                msg = self.message_history[-(self.history_index + 1)]
                if isinstance(msg, dict) and msg.get("type") == "user":
                    msg_text = msg.get("message", "")
                    if msg_text:
                        self.input_text.delete("1.0", tk.END)
                        self.input_text.insert("1.0", msg_text)
            else:
                self.input_text.delete("1.0", tk.END)
            return "break"

    def _restore_draft_text(self):
        """Восстанавливает черновик текста ввода."""
        try:
            if self._draft_text and self.input_text and self.input_text.winfo_exists():
                self.input_text.insert("1.0", self._draft_text)
                self.input_text.see(tk.END)
        except (AttributeError, TypeError, RuntimeError, tk.TclError):
            pass
