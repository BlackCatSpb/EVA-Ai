"""Copy, like, dislike, regenerate, and context menu actions for the chat module."""
import tkinter as tk
from tkinter import ttk, Menu, messagebox
import threading
import logging
from typing import Optional

logger = logging.getLogger("eva_ai.gui.chat")


class ChatActionsMixin:
    """Mixin providing copy, context menu, and action button functionality."""

    def _create_context_menu(self):
        """Создает контекстное меню."""
        self.context_menu = Menu(self.chat_display, tearoff=0)
        self.context_menu.add_command(label="Копировать", command=self._copy_selected)

        # Подменю для работы с выделением
        selection_menu = Menu(self.context_menu, tearoff=0)
        selection_menu.add_command(label="Спросить об этом",
            command=lambda: self._run_command_on_selection('ask'))
        selection_menu.add_command(label="Объяснить",
            command=lambda: self._run_command_on_selection('explain'))
        selection_menu.add_command(label="Оспорить",
            command=lambda: self._run_command_on_selection('challenge'))
        selection_menu.add_separator()
        selection_menu.add_command(label="Добавить в граф знаний",
            command=lambda: self._run_command_on_selection('add_to_graph'))
        selection_menu.add_separator()
        selection_menu.add_command(label="Использовать как контекст (Ctrl+Q)",
            command=self._on_use_selection_as_context)

        self.context_menu.add_cascade(label="По выделению...", menu=selection_menu)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Копировать все", command=self._copy_all)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Очистить чат", command=self._clear_chat)

    def _create_response_action_bar(self):
        """Создает панель действий с ответом (копировать, полезно, неверно, перегенерировать)."""
        try:
            self.action_bar = ttk.Frame(self.chat_frame)
            
            btn_style = 'action.TButton'
            style = ttk.Style()
            style.configure(btn_style, font=('Segoe UI', 9))
            
            # Кнопка копирования
            self.btn_copy = ttk.Button(
                self.action_bar,
                text="📋 Копировать",
                style=btn_style,
                command=self._action_copy_response,
                width=12
            )
            self.btn_copy.pack(side=tk.LEFT, padx=2)
            
            # Кнопка полезно
            self.btn_useful = ttk.Button(
                self.action_bar,
                text="👍 Полезно",
                style=btn_style,
                command=self._action_mark_useful,
                width=12
            )
            self.btn_useful.pack(side=tk.LEFT, padx=2)
            
            # Кнопка неверно
            self.btn_incorrect = ttk.Button(
                self.action_bar,
                text="👎 Неверно",
                style=btn_style,
                command=self._action_mark_incorrect,
                width=12
            )
            self.btn_incorrect.pack(side=tk.LEFT, padx=2)
            
            # Кнопка перегенерировать
            self.btn_regenerate = ttk.Button(
                self.action_bar,
                text="🔄 Перегенерировать",
                style=btn_style,
                command=self._action_regenerate,
                width=15
            )
            self.btn_regenerate.pack(side=tk.LEFT, padx=2)
            
            # Скрываем панель по умолчанию
            self.action_bar.pack_forget()
            
            logger.debug("Response action bar created")
        except Exception as e:
            logger.debug(f"Error creating response action bar: {e}")
            self.action_bar = None

    def _show_response_actions(self):
        """Показывает панель действий с ответом."""
        try:
            if self.action_bar and self.chat_frame.winfo_exists():
                if not self.action_bar.winfo_ismapped():
                    self.action_bar.pack(fill=tk.X, padx=5, pady=(0, 5), before=self._get_input_frame())
        except Exception as e:
            logger.debug(f"Error showing response actions: {e}")

    def _hide_response_actions(self):
        """Скрывает панель действий с ответом."""
        try:
            if self.action_bar and self.action_bar.winfo_ismapped():
                self.action_bar.pack_forget()
        except Exception as e:
            logger.debug(f"Error hiding response actions: {e}")

    def _get_input_frame(self):
        """Возвращает фрейм ввода."""
        return getattr(self, 'input_frame', None)

    def _action_copy_response(self):
        """Копирует последний ответ EVA в буфер обмена."""
        try:
            if self.message_history:
                for msg in reversed(self.message_history):
                    if msg.get('type') == 'system' and msg.get('sender') == 'ЕВА':
                        text = msg.get('message', '')
                        self.gui.root.clipboard_clear()
                        self.gui.root.clipboard_append(text)
                        self.gui.show_toast("Ответ скопирован", "info")
                        return
        except Exception as e:
            logger.error(f"Error copying response: {e}")

    def _action_mark_useful(self):
        """Отмечает ответ как полезный."""
        try:
            self.gui.show_toast("Спасибо за обратную связь! Ответ отмечен как полезный.", "info")
            logger.debug("Response marked as useful")
        except Exception as e:
            logger.debug(f"Error marking response as useful: {e}")

    def _action_mark_incorrect(self):
        """Отмечает ответ как неверный."""
        try:
            self.gui.show_toast("Спасибо за обратную связь! Мы учтём это.", "info")
            logger.debug("Response marked as incorrect")
        except Exception as e:
            logger.debug(f"Error marking response as incorrect: {e}")

    def _action_regenerate(self):
        """Перегенерирует последний ответ."""
        try:
            # Находим последний запрос пользователя
            last_user_query = None
            for msg in reversed(self.message_history):
                if msg.get('type') == 'user':
                    last_user_query = msg.get('message', '')
                    break
            
            if last_user_query:
                # Скрываем панель действий
                self._hide_response_actions()
                
                # Удаляем последний ответ EVA
                self._remove_last_message()
                
                # Вставляем запрос обратно в поле ввода
                if hasattr(self, 'input_text'):
                    self.input_text.delete("1.0", tk.END)
                    self.input_text.insert("1.0", last_user_query)
                    self.input_text.focus_set()
                
                # Отправляем запрос повторно
                self._send_message()
            else:
                self.gui.show_toast("Нечего перегенерировать", "warning")
        except Exception as e:
            logger.error(f"Error regenerating response: {e}")

    def _show_context_menu(self, event):
        """Показывает контекстное меню."""
        try:
            if not self.chat_display.winfo_exists():
                return

            has_selection = False
            try:
                _ = self.chat_display.selection_get()
                has_selection = True
            except tk.TclError:
                pass

            menu = Menu(self.chat_display, tearoff=0)

            if has_selection:
                menu.add_command(label="Копировать", command=self._copy_selected)
                menu.add_command(label="Цитировать в ввод", command=self._quote_selection_to_input)
                menu.add_separator()
                menu.add_command(label="Спросить по цитате",
                    command=lambda: self._run_command_on_selection('ask'))
                menu.add_command(label="Оспорить цитату",
                    command=lambda: self._run_command_on_selection('challenge'))
                menu.add_command(label="Объяснить цитату",
                    command=lambda: self._run_command_on_selection('explain'))
                menu.add_command(label="Добавить в граф знаний",
                    command=lambda: self._run_command_on_selection('add_to_graph'))
                menu.add_separator()

            menu.add_command(label="Копировать все", command=self._copy_all)
            menu.add_separator()
            menu.add_command(label="Очистить чат", command=self._clear_chat)

            menu.tk_popup(event.x_root, event.y_root)

        finally:
            try:
                menu.grab_release()
            except Exception as e:
                logger.debug(f"Error releasing menu grab: {e}")

    def _get_selected_chat_text(self) -> Optional[str]:
        """Возвращает выделенный текст из чата."""
        try:
            if not self.chat_display.winfo_exists():
                return None
            return self.chat_display.selection_get()
        except (tk.TclError, Exception):
            return None

    def _quote_selection_to_input(self):
        """Вставляет выделение как цитату."""
        try:
            text = self._get_selected_chat_text()
            if not text:
                return

            quoted = "> " + "\n> ".join(text.strip().splitlines()) + "\n"
            self.input_text.insert(tk.INSERT, quoted)
            self.input_text.focus_set()
        except Exception as e:
            logger.error(f"Ошибка цитирования выделения: {e}", exc_info=True)

    def _run_command_on_selection(self, cmd: str):
        """Выполняет команду над выделением."""
        try:
            text = self._get_selected_chat_text()
            if not text:
                return

            text = text.strip()

            if cmd == 'add_to_graph':
                concept = text if len(text) <= 256 else text[:256]
                try:
                    threading.Thread(
                        target=self._invoke_knowledge_integration,
                        args=(concept,),
                        name="KnowledgeIntegrateSel",
                        daemon=True).start()
                    self._add_message("ЕВА",
                        f"Добавляю в граф знаний: \"{concept}\"", "system")
                except Exception as e:
                    logger.debug(f"Error adding to knowledge graph: {e}")
                    self._add_message("ЕВА",
                        "Не удалось запустить интеграцию знаний для выделенного текста.", "system")
                return

            # Подготовка промпта
            if cmd == 'ask':
                # Сохраняем выделенный текст как контекст
                self._selection_context = text
                self._update_context_indicator()
                
                # Вставляем подсказку для пользователя
                self.input_text.insert("1.0", f"📎 Контекст: \"{text[:50]}{'...' if len(text) > 50 else ''}\"\n\n")
                self.input_text.focus_set()
                self.gui.show_toast("Контекст добавлен. Введите ваш вопрос.", "info")
                return  # Не отправляем сразу
            elif cmd == 'challenge':
                prompt = f"Оспорь утверждение из цитаты и укажи возможные контраргументы:\n\"{text}\""
            elif cmd == 'explain':
                prompt = f"Объясни смысл следующей цитаты понятным языком:\n\"{text}\""
            else:
                prompt = text

            self.input_text.insert("1.0", prompt + "\n")
            self._send_message()

        except Exception as e:
            logger.error(f"Ошибка выполнения команды по выделению ({cmd}): {e}", exc_info=True)

    def _copy_selected(self):
        """Копирует выделенный текст."""
        try:
            if not self.chat_display.winfo_exists():
                return
            selected_text = self.chat_display.selection_get()
            self.gui.root.clipboard_clear()
            self.gui.root.clipboard_append(selected_text)
        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка копирования выделенного текста: {e}", exc_info=True)

    def _copy_all(self):
        """Копирует весь текст чата."""
        try:
            if not self.chat_display.winfo_exists():
                return
            self.chat_display.config(state=tk.NORMAL)
            all_text = self.chat_display.get("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.gui.root.clipboard_clear()
            self.gui.root.clipboard_append(all_text)
        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка копирования всего текста: {e}", exc_info=True)

    def _clear_chat(self):
        """Очищает чат."""
        if not messagebox.askyesno("Очистка чата", "Вы действительно хотите очистить чат?"):
            return

        try:
            if not self.chat_display.winfo_exists():
                return

            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)

            with self._history_lock:
                self.message_history = []

            self._show_welcome_message()

        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка очистки чата: {e}", exc_info=True)

    def _cut_text(self, widget):
        """Вырезает выделенный текст."""
        try:
            if not widget.winfo_exists():
                return
            selected_text = widget.selection_get()
            widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            self.gui.root.clipboard_clear()
            self.gui.root.clipboard_append(selected_text)
        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка вырезания текста: {e}", exc_info=True)

    def _copy_text(self, widget):
        """Копирует выделенный текст."""
        try:
            if not widget.winfo_exists():
                return
            selected_text = widget.selection_get()
            self.gui.root.clipboard_clear()
            self.gui.root.clipboard_append(selected_text)
        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка копирования текста: {e}", exc_info=True)

    def _paste_text(self, widget):
        """Вставляет текст из буфера."""
        try:
            if not widget.winfo_exists():
                return
            clipboard_text = self.gui.root.clipboard_get()
            widget.insert(tk.INSERT, clipboard_text)
        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка вставки текста: {e}", exc_info=True)

    def _on_chat_copy(self, event):
        """Обработка Ctrl+C в чате"""
        try:
            self._copy_selected()
            return "break"
        except Exception as e:
            logger.debug(f"Error in _on_chat_copy: {e}")
            return None

    def _on_chat_select_all(self, event):
        try:
            self.chat_display.tag_add(tk.SEL, "1.0", tk.END)
            self.chat_display.mark_set(tk.INSERT, "1.0")
            self.chat_display.see(tk.INSERT)
            return "break"
        except Exception as e:
            logger.debug(f"Error in _on_chat_select_all: {e}")
            return None

    def _on_copy_shortcut(self, event):
        """Обработчик Ctrl+C."""
        try:
            widget = event.widget
            if not widget.winfo_exists():
                return "break"
            # Проверяем есть ли выделение
            try:
                selected = widget.selection_get()
                if selected:
                    self.gui.root.clipboard_clear()
                    self.gui.root.clipboard_append(selected)
            except tk.TclError:
                pass  # Нет выделения - ничего не делаем
            return "break"
        except Exception as e:
            logger.debug(f"Copy error: {e}")
            return "break"

    def _on_paste_shortcut(self, event):
        """Обработчик Ctrl+V."""
        try:
            widget = event.widget
            if not widget.winfo_exists():
                return "break"
            # Получаем текст из буфера обмена
            try:
                clipboard = self.gui.root.clipboard_get()
                if clipboard:
                    # Вставляем в позицию курсора
                    widget.insert(tk.INSERT, clipboard)
            except tk.TclError:
                pass  # Буфер пустой
            return "break"
        except Exception as e:
            logger.debug(f"Paste error: {e}")
            return "break"

    def _on_cut_shortcut(self, event):
        """Обработчик Ctrl+X."""
        try:
            widget = event.widget
            if not widget.winfo_exists():
                return "break"
            try:
                selected = widget.selection_get()
                if selected:
                    # Копируем в буфер
                    self.gui.root.clipboard_clear()
                    self.gui.root.clipboard_append(selected)
                    # Удаляем выделенный текст
                    widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                pass  # Нет выделения
            return "break"
        except Exception as e:
            logger.debug(f"Cut error: {e}")
            return "break"

    def _on_select_all_shortcut(self, event):
        try:
            self.input_text.tag_add(tk.SEL, "1.0", tk.END)
            self.input_text.mark_set(tk.INSERT, "1.0")
            self.input_text.see(tk.INSERT)
            return "break"
        except (AttributeError, tk.TclError):
            return None

    def _on_use_selection_as_context(self, event):
        """Обработка Ctrl+Enter или Ctrl+Q - использовать выделенный текст как контекст"""
        try:
            selected = None
            # Получаем выделенный текст
            try:
                if self.chat_display.tag_ranges(tk.SEL):
                    selected = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                # Нет выделения - пробуем получить текущую строку
                pass

            if not selected:
                try:
                    current_pos = self.chat_display.index(tk.INSERT)
                    line_start = f"{current_pos.split('.')[0]}.0"
                    line_end = f"{current_pos.split('.')[0]}.end"
                    selected = self.chat_display.get(line_start, line_end)
                except Exception as e:
                    logger.debug(f"Error getting current line: {e}")
                    return None

            if selected and selected.strip():
                # Формируем запрос с контекстом
                context_prompt = f"Относительно этого текста: \"{selected.strip()}\"\n\n"
                # Вставляем в поле ввода
                self.input_text.insert("1.0", context_prompt)
                self.input_text.focus_set()
                # Прокручиваем курсор в конец
                self.input_text.mark_set(tk.INSERT, "1.0")
                self.input_text.see(tk.INSERT)

                # Показываем подсказку
                self.gui.gui_queue.put(lambda: self.gui.show_toast("Контекст добавлен. Введите ваш вопрос.", "info"))

            return "break"
        except Exception as e:
            logger.error(f"Ошибка использования выделения как контекста: {e}")
            return None

    def _on_help(self, event=None):
        """Обработчик F1."""
        import os
        import webbrowser
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            docs_md = os.path.join(base_dir, "docs", "api_reference.md")
            readme_md = os.path.join(base_dir, "README.md")

            opened = False
            if os.path.exists(docs_md):
                webbrowser.open_new_tab(f"file://{docs_md}")
                opened = True
            elif os.path.exists(readme_md):
                webbrowser.open_new_tab(f"file://{readme_md}")
                opened = True

            if not opened:
                help_text = (
                    "Горячие клавиши:\n"
                    "• Enter — отправить\n"
                    "• Shift+Enter / Ctrl+Enter — новая строка\n"
                    "• Ctrl+C / Ctrl+V / Ctrl+X — копировать/вставить/вырезать\n"
                    "• Ctrl+A — выделить все\n"
                    "Команды:\n"
                    "• Импорт документов: кнопка 'Импорт' или меню\n"
                    "• Подсветка ссылок и изображений поддерживается\n"
                )
                self._add_message("Справка", help_text, "system")

        except Exception as e:
            logger.debug(f"Error showing help: {e}")
            try:
                messagebox.showinfo("Справка",
                    "Нажимайте Enter для отправки, Shift+Enter — новая строка.")
            except Exception as e2:
                logger.debug(f"Error showing fallback help dialog: {e2}")
        finally:
            try:
                if self.input_text and self.input_text.winfo_exists():
                    self.input_text.focus_set()
            except Exception as e3:
                logger.debug(f"Error setting focus: {e3}")
                pass

    def _on_text_selection_changed(self, event=None):
        """Обработчик изменения выделения текста - показывает popup с быстрыми действиями."""
        try:
            # Проверяем есть ли выделение в чате
            selection = None
            try:
                selection = self.chat_display.selection_get()
            except (tk.TclError, Exception):
                selection = None
            
            if selection and selection.strip() and len(selection.strip()) > 3:
                # Показываем popup с быстрыми действиями
                self._show_selection_popup()
            else:
                self._hide_selection_popup()
                self._hide_response_actions()
        except Exception as e:
            logger.debug(f"Error handling text selection: {e}")

    def _show_selection_popup(self):
        """Показывает popup с быстрыми действиями при выделении текста."""
        try:
            if not hasattr(self, '_selection_popup') or self._selection_popup is None:
                self._selection_popup = Menu(self.chat_display, tearoff=0)
                self._selection_popup.add_command(
                    label="💬 Задать вопрос по выделенному",
                    command=lambda: self._run_command_on_selection('ask')
                )
                self._selection_popup.add_command(
                    label="📋 Копировать",
                    command=self._copy_selected
                )
                self._selection_popup.add_separator()
                self._selection_popup.add_command(
                    label="📖 Объяснить",
                    command=lambda: self._run_command_on_selection('explain')
                )
                self._selection_popup.add_command(
                    label="❓ Оспорить",
                    command=lambda: self._run_command_on_selection('challenge')
                )
            
            # Показываем popup в позиции курсора
            if self.chat_display.winfo_exists():
                try:
                    x = self.chat_display.winfo_rootx() + 50
                    y = self.chat_display.winfo_rooty() + self.chat_display.winfo_height() // 2
                    self._selection_popup.tk_popup(x, y, 0)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Error showing selection popup: {e}")

    def _hide_selection_popup(self):
        """Скрывает popup выделения."""
        try:
            if hasattr(self, '_selection_popup') and self._selection_popup:
                try:
                    self._selection_popup.grab_release()
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Error hiding selection popup: {e}")

    def _update_context_indicator(self):
        """Обновляет индикатор контекста в поле ввода."""
        try:
            if not hasattr(self, '_context_indicator') or self._context_indicator is None:
                return
            
            if self._selection_context:
                context_text = self._selection_context[:100]
                if len(self._selection_context) > 100:
                    context_text += "..."
                self._context_indicator.config(
                    text=f"📎 Контекст: {context_text}",
                    foreground=self.gui.colors.get('primary', '#4A90D9')
                )
                self._context_indicator.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 0))
            else:
                self._context_indicator.pack_forget()
        except Exception as e:
            logger.debug(f"Error updating context indicator: {e}")

    def _clear_selection_context(self):
        """Очищает контекст из выделения."""
        self._selection_context = None
        self._update_context_indicator()
        self._hide_selection_popup()

    def _get_context_for_prompt(self) -> str:
        """Возвращает контекст из выделения для включения в промпт."""
        if self._selection_context:
            return f"Контекст для ответа:\n\"{self._selection_context}\"\n\n"
        return ""
