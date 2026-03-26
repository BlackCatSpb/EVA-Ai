"""Модуль чата для CogniFlex GUI - полнофункциональная реализация"""
import tkinter as tk
from tkinter import ttk, scrolledtext, Menu, messagebox, font, filedialog
import logging
import webbrowser
import re
import time
import random
import json
import os
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable

try:
    from cogniflex.tools.import_pipeline import ImportPipeline
except ImportError:
    ImportPipeline = None

logger = logging.getLogger("cogniflex.gui.chat")


# ============================================================================
# Утилиты обработки текста
# ============================================================================

def _to_display_str(val: Any) -> str:
    """Безопасно преобразует значение к строке для отображения в UTF-8."""
    try:
        if isinstance(val, bytes):
            s = val.decode("utf-8", errors="replace")
        else:
            s = str(val)
        return s.replace("\r\n", "\n").replace("\r", "\n")
    except (UnicodeError, TypeError, ValueError, AttributeError) as e:
        logger.debug(f"Error converting value to display string: {e}")
        return str(val)


def _looks_mojibake(s: str) -> bool:
    """Грубая эвристика для детекции mojibake."""
    if not s:
        return False
    bad_chars = set("ÐÑÂÃĤĭĮıİıĝĞġĠ")
    return any(ch in bad_chars for ch in s)


def _fix_mojibake(s: str) -> str:
    """Пытается исправить типичный mojibake (UTF-8, показанный как Latin-1)."""
    try:
        s0 = _to_display_str(s)
        if _looks_mojibake(s0):
            try:
                repaired = s0.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
                if repaired and repaired != s0:
                    return repaired
            except (UnicodeError, TypeError, ValueError, AttributeError):
                pass
        return s0
    except (UnicodeError, TypeError, ValueError, AttributeError):
        return _to_display_str(s)


# ============================================================================
# Основной класс модуля чата
# ============================================================================

class ChatModule:
    """Модуль чата для взаимодействия с пользователем."""
    
    def __init__(self, gui):
        self.gui = gui
        self.chat_frame = None
        self.chat_display = None
        
        # Панель рассуждений (коллапсируемая)
        self.reasoning_frame = None
        self.reasoning_header = None
        self.reasoning_toggle_btn = None
        self.reasoning_text = None
        self.reasoning_visible = False
        
        # Область ввода
        self.input_frame = None
        self.input_text = None
        self.send_button = None
        self.import_button = None
        
        # Контекстное меню
        self.context_menu = None
        
        # История и очередь запросов
        self.message_history = []
        self.history_index = -1
        self.pending_requests = set()
        self.request_queue = queue.Queue()
        self.processing_thread = None
        self.stop_event = threading.Event()
        self._status_updater_id = None
        self.typing_text = "Печатает..."
        self.formatting_pattern = re.compile(r'\*\*(.*?)\*\*|__(.*?)__|_(.*?)_|`(.*?)`')
        self.markdown_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        self.url_pattern = re.compile(r'https?://\S+')
        self.emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
        self.image_pattern = re.compile(r'\.(?:jpg|jpeg|png|gif|bmp|webp)', re.IGNORECASE)
        
        # Инициализация флага рассуждений
        if not hasattr(self.gui, 'reasoning_active'):
            self.gui.reasoning_active = True
        
        # Инициализация состояний
        self._suppress_history_append = False
        self._ml_ready_cached = False
        self._import_pipeline = None  # Lazy initialized import pipeline
            
        logger.info("Модуль чата инициализирован")
    
    # =========================================================================
    # Управление жизненным циклом
    # =========================================================================
    
    def activate(self):
        """Активирует модуль чата."""
        # Очищаем область контента
        for widget in self.gui.content_area.winfo_children():
            widget.destroy()
        
        # Создаем интерфейс
        self._create_chat_interface()
        
        # Запускаем фоновую обработку
        self._start_processing_thread()
        
        # Подписки на события
        self._setup_event_subscriptions()
        
        logger.info("Модуль чата активирован")
    
    def deactivate(self):
        """Деактивирует модуль чата."""
        # Останавливаем фоновые процессы
        self._stop_processing_thread()
        
        # Отменяем обновление статуса
        self._cancel_status_update()
        
        # Сохраняем черновик
        self._save_draft_text()
        
        # Сохраняем историю
        self._save_chat_history()
        
        logger.info("Модуль чата деактивирован")
    
    def _setup_event_subscriptions(self):
        """Настраивает подписки на события загрузки моделей."""
        try:
            brain = getattr(self.gui, 'brain', None)
            if not brain:
                return
            
            handler = self._handle_model_load_event_chat
            
            # Через events
            if hasattr(brain, 'events') and brain.events:
                try:
                    brain.events.on('model_load', 
                        lambda data: self.gui.gui_queue.put(lambda: handler(data)))
                    brain.events.on('models_ready', 
                        lambda data=None: self.gui.gui_queue.put(self._set_ml_ready))
                except (AttributeError, TypeError, RuntimeError):
                    pass
            
            # Фолбэк через списки
            if not hasattr(brain, 'on_model_load'):
                setattr(brain, 'on_model_load', [])
            brain.on_model_load.append(
                lambda data: self.gui.gui_queue.put(lambda: handler(data)))
            
            if not hasattr(brain, 'on_models_ready'):
                setattr(brain, 'on_models_ready', [])
            brain.on_models_ready.append(
                lambda data=None: self.gui.gui_queue.put(self._set_ml_ready))
            
            # Инициализация состояния
            ready = bool(getattr(brain, 'models_ready', False))
            self._ml_ready_cached = ready
            self._apply_ml_ready_state(ready)
            
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Error setting up event subscriptions: {e}")
    
    def _start_processing_thread(self):
        """Запускает фоновый поток для обработки запросов."""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="ChatProcessing",
            daemon=True
        )
        self.processing_thread.start()
        logger.debug("Фоновый поток обработки запросов чата запущен")
    
    def _stop_processing_thread(self):
        """Останавливает фоновый поток обработки запросов."""
        if not self.processing_thread:
            return
        
        self.stop_event.set()
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        logger.debug("Фоновый поток обработки запросов чата остановлен")
    
    def _cancel_status_update(self):
        """Отменяет периодическое обновление статуса."""
        try:
            if self._status_updater_id is not None and hasattr(self.gui, 'root'):
                self.gui.root.after_cancel(self._status_updater_id)
        except (AttributeError, TypeError, RuntimeError):
            pass
        finally:
            self._status_updater_id = None
    
    def _save_draft_text(self):
        """Сохраняет черновик текста ввода."""
        try:
            if self.input_text and self.input_text.winfo_exists():
                self._draft_text = self.input_text.get("1.0", tk.END)
        except (AttributeError, TypeError, RuntimeError):
            pass
    
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
    
    # =========================================================================
    # Обработка запросов
    # =========================================================================
    
    def _processing_loop(self):
        """Цикл обработки запросов в фоновом потоке."""
        while not self.stop_event.is_set():
            try:
                request = self.request_queue.get(timeout=0.5)
                self._process_request(request)
                self.request_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка в цикле обработки запросов чата: {e}", exc_info=True)
    
    def _process_request(self, request: Dict[str, Any]):
        """Обрабатывает запрос в фоновом потоке."""
        user_message = request.get("message", "")
        request_id = request.get("request_id")
        start_time = request.get("start_time", time.time())
        
        try:
            # Проверка отмены
            if request_id not in self.pending_requests:
                return
            
            # Индикатор ожидания
            self.gui.gui_queue.put(self._show_typing)
            self.gui.gui_queue.put(
                lambda: self._add_message("CogniFlex", "Обрабатываю запрос...", "system"))
            
            # Проверка готовности системы
            brain = getattr(self.gui, 'brain', None)
            if not brain:
                self.gui.gui_queue.put(lambda: self._add_message(
                    "CogniFlex", 
                    "Ядро системы недоступно. Проверьте, что все компоненты системы запущены.",
                    "system"))
                return
            
            has_fractal = hasattr(brain, 'fractal_ready') and brain.fractal_ready
            has_memory = hasattr(brain, 'memory_manager') and brain.memory_manager
            
            if not has_fractal and not has_memory:
                self.gui.gui_queue.put(lambda: self._add_message(
                    "CogniFlex",
                    "Фрактальное хранилище и граф памяти недоступны. Чат не может работать.",
                    "system"))
                return
            
            if not hasattr(brain, 'process_query'):
                self.gui.gui_queue.put(lambda: self._add_message(
                    "CogniFlex",
                    "Метод обработки запросов недоступен в ядре системы.",
                    "system"))
                return
            
            # Выполнение запроса
            try:
                timeout_sec = getattr(self.gui, 'chat_timeout_sec', 30.0)
                response_obj, processing_time = self._generate_response_obj(
                    user_message, timeout_sec=timeout_sec)
            except Exception as e:
                logger.error(f"Ошибка при обработке запроса через ядро: {e}", exc_info=True)
                response_obj, processing_time = (
                    {"text": f"Ошибка обработки запроса: {str(e)}"}, 
                    time.time() - start_time)
            
            # Проверка отмены после выполнения
            if request_id not in self.pending_requests:
                return
            
            # Удаление индикатора
            self.gui.gui_queue.put(self._remove_last_message)
            self.gui.gui_queue.put(self._hide_typing)
            
            # Извлечение полей ответа
            text, tokens, sentiment, reasoning, contradictions, contradiction_flag, metadata = \
                self._extract_response_fields(response_obj, fallback_input=user_message)
            
            # Форматирование ответа
            display_text = _fix_mojibake(text).strip() if text and str(text).strip() \
                else "Извините, система не смогла сформировать ответ."
            
            # Аналитика
            analytics_lines = self._build_analytics_lines(tokens, sentiment, contradictions, processing_time)
            
            # Проверка отмены перед обновлением UI
            if request_id not in self.pending_requests:
                return
            
            # Обновление панели рассуждений
            self.gui.gui_queue.put(lambda r=reasoning: self._set_reasoning_content(r, auto_expand=True))
            
            # Формирование финального сообщения
            final_message = display_text + "\n" + "\n".join(analytics_lines) if analytics_lines else display_text
            extras = {
                "processing_time": processing_time,
                "tokens": tokens,
                "sentiment": sentiment,
                "contradictions": contradictions,
                "metadata": metadata,
                "reasoning": reasoning,
            }
            
            # Проверка отмены перед добавлением
            if request_id not in self.pending_requests:
                return
            
            self.gui.gui_queue.put(
                lambda m=final_message, ex=extras: self._add_message("CogniFlex", m, "system", extras=ex))
            
            # Уведомление о противоречиях
            if contradiction_flag:
                num = len(contradictions) if contradictions else 1
                self.gui.gui_queue.put(lambda: self.gui.show_notification(
                    f"Обнаружено {num} противоречение(ий) в знаниях",
                    "warning",
                    actions=[{"text": "Посмотреть", 
                             "command": lambda: self.gui._switch_view("contradictions")}]
                ))
            
            # Интеграция знаний в фоне
            concept_for_kg = self._extract_concept_for_integration(metadata, tokens, user_message)
            if request_id in self.pending_requests and concept_for_kg:
                threading.Thread(
                    target=self._invoke_knowledge_integration, 
                    args=(concept_for_kg,), 
                    name="KnowledgeIntegrate", 
                    daemon=True).start()
                    
        except Exception as e:
            logger.error(f"Ошибка обработки ответа в _process_request: {e}", exc_info=True)
            error_msg = (
                "Произошла ошибка при обработке запроса:\n"
                f"{str(e)}\n"
                "Попробуйте повторить запрос или обратиться к системному администратору."
            )
            self.gui.gui_queue.put(lambda: self._add_message("CogniFlex", error_msg, "system"))
        finally:
            # Очистка pending requests
            if request_id in self.pending_requests:
                try:
                    self.pending_requests.remove(request_id)
                except (KeyError, ValueError, AttributeError):
                    logger.debug("Не удалось удалить request_id из pending_requests", exc_info=True)
            
            # Скрытие индикатора
            try:
                self.gui.gui_queue.put(self._hide_typing)
            except (AttributeError, RuntimeError, TypeError):
                pass
    
    def _build_analytics_lines(self, tokens, sentiment, contradictions, processing_time):
        """Формирует строки аналитики для отображения."""
        analytics_lines = []
        
        if tokens:
            if isinstance(tokens[0], (list, tuple)) and len(tokens[0]) >= 1:
                keywords_display = ", ".join([_fix_mojibake(k[0]) for k in tokens[:5]])
            else:
                keywords_display = ", ".join([_fix_mojibake(k) for k in tokens[:5]])
            if keywords_display.strip():
                analytics_lines.append(f"Ключевые слова: {keywords_display}")
        
        if sentiment:
            if isinstance(sentiment, dict):
                sentiment_display = sentiment.get("compound") if "compound" in sentiment \
                    else sentiment.get("label", "")
            else:
                sentiment_display = _fix_mojibake(sentiment)
            if str(sentiment_display).strip():
                analytics_lines.append(f"Тональность: {sentiment_display}")
        
        if contradictions:
            short_contra = self._summarize_contradictions(contradictions)
            if short_contra:
                analytics_lines.append(f"Противоречия: {short_contra}")
        
        analytics_lines.append(f"Время обработки: {processing_time:.2f} сек")
        
        return analytics_lines
    
    # =========================================================================
    # Создание интерфейса
    # =========================================================================
    
    def _create_chat_interface(self):
        """Создает интерфейс чата."""
        self.chat_frame = ttk.Frame(self.gui.content_area)
        self.chat_frame.pack(fill=tk.BOTH, expand=True)
        
        # Область отображения сообщений
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg=self.gui.colors['card-bg'],
            fg=self.gui.colors['text'],
            font=('Segoe UI', 10),
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Панель рассуждений
        self._init_reasoning_panel()
        
        # Настройка стилей
        self._configure_chat_tags()
        
        # Обработчики событий
        self._setup_chat_event_handlers()
        
        # Контекстное меню
        self._create_context_menu()
        
        # Область ввода
        self._create_input_area()
        
        # Статус-бар
        self._create_status_bar()
        
        # Индикатор набора
        self._create_typing_indicator()
        
        # Загрузка истории
        self._load_chat_history()
        
        # Приветствие
        if not self.message_history:
            self._show_welcome_message()
        
        # Восстановление черновика
        self._restore_draft_text()
        
        # Запуск обновления статуса
        self._schedule_status_update()
        
        # Применение состояния готовности
        self._apply_initial_ml_state()
    
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
    
    def _setup_chat_event_handlers(self):
        """Настраивает обработчики событий для области чата."""
        self.chat_display.tag_bind("url", "<Enter>", self._handle_url_enter)
        self.chat_display.tag_bind("url", "<Leave>", self._handle_url_leave)
        self.chat_display.bind("<Button-1>", self._handle_url_click)
        self.chat_display.bind("<Button-3>", self._show_context_menu)
        
        # Горячие клавиши - исправляем проблему с копированием
        self.chat_display.bind("<Control-Key-c>", self._on_chat_copy)
        self.chat_display.bind("<Control-Key-C>", self._on_chat_copy)
        self.chat_display.bind("<Control-Key-a>", self._on_chat_select_all)
        self.chat_display.bind("<Control-Key-A>", self._on_chat_select_all)
        
        # Ctrl+Q - использовать выделенный текст как контекст
        self.chat_display.bind("<Control-Key-q>", self._on_use_selection_as_context)
        self.chat_display.bind("<Control-Key-Q>", self._on_use_selection_as_context)
        
        # Ctrl+Enter - использовать выделенный текст как контекст (как в GPT)
        self.chat_display.bind("<Control-Key-Return>", self._on_use_selection_as_context)
        self.chat_display.bind("<Control-Key-KP_Enter>", self._on_use_selection_as_context)
        
        # Ctrl+Shift+Q - уточнить по выделению
        self.chat_display.bind("<Control-Shift-Key-q>", lambda e: (self._run_command_on_selection('ask'), "break"))
        self.chat_display.bind("<Control-Shift-Key-Q>", lambda e: (self._run_command_on_selection('ask'), "break"))
        self.chat_display.bind("<Control-Shift-Key-G>", 
            lambda e: (self._run_command_on_selection('add_to_graph'), "break"))
    
    def _on_chat_copy(self, event):
        """Обработка Ctrl+C в чате"""
        try:
            self._copy_selected()
            return "break"
        except Exception as e:
            logger.debug(f"Error in _on_chat_copy: {e}")
            return None
    
    def _on_use_selection_as_context(self, event):
        """Обработка Ctrl+Enter или Ctrl+Q - использовать выделенный текст как контекст"""
        try:
            # Получаем выделенный текст
            try:
                selected = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                # Нет выделения - пробуем получить текущую строку
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
                self.gui.gui_queue.put(lambda: self._show_toast("Контекст добавлен. Введите ваш вопрос.", "info"))
            
            return "break"
        except Exception as e:
            logger.error(f"Ошибка использования выделения как контекста: {e}")
            return None
    
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
    
    def _create_input_area(self):
        """Создает область ввода сообщения."""
        self.input_frame = ttk.Frame(self.chat_frame)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.input_text = scrolledtext.ScrolledText(
            self.input_frame,
            height=3,
            wrap=tk.WORD,
            bg=self.gui.colors['card-bg'],
            fg=self.gui.colors['text'],
            font=('Segoe UI', 10),
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
        # Кнопка импорта
        self.import_button = ttk.Button(
            self.input_frame,
            text="Импорт",
            command=self._on_import_document
        )
        self.import_button.pack(side=tk.RIGHT, padx=(5, 0))
        
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
        
        # Горячие клавиши - используем стандартное поведение + свои обработчики
        # Явно добавляем Ctrl+C, Ctrl+V, Ctrl+X для надежности
        self.input_text.bind("<Control-c>", self._on_copy_shortcut)
        self.input_text.bind("<Control-C>", self._on_copy_shortcut)
        self.input_text.bind("<Control-v>", self._on_paste_shortcut)
        self.input_text.bind("<Control-V>", self._on_paste_shortcut)
        self.input_text.bind("<Control-x>", self._on_cut_shortcut)
        self.input_text.bind("<Control-X>", self._on_cut_shortcut)
        # Ctrl+A для выделения всего
        self.input_text.bind("<Control-a>", self._on_select_all_shortcut)
        self.input_text.bind("<Control-A>", self._on_select_all_shortcut)
        self.input_text.bind("<Control-q>", 
            lambda e: (self._quote_selection_to_input(), "break"))
        self.input_text.bind("<Control-Q>", 
            lambda e: (self._quote_selection_to_input(), "break"))
        # Ctrl+Shift+V для вставки без форматирования
        self.input_text.bind("<Control-Shift-V>", self._on_paste_shortcut)
        self.input_text.bind("<Control-Shift-v>", self._on_paste_shortcut)
    
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
        """Создает индикатор набора текста."""
        try:
            self.typing_indicator = ttk.Label(
                self.chat_frame, 
                text=self.typing_text, 
                foreground=self.gui.colors['text-muted'])
            self.typing_indicator.pack_forget()  # Скрыт по умолчанию
        except (AttributeError, TypeError, RuntimeError, tk.TclError):
            self.typing_indicator = None
    
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
    
    # =========================================================================
    # Обработка сообщений
    # =========================================================================
    
    def _add_message(self, sender: str, message: str, msg_type: str = "user",
                     timestamp: Optional[float] = None, 
                     process_formatting: bool = True, 
                     extras: Optional[Dict[str, Any]] = None):
        """Добавляет сообщение в чат."""
        if timestamp is None:
            timestamp = time.time()
        
        # Сохранение в историю
        if not self._suppress_history_append:
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
            
            self.message_history.append(entry)
            self._save_history_incremental()
            
            # Ограничение размера истории
            if len(self.message_history) > 500:
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
    
    # =========================================================================
    # Панель рассуждений
    # =========================================================================
    
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
    
    # =========================================================================
    # Индикаторы состояния
    # =========================================================================
    
    def _set_ml_ready(self):
        """Устанавливает состояние готовности ML."""
        self._ml_ready_cached = True
        self._apply_ml_ready_state(True)
    
    def _apply_ml_ready_state(self, ready: bool):
        """Применяет состояние готовности ML к UI."""
        try:
            brain = getattr(self.gui, 'brain', None)
            fractal_ready = bool(getattr(brain, 'fractal_ready', False)) if brain else False
            
            ml_unit_ready = False
            if brain and hasattr(brain, 'ml_unit') and brain.ml_unit is not None:
                ml_unit_ready = getattr(brain.ml_unit, 'models_ready', False)
                if not ml_unit_ready:
                    ml_unit_ready = getattr(brain.ml_unit, 'initialized', False)
                if not ml_unit_ready:
                    ml_unit_ready = getattr(brain.ml_unit, 'running', False)
            
            is_ready = ready or fractal_ready or ml_unit_ready
            
            # Проверяем Qwen API статус
            qwen_status = ""
            if brain and hasattr(brain, 'qwen_api_enhancer') and brain.qwen_api_enhancer:
                status = brain.qwen_api_enhancer.get_status()
                if status.get('enabled'):
                    source = status.get('current_source', 'unknown')
                    enhancements = status.get('total_enhancements', 0)
                    if source == 'qwen_api':
                        qwen_status = f" | 🤖 Qwen API: ✅"
                    elif source == 'wikipedia':
                        qwen_status = f" | 🌐 Wiki"
                    elif source == 'websearch':
                        qwen_status = f" | 🌐 Web"
                    else:
                        qwen_status = f" | 🤖 Локально"
            
            # Индикатор
            if self.ml_status_canvas and self.ml_status_canvas.winfo_exists():
                color = self.gui.colors['success'] if is_ready else self.gui.colors['warning']
                self.ml_status_canvas.itemconfig("ml_indicator", fill=color)
            
            # Лейбл
            if self.ml_status_label and self.ml_status_label.winfo_exists():
                if is_ready:
                    if fractal_ready:
                        self.ml_status_label.config(text=f"Фрактальное хранилище: активно{qwen_status}")
                    else:
                        self.ml_status_label.config(text=f"ML: готово{qwen_status}")
                else:
                    prog_text = self._current_ml_progress_text()
                    self.ml_status_label.config(text=f"ML: {prog_text}")
            
            # Кнопки
            if self.send_button and self.send_button.winfo_exists():
                state = tk.NORMAL if is_ready else tk.DISABLED
                self.send_button.config(state=state)
                
        except (AttributeError, TypeError, RuntimeError, tk.TclError):
            pass
    
    def _apply_initial_ml_state(self):
        """Применяет начальное состояние готовности ML."""
        try:
            brain = getattr(self.gui, 'brain', None)
            fractal_ready = bool(getattr(brain, 'fractal_ready', False)) if brain else False
            ml_unit_ready = False
            if brain and hasattr(brain, 'ml_unit') and brain.ml_unit is not None:
                ml_unit_ready = getattr(brain.ml_unit, 'models_ready', False)
                if not ml_unit_ready:
                    ml_unit_ready = getattr(brain.ml_unit, 'initialized', False)
                if not ml_unit_ready:
                    ml_unit_ready = getattr(brain.ml_unit, 'running', False)
            ready = fractal_ready or ml_unit_ready
            self._apply_ml_ready_state(ready)
        except (AttributeError, TypeError, RuntimeError):
            pass
    
    def _current_ml_progress_text(self) -> str:
        """Возвращает текст статуса модели."""
        try:
            brain = getattr(self.gui, 'brain', None)
            if brain and getattr(brain, 'fractal_ready', False):
                return "активно"
            
            st = getattr(self.gui, 'model_loading_state', None)
            if not st:
                return "недоступно"
            
            if st.get('error'):
                return "ошибка загрузки"
            
            if st.get('active'):
                action = st.get('action') or 'load'
                if action == 'unload':
                    return "выгрузка..."
                return f"загрузка {int(st.get('progress') or 0)}%"
            
            prog = int(st.get('progress') or 0)
            return f"загрузка {prog}%" if prog and prog < 100 else "недоступно"
            
        except (AttributeError, TypeError, ValueError):
            return "недоступно"
    
    def _handle_model_load_event_chat(self, data: Dict[str, Any]):
        """Обработчик событий загрузки модели."""
        try:
            if not isinstance(data, dict):
                return
            
            event = data.get('event')
            brain = getattr(self.gui, 'brain', None)
            fractal_ready = bool(getattr(brain, 'fractal_ready', False)) if brain else False
            
            if event == 'model_load_progress':
                if not fractal_ready:
                    self._apply_ml_ready_state(False)
            elif event == 'model_load_complete':
                if fractal_ready:
                    self._set_ml_ready()
                else:
                    self._apply_ml_ready_state(False)
            elif event == 'model_load_error':
                if fractal_ready:
                    self._set_ml_ready()
                else:
                    self._ml_ready_cached = False
                    self._apply_ml_ready_state(False)
            elif event == 'model_unload_start':
                if not fractal_ready:
                    self._ml_ready_cached = False
                    self._apply_ml_ready_state(False)
            elif event == 'model_unload_complete':
                ready = fractal_ready or bool(getattr(brain, 'models_ready', False)) if brain else False
                self._ml_ready_cached = ready
                self._apply_ml_ready_state(ready)
            elif event == 'model_unload_error':
                self._apply_ml_ready_state(self._ml_ready_cached or fractal_ready)
                
        except (AttributeError, TypeError, KeyError, RuntimeError):
            pass
    
    # =========================================================================
    # Статус-бар
    # =========================================================================
    
    def _schedule_status_update(self, interval_ms: int = 2000):
        """Планирует периодическое обновление статус-бара."""
        try:
            if not hasattr(self.gui, 'root') or not self.gui.root or \
               not self.chat_frame or not self.chat_frame.winfo_exists():
                return
            
            self._update_status_bar()
            self._status_updater_id = self.gui.root.after(
                interval_ms, 
                lambda: self._schedule_status_update(interval_ms))
        except Exception as e:
            logger.debug(f"Error in _schedule_status_update: {e}")
            self._status_updater_id = None
    
    def _update_status_bar(self):
        """Обновляет значения CPU/RAM в статус-баре."""
        try:
            if not self.status_frame or not self.status_frame.winfo_exists():
                return
            
            cpu_pct = "--"
            mem_pct = "--"
            
            brain = getattr(self.gui, 'brain', None)
            if brain and hasattr(brain, 'get_system_metrics'):
                metrics = brain.get_system_metrics() or {}
                
                cpu = metrics.get("cpu_usage")
                if isinstance(cpu, (int, float)):
                    cpu_pct_val = cpu * 100.0 if cpu <= 1.5 else float(cpu)
                    cpu_pct = f"{cpu_pct_val:.0f}"
                
                mem = metrics.get("memory_usage")
                if isinstance(mem, (int, float)):
                    mem_pct_val = mem * 100.0 if mem <= 1.5 else float(mem)
                    mem_pct = f"{mem_pct_val:.0f}"
            
            if self.cpu_label:
                self.cpu_label.config(text=f"CPU: {cpu_pct}%")
            if self.mem_label:
                self.mem_label.config(text=f"RAM: {mem_pct}%")
                
        except Exception as e:
            logger.debug(f"Error updating status bar: {e}")
    
    # =========================================================================
    # История чата
    # =========================================================================
    
    def _load_chat_history(self):
        """Загружает историю чата из файла."""
        try:
            if self.gui.cache_dir:
                os.makedirs(self.gui.cache_dir, exist_ok=True)
                history_file = os.path.join(self.gui.cache_dir, "chat_history.json")
                
                if os.path.exists(history_file):
                    with open(history_file, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    
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
    
    def _show_welcome_message(self):
        """Показывает приветственное сообщение."""
        welcome_msg = (
            "Добро пожаловать в CogniFlex!\n"
            "Я - когнитивная система с поддержкой:\n"
            "• Этического анализа\n"
            "• Адаптации под пользователя\n"
            "• Распределенных вычислений\n"
            "• Управления знаниями\n"
            "Задайте ваш первый вопрос или нажмите F1 для просмотра справки."
        )
        self._add_message("CogniFlex", welcome_msg, "system")
    
    def _restore_draft_text(self):
        """Восстанавливает черновик текста ввода."""
        try:
            if self._draft_text and self.input_text and self.input_text.winfo_exists():
                self.input_text.insert("1.0", self._draft_text)
                self.input_text.see(tk.END)
        except (AttributeError, TypeError, RuntimeError, tk.TclError):
            pass
    
    # =========================================================================
    # Индикатор набора
    # =========================================================================
    
    def _show_typing(self):
        """Показывает индикатор набора текста."""
        try:
            self.typing_active = True
            if self.typing_indicator and self.chat_frame and self.chat_frame.winfo_exists():
                if not self.typing_indicator.winfo_ismapped():
                    self.typing_indicator.pack(fill=tk.X, padx=5, pady=(0, 5))
        except (AttributeError, TypeError, RuntimeError, tk.TclError):
            pass
    
    def _hide_typing(self):
        """Скрывает индикатор набора текста."""
        try:
            self.typing_active = False
            if self.typing_indicator and self.typing_indicator.winfo_ismapped():
                self.typing_indicator.pack_forget()
        except (AttributeError, TypeError, RuntimeError, tk.TclError):
            pass
    
    # =========================================================================
    # Отправка сообщений
    # =========================================================================
    
    def _send_message(self):
        """Отправляет сообщение из поля ввода."""
        message = self.input_text.get("1.0", tk.END).strip()
        if not message:
            return
        
        # Проверка готовности - расширенная проверка
        try:
            brain = getattr(self.gui, 'brain', None)
            ml_ready = bool(getattr(brain, 'models_ready', False)) if brain else False
            fractal_ready = bool(getattr(brain, 'fractal_ready', False)) if brain else False
            has_ml_unit = hasattr(brain, 'ml_unit') and brain.ml_unit is not None if brain else False
            
            print(f"[DEBUG CHAT] brain={brain is not None}, models_ready={ml_ready}, fractal_ready={fractal_ready}, has_ml_unit={has_ml_unit}")
            
            # Если есть ml_unit - считаем что система может работать
            if has_ml_unit and not ml_ready:
                ml_ready = True
                fractal_ready = True
                print(f"[DEBUG CHAT] Enabled ml_ready=True because has_ml_unit={has_ml_unit}")
            
            if not ml_ready and not fractal_ready:
                info = "Модель ещё загружается. Пожалуйста, дождитесь готовности (" + \
                       self._current_ml_progress_text() + ")."
                self._add_message("CogniFlex", info, "system")
                return
        except Exception as e:
            print(f"[DEBUG CHAT] Exception in check: {e}")
        
        # Добавление сообщения
        self._add_message("Вы", message, "user")
        
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
    
    # =========================================================================
    # Горячие клавиши
    # =========================================================================
    
    def _on_copy_shortcut(self, event):
        try:
            self._copy_text(self.input_text)
            return "break"
        except (AttributeError, tk.TclError):
            return None
    
    def _on_paste_shortcut(self, event):
        try:
            # Используем стандартную вставку
            self.input_text.event_generate('<<Paste>>')
            return "break"
        except (AttributeError, tk.TclError):
            return None
    
    def _on_cut_shortcut(self, event):
        try:
            self._cut_text(self.input_text)
            return "break"
        except (AttributeError, tk.TclError):
            return None
    
    def _on_select_all_shortcut(self, event):
        try:
            self.input_text.tag_add(tk.SEL, "1.0", tk.END)
            self.input_text.mark_set(tk.INSERT, "1.0")
            self.input_text.see(tk.INSERT)
            return "break"
        except (AttributeError, tk.TclError):
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
    
    def _on_help(self, event=None):
        """Обработчик F1."""
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
    
    # =========================================================================
    # Контекстное меню и операции с текстом
    # =========================================================================
    
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
                    self._add_message("CogniFlex", 
                        f"Добавляю в граф знаний: \"{concept}\"", "system")
                except Exception as e:
                    logger.debug(f"Error adding to knowledge graph: {e}")
                    self._add_message("CogniFlex", 
                        "Не удалось запустить интеграцию знаний для выделенного текста.", "system")
                return
            
            # Подготовка промпта
            if cmd == 'ask':
                prompt = f"Вопрос по цитате:\n\"{text}\""
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
    
    # =========================================================================
    # Удаление сообщений
    # =========================================================================
    
    def _remove_last_message(self):
        """Удаляет последнее сообщение из чата."""
        try:
            if not self.chat_display.winfo_exists():
                return
            
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("end-2l", "end")
            self.chat_display.config(state=tk.DISABLED)
            
            if self.message_history:
                self.message_history.pop()
                
        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка удаления последнего сообщения: {e}", exc_info=True)
    
    # =========================================================================
    # Импорт документов
    # =========================================================================
    
    def _on_import_document(self):
        """Открывает диалог импорта документа."""
        try:
            filetypes = [
                ("Документы", "*.txt *.md *.log *.pdf *.epub"),
                ("Текстовые файлы", "*.txt *.md *.log"),
                ("PDF", "*.pdf"),
                ("EPUB", "*.epub"),
                ("Все файлы", "*.*"),
            ]
            
            filename = filedialog.askopenfilename(
                title="Выберите документ для импорта", 
                filetypes=filetypes)
            
            if not filename:
                return
            
            self._add_message("CogniFlex", 
                f"Импорт файла: {os.path.basename(filename)}", "system")
            
            threading.Thread(
                target=self._import_and_maybe_train,
                args=(filename,),
                name="ImportAndTrain",
                daemon=True,
            ).start()
            
        except Exception as e:
            logger.error(f"Ошибка в обработчике импорта: {e}", exc_info=True)
            try:
                messagebox.showerror("Импорт", f"Ошибка импорта: {str(e)}")
            except (tk.TclError, RuntimeError, AttributeError):
                pass
    
    def _import_and_maybe_train(self, path: str):
        """Импортирует документ и запускает обучение."""
        try:
            brain = getattr(self.gui, 'brain', None)
            if not brain:
                self.gui.gui_queue.put(lambda: self._add_message(
                    "CogniFlex", "Ядро недоступно — импорт невозможен.", "system"))
                return
            
            # Инициализация пайплайна
            if ImportPipeline is None:
                self.gui.gui_queue.put(lambda: self._add_message(
                    "CogniFlex", "ImportPipeline не доступен.", "system"))
                return
            
            try:
                if self._import_pipeline is None:
                    self._import_pipeline = ImportPipeline(brain=brain)
            except Exception as e:
                logger.error(f"Не удалось инициализировать ImportPipeline: {e}", exc_info=True)
                self.gui.gui_queue.put(lambda: self._add_message(
                    "CogniFlex", f"Не удалось инициализировать импорт: {str(e)}", "system"))
                return
            
            # Импорт
            imported = self._import_pipeline.import_path(path)
            segments = list(imported.iter_segments())
            seg_count = len(segments)
            
            self.gui.gui_queue.put(lambda: self._add_message(
                "CogniFlex", f"Импорт завершён: найдено сегментов — {seg_count}", "system"))
            
            # Обучение
            self._run_training(imported)
            
        except Exception as e:
            logger.error(f"Общая ошибка импорта/обучения: {e}", exc_info=True)
            self.gui.gui_queue.put(lambda: self._add_message(
                "CogniFlex", f"Ошибка импорта: {str(e)}", "system"))
    
    def _run_training(self, imported):
        """Запускает обучение на импортированных данных."""
        try:
            brain = getattr(self.gui, 'brain', None)
            tor = None
            
            ml_unit = getattr(brain, 'ml_unit', None)
            if ml_unit and hasattr(ml_unit, 'training_orchestrator'):
                tor = ml_unit.training_orchestrator
            
            if tor is None:
                from cogniflex.mlearning.training_orchestrator import TrainingOrchestrator
                tor = TrainingOrchestrator(brain=brain, progress_cb=self._training_progress_cb)
            
            result = tor.train_from_document(imported)
            status = (result or {}).get("status")
            
            if status == "completed":
                self.gui.gui_queue.put(lambda: self._add_message(
                    "CogniFlex", "Обучение по документу завершено успешно.", "system"))
            elif status == "deferred":
                reason = (result or {}).get("reason", "модели не готовы или кэш недоступен")
                self.gui.gui_queue.put(lambda r=reason: self._add_message(
                    "CogniFlex", 
                    f"Обучение отложено: {r}. Документ будет обработан позже.", 
                    "system"))
            elif status == "failed":
                err = (result or {}).get("error", "неизвестная ошибка")
                self.gui.gui_queue.put(lambda e=err: self._add_message(
                    "CogniFlex", f"Ошибка обучения: {e}", "system"))
            else:
                self.gui.gui_queue.put(lambda: self._add_message(
                    "CogniFlex", f"Статус обучения: {status or 'неизвестно'}", "system"))
                    
        except Exception as e:
            logger.error(f"Ошибка запуска обучения: {e}", exc_info=True)
            self.gui.gui_queue.put(lambda: self._add_message(
                "CogniFlex", f"Не удалось запустить обучение: {str(e)}", "system"))
    
    def _training_progress_cb(self, data: Dict[str, Any]):
        """Колбэк прогресса обучения."""
        try:
            if not isinstance(data, dict):
                return
            
            event = data.get("event")
            
            def _post(msg: str):
                try:
                    self._add_message("Обучение", msg, "system")
                except (AttributeError, RuntimeError, TypeError):
                    pass
            
            if hasattr(self.gui, 'gui_queue') and self.gui.gui_queue:
                if event == "start":
                    total = data.get("total_chunks")
                    self.gui.gui_queue.put(
                        lambda t=total: _post(f"Старт обучения документа (всего сегментов: {t})."))
                elif event == "batch_start":
                    s = data.get("start_idx")
                    e = data.get("end_idx")
                    a = data.get("attempt")
                    self.gui.gui_queue.put(
                        lambda s=s, e=e, a=a: _post(f"Обработка сегментов {s}-{e} (попытка {a})."))
                elif event == "batch_end":
                    pc = data.get("processed_chunks")
                    tt = data.get("total_chunks")
                    self.gui.gui_queue.put(
                        lambda pc=pc, tt=tt: _post(f"Готово: {pc}/{tt} сегментов."))
                elif event == "batch_retry":
                    a = data.get("attempt")
                    self.gui.gui_queue.put(
                        lambda a=a: _post(f"Повторная попытка обработки батча (попытка {a})."))
                elif event == "paused":
                    r = data.get("reason")
                    self.gui.gui_queue.put(
                        lambda r=r: _post(f"Пауза обучения (причина: {r})."))
                elif event == "completed":
                    self.gui.gui_queue.put(lambda: _post("Обучение по документу завершено."))
                elif event == "failed":
                    err = data.get("error")
                    self.gui.gui_queue.put(
                        lambda err=err: _post(f"Обучение завершилось ошибкой: {err}"))
                elif event == "deferred":
                    r = data.get("reason")
                    self.gui.gui_queue.put(
                        lambda r=r: _post(f"Обучение отложено: {r}."))
                elif event == "resource_adjustment":
                    nb = data.get("new_batch_size")
                    self.gui.gui_queue.put(
                        lambda nb=nb: _post(f"Адаптация ресурсов: новый размер батча {nb}."))
                        
        except (AttributeError, TypeError, RuntimeError, KeyError):
            pass
    
    # =========================================================================
    # Вспомогательные методы
    # =========================================================================
    
    def _generate_response_obj(self, message: str, timeout_sec: Optional[float] = None, 
                               fallback_response: Optional[str] = None) -> Tuple[Any, float]:
        """Генерирует объект ответа от ядра."""
        start = time.time()
        brain = getattr(self.gui, 'brain', None)
        
        # Debug logging
        if brain:
            print(f"[DEBUG CHAT] brain found: {type(brain)}, has process_query: {hasattr(brain, 'process_query')}")
            if hasattr(brain, 'ml_unit'):
                print(f"[DEBUG CHAT] brain.ml_unit found: {type(brain.ml_unit)}")
            else:
                print("[DEBUG CHAT] brain.ml_unit NOT FOUND")
        else:
            print("[DEBUG CHAT] brain is None!")
        
        def _select_callable():
            try:
                if brain and hasattr(brain, 'process_query'):
                    return brain.process_query
                if brain and hasattr(brain, 'ml_unit') and hasattr(brain.ml_unit, 'process_query'):
                    return brain.ml_unit.process_query
                if brain and hasattr(brain, 'ml_unit') and hasattr(brain.ml_unit, 'generate_response'):
                    return brain.ml_unit.generate_response
            except (AttributeError, TypeError) as e:
                logger.debug(f"Error in _select_callable: {e}")
                pass
            return None
        
        try:
            target = _select_callable()
            
            if target is None:
                response_obj = {"text": "Ядро и ML-модуль недоступны для обработки запроса."}
            else:
                # Убираем ограничение по времени - запросы могут выполняться без таймаута
                response_obj = target(message)
                
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}", exc_info=True)
            response_obj = {"text": f"Ошибка обработки запроса: {str(e)}"}
        
        return response_obj, (time.time() - start)
    
    def _extract_response_fields(self, response_obj: Any, 
                                 fallback_input: Optional[str] = None) -> Tuple[str, List[Any], Any, Optional[str], List[Dict[str, Any]], bool, Dict[str, Any]]:
        """Извлекает поля из ответа ядра."""
        text = ""
        tokens: List[Any] = []
        sentiment = None
        reasoning = None
        contradictions: List[Dict[str, Any]] = []
        contradiction_flag = False
        metadata: Dict[str, Any] = {}
        
        try:
            if isinstance(response_obj, dict):
                text = response_obj.get("text", "") or response_obj.get("response", "") or ""
                tokens = response_obj.get("tokens", []) or response_obj.get("token_list", []) or []
                reasoning = response_obj.get("reasoning") or response_obj.get("explanation")
                contradictions = response_obj.get("contradictions", []) or []
                contradiction_flag = bool(response_obj.get("contradiction_detected", False) or contradictions)
                sentiment = response_obj.get("sentiment")
                metadata = response_obj.get("metadata", {})
            else:
                text = str(response_obj)
            
            # NLP обработка
            ml_unit = getattr(getattr(self.gui, 'brain', None), 'ml_unit', None)
            utp = getattr(ml_unit, 'unified_text_processor', None) if ml_unit else None
            
            if utp and hasattr(utp, 'process_text'):
                processed = utp.process_text(text if text else (fallback_input or ""))
                tokens = processed.get("keywords") or processed.get("tokens") or []
                if tokens and isinstance(tokens[0], tuple):
                    tokens = [k for k, _ in tokens]
                sentiment = processed.get("sentiment")
                
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.debug(f"_extract_response_fields: не удалось распарсить ответ: {e}", exc_info=True)
        
        return text, tokens, sentiment, reasoning, contradictions, contradiction_flag, metadata
    
    def _summarize_contradictions(self, contradictions: List[Dict[str, Any]], limit: int = 3) -> str:
        """Формирует краткую сводку противоречий."""
        try:
            items = []
            for c in contradictions[:limit]:
                if isinstance(c, dict):
                    s = c.get("summary") or c.get("title") or c.get("type") or c.get("description")
                    if not s and "concept" in c and "domains" in c:
                        s = f"{c['concept']}: конфликт доменов"
                    items.append(_fix_mojibake(str(s)))
                else:
                    items.append(_fix_mojibake(str(c)))
            return "; ".join([i for i in items if i])
        except (AttributeError, TypeError, ValueError, UnicodeError):
            return ""
    
    def _extract_concept_for_integration(self, metadata: Dict[str, Any], 
                                         tokens: List[Any], 
                                         fallback_text: str) -> Optional[str]:
        """Извлекает концепт для интеграции знаний."""
        try:
            for key in ("concept", "topic", "focus_concept"):
                val = metadata.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()[:128]
            
            nlp_info = metadata.get("nlp") if isinstance(metadata, dict) else None
            if isinstance(nlp_info, dict):
                for key in ("main_concept", "entities", "keywords"):
                    val = nlp_info.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()[:128]
                    if isinstance(val, list) and val:
                        first = val[0]
                        return (first[0] if isinstance(first, (list, tuple)) else str(first)).strip()[:128]
            
            if tokens:
                first = tokens[0]
                return (first[0] if isinstance(first, (list, tuple)) else str(first)).strip()[:128]
            
            return " ".join(fallback_text.split()[:10]) if fallback_text else None
            
        except (AttributeError, TypeError, ValueError, IndexError, UnicodeError):
            return None
    
    def _invoke_knowledge_integration(self, concept: str):
        """Вызывает интеграцию знаний."""
        try:
            from cogniflex.knowledge.knowledge_integrator import KnowledgeIntegrator
        except Exception as e:
            logger.debug(f"Не удалось импортировать KnowledgeIntegrator: {e}")
            return
        
        try:
            integrator = None
            bi = getattr(self.gui, 'brain', None)
            if bi:
                integrator = getattr(bi, 'knowledge_integrator', None)

            if integrator is None:
                integrator = KnowledgeIntegrator(brain=bi)
                if bi:
                    bi.knowledge_integrator = integrator
                    logger.debug("Создан новый KnowledgeIntegrator и сохранен в brain")
                else:
                    logger.debug("Создан новый KnowledgeIntegrator без brain")

            integrator.integrate_knowledge(concept, depth=1)
            
        except Exception as e:
            logger.debug(f"Ошибка вызова интеграции знаний: {e}")
    
    def _handle_url_click(self, event):
        """Обрабатывает клик по гиперссылке."""
        try:
            if not self.chat_display.winfo_exists():
                return
            
            index = self.chat_display.index(f"@{event.x},{event.y}")
            for tag in self.chat_display.tag_names(index):
                if tag.startswith("url_"):
                    url = tag[4:]
                    webbrowser.open(url)
                    break
                    
        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка обработки клика по URL: {e}", exc_info=True)
    
    def _handle_url_enter(self, event):
        """Изменяет курсор при наведении на гиперссылку."""
        try:
            if not self.chat_display.winfo_exists():
                return
            self.chat_display.config(cursor="hand2")
        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка изменения курсора (enter): {e}", exc_info=True)
    
    def _handle_url_leave(self, event):
        """Возвращает обычный курсор."""
        try:
            if not self.chat_display.winfo_exists():
                return
            self.chat_display.config(cursor="")
        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка изменения курсора (leave): {e}", exc_info=True)
    
    def _open_image(self, url: str):
        """Открывает изображение в браузере."""
        webbrowser.open(url)
    
    def update_theme(self):
        """Обновляет тему чата."""
        if not hasattr(self, 'chat_display') or not self.chat_display or \
           not self.chat_display.winfo_exists():
            return
        
        try:
            self.chat_display.config(
                bg=self.gui.colors['card-bg'],
                fg=self.gui.colors['text']
            )
            
            self.chat_display.tag_configure("user", foreground=self.gui.colors['primary'])
            self.chat_display.tag_configure("system", foreground=self.gui.colors['text'])
            self.chat_display.tag_configure("reasoning", foreground=self.gui.colors['text-muted'])
            self.chat_display.tag_configure("timestamp", foreground=self.gui.colors['text-muted'])
            self.chat_display.tag_configure("url", foreground=self.gui.colors['primary'])
            
            self._redraw_chat()
            
        except (tk.TclError, Exception) as e:
            logger.error(f"Ошибка обновления темы чата: {e}", exc_info=True)
    
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