"""Модуль чата для ЕВА GUI - полнофункциональная реализация"""
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
    from eva_ai.tools.import_pipeline import ImportPipeline
except ImportError:
    ImportPipeline = None

from eva_ai.gui.chat_messages import ChatMessagesMixin
from eva_ai.gui.chat_input import ChatInputMixin
from eva_ai.gui.chat_history import ChatHistoryMixin
from eva_ai.gui.chat_actions import ChatActionsMixin
from eva_ai.gui.chat_reasoning import ChatReasoningMixin
from eva_ai.gui.chat_text_utils import _to_display_str, _fix_mojibake

logger = logging.getLogger("eva_ai.gui.chat")


class ChatModule(ChatMessagesMixin, ChatInputMixin, ChatHistoryMixin, ChatActionsMixin, ChatReasoningMixin):
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
        
        # Контекст из выделенного текста
        self._selection_context = None
        self._context_label = None
        
        # История и очередь запросов
        self.message_history = []
        self._history_lock = threading.Lock()
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
        self.heading_pattern = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)
        self.list_pattern = re.compile(r'^[\-\*]\s+(.+)$', re.MULTILINE)
        self.numbered_list_pattern = re.compile(r'^\d+\.\s+(.+)$', re.MULTILINE)
        self.bullet_emoji_pattern = re.compile(r'^[\U0001F537-\U0001F93A]\s+(.+)$', re.MULTILINE)
        self.hr_pattern = re.compile(r'^[\-\*]{3,}$', re.MULTILINE)
        self.comment_pattern = re.compile(r'§([^§]+)§')
        
        # Инициализация флага рассуждений
        setattr(self.gui, 'reasoning_active', True)
        self.gui.reasoning_active = True
        
        # Инициализация состояний
        self._suppress_history_append = False
        self._ml_ready_cached = False
        self._import_pipeline = None
        self._draft_text = None
            
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
                    brain.events.subscribe('model_load', 
                        lambda data: self.gui.gui_queue.put(lambda: handler(data)))
                    brain.events.subscribe('models_ready', 
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
                lambda: self._add_message("ЕВА", "Обрабатываю запрос...", "system"))
            
            # Проверка готовности системы
            brain = getattr(self.gui, 'brain', None)
            if not brain:
                self.gui.gui_queue.put(lambda: self._add_message(
                    "ЕВА", 
                    "Ядро системы недоступно. Проверьте, что все компоненты системы запущены.",
                    "system"))
                return
            
            components = getattr(brain, 'components', {}) or {}
            component = components.get('query_processor') if hasattr(brain, 'components') else None
            has_query_processor = (
                brain and 
                component is not None and
                callable(getattr(component, 'process_query', None))
            )
            has_fractal = getattr(brain, 'fractal_ready', False)
            has_memory = getattr(brain, 'memory_manager', None) is not None
            
            if not has_query_processor and not has_fractal and not has_memory:
                self.gui.gui_queue.put(lambda: self._add_message(
                    "ЕВА",
                    "Фрактальное хранилище и граф памяти недоступны. Чат не может работать.",
                    "system"))
                return
            elif not has_fractal and not has_memory:
                self.gui.gui_queue.put(lambda: self._add_message(
                    "ЕВА",
                    "Фрактальное хранилище и граф памяти недоступны. Чат не может работать.",
                    "system"))
                return
            
            if brain is None or not hasattr(brain, 'process_query'):
                self.gui.gui_queue.put(lambda: self._add_message(
                    "ЕВА",
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
            extracted = self._extract_response_fields(response_obj, fallback_input=user_message)
            if len(extracted) == 7:
                text, tokens, sentiment, reasoning, contradictions, contradiction_flag, metadata = extracted
            else:
                text, tokens, sentiment, reasoning, contradictions, contradiction_flag, metadata = "", [], "", [], [], False, {}
            
            # Форматирование ответа
            display_text = _fix_mojibake(str(text)).strip() if text else "Извините, система не смогла сформировать ответ."
            
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
                lambda m=final_message, ex=extras: self._add_message("ЕВА", m, "system", extras=ex))
            
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
            self.gui.gui_queue.put(lambda: self._add_message("ЕВА", error_msg, "system"))
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
        
        # Панель действий с ответом
        self._create_response_action_bar()
        
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
    
    def _setup_chat_event_handlers(self):
        """Настраивает обработчики событий для области чата."""
        self.chat_display.tag_bind("url", "<Enter>", self._handle_url_enter)
        self.chat_display.tag_bind("url", "<Leave>", self._handle_url_leave)
        self.chat_display.bind("<Button-1>", self._handle_url_click)
        self.chat_display.bind("<Button-3>", self._show_context_menu)
        
        # Привязка для показа панели действий при выборе
        self.chat_display.bind("<<Selection>>", self._on_text_selection_changed)
        self.chat_display.bind("<ButtonRelease-1>", self._on_text_selection_changed)
        
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
                if status and isinstance(status, dict) and status.get('enabled'):
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
            
        except (AttributeError, TypeError, ValueError, tk.TclError):
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
            
            ext = os.path.splitext(filename)[1].lower()
            
            # Для текстовых файлов - сначала показать в чате
            if ext in {'.txt', '.md', '.log', '.json', '.xml', '.csv', '.yaml', '.yml'}:
                self._display_text_file(filename)
            else:
                # Для PDF/EPUB - импорт и обучение
                self._add_message("ЕВА", 
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
    
    def _display_text_file(self, filepath: str):
        """Отображает содержимое текстового файла в чате."""
        try:
            from eva_ai.tools.document_reader import DocumentTextReader
            
            reader = DocumentTextReader(max_chars=50000)
            messages = reader.read_as_messages(filepath, max_lines=150)
            
            for msg in messages:
                self._add_message(
                    sender=msg["sender"],
                    message=msg["text"],
                    msg_type=msg["type"]
                )
                
        except Exception as e:
            logger.error(f"Ошибка отображения файла: {e}", exc_info=True)
            self._add_message("ЕВА", f"Ошибка чтения файла: {str(e)}", "system")
    
    def _import_and_maybe_train(self, path: str):
        """Импортирует документ и запускает обучение."""
        try:
            brain = getattr(self.gui, 'brain', None)
            if not brain:
                self.gui.gui_queue.put(lambda: self._add_message(
                    "ЕВА", "Ядро недоступно — импорт невозможен.", "system"))
                return
            
            # Инициализация пайплайна
            if ImportPipeline is None:
                self.gui.gui_queue.put(lambda: self._add_message(
                    "ЕВА", "ImportPipeline не доступен.", "system"))
                return
            
            try:
                if self._import_pipeline is None:
                    self._import_pipeline = ImportPipeline(brain=brain)
            except Exception as e:
                logger.error(f"Не удалось инициализировать ImportPipeline: {e}", exc_info=True)
                self.gui.gui_queue.put(lambda: self._add_message(
                    "ЕВА", f"Не удалось инициализировать импорт: {str(e)}", "system"))
                return
            
            # Импорт
            imported = self._import_pipeline.import_path(path)
            segments = list(imported.iter_segments())
            seg_count = len(segments)
            
            self.gui.gui_queue.put(lambda: self._add_message(
                "ЕВА", f"Импорт завершён: найдено сегментов — {seg_count}", "system"))
            
            # Обучение
            self._run_training(imported)
            
        except Exception as e:
            logger.error(f"Общая ошибка импорта/обучения: {e}", exc_info=True)
            self.gui.gui_queue.put(lambda: self._add_message(
                "ЕВА", f"Ошибка импорта: {str(e)}", "system"))
    
    def _run_training(self, imported):
        """Запускает обучение на импортированных данных."""
        try:
            brain = getattr(self.gui, 'brain', None)
            tor = None
            
            ml_unit = getattr(brain, 'ml_unit', None)
            if ml_unit and hasattr(ml_unit, 'training_orchestrator'):
                tor = ml_unit.training_orchestrator
            
            if tor is None:
                                pass  # Training disabled
            
            result = tor.train_from_document(imported)
            status = (result or {}).get("status")
            
            if status == "completed":
                self.gui.gui_queue.put(lambda: self._add_message(
                    "ЕВА", "Обучение по документу завершено успешно.", "system"))
            elif status == "deferred":
                reason = (result or {}).get("reason", "модели не готовы или кэш недоступен")
                self.gui.gui_queue.put(lambda r=reason: self._add_message(
                    "ЕВА", 
                    f"Обучение отложено: {r}. Документ будет обработан позже.", 
                    "system"))
            elif status == "failed":
                err = (result or {}).get("error", "неизвестная ошибка")
                self.gui.gui_queue.put(lambda e=err: self._add_message(
                    "ЕВА", f"Ошибка обучения: {e}", "system"))
            else:
                self.gui.gui_queue.put(lambda: self._add_message(
                    "ЕВА", f"Статус обучения: {status or 'неизвестно'}", "system"))
                    
        except Exception as e:
            logger.error(f"Ошибка запуска обучения: {e}", exc_info=True)
            self.gui.gui_queue.put(lambda: self._add_message(
                "ЕВА", f"Не удалось запустить обучение: {str(e)}", "system"))
    
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
            logger.debug(f"[DEBUG CHAT] brain found: {type(brain)}, has process_query: {hasattr(brain, 'process_query')}")
            if hasattr(brain, 'ml_unit'):
                logger.debug(f"[DEBUG CHAT] brain.ml_unit found: {type(brain.ml_unit)}")
            else:
                logger.debug("[DEBUG CHAT] brain.ml_unit NOT FOUND")
        else:
            logger.debug("[DEBUG CHAT] brain is None!")
        
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
            from eva_ai.knowledge.knowledge_integrator import KnowledgeIntegrator
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

            if hasattr(integrator, 'integrate_knowledge'):
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
