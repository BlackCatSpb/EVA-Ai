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
from cogniflex.tools.import_pipeline import ImportPipeline

logger = logging.getLogger("cogniflex.gui.chat")

def _to_display_str(val: Any) -> str:
    """Безопасно преобразует значение к строке для отображения в UTF-8.
    Преобразует bytes -> utf-8 с заменой, нормализует переводы строк.
    """
    try:
        if isinstance(val, bytes):
            s = val.decode("utf-8", errors="replace")
        else:
            s = str(val)
        # Нормализуем CRLF -> LF
        return s.replace("\r\n", "\n").replace("\r", "\n")
    except Exception:
        return str(val)

def _looks_mojibake(s: str) -> bool:
    """Грубая эвристика для детекции 'Ð', 'Ñ' и похожих артефактов mojibake."""
    if not s:
        return False
    bad_chars = set("ÐÑÂÃĤĭĮıİıĝĞġĠ")
    return any(ch in bad_chars for ch in s)

def _fix_mojibake(s: str) -> str:
    """Пытается исправить типичный mojibake (UTF-8, показанный как Latin-1).
    Ничего не ломает, если строка нормальная.
    """
    try:
        s0 = _to_display_str(s)
        if _looks_mojibake(s0):
            try:
                repaired = s0.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
                if repaired and repaired != s0:
                    return repaired
            except Exception:
                pass
        return s0
    except Exception:
        return _to_display_str(s)

class ChatModule:
    """Модуль чата для взаимодействия с пользователем."""
    
    def __init__(self, gui):
        self.gui = gui
        self.chat_frame = None
        self.chat_display = None
        self.input_frame = None
        self.input_text = None
        self.send_button = None
        self.import_button = None
        self.context_menu = None
        self.message_history = []
        self.history_index = -1
        self.pending_requests = set()
        self.request_queue = queue.Queue()
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.url_pattern = re.compile(r'(https?://[^\s]+)')
        self.markdown_link_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\s)]+)\)')
        self.formatting_pattern = re.compile(r'\*\*(.*?)\*\*|__(.*?)__|_(.*?)_|`(.*?)`')
        self.emoji_pattern = re.compile(r'[:;=][-o*]?[)D\]\(\[/\\|Pp]')
        self.image_pattern = re.compile(r'!\[.*?\]\((https?://[^\s]+)\)')
        self.typing_indicator = None
        self.typing_active = False
        self.typing_text = "CogniFlex печатает..."
        # Статус-бар системных метрик
        self.status_frame = None
        self.cpu_label = None
        self.mem_label = None
        self._status_updater_id = None
        # Черновик текста ввода для сохранения между переключениями вкладок
        self._draft_text = ""
        # Флаг подавления записи в историю при восстановлении сообщений
        self._suppress_history_append = False
        # Пайплайн импорта документов
        self._import_pipeline: Optional[ImportPipeline] = None
        # Режим автодиалога (диалог системы сама с собой)
        self.self_dialog_button = None
        self.self_dialog_active = False
        self.self_dialog_thread: Optional[threading.Thread] = None
        self.self_dialog_stop = threading.Event()
        
        # Добавлено для совместимости с ядром
        if not hasattr(self.gui, 'reasoning_active'):
            self.gui.reasoning_active = True
            
        logger.info("Модуль чата инициализирован")

    def activate(self):
        """Активирует модуль чата."""
        # Очищаем область контента
        for widget in self.gui.content_area.winfo_children():
            widget.destroy()
            
        # Создаем интерфейс чата
        self._create_chat_interface()
        
        # Запускаем фоновый процесс обработки запросов
        self._start_processing_thread()
        
        logger.info("Модуль чата активирован")

    def deactivate(self):
        """Деактивирует модуль чата."""
        # Останавливаем фоновый процесс
        self._stop_processing_thread()
        # Останавливаем автодиалог, если активен
        try:
            if self.self_dialog_active:
                self.self_dialog_stop.set()
                if self.self_dialog_thread and self.self_dialog_thread.is_alive():
                    self.self_dialog_thread.join(timeout=2.0)
        except Exception:
            pass
        # Отменяем периодическое обновление статуса
        try:
            if self._status_updater_id is not None and hasattr(self.gui, 'root'):
                self.gui.root.after_cancel(self._status_updater_id)
        except Exception:
            pass
        finally:
            self._status_updater_id = None
        # Сохраняем черновик текста ввода, чтобы не потерять набранный текст
        try:
            if self.input_text and self.input_text.winfo_exists():
                self._draft_text = self.input_text.get("1.0", tk.END)
        except Exception:
            pass
        
        # Сохраняем историю сообщений
        if hasattr(self, 'message_history'):
            try:
                # Создаем директорию кэша, если она не существует
                if self.gui.cache_dir:
                    os.makedirs(self.gui.cache_dir, exist_ok=True)
                    
                history_file = os.path.join(self.gui.cache_dir, "chat_history.json")
                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump(self.message_history, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Ошибка сохранения истории чата: {e}")
        
        logger.info("Модуль чата деактивирован")

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

    def _processing_loop(self):
        """Цикл обработки запросов в фоновом потоке."""
        while not self.stop_event.is_set():
            try:
                # Получаем запрос из очереди
                request = self.request_queue.get(timeout=0.5)
                
                # Обрабатываем запрос
                self._process_request(request)
                
                # Сигнализируем о завершении
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка в цикле обработки запросов чата: {e}", exc_info=True)

    def _process_request(self, request: Dict[str, Any]):
        """Обрабатывает запрос в фоновом потоке. Поддерживает строковый и структурированный ответ от ядра."""
        user_message = request.get("message", "")
        request_id = request.get("request_id")
        start_time = request.get("start_time", time.time())

        try:
            # Если уже отменён/обработан
            if request_id not in self.pending_requests:
                return

            # Показать индикатор ожидания
            self.gui.gui_queue.put(self._show_typing)
            self.gui.gui_queue.put(lambda: self._add_message("CogniFlex", "Обрабатываю запрос...", "system"))

            # Проверки ядра
            if not self.gui.brain:
                error_msg = "Ядро системы недоступно. Проверьте, что все компоненты системы запущены."
                self.gui.gui_queue.put(lambda: self._add_message("CogniFlex", error_msg, "system"))
                return

            if not hasattr(self.gui.brain, 'process_query') and not hasattr(self.gui.brain, 'ml_unit'):
                error_msg = "Ядро не поддерживает обработку запросов. Проверьте конфигурацию."
                self.gui.gui_queue.put(lambda: self._add_message("CogniFlex", error_msg, "system"))
                return

            # Выполнить вызов ядра — пытаемся вызвать наиболее подходящий API
            try:
                # Сначала пробуем новый API: ml_unit.generate_response возвращает либо str, либо dict
                if hasattr(self.gui.brain, 'ml_unit') and hasattr(self.gui.brain.ml_unit, 'generate_response'):
                    response_obj = self.gui.brain.ml_unit.generate_response(user_message)
                else:
                    # Бэкап — старый API process_query
                    response_obj = self.gui.brain.process_query(user_message)
            except Exception as e:
                logger.error(f"Ошибка при обработке запроса через ядро: {e}", exc_info=True)
                response_obj = {"text": f"Ошибка обработки запроса: {str(e)}"}

            # Если запрос был отменён параллельно — выйти
            if request_id not in self.pending_requests:
                return

            # Удаляем индикатор ожидания
            self.gui.gui_queue.put(self._remove_last_message)
            self.gui.gui_queue.put(self._hide_typing)

            # Подготовка данных для отображения
            processing_time = time.time() - start_time

            # Унифицируем формат: получаем text, tokens, reasoning, contradictions, sentiment, metadata
            text = ""
            tokens = []
            reasoning = None
            contradictions = []
            contradiction_flag = False
            sentiment = None
            metadata = {}

            # Если ядро вернуло словарь/структуру
            if isinstance(response_obj, dict):
                text = response_obj.get("text", "") or response_obj.get("response", "") or ""
                tokens = response_obj.get("tokens", []) or response_obj.get("token_list", []) or []
                reasoning = response_obj.get("reasoning") or response_obj.get("explanation")
                contradictions = response_obj.get("contradictions", []) or []
                contradiction_flag = bool(response_obj.get("contradiction_detected", False) or contradictions)
                sentiment = response_obj.get("sentiment")  # ожидается словарь/строка
                metadata = response_obj.get("metadata", {})
            else:
                # Если вернулась строка — используем её как основной текст
                text = str(response_obj)

                # Попытаемся извлечь токены/ключевые слова/тональность через ml_unit.unified_text_processor (если доступен)
                try:
                    ml_unit = getattr(self.gui.brain, 'ml_unit', None)
                    utp = getattr(ml_unit, 'unified_text_processor', None) if ml_unit else None
                    if utp and hasattr(utp, 'process_text'):
                        processed = utp.process_text(text if text else user_message)
                        tokens = processed.get("keywords") or processed.get("tokens") or []
                        # keywords may be list of tuples -> normalize to strings
                        if tokens and isinstance(tokens[0], tuple):
                            tokens = [k for k, _ in tokens]
                        sentiment = processed.get("sentiment")
                except Exception:
                    # не критично — оставляем пустые tokens/sentiment
                    logger.debug("Не удалось получить ключевые слова/тональность из unified_text_processor", exc_info=True)

            # Форматируем отображаемый ответ: текст + вспомогательная аналитика
            # Если text пуст — сформируем осмысленную строку (защитный fallback)
            if not text or str(text).strip() == "":
                display_text = "Извините, система не смогла сформировать ответ."
            else:
                display_text = _fix_mojibake(text).strip()

            # Формируем блок аналитики (ключевые слова, тональность, время)
            analytics_lines = []

            # Ключевые слова показываем только если они реально получены
            if tokens:
                # tokens может быть список кортежей (keyword, score) или список строк
                if isinstance(tokens[0], (list, tuple)) and len(tokens[0]) >= 1:
                    keywords_display = ", ".join([_fix_mojibake(k[0]) for k in tokens[:5]])
                else:
                    keywords_display = ", ".join([_fix_mojibake(k) for k in tokens[:5]])
                if keywords_display.strip():
                    analytics_lines.append(f"Ключевые слова: {keywords_display}")

            # Тональность показываем только если доступна
            sentiment_display = ""
            if sentiment:
                if isinstance(sentiment, dict):
                    # простой выбор: использовать compound или label
                    sentiment_display = sentiment.get("compound") if "compound" in sentiment else sentiment.get("label", "")
                else:
                    sentiment_display = _fix_mojibake(sentiment)
                if str(sentiment_display).strip():
                    analytics_lines.append(f"Тональность: {sentiment_display}")

            # Краткое отображение противоречий (если есть)
            if contradictions:
                short_contra = self._summarize_contradictions(contradictions)
                if short_contra:
                    analytics_lines.append(f"Противоречия: {short_contra}")

            # Всегда показываем время обработки
            analytics_lines.append(f"Время обработки: {processing_time:.2f} сек")

            # Добавляем reasoning, если он есть и пользователь включил отображение рассуждений
            # Показываем рассуждения, если доступны
            if reasoning and getattr(self.gui, "reasoning_active", True):
                # Показываем reasoning отдельно
                self.gui.gui_queue.put(lambda: self._add_message("CogniFlex (рассуждения)", _fix_mojibake(reasoning), "reasoning"))

            # Добавляем основной текст + аналитику в GUI
            if analytics_lines:
                final_message = display_text + "\n\n" + "\n".join(analytics_lines)
            else:
                final_message = display_text

            # Сохраняем расширенные метаданные вместе с сообщением
            extras = {
                "processing_time": processing_time,
                "tokens": tokens,
                "sentiment": sentiment,
                "contradictions": contradictions,
                "metadata": metadata,
                "reasoning": reasoning,
            }
            self.gui.gui_queue.put(lambda m=final_message, ex=extras: self._add_message("CogniFlex", m, "system", extras=ex))

            # Если были противоречия — уведомляем
            if contradiction_flag:
                num = len(contradictions) if contradictions else 1
                self.gui.gui_queue.put(lambda: self.gui.show_notification(
                    f"Обнаружено {num} противоречение(ий) в знаниях",
                    "warning",
                    actions=[{"text": "Посмотреть", "command": lambda: self.gui._switch_view("contradictions")}]
                ))

            # Вызов интеграции знаний в фоне (не блокируем UI)
            concept_for_kg = self._extract_concept_for_integration(metadata, tokens, user_message)
            if concept_for_kg:
                threading.Thread(target=self._invoke_knowledge_integration, args=(concept_for_kg,), name="KnowledgeIntegrate", daemon=True).start()

        except Exception as e:
            logger.error(f"Ошибка обработки ответа в _process_request: {e}", exc_info=True)
            error_msg = (
                "Произошла ошибка при обработке запроса:\n"
                f"{str(e)}\n\n"
                "Попробуйте повторить запрос или обратиться к системному администратору."
            )
            self.gui.gui_queue.put(lambda: self._add_message("CogniFlex", error_msg, "system"))
        finally:
            # Всегда удаляем из очереди pending
            if request_id in self.pending_requests:
                try:
                    self.pending_requests.remove(request_id)
                except Exception:
                    logger.debug("Не удалось удалить request_id из pending_requests", exc_info=True)
            # На всякий случай скрываем индикатор
            try:
                self.gui.gui_queue.put(self._hide_typing)
            except Exception:
                pass


    
    def _create_chat_interface(self):
        """Создает интерфейс чата."""
        self.chat_frame = ttk.Frame(self.gui.content_area)
        self.chat_frame.pack(fill=tk.BOTH, expand=True)
        
        # Область отображения сообщений с поддержкой гиперссылок
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
        
        # Настройка стилей для сообщений
        self.chat_display.tag_configure("user", foreground=self.gui.colors['primary'], font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_configure("system", foreground=self.gui.colors['text'], font=('Segoe UI', 10))
        self.chat_display.tag_configure("reasoning", foreground=self.gui.colors['text-muted'], font=('Segoe UI', 10, 'italic'))
        self.chat_display.tag_configure("timestamp", foreground=self.gui.colors['text-muted'], font=('Segoe UI', 8))
        self.chat_display.tag_configure("url", foreground=self.gui.colors['primary'], underline=True)
        self.chat_display.tag_configure("bold", font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_configure("italic", font=('Segoe UI', 10, 'italic'))
        self.chat_display.tag_configure("code", background=self.gui.colors['bg'], font=('Consolas', 9))
        self.chat_display.tag_configure("emoji", font=('Segoe UI Emoji', 10))
        
        # Обработчик кликов по гиперссылкам
        self.chat_display.tag_bind("url", "<Button-1>", self._handle_url_click)
        self.chat_display.tag_bind("url", "<Enter>", self._handle_url_enter)
        self.chat_display.tag_bind("url", "<Leave>", self._handle_url_leave)
        
        # Создаем контекстное меню
        self.context_menu = Menu(self.chat_display, tearoff=0)
        self.context_menu.add_command(label="Копировать", command=self._copy_selected)
        self.context_menu.add_command(label="Копировать все", command=self._copy_all)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Очистить чат", command=self._clear_chat)
        
        # Привязываем контекстное меню
        self.chat_display.bind("<Button-3>", self._show_context_menu)
        
        # Горячие клавиши для области чата (копирование/выделить всё)
        try:
            self.chat_display.bind("<Control-c>", lambda e: (self._copy_selected(), "break"))
            self.chat_display.bind("<Control-C>", lambda e: (self._copy_selected(), "break"))
            self.chat_display.bind("<Control-a>", self._on_chat_select_all)
            self.chat_display.bind("<Control-A>", self._on_chat_select_all)
        except Exception:
            pass
        
        # Область ввода
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
        
        # Устанавливаем фокус на поле ввода, чтобы горячие клавиши сразу работали
        try:
            self.input_text.focus_set()
        except Exception:
            pass
        
        # Настройка контекстного меню для поля ввода
        input_context_menu = Menu(self.input_text, tearoff=0)
        input_context_menu.add_command(label="Вырезать", command=lambda: self._cut_text(self.input_text))
        input_context_menu.add_command(label="Копировать", command=lambda: self._copy_text(self.input_text))
        input_context_menu.add_command(label="Вставить", command=lambda: self._paste_text(self.input_text))
        input_context_menu.add_separator()
        input_context_menu.add_command(label="Очистить", command=lambda: self.input_text.delete("1.0", tk.END))
        
        self.input_text.bind("<Button-3>", lambda event: input_context_menu.tk_popup(event.x_root, event.y_root))
        
        # Кнопка импорта документа (TXT/PDF/EPUB)
        self.import_button = ttk.Button(
            self.input_frame,
            text="Импорт",
            command=self._on_import_document
        )
        self.import_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Кнопка автодиалога (старт/стоп)
        self.self_dialog_button = ttk.Button(
            self.input_frame,
            text="Автодиалог",
            command=self._toggle_self_dialog
        )
        self.self_dialog_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Кнопка отправки
        self.send_button = ttk.Button(
            self.input_frame,
            text="Отправить",
            command=self._send_message
        )
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Привязываем события
        self.input_text.bind("<Return>", self._on_enter_pressed)
        self.input_text.bind("<Up>", self._on_history_up)
        self.input_text.bind("<Down>", self._on_history_down)
        # Горячие клавиши: копирование/вставка/вырезание/выделить всё
        self.input_text.bind("<Control-c>", self._on_copy_shortcut)
        self.input_text.bind("<Control-C>", self._on_copy_shortcut)
        self.input_text.bind("<Control-v>", self._on_paste_shortcut)
        self.input_text.bind("<Control-V>", self._on_paste_shortcut)
        self.input_text.bind("<Control-x>", self._on_cut_shortcut)
        self.input_text.bind("<Control-X>", self._on_cut_shortcut)
        self.input_text.bind("<Control-a>", self._on_select_all_shortcut)
        self.input_text.bind("<Control-A>", self._on_select_all_shortcut)

        # Глобальная горячая клавиша справки (F1)
        try:
            if hasattr(self.gui, 'root') and self.gui.root:
                self.gui.root.bind("<F1>", self._on_help)
            else:
                # Резервно: биндим на фрейм
                self.chat_frame.bind_all("<F1>", self._on_help)
        except Exception:
            pass

        
        # Статус-бар с метриками CPU/RAM
        self.status_frame = ttk.Frame(self.chat_frame)
        self.status_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        self.cpu_label = ttk.Label(self.status_frame, text="CPU: --%")
        self.cpu_label.pack(side=tk.LEFT)
        self.mem_label = ttk.Label(self.status_frame, text="RAM: --%")
        self.mem_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Индикатор набора текста (видимый индикатор печати)
        try:
            self.typing_indicator = ttk.Label(self.chat_frame, text=self.typing_text, foreground=self.gui.colors['text-muted'])
            # Скрыт по умолчанию
            self.typing_indicator.pack_forget()
        except Exception:
            self.typing_indicator = None
        
        # Загружаем историю чата
        self._load_chat_history()
        
        # Показываем приветствие только если нет истории
        if not self.message_history:
            welcome_msg = (
                "Добро пожаловать в CogniFlex!\n\n"
                "Я - когнитивная система с поддержкой:\n"
                "• Этического анализа\n"
                "• Адаптации под пользователя\n"
                "• Распределенных вычислений\n"
                "• Управления знаниями\n\n"
                "Задайте ваш первый вопрос или нажмите F1 для просмотра справки."
            )
            self._add_message("CogniFlex", welcome_msg, "system")

        # Восстанавливаем черновик текста ввода, если он есть
        try:
            if self._draft_text:
                self.input_text.insert("1.0", self._draft_text)
                self.input_text.see(tk.END)
        except Exception:
            pass

        # Запускаем обновление статуса метрик
        self._schedule_status_update()

    # ----- Вспомогательные методы интеграции/извлечения -----
    def _summarize_contradictions(self, contradictions: List[Dict[str, Any]], limit: int = 3) -> str:
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
        except Exception:
            return ""

    def _extract_concept_for_integration(self, metadata: Dict[str, Any], tokens: List[Any], fallback_text: str) -> Optional[str]:
        try:
            # Пытаемся достать из metadata
            for key in ("concept", "topic", "focus_concept"):
                val = metadata.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()[:128]
            # Пробуем из NLP
            nlp_info = metadata.get("nlp") if isinstance(metadata, dict) else None
            if isinstance(nlp_info, dict):
                for key in ("main_concept", "entities", "keywords"):
                    val = nlp_info.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()[:128]
                    if isinstance(val, list) and val:
                        first = val[0]
                        return (first[0] if isinstance(first, (list, tuple)) else str(first)).strip()[:128]
            # Из tokens
            if tokens:
                first = tokens[0]
                return (first[0] if isinstance(first, (list, tuple)) else str(first)).strip()[:128]
            # Фолбэк — первые 10 слов запроса
            return " ".join(fallback_text.split()[:10]) if fallback_text else None
        except Exception:
            return None

    def _invoke_knowledge_integration(self, concept: str):
        try:
            from cogniflex.knowledge.knowledge_integrator import KnowledgeIntegrator
        except Exception as e:
            logger.debug(f"Не удалось импортировать KnowledgeIntegrator: {e}")
            return
        try:
            integrator = None
            # Если у ядра уже есть интегратор — используем его
            bi = getattr(self.gui, 'brain', None)
            integrator = getattr(bi, 'knowledge_integrator', None) if bi else None
            if integrator is None:
                integrator = KnowledgeIntegrator(brain=bi)
            integrator.integrate_knowledge(concept, depth=1)
        except Exception as e:
            logger.debug(f"Ошибка вызова интеграции знаний: {e}")

    def _generate_response_obj(self, message: str) -> Tuple[Any, float]:
        start = time.time()
        try:
            brain = getattr(self.gui, 'brain', None)
            if brain and hasattr(brain, 'process_query'):
                response_obj = brain.process_query(message)
            elif brain and hasattr(brain, 'ml_unit') and hasattr(brain.ml_unit, 'process_query'):
                response_obj = brain.ml_unit.process_query(message)
            elif brain and hasattr(brain, 'ml_unit') and hasattr(brain.ml_unit, 'generate_response'):
                response_obj = brain.ml_unit.generate_response(message)
            else:
                response_obj = {"text": "Ядро и ML-модуль недоступны для обработки запроса."}
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}", exc_info=True)
            response_obj = {"text": f"Ошибка обработки запроса: {str(e)}"}
        return response_obj, (time.time() - start)

    def _extract_response_fields(self, response_obj: Any, fallback_input: Optional[str] = None) -> Tuple[str, List[Any], Any, Optional[str], List[Dict[str, Any]], bool, Dict[str, Any]]:
        """Унифицирует ответ ядра в поля: text, tokens, sentiment, reasoning, contradictions, contradiction_flag, metadata."""
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
                # Попытка извлечь базовую аналитiku через unified_text_processor
                ml_unit = getattr(getattr(self.gui, 'brain', None), 'ml_unit', None)
                utp = getattr(ml_unit, 'unified_text_processor', None) if ml_unit else None
                if utp and hasattr(utp, 'process_text'):
                    processed = utp.process_text(text if text else (fallback_input or ""))
                    tokens = processed.get("keywords") or processed.get("tokens") or []
                    if tokens and isinstance(tokens[0], tuple):
                        tokens = [k for k, _ in tokens]
                    sentiment = processed.get("sentiment")
        except Exception:
            logger.debug("_extract_response_fields: не удалось распарсить ответ", exc_info=True)
        return text, tokens, sentiment, reasoning, contradictions, contradiction_flag, metadata

    # ----- Автодиалог: система ведет диалог сама с собой -----
    def _toggle_self_dialog(self):
        try:
            if not self.self_dialog_active:
                # Запуск
                if not getattr(self.gui, 'brain', None):
                    self._add_message("CogniFlex", "Невозможно запустить автодиалог: ядро недоступно.", "system")
                    return
                self.self_dialog_active = True
                self.self_dialog_stop.clear()
                if self.self_dialog_button:
                    self.self_dialog_button.config(text="Стоп автодиалога")
                self.self_dialog_thread = threading.Thread(target=self._self_dialog_loop, name="SelfDialog", daemon=True)
                self.self_dialog_thread.start()
                self._add_message("CogniFlex", "Автодиалог запущен.", "system")
            else:
                # Остановка
                self.self_dialog_active = False
                self.self_dialog_stop.set()
                if self.self_dialog_button:
                    self.self_dialog_button.config(text="Автодиалог")
                self._add_message("CogniFlex", "Автодиалог остановлен.", "system")
        except Exception as e:
            logger.error(f"Ошибка переключения автодиалога: {e}", exc_info=True)

    def _self_dialog_loop(self):
        """Фоновая петля автодиалога. Запускается/останавливается через кнопку."""
        try:
            # Набор стартовых тем
            starters = [
                "Привет! Давай обсудим преимущества и недостатки различных моделей языковых моделей.",
                "Начнем с приветствия: как ты оцениваешь важность качества данных при обучении моделей?",
                "Предлагаю тему: как улучшить устойчивость моделей к некорректным входным данным?",
            ]
            current_prompt = random.choice(starters)
            # Бесконечный цикл до явной остановки пользователем
            while not self.self_dialog_stop.is_set():
                # 1) Добавляем как сообщение пользователя
                self.gui.gui_queue.put(lambda p=current_prompt: self._add_message("Вы", p, "user"))

                # 2) Получаем ответ модели синхронно в этом потоке
                response_obj, processing_time = self._generate_response_obj(current_prompt)

                # 3) Подготовка отображаемого текста (повторяет форматирование из _process_request)
                text, tokens, sentiment, reasoning, contradictions, contradiction_flag, metadata = self._extract_response_fields(response_obj, fallback_input=current_prompt)

                display_text = _fix_mojibake(text).strip() if text and str(text).strip() else "Извините, система не смогла сформировать ответ."
                analytics_lines = []
                if tokens:
                    if isinstance(tokens[0], (list, tuple)) and len(tokens[0]) >= 1:
                        keywords_display = ", ".join([_fix_mojibake(k[0]) for k in tokens[:5]])
                    else:
                        keywords_display = ", ".join([_fix_mojibake(k) for k in tokens[:5]])
                    if keywords_display.strip():
                        analytics_lines.append(f"Ключевые слова: {keywords_display}")
                sentiment_display = ""
                if sentiment:
                    if isinstance(sentiment, dict):
                        sentiment_display = sentiment.get("compound") if "compound" in sentiment else sentiment.get("label", "")
                    else:
                        sentiment_display = _fix_mojibake(sentiment)
                    if str(sentiment_display).strip():
                        analytics_lines.append(f"Тональность: {sentiment_display}")
                analytics_lines.append(f"Время обработки: {processing_time:.2f} сек")

                if reasoning and getattr(self.gui, "reasoning_active", True):
                    self.gui.gui_queue.put(lambda r=reasoning: self._add_message("CogniFlex (рассуждения)", _fix_mojibake(r), "reasoning"))

                if analytics_lines:
                    final_message = display_text + "\n\n" + "\n".join(analytics_lines)
                else:
                    final_message = display_text

                # 4) Публикуем ответ системы
                self.gui.gui_queue.put(lambda m=final_message: self._add_message("CogniFlex", m, "system"))

                # 5) Подготовка следующего шага: вместо эха ответа — формируем уточняющий вопрос
                current_prompt = self._make_followup_prompt(text if isinstance(text, str) else display_text,
                                                            seed_topic=starters[0])

                # Небольшая пауза, чтобы не перегружать интерфейс
                for _ in range(10):
                    if self.self_dialog_stop.is_set():
                        break
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Ошибка в автодиалоге: {e}", exc_info=True)
        finally:
            # Обновляем состояние кнопки при завершении цикла
            def _finalize_btn():
                self.self_dialog_active = False
                if self.self_dialog_button:
                    self.self_dialog_button.config(text="Автодиалог")
            try:
                if hasattr(self.gui, 'root') and self.gui.root:
                    self.gui.root.after(0, _finalize_btn)
                else:
                    _finalize_btn()
            except Exception:
                # Безопасная деградация при завершении
                try:
                    _finalize_btn()
                except Exception:
                    pass

    def _make_followup_prompt(self, reply_text: str, seed_topic: str = "") -> str:
        """Формирует следующий краткий вопрос по теме, чтобы не эхо-нить ответ целиком."""
        try:
            base = _fix_mojibake(reply_text or "").strip()
            if not base:
                return seed_topic or "Продолжим: уточни сильные и слабые стороны подхода?"

            # Выделим предложения и ключевые слова простейшей эвристикой
            sentences = re.split(r"(?<=[.!?])\s+", base)
            first = sentences[0] if sentences else base
            # Уберем маркдауны/мусор
            first = re.sub(r"[`*_#>]+", "", first)
            # Небольшой срез темы
            words = [w for w in re.findall(r"[\wА-Яа-яЁё\-]+", first) if len(w) > 2]
            topic = " ".join(words[:8]) if words else "тему"

            templates = [
                f"Можешь кратко перечислить плюсы и минусы по теме: {topic}?",
                f"В чём основные компромиссы для {topic}?",
                f"Назови 2-3 ключевых риска и способ их снижения для {topic}.",
                f"Сформулируй критерии выбора и порекомендуй подход для {topic}.",
                f"Какие метрики качества важнее всего для {topic}?",
            ]
            return random.choice(templates)
        except Exception:
            return seed_topic or "Продолжим: какие метрики и компромиссы важны здесь?"

    def _show_typing(self):
        """Показывает индикатор набора текста."""
        try:
            self.typing_active = True
            if self.typing_indicator and self.chat_frame and self.chat_frame.winfo_exists():
                # Показываем индикатор под статус-баром
                # Если уже упакован — ничего не делаем
                if not self.typing_indicator.winfo_ismapped():
                    self.typing_indicator.pack(fill=tk.X, padx=5, pady=(0, 5))
        except Exception:
            pass

    def _hide_typing(self):
        """Скрывает индикатор набора текста."""
        try:
            self.typing_active = False
            if self.typing_indicator and self.typing_indicator.winfo_ismapped():
                self.typing_indicator.pack_forget()
        except Exception:
            pass

    # ----- Горячие клавиши ввода -----
    def _on_copy_shortcut(self, event):
        try:
            self._copy_text(self.input_text)
            return "break"
        except Exception:
            return None

    def _on_paste_shortcut(self, event):
        try:
            self._paste_text(self.input_text)
            return "break"
        except Exception:
            return None

    def _on_cut_shortcut(self, event):
        try:
            self._cut_text(self.input_text)
            return "break"
        except Exception:
            return None

    def _on_select_all_shortcut(self, event):
        try:
            self.input_text.tag_add(tk.SEL, "1.0", tk.END)
            self.input_text.mark_set(tk.INSERT, "1.0")
            self.input_text.see(tk.INSERT)
            return "break"
        except Exception:
            return None

    def _on_help(self, event=None):
        """Обработчик F1: открывает документацию или выводит краткую подсказку."""
        try:
            # Пытаемся открыть локальную документацию, если есть
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
                # Выводим краткую справку в чат
                help_text = (
                    "Горячие клавиши:\n"
                    "• Enter — отправить\n"
                    "• Shift+Enter / Ctrl+Enter — новая строка\n"
                    "• Ctrl+C / Ctrl+V / Ctrl+X — копировать/вставить/вырезать\n"
                    "• Ctrl+A — выделить все\n\n"
                    "Команды:\n"
                    "• Импорт документов: кнопка ‘Импорт’ или меню\n"
                    "• Подсветка ссылок и изображений поддерживается\n"
                )
                self._add_message("Справка", help_text, "system")
        except Exception:
            try:
                messagebox.showinfo("Справка", "Нажимайте Enter для отправки, Shift+Enter — новая строка.")
            except Exception:
                pass
        finally:
            try:
                if self.input_text and self.input_text.winfo_exists():
                    self.input_text.focus_set()
            except Exception:
                pass

    def _schedule_status_update(self, interval_ms: int = 2000):
        """Планирует периодическое обновление статус-бара."""
        try:
            if not hasattr(self.gui, 'root') or not self.gui.root or not self.chat_frame or not self.chat_frame.winfo_exists():
                return
            # Выполнить сразу
            self._update_status_bar()
            # Запланировать следующее обновление
            self._status_updater_id = self.gui.root.after(interval_ms, lambda: self._schedule_status_update(interval_ms))
        except Exception:
            # В случае ошибки не прерываем работу чата
            self._status_updater_id = None

    def _update_status_bar(self):
        """Обновляет значения CPU/RAM в статус-баре."""
        try:
            if not self.status_frame or not self.status_frame.winfo_exists():
                return
            cpu_pct = "--"
            mem_pct = "--"
            if getattr(self.gui, 'brain', None) and hasattr(self.gui.brain, 'get_system_metrics'):
                metrics = self.gui.brain.get_system_metrics() or {}
                # В Analytics значения ожидаются как доли (0..1). Приводим к %.
                cpu = metrics.get("cpu_usage")
                mem = metrics.get("memory_usage")
                if isinstance(cpu, (int, float)):
                    # Поддержка как долей, так и процентов
                    cpu_pct_val = cpu * 100.0 if cpu <= 1.5 else float(cpu)
                    cpu_pct = f"{cpu_pct_val:.0f}"
                if isinstance(mem, (int, float)):
                    mem_pct_val = mem * 100.0 if mem <= 1.5 else float(mem)
                    mem_pct = f"{mem_pct_val:.0f}"
            if self.cpu_label:
                self.cpu_label.config(text=f"CPU: {cpu_pct}%")
            if self.mem_label:
                self.mem_label.config(text=f"RAM: {mem_pct}%")
        except Exception:
            # Безопасная деградация
            pass
    
    def _load_chat_history(self):
        """Загружает историю чата из файла."""
        try:
            # Создаем директорию кэша, если она не существует
            if self.gui.cache_dir:
                os.makedirs(self.gui.cache_dir, exist_ok=True)
                
            history_file = os.path.join(self.gui.cache_dir, "chat_history.json")
            if os.path.exists(history_file):
                with open(history_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    # Заменяем текущую историю загруженной, но не дублируем при отрисовке
                    self.message_history = list(loaded)
                
                # Восстанавливаем историю в интерфейсе
                self._suppress_history_append = True
                try:
                    for msg in loaded:
                        self._add_message(
                            msg["sender"], 
                            msg["message"], 
                            msg["type"], 
                            timestamp=msg["timestamp"],
                            process_formatting=False  # Не обрабатываем форматирование при загрузке
                        )
                finally:
                    self._suppress_history_append = False
        except Exception as e:
            logger.error(f"Ошибка загрузки истории чата: {e}", exc_info=True)

    def _add_message(self, sender: str, message: str, msg_type: str = "user", 
                timestamp: Optional[float] = None, process_formatting: bool = True, extras: Optional[Dict[str, Any]] = None):
        """Добавляет сообщение в чат с поддержкой гиперссылок и форматирования."""
        if timestamp is None:
            timestamp = time.time()
        
        # Сохраняем в историю, если не в режиме восстановления
        if not self._suppress_history_append:
            entry = {
                "sender": sender,
                "message": message,
                "type": msg_type,
                "timestamp": timestamp
            }
            # Включаем дополнительные поля (метаданные, тональность, токены, противоречия)
            if extras and isinstance(extras, dict):
                # Только сериализуемые типы
                try:
                    safe_extras = json.loads(json.dumps(extras, ensure_ascii=False, default=str))
                except Exception:
                    safe_extras = {}
                    for k, v in extras.items():
                        try:
                            safe_extras[k] = json.loads(json.dumps(v, ensure_ascii=False, default=str))
                        except Exception:
                            safe_extras[k] = str(v)
                entry.update({"extras": safe_extras})
            self.message_history.append(entry)
            # Инкрементальное сохранение истории в файл
            self._save_history_incremental()
        
        # Ограничиваем размер истории
        if len(self.message_history) > 500:
            self.message_history = self.message_history[-500:]
        
        # Отображаем сообщение
        try:
            if not self.chat_display.winfo_exists():
                return
                
            self.chat_display.config(state=tk.NORMAL)
            # Гарантируем, что каждое сообщение начинается с новой строки
            try:
                last_char = self.chat_display.get("end-2c", "end-1c")
                if last_char not in ("\n", ""):
                    self.chat_display.insert(tk.END, "\n")
            except Exception:
                pass
            
            # Форматируем временную метку
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            self.chat_display.insert(tk.END, f"[{time_str}] ", "timestamp")
            
            # Добавляем отправителя
            tag = "user" if msg_type == "user" else msg_type
            self.chat_display.insert(tk.END, f"{sender}: ", tag)
            
            # Обрабатываем форматирование, если нужно
            if process_formatting:
                self._process_and_insert_formatted_message(_to_display_str(message))
            else:
                # Простое добавление сообщения без обработки форматирования
                self.chat_display.insert(tk.END, _to_display_str(message))
            
            # Новая строка
            self.chat_display.insert(tk.END, "\n\n")
            self.chat_display.config(state=tk.DISABLED)
            self.chat_display.see(tk.END)
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка добавления сообщения в чат: {e}", exc_info=True)

    def _save_history_incremental(self):
        """Безопасно сохраняет историю чата в chat_history.json после каждого добавления сообщения."""
        try:
            if not getattr(self.gui, 'cache_dir', None):
                return
            os.makedirs(self.gui.cache_dir, exist_ok=True)
            history_file = os.path.join(self.gui.cache_dir, "chat_history.json")
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(self.message_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Не удалось инкрементально сохранить историю чата: {e}")

    def _process_and_insert_formatted_message(self, message: str):
        """Обрабатывает и вставляет форматированное сообщение в чат."""
        try:
            if not self.chat_display.winfo_exists():
                return
                
            # Сохраняем позицию начала сообщения
            start_index = self.chat_display.index(tk.END)
            
            # Сначала вставляем все сообщение как обычный текст
            self.chat_display.insert(tk.END, message)
            
            # Теперь обрабатываем различные форматы
            current_pos = 0
            for match in self.formatting_pattern.finditer(message):
                start_idx = match.start()
                end_idx = match.end()
                matched_text = match.group(0)
                
                # Определяем тип форматирования
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
                
                # Вычисляем позиции в текстовом виджете
                start_tag_pos = f"{start_index}+{start_idx + current_pos}c"
                end_tag_pos = f"{start_index}+{start_idx + current_pos + len(matched_text)}c"
                content_start = f"{start_index}+{start_idx + current_pos + 2}c"
                content_end = f"{start_index}+{start_idx + current_pos + len(content) + 2}c"
                
                # Удаляем теги форматирования
                self.chat_display.delete(start_tag_pos, end_tag_pos)
                self.chat_display.insert(start_tag_pos, content)
                
                # Применяем форматирование к контенту
                self.chat_display.tag_add(format_type, content_start, content_end)
                
                # Корректируем позицию для следующей итерации
                current_pos -= len(matched_text) - len(content)
            
            # Обрабатываем гиперссылки
            # Сначала markdown-ссылки [текст](url) — они заменяются на текст и получают теги
            self._process_markdown_links(message, start_index)
            # Затем «сырые» URL, чтобы не дублировать обработку внутри markdown
            self._process_urls(message, start_index)
            
            # Обрабатываем эмодзи
            self._process_emojis(message, start_index)
            
            # Обрабатываем изображения
            self._process_images(message, start_index)
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки форматирования сообщения: {e}", exc_info=True)

    def _process_urls(self, message: str, start_index: str):
        """Обрабатывает и форматирует URL в сообщении."""
        try:
            if not self.chat_display.winfo_exists():
                return
                
            for match in self.url_pattern.finditer(message):
                start_idx = match.start()
                end_idx = match.end()
                url = match.group(0)
                
                # Вычисляем позиции в текстовом виджете
                url_start = f"{start_index}+{start_idx}c"
                url_end = f"{start_index}+{end_idx}c"
                
                # Применяем стиль к URL
                self.chat_display.tag_add("url", url_start, url_end)
                self.chat_display.tag_add(f"url_{url}", url_start, url_end)
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки URL: {e}", exc_info=True)

    def _process_markdown_links(self, message: str, start_index: str):
        """Находит markdown-ссылки вида [текст](url),
        заменяет их на «текст» и помечает диапазон тегами гиперссылки.
        """
        try:
            if not self.chat_display.winfo_exists():
                return

            offset_correction = 0
            for match in self.markdown_link_pattern.finditer(message):
                link_text = match.group(1)
                url = match.group(2)

                md_start = match.start()
                md_end = match.end()

                # Позиции в виджете с учётом уже произведённых замен
                start_pos = f"{start_index}+{md_start + offset_correction}c"
                end_pos = f"{start_index}+{md_end + offset_correction}c"

                # Заменяем весь markdown-конструкт на отображаемый текст
                self.chat_display.delete(start_pos, end_pos)
                self.chat_display.insert(start_pos, link_text)

                # Отметим новый диапазон гиперссылочным стилем и индивидуальным тегом
                link_end_pos = f"{start_pos}+{len(link_text)}c"
                self.chat_display.tag_add("url", start_pos, link_end_pos)
                self.chat_display.tag_add(f"url_{url}", start_pos, link_end_pos)

                # Скорректируем смещение для следующих совпадений
                offset_correction -= (md_end - md_start) - len(link_text)
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки markdown-ссылок: {e}", exc_info=True)

    def _process_emojis(self, message: str, start_index: str):
        """Обрабатывает и форматирует эмодзи в сообщении."""
        try:
            if not self.chat_display.winfo_exists():
                return
                
            for match in self.emoji_pattern.finditer(message):
                start_idx = match.start()
                end_idx = match.end()
                emoji = match.group(0)
                
                # Вычисляем позиции в текстовом виджете
                emoji_start = f"{start_index}+{start_idx}c"
                emoji_end = f"{start_index}+{end_idx}c"
                
                # Применяем стиль к эмодзи
                self.chat_display.tag_add("emoji", emoji_start, emoji_end)
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки эмодзи: {e}", exc_info=True)

    def _process_images(self, message: str, start_index: str):
        """Обрабатывает и форматирует изображения в сообщении."""
        try:
            if not self.chat_display.winfo_exists():
                return
                
            for match in self.image_pattern.finditer(message):
                url = match.group(1)
                start_idx = match.start(1)
                end_idx = match.end(1)
                
                # Вычисляем позиции в текстовом виджете
                url_start = f"{start_index}+{start_idx}c"
                url_end = f"{start_index}+{end_idx}c"
                
                # Заменяем URL на текст "[Изображение]"
                self.chat_display.delete(url_start, url_end)
                self.chat_display.insert(url_start, "[Изображение]")
                
                # Применяем стиль к тексту изображения
                self.chat_display.tag_add("url", url_start, f"{url_start}+10c")
                self.chat_display.tag_bind("url", "<Button-1>", lambda e, u=url: self._open_image(u))
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки изображений: {e}", exc_info=True)

    def _open_image(self, url: str):
        """Открывает изображение в браузере."""
        webbrowser.open(url)

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
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки клика по URL: {e}", exc_info=True)

    def _handle_url_enter(self, event):
        """Изменяет курсор при наведении на гиперссылку."""
        try:
            if not self.chat_display.winfo_exists():
                return
                
            self.chat_display.config(cursor="hand2")
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка изменения курсора (enter): {e}", exc_info=True)

    def _handle_url_leave(self, event):
        """Возвращает обычный курсор после ухода с гиперссылки."""
        try:
            if not self.chat_display.winfo_exists():
                return
                
            self.chat_display.config(cursor="")
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка изменения курсора (leave): {e}", exc_info=True)

    def _show_context_menu(self, event):
        """Показывает контекстное меню."""
        try:
            if not self.chat_display.winfo_exists():
                return
                
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _copy_selected(self):
        """Копирует выделенный текст в буфер обмена."""
        try:
            if not self.chat_display.winfo_exists():
                return
                
            selected_text = self.chat_display.selection_get()
            self.gui.root.clipboard_clear()
            self.gui.root.clipboard_append(selected_text)
        except tk.TclError:
            pass  # Нет выделенного текста
        except Exception as e:
            logger.error(f"Ошибка копирования выделенного текста: {e}", exc_info=True)

    def _copy_all(self):
        """Копирует весь текст чата в буфер обмена."""
        try:
            if not self.chat_display.winfo_exists():
                return
                
            self.chat_display.config(state=tk.NORMAL)
            all_text = self.chat_display.get("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)
            
            self.gui.root.clipboard_clear()
            self.gui.root.clipboard_append(all_text)
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка копирования всего текста: {e}", exc_info=True)

    def _clear_chat(self):
        """Очищает чат."""
        if messagebox.askyesno("Очистка чата", "Вы действительно хотите очистить чат?"):
            try:
                if not self.chat_display.winfo_exists():
                    return
                    
                self.chat_display.config(state=tk.NORMAL)
                self.chat_display.delete("1.0", tk.END)
                self.chat_display.config(state=tk.DISABLED)
                self.message_history = []
                
                # Добавляем приветственное сообщение
                welcome_msg = (
                    "Чат очищен.\n\n"
                    "Я - когнитивная система с поддержкой:\n"
                    "• Этического анализа\n"
                    "• Адаптации под пользователя\n"
                    "• Распределенных вычислений\n"
                    "• Управления знаниями\n\n"
                    "Задайте ваш первый вопрос или нажмите F1 для просмотра справки."
                )
                self._add_message("CogniFlex", welcome_msg, "system")
            except tk.TclError:
                # GUI уже уничтожен
                pass
            except Exception as e:
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
        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка вырезания текста: {e}", exc_info=True)

    def _copy_text(self, widget):
        """Копирует выделенный текст."""
        try:
            if not widget.winfo_exists():
                return
                
            selected_text = widget.selection_get()
            self.gui.root.clipboard_clear()
            self.gui.root.clipboard_append(selected_text)
        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка копирования текста: {e}", exc_info=True)

    def _paste_text(self, widget):
        """Вставляет текст из буфера обмена."""
        try:
            if not widget.winfo_exists():
                return
                
            clipboard_text = self.gui.root.clipboard_get()
            widget.insert(tk.INSERT, clipboard_text)
        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка вставки текста: {e}", exc_info=True)

    def _on_import_document(self):
        """Открывает диалог выбора файла и импортирует документ через ImportPipeline.
        Поддерживает TXT, PDF, EPUB. Импорт выполняется в фоне; результат отображается в чате.
        """
        try:
            file_path = filedialog.askopenfilename(
                title="Выберите документ для импорта",
                filetypes=[
                    ("Документы", "*.txt *.pdf *.epub"),
                    ("Текстовые файлы", "*.txt"),
                    ("PDF", "*.pdf"),
                    ("EPUB", "*.epub"),
                    ("Все файлы", "*.*"),
                ],
            )
            if not file_path:
                return

            # Ленивая инициализация пайплайна
            if self._import_pipeline is None:
                try:
                    brain = getattr(self.gui, 'brain', None)
                    self._import_pipeline = ImportPipeline(brain, chunk_tokens=512, overlap_tokens=64)
                except Exception as e:
                    messagebox.showerror("Импорт", f"Не удалось инициализировать импорт: {e}")
                    return

            # Сообщение о старте импорта
            self._add_message("CogniFlex", f"Импортирую документ: {os.path.basename(file_path)}...", "system")

            def worker():
                try:
                    imported = self._import_pipeline.import_path(file_path)
                    # Безопасно обновляем GUI
                    def on_done():
                        try:
                            name = getattr(imported, "name", os.path.basename(file_path))
                            segs = getattr(imported, "segments", []) or []
                            seg_count = len(segs)
                            preview = ""
                            if segs:
                                first = segs[0]
                                # segment may be dict or object; try common access
                                text = first.get("text") if isinstance(first, dict) else getattr(first, "text", str(first))
                                if text:
                                    preview = (text[:400] + ("…" if len(text) > 400 else ""))
                            msg = (
                                f"Импорт завершен: '{name}'. Найдено сегментов: {seg_count}.\n"
                                + (f"Предпросмотр первого сегмента:\n{preview}" if preview else "")
                            )
                            self._remove_last_message()  # убрать индикатор
                            self._add_message("CogniFlex", msg, "system")
                        except Exception as ie:
                            self._add_message("CogniFlex", f"Ошибка отображения результата импорта: {ie}", "system")

                    if hasattr(self.gui, "root") and self.gui.root:
                        self.gui.root.after(0, on_done)
                except Exception as e:
                    def on_fail():
                        try:
                            self._remove_last_message()
                        except Exception:
                            pass
                        self._add_message("CogniFlex", f"Ошибка импорта: {e}", "system")
                    if hasattr(self.gui, "root") and self.gui.root:
                        self.gui.root.after(0, on_fail)

            threading.Thread(target=worker, name="ChatImport", daemon=True).start()
        except Exception as e:
            logger.error(f"Ошибка импорта: {e}", exc_info=True)

    def _on_enter_pressed(self, event):
        """Отправка по Enter. Shift+Enter и Ctrl+Enter — новая строка."""
        try:
            # Bit masks: Shift=0x0001, Control=0x0004
            if (event.state & 0x0001) or (event.state & 0x0004):
                # Разрешаем стандартную вставку новой строки
                return None
            # Без модификаторов — отправляем
            self._send_message()
            return "break"
        except Exception:
            return None

    def _on_history_up(self, event):
        """Обрабатывает нажатие стрелки вверх для просмотра истории."""
        if self.message_history and self.history_index < len(self.message_history) - 1:
            self.history_index += 1
            msg = self.message_history[-(self.history_index + 1)]
            if msg["type"] == "user":
                self.input_text.delete("1.0", tk.END)
                self.input_text.insert("1.0", msg["message"])
        return "break"

    def _on_history_down(self, event):
        """Обрабатывает нажатие стрелки вниз для просмотра истории."""
        if self.history_index > 0:
            self.history_index -= 1
            if self.history_index >= 0:
                msg = self.message_history[-(self.history_index + 1)]
                if msg["type"] == "user":
                    self.input_text.delete("1.0", tk.END)
                    self.input_text.insert("1.0", msg["message"])
            else:
                self.input_text.delete("1.0", tk.END)
        return "break"

    def _on_chat_select_all(self, event):
        """Выделяет весь текст в области отображения чата."""
        try:
            self.chat_display.tag_add(tk.SEL, "1.0", tk.END)
            self.chat_display.mark_set(tk.INSERT, "1.0")
            self.chat_display.see(tk.INSERT)
            return "break"
        except Exception:
            return None

    def _send_message(self):
        """Отправляет сообщение из поля ввода."""
        message = self.input_text.get("1.0", tk.END).strip()
        if not message:
            return
            
        # Добавляем сообщение пользователя
        self._add_message("Вы", message, "user")
        
        # Очищаем поле ввода
        self.input_text.delete("1.0", tk.END)
        
        # Добавляем запрос в очередь
        request_id = f"req_{int(time.time())}_{random.randint(1000, 9999)}"
        self.pending_requests.add(request_id)
        self.request_queue.put({
            "message": message,
            "request_id": request_id
        })

    def _remove_last_message(self):
        """Удаляет последнее сообщение из чата."""
        try:
            if not self.chat_display.winfo_exists():
                return
                
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("end-2l", "end")
            self.chat_display.config(state=tk.DISABLED)
            
            # Удаляем из истории
            if self.message_history:
                self.message_history.pop()
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка удаления последнего сообщения: {e}", exc_info=True)
    
    def update_theme(self):
        """Обновляет тему чата при смене темы интерфейса."""
        if not hasattr(self, 'chat_display') or not self.chat_display or not self.chat_display.winfo_exists():
            return
            
        try:
            # Обновляем цвета
            self.chat_display.config(
                bg=self.gui.colors['card-bg'],
                fg=self.gui.colors['text']
            )
            
            # Обновляем теги
            self.chat_display.tag_configure("user", foreground=self.gui.colors['primary'])
            self.chat_display.tag_configure("system", foreground=self.gui.colors['text'])
            self.chat_display.tag_configure("reasoning", foreground=self.gui.colors['text-muted'])
            self.chat_display.tag_configure("timestamp", foreground=self.gui.colors['text-muted'])
            self.chat_display.tag_configure("url", foreground=self.gui.colors['primary'])
            
            # Перерисовываем чат
            self._redraw_chat()
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка обновления темы чата: {e}", exc_info=True)

    def _redraw_chat(self):
        """Перерисовывает чат с учетом новой темы."""
        if not self.message_history:
            return
            
        try:
            if not self.chat_display.winfo_exists():
                return
                
            # Сохраняем текущую позицию прокрутки
            current_pos = self.chat_display.yview()
            
            # Очищаем чат
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            
            # Перерисовываем сообщения
            for msg in self.message_history:
                self._add_message(
                    msg["sender"], 
                    msg["message"], 
                    msg["type"], 
                    timestamp=msg["timestamp"],
                    process_formatting=False
                )
            
            # Восстанавливаем позицию прокрутки
            self.chat_display.yview_moveto(current_pos[0])
            self.chat_display.config(state=tk.DISABLED)
        except tk.TclError:
            # GUI уже уничтожен
            pass
        except Exception as e:
            logger.error(f"Ошибка перерисовки чата: {e}", exc_info=True)