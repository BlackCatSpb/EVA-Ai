"""
Графический интерфейс пользователя для CogniFlex - основной модуль с полной функциональностью
"""
import os
import sys
import logging
import threading
import queue
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use('TkAgg')

# Импортируем утилиты и настройки
from .settings import load_settings, save_settings

logger = logging.getLogger("cogniflex.gui.core")

class CogniFlexGUI:
    """Полнофункциональный графический интерфейс для CogniFlex с поддержкой всех расширенных функций."""
    
    def __init__(self, brain=None, integrator=None, cache_dir: Optional[str] = None):
        logger.debug("Инициализация графического интерфейса...")
        self.brain = brain
        self.integrator = integrator  # Новый интегратор системы
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cogniflex_gui_cache")
        self.cache_dir = os.path.abspath(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.settings = load_settings(os.path.join(self.cache_dir, "gui_settings.json"))
        # Настройка троттлинга уведомлений (по умолчанию 30 секунд)
        self.notification_throttle_seconds = self.settings.get("gui", {}).get("notification_throttle_seconds", 30)
        
        self.theme_colors = {
            "light": {
                'bg': '#f0f0f0', 'card-bg': 'white', 'text': '#333333',
                'text-muted': '#666666', 'border': '#cccccc', 'primary': '#0078d7',
                'success': '#28a745', 'danger': '#dc3545', 'warning': '#ffc107',
            },
            "dark": {
                'bg': '#1e1e1e', 'card-bg': '#2d2d2d', 'text': '#e0e0e0',
                'text-muted': '#a0a0a0', 'border': '#444444', 'primary': '#0091ff',
                'success': '#4caf50', 'danger': '#f44336', 'warning': '#ff9800',
            }
        }
        
        self.theme = self.settings.get("gui", {}).get("theme", "light")
        self.colors = self.theme_colors[self.theme]
        
        self.gui_queue = queue.Queue()
        self.current_view = "chat"
        self.system_status = "initializing"
        self.running = False
        self.stop_event = threading.Event()
        self.compact_mode = self.settings.get("gui", {}).get("compact_mode", False)
        # Текущее состояние загрузки моделей для индикатора
        self.model_loading_state = {
            "active": False,
            "model_id": None,
            "name": None,
            "progress": 0,
            "error": None,
            "action": "load",
        }
        
        self.dashboard_data = {}
        self.last_notification_times = {}
        self.gui_queue_job = None  # ID запланированного after для очереди GUI
        self.active_notifications = []  # Активные уведомления
        
        # Добавляем логгирование для чата
        self.chat_logger = logging.getLogger("cogniflex.gui.chat")
        self.chat_logger.info("Инициализирован чат-логгер")

        self.root = None
        self.content_area = None
        self.update_job = None
        
        self.chat_module = None
        self.analytics_module = None
        self.knowledge_module = None
        self.contradiction_module = None
        self.memory_module = None
        self.learning_module = None
        self.settings_module = None
        self.neuromorphic_module = None
        
        logger.info("GUI инициализирован")

    def process_query_via_integrator(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Обработка запроса через интегратор системы.

        Args:
            query: Текст запроса
            context: Дополнительный контекст

        Returns:
            Dict с результатом обработки
        """
        if not self.integrator:
            logger.warning("Интегратор не доступен, использую прямую обработку через brain")
            return self._fallback_query_processing(query, context)

        try:
            logger.info(f"Отправка запроса через интегратор: '{query[:50]}...'")

            # Публикуем событие query_received через событийную шину интегратора
            query_data = {
                'query': query,
                'context': context or {},
                'source': 'gui',
                'timestamp': time.time()
            }

            # Используем событийную шину интегратора для публикации события
            if hasattr(self.integrator, 'event_bus'):
                self.integrator.event_bus.trigger('query_received', query_data)

                # Ожидаем ответ (асинхронно через события)
                return self._wait_for_response(query_data)

            else:
                # Fallback: прямой вызов
                return self.integrator.process_query(query, context)

        except Exception as e:
            logger.error(f"Ошибка обработки запроса через интегратор: {e}")
            return self._fallback_query_processing(query, context)

    def _wait_for_response(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ожидание ответа на запрос через событийную шину.
        """
        timeout = 30.0  # 30 секунд таймаут
        start_time = time.time()
        request_id = f"gui_{int(start_time)}"

        # Создаем объект для ожидания ответа
        response_received = threading.Event()
        response_data = {}

        def on_response_received(data):
            nonlocal response_data
            if data.get('request_id') == request_id or 'response' in data:
                response_data = data
                response_received.set()

        # Подписываемся на событие response_generated
        if hasattr(self.integrator, 'event_bus'):
            self.integrator.event_bus.subscribe(
                'response_generated',
                on_response_received,
                priority=10
            )

        # Ждем ответ
        if response_received.wait(timeout):
            return response_data
        else:
            return {
                'status': 'timeout',
                'error': 'Превышено время ожидания ответа',
                'response': 'Извините, обработка запроса заняла слишком много времени.'
            }

    def _fallback_query_processing(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Запасной вариант обработки запроса напрямую через brain.
        """
        try:
            if not self.brain:
                return {
                    'status': 'error',
                    'error': 'Система недоступна',
                    'response': 'Извините, система временно недоступна.'
                }

            # Пытаемся использовать response_generator
            if hasattr(self.brain, 'process_query'):
                result = self.brain.process_query(query, context)
                if result:
                    self.gui_queue.put(lambda: self._add_message("CogniFlex", result.get("text", "Ошибка обработки"), 0))
                else:
                    self.gui_queue.put(lambda: self._add_message("CogniFlex", "Пустой ответ от системы", 0))
                return {
                    'status': 'ok',
                    'response': result.get("text", result.get("response", "")) if isinstance(result, dict) else str(result)
                }
            else:
                # Fallback
                self.gui_queue.put(lambda: self._add_message("CogniFlex", "Система обработки запросов недоступна", 0))
                return {
                    'status': 'error',
                    'error': 'No brain available',
                    'response': 'Система временно недоступна.'
                }

        except Exception as e:
            logger.error(f"Ошибка fallback обработки: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'response': 'Произошла ошибка при обработке запроса.'
            }

    def get_system_status_via_integrator(self) -> Dict[str, Any]:
        """
        Получение статуса системы через интегратор.
        """
        if self.integrator and hasattr(self.integrator, 'get_system_health'):
            return self.integrator.get_system_health()
        elif self.integrator and hasattr(self.integrator, 'get_system_stats'):
            return self.integrator.get_system_stats()
        else:
            # Fallback через brain
            return self._get_system_status_fallback()

    def _get_system_status_fallback(self) -> Dict[str, Any]:
        """Запасной вариант получения статуса системы."""
        try:
            if not self.brain:
                return {'status': 'disconnected'}

            status = {'status': 'unknown'}

            if hasattr(self.brain, 'running') and self.brain.running:
                status['status'] = 'active'
            elif hasattr(self.brain, 'components') and self.brain.components:
                status['status'] = 'active'
            else:
                status['status'] = 'disconnected'

            # Добавляем информацию о компонентах
            if hasattr(self.brain, 'components'):
                status['components_count'] = len(self.brain.components)

            return status

        except Exception as e:
            logger.error(f"Ошибка получения статуса системы: {e}")
            return {'status': 'error', 'error': str(e)}

    def start_self_dialog_via_integrator(self):
        """Запуск самодиалога через интегратор."""
        try:
            if self.integrator and hasattr(self.integrator, 'start_self_dialog'):
                self.integrator.start_self_dialog()
                self.show_toast("Самодиалог запущен", "info")
            else:
                self.show_toast("Самодиалог недоступен", "warning")
        except Exception as e:
            logger.error(f"Ошибка запуска самодиалога: {e}")
            self.show_toast(f"Ошибка самодиалога: {e}", "error")

    def optimize_system_via_integrator(self):
        """Оптимизация системы через интегратор."""
        try:
            if self.integrator and hasattr(self.integrator, 'optimize_system'):
                self.integrator.optimize_system()
                self.show_toast("Оптимизация системы запущена", "info")
            else:
                self.show_toast("Оптимизация недоступна", "warning")
        except Exception as e:
            logger.error(f"Ошибка оптимизации системы: {e}")
            self.show_toast(f"Ошибка оптимизации: {e}", "error")

    def _init_modules(self):
        """Инициализирует модули GUI с улучшенной обработкой ошибок и приоритетом chat модуля."""
        logger.debug("DEBUG: _init_modules() вызван!")
        if not self.content_area:
            logger.warning("Контентная область не создана, создаём базовую")
            if self.root:
                self.content_area = ttk.Frame(self.root)
                self.content_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            else:
                logger.error("Невозможно инициализировать модули: root окно не создано")
                return
        
        logger.debug(f"DEBUG: Начинаем инициализацию модулей GUI")
        logger.info("Начинаем инициализацию модулей GUI")
        
        # Приоритизируем chat модуль - инициализируем его первым
        chat_initialized = False

        # Сначала пытаемся инициализировать chat модуль отдельно
        try:
            logger.info(" Инициализация chat модуля с приоритетом...")
            from .chat_module import ChatModule

            # Проверяем, что класс существует
            if hasattr(ChatModule, '__init__'):
                self.chat_module = ChatModule(self)
                chat_initialized = True
                logger.info("[OK] Chat модуль инициализирован успешно (приоритет)")
            else:
                logger.warning("Класс ChatModule не найден")

        except ImportError as e:
            logger.warning(f"Не удалось импортировать ChatModule: {e}")
        except Exception as e:
            logger.error(f"Ошибка инициализации ChatModule: {e}", exc_info=True)

        # Если chat не инициализирован, пробуем fallback
        if not chat_initialized:
            try:
                logger.info("Попытка fallback инициализации chat модуля...")
                module = __import__('cogniflex.gui.chat_module', fromlist=['ChatModule'])
                module_class = getattr(module, 'ChatModule')
                self.chat_module = module_class(self)
                chat_initialized = True
                logger.info("[OK] Chat модуль инициализирован через fallback")
            except Exception as e:
                logger.critical(f"Не удалось инициализировать ChatModule даже через fallback: {e}")

        # Теперь инициализируем только необходимые модули
        module_map = {
            "memory": ("cogniflex.gui.memory_module", "MemoryModule"),
        }

        for name, (module_path, class_name) in module_map.items():
            try:
                logger.info(f"Инициализация модуля: {name}")
                module = __import__(module_path, fromlist=[None])
                
                if not hasattr(module, class_name):
                    logger.warning(f"Класс {class_name} не найден в модуле {module_path}")
                    continue

                module_class = getattr(module, class_name)
                instance = module_class(self)
                setattr(self, f"{name}_module", instance)
                logger.info(f"[OK] Модуль '{name}' инициализирован успешно")

            except ImportError as e:
                logger.error(f"Не удалось импортировать модуль '{name}': {e}")
            except Exception as e:
                logger.error(f"Ошибка инициализации модуля '{name}': {e}")

        # Финальная проверка chat модуля
        if not hasattr(self, 'chat_module') or self.chat_module is None:
            logger.error("КРИТИЧЕСКАЯ ОШИБКА: Chat модуль не инициализирован после всех попыток!")
            # Создаем заглушку для chat модуля
            try:
                logger.info("Создание заглушки для chat модуля...")

                class ChatModuleStub:
                    def __init__(self, gui):
                        self.gui = gui
                        self.message_history = []
                        logger.warning("Chat модуль заменен заглушкой - функциональность ограничена")

                    def activate(self):
                        # Создаем простой интерфейс
                        for widget in self.gui.content_area.winfo_children():
                            widget.destroy()

                        from tkinter import ttk
                        frame = ttk.Frame(self.gui.content_area)
                        frame.pack(fill="both", expand=True, padx=20, pady=20)

                        ttk.Label(frame, text="⚠️ Chat модуль временно недоступен",
                                font=("Segoe UI", 14, "bold")).pack(pady=(0, 10))

                        ttk.Label(frame, text="Chat работает только с фрактальным хранилищем.\n"
                                "Проверьте, что memory_manager инициализирован.",
                                wraplength=400, justify="center").pack(pady=(0, 20))

                        ttk.Button(frame, text="Перезагрузить систему",
                                 command=lambda: self.gui._reboot_system()).pack()

                        logger.warning("Chat модуль заменен заглушкой")

                    def deactivate(self):
                        """Деактивирует заглушку модуля чата"""
                        try:
                            logger.debug("ChatModuleStub деактивирован")
                        except Exception as e:
                            logger.debug(f"Ошибка деактивации ChatModuleStub: {e}")

                self.chat_module = ChatModuleStub(self)
                logger.info("[OK] Заглушка для chat модуля создана")

            except Exception as e:
                logger.critical(f"Не удалось создать даже заглушку для chat модуля: {e}")

        # Проверяем успешность инициализации
        final_chat_status = hasattr(self, 'chat_module') and self.chat_module is not None
        logger.info(f"Итог инициализации модулей: chat={'успешно' if final_chat_status else 'не удалось'}")

        self._switch_view("chat")

    def _create_styles(self):
        style = ttk.Style()
        self.theme = self.settings.get("gui", {}).get("theme", "light")
        self.colors = self.theme_colors[self.theme]

        if self.theme == "dark":
            style.theme_use("clam")
            style.configure(".", background=self.colors['bg'], foreground=self.colors['text'], fieldbackground=self.colors['card-bg'], bordercolor=self.colors['border'])
            style.configure("TFrame", background=self.colors['bg'])
            style.configure("TLabel", background=self.colors['bg'], foreground=self.colors['text'])
            style.configure("TButton", background=self.colors['card-bg'], foreground=self.colors['text'], borderwidth=1)
            style.map("TButton", background=[("active", self.colors['primary'])])
        else:
            style.theme_use("default")
            style.configure(".", background=self.colors['bg'], foreground=self.colors['text'], fieldbackground=self.colors['card-bg'])

        # Navigation button styles
        style.configure("Nav.TButton", padding=(10, 5), relief=tk.FLAT)
        style.configure("NavActive.TButton", padding=(10, 5), relief=tk.FLAT, background=self.colors['primary'])
        style.map("NavActive.TButton", background=[("active", self.colors['primary'])])
        
        # Notebook tab styles
        style.configure("TNotebook", background=self.colors['bg'])
        style.configure("TNotebook.Tab", padding=(10, 5), background=self.colors['card-bg'], foreground=self.colors['text'])
        style.map("TNotebook.Tab", background=[("selected", self.colors['primary'])], foreground=[("selected", "white")])

        if self.root:
            self.root.configure(bg=self.colors['bg'])

    def _create_interface(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        self._create_navbar()
        self._create_notebook()
        self._create_status_bar()

    def _create_navbar(self):
        """Создаёт упрощённую панель навигации (без кнопок переключения вкладок)."""
        navbar = ttk.Frame(self.main_container, height=40)
        navbar.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(navbar, text="CogniFlex", font=("Segoe UI", 14, "bold"), foreground=self.colors['primary']).pack(side=tk.LEFT, padx=10)
        right_frame = ttk.Frame(navbar)
        right_frame.pack(side=tk.RIGHT)
        ttk.Button(right_frame, text="Перезагрузить", command=self._reboot_system).pack(side=tk.LEFT, padx=5)
        ttk.Button(right_frame, text="Горячая перезагрузка", command=self._soft_reload).pack(side=tk.LEFT, padx=5)


    def _create_notebook(self):
        """Создаёт Notebook с 3 вкладками: Чат, Память, Система."""
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)
        
        self.tabs = {}
        tab_names = [
            ("chat", "Чат"),
            ("memory", "Память"),
            ("system", "Система")
        ]
        
        for tab_id, tab_title in tab_names:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=tab_title)
            self.tabs[tab_id] = frame
        
        self.content_area = self.tabs.get("chat")
        self.tab_order = [tab_id for tab_id, _ in tab_names]

    def _on_tab_changed(self, event):
        """Обрабатывает событие переключения вкладки пользователем."""
        try:
            selected_index = self.notebook.index(self.notebook.select())
            if 0 <= selected_index < len(self.tab_order):
                view_id = self.tab_order[selected_index]
                self._switch_view(view_id)
        except Exception as e:
            logger.debug(f"Ошибка при обработке смены вкладки: {e}")

    def _update_nav_visual_state(self, active_view_id: str):
        """Обновляет визуальное состояние кнопок навигации (теперь для вкладок)."""
        # ttk.Notebook сам управляет подсветкой активной вкладки
        pass

    def _create_status_bar(self):
        self.status_bar = ttk.Frame(self.root, height=30)
        self.status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.status_indicator = tk.Canvas(self.status_bar, width=15, height=15, highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 5))
        self.status_indicator.create_oval(2, 2, 13, 13, fill=self.colors['warning'], tags="indicator")
        self.connection_status = ttk.Label(self.status_bar, text="Соединение: инициализация...")
        self.connection_status.pack(side=tk.LEFT)
        # Компактный индикатор загрузки моделей рядом со статусом
        self.model_load_label = ttk.Label(self.status_bar, text="")
        self.model_load_label.pack(side=tk.LEFT, padx=(10, 0))
        self.timestamp_label = ttk.Label(self.status_bar, text="--:--:--")
        self.timestamp_label.pack(side=tk.RIGHT, padx=10)
        metrics_frame = ttk.Frame(self.status_bar)
        metrics_frame.pack(side=tk.RIGHT, padx=10)
        ttk.Label(metrics_frame, text="CPU:").pack(side=tk.LEFT)
        self.cpu_label = ttk.Label(metrics_frame, text="0%")
        self.cpu_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(metrics_frame, text="RAM:").pack(side=tk.LEFT)
        self.memory_label = ttk.Label(metrics_frame, text="0%")
        self.memory_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(metrics_frame, text="Противоречия:").pack(side=tk.LEFT)
        self.contradictions_label = ttk.Label(metrics_frame, text="0")
        self.contradictions_label.pack(side=tk.LEFT, padx=(0, 10))
        # Доп. метрики из CoreBrain: кэш и I/O
        ttk.Label(metrics_frame, text="HitRate:").pack(side=tk.LEFT)
        self.hit_rate_label = ttk.Label(metrics_frame, text="0.0%")
        self.hit_rate_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(metrics_frame, text="CacheUtil:").pack(side=tk.LEFT)
        self.cache_util_label = ttk.Label(metrics_frame, text="0.0%")
        self.cache_util_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(metrics_frame, text="DiskEntries:").pack(side=tk.LEFT)
        self.disk_entries_label = ttk.Label(metrics_frame, text="0")
        self.disk_entries_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(metrics_frame, text="IOtokens:").pack(side=tk.LEFT)
        self.io_tokens_label = ttk.Label(metrics_frame, text="0")
        self.io_tokens_label.pack(side=tk.LEFT)

    def on_close(self):
        if messagebox.askyesno("Подтверждение", "Вы действительно хотите выйти?"):
            self.stop()

    def _switch_view(self, view_id: str):
        logger.debug(f"Переключение на представление: {view_id}")
        
        if not self.tabs or view_id not in self.tabs:
            logger.warning(f"Попытка переключения на несуществующую вкладку: {view_id}")
            return

        self.content_area = self.tabs[view_id]
        
        try:
            tab_index = self.tab_order.index(view_id)
            self.notebook.select(tab_index)
        except (ValueError, IndexError) as e:
            logger.warning(f"Не удалось переключить вкладку {view_id}: {e}")

        try:
            prev_view = getattr(self, "current_view", None)
            if prev_view and prev_view != view_id:
                prev_module = getattr(self, f"{prev_view}_module", None)
                if prev_module and hasattr(prev_module, "deactivate"):
                    try:
                        prev_module.deactivate()
                    except Exception as e:
                        logger.warning(f"Error deactivating previous module: {e}")
        except Exception as e:
            logger.warning(f"Error in _switch_view: {e}")

        for widget in self.content_area.winfo_children():
            widget.destroy()

        if view_id == "memory":
            self.memory_tab_instance = MemoryTab(self)
            self.memory_tab_instance.activate()
            self.current_view = view_id
            return
        elif view_id == "system":
            self.system_tab_instance = SystemTab(self)
            self.system_tab_instance.activate()
            self.current_view = view_id
            return
        
        module = getattr(self, f"{view_id}_module", None)
        if module and hasattr(module, 'activate'):
            self.chat_logger.info(f"Активация модуля: {view_id}")
            module.activate()
        else:
            ttk.Label(self.content_area, text=f"Модуль '{view_id}' недоступен.").pack()
        self.current_view = view_id

    def _schedule_update(self):
        try:
            if not self.running:
                return
            if not self.root:
                return
            self._update_interface()
            interval = self.settings.get("gui", {}).get("auto_update_interval", 5000)
            self.update_job = self.root.after(interval, self._schedule_update)
        except Exception as e:
            try:
                self.chat_logger.debug(f"_schedule_update error: {e}")
            except Exception as e2:
                logger.warning(f"Error logging _schedule_update error: {e2}")

    def _start_background_services(self):
        self._schedule_update()
        self._process_gui_queue()
        # Подписка на события загрузки модели (через EventSystem или фолбэк)
        try:
            if self.brain:
                handler = self._handle_model_load_event
                # Предпочтительно через шину событий
                if hasattr(self.brain, 'events') and self.brain.events and hasattr(self.brain.events, 'on'):
                    try:
                        self.brain.events.on('model_load', handler)
                        self.chat_logger.info("GUI подписан на события model_load через EventSystem")
                        # Подписка на готовность моделей
                        try:
                            self.brain.events.on('models_ready', lambda data=None: self.gui_queue.put(self._handle_models_ready_event))
                            self.chat_logger.info("GUI подписан на событие models_ready")
                        except Exception:
                            pass
                        # Подписка на запрос перезагрузки GUI
                        try:
                            self.brain.events.on('request_gui_reload', lambda data=None: self.gui_queue.put(self.reload))
                            self.chat_logger.info("GUI подписан на событие request_gui_reload")
                        except Exception:
                            pass
                    except Exception:
                        # Фолбэк на список колбэков
                        if not hasattr(self.brain, 'on_model_load'):
                            setattr(self.brain, 'on_model_load', [])
                        self.brain.on_model_load.append(handler)
                        self.chat_logger.info("GUI подписан на события model_load через on_model_load")
                        # Фолбэк подписки на models_ready
                        try:
                            if not hasattr(self.brain, 'on_models_ready'):
                                setattr(self.brain, 'on_models_ready', [])
                            self.brain.on_models_ready.append(lambda data=None: self.gui_queue.put(self._handle_models_ready_event))
                        except Exception:
                            pass
                else:
                    if not hasattr(self.brain, 'on_model_load'):
                        setattr(self.brain, 'on_model_load', [])
                    self.brain.on_model_load.append(handler)
                    self.chat_logger.info("GUI подписан на события model_load (fallback)")
                    # Фолбэк подписки на models_ready
                    try:
                        if not hasattr(self.brain, 'on_models_ready'):
                            setattr(self.brain, 'on_models_ready', [])
                        self.brain.on_models_ready.append(lambda data=None: self.gui_queue.put(self._handle_models_ready_event))
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Не удалось подписаться на события model_load: {e}")

        # Фолбэк: если нет шины событий, можно дергать reload() из ядра через прямой атрибут
        try:
            if self.brain and not (hasattr(self.brain, 'events') and self.brain.events):
                setattr(self.brain, 'request_gui_reload', lambda reason=None: self.gui_queue.put(self.reload))
        except Exception:
            pass

    def reload(self):
        """Горячая перезагрузка GUI: обновление стилей/настроек и повторная инициализация модулей."""
        try:
            if not self.root or not self.running:
                return
            self.chat_logger.info("Выполняется горячая перезагрузка GUI...")
            # Перечитать настройки и обновить тему/цвета
            try:
                self.settings = load_settings(os.path.join(self.cache_dir, "gui_settings.json"))
            except Exception:
                pass
            self.theme = self.settings.get("gui", {}).get("theme", self.theme)
            self.colors = self.theme_colors.get(self.theme, self.colors)
            # Применить стили повторно
            try:
                self._create_styles()
            except Exception:
                pass
            # Переинициализировать модули (сохраняя текущий view)
            cur_view = getattr(self, 'current_view', 'chat')
            try:
                if self.content_area:
                    for w in self.content_area.winfo_children():
                        w.destroy()
                self._init_modules()
                # Вернуться к предыдущему виду
                self._switch_view(cur_view)
            except Exception:
                pass
            self._update_interface()
            self.show_toast("GUI обновлён (soft-reload)", "info")
            self.chat_logger.info("Горячая перезагрузка GUI завершена")
        except tk.TclError:
            pass
        except Exception as e:
            self.chat_logger.warning(f"Ошибка горячей перезагрузки GUI: {e}")

        if not self.running: return
        self._update_interface()
        if self.root:
            self.update_job = self.root.after(self.settings.get("gui", {}).get("auto_update_interval", 5000), self._schedule_update)

    def _update_interface(self):
        if not self.root or not self.running: return
        
        try:
            # Получаем данные дашборда через интегратор
            if self.integrator and hasattr(self.integrator, 'get_system_stats'):
                self.dashboard_data = self.integrator.get_system_stats()
            elif self.brain and hasattr(self.brain, 'get_system_dashboard_data'):
                self.dashboard_data = self.brain.get_system_dashboard_data()
            
            # Безопасно обновляем снимки ресурсов и статистику кэша
            self.resource_snapshot = {}
            self.cache_stats = {}
            try:
                if self.brain and hasattr(self.brain, 'get_resource_snapshot'):
                    self.resource_snapshot = self.brain.get_resource_snapshot() or {}
            except Exception:
                self.resource_snapshot = {}
            try:
                if self.brain and hasattr(self.brain, 'get_cache_stats'):
                    self.cache_stats = self.brain.get_cache_stats() or {}
            except Exception:
                self.cache_stats = {}
            
            if hasattr(self, 'timestamp_label') and self.timestamp_label:
                self.timestamp_label.config(text=datetime.now().strftime("%H:%M:%S"))
            self._update_status_indicator()
            self._update_system_metrics()
            self._handle_notifications()

            active_module = getattr(self, f"{self.current_view}_module", None)
            if active_module and hasattr(active_module, 'update'):
                active_module.update()

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обновления интерфейса: {e}", exc_info=True)

    def _update_system_metrics(self):
        metrics = self.dashboard_data.get('metrics', {})
        # Поддерживаем как доли (0..1), так и проценты (0..100)
        cpu = metrics.get('cpu_usage', 0.0)
        mem = metrics.get('memory_usage', 0.0)
        try:
            cpu_val = float(cpu)
            if cpu_val <= 1.5:
                cpu_val *= 100.0
        except Exception:
            cpu_val = 0.0
        try:
            mem_val = float(mem)
            if mem_val <= 1.5:
                mem_val *= 100.0
        except Exception:
            mem_val = 0.0
        if hasattr(self, 'cpu_label') and self.cpu_label:
            self.cpu_label.config(text=f"{cpu_val:.1f}%")
        if hasattr(self, 'memory_label') and self.memory_label:
            self.memory_label.config(text=f"{mem_val:.1f}%")
        contradiction_stats = self.dashboard_data.get('contradiction_stats', {})
        if hasattr(self, 'contradictions_label') and self.contradictions_label:
            self.contradictions_label.config(text=str(contradiction_stats.get('total', 0)))
        # Обновляем метрики кэша и I/O из CoreBrain API (если доступны)
        try:
            hit_rate = float(self.cache_stats.get('hit_rate', 0.0)) * (100.0 if self.cache_stats.get('hit_rate', 0.0) <= 1.5 else 1.0)
        except Exception:
            hit_rate = 0.0
        try:
            util = float(self.cache_stats.get('cache_utilization_percent', 0.0))
            if util <= 1.5:
                util *= 100.0
        except Exception:
            util = 0.0
        try:
            disk_entries = int(self.cache_stats.get('disk_stats', {}).get('entries', 0))
        except Exception:
            disk_entries = 0
        try:
            io_tokens = float(self.resource_snapshot.get('io_tokens', 0.0))
        except Exception:
            io_tokens = 0.0
        # Отображаем
        if hasattr(self, 'hit_rate_label'):
            self.hit_rate_label.config(text=f"{hit_rate:.1f}%")
        if hasattr(self, 'cache_util_label'):
            self.cache_util_label.config(text=f"{util:.1f}%")
        if hasattr(self, 'disk_entries_label'):
            self.disk_entries_label.config(text=str(disk_entries))
        if hasattr(self, 'io_tokens_label'):
            # Компактный формат без дробной части для больших значений
            try:
                if io_tokens >= 1_000_000_000:
                    io_disp = f"{io_tokens/1_000_000_000:.1f}G"
                elif io_tokens >= 1_000_000:
                    io_disp = f"{io_tokens/1_000_000:.1f}M"
                elif io_tokens >= 1_000:
                    io_disp = f"{io_tokens/1_000:.1f}K"
                else:
                    io_disp = f"{int(io_tokens)}"
            except Exception:
                io_disp = "0"
            self.io_tokens_label.config(text=io_disp)

    def _update_status_indicator(self):
        """Обновляет индикатор статуса соединения с улучшенной логикой."""
        try:
            # Более надежная проверка статуса brain
            brain_active = False
            if self.brain:
                # Проверяем несколько признаков активности brain
                brain_active = (
                    hasattr(self.brain, 'running') and self.brain.running
                ) or (
                    hasattr(self.brain, 'components') and self.brain.components
                ) or (
                    hasattr(self.brain, 'get_system_status')
                )

            status = "active" if brain_active else "disconnected"
            color = self.colors['success'] if status == "active" else self.colors['danger']
            status_text = f"Соединение: {'активно' if status == 'active' else 'отключено'}"

            # Обновляем только если статус изменился
            if hasattr(self, '_last_status') and self._last_status != status:
                self.chat_logger.info(f"Статус соединения изменился: {self._last_status} -> {status}")
            self._last_status = status

            if hasattr(self, 'status_indicator') and self.status_indicator:
                self.status_indicator.itemconfig("indicator", fill=color)
            if hasattr(self, 'connection_status') and self.connection_status:
                self.connection_status.config(text=status_text)

            # Дополнительная информация для отладки
            if not brain_active and self.brain:
                self.chat_logger.debug(f"Brain найден, но не активен. Свойства brain: {dir(self.brain)[:10]}...")
            elif brain_active:
                self.chat_logger.debug("Brain активен и работает нормально")

        except Exception as e:
            # Fallback на disconnected при ошибке
            if hasattr(self, 'status_indicator') and self.status_indicator:
                self.status_indicator.itemconfig("indicator", fill=self.colors['danger'])
            if hasattr(self, 'connection_status') and self.connection_status:
                self.connection_status.config(text="Соединение: ошибка")
            self.chat_logger.error(f"Ошибка проверки статуса соединения: {e}")

        # Обновляем компактный индикатор загрузки моделей
        self._update_model_loading_indicator()

    def _update_model_loading_indicator(self):
        try:
            if not hasattr(self, 'model_load_label') or not self.model_load_label:
                return
            st = self.model_loading_state
            if not st.get("active"):
                self.model_load_label.config(text="")
                return
            name = st.get("name") or st.get("model_id") or "модель"
            prog = int(st.get("progress") or 0)
            action = st.get("action") or "load"
            if st.get("error"):
                if action == "unload":
                    self.model_load_label.config(text=f"Выгрузка модели '{name}': ошибка")
                else:
                    self.model_load_label.config(text=f"Загрузка модели '{name}': ошибка")
            elif prog >= 100 and action != "unload":
                self.model_load_label.config(text=f"Загрузка модели '{name}': завершено")
            else:
                if action == "unload":
                    # Для выгрузки обычно не считаем проценты
                    self.model_load_label.config(text=f"Выгрузка модели '{name}'...")
                else:
                    self.model_load_label.config(text=f"Загрузка модели '{name}': {prog}%")
        except Exception:
            pass

    def _handle_model_load_event(self, data: Dict[str, Any]):
        """Обрабатывает события загрузки модели из ядра (вызывается из фоновых потоков)."""
        try:
            event = data.get('event') if isinstance(data, dict) else None
            if not event:
                return
            def apply_update():
                try:
                    if event == 'model_load_start':
                        self.model_loading_state.update({
                            "active": True,
                            "model_id": data.get('model_id'),
                            "name": data.get('name'),
                            "progress": 0,
                            "error": None,
                            "action": "load",
                        })
                    elif event == 'model_load_progress':
                        self.model_loading_state.update({
                            "active": True,
                            "model_id": data.get('model_id'),
                            "progress": max(0, min(100, int(data.get('progress', 0)))),
                            "action": "load",
                        })
                    elif event == 'model_load_complete':
                        self.model_loading_state.update({
                            "active": False,
                            "progress": 100,
                            "error": None,
                            "action": "load",
                        })
                    elif event == 'model_load_error':
                        self.model_loading_state.update({
                            "active": False,
                            "error": data.get('error') or 'unknown',
                            "action": "load",
                        })
                    elif event == 'model_unload_start':
                        self.model_loading_state.update({
                            "active": True,
                            "model_id": data.get('model_id'),
                            "name": data.get('name'),
                            "progress": 0,
                            "error": None,
                            "action": "unload",
                        })
                    elif event == 'model_unload_complete':
                        self.model_loading_state.update({
                            "active": False,
                            "progress": 0,
                            "error": None,
                            "action": "unload",
                        })
                    elif event == 'model_unload_error':
                        self.model_loading_state.update({
                            "active": False,
                            "error": data.get('error') or 'unknown',
                            "action": "unload",
                        })
                    self._update_model_loading_indicator()
                except tk.TclError:
                    pass
                except Exception:
                    pass
            # Планируем в GUI-потоке
            self.gui_queue.put(apply_update)
        except Exception:
            pass

    def _handle_models_ready_event(self):
        """Обработка события глобовой готовности моделей: очищаем индикатор загрузки."""
        try:
            def apply_ready():
                try:
                    self.model_loading_state.update({
                        "active": False,
                        "progress": 100,
                        "error": None,
                        "action": "load",
                    })
                    self._update_model_loading_indicator()
                    self.show_toast("Модели готовы к работе", "info")
                except Exception:
                    pass
            self.gui_queue.put(apply_ready)
        except Exception:
            pass

    def _handle_notifications(self):
        contradiction_stats = self.dashboard_data.get('contradiction_stats', {})
        high_severity_count = contradiction_stats.get('by_severity', {}).get('high', 0)
        critical_count = contradiction_stats.get('by_severity', {}).get('critical', 0)
        serious_contradictions = high_severity_count + critical_count

        if serious_contradictions > 0:
            self.show_toast(f"Обнаружено {serious_contradictions} серьезных противоречий!", "warning", key="serious_contradictions")

    def _process_gui_queue(self):
        # Если GUI остановлен или окно уничтожено – не продолжаем
        if not self.running or not self.root:
            return
        try:
            while True:
                task = self.gui_queue.get_nowait()
                if callable(task):
                    task()
        except queue.Empty:
            pass
        except tk.TclError:
            # Окно могло быть уничтожено – прекращаем цикл
            return
        finally:
            try:
                if self.running and self.root and self.root.winfo_exists():
                    # Сохраняем ID, чтобы можно было отменить при остановке
                    self.gui_queue_job = self.root.after(100, self._process_gui_queue)
            except tk.TclError:
                # Окно уничтожено – не планируем повторно
                self.gui_queue_job = None

    def process_query(self, query: str) -> str:
        """
        Обрабатывает запрос пользователя с полным логгированием всего процесса.
        
        Args:
            query: Текст запроса от пользователя
            
        Returns:
            str: Ответ системы
        """
        # 1. Логируем получение запроса от пользователя
        self.chat_logger.info(f"Получен запрос от пользователя: '{query}'")
        
        # 2. Логируем начало обработки запроса
        self.chat_logger.debug(f"Начало обработки запроса: '{query}'")
        start_time = time.time()
        
        # 3. Проверяем доступность ядра системы
        if not self.brain:
            error_msg = "Ошибка: ядро системы недоступно. Пожалуйста, перезагрузите систему."
            self.chat_logger.error(error_msg)
            return error_msg
        
        try:
            # 4. Логируем передачу запроса в ядро
            self.chat_logger.info(f"Передача запроса в ядро системы: '{query}'")
            
            # 5. Логируем информацию о ядре
            if hasattr(self.brain, 'get_system_status'):
                system_status = self.brain.get_system_status()
                self.chat_logger.debug(f"Состояние системы перед обработкой запроса: {system_status}")
            
            # 6. Логируем токенизацию запроса
            self.chat_logger.debug(f"Начало токенизации запроса: '{query}'")
            tokenization_start = time.time()
            
            # 7. Выполняем токенизацию (пример, если есть метод токенизации)
            if hasattr(self.brain, 'tokenize_query'):
                tokens = self.brain.tokenize_query(query)
                tokenization_time = time.time() - tokenization_start
                self.chat_logger.info(f"Токенизация завершена за {tokenization_time:.4f} сек. Количество токенов: {len(tokens)}")
                self.chat_logger.debug(f"Токены: {tokens[:10]}..." if len(tokens) > 10 else f"Токены: {tokens}")
            else:
                self.chat_logger.warning("Метод tokenize_query не найден в ядре. Используется базовая токенизация.")
                tokens = query.split()
                self.chat_logger.info(f"Базовая токенизация выполнена. Количество токенов: {len(tokens)}")
            
            # 8. Логируем передачу токенов в обработку
            processing_start = time.time()
            self.chat_logger.info(f"Начало обработки запроса в ядре. Токенов: {len(tokens)}")
            
            # 9. Вызываем обработку запроса
            # Передаем историю сообщений как контекст для сохранения памяти разговора
            history_context = None
            if hasattr(self, 'chat_module') and hasattr(self.chat_module, 'message_history'):
                # Ограничиваем историю последними 10 сообщениями для экономии токенов
                recent_history = self.chat_module.message_history[-10:] if self.chat_module.message_history else []
                if recent_history:
                    history_context = {"conversation_history": recent_history}
            
            response_obj = self.brain.process_query(query, context=history_context)
            
            # Извлекаем текст из ответа (может быть dict или str)
            if isinstance(response_obj, dict):
                response = response_obj.get('text') or response_obj.get('response') or str(response_obj)
            else:
                response = str(response_obj) if response_obj else "нет ответа"
            
            # 10. Логируем время обработки
            processing_time = time.time() - processing_start
            
            # 11. Логируем получение ответа
            response_preview = response[:100] if response else ""
            self.chat_logger.info(f"Получен ответ от ядра: '{response_preview}...'")
            
            # 12. Проверяем валидность ответа
            if not response or (isinstance(response, str) and response.strip().lower() == "нет ответа"):
                self.chat_logger.error(f"Ядро вернуло недопустимый ответ на запрос '{query}'")
                response = (
                    "Система временно не может предоставить полный ответ. "
                    "Попробуйте перефразировать запрос или задать другой вопрос."
                )
                self.chat_logger.warning("Сгенерирован альтернативный ответ из-за недопустимого ответа ядра")
            
            # 13. Логируем сохранение в историю
            self.chat_logger.debug("Сохранение запроса и ответа в историю")
            self._save_to_history(query, response)
            
            # 14. Логируем общее время обработки запроса
            total_time = time.time() - start_time
            self.chat_logger.info(f"Общий процесс обработки запроса завершен за {total_time:.4f} сек")
            
            return response
        except Exception as e:
            # 15. Логируем критические ошибки
            self.chat_logger.exception(f"Критическая ошибка при обработке запроса: {e}")
            error_response = (
                "Произошла ошибка при обработке запроса. "
                "Пожалуйста, попробуйте повторить запрос позже."
            )
            self.chat_logger.error(f"Возвращен ошибочный ответ: '{error_response}'")
            return error_response

    def _save_to_history(self, query: str, response: str):
        """Сохраняет запрос и ответ в историю с подробным логгированием."""
        # 1. Логируем начало сохранения
        self.chat_logger.debug(f"Начало сохранения в историю: Q: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # 2. Логируем длину запроса и ответа
        self.chat_logger.info(f"Сохранение в историю: длина запроса={len(query)}, длина ответа={len(response)}")
        
        # 3. Логируем содержимое (с обрезкой для длинных сообщений)
        if len(query) > 100:
            self.chat_logger.debug(f"Запрос (первые 100 символов): {query[:100]}...")
        else:
            self.chat_logger.debug(f"Запрос: {query}")
            
        if len(response) > 100:
            self.chat_logger.debug(f"Ответ (первые 100 символов): {response[:100]}...")
        else:
            self.chat_logger.debug(f"Ответ: {response}")
        
        # 4. Логируем временные метки
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.chat_logger.info(f"Сохранение в историю с временной меткой: {timestamp}")
        
        # 5. Попытка сохранения в файл (пример)
        try:
            history_file = os.path.join(self.cache_dir, "chat_history.json")
            history = []
            
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            # Добавляем новую запись
            history.append({
                "timestamp": timestamp,
                "query": query,
                "response": response,
                "query_length": len(query),
                "response_length": len(response)
            })
            
            # Ограничиваем историю 500 записями
            if len(history) > 500:
                history = history[-500:]
            
            # Сохраняем
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            # 6. Логируем успешное сохранение
            self.chat_logger.info(f"История чата успешно сохранена в {history_file}. Текущая длина истории: {len(history)}")
        
        except Exception as e:
            # 7. Логируем ошибки сохранения
            self.chat_logger.error(f"Ошибка при сохранении истории чата: {str(e)}", exc_info=True)
        
        # 8. Логируем завершение сохранения
        self.chat_logger.debug("Сохранение в историю завершено")

    def show_toast(self, message: str, level: str = "info", duration: int = 5000, key: str = None):
        if key:
            now = time.time()
            last_time = self.last_notification_times.get(key, 0)
            if now - last_time < self.notification_throttle_seconds:
                self.chat_logger.debug(f"Подавлено повторное уведомление с ключом '{key}'")
                return
            self.last_notification_times[key] = now
            self.chat_logger.info(f"Показ уведомления с ключом '{key}': {message}")
        
        self.gui_queue.put(lambda: self._create_toast_window(message, level, duration))

    def _create_toast_window(self, message, level, duration):
        """Создает всплывающее окно с уведомлением"""
        try:
            toast = tk.Toplevel(self.root)
            toast.overrideredirect(True)
            
            # Цвета по уровню
            colors = {
                'info': ('#0078d7', 'white'),
                'success': ('#28a745', 'white'),
                'warning': ('#ffc107', 'black'),
                'error': ('#dc3545', 'white')
            }
            bg_color, fg_color = colors.get(level, ('#333333', 'white'))
            
            toast.configure(bg=bg_color)
            
            # Размещение в правом нижнем углу
            if not self.root:
                return
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            label = tk.Label(toast, text=message, bg=bg_color, fg=fg_color,
                           font=('Segoe UI', 10), wraplength=300, padx=15, pady=10)
            label.pack()
            
            toast.update_idletasks()
            width = toast.winfo_width()
            height = toast.winfo_height()
            
            x = screen_width - width - 20
            y = screen_height - height - 100
            toast.geometry(f"{width}x{height}+{x}+{y}")
            
            # Автоматическое закрытие
            if self.root:
                self.root.after(int(duration * 1000), toast.destroy)
            
        except Exception as e:
            logger.debug(f"Ошибка создания toast окна: {e}")

    def _reboot_system(self):
        if not messagebox.askyesno("Подтверждение", "Вы действительно хотите перезагрузить систему?"):
            self.chat_logger.info("Отмена перезагрузки системы по запросу пользователя")
            return
            
        self.chat_logger.warning("Инициирована перезагрузка системы по запросу пользователя")
        self.show_toast("Перезагрузка системы...", "info")
        logger.info("Инициирована перезагрузка системы")
        
        if self.brain and hasattr(self.brain, 'reboot'):
            self.chat_logger.info("Запуск фонового потока для перезагрузки ядра")
            threading.Thread(target=self.brain.reboot, daemon=True).start()
        else:
            self.chat_logger.error("Попытка перезагрузки без поддержки функции reboot в ядре")
            self.show_toast("Функция перезагрузки не поддерживается", "error")

    def _soft_reload(self):
        """Инициирует soft-reload ядра с сохранением ML и обновлением GUI."""
        try:
            if not self.brain or not hasattr(self.brain, 'soft_reload'):
                self.show_toast("Soft-reload не поддерживается ядром", "error")
                return
            self.show_toast("Горячая перезагрузка...", "info")
            self.chat_logger.warning("Инициирован soft-reload ядра по запросу пользователя")
            def _do_reload():
                try:
                    ok = self.brain.soft_reload(reload_gui=True)
                    if not ok:
                        self.gui_queue.put(lambda: self.show_toast("Soft-reload завершился с ошибкой", "error"))
                except Exception as e:
                    self.gui_queue.put(lambda: self.show_toast(f"Soft-reload ошибка: {e}", "error"))
            threading.Thread(target=_do_reload, daemon=True).start()
        except Exception:
            pass

    def start_gui(self) -> None:
        """Запуск GUI для CoreBrain."""
        if not self.running:
            self.chat_logger.info("Запуск GUI после полной инициализации системы...")
            self.start()
        else:
            self.chat_logger.info("GUI уже запущен")
    
    def start(self):
        if self.running: 
            self.chat_logger.warning("Попытка запуска GUI, когда он уже запущен")
            return
            
        self.running = True
        self.chat_logger.info("Запуск графического интерфейса...")
        
        try:
            # НЕ создаем root если он уже существует (из run_gui.py)
            # if self.root is None:
            #     self.root = tk.Tk()
            #     self.root.title("CogniFlex - Адаптивная когнитивная система")
            #     self.root.geometry("1280x800")
            #     self.root.minsize(800, 600)
            #     self._create_styles()
            
            # Проверяем, были ли модули уже инициализированы
            if not hasattr(self, 'analytics_module') or self.analytics_module is None:
                # self._create_interface()  # Не вызываем, так как интерфейс уже создан
                self._init_modules()
            
            self._load_state()
            self._start_background_services()
            
            # Устанавливаем обработчик закрытия только если root существует
            if self.root is not None:
                self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            
            # Логируем успешный запуск
            self.chat_logger.info("Графический интерфейс успешно запущен")
            if self.root is not None:
                self.chat_logger.info(f"Разрешение окна: {self.root.winfo_width()}x{self.root.winfo_height()}")
            else:
                self.chat_logger.warning("Окно root не создано, пропускаем логирование размера")
            
            if self.root is not None:
                self.root.mainloop()
        except Exception as e:
            self.chat_logger.critical(f"Критическая ошибка запуска GUI: {e}", exc_info=True)

    def stop(self):
        if not self.running: 
            self.chat_logger.warning("Попытка остановки GUI, когда он не запущен")
            return
            
        self.running = False
        self.chat_logger.info("Остановка графического интерфейса...")
        
        self.stop_event.set()
        if self.update_job and self.root:
            try:
                self.root.after_cancel(self.update_job)
                self.chat_logger.debug("Отменено запланированное обновление интерфейса")
            except tk.TclError:
                pass
            finally:
                self.update_job = None
        if self.gui_queue_job and self.root:
            try:
                self.root.after_cancel(self.gui_queue_job)
                self.chat_logger.debug("Отменена запланированная обработка GUI очереди")
            except tk.TclError:
                pass
            finally:
                self.gui_queue_job = None
            
        self._save_state()
        
        # Очищаем все модули с after задачами
        self._cleanup_modules()
        
        if self.brain: 
            self.brain.stop()
            self.chat_logger.info("Ядро системы остановлено")
            
        if self.root: 
            self.root.destroy()
            self.chat_logger.info("Окно GUI уничтожено")
        
        self.chat_logger.info("Графический интерфейс остановлен")

    def _cleanup_modules(self):
        """Очищает все модули с after задачами."""
        try:
            modules_to_cleanup = [
                'contradiction_module',
                'neuromorphic_module', 
                'memory_module',
                'learning_module'
            ]
            
            for module_name in modules_to_cleanup:
                module = getattr(self, module_name, None)
                if module and hasattr(module, 'cleanup'):
                    module.cleanup()
                    self.chat_logger.debug(f"Очищен модуль: {module_name}")
                elif module and hasattr(module, '_after_jobs'):
                    # Прямая очистка если нет метода cleanup
                    if hasattr(self, 'root') and self.root:
                        for job_id in module._after_jobs:
                            try:
                                self.root.after_cancel(job_id)
                            except Exception:
                                pass
                        module._after_jobs.clear()
                        self.chat_logger.debug(f"Прямая очистка after задач в модуле: {module_name}")
            
            # Очищаем активные уведомления
            if hasattr(self, 'active_notifications') and self.root:
                for notification in self.active_notifications[:]:
                    try:
                        job_id = notification.get('job_id')
                        if job_id:
                            self.root.after_cancel(job_id)
                    except Exception:
                        pass
                self.active_notifications.clear()
                self.chat_logger.debug("Очищены активные уведомления")
                        
            self.chat_logger.info("Очистка модулей завершена")
        except Exception as e:
            self.chat_logger.error(f"Ошибка при очистке модулей: {e}")

    def _save_state(self):
        """Сохраняет состояние GUI с подробным логгированием."""
        try:
            state = {
                "current_view": self.current_view,
                "theme": self.theme,
                "compact_mode": self.compact_mode,
                "window_size": {
                    "width": self.root.winfo_width() if self.root else 1280,
                    "height": self.root.winfo_height() if self.root else 800
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            state_file = os.path.join(self.cache_dir, "gui_state.json")
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                
            self.chat_logger.info(f"Состояние GUI сохранено в {state_file}")
            self.chat_logger.debug(f"Сохраненное состояние: {state}")
            
        except Exception as e:
            self.chat_logger.error(f"Ошибка при сохранении состояния GUI: {str(e)}", exc_info=True)

    def _load_state(self):
        """Загружает состояние GUI с подробным логгированием."""
        try:
            state_file = os.path.join(self.cache_dir, "gui_state.json")
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # Восстанавливаем состояние
                self.current_view = state.get("current_view", "chat")
                self.theme = state.get("theme", "light")
                self.compact_mode = state.get("compact_mode", False)
                
                # Логируем загруженное состояние
                self.chat_logger.info(f"Состояние GUI загружено из {state_file}")
                self.chat_logger.debug(f"Загруженное состояние: {state}")
                
                # Восстанавливаем размер окна
                if "window_size" in state and self.root:
                    width = state["window_size"].get("width", 1280)
                    height = state["window_size"].get("height", 800)
                    self.root.geometry(f"{width}x{height}")
                    self.chat_logger.info(f"Восстановлен размер окна: {width}x{height}")
            else:
                self.chat_logger.info("Файл состояния GUI не найден, используется состояние по умолчанию")
                
        except Exception as e:
            self.chat_logger.error(f"Ошибка при загрузке состояния GUI: {str(e)}", exc_info=True)
            self.chat_logger.info("Используется состояние по умолчанию")

    def update_status(self, status: str, details: Optional[Dict[str, Any]] = None):
        """Обновляет статус системы в GUI."""
        self.system_status = status
        if details:
            self.dashboard_data.update(details)
        logger.debug(f"Статус обновлен: {status}")

    def show_error(self, title: str, message: str):
        """Показывает сообщение об ошибке."""
        logger.error(f"GUI Error - {title}: {message}")
        if self.root:
            messagebox.showerror(title, message)

    def show_message(self, title: str, message: str, msg_type: str = "info"):
        """Показывает информационное сообщение."""
        logger.info(f"GUI Message - {title}: {message}")
        if self.root:
            if msg_type == "info":
                messagebox.showinfo(title, message)
            elif msg_type == "warning":
                messagebox.showwarning(title, message)
            elif msg_type == "error":
                messagebox.showerror(title, message)

    def create_main_window(self):
        """Создает главное окно приложения."""
        if self.root:
            return
            
        self.root = tk.Tk()
        self.root.title("CogniFlex - Cognitive AI System")
        self.root.geometry("1280x800")
        self.root.configure(bg=self.colors['bg'])
        
        # Создаем основную область контента
        self.content_area = ttk.Frame(self.root)
        self.content_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        logger.info("Главное окно GUI создано")

    def show_notification(self, message: str, msg_type: str = "info"):
        """Показывает уведомление пользователю."""
        logger.info(f"GUI Notification - {msg_type}: {message}")
        if self.root:
            if msg_type == "info":
                messagebox.showinfo("Уведомление", message)
            elif msg_type == "warning":
                messagebox.showwarning("Предупреждение", message)
            elif msg_type == "error":
                messagebox.showerror("Ошибка", message)


class MemoryTab:
    """Упрощённый модуль отображения памяти - learned entities и статистика."""
    
    def __init__(self, gui):
        self.gui = gui
        self.frame = None
        self.entity_list = None
        self.stats_label = None
        self.curiosity_label = None
        self._after_jobs = []
    
    def activate(self):
        """Активирует вкладку памяти с упрощённым отображением."""
        self.frame = self.gui.content_area
        
        title_label = ttk.Label(self.frame, text="Память системы", font=("Segoe UI", 14, "bold"))
        title_label.pack(pady=(10, 5))
        
        stats_frame = ttk.LabelFrame(self.frame, text="Статистика знаний")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_label = ttk.Label(stats_frame, text="Загрузка...")
        self.stats_label.pack(pady=10, padx=10)
        
        curiosity_frame = ttk.LabelFrame(self.frame, text="Триггеры любопытства")
        curiosity_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.curiosity_label = ttk.Label(curiosity_frame, text="Загрузка...", wraplength=500)
        self.curiosity_label.pack(pady=10, padx=10)
        
        entities_frame = ttk.LabelFrame(self.frame, text="Недавние изученные сущности")
        entities_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(entities_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.entity_list = tk.Listbox(entities_frame, yscrollcommand=scrollbar.set, height=15)
        self.entity_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.entity_list.yview)
        
        self.update()
    
    def deactivate(self):
        """Деактивирует вкладку памяти."""
        for job_id in self._after_jobs:
            try:
                if self.gui.root:
                    self.gui.root.after_cancel(job_id)
            except Exception:
                pass
        self._after_jobs.clear()
    
    def update(self):
        """Обновляет данные памяти."""
        try:
            verified = 0
            generated = 0
            entities = []
            
            if self.gui.brain and hasattr(self.gui.brain, 'memory_manager'):
                mm = self.gui.brain.memory_manager
                if hasattr(mm, 'get_stats'):
                    stats = mm.get_stats()
                    verified = stats.get('verified_entities', 0)
                    generated = stats.get('generated_entities', 0)
                if hasattr(mm, 'get_recent_entities'):
                    entities = mm.get_recent_entities(limit=20)
            
            self.stats_label.config(text=f"Подтверждённые: {verified} | Сгенерированные: {generated}")
            
            curiosity_triggers = []
            if self.gui.brain and hasattr(self.gui.brain, 'get_curiosity_triggers'):
                curiosity_triggers = self.gui.brain.get_curiosity_triggers()[:5]
            
            if curiosity_triggers:
                trigger_text = "\n".join([f"• {t}" for t in curiosity_triggers])
                self.curiosity_label.config(text=trigger_text)
            else:
                self.curiosity_label.config(text="Нет активных триггеров")
            
            self.entity_list.delete(0, tk.END)
            for entity in entities:
                self.entity_list.insert(tk.END, entity)
            
            job_id = self.gui.root.after(5000, self.update) if self.gui.root else None
            if job_id:
                self._after_jobs.append(job_id)
        except Exception as e:
            logger.debug(f"MemoryTab update error: {e}")


class SystemTab:
    """Упрощённый модуль отображения системного статуса и здоровья."""
    
    def __init__(self, gui):
        self.gui = gui
        self.frame = None
        self.status_label = None
        self.health_label = None
        self._after_jobs = []
    
    def activate(self):
        """Активирует вкладку системы с упрощённым отображением."""
        self.frame = self.gui.content_area
        
        title_label = ttk.Label(self.frame, text="Система", font=("Segoe UI", 14, "bold"))
        title_label.pack(pady=(10, 5))
        
        status_frame = ttk.LabelFrame(self.frame, text="Статус")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Загрузка...", wraplength=500)
        self.status_label.pack(pady=10, padx=10)
        
        health_frame = ttk.LabelFrame(self.frame, text="Здоровье системы")
        health_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.health_label = ttk.Label(health_frame, text="Загрузка...", wraplength=500)
        self.health_label.pack(pady=10, padx=10)
        
        metrics_frame = ttk.LabelFrame(self.frame, text="Метрики")
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.metrics_text = tk.Text(metrics_frame, height=10, state=tk.DISABLED)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.update()
    
    def deactivate(self):
        """Деактивирует вкладку системы."""
        for job_id in self._after_jobs:
            try:
                if self.gui.root:
                    self.gui.root.after_cancel(job_id)
            except Exception:
                pass
        self._after_jobs.clear()
    
    def update(self):
        """Обновляет данные системы."""
        try:
            status = "Неизвестно"
            components = 0
            if self.gui.brain:
                if hasattr(self.gui.brain, 'running') and self.gui.brain.running:
                    status = "Активен"
                if hasattr(self.gui.brain, 'components'):
                    components = len(self.gui.brain.components) if self.gui.brain.components else 0
            
            self.status_label.config(text=f"Статус: {status}\nКомпоненты: {components}")
            
            health_data = {}
            if self.gui.integrator and hasattr(self.gui.integrator, 'get_system_health'):
                health_data = self.gui.integrator.get_system_health()
            elif self.gui.brain and hasattr(self.gui.brain, 'get_system_health'):
                health_data = self.gui.brain.get_system_health()
            
            health_text = f"Общее: {health_data.get('overall', 'N/A')}"
            if 'issues' in health_data and health_data['issues']:
                health_text += f"\nПроблемы: {len(health_data['issues'])}"
            self.health_label.config(text=health_text)
            
            self.metrics_text.config(state=tk.NORMAL)
            self.metrics_text.delete(1.0, tk.END)
            
            dash_data = {}
            if self.gui.integrator and hasattr(self.gui.integrator, 'get_system_stats'):
                dash_data = self.gui.integrator.get_system_stats()
            
            metrics = dash_data.get('metrics', {})
            cache_stats = dash_data.get('cache_stats', {})
            
            lines = [
                f"CPU: {metrics.get('cpu_usage', 0):.1f}%",
                f"Memory: {metrics.get('memory_usage', 0):.1f}%",
                f"Cache Hit Rate: {cache_stats.get('hit_rate', 0):.1%}",
                f"Cache Utilization: {cache_stats.get('cache_utilization_percent', 0):.1f}%",
            ]
            self.metrics_text.insert(tk.END, "\n".join(lines))
            self.metrics_text.config(state=tk.DISABLED)
            
            job_id = self.gui.root.after(5000, self.update) if self.gui.root else None
            if job_id:
                self._after_jobs.append(job_id)
        except Exception as e:
            logger.debug(f"SystemTab update error: {e}")


def create_gui(brain=None, cache_dir: str = None, integrator=None):
    """Создает и возвращает экземпляр GUI с подробным логгированием."""
    logger.info("Создание экземпляра GUI")
    gui = CogniFlexGUI(brain, integrator=integrator, cache_dir=cache_dir)
    
    # Создаем главное окно ПЕРЕД инициализацией интерфейса
    try:
        gui.create_main_window()
        logger.info("Главное окно создано в create_gui()")
    except Exception as e:
        logger.error(f"Ошибка создания главного окна: {e}", exc_info=True)
    
    # Инициализируем модули GUI
    try:
        gui._create_interface()
        gui._init_modules()
        logger.info("Модули GUI инициализированы в create_gui()")
    except Exception as e:
        logger.error(f"Ошибка инициализации модулей GUI: {e}", exc_info=True)
    
    # Логируем создание экземпляра
    logger.info("Экземпляр GUI создан")
    logger.debug(f"Путь к кэшу GUI: {gui.cache_dir}")
    
    return gui