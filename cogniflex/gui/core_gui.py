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
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        logger.debug("Инициализация графического интерфейса...")
        self.brain = brain
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
        }
        
        self.dashboard_data = {}
        self.last_notification_times = {}
        self.gui_queue_job = None  # ID запланированного after для очереди GUI
        
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

    def _init_modules(self):
        if not self.content_area:
            logger.error("Попытка инициализации модулей до создания интерфейса")
            return

        module_map = {
            "chat": ".chat_module.ChatModule",
            "analytics": ".analytics_module.AnalyticsModule",
            "knowledge": ".knowledge_graph_module.KnowledgeGraphModule",
            "contradictions": ".contradiction_module.ContradictionModule",
            "memory": ".memory_module.MemoryModule",
            "learning": ".learning_module.LearningModule",
            "settings": ".settings_module.SettingsModule",
            "neuromorphic": ".neuromorphic_module.NeuromorphicModule",
        }

        for name, path in module_map.items():
            try:
                module_path, class_name = path.rsplit('.', 1)
                module = __import__(f"cogniflex.gui{module_path}", fromlist=[class_name])
                module_class = getattr(module, class_name)
                instance = module_class(self)
                setattr(self, f"{name}_module", instance)
                logger.info(f"Модуль '{name}' инициализирован")
            except Exception as e:
                logger.warning(f"Модуль '{name}' недоступен: {e}", exc_info=True)
        
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

        self.root.configure(bg=self.colors['bg'])

    def _create_interface(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        self._create_navbar()
        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.content_area = ttk.Frame(self.content_frame)
        self.content_area.pack(fill=tk.BOTH, expand=True)
        self._create_status_bar()

    def _create_navbar(self):
        navbar = ttk.Frame(self.main_container, height=50)
        navbar.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(navbar, text="CogniFlex", font=("Segoe UI", 16, "bold"), foreground=self.colors['primary']).pack(side=tk.LEFT)
        nav_frame = ttk.Frame(navbar)
        nav_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        nav_items = [
            ("Чат", "chat"), ("Аналитика", "analytics"), ("Граф знаний", "knowledge"),
            ("Противоречия", "contradictions"), ("Память", "memory"),
            ("Обучение", "learning"), ("Нейроморфика", "neuromorphic"), ("Настройки", "settings")
        ]
        for text, view_id in nav_items:
            btn = ttk.Button(nav_frame, text=text, command=lambda v=view_id: self._switch_view(v))
            btn.pack(side=tk.LEFT, padx=5)
        right_frame = ttk.Frame(navbar)
        right_frame.pack(side=tk.RIGHT)
        ttk.Button(right_frame, text="Перезагрузить", command=self._reboot_system).pack(side=tk.LEFT, padx=5)

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
        self.contradictions_label.pack(side=tk.LEFT)

    def on_close(self):
        if messagebox.askyesno("Подтверждение", "Вы действительно хотите выйти?"):
            self.stop()

    def _switch_view(self, view_id: str):
        logger.debug(f"Переключение на представление: {view_id}")
        if not self.content_area: return

        # Деактивируем предыдущий модуль, чтобы отменить таймеры/задания
        try:
            prev_view = getattr(self, "current_view", None)
            if prev_view and prev_view != view_id:
                prev_module = getattr(self, f"{prev_view}_module", None)
                if prev_module and hasattr(prev_module, "deactivate"):
                    try:
                        prev_module.deactivate()
                    except Exception:
                        # Безопасно игнорируем любые ошибки деактивации
                        pass
        except Exception:
            pass

        for widget in self.content_area.winfo_children():
            widget.destroy()

        module = getattr(self, f"{view_id}_module", None)
        if module and hasattr(module, 'activate'):
            # Логируем активацию модуля
            self.chat_logger.info(f"Активация модуля: {view_id}")
            module.activate()
        else:
            ttk.Label(self.content_area, text=f"Модуль '{view_id}' недоступен.").pack()
        self.current_view = view_id

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
                    except Exception:
                        # Фолбэк на список колбэков
                        if not hasattr(self.brain, 'on_model_load'):
                            setattr(self.brain, 'on_model_load', [])
                        self.brain.on_model_load.append(handler)
                        self.chat_logger.info("GUI подписан на события model_load через on_model_load")
                else:
                    if not hasattr(self.brain, 'on_model_load'):
                        setattr(self.brain, 'on_model_load', [])
                    self.brain.on_model_load.append(handler)
                    self.chat_logger.info("GUI подписан на события model_load (fallback)")
        except Exception as e:
            logger.warning(f"Не удалось подписаться на события model_load: {e}")

    def _schedule_update(self):
        if not self.running: return
        self._update_interface()
        self.update_job = self.root.after(self.settings.get("gui", {}).get("auto_update_interval", 5000), self._schedule_update)

    def _update_interface(self):
        if not self.root or not self.running: return
        
        try:
            if self.brain and hasattr(self.brain, 'get_system_dashboard_data'):
                self.dashboard_data = self.brain.get_system_dashboard_data()
            
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
        self.cpu_label.config(text=f"{cpu_val:.1f}%")
        self.memory_label.config(text=f"{mem_val:.1f}%")
        contradiction_stats = self.dashboard_data.get('contradiction_stats', {})
        self.contradictions_label.config(text=str(contradiction_stats.get('total', 0)))

    def _update_status_indicator(self):
        status = "active" if self.brain and self.brain.running else "disconnected"
        color = self.colors['success'] if status == "active" else self.colors['danger']
        status_text = f"Соединение: {'активно' if status == 'active' else 'отключено'}"
        self.status_indicator.itemconfig("indicator", fill=color)
        self.connection_status.config(text=status_text)
        # Обновляем компактный индикатор загрузки моделей
        self._update_model_loading_indicator()

    def _update_model_loading_indicator(self):
        try:
            st = self.model_loading_state
            if not st.get("active"):
                self.model_load_label.config(text="")
                return
            name = st.get("name") or st.get("model_id") or "модель"
            prog = int(st.get("progress") or 0)
            if st.get("error"):
                self.model_load_label.config(text=f"Загрузка модели '{name}': ошибка")
            elif prog >= 100:
                self.model_load_label.config(text=f"Загрузка модели '{name}': завершено")
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
                        })
                    elif event == 'model_load_progress':
                        self.model_loading_state.update({
                            "active": True,
                            "model_id": data.get('model_id'),
                            "progress": max(0, min(100, int(data.get('progress', 0))))
                        })
                    elif event == 'model_load_complete':
                        self.model_loading_state.update({
                            "active": False,
                            "progress": 100,
                            "error": None,
                        })
                    elif event == 'model_load_error':
                        self.model_loading_state.update({
                            "active": False,
                            "error": data.get('error') or 'unknown',
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

    def _handle_notifications(self):
        contradiction_stats = self.dashboard_data.get('contradiction_stats', {})
        high_severity_count = contradiction_stats.get('by_severity', {}).get('high', 0)
        critical_count = contradiction_stats.get('by_severity', {}).get('critical', 0)
        serious_contradictions = high_severity_count + critical_count

        if serious_contradictions > 0:
            self.show_toast(f"Обнаружено {serious_contradictions} серьезных противоречий!", "warning", key="serious_contradictions")

        opportunities = self.dashboard_data.get('learning_opportunities', [])
        if len(opportunities) > 0:
            self.show_toast(f"Найдено {len(opportunities)} новых возможностей для обучения.", "info", key="learning_opportunities")

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
            response = self.brain.process_query(query)
            
            # 10. Логируем время обработки
            processing_time = time.time() - processing_start
            self.chat_logger.info(f"Обработка запроса завершена за {processing_time:.4f} сек")
            
            # 11. Логируем получение ответа
            self.chat_logger.info(f"Получен ответ от ядра: '{response[:100]}{'...' if len(response) > 100 else ''}'")
            
            # 12. Проверяем валидность ответа
            if not response or response.strip().lower() == "нет ответа":
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
        pass # Placeholder

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

    def start(self):
        if self.running: 
            self.chat_logger.warning("Попытка запуска GUI, когда он уже запущен")
            return
            
        self.running = True
        self.chat_logger.info("Запуск графического интерфейса...")
        
        try:
            self.root = tk.Tk()
            self.root.title("CogniFlex - Адаптивная когнитивная система")
            self.root.geometry("1280x800")
            self.root.minsize(800, 600)
            self._create_styles()
            self._create_interface()
            self._init_modules()
            self._load_state()
            self._start_background_services()
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            
            # Логируем успешный запуск
            self.chat_logger.info("Графический интерфейс успешно запущен")
            self.chat_logger.info(f"Разрешение окна: {self.root.winfo_width()}x{self.root.winfo_height()}")
            
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
        
        if self.brain: 
            self.brain.stop()
            self.chat_logger.info("Ядро системы остановлено")
            
        if self.root: 
            self.root.destroy()
            self.chat_logger.info("Окно GUI уничтожено")
        
        self.chat_logger.info("Графический интерфейс остановлен")

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

def create_gui(brain=None, cache_dir: str = None):
    """Создает и возвращает экземпляр GUI с подробным логгированием."""
    logger.info("Создание экземпляра GUI")
    gui = CogniFlexGUI(brain, cache_dir)
    
    # Логируем создание экземпляра
    logger.info("Экземпляр GUI создан")
    logger.debug(f"Путь к кэшу GUI: {gui.cache_dir}")
    
    return gui