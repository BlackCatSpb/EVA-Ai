"""
Интегрированный графический интерфейс CogniFlex с поддержкой фрактальной архитектуры.

Этот интерфейс работает с CogniFlexIntegrator для полной интеграции всех компонентов
системы через централизованную событийную шину.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from cogniflex.core.integration_layer import CogniFlexIntegrator

logger = logging.getLogger("cogniflex.integrated_gui")

class IntegratedCogniFlexGUI:
    """
    Интегрированный графический интерфейс для CogniFlex с поддержкой фрактальной архитектуры.
    """

    def __init__(self, brain_or_integrator):
        """
        Инициализация интегрированного GUI.

        Args:
            brain_or_integrator: Экземпляр CoreBrain или CogniFlexIntegrator
        """
        # Проверяем тип переданного объекта
        if hasattr(brain_or_integrator, 'components'):
            # Это CoreBrain
            self.brain = brain_or_integrator
            self.integrator = None
        else:
            # Это CogniFlexIntegrator
            self.integrator = brain_or_integrator
            self.brain = getattr(brain_or_integrator, 'core_brain', None)
        
        self.root = None
        self.running = False
        self.chat_history = []
        self.current_status = "Инициализация..."

        # Настройка интерфейса
        self.setup_gui()

    def setup_gui(self):
        """Настройка графического интерфейса."""
        self.root = tk.Tk()
        self.root.title("CogniFlex - Единая Фрактальная Архитектура")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Обработка закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Создание интерфейса
        self.create_widgets()
        self.setup_event_handlers()

        logger.info("Интегрированный GUI инициализирован")
    
    def on_closing(self):
        """Обработка закрытия окна."""
        if messagebox.askokcancel("Выход", "Вы хотите выйти из CogniFlex?"):
            self.running = False
            try:
                if hasattr(self, 'brain') and self.brain:
                    logger.info("Остановка CoreBrain...")
                    if hasattr(self.brain, 'shutdown'):
                        self.brain.shutdown()
                    elif hasattr(self.brain, 'stop'):
                        self.brain.stop()
            except Exception as e:
                logger.error(f"Ошибка при остановке brain: {e}")
            
            try:
                if self.root:
                    self.root.quit()
                    self.root.destroy()
            except Exception:
                pass
            
            logger.info("CogniFlex завершен")
            os._exit(0)

    def create_widgets(self):
        """Создание виджетов интерфейса."""
        # Главный контейнер
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Заголовок
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(
            title_frame,
            text="🧠 CogniFlex - Единая Фрактальная Архитектура",
            font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.LEFT)

        # Статус системы
        self.status_label = ttk.Label(
            title_frame,
            text=f"Статус: {self.current_status}",
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.RIGHT)

        # Панель инструментов
        toolbar_frame = ttk.Frame(main_frame)
        toolbar_frame.pack(fill=tk.X, pady=(0, 10))

        # Кнопки управления
        ttk.Button(
            toolbar_frame,
            text="🚀 Запустить самодиалог",
            command=self.start_self_dialog
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            toolbar_frame,
            text="⚡ Оптимизировать систему",
            command=self.optimize_system
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            toolbar_frame,
            text="📊 Статистика системы",
            command=self.show_system_stats
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Выбор модели
        ttk.Label(toolbar_frame, text="Модель:").pack(side=tk.LEFT, padx=(10, 5))
        
        self.model_var = tk.StringVar(value="RuGPT-3 Large")
        self.model_combo = ttk.Combobox(
            toolbar_frame,
            textvariable=self.model_var,
            values=["RuGPT-3 Large", "Qwen3.5-0.8B", "Qwen3.5-2B", "BitNet 2B"],
            state="readonly",
            width=18
        )
        self.model_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)

        ttk.Button(
            toolbar_frame,
            text="🛑 Остановить",
            command=self.stop_system
        ).pack(side=tk.RIGHT)

        # Основная область
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Левая панель - чат
        left_panel = ttk.LabelFrame(content_frame, text="💬 Диалог", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Область чата
        self.chat_area = scrolledtext.ScrolledText(
            left_panel,
            wrap=tk.WORD,
            height=20,
            font=("Consolas", 10)
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Поле ввода
        input_frame = ttk.Frame(left_panel)
        input_frame.pack(fill=tk.X)

        self.input_field = ttk.Entry(input_frame, font=("Arial", 11))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_field.bind("<Return>", self.send_message)
        
        # Добавляем шоткаты
        self.input_field.bind("<Control-a>", lambda e: (self.input_field.select_range(0, 'end'), "break"))
        self.input_field.bind("<Control-A>", lambda e: (self.input_field.select_range(0, 'end'), "break"))
        self.input_field.bind("<Control-q>", lambda e: (self.input_field.delete(0, 'end'), "break"))
        self.input_field.bind("<Control-Q>", lambda e: (self.input_field.delete(0, 'end'), "break"))

        ttk.Button(
            input_frame,
            text="📤 Отправить",
            command=self.send_message
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Правая панель - информация о системе
        right_panel = ttk.LabelFrame(content_frame, text="📊 Состояние системы", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        # Метрики системы
        self.metrics_frame = ttk.Frame(right_panel)
        self.metrics_frame.pack(fill=tk.BOTH, expand=True)

        # Метки для отображения метрик
        self.create_metrics_labels()

        # Нижняя панель - логи
        log_frame = ttk.LabelFrame(main_frame, text="📝 Логи системы", padding=5)
        log_frame.pack(fill=tk.X, pady=(10, 0))

        self.log_area = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            height=8,
            font=("Consolas", 9)
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def create_metrics_labels(self):
        """Создание меток для отображения метрик системы."""
        metrics = [
            ("Статус:", "status", "Неизвестен"),
            ("Компонентов:", "components", "0"),
            ("Запросов:", "requests", "0"),
            ("Ответов:", "responses", "0"),
            ("Среднее время:", "avg_time", "0.0 сек"),
            ("Активных запросов:", "active", "0"),
            ("Фокус внимания:", "focus", "Нет"),
            ("Противоречий:", "contradictions", "0"),
            ("Возможностей обучения:", "learning", "0")
        ]

        self.metrics_labels = {}

        for i, (label_text, key, default_value) in enumerate(metrics):
            ttk.Label(self.metrics_frame, text=label_text).grid(
                row=i, column=0, sticky=tk.W, pady=2
            )
            label = ttk.Label(self.metrics_frame, text=default_value)
            label.grid(row=i, column=1, sticky=tk.W, pady=2)
            self.metrics_labels[key] = label

    def setup_event_handlers(self):
        """Настройка обработчиков событий."""
        try:
            # Используем brain напрямую если доступен
            event_bus = None
            if self.brain and hasattr(self.brain, 'event_bus'):
                event_bus = self.brain.event_bus
            elif self.integrator and hasattr(self.integrator, 'event_bus'):
                event_bus = self.integrator.event_bus
            
            if event_bus:
                # Подписываемся на события системы
                event_bus.subscribe(
                    'response_generated',
                    self.handle_response_generated,
                    priority=10
                )

                event_bus.subscribe(
                    'system_health_check',
                    self.handle_system_health,
                    priority=9
                )

                event_bus.subscribe(
                    'learning_opportunity',
                    self.handle_learning_opportunity,
                    priority=7
                )

                logger.info("Обработчики событий настроены")

        except Exception as e:
            logger.error(f"Ошибка настройки обработчиков событий: {e}")

    def send_message(self, event=None):
        """Отправка сообщения в систему."""
        message = self.input_field.get().strip()
        if not message:
            return

        # Очищаем поле ввода
        self.input_field.delete(0, tk.END)

        # Добавляем сообщение пользователя в чат
        self.add_to_chat("Вы", message)

        # Отправляем запрос в систему
        self.process_query(message)

    def process_query(self, query: str):
        """Обрабатывает пользовательский запрос."""
        if not query.strip():
            return
        
        self.add_to_log(f"👤 Пользователь: {query}")
        self.update_status("Обработка...")
        
        def process_async():
            try:
                # Обрабатываем запрос через brain или интегратор
                if self.brain and hasattr(self.brain, 'process_query'):
                    result = self.brain.process_query(query)
                elif self.integrator:
                    result = self.integrator.process_query(query)
                else:
                    result = {"response": "Система временно недоступна", "status": "error"}

                # Обновляем интерфейс в главном потоке
                self.root.after(0, lambda: self.handle_query_result(result))
            except Exception as e:
                logger.error(f"Ошибка обработки запроса: {e}")
                self.add_to_chat("Система", f"Ошибка: {e}")
        
        threading.Thread(target=process_async, daemon=True).start()

    def handle_query_result(self, result: Dict[str, Any]):
        """Обработка результата запроса."""
        try:
            # brain.process_query() returns dict with 'text' key
            if isinstance(result, dict):
                if 'text' in result:
                    response = result.get('text', 'Нет ответа')
                    self.add_to_chat("CogniFlex", response)
                    self.update_status("Готов")
                    self.add_to_log(f"✅ Запрос обработан успешно за {result.get('processing_time', 0):.2f} сек")
                    return
                elif result.get('status') == 'error':
                    error_msg = result.get('error', 'Неизвестная ошибка')
                    self.add_to_chat("Система", f"Ошибка: {error_msg}")
                    self.update_status("Ошибка")
                    self.add_to_log(f"❌ Ошибка обработки: {error_msg}")
                    return

            if isinstance(result, str):
                self.add_to_chat("CogniFlex", result)
                self.update_status("Готов")
            else:
                self.add_to_chat("Система", "Неожиданный формат ответа")
                self.update_status("Ошибка")

        except Exception as e:
            logger.error(f"Ошибка обработки результата: {e}")
            self.add_to_chat("Система", f"Ошибка обработки результата: {e}")

    def handle_query_error(self, error: str):
        """Обработка ошибки запроса."""
        self.add_to_chat("Система", f"Ошибка обработки запроса: {error}")
        self.update_status("Ошибка")
        self.add_to_log(f"❌ Ошибка: {error}")

    def handle_response_generated(self, data: Dict[str, Any]):
        """Обработка события response_generated."""
        try:
            if data.get('request_id'):
                response = data.get('response', {})
                if isinstance(response, dict):
                    response_text = response.get('text', str(response))
                else:
                    response_text = str(response)

                self.add_to_chat("CogniFlex", response_text)
                self.add_to_log(f"📝 Сгенерирован ответ для запроса {data['request_id']}")

        except Exception as e:
            logger.error(f"Ошибка обработки response_generated: {e}")

    def handle_system_health(self, data: Dict[str, Any]):
        """Обработка события system_health_check."""
        try:
            status = data.get('status', 'unknown')
            if status == 'healthy':
                self.update_status("Здоров")
            elif status == 'degraded':
                self.update_status("Деградирован")
            else:
                self.update_status("Проблемы")

            self.add_to_log(f"🏥 Статус системы: {status}")

        except Exception as e:
            logger.error(f"Ошибка обработки system_health: {e}")

    def handle_learning_opportunity(self, data: Dict[str, Any]):
        """Обработка события learning_opportunity."""
        try:
            opportunities = data.get('opportunities', [])
            if opportunities:
                self.add_to_log(f"🎓 Обнаружено {len(opportunities)} возможностей обучения")
                for opp in opportunities[:3]:  # Показываем первые 3
                    desc = opp.get('description', 'Неизвестная возможность')
                    self.add_to_log(f"   • {desc}")

        except Exception as e:
            logger.error(f"Ошибка обработки learning_opportunity: {e}")

    def add_to_chat(self, sender: str, message: str):
        """Добавление сообщения в чат."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {sender}: {message}\n"

            self.chat_area.insert(tk.END, formatted_message)
            self.chat_area.see(tk.END)

            # Сохраняем в истории
            self.chat_history.append({
                'timestamp': timestamp,
                'sender': sender,
                'message': message
            })

            # Ограничиваем историю
            if len(self.chat_history) > 1000:
                self.chat_history = self.chat_history[-500:]

        except Exception as e:
            logger.error(f"Ошибка добавления сообщения в чат: {e}")

    def add_to_log(self, message: str):
        """Добавление сообщения в лог."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}\n"

            self.log_area.insert(tk.END, formatted_message)
            self.log_area.see(tk.END)

        except Exception as e:
            logger.error(f"Ошибка добавления в лог: {e}")

    def update_status(self, status: str):
        """Обновление статуса системы."""
        self.current_status = status
        if hasattr(self, 'status_label'):
            model = self.model_var.get() if hasattr(self, 'model_var') else "RuGPT"
            self.status_label.config(text=f"Модель: {model} | Статус: {status}")

    def update_metrics(self):
        """Обновление метрик системы."""
        try:
            if not hasattr(self.integrator, 'get_system_stats'):
                return

            stats = self.integrator.get_system_stats()

            # Обновляем метки
            metrics_map = {
                'status': lambda: stats.get('health', {}).get('status', 'Неизвестен'),
                'components': lambda: str(stats.get('health', {}).get('components_count', 0)),
                'requests': lambda: str(stats.get('metrics', {}).get('total_requests', 0)),
                'responses': lambda: str(stats.get('metrics', {}).get('successful_responses', 0)),
                'avg_time': lambda: ".2f",
                'active': lambda: str(stats.get('active_requests', 0)),
                'focus': lambda: "Активен" if stats.get('attention_focus') else "Нет",
                'contradictions': lambda: str(len(self.integrator.contradiction_resolver.active_contradictions) if hasattr(self.integrator, 'contradiction_resolver') else 0),
                'learning': lambda: str(len(self.integrator.learning_scheduler.pending_opportunities) if hasattr(self.integrator, 'learning_scheduler') else 0)
            }

            for key, getter in metrics_map.items():
                if key in self.metrics_labels:
                    try:
                        value = getter()
                        self.metrics_labels[key].config(text=value)
                    except Exception as e:
                        logger.debug(f"Ошибка обновления метрики {key}: {e}")

        except Exception as e:
            logger.error(f"Ошибка обновления метрик: {e}")

    def start_self_dialog(self):
        """Запуск самодиалога."""
        try:
            if hasattr(self.integrator, 'start_self_dialog'):
                self.integrator.start_self_dialog()
                self.add_to_log("🚀 Запущен самодиалог")
                messagebox.showinfo("Самодиалог", "Самодиалог запущен в фоновом режиме")
            else:
                messagebox.showwarning("Самодиалог", "Самодиалог недоступен")
        except Exception as e:
            logger.error(f"Ошибка запуска самодиалога: {e}")
            messagebox.showerror("Ошибка", f"Не удалось запустить самодиалог: {e}")

    def optimize_system(self):
        """Оптимизация системы."""
        try:
            if hasattr(self.integrator, 'optimize_system'):
                self.integrator.optimize_system()
                self.add_to_log("⚡ Запущена оптимизация системы")
                messagebox.showinfo("Оптимизация", "Оптимизация системы запущена")
            else:
                messagebox.showwarning("Оптимизация", "Оптимизация недоступна")
        except Exception as e:
            logger.error(f"Ошибка оптимизации системы: {e}")
            messagebox.showerror("Ошибка", f"Не удалось оптимизировать систему: {e}")

    def _on_model_changed(self, event=None):
        """Обработка смены модели."""
        selected = self.model_var.get()
        model_map = {
            "RuGPT-3 Large": "rugpt3large",
            "Qwen3.5-0.8B": "qwen3.5-0.8b",
            "Qwen3.5-2B": "qwen3.5-2b",
            "BitNet 2B": "bitnet-2b"
        }
        
        model_id = model_map.get(selected, "rugpt3large")
        self.add_to_log(f"Смена модели на: {selected}")
        
        try:
            from cogniflex.mlearning.model_selector import MODEL_CONFIGS
            if model_id in MODEL_CONFIGS:
                MODEL_CONFIGS[model_id]["status"] = "ready"
                messagebox.showinfo("Модель", f"Модель {selected} выбрана")
        except Exception as e:
            logger.error(f"Ошибка смены модели: {e}")
            messagebox.showwarning("Модель", f"Модель {selected} недоступна: {e}")

    def show_system_stats(self):
        """Показать статистику системы."""
        try:
            if hasattr(self.integrator, 'get_system_stats'):
                stats = self.integrator.get_system_stats()

                stats_text = "📊 Статистика системы CogniFlex\n\n"
                stats_text += f"Статус: {stats.get('health', {}).get('status', 'Неизвестен')}\n"
                stats_text += f"Компонентов: {stats.get('health', {}).get('components_count', 0)}\n"
                stats_text += f"Всего запросов: {stats.get('metrics', {}).get('total_requests', 0)}\n"
                stats_text += f"Успешных ответов: {stats.get('metrics', {}).get('successful_responses', 0)}\n"
                stats_text += ".2f"
                stats_text += f"Активных запросов: {stats.get('active_requests', 0)}\n\n"

                if 'event_bus_stats' in stats:
                    stats_text += "📡 Событийная шина:\n"
                    stats_text += f"  Всего событий: {stats['event_bus_stats'].get('total_events_triggered', 0)}\n"
                    stats_text += f"  Слушателей: {stats['event_bus_stats'].get('total_listeners', 0)}\n"

                messagebox.showinfo("Статистика системы", stats_text)
            else:
                messagebox.showwarning("Статистика", "Статистика недоступна")

        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            messagebox.showerror("Ошибка", f"Не удалось получить статистику: {e}")

    def stop_system(self):
        """Остановка системы."""
        try:
            if messagebox.askyesno("Подтверждение", "Остановить систему?"):
                self.update_status("Останавливается...")
                if hasattr(self.integrator, 'shutdown'):
                    self.integrator.shutdown()
                self.running = False
                self.root.quit()
                logger.info("Система остановлена через GUI")
        except Exception as e:
            logger.error(f"Ошибка остановки системы: {e}")

    def start_gui(self) -> None:
        """Запуск GUI из CoreBrain после полной инициализации системы."""
        if not self.running:
            logger.info("Запуск GUI после полной инициализации системы...")
            self.start()
        else:
            logger.debug("GUI уже запущен")
    
    def start(self):
        """Запуск интерфейса."""
        if self.running:
            return

        self.running = True
        self.update_status("Готов")

        # Запуск фонового обновления метрик
        self.start_metrics_update()

        logger.info("Интегрированный GUI запущен")
        self.root.mainloop()

    def start_metrics_update(self):
        """Запуск фонового обновления метрик."""
        def update_loop():
            while self.running:
                try:
                    self.root.after(0, self.update_metrics)
                    time.sleep(5)  # Обновление каждые 5 секунд
                except Exception:
                    break

        threading.Thread(target=update_loop, daemon=True).start()

    def stop(self):
        """Остановка интерфейса."""
        self.running = False
        if self.root:
            self.root.quit()


def create_integrated_gui(integrator: CogniFlexIntegrator) -> IntegratedCogniFlexGUI:
    """
    Создание интегрированного GUI для CogniFlex.

    Args:
        integrator: Экземпляр CogniFlexIntegrator

    Returns:
        IntegratedCogniFlexGUI: Экземпляр интегрированного GUI
    """
    return IntegratedCogniFlexGUI(integrator)
