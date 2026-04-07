"""
Tab creation, management, and view switching for ЕВА GUI.
"""
import logging
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger("eva_ai.gui.core")


class TabManagerMixin:
    """Mixin for tab management and view switching."""

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
            try:
                if hasattr(self, 'memory_module') and self.memory_module:
                    self.memory_module.activate()
                else:
                    ttk.Label(self.content_area, text="Модуль памяти недоступен").pack()
            except Exception as e:
                logger.error(f"Error activating memory module: {e}")
                ttk.Label(self.content_area, text="Модуль памяти недоступен").pack()
            self.current_view = view_id
            return
        elif view_id == "system":
            try:
                ttk.Label(self.content_area, text="Модуль системы в разработке", font=("Segoe UI", 12)).pack(pady=20)
            except Exception as e:
                logger.error(f"Error activating SystemTab: {e}")
                ttk.Label(self.content_area, text="Модуль системы недоступен").pack()
            self.current_view = view_id
            return
        
        module = getattr(self, f"{view_id}_module", None)
        if module and hasattr(module, 'activate'):
            self.chat_logger.info(f"Активация модуля: {view_id}")
            module.activate()
        else:
            ttk.Label(self.content_area, text=f"Модуль '{view_id}' недоступен.").pack()
        self.current_view = view_id

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
        """Обновляет визуальное состояние кнопок навигации."""
        pass

    def _init_modules(self):
        """Инициализирует модули GUI."""
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
        
        chat_initialized = False

        try:
            logger.info(" Инициализация chat модуля с приоритетом...")
            from .chat_module import ChatModule

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

        if not chat_initialized:
            try:
                logger.info("Попытка fallback инициализации chat модуля...")
                module_path = 'eva_ai.gui.chat_module'
                if not _validate_module_path(module_path):
                    raise ValueError(f"Module path not allowed: {module_path}")
                module = __import__(module_path, fromlist=['ChatModule'])
                module_class = getattr(module, 'ChatModule')
                self.chat_module = module_class(self)
                chat_initialized = True
                logger.info("[OK] Chat модуль инициализирован через fallback")
            except Exception as e:
                logger.critical(f"Не удалось инициализировать ChatModule даже через fallback: {e}")
                self._chat_init_error = str(e)

        module_map = {
            "memory": ("eva_ai.gui.memory_module", "MemoryModule"),
        }

        for name, (module_path, class_name) in module_map.items():
            try:
                logger.info(f"Инициализация модуля: {name}")
                if not _validate_module_path(module_path):
                    raise ValueError(f"Module path not allowed: {module_path}")
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

        if not hasattr(self, 'chat_module') or self.chat_module is None:
            logger.error("КРИТИЧЕСКАЯ ОШИБКА: Chat модуль не инициализирован после всех попыток!")
            try:
                logger.info("Создание заглушки для chat модуля...")

                class ChatModuleStub:
                    def __init__(self, gui, init_error=None):
                        self.gui = gui
                        self.message_history = []
                        self.init_error = init_error
                        logger.warning(f"Chat модуль заменен заглушкой - функциональность ограничена. Причина: {init_error}")

                    def activate(self):
                        for widget in self.gui.content_area.winfo_children():
                            widget.destroy()

                        from tkinter import ttk
                        frame = ttk.Frame(self.gui.content_area)
                        frame.pack(fill="both", expand=True, padx=20, pady=20)

                        ttk.Label(frame, text="⚠️ Chat модуль недоступен",
                                font=("Segoe UI", 14, "bold")).pack(pady=(0, 10))

                        error_detail = getattr(self, '_chat_init_error', 'Неизвестная ошибка')
                        ttk.Label(frame, text=f"Ошибка инициализации: {error_detail}\n\n"
                                "Проверьте логи для получения подробной информации.",
                                wraplength=400, justify="center").pack(pady=(0, 20))

                        ttk.Button(frame, text="Перезагрузить систему",
                                 command=lambda: self.gui._reboot_system()).pack()

                        logger.warning("Chat модуль заменен заглушкой")

                    def deactivate(self):
                        try:
                            logger.debug("ChatModuleStub деактивирован")
                        except Exception as e:
                            logger.debug(f"Ошибка деактивации ChatModuleStub: {e}")

                self.chat_module = ChatModuleStub(self, init_error=getattr(self, '_chat_init_error', 'Неизвестная ошибка'))
                logger.critical(f"Chat модуль не инициализирован. Установлена заглушка. Ошибка: {getattr(self, '_chat_init_error', 'Неизвестная ошибка')}")
                logger.info("[OK] Заглушка для chat модуля создана")

            except Exception as e:
                logger.critical(f"Не удалось создать даже заглушку для chat модуля: {e}")

        final_chat_status = hasattr(self, 'chat_module') and self.chat_module is not None
        logger.info(f"Итог инициализации модулей: chat={'успешно' if final_chat_status else 'не удалось'}")

        self._switch_view("chat")

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
                    if hasattr(self, 'root') and self.root:
                        for job_id in module._after_jobs:
                            try:
                                self.root.after_cancel(job_id)
                            except Exception:
                                pass
                        module._after_jobs.clear()
                        self.chat_logger.debug(f"Прямая очистка after задач в модуле: {module_name}")
            
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


def _validate_module_path(module_path: str) -> bool:
    if not module_path or ".." in module_path or module_path.startswith("/"):
        return False
    return module_path in frozenset([
        "eva_ai.gui.chat_module",
        "eva_ai.gui.memory_module",
    ])


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
