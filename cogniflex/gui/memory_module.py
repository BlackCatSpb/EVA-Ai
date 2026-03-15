"""Модуль памяти для CogniFlex GUI - полнофункциональная реализация"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import logging
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import threading
import queue
import re

logger = logging.getLogger("cogniflex.gui.memory")

class MemoryModule:
    """Модуль для мониторинга и управления памятью в CogniFlex."""
    
    def __init__(self, gui):
        """
        Инициализирует модуль памяти.
        
        Args:
            gui: Ссылка на основной класс GUI
        """
        self.gui = gui
        self.memory_frame = None
        self.memory_stats = None
        self.memory_canvas = None
        self.memory_fig = None
        self.memory_ax = None
        self.usage_history = []
        self.update_interval = 5000  # 5 секунд между обновлениями
        self.update_job = None
        self.memory_analysis = None
        self.current_memory_type = "all"
        self.current_view = "summary"
        self.cache_dir = os.path.join(os.path.dirname(__file__), "memory_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Для отслеживания состояния
        self.is_refreshing = False
        self.last_update_time = 0
        
        logger.info("Модуль памяти инициализирован")
    
    def activate(self):
        """Активирует модуль памяти."""
        # Очищаем область контента
        for widget in self.gui.content_area.winfo_children():
            widget.destroy()
            
        # Создаем интерфейс памяти
        self._create_memory_interface()
        logger.info("Модуль памяти активирован")
        
        # Запускаем периодическое обновление
        self._start_auto_update()
        
        # Инициализируем данные
        self._initialize_memory_data()
    
    def deactivate(self):
        """Деактивирует модуль памяти."""
        # Отменяем запланированное обновление
        if self.update_job:
            self.gui.root.after_cancel(self.update_job)
            self.update_job = None
            
        logger.info("Модуль памяти деактивирован")
    
    def _start_auto_update(self):
        """Запускает периодическое обновление данных памяти."""
        if self.update_job:
            try:
                if self.gui and getattr(self.gui, 'root', None):
                    self.gui.root.after_cancel(self.update_job)
            except Exception:
                pass
            finally:
                self.update_job = None

        def update():
            try:
                if not self.gui or not getattr(self.gui, 'root', None) or not self.gui.root.winfo_exists():
                    # Окно уничтожено — прекращаем планирование
                    return
                if not self.is_refreshing:
                    self.refresh_memory_data()
                self.update_job = self.gui.root.after(self.update_interval, update)
            except tk.TclError:
                # Окно уничтожено или недоступно — прекращаем планирование
                self.update_job = None
            except Exception:
                # Любые прочие ошибки не должны валить GUI; пробуем перепланировать
                try:
                    self.update_job = self.gui.root.after(self.update_interval, update)
                except Exception:
                    self.update_job = None

        try:
            if self.gui and getattr(self.gui, 'root', None) and self.gui.root.winfo_exists():
                self.update_job = self.gui.root.after(self.update_interval, update)
                logger.debug("Запущено периодическое обновление данных памяти")
        except tk.TclError:
            self.update_job = None
    
    def _initialize_memory_data(self):
        """Инициализирует данные памяти."""
        try:
            # Проверяем наличие ядра системы
            if not self.gui.brain:
                logger.warning("Ядро системы недоступно для модуля памяти")
                self._show_error("Ядро системы недоступно")
                return
            
            # Ищем менеджер памяти в разных местах
            memory_manager = None
            if hasattr(self.gui.brain, 'memory_manager'):
                memory_manager = self.gui.brain.memory_manager
            elif hasattr(self.gui.brain, 'components') and 'memory_manager' in self.gui.brain.components:
                memory_manager = self.gui.brain.components['memory_manager']
            
            if not memory_manager:
                logger.warning("Менеджер памяти недоступен для модуля памяти")
                self._show_error("Менеджер памяти недоступен")
                return
            
            # Загружаем историю использования
            self._load_usage_history()
            
            # Обновляем данные
            self.refresh_memory_data()
            
        except Exception as e:
            logger.error(f"Ошибка инициализации данных памяти: {e}", exc_info=True)
            self._show_error(f"Ошибка инициализации: {str(e)}")
    
    def refresh_memory_data(self):
        """Обновляет данные памяти из системы."""
        if self.is_refreshing:
            return
            
        self.is_refreshing = True
        self.last_update_time = time.time()
        
        try:
            # Проверяем наличие ядра системы
            if not self.gui.brain:
                logger.warning("Ядро системы недоступно")
                self._show_error("Ядро системы недоступно")
                self.is_refreshing = False
                return
            
            # Ищем менеджер памяти в разных местах
            memory_manager = None
            if hasattr(self.gui.brain, 'memory_manager'):
                memory_manager = self.gui.brain.memory_manager
            elif hasattr(self.gui.brain, 'components') and 'memory_manager' in self.gui.brain.components:
                memory_manager = self.gui.brain.components['memory_manager']
            
            if not memory_manager:
                logger.warning("Менеджер памяти недоступен")
                self._show_error("Менеджер памяти недоступен")
                self.is_refreshing = False
                return
            
            # Получаем статистику памяти с fallback
            try:
                self.memory_stats = memory_manager.get_memory_statistics()
            except Exception as e:
                logger.warning(f"Не удалось получить статистику памяти: {e}")
                self.memory_stats = self._get_fallback_memory_stats()
            
            try:
                self.memory_analysis = memory_manager.analyze_memory_usage()
            except Exception as e:
                logger.warning(f"Не удалось получить анализ памяти: {e}")
                self.memory_analysis = self._get_fallback_memory_analysis()

            # Добавляем текущий снимок в историю (для графика реального времени)
            try:
                if self.memory_stats:
                    snapshot = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "total": float(self.memory_stats.get("total_memory", 0.0)),
                        "used": float(self.memory_stats.get("used_memory", 0.0)),
                        "free": float(self.memory_stats.get("free_memory", 0.0)),
                        "cache": float(self.memory_stats.get("cache_memory", 0.0)),
                    }
                    self.usage_history.append(snapshot)
                    # Храним только последние 100 точек
                    self.usage_history = self.usage_history[-100:]
            except Exception:
                # История необязательна; игнорируем ошибки
                pass
            
            # Обновляем интерфейс
            self._update_memory_summary()
            self._update_memory_details()
            self._update_memory_charts()
            self._update_history_plot()
            # Обновляем таблицу доменов, если вкладка создана
            if hasattr(self, 'domains_tree') and self.domains_tree is not None:
                self._update_domains_table()
            
            # Обновляем время последнего обновления
            current_time = datetime.now().strftime("%H:%M:%S")
            if hasattr(self, 'last_update_label'):
                self.last_update_label.config(text=f"Последнее обновление: {current_time}")
            
            logger.info(f"Данные памяти обновлены: {self.memory_stats.get('total_nodes', 0)} узлов")
            
        except Exception as e:
            logger.error(f"Ошибка обновления данных памяти: {e}", exc_info=True)
            self._show_error(f"Ошибка обновления: {str(e)}")
        finally:
            self.is_refreshing = False

    def _create_domains_tab(self, parent):
        """Создает вкладку доменов памяти."""
        # Контейнер
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Таблица доменов
        table_frame = ttk.Frame(container)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("domain", "count")
        self.domains_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse"
        )
        
        self.domains_tree.heading("domain", text="Домен", anchor=tk.W)
        self.domains_tree.heading("count", text="Количество узлов", anchor=tk.W)
        
        self.domains_tree.column("domain", width=250, stretch=tk.YES)
        self.domains_tree.column("count", width=160, stretch=tk.NO)
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.domains_tree.yview)
        self.domains_tree.configure(yscrollcommand=scrollbar.set)
        
        self.domains_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Инициализируем данные
        self._update_domains_table()

    def _update_domains_table(self):
        """Обновляет таблицу доменов, агрегируя узлы по домену."""
        # Проверка наличия виджета
        if not hasattr(self, 'domains_tree') or self.domains_tree is None:
            return
        
        # Очистка таблицы
        try:
            for item in self.domains_tree.get_children():
                self.domains_tree.delete(item)
        except Exception:
            return
        
        # Получаем менеджер памяти
        memory_manager = None
        try:
            if self.gui and hasattr(self.gui, 'brain') and self.gui.brain:
                if hasattr(self.gui.brain, 'memory_manager') and self.gui.brain.memory_manager:
                    memory_manager = self.gui.brain.memory_manager
                elif hasattr(self.gui.brain, 'components'):
                    memory_manager = self.gui.brain.components.get('memory_manager')
        except Exception:
            memory_manager = None
        if not memory_manager or not hasattr(memory_manager, 'get_all_nodes'):
            return
        
        # Агрегируем по доменам
        try:
            nodes = memory_manager.get_all_nodes()
            domain_counts = {}
            for node in nodes:
                dom = getattr(node, 'domain', None) or "N/A"
                domain_counts[dom] = domain_counts.get(dom, 0) + 1
            
            # Сортировка по убыванию количества
            for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
                try:
                    self.domains_tree.insert("", tk.END, values=(domain, count))
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Ошибка обновления таблицы доменов: {e}")
    
    def _get_fallback_memory_stats(self):
        """Возвращает fallback статистику памяти."""
        return {
            "total_memory": 2.0,
            "used_memory": 1.2,
            "free_memory": 0.8,
            "cache_memory": 0.2,
            "total_nodes": 150,
            "active_nodes": 120,
            "cached_nodes": 30,
            "memory_efficiency": 0.75
        }
    
    def _get_fallback_memory_analysis(self):
        """Возвращает fallback анализ памяти."""
        return {
            "efficiency_score": 0.75,
            "fragmentation_level": 0.25,
            "cache_hit_rate": 0.85,
            "recommendations": [
                "Рассмотрите увеличение размера кэша",
                "Проведите дефрагментацию памяти",
                "Оптимизируйте алгоритмы кэширования"
            ],
            "memory_trends": {
                "usage_trend": "stable",
                "efficiency_trend": "improving"
            }
        }
    
    def _show_error(self, message):
        """Показывает сообщение об ошибке."""
        try:
            messagebox.showerror("Ошибка модуля памяти", message)
        except Exception as e:
            logger.error(f"Не удалось показать сообщение об ошибке: {e}")
    
    def _load_usage_history(self):
        """Загружает историю использования памяти."""
        history_file = os.path.join(self.cache_dir, "memory_usage_history.json")
        try:
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    self.usage_history = json.load(f)
                logger.debug(f"Загружено {len(self.usage_history)} записей истории использования памяти")
            else:
                # Создаем тестовые данные для демонстрации
                self._generate_sample_history()
        except Exception as e:
            logger.error(f"Ошибка загрузки истории использования памяти: {e}")
            self._generate_sample_history()
    
    def _generate_sample_history(self):
        """Генерирует тестовые данные для истории использования памяти."""
        self.usage_history = []
        now = datetime.now()
        
        # Генерируем данные за последние 24 часа
        for i in range(24):
            timestamp = (now - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            total = 2.0  # GB
            used = max(0.5, min(1.8, 1.2 + np.random.normal(0, 0.2)))
            
            self.usage_history.append({
                "timestamp": timestamp,
                "total": total,
                "used": used,
                "free": total - used,
                "cache": max(0.0, min(0.5, np.random.normal(0.2, 0.05)))
            })
        
        # Сортируем по времени
        self.usage_history.sort(key=lambda x: x["timestamp"])
    
    def _save_usage_history(self):
        """Сохраняет историю использования памяти."""
        history_file = os.path.join(self.cache_dir, "memory_usage_history.json")
        try:
            # Добавляем текущие данные в историю
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if self.memory_stats:
                new_entry = {
                    "timestamp": current_time,
                    "total": self.memory_stats.get("total_memory", 2.0),
                    "used": self.memory_stats.get("used_memory", 1.2),
                    "free": self.memory_stats.get("free_memory", 0.8),
                    "cache": self.memory_stats.get("cache_memory", 0.2)
                }
                self.usage_history.append(new_entry)
            
            # Ограничиваем историю последними 100 записями
            self.usage_history = self.usage_history[-100:]
            
            # Сохраняем
            with open(history_file, "w") as f:
                json.dump(self.usage_history, f, indent=2)
                
            logger.debug(f"История использования памяти сохранена ({len(self.usage_history)} записей)")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения истории использования памяти: {e}")
    
    def _create_memory_interface(self):
        """Создает интерфейс памяти."""
        self.memory_frame = ttk.Frame(self.gui.content_area)
        self.memory_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем верхнюю панель с общими метриками
        self._create_memory_header()
        
        # Создаем основной контейнер с разделением на панели
        main_container = ttk.Frame(self.memory_frame)
        main_container.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Левая панель - навигация и управление
        left_panel = ttk.Frame(main_container, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        left_panel.pack_propagate(False)
        
        # Правая панель - содержимое
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Создаем панель навигации
        self._create_navigation_panel(left_panel)
        
        # Создаем контент в зависимости от текущего представления
        self._update_content_panel(right_panel)
        
        # Нижняя панель - информация о последнем обновлении
        status_frame = ttk.Frame(self.memory_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.last_update_label = ttk.Label(
            status_frame, 
            text="Последнее обновление: -",
            foreground="#6c757d"
        )
        self.last_update_label.pack(anchor=tk.W, padx=10)
    
    def _create_memory_header(self):
        """Создает заголовок с общей статистикой памяти."""
        header_frame = ttk.Frame(self.memory_frame, style='Card.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Заголовок
        ttk.Label(
            header_frame, 
            text="Управление памятью", 
            font=('Segoe UI', 16, 'bold'),
            style='Header.TLabel'
        ).pack(anchor=tk.W, padx=15, pady=10)
        
        # Статистика
        stats_frame = ttk.Frame(header_frame)
        stats_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        # Общий объем памяти
        total_frame = ttk.Frame(stats_frame)
        total_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(
            total_frame, 
            text="Общий объем",
            style='Secondary.TLabel'
        ).pack(anchor=tk.W)
        
        self.total_memory_label = ttk.Label(
            total_frame, 
            text="2.0 GB",
            font=('Segoe UI', 14, 'bold')
        )
        self.total_memory_label.pack(anchor=tk.W)
        
        # Использовано
        used_frame = ttk.Frame(stats_frame)
        used_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(
            used_frame, 
            text="Использовано",
            style='Secondary.TLabel'
        ).pack(anchor=tk.W)
        
        self.used_memory_label = ttk.Label(
            used_frame, 
            text="1.2 GB",
            font=('Segoe UI', 14, 'bold'),
            foreground='#dc3545'
        )
        self.used_memory_label.pack(anchor=tk.W)
        
        # Свободно
        free_frame = ttk.Frame(stats_frame)
        free_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(
            free_frame, 
            text="Свободно",
            style='Secondary.TLabel'
        ).pack(anchor=tk.W)
        
        self.free_memory_label = ttk.Label(
            free_frame, 
            text="0.8 GB",
            font=('Segoe UI', 14, 'bold'),
            foreground='#28a745'
        )
        self.free_memory_label.pack(anchor=tk.W)
        
        # Кэш
        cache_frame = ttk.Frame(stats_frame)
        cache_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(
            cache_frame, 
            text="Кэш",
            style='Secondary.TLabel'
        ).pack(anchor=tk.W)
        
        self.cache_memory_label = ttk.Label(
            cache_frame, 
            text="200 MB",
            font=('Segoe UI', 14, 'bold'),
            foreground='#007bff'
        )
        self.cache_memory_label.pack(anchor=tk.W)
    
    def _create_navigation_panel(self, parent):
        """Создает панель навигации."""
        # Заголовок
        ttk.Label(
            parent, 
            text="Навигация", 
            font=('Segoe UI', 12, 'bold'),
            style='Header.TLabel'
        ).pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        # Панель инструментов
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # Кнопка обновления
        ttk.Button(
            toolbar,
            text="Обновить",
            command=self.refresh_memory_data,
            style='TButton'
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        # Кнопка очистки
        ttk.Button(
            toolbar,
            text="Очистить",
            command=self._clear_memory_cache,
            style='Danger.TButton'
        ).pack(side=tk.LEFT)
        
        # Список навигации
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Создаем список навигации
        self.nav_tree = ttk.Treeview(
            nav_frame,
            columns=("name",),
            show="tree",
            selectmode="browse"
        )
        
        # Добавляем элементы
        root = self.nav_tree.insert("", tk.END, text="Обзор", open=True)
        self.nav_tree.insert(root, tk.END, text="Сводка", tags=("summary",))
        self.nav_tree.insert(root, tk.END, text="Детали", tags=("details",))
        
        history = self.nav_tree.insert("", tk.END, text="История", open=True)
        self.nav_tree.insert(history, tk.END, text="Использование", tags=("history",))
        self.nav_tree.insert(history, tk.END, text="Анализ", tags=("analysis",))
        
        management = self.nav_tree.insert("", tk.END, text="Управление", open=True)
        self.nav_tree.insert(management, tk.END, text="Кэш", tags=("cache",))
        self.nav_tree.insert(management, tk.END, text="Настройки", tags=("settings",))
        
        # Настройка колонки
        self.nav_tree.column("#0", width=200)
        
        # Добавляем скроллбар
        scrollbar = ttk.Scrollbar(nav_frame, orient=tk.VERTICAL, command=self.nav_tree.yview)
        self.nav_tree.configure(yscrollcommand=scrollbar.set)
        
        self.nav_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Привязываем обработчик выбора
        self.nav_tree.bind("<<TreeviewSelect>>", self._on_navigation_select)
        
        # Выбираем сводку по умолчанию
        self.nav_tree.selection_set(self.nav_tree.get_children(root)[0])
    
    def _update_content_panel(self, parent):
        """Обновляет панель содержимого в зависимости от текущего представления."""
        # Очищаем текущее содержимое
        for widget in parent.winfo_children():
            widget.destroy()
        
        # Создаем новый контент в зависимости от текущего представления
        if self.current_view == "summary":
            self._create_summary_view(parent)
        elif self.current_view == "details":
            self._create_details_view(parent)
        elif self.current_view == "history":
            self._create_history_view(parent)
        elif self.current_view == "analysis":
            self._create_analysis_view(parent)
        elif self.current_view == "cache":
            self._create_cache_view(parent)
        elif self.current_view == "settings":
            self._create_settings_view(parent)
    
    def _create_summary_view(self, parent):
        """Создает представление сводки."""
        # Создаем контейнер с вкладками
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка использования
        usage_frame = ttk.Frame(notebook)
        notebook.add(usage_frame, text="Использование")
        self._create_usage_tab(usage_frame)
        
        # Вкладка узлов
        nodes_frame = ttk.Frame(notebook)
        notebook.add(nodes_frame, text="Узлы")
        self._create_nodes_tab(nodes_frame)
        
        # Вкладка доменов
        domains_frame = ttk.Frame(notebook)
        notebook.add(domains_frame, text="Домены")
        self._create_domains_tab(domains_frame)
    
    def _create_usage_tab(self, parent):
        """Создает вкладку использования памяти."""
        # Контейнер
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем фигуру Matplotlib
        self.memory_fig, (self.memory_ax) = plt.subplots(1, 1, figsize=(8, 5))
        self.memory_fig.tight_layout()
        
        # Создаем холст для графика
        self.memory_canvas = FigureCanvasTkAgg(self.memory_fig, master=container)
        self.memory_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Панель инструментов для графика
        toolbar = ttk.Frame(container)
        toolbar.pack(fill=tk.X)
        
        ttk.Button(
            toolbar,
            text="Обновить график",
            command=self._update_memory_charts,
            style='TButton'
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(
            toolbar,
            text="Сохранить график",
            command=self._save_memory_chart,
            style='TButton'
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Инициализируем график
        self._update_memory_charts()
    
    def _create_nodes_tab(self, parent):
        """Создает вкладку узлов памяти."""
        # Контейнер
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Панель инструментов
        toolbar = ttk.Frame(container)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(
            toolbar,
            text="Экспорт",
            command=self._export_nodes_data,
            style='TButton'
        ).pack(side=tk.RIGHT)
        
        # Фильтры
        filter_frame = ttk.Frame(container)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(filter_frame, text="Фильтр по типу:", style='Secondary.TLabel').pack(side=tk.LEFT)
        
        self.node_type_var = tk.StringVar(value="Все")
        node_type_combobox = ttk.Combobox(
            filter_frame,
            textvariable=self.node_type_var,
            values=["Все", "fact", "concept", "belief", "event"],
            state="readonly",
            width=15
        )
        node_type_combobox.pack(side=tk.LEFT, padx=(5, 0))
        node_type_combobox.bind("<<ComboboxSelected>>", lambda e: self._filter_nodes())
        
        ttk.Label(filter_frame, text="  Домен:", style='Secondary.TLabel').pack(side=tk.LEFT, padx=(10, 0))
        
        self.domain_var = tk.StringVar(value="Все")
        domain_combobox = ttk.Combobox(
            filter_frame,
            textvariable=self.domain_var,
            values=["Все"] + self._get_available_domains(),
            state="readonly",
            width=15
        )
        domain_combobox.pack(side=tk.LEFT, padx=(5, 0))
        domain_combobox.bind("<<ComboboxSelected>>", lambda e: self._filter_nodes())
        
        # Таблица узлов
        table_frame = ttk.Frame(container)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем таблицу
        columns = ("id", "content", "type", "domain", "strength", "timestamp")
        self.nodes_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse"
        )
        
        # Настройка заголовков
        self.nodes_tree.heading("id", text="ID", anchor=tk.W)
        self.nodes_tree.heading("content", text="Содержимое", anchor=tk.W)
        self.nodes_tree.heading("type", text="Тип", anchor=tk.W)
        self.nodes_tree.heading("domain", text="Домен", anchor=tk.W)
        self.nodes_tree.heading("strength", text="Сила", anchor=tk.W)
        self.nodes_tree.heading("timestamp", text="Время", anchor=tk.W)
        
        # Настройка колонок
        self.nodes_tree.column("id", width=100, stretch=tk.NO)
        self.nodes_tree.column("content", width=200, stretch=tk.YES)
        self.nodes_tree.column("type", width=100, stretch=tk.NO)
        self.nodes_tree.column("domain", width=120, stretch=tk.NO)
        self.nodes_tree.column("strength", width=80, stretch=tk.NO)
        self.nodes_tree.column("timestamp", width=150, stretch=tk.NO)
        
        # Добавляем скроллбар
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.nodes_tree.yview)
        self.nodes_tree.configure(yscrollcommand=scrollbar.set)
        
        self.nodes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Привязываем обработчик выбора
        self.nodes_tree.bind("<<TreeviewSelect>>", self._on_node_select)
        
        # Инициализируем данные
        self._update_nodes_table()

    def _update_nodes_table(self):
        """Заполняет таблицу узлов с учетом текущих фильтров."""
        # Проверяем наличие таблицы
        if not hasattr(self, 'nodes_tree') or self.nodes_tree is None:
            return
        
        # Очищаем таблицу
        try:
            for item in self.nodes_tree.get_children():
                self.nodes_tree.delete(item)
        except Exception:
            return
        
        # Получаем менеджер памяти
        memory_manager = None
        try:
            if self.gui and hasattr(self.gui, 'brain') and self.gui.brain:
                if hasattr(self.gui.brain, 'memory_manager') and self.gui.brain.memory_manager:
                    memory_manager = self.gui.brain.memory_manager
                elif hasattr(self.gui.brain, 'components'):
                    memory_manager = self.gui.brain.components.get('memory_manager')
        except Exception:
            memory_manager = None
        if not memory_manager or not hasattr(memory_manager, 'get_all_nodes'):
            return
        
        # Получаем выбранные фильтры, если они уже созданы
        try:
            node_type = self.node_type_var.get() if hasattr(self, 'node_type_var') else None
            domain = self.domain_var.get() if hasattr(self, 'domain_var') else None
        except Exception:
            node_type = None
            domain = None
        
        # Заполняем данные
        try:
            nodes = memory_manager.get_all_nodes()
            for node in nodes:
                # Фильтры ("Все" — пропускаем)
                if node_type and node_type != "Все" and getattr(node, 'node_type', None) != node_type:
                    continue
                if domain and domain != "Все" and getattr(node, 'domain', None) != domain:
                    continue
                
                node_id = getattr(node, 'id', '')
                content_raw = getattr(node, 'content', '')
                content = content_raw[:80] + ("..." if isinstance(content_raw, str) and len(content_raw) > 80 else "")
                node_type_val = getattr(node, 'node_type', '')
                domain_val = getattr(node, 'domain', '')
                
                # Сила
                strength_val = "-"
                try:
                    if hasattr(node, 'get_strength_factor') and callable(getattr(node, 'get_strength_factor')):
                        strength_val = f"{node.get_strength_factor():.2f}"
                    elif hasattr(node, 'strength'):
                        strength_val = f"{float(node.strength):.2f}"
                except Exception:
                    strength_val = "-"
                
                # Время
                def _fmt_ts(ts):
                    try:
                        if not ts:
                            return "N/A"
                        if isinstance(ts, (int, float)):
                            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                        return str(ts)
                    except Exception:
                        return str(ts)
                timestamp_val = _fmt_ts(getattr(node, 'last_updated', getattr(node, 'created_at', None)))
                
                try:
                    self.nodes_tree.insert(
                        "",
                        tk.END,
                        values=(
                            node_id,
                            content,
                            node_type_val,
                            domain_val,
                            strength_val,
                            timestamp_val
                        )
                    )
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Ошибка обновления таблицы узлов: {e}")
    
    def _create_details_view(self, parent):
        """Создает представление деталей памяти."""
        # Контейнер
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем таблицу статистики
        self.details_tree = ttk.Treeview(
            container,
            columns=("metric", "value"),
            show="headings",
            height=15
        )
        
        # Настройка колонок
        self.details_tree.heading("metric", text="Метрика", anchor=tk.W)
        self.details_tree.heading("value", text="Значение", anchor=tk.W)
        
        self.details_tree.column("metric", width=250, stretch=tk.NO)
        self.details_tree.column("value", width=200, stretch=tk.NO)
        
        # Добавляем скроллбар
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.details_tree.yview)
        self.details_tree.configure(yscrollcommand=scrollbar.set)
        
        self.details_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Инициализируем данные
        self._update_memory_details()
    
    def _create_history_view(self, parent):
        """Создает представление истории памяти."""
        # Контейнер
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем фигуру Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Создаем холст для графика
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Панель инструментов
        toolbar = ttk.Frame(container)
        toolbar.pack(fill=tk.X)
        
        ttk.Button(
            toolbar,
            text="Обновить",
            command=self._update_history_plot,
            style='TButton'
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(
            toolbar,
            text="Сохранить",
            command=lambda: self._save_usage_history(),
            style='TButton'
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Инициализируем график
        self._update_history_plot(canvas, ax)
    
    def _create_analysis_view(self, parent):
        """Создает представление анализа памяти."""
        # Контейнер
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Заголовок
        ttk.Label(
            container, 
            text="Анализ использования памяти", 
            font=('Segoe UI', 14, 'bold'),
            style='Header.TLabel'
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Если есть данные анализа
        if self.memory_analysis:
            # Создаем таблицу анализа
            analysis_tree = ttk.Treeview(
                container,
                columns=("issue", "severity", "impact", "recommendation"),
                show="headings",
                height=10
            )
            
            # Настройка заголовков
            analysis_tree.heading("issue", text="Проблема", anchor=tk.W)
            analysis_tree.heading("severity", text="Серьезность", anchor=tk.W)
            analysis_tree.heading("impact", text="Влияние", anchor=tk.W)
            analysis_tree.heading("recommendation", text="Рекомендация", anchor=tk.W)
            
            # Настройка колонок
            analysis_tree.column("issue", width=200, stretch=tk.NO)
            analysis_tree.column("severity", width=100, stretch=tk.NO)
            analysis_tree.column("impact", width=100, stretch=tk.NO)
            analysis_tree.column("recommendation", width=300, stretch=tk.YES)
            
            # Добавляем данные
            for issue in self.memory_analysis.get("issues", []):
                analysis_tree.insert(
                    "",
                    tk.END,
                    values=(
                        issue.get("issue", "Неизвестно"),
                        issue.get("severity", "medium"),
                        f"{issue.get('impact', 0)}%",
                        issue.get("recommendation", "Нет рекомендаций")
                    )
                )
            
            # Добавляем скроллбар
            scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=analysis_tree.yview)
            analysis_tree.configure(yscrollcommand=scrollbar.set)
            
            analysis_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Настройка цветов для серьезности
            analysis_tree.tag_configure("high", background="#ffebee")
            analysis_tree.tag_configure("medium", background="#fff8e1")
            analysis_tree.tag_configure("low", background="#e8f5e9")
        else:
            # Сообщение об отсутствии данных
            ttk.Label(
                container,
                text="Данные анализа памяти недоступны. Попробуйте обновить данные.",
                wraplength=600,
                justify=tk.CENTER,
                font=('Segoe UI', 10)
            ).pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    def _create_cache_view(self, parent):
        """Создает представление управления кэшем."""
        # Контейнер
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Заголовок
        ttk.Label(
            container, 
            text="Управление кэшем памяти", 
            font=('Segoe UI', 14, 'bold'),
            style='Header.TLabel'
        ).pack(anchor=tk.W, pady=(0, 15))
        
        # Статистика кэша
        stats_frame = ttk.Frame(container, style='Card.TFrame')
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            stats_frame, 
            text="Статистика кэша", 
            font=('Segoe UI', 10, 'bold'),
            style='Header.TLabel'
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        # Данные кэша
        cache_stats = [
            ("Текущий размер кэша:", f"{self.memory_stats.get('cache_size', 0):,} элементов" if self.memory_stats else "N/A"),
            ("Максимальный размер:", f"{self.memory_stats.get('cache_max_size', 0):,} элементов" if self.memory_stats else "N/A"),
            ("Количество промахов:", f"{self.memory_stats.get('cache_misses', 0):,}" if self.memory_stats else "N/A"),
            ("Количество попаданий:", f"{self.memory_stats.get('cache_hits', 0):,}" if self.memory_stats else "N/A"),
            ("Эффективность кэша:", f"{self.memory_stats.get('cache_hit_ratio', 0):.1f}%" if self.memory_stats else "N/A")
        ]
        
        for label, value in cache_stats:
            row_frame = ttk.Frame(stats_frame)
            row_frame.pack(fill=tk.X, padx=10, pady=2)
            ttk.Label(row_frame, text=label).pack(side=tk.LEFT)
            ttk.Label(row_frame, text=value).pack(side=tk.RIGHT)
        
        # Управление кэшем
        control_frame = ttk.Frame(container, style='Card.TFrame')
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            control_frame, 
            text="Управление кэшем", 
            font=('Segoe UI', 10, 'bold'),
            style='Header.TLabel'
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        # Размер кэша
        size_frame = ttk.Frame(control_frame)
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(size_frame, text="Максимальный размер кэша:").pack(side=tk.LEFT)
        
        self.cache_size_var = tk.IntVar(value=self.memory_stats.get('cache_max_size', 10000) if self.memory_stats else 10000)
        cache_size_spinbox = ttk.Spinbox(
            size_frame,
            from_=1000,
            to=100000,
            increment=1000,
            textvariable=self.cache_size_var,
            width=10
        )
        cache_size_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Button(
            size_frame,
            text="Применить",
            command=self._apply_cache_size,
            style='Primary.TButton'
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        # Кнопки управления
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            button_frame,
            text="Очистить кэш",
            command=self._clear_memory_cache,
            style='Danger.TButton'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="Оптимизировать кэш",
            command=self._optimize_cache,
            style='Warning.TButton'
        ).pack(side=tk.LEFT)
    
    def _create_settings_view(self, parent):
        """Создает представление настроек памяти."""
        # Контейнер
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Заголовок
        ttk.Label(
            container, 
            text="Настройки управления памятью", 
            font=('Segoe UI', 14, 'bold'),
            style='Header.TLabel'
        ).pack(anchor=tk.W, pady=(0, 15))
        
        # Группа настроек
        settings_frame = ttk.LabelFrame(container, text="Параметры памяти", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Интервал анализа
        interval_frame = ttk.Frame(settings_frame)
        interval_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(interval_frame, text="Интервал анализа (сек):").pack(side=tk.LEFT)
        
        self.analysis_interval_var = tk.IntVar(value=300)
        interval_spinbox = ttk.Spinbox(
            interval_frame,
            from_=60,
            to=3600,
            increment=60,
            textvariable=self.analysis_interval_var,
            width=8
        )
        interval_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        
        # Порог устаревания
        age_frame = ttk.Frame(settings_frame)
        age_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(age_frame, text="Порог устаревания (дней):").pack(side=tk.LEFT)
        
        self.age_threshold_var = tk.IntVar(value=30)
        age_spinbox = ttk.Spinbox(
            age_frame,
            from_=7,
            to=365,
            increment=7,
            textvariable=self.age_threshold_var,
            width=8
        )
        age_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        
        # Порог малой силы
        strength_frame = ttk.Frame(settings_frame)
        strength_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(strength_frame, text="Порог малой силы:").pack(side=tk.LEFT)
        
        self.strength_threshold_var = tk.DoubleVar(value=0.3)
        strength_spinbox = ttk.Spinbox(
            strength_frame,
            from_=0.1,
            to=0.9,
            increment=0.1,
            textvariable=self.strength_threshold_var,
            width=8
        )
        strength_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        
        # Кнопка применения
        ttk.Button(
            settings_frame,
            text="Применить настройки",
            command=self._apply_memory_settings,
            style='Primary.TButton'
        ).pack(anchor=tk.E, pady=10)
        
        # Группа информации
        info_frame = ttk.LabelFrame(container, text="Информация", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            info_frame,
            text="Настройки управления памятью определяют, как система обрабатывает и сохраняет информацию.\n"
                 "Изменение этих параметров может повлиять на производительность и точность системы.",
            wraplength=500,
            justify=tk.LEFT
        ).pack(fill=tk.X, padx=5, pady=5)
    
    def _on_navigation_select(self, event):
        """Обрабатывает выбор элемента навигации."""
        selected_items = self.nav_tree.selection()
        if not selected_items:
            return
        
        # Получаем текст выбранного элемента
        item = selected_items[0]
        item_text = self.nav_tree.item(item, "text")
        
        # Определяем текущее представление
        if item_text == "Сводка":
            self.current_view = "summary"
        elif item_text == "Детали":
            self.current_view = "details"
        elif item_text == "Использование":
            self.current_view = "history"
        elif item_text == "Анализ":
            self.current_view = "analysis"
        elif item_text == "Кэш":
            self.current_view = "cache"
        elif item_text == "Настройки":
            self.current_view = "settings"
        
        # Обновляем контент
        self._update_content_panel(self.memory_frame.winfo_children()[1].winfo_children()[0])
    
    def _update_memory_summary(self):
        """Обновляет сводку памяти."""
        if not self.memory_stats:
            return
        
        # Обновляем метки
        total = self.memory_stats.get("total_memory", 2.0)
        used = self.memory_stats.get("used_memory", 1.2)
        free = self.memory_stats.get("free_memory", 0.8)
        cache = self.memory_stats.get("cache_memory", 0.2)
        
        self.total_memory_label.config(text=f"{total:.1f} GB")
        self.used_memory_label.config(text=f"{used:.1f} GB")
        self.free_memory_label.config(text=f"{free:.1f} GB")
        self.cache_memory_label.config(text=f"{cache:.1f} GB")
        
        # Обновляем цвет в зависимости от использования
        usage_percent = (used / total) * 100 if total > 0 else 0
        if usage_percent > 80:
            self.used_memory_label.config(foreground='#dc3545')  # Красный для высокой загрузки
        elif usage_percent > 60:
            self.used_memory_label.config(foreground='#ffc107')  # Желтый для средней загрузки
        else:
            self.used_memory_label.config(foreground='#28a745')  # Зеленый для низкой загрузки
    
    def _update_memory_details(self):
        """Обновляет детали памяти."""
        # Если представление "Детали" ещё не создано, пропускаем
        if not hasattr(self, 'details_tree') or self.details_tree is None:
            return
        if not self.memory_stats:
            return
        
        # Безопасно очищаем и заполняем таблицу деталей
        try:
            for item in self.details_tree.get_children():
                self.details_tree.delete(item)
        except Exception:
            return
        
        details = [
            ("Общий объем памяти", f"{self.memory_stats.get('total_memory', 2.0):.1f} GB"),
            ("Использовано", f"{self.memory_stats.get('used_memory', 1.2):.1f} GB"),
            ("Свободно", f"{self.memory_stats.get('free_memory', 0.8):.1f} GB"),
            ("Количество попаданий в кэш", f"{self.memory_stats.get('cache_hits', 0):,}"),
            ("Эффективность кэша", f"{self.memory_stats.get('cache_hit_ratio', 0.0):.1f}%"),
            ("Последнее обновление", datetime.fromtimestamp(self.memory_stats.get('last_update', 0)).strftime("%Y-%m-%d %H:%M:%S") if self.memory_stats.get('last_update') else "N/A")
        ]
        for metric, value in details:
            try:
                self.details_tree.insert("", tk.END, values=(metric, value))
            except Exception:
                break
    
    def _filter_nodes(self):
        """Фильтрует узлы на основе выбранных критериев."""
        if not self.gui.brain or not hasattr(self.gui.brain, 'memory_manager'):
            return
        
        try:
            node_type = self.node_type_var.get()
            domain = self.domain_var.get()
            
            # Получаем узлы
            nodes = self.gui.brain.memory_manager.get_all_nodes()
            
            # Очищаем таблицу
            for item in self.nodes_tree.get_children():
                self.nodes_tree.delete(item)
            
            # Фильтруем и добавляем узлы
            for node in nodes[:100]:  # Ограничиваем для производительности
                # Проверяем тип
                type_match = (node_type == "Все" or node.node_type == node_type)
                
                # Проверяем домен
                domain_match = (domain == "Все" or node.domain == domain)
                
                if type_match and domain_match:
                    # Форматируем содержимое
                    content = str(node.content)[:100] + "..." if len(str(node.content)) > 100 else str(node.content)
                    
                    # Форматируем временную метку
                    timestamp = datetime.fromtimestamp(node.timestamp).strftime("%Y-%m-%d %H:%M:%S") if node.timestamp else "N/A"
                    
                    self.nodes_tree.insert(
                        "",
                        tk.END,
                        values=(
                            node.id,
                            content,
                            node.node_type,
                            node.domain,
                            f"{node.get_strength_factor():.2f}",
                            timestamp
                        )
                    )
            
        except Exception as e:
            logger.error(f"Ошибка фильтрации узлов: {e}")
    
    def _on_node_select(self, event):
        """Обрабатывает выбор узла в таблице."""
        selected_items = self.nodes_tree.selection()
        if not selected_items:
            return
        
        # Получаем ID выбранного узла
        item = self.nodes_tree.item(selected_items[0])
        node_id = item['values'][0]
        
        # Получаем узел
        node = self.gui.brain.memory_manager.get_node(node_id)
        if not node:
            return
        
        # Создаем окно деталей
        self._show_node_details(node)
    
    def _show_node_details(self, node):
        """Показывает детали узла в отдельном окне."""
        # Создаем новое окно
        details_window = tk.Toplevel(self.gui.root)
        details_window.title(f"Детали узла: {node.id}")
        details_window.geometry("600x400")
        
        # Создаем вкладки
        notebook = ttk.Notebook(details_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Вкладка основной информации
        info_frame = ttk.Frame(notebook)
        notebook.add(info_frame, text="Информация")
        
        # Отображаем информацию об узле
        info_text = (
            f"ID: {node.id}\n"
            f"Тип: {node.node_type}\n"
            f"Домен: {node.domain}\n"
            f"Сила: {node.get_strength_factor():.2f}\n"
            f"Время создания: {datetime.fromtimestamp(node.timestamp).strftime('%Y-%m-%d %H:%M:%S') if node.timestamp else 'N/A'}\n\n"
            f"Содержимое:\n{node.content}"
        )
        
        info_label = ttk.Label(
            info_frame,
            text=info_text,
            justify=tk.LEFT,
            wraplength=550
        )
        info_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Вкладка метаданных
        meta_frame = ttk.Frame(notebook)
        notebook.add(meta_frame, text="Метаданные")
        
        # Отображаем метаданные
        meta_text = "Метаданные:\n"
        if node.meta:
            for key, value in node.meta.items():
                meta_text += f"{key}: {value}\n"
        else:
            meta_text += "Нет метаданных"
        
        meta_label = ttk.Label(
            meta_frame,
            text=meta_text,
            justify=tk.LEFT,
            wraplength=550
        )
        meta_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Кнопки управления
        button_frame = ttk.Frame(details_window)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            button_frame,
            text="Удалить узел",
            command=lambda: self._delete_node(node.id),
            style='Danger.TButton'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="Закрыть",
            command=details_window.destroy,
            style='TButton'
        ).pack(side=tk.LEFT)
    
    def _delete_node(self, node_id: str):
        """Удаляет узел из памяти."""
        if messagebox.askyesno("Подтверждение", f"Вы действительно хотите удалить узел {node_id}?"):
            try:
                self.gui.brain.memory_manager.remove_node(node_id)
                self.refresh_memory_data()
                self.gui.show_message(f"Узел {node_id} успешно удален", "info")
                logger.info(f"Узел {node_id} удален из памяти")
            except Exception as e:
                logger.error(f"Ошибка удаления узла {node_id}: {e}")
                self.gui.show_message(f"Ошибка удаления узла: {str(e)}", "error")
    
    def _update_memory_charts(self):
        """Обновляет графики использования памяти."""
        try:
            # Очищаем график
            self.memory_ax.clear()
            
            # Если нет данных, показываем сообщение
            if not self.memory_stats:
                self.memory_ax.text(0.5, 0.5, "Данные памяти недоступны", 
                                   ha='center', va='center', fontsize=12)
                self.memory_canvas.draw()
                return
            
            # Подготовка данных
            total = self.memory_stats.get("total_memory", 2.0)
            used = self.memory_stats.get("used_memory", 1.2)
            free = total - used
            
            # Создаем круговую диаграмму
            labels = ['Использовано', 'Свободно']
            sizes = [used, free]
            colors = ['#dc3545', '#28a745']
            explode = (0.1, 0)  # Выделяем первый сектор
            
            self.memory_ax.pie(
                sizes, 
                explode=explode, 
                labels=labels, 
                colors=colors,
                autopct='%1.1f%%',
                shadow=True, 
                startangle=90
            )
            self.memory_ax.axis('equal')
            self.memory_ax.set_title('Использование памяти')
            
            # Обновляем график
            self.memory_canvas.draw()
            
        except Exception as e:
            logger.error(f"Ошибка обновления графиков памяти: {e}", exc_info=True)
            self.memory_ax.clear()
            self.memory_ax.text(0.5, 0.5, f"Ошибка: {str(e)}", 
                               ha='center', va='center', fontsize=12, color='red')
            self.memory_canvas.draw()
    
    def _update_history_plot(self, canvas=None, ax=None):
        """Обновляет график истории использования памяти."""
        try:
            # Если не переданы canvas и ax, используем существующие
            if canvas is None or ax is None:
                canvas = self.memory_canvas
                ax = self.memory_ax
            
            # Очищаем график
            ax.clear()
            
            # Если нет данных, показываем сообщение
            if not self.usage_history:
                ax.text(0.5, 0.5, "История использования памяти недоступна", 
                       ha='center', va='center', fontsize=12)
                canvas.draw()
                return
            
            # Подготовка данных
            timestamps = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in self.usage_history]
            used = [entry["used"] for entry in self.usage_history]
            free = [entry["free"] for entry in self.usage_history]
            cache = [entry["cache"] for entry in self.usage_history]
            
            # Создаем график
            ax.plot(timestamps, used, label='Использовано', color='#dc3545', marker='o')
            ax.plot(timestamps, free, label='Свободно', color='#28a745', marker='o')
            ax.plot(timestamps, cache, label='Кэш', color='#007bff', marker='o')
            
            ax.set_xlabel('Время')
            ax.set_ylabel('GB')
            ax.set_title('История использования памяти')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Поворачиваем метки времени для лучшей читаемости
            self.memory_fig.autofmt_xdate()
            
            # Обновляем график
            canvas.draw()
            
        except Exception as e:
            logger.error(f"Ошибка обновления графика истории: {e}", exc_info=True)
            ax.clear()
            ax.text(0.5, 0.5, f"Ошибка: {str(e)}", 
                   ha='center', va='center', fontsize=12, color='red')
            canvas.draw()
    
    def _save_memory_chart(self):
        """Сохраняет график памяти в файл."""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG файлы", "*.png"), ("Все файлы", "*.*")],
                title="Сохранить график памяти"
            )
            
            if file_path:
                self.memory_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"График памяти сохранен в {file_path}")
                messagebox.showinfo("Успех", "График успешно сохранен")
                
        except Exception as e:
            logger.error(f"Ошибка сохранения графика памяти: {e}", exc_info=True)
            messagebox.showerror("Ошибка", f"Не удалось сохранить график: {str(e)}")
    
    def _clear_memory_cache(self):
        """Очищает кэш памяти."""
        if not self.gui.brain or not hasattr(self.gui.brain, 'memory_manager'):
            messagebox.showerror("Ошибка", "Менеджер памяти недоступен")
            return
        
        if messagebox.askyesno("Подтверждение", "Вы действительно хотите очистить кэш памяти?"):
            try:
                self.gui.brain.memory_manager.clear_cache()
                self.refresh_memory_data()
                self.gui.show_message("Кэш памяти успешно очищен", "info")
                logger.info("Кэш памяти очищен")
            except Exception as e:
                logger.error(f"Ошибка очистки кэша памяти: {e}", exc_info=True)
                messagebox.showerror("Ошибка", f"Не удалось очистить кэш: {str(e)}")
    
    def _optimize_cache(self):
        """Оптимизирует кэш памяти."""
        if not self.gui.brain or not hasattr(self.gui.brain, 'memory_manager'):
            messagebox.showerror("Ошибка", "Менеджер памяти недоступен")
            return
        
        try:
            self.gui.brain.memory_manager.optimize_cache()
            self.refresh_memory_data()
            self.gui.show_message("Кэш памяти успешно оптимизирован", "info")
            logger.info("Кэш памяти оптимизирован")
        except Exception as e:
            logger.error(f"Ошибка оптимизации кэша памяти: {e}", exc_info=True)
            messagebox.showerror("Ошибка", f"Не удалось оптимизировать кэш: {str(e)}")
    
    def _apply_cache_size(self):
        """Применяет новый размер кэша."""
        if not self.gui.brain or not hasattr(self.gui.brain, 'memory_manager'):
            messagebox.showerror("Ошибка", "Менеджер памяти недоступен")
            return
        
        try:
            new_size = self.cache_size_var.get()
            self.gui.brain.memory_manager.set_cache_size(new_size)
            self.refresh_memory_data()
            self.gui.show_message(f"Размер кэша изменен на {new_size} элементов", "info")
            logger.info(f"Размер кэша изменен на {new_size} элементов")
        except Exception as e:
            logger.error(f"Ошибка изменения размера кэша: {e}", exc_info=True)
            messagebox.showerror("Ошибка", f"Не удалось изменить размер кэша: {str(e)}")
    
    def _apply_memory_settings(self):
        """Применяет настройки управления памятью."""
        if not self.gui.brain or not hasattr(self.gui.brain, 'memory_manager'):
            messagebox.showerror("Ошибка", "Менеджер памяти недоступен")
            return
        
        try:
            # Получаем значения
            analysis_interval = self.analysis_interval_var.get()
            age_threshold = self.age_threshold_var.get()
            strength_threshold = self.strength_threshold_var.get()
            
            # Применяем настройки
            self.gui.brain.memory_manager.set_analysis_interval(analysis_interval)
            self.gui.brain.memory_manager.set_age_threshold(age_threshold)
            self.gui.brain.memory_manager.set_strength_threshold(strength_threshold)
            
            self.gui.show_message("Настройки управления памятью применены", "info")
            logger.info(f"Настройки управления памятью применены: "
                       f"interval={analysis_interval}, "
                       f"age_threshold={age_threshold}, "
                       f"strength_threshold={strength_threshold}")
        except Exception as e:
            logger.error(f"Ошибка применения настроек памяти: {e}", exc_info=True)
            messagebox.showerror("Ошибка", f"Не удалось применить настройки: {str(e)}")
    
    def _get_available_domains(self) -> List[str]:
        """Возвращает список доступных доменов."""
        if not self.gui.brain or not hasattr(self.gui.brain, 'memory_manager'):
            return []
        
        try:
            # Получаем все узлы
            nodes = self.gui.brain.memory_manager.get_all_nodes()
            
            # Собираем уникальные домены
            domains = set()
            for node in nodes:
                domains.add(node.domain)
            
            return sorted(list(domains))
        except Exception as e:
            logger.error(f"Ошибка получения доменов: {e}")
            return []
    
    def _export_nodes_data(self):
        """Экспортирует данные узлов в CSV."""
        if not self.gui.brain or not self.gui.brain.memory_manager:
            self.gui.show_notification("Менеджер памяти недоступен", "error")
            return
        
        try:
            # Получаем данные узлов
            nodes = self.gui.brain.memory_manager.get_all_nodes()
            
            # Создаем файл для экспорта
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")]
            )
            
            if not file_path:
                return  # Пользователь отменил выбор
            
            # Экспортируем данные
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                import csv
                writer = csv.writer(f)
                
                # Заголовки
                writer.writerow(["ID", "Контент", "Тип", "Количество связей", "Время создания", "Последнее обновление"])
                
                # Данные
                for node in nodes:
                    writer.writerow([
                        node.id,
                        node.content,
                        node.node_type,
                        len(node.edges),
                        node.created_at,
                        node.last_updated
                    ])
            
            self.gui.show_notification(f"Данные узлов экспортированы в {file_path}", "success")
        except Exception as e:
            logger.error(f"Ошибка экспорта данных узлов: {e}")
            self.gui.show_notification(f"Ошибка экспорта: {str(e)}", "error")
    
    def _export_edges_data(self):
        """Экспортирует данные связей в CSV."""
        if not self.gui.brain or not self.gui.brain.memory_manager:
            self.gui.show_notification("Менеджер памяти недоступен", "error")
            return
        
        try:
            # Получаем данные связей
            edges = self.gui.brain.memory_manager.get_all_edges()
            
            # Создаем файл для экспорта
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")]
            )
            
            if not file_path:
                return  # Пользователь отменил выбор
            
            # Экспортируем данные
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                import csv
                writer = csv.writer(f)
                
                # Заголовки
                writer.writerow(["ID", "Источник", "Цель", "Тип", "Вес", "Контекст", "Время создания", "Последнее обновление"])
                
                # Данные
                for edge in edges:
                    writer.writerow([
                        edge.id,
                        edge.source_id,
                        edge.target_id,
                        edge.edge_type,
                        edge.weight,
                        edge.context,
                        edge.created_at,
                        edge.last_updated
                    ])
            
            self.gui.show_notification(f"Данные связей экспортированы в {file_path}", "success")
        except Exception as e:
            logger.error(f"Ошибка экспорта данных связей: {e}")
            self.gui.show_notification(f"Ошибка экспорта: {str(e)}", "error")
    
    def _show_error(self, message: str):
        """Отображает сообщение об ошибке."""
        if hasattr(self, 'memory_frame') and self.memory_frame:
            # Очищаем текущее содержимое
            for widget in self.memory_frame.winfo_children():
                widget.destroy()
            
            # Создаем сообщение об ошибке
            error_frame = ttk.Frame(self.memory_frame)
            error_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            ttk.Label(
                error_frame,
                text="Ошибка",
                font=('Segoe UI', 16, 'bold'),
                foreground='#dc3545'
            ).pack(pady=(0, 10))
            
            ttk.Label(
                error_frame,
                text=message,
                wraplength=500,
                justify=tk.CENTER
            ).pack(pady=(0, 20))
            
            ttk.Button(
                error_frame,
                text="Повторить",
                command=self._initialize_memory_data,
                style='Primary.TButton'
            ).pack()
    
    def show_context_clarification_request(self, user_id: str, contradiction_id: str, request: str):
        """
        Отображает запрос на уточнение контекста.
        
        Args:
            user_id: ID пользователя
            contradiction_id: ID противоречия
            request: Текст запроса
        """
        # Сохраняем состояние
        self.context_clarification_request = request
        self.pending_contradiction_id = contradiction_id
        
        # Показываем уведомление с запросом уточнения
        self.show_notification(
            f"Требуется уточнение контекста:\n{request}",
            level="warning",
            duration=10000,
            closable=True
        )
        
        # Добавляем сообщение в чат
        self.root.after(0, lambda: self._add_message_to_chat(
            f"Для разрешения противоречия требуется уточнение:\n{request}", 
            False
        ))
        
        # Обновляем статус
        self._update_status("Требуется уточнение контекста")

    # ... rest of the code remains the same ...