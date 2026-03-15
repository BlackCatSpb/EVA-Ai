"""Модуль аналитики для CogniFlex GUI - полнофункциональная реализация"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
import base64
from io import BytesIO

logger = logging.getLogger("cogniflex.gui.analytics")

class AnalyticsModule:
    """Модуль аналитики для мониторинга состояния системы."""
    
    def __init__(self, gui):
        self.gui = gui
        self.analytics_frame = None
        self.notebook = None
        self.canvas_objects = {}
        self.update_interval = 5000  # Интервал обновления в миллисекундах (5 секунд)
        self.running = False
        self.update_thread = None
        self.stop_event = threading.Event()
        self.data_queue = queue.Queue()
        self.system_data = {
            "cpu": {"timestamps": [], "values": []},
            "memory": {"timestamps": [], "values": []},
            "tasks": {"timestamps": [], "values": []},
            "contradictions": {"timestamps": [], "values": []},
            "throughput": {"timestamps": [], "values": []},
            "response_time": {"timestamps": [], "values": []},
            # Добавлены недостающие серии для вкладки обучения
            "learning_opportunities": {"timestamps": [], "values": []},
            "learning_progress": {"timestamps": [], "values": []}
        }
        self.current_view = "system"
        self.max_data_points = 100  # Максимальное количество точек для отображения
        self.dashboard_data = None
        logger.info("Модуль аналитики инициализирован")
    
    def activate(self):
        """Активирует модуль аналитики."""
        # Очищаем область контента
        for widget in self.gui.content_area.winfo_children():
            widget.destroy()
            
        # Создаем интерфейс аналитики
        self._create_analytics_interface()
        
        # Начинаем сбор данных
        self._start_data_collection()
        
        logger.info("Модуль аналитики активирован")
    
    def deactivate(self):
        """Деактивирует модуль аналитики."""
        # Останавливаем сбор данных
        self._stop_data_collection()
        
        # Очищаем очередь данных
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
                
        # Удаляем ссылки на GUI элементы
        self.analytics_frame = None
        self.notebook = None
        self.canvas_objects = {}
        
        logger.info("Модуль аналитики деактивирован")
    
    def _start_data_collection(self):
        """Начинает сбор данных в фоновом потоке."""
        if self.running:
            return
            
        self.stop_event.clear()
        self.running = True
        self.update_thread = threading.Thread(
            target=self._data_collection_loop,
            name="AnalyticsDataCollector",
            daemon=True
        )
        self.update_thread.start()
        logger.debug("Поток сбора данных аналитики запущен")
    
    def _stop_data_collection(self):
        """Останавливает сбор данных."""
        if not self.running:
            return
            
        self.stop_event.set()
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
        self.running = False
        logger.debug("Поток сбора данных аналитики остановлен")
    
    def _data_collection_loop(self):
        """Цикл сбора данных аналитики."""
        while not self.stop_event.is_set():
            try:
                # Собираем данные
                self._collect_system_data()
                
                # Отправляем сигнал обновления
                self.gui.gui_queue.put(self._update_charts)
                
                # Ждем перед следующим сбором
                self.stop_event.wait(timeout=self.update_interval / 1000.0)
                
            except Exception as e:
                logger.error(f"Ошибка в цикле сбора данных аналитики: {e}", exc_info=True)
                self.stop_event.wait(timeout=5.0)
    
    def _collect_system_data(self):
        """Собирает системные данные для аналитики."""
        try:
            # Проверяем доступность ядра
            if not self.gui.brain or not hasattr(self.gui.brain, 'get_system_metrics'):
                logger.warning("Ядро системы недоступно для сбора аналитики")
                return
                
            # Получаем системные метрики
            metrics = self.gui.brain.get_system_metrics()
            
            # Получаем статистику по противоречиям
            contradiction_stats = {}
            if hasattr(self.gui.brain, 'get_contradiction_statistics'):
                try:
                    contradiction_stats = self.gui.brain.get_contradiction_statistics()
                except Exception as e:
                    logger.warning(f"Ошибка получения статистики противоречий: {e}")
            
            # Получаем данные для аналитики
            current_time = time.time()
            
            # Собираем данные для CPU
            cpu_usage = metrics.get("cpu_usage", 0.0) * 100
            self._add_data_point("cpu", current_time, cpu_usage)
            
            # Собираем данные для памяти
            memory_usage = metrics.get("memory_usage", 0.0) * 100
            self._add_data_point("memory", current_time, memory_usage)
            
            # Собираем данные для активных задач
            active_tasks = metrics.get("active_tasks", 0)
            self._add_data_point("tasks", current_time, active_tasks)
            
            # Собираем данные для противоречий
            contradictions = contradiction_stats.get("total", 0) if isinstance(contradiction_stats, dict) else 0
            self._add_data_point("contradictions", current_time, contradictions)
            
            # Собираем данные для пропускной способности
            throughput = metrics.get("request_throughput", 0.0)
            self._add_data_point("throughput", current_time, throughput)
            
            # Собираем данные для времени ответа
            response_time = metrics.get("response_time", 0.0)
            self._add_data_point("response_time", current_time, response_time)
            
            logger.debug(f"Собраны данные аналитики: CPU={cpu_usage:.1f}%, Memory={memory_usage:.1f}%, Tasks={active_tasks}, Contradictions={contradictions}")
            
        except Exception as e:
            logger.error(f"Ошибка сбора системных данных: {e}", exc_info=True)
    
    def _add_data_point(self, data_type: str, timestamp: float, value: float):
        """Добавляет точку данных в соответствующий набор."""
        if data_type not in self.system_data:
            logger.warning(f"Неизвестный тип данных: {data_type}")
            return
            
        # Добавляем точку
        self.system_data[data_type]["timestamps"].append(timestamp)
        self.system_data[data_type]["values"].append(value)
        
        # Ограничиваем размер истории
        if len(self.system_data[data_type]["timestamps"]) > self.max_data_points:
            self.system_data[data_type]["timestamps"] = self.system_data[data_type]["timestamps"][-self.max_data_points:]
            self.system_data[data_type]["values"] = self.system_data[data_type]["values"][-self.max_data_points:]
    
    def _create_analytics_interface(self):
        """Создает интерфейс аналитики."""
        # Создаем основной фрейм
        self.analytics_frame = ttk.Frame(self.gui.content_area)
        self.analytics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Заголовок
        header_frame = ttk.Frame(self.analytics_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, text="Системная аналитика", 
                 font=("Segoe UI", 16, "bold"),
                 foreground=self.gui.colors['primary']).pack(side=tk.LEFT)
        
        # Кнопки управления
        btn_frame = ttk.Frame(header_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="Обновить", 
                 command=self._manual_refresh).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Экспорт", 
                 command=self._export_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Настройки", 
                 command=self._show_settings).pack(side=tk.LEFT, padx=5)
        
        # Создаем ноутбук для разных представлений
        self.notebook = ttk.Notebook(self.analytics_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка системных метрик
        system_frame = ttk.Frame(self.notebook)
        self.notebook.add(system_frame, text="Система")
        self._create_system_metrics_tab(system_frame)
        
        # Вкладка аналитики знаний
        knowledge_frame = ttk.Frame(self.notebook)
        self.notebook.add(knowledge_frame, text="Знания")
        self._create_knowledge_analytics_tab(knowledge_frame)
        
        # Вкладка производительности
        performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(performance_frame, text="Производительность")
        self._create_performance_tab(performance_frame)
        
        # Вкладка противоречий
        contradictions_frame = ttk.Frame(self.notebook)
        self.notebook.add(contradictions_frame, text="Противоречия")
        self._create_contradictions_tab(contradictions_frame)
        
        # Вкладка обучения
        learning_frame = ttk.Frame(self.notebook)
        self.notebook.add(learning_frame, text="Обучение")
        self._create_learning_tab(learning_frame)
    
    def _create_system_metrics_tab(self, parent_frame):
        """Создает вкладку системных метрик."""
        # Создаем контейнер для графиков
        charts_frame = ttk.Frame(parent_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем график использования CPU
        cpu_frame = ttk.LabelFrame(charts_frame, text="Использование CPU")
        cpu_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем фигуру matplotlib
        fig = plt.Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Использование CPU')
        ax.set_xlabel('Время')
        ax.set_ylabel('Проценты')
        ax.set_ylim(0, 100)
        ax.grid(True)
        
        # Создаем линию
        line, = ax.plot([], [], 'b-')
        
        # Встраиваем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, cpu_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем объекты для обновления
        self.canvas_objects["cpu"] = {
            "figure": fig,
            "axis": ax,
            "line": line,
            "canvas": canvas
        }
        
        # Создаем график использования памяти
        memory_frame = ttk.LabelFrame(charts_frame, text="Использование памяти")
        memory_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем фигуру matplotlib
        fig = plt.Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Использование памяти')
        ax.set_xlabel('Время')
        ax.set_ylabel('Проценты')
        ax.set_ylim(0, 100)
        ax.grid(True)
        
        # Создаем линию
        line, = ax.plot([], [], 'g-')
        
        # Встраиваем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, memory_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем объекты для обновления
        self.canvas_objects["memory"] = {
            "figure": fig,
            "axis": ax,
            "line": line,
            "canvas": canvas
        }
    
    def _create_knowledge_analytics_tab(self, parent_frame):
        """Создает вкладку аналитики знаний."""
        # Создаем контейнер для графиков
        charts_frame = ttk.Frame(parent_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем график количества знаний по доменам
        domains_frame = ttk.LabelFrame(charts_frame, text="Знания по доменам")
        domains_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем фигуру matplotlib
        fig = plt.Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Распределение знаний по доменам')
        ax.set_xlabel('Домен')
        ax.set_ylabel('Количество')
        
        # Встраиваем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, domains_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем объекты для обновления
        self.canvas_objects["knowledge_domains"] = {
            "figure": fig,
            "axis": ax,
            "canvas": canvas
        }
        
        # Создаем график активности знаний
        activity_frame = ttk.LabelFrame(charts_frame, text="Активность знаний")
        activity_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем фигуру matplotlib
        fig = plt.Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Активность знаний')
        ax.set_xlabel('Время')
        ax.set_ylabel('Количество изменений')
        
        # Встраиваем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, activity_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем объекты для обновления
        self.canvas_objects["knowledge_activity"] = {
            "figure": fig,
            "axis": ax,
            "canvas": canvas
        }
    
    def _create_performance_tab(self, parent_frame):
        """Создает вкладку производительности."""
        # Создаем контейнер для графиков
        charts_frame = ttk.Frame(parent_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем график пропускной способности
        throughput_frame = ttk.LabelFrame(charts_frame, text="Пропускная способность")
        throughput_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем фигуру matplotlib
        fig = plt.Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Пропускная способность')
        ax.set_xlabel('Время')
        ax.set_ylabel('Запросы/сек')
        ax.grid(True)
        
        # Создаем линию
        line, = ax.plot([], [], 'r-')
        
        # Встраиваем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, throughput_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем объекты для обновления
        self.canvas_objects["throughput"] = {
            "figure": fig,
            "axis": ax,
            "line": line,
            "canvas": canvas
        }
        
        # Создаем график времени ответа
        response_frame = ttk.LabelFrame(charts_frame, text="Время ответа")
        response_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем фигуру matplotlib
        fig = plt.Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Время ответа')
        ax.set_xlabel('Время')
        ax.set_ylabel('Секунды')
        ax.grid(True)
        
        # Создаем линию
        line, = ax.plot([], [], 'm-')
        
        # Встраиваем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, response_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем объекты для обновления
        self.canvas_objects["response_time"] = {
            "figure": fig,
            "axis": ax,
            "line": line,
            "canvas": canvas
        }
    
    def _create_contradictions_tab(self, parent_frame):
        """Создает вкладку противоречий."""
        # Создаем контейнер для графиков
        charts_frame = ttk.Frame(parent_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем график количества противоречий
        count_frame = ttk.LabelFrame(charts_frame, text="Количество противоречий")
        count_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем фигуру matplotlib
        fig = plt.Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Количество противоречий')
        ax.set_xlabel('Время')
        ax.set_ylabel('Количество')
        ax.grid(True)
        
        # Создаем линию
        line, = ax.plot([], [], 'b-')
        
        # Встраиваем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, count_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем объекты для обновления
        self.canvas_objects["contradictions"] = {
            "figure": fig,
            "axis": ax,
            "line": line,
            "canvas": canvas
        }
        
        # Создаем график серьезности противоречий
        severity_frame = ttk.LabelFrame(charts_frame, text="Серьезность противоречий")
        severity_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем фигуру matplotlib
        fig = plt.Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Распределение серьезности противоречий')
        ax.set_xlabel('Серьезность')
        ax.set_ylabel('Количество')
        
        # Встраиваем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, severity_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем объекты для обновления
        self.canvas_objects["contradictions_severity"] = {
            "figure": fig,
            "axis": ax,
            "canvas": canvas
        }
    
    def _create_learning_tab(self, parent_frame):
        """Создает вкладку обучения."""
        # Создаем контейнер для графиков
        charts_frame = ttk.Frame(parent_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем график возможностей обучения
        opportunities_frame = ttk.LabelFrame(charts_frame, text="Возможности обучения")
        opportunities_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем фигуру matplotlib
        fig = plt.Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Возможности обучения')
        ax.set_xlabel('Время')
        ax.set_ylabel('Количество')
        ax.grid(True)
        
        # Создаем линию
        line, = ax.plot([], [], 'g-')
        
        # Встраиваем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, opportunities_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем объекты для обновления
        self.canvas_objects["learning_opportunities"] = {
            "figure": fig,
            "axis": ax,
            "line": line,
            "canvas": canvas
        }
        
        # Создаем график прогресса обучения
        progress_frame = ttk.LabelFrame(charts_frame, text="Прогресс обучения")
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем фигуру matplotlib
        fig = plt.Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Прогресс обучения')
        ax.set_xlabel('Время')
        ax.set_ylabel('Проценты')
        ax.set_ylim(0, 100)
        ax.grid(True)
        
        # Создаем линию
        line, = ax.plot([], [], 'm-')
        
        # Встраиваем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, progress_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем объекты для обновления
        self.canvas_objects["learning_progress"] = {
            "figure": fig,
            "axis": ax,
            "line": line,
            "canvas": canvas
        }
    
    def _update_charts(self):
        """Обновляет все графики на основе собранных данных."""
        if not self.analytics_frame or not self.analytics_frame.winfo_exists():
            return
            
        try:
            # Обновляем график CPU
            if "cpu" in self.canvas_objects:
                self._update_line_chart(
                    "cpu",
                    self.system_data["cpu"]["timestamps"],
                    self.system_data["cpu"]["values"],
                    "Использование CPU",
                    "Проценты",
                    0,
                    100
                )
            
            # Обновляем график памяти
            if "memory" in self.canvas_objects:
                self._update_line_chart(
                    "memory",
                    self.system_data["memory"]["timestamps"],
                    self.system_data["memory"]["values"],
                    "Использование памяти",
                    "Проценты",
                    0,
                    100
                )
            
            # Обновляем график пропускной способности
            if "throughput" in self.canvas_objects:
                self._update_line_chart(
                    "throughput",
                    self.system_data["throughput"]["timestamps"],
                    self.system_data["throughput"]["values"],
                    "Пропускная способность",
                    "Запросы/сек"
                )
            
            # Обновляем график времени ответа
            if "response_time" in self.canvas_objects:
                self._update_line_chart(
                    "response_time",
                    self.system_data["response_time"]["timestamps"],
                    self.system_data["response_time"]["values"],
                    "Время ответа",
                    "Секунды"
                )
            
            # Обновляем график противоречий
            if "contradictions" in self.canvas_objects:
                self._update_line_chart(
                    "contradictions",
                    self.system_data["contradictions"]["timestamps"],
                    self.system_data["contradictions"]["values"],
                    "Количество противоречий",
                    "Количество"
                )
            
            # Обновляем график возможностей обучения
            if "learning_opportunities" in self.canvas_objects:
                opportunities = 0
                if self.gui.brain and hasattr(self.gui.brain, 'get_system_dashboard_data'):
                    dashboard_data = self.gui.brain.get_system_dashboard_data()
                    opportunities = len(dashboard_data.get("learning_opportunities", []))
                
                # Создаем временные данные для демонстрации
                current_time = time.time()
                self._add_data_point("learning_opportunities", current_time, opportunities)
                
                self._update_line_chart(
                    "learning_opportunities",
                    self.system_data["learning_opportunities"]["timestamps"],
                    self.system_data["learning_opportunities"]["values"],
                    "Возможности обучения",
                    "Количество"
                )
            
            # Обновляем график прогресса обучения
            if "learning_progress" in self.canvas_objects and self.gui.brain:
                progress = 0.0
                if hasattr(self.gui.brain, 'get_system_metrics'):
                    metrics = self.gui.brain.get_system_metrics()
                    progress = metrics.get("learning_progress", 0.0) * 100
                
                # Создаем временные данные для демонстрации
                current_time = time.time()
                self._add_data_point("learning_progress", current_time, progress)
                
                self._update_line_chart(
                    "learning_progress",
                    self.system_data["learning_progress"]["timestamps"],
                    self.system_data["learning_progress"]["values"],
                    "Прогресс обучения",
                    "Проценты",
                    0,
                    100
                )
            
            # Обновляем график серьезности противоречий
            if "contradictions_severity" in self.canvas_objects and self.gui.brain:
                self._update_contradictions_severity_chart()
            
            # Обновляем график знаний по доменам
            if "knowledge_domains" in self.canvas_objects and self.gui.brain:
                self._update_knowledge_domains_chart()
            
        except Exception as e:
            logger.error(f"Ошибка обновления графиков аналитики: {e}", exc_info=True)
    
    def _update_line_chart(self, chart_key: str, timestamps: List[float], values: List[float], 
                          title: str, ylabel: str, ymin: Optional[float] = None, 
                          ymax: Optional[float] = None):
        """Обновляет линейный график."""
        if not self.canvas_objects.get(chart_key):
            return
            
        try:
            obj = self.canvas_objects[chart_key]
            
            # Преобразуем временные метки в относительное время
            if timestamps:
                start_time = timestamps[0]
                x = [t - start_time for t in timestamps]
            else:
                x = []
            
            # Обновляем данные линии
            obj["line"].set_data(x, values)
            
            # Обновляем пределы осей
            obj["axis"].relim()
            obj["axis"].autoscale_view()
            
            # Устанавливаем минимальные пределы, если указано
            if ymin is not None and ymax is not None:
                obj["axis"].set_ylim(ymin, ymax)
            
            # Обновляем заголовок и метки
            obj["axis"].set_title(title)
            obj["axis"].set_ylabel(ylabel)
            
            # Обновляем график
            obj["canvas"].draw()
            
        except Exception as e:
            logger.error(f"Ошибка обновления линейного графика {chart_key}: {e}", exc_info=True)
    
    def _update_contradictions_severity_chart(self):
        """Обновляет график серьезности противоречий."""
        if not self.canvas_objects.get("contradictions_severity"):
            return
            
        try:
            obj = self.canvas_objects["contradictions_severity"]
            
            # Получаем статистику по противоречиям
            severity_data = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            if self.gui.brain and hasattr(self.gui.brain, 'get_contradiction_statistics'):
                stats = self.gui.brain.get_contradiction_statistics()
                if isinstance(stats, dict) and "by_severity" in stats:
                    severity_data = stats["by_severity"]
            
            # Очищаем график
            obj["axis"].clear()
            
            # Создаем данные для бар-чарта
            severities = list(severity_data.keys())
            counts = [severity_data[s] for s in severities]
            
            # Строим бар-чарт
            colors = {
                "critical": "red",
                "high": "orange",
                "medium": "yellow",
                "low": "green"
            }
            bar_colors = [colors.get(s, "gray") for s in severities]
            
            obj["axis"].bar(severities, counts, color=bar_colors)
            obj["axis"].set_title('Распределение серьезности противоречий')
            obj["axis"].set_xlabel('Серьезность')
            obj["axis"].set_ylabel('Количество')
            obj["axis"].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Обновляем график
            obj["canvas"].draw()
            
        except Exception as e:
            logger.error(f"Ошибка обновления графика серьезности противоречий: {e}", exc_info=True)
    
    def _update_knowledge_domains_chart(self):
        """Обновляет график знаний по доменам."""
        if not self.canvas_objects.get("knowledge_domains"):
            return
            
        try:
            obj = self.canvas_objects["knowledge_domains"]
            
            # Получаем статистику по доменам
            domain_data = {}
            if self.gui.brain and hasattr(self.gui.brain, 'get_system_dashboard_data'):
                dashboard_data = self.gui.brain.get_system_dashboard_data()
                domain_stats = dashboard_data.get("metrics", {}).get("by_domain", {})
                if isinstance(domain_stats, dict):
                    domain_data = domain_stats
            
            # Очищаем график
            obj["axis"].clear()
            
            # Создаем данные для бар-чарта
            domains = list(domain_data.keys())
            counts = [domain_data[d] for d in domains]
            
            # Строим бар-чарт
            obj["axis"].bar(domains, counts, color=self.gui.colors['primary'])
            obj["axis"].set_title('Распределение знаний по доменам')
            obj["axis"].set_xlabel('Домен')
            obj["axis"].set_ylabel('Количество')
            obj["axis"].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Поворачиваем метки доменов для лучшей читаемости
            plt.setp(obj["axis"].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Обновляем график
            obj["canvas"].draw()
            
        except Exception as e:
            logger.error(f"Ошибка обновления графика знаний по доменам: {e}", exc_info=True)
    
    def _manual_refresh(self):
        """Обновляет данные аналитики вручную."""
        self._collect_system_data()
        self._update_charts()
        logger.info("Данные аналитики обновлены вручную")
    
    def _export_data(self):
        """Экспортирует данные аналитики."""
        try:
            # Создаем имя файла
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                initialfile=f"analytics_export_{timestamp}.json",
                defaultextension=".json",
                filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")]
            )
            
            if not filename:
                return
                
            # Подготавливаем данные для экспорта
            export_data = {
                "timestamp": time.time(),
                "system_data": self.system_data,
                "metadata": {
                    "cogniflex_version": "1.0",
                    "export_date": datetime.now().isoformat()
                }
            }
            
            # Добавляем дополнительные данные, если доступны
            if self.gui.brain:
                try:
                    if hasattr(self.gui.brain, 'get_system_metrics'):
                        export_data["system_metrics"] = self.gui.brain.get_system_metrics()
                    if hasattr(self.gui.brain, 'get_contradiction_statistics'):
                        export_data["contradiction_statistics"] = self.gui.brain.get_contradiction_statistics()
                except Exception as e:
                    logger.warning(f"Ошибка добавления дополнительных данных в экспорт: {e}")
            
            # Сохраняем в файл
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("Экспорт данных", f"Данные успешно экспортированы в {os.path.basename(filename)}")
            logger.info(f"Данные аналитики экспортированы в {filename}")
        except Exception as e:
            logger.error(f"Ошибка экспорта данных: {e}", exc_info=True)
            messagebox.showerror("Ошибка", f"Не удалось экспортировать данные: {str(e)}")
    
    def _show_settings(self):
        """Показывает окно настроек аналитики."""
        settings_window = tk.Toplevel(self.gui.root)
        settings_window.title("Настройки аналитики")
        settings_window.geometry("400x300")
        settings_window.transient(self.gui.root)
        settings_window.grab_set()
        
        # Создаем фрейм с отступами
        main_frame = ttk.Frame(settings_window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Интервал обновления
        ttk.Label(main_frame, text="Интервал обновления (мс):").pack(anchor=tk.W, pady=(0, 5))
        interval_var = tk.StringVar(value=str(self.update_interval))
        interval_entry = ttk.Entry(main_frame, textvariable=interval_var)
        interval_entry.pack(fill=tk.X, pady=(0, 15))
        
        # Максимальное количество точек
        ttk.Label(main_frame, text="Максимальное количество точек:").pack(anchor=tk.W, pady=(0, 5))
        points_var = tk.StringVar(value=str(self.max_data_points))
        points_entry = ttk.Entry(main_frame, textvariable=points_var)
        points_entry.pack(fill=tk.X, pady=(0, 20))
        
        # Кнопки
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        def apply_settings():
            try:
                new_interval = int(interval_var.get())
                new_points = int(points_var.get())
                
                if new_interval < 1000:
                    messagebox.showwarning("Недопустимое значение", "Интервал обновления должен быть не менее 1000 мс")
                    return
                    
                if new_points < 10:
                    messagebox.showwarning("Недопустимое значение", "Минимальное количество точек - 10")
                    return
                
                self.update_interval = new_interval
                self.max_data_points = new_points
                
                # Перезапускаем сбор данных с новыми настройками
                self._stop_data_collection()
                self._start_data_collection()
                
                settings_window.destroy()
                messagebox.showinfo("Настройки", "Настройки успешно применены")
            except ValueError:
                messagebox.showerror("Ошибка", "Пожалуйста, введите числовые значения")
        
        ttk.Button(btn_frame, text="Применить", command=apply_settings).pack(side=tk.LEFT, expand=True)
        ttk.Button(btn_frame, text="Отмена", command=settings_window.destroy).pack(side=tk.LEFT, expand=True)
    
    def update_theme(self):
        """Обновляет тему при смене темы интерфейса."""
        if not self.analytics_frame or not self.analytics_frame.winfo_exists():
            return
            
        try:
            # Обновляем цвета графиков
            for chart_key, chart_obj in self.canvas_objects.items():
                if "axis" in chart_obj:
                    # Устанавливаем цвета в зависимости от темы
                    if self.gui.theme == "dark":
                        chart_obj["axis"].set_facecolor('#2d2d2d')
                        chart_obj["axis"].tick_params(colors='white')
                        chart_obj["axis"].xaxis.label.set_color('white')
                        chart_obj["axis"].yaxis.label.set_color('white')
                        chart_obj["axis"].title.set_color('white')
                    else:
                        chart_obj["axis"].set_facecolor('white')
                        chart_obj["axis"].tick_params(colors='black')
                        chart_obj["axis"].xaxis.label.set_color('black')
                        chart_obj["axis"].yaxis.label.set_color('black')
                        chart_obj["axis"].title.set_color('black')
                    
                    # Обновляем график
                    chart_obj["canvas"].draw()
        except Exception as e:
            logger.error(f"Ошибка обновления темы аналитики: {e}", exc_info=True)
    
    def get_system_summary(self) -> str:
        """Возвращает краткую сводку о состоянии системы.
        
        Returns:
            str: Сводка о состоянии
        """
        try:
            if not self.gui.brain:
                return "Система: Не подключена"
                
            # Получаем системные метрики
            metrics = self.gui.brain.get_system_metrics()
            
            # Формируем сводку
            summary = "СИСТЕМА COGNIFLEX\n"
            summary += "=" * 50 + "\n\n"
            
            # Статус системы
            status = "АКТИВНА" if self.gui.brain.running else "НЕ АКТИВНА"
            summary += f"Статус системы: {status}\n"
            
            # Время работы
            uptime = metrics.get("uptime", 0)
            if uptime > 0:
                days = int(uptime // (24 * 3600))
                hours = int((uptime % (24 * 3600)) // 3600)
                minutes = int((uptime % 3600) // 60)
                summary += f"Время работы: {days}д {hours}ч {minutes}м\n"
            
            # Использование ресурсов
            cpu = metrics.get("cpu_usage", 0) * 100
            memory = metrics.get("memory_usage", 0) * 100
            summary += f"Использование CPU: {cpu:.1f}%\n"
            summary += f"Использование памяти: {memory:.1f}%\n"
            
            # Производительность
            throughput = metrics.get("request_throughput", 0)
            response_time = metrics.get("response_time", 0)
            summary += f"Пропускная способность: {throughput:.2f} запросов/сек\n"
            summary += f"Среднее время ответа: {response_time:.3f} сек\n"
            
            # Противоречия
            contradictions = 0
            if hasattr(self.gui.brain, 'get_contradiction_statistics'):
                stats = self.gui.brain.get_contradiction_statistics()
                if isinstance(stats, dict):
                    contradictions = stats.get("total", 0)
            summary += f"Обнаружено противоречий: {contradictions}\n"
            
            # Возможности обучения
            opportunities = 0
            if hasattr(self.gui.brain, 'get_system_dashboard_data'):
                dashboard_data = self.gui.brain.get_system_dashboard_data()
                opportunities = len(dashboard_data.get("learning_opportunities", []))
            summary += f"Возможности для обучения: {opportunities}\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка формирования сводки системы: {e}", exc_info=True)
            return "Ошибка формирования сводки системы"