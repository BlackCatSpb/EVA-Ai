# gui/knowledge_graph_module.py
"""Модуль графа знаний для CogniFlex GUI - полнофункциональная реализация"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog, font
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import time
import json
import os
import threading
import queue
import webbrowser
import re
from tkinter import Menu
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from cogniflex.knowledge.knowledge_graph import KnowledgeGraph



logger = logging.getLogger("cogniflex.gui.knowledge")

class KnowledgeGraphModule:
    """Модуль для визуализации и работы с графом знаний."""
    
    def __init__(self, gui):
        self.gui = gui
        
        # Инициализация атрибутов
        self.knowledge_frame = None
        self.notebook = None
        self.graph_canvas = None
        self.graph = None
        self.pos = None
        self.selected_node = None
        self.selected_edge = None
        self.current_view = "all"
        self.search_query = ""
        
        # Поток обновления
        self.update_thread = None
        self.update_running = False
        
        # Цветовая схема
        self.color_scheme = {
            "background": {"dark": "#1e1e1e", "light": "#ffffff"},
            "node": {"default": "#4a90e2", "selected": "#e74c3c"},
            "edge": {"default": "#95a5a6", "selected": "#f39c12"},
            "text": {"dark": "#ffffff", "light": "#000000"},
            "fact": {"node": "#1f77b4", "text": "white"},
            "concept": {"node": "#ff7f0e", "text": "black"},
            "entity": {"node": "#2ca02c", "text": "white"},
            "relation": {"edge": "#9467bd"},
            "contradiction": {"edge": "#d62728", "width": 2.5}
        }
        
        # Отложенные события
        self.pending_after_events = []
        
        # Потоковые атрибуты
        self.search_results = []
        self.search_index = 0
        self.update_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.update_interval = 5000  # Интервал обновления в миллисекундах (5 секунд)
        self.pending_after_ids = []  # Отслеживание after событий
        self.node_size = 300
        self.edge_width = 1.5
        self.font_size = 9
        
    def _safe_brain_call(self, method_name: str, *args, **kwargs):
        """Безопасный вызов метода brain с логированием"""
        if hasattr(self, 'gui') and hasattr(self.gui, 'brain') and self.gui.brain:
            try:
                method = getattr(self.gui.brain, method_name, None)
                if method:
                    result = method(*args, **kwargs)
                    logger.debug(f"[{self.__class__.__name__}] Успешный вызов brain.{method_name}()")
                    return result
                else:
                    logger.warning(f"[{self.__class__.__name__}] Метод brain.{method_name} не найден")
                    return None
            except Exception as e:
                logger.error(f"[{self.__class__.__name__}] Ошибка вызова brain.{method_name}: {e}")
                return None
        else:
            logger.warning(f"[{self.__class__.__name__}] Brain недоступен")
            return None
    
    def activate(self):
        """Активирует модуль графа знаний."""
        logger.info("Активация модуля графа знаний")
        
        # Проверяем доступность компонентов
        if not self._log_brain_access("knowledge_graph"):
            logger.error("Компонент knowledge_graph недоступен")
            return
        
        # Очищаем область контента
        for widget in self.gui.content_area.winfo_children():
            widget.destroy()
            
        # Создаем интерфейс графа знаний
        self._create_knowledge_interface()
        
        # Начинаем обновление данных
        self._start_update_thread()
        
        logger.info("Модуль графа знаний активирован")
        
        # Начинаем обновление данных
        self._start_update_thread()
        
        logger.info("Модуль графа знаний активирован")

    def deactivate(self):
        """Деактивирует модуль графа знаний."""
        # Останавливаем обновление данных
        self._stop_update_thread()
        
        # Отменяем все pending after события
        self._cancel_pending_after_events()
        
        logger.info("Модуль графа знаний деактивирован")

    def _log_brain_access(self, component_name: str) -> bool:
        """Проверяет и логирует доступ к компоненту brain."""
        if not hasattr(self.gui, 'brain') or not self.gui.brain:
            logger.warning(f"Brain недоступен для {component_name}")
            return False
        
        if not hasattr(self.gui.brain, component_name):
            logger.warning(f"Компонент {component_name} недоступен в brain")
            return False
        
        logger.debug(f"Доступ к {component_name} получен")
        return True

    def _start_update_thread(self):
        """Запускает фоновый поток для обновления данных."""
        if self.update_thread and self.update_thread.is_alive():
            return
            
        self.stop_event.clear()
        self.update_thread = threading.Thread(
            target=self._update_data_loop,
            name="KnowledgeGraphUpdater",
            daemon=True
        )
        self.update_thread.start()
        logger.debug("Фоновый поток обновления графа знаний запущен")

    def _stop_update_thread(self):
        """Останавливает фоновый поток обновления данных."""
        if not self.update_thread:
            return
            
        self.stop_event.set()
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
        logger.debug("Фоновый поток обновления графа знаний остановлен")
    
    def _cancel_pending_after_events(self):
        """Отменяет все ожидающие after события."""
        if not self.gui or not hasattr(self.gui, 'root') or not self.gui.root:
            return
            
        try:
            for after_id in list(self.pending_after_ids):
                try:
                    self.gui.root.after_cancel(after_id)
                except tk.TclError:
                    pass  # Событие уже выполнено или отменено
            self.pending_after_ids.clear()
            logger.debug("Все after события отменены")
        except Exception as e:
            logger.error(f"Ошибка отмены after событий: {e}")
    
    def _safe_after(self, delay_ms: int, callback):
        """Безопасно планирует after событие с отслеживанием."""
        if not self.gui or not hasattr(self.gui, 'root') or not self.gui.root:
            return None
            
        try:
            after_id = self.gui.root.after(delay_ms, callback)
            self.pending_after_ids.append(after_id)
            return after_id
        except (tk.TclError, RuntimeError):
            # Окно уничтожено или не в главном потоке
            return None

    def _update_data_loop(self):
        """Цикл обновления данных графа знаний."""
        while not self.stop_event.is_set():
            try:
                # Проверяем, нужно ли обновить граф
                if self.current_view == "all" or self.current_view == "search":
                    self._update_graph_data()
                
                # Отправляем сигнал обновления
                self._safe_after(100, self._process_update_queue)
                
                # Ждем перед следующим обновлением
                time.sleep(self.update_interval / 1000.0)
                
            except Exception as e:
                logger.error(f"Ошибка в цикле обновления данных графа: {e}", exc_info=True)
                time.sleep(5.0)

    def _update_graph_data(self):
        """Обновляет данные графа знаний из системы."""
        try:
            # Получаем доступ к графу знаний
            if not self._log_brain_access('knowledge_graph'):
                logger.warning("Граф знаний недоступен для загрузки доменов")
                return
            
            knowledge_graph = self.gui.brain.knowledge_graph
            
            # Получаем данные в зависимости от текущего вида
            if self.current_view == "all":
                # Получаем все узлы и связи
                nodes = knowledge_graph.get_all_nodes()
                edges = knowledge_graph.get_all_edges()
            elif self.current_view == "search" and self.search_query:
                # Ищем узлы по запросу
                nodes = knowledge_graph.search_nodes(self.search_query)
                # Получаем связанные узлы и связи
                node_ids = [getattr(node, "id", None) if not isinstance(node, dict) else node.get("id") for node in nodes]
                node_ids = [nid for nid in node_ids if nid]
                edges = []
                for node_id in node_ids:
                    node_edges = knowledge_graph.get_edges(node_id)
                    edges.extend(node_edges)
                
                # Получаем связанные узлы
                related_node_ids = set()
                for edge in edges:
                    # Поддержка обоих вариантов: объект и словарь
                    src = getattr(edge, "source_id", None)
                    if src is None:
                        src = getattr(edge, "source", None)
                    if src is None and isinstance(edge, dict):
                        src = edge.get("source_id") or edge.get("source")
                    tgt = getattr(edge, "target_id", None)
                    if tgt is None:
                        tgt = getattr(edge, "target", None)
                    if tgt is None and isinstance(edge, dict):
                        tgt = edge.get("target_id") or edge.get("target")
                    if src:
                        related_node_ids.add(src)
                    if tgt:
                        related_node_ids.add(tgt)
                
                # Получаем данные для связанных узлов
                related_nodes = []
                for node_id in related_node_ids:
                    if node_id not in node_ids:
                        node = knowledge_graph.get_node(node_id)
                        if node:
                            related_nodes.append(node)
                
                nodes.extend(related_nodes)
            elif self.current_view.startswith("domain:"):
                # Получаем узлы по домену
                domain = self.current_view.split(":", 1)[1]
                nodes = knowledge_graph.get_nodes_by_domain(domain)
                # Получаем связанные узлы и связи
                node_ids = [getattr(node, "id", None) if not isinstance(node, dict) else node.get("id") for node in nodes]
                node_ids = [nid for nid in node_ids if nid]
                edges = []
                for node_id in node_ids:
                    node_edges = knowledge_graph.get_edges(node_id)
                    edges.extend(node_edges)
            else:
                return
            
            # Создаем граф
            G = nx.DiGraph()
            
            # Добавляем узлы
            for node in nodes:
                if isinstance(node, dict):
                    node_id = node.get("id")
                    label = node.get("name") or node.get("label") or str(node_id)
                    node_type = node.get("node_type") or node.get("type") or "other"
                    domain = node.get("domain") or "general"
                    strength = node.get("strength", 0.0)
                    description = node.get("description") or node.get("content") or ""
                else:
                    node_id = getattr(node, "id", None)
                    label = getattr(node, "name", None) or getattr(node, "content", None) or str(node_id)
                    node_type = getattr(node, "node_type", "other")
                    domain = getattr(node, "domain", "general")
                    strength = getattr(node, "strength", 0.0)
                    description = getattr(node, "description", None) or getattr(node, "content", "")
                if not node_id:
                    continue
                G.add_node(
                    node_id,
                    label=label,
                    type=node_type,
                    domain=domain,
                    strength=strength,
                    description=description
                )
            
            # Добавляем связи
            for edge in edges:
                if isinstance(edge, dict):
                    src = edge.get("source_id") or edge.get("source")
                    tgt = edge.get("target_id") or edge.get("target")
                    label = edge.get("relation_type") or edge.get("relation") or ""
                    strength = edge.get("strength", 0.0)
                    eid = edge.get("id")
                else:
                    src = getattr(edge, "source_id", None) or getattr(edge, "source", None)
                    tgt = getattr(edge, "target_id", None) or getattr(edge, "target", None)
                    label = getattr(edge, "relation_type", None) or getattr(edge, "relation", "")
                    strength = getattr(edge, "strength", 0.0)
                    eid = getattr(edge, "id", None)
                if not src or not tgt:
                    continue
                G.add_edge(
                    src,
                    tgt,
                    label=label,
                    strength=strength,
                    id=eid
                )
            
            # Сохраняем в очередь
            self.update_queue.put({
                "graph": G,
                "nodes": nodes,
                "edges": edges
            })
            
        except Exception as e:
            logger.error(f"Ошибка обновления данных графа: {e}", exc_info=True)

    def _process_update_queue(self):
        """Обрабатывает очередь обновления графа."""
        try:
            # Получаем данные из очереди
            while not self.update_queue.empty():
                data = self.update_queue.get_nowait()
                
                # Обновляем граф
                self.graph = data["graph"]
                self._update_graph_visualization()
                
                self.update_queue.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки очереди графа: {e}", exc_info=True)

    def _create_knowledge_interface(self):
        """Создает интерфейс графа знаний с вкладками."""
        self.knowledge_frame = ttk.Frame(self.gui.content_area)
        self.knowledge_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Заголовок и панель инструментов
        header_frame = ttk.Frame(self.knowledge_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            header_frame, 
            text="Граф знаний", 
            font=('Segoe UI', 16, 'bold')
        ).pack(side=tk.LEFT)
        
        # Панель инструментов
        toolbar = ttk.Frame(header_frame)
        toolbar.pack(side=tk.RIGHT)
        
        ttk.Button(
            toolbar, 
            text="Обновить", 
            command=self._refresh_graph
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            toolbar, 
            text="Поиск", 
            command=self._show_search_dialog
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            toolbar, 
            text="Экспорт", 
            command=self._export_graph
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            toolbar, 
            text="Настройки", 
            command=self._show_settings_dialog
        ).pack(side=tk.LEFT, padx=2)
        
        # Создаем вкладки
        self.notebook = ttk.Notebook(self.knowledge_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка визуализации графа
        graph_tab = ttk.Frame(self.notebook)
        self.notebook.add(graph_tab, text="Визуализация")
        self._create_graph_tab(graph_tab)
        
        # Вкладка поиска
        search_tab = ttk.Frame(self.notebook)
        self.notebook.add(search_tab, text="Поиск")
        self._create_search_tab(search_tab)
        
        # Вкладка доменов
        domains_tab = ttk.Frame(self.notebook)
        self.notebook.add(domains_tab, text="Домены")
        self._create_domains_tab(domains_tab)
        
        # Вкладка статистики
        stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(stats_tab, text="Статистика")
        self._create_stats_tab(stats_tab)
        
        # Привязываем событие смены вкладки
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        
        # Инициализируем граф
        self._initialize_graph()

    def _create_graph_tab(self, parent):
        """Создает вкладку визуализации графа."""
        # Создаем фрейм для графа
        graph_frame = ttk.Frame(parent)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем область для графа
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Граф знаний', fontsize=12)
        ax.axis('off')
        
        # Создаем холст для графа
        self.graph_canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Добавляем навигационную панель
        toolbar = NavigationToolbar2Tk(self.graph_canvas, graph_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Привязываем события
        self.graph_canvas.mpl_connect('button_press_event', self._on_graph_click)
        self.graph_canvas.mpl_connect('motion_notify_event', self._on_graph_hover)
        self.graph_canvas.mpl_connect('scroll_event', self._on_graph_scroll)
        
        # Создаем контекстное меню
        self._create_graph_context_menu()
        
        # Создаем информационную панель
        self._create_info_panel(parent)

    def _create_graph_context_menu(self):
        """Создает контекстное меню для графа."""
        self.graph_context_menu = Menu(self.gui.root, tearoff=0)
        self.graph_context_menu.add_command(
            label="Показать детали", 
            command=self._show_node_details
        )
        self.graph_context_menu.add_command(
            label="Показать связанные узлы", 
            command=self._show_related_nodes
        )
        self.graph_context_menu.add_separator()
        self.graph_context_menu.add_command(
            label="Поиск по домену", 
            command=self._search_by_domain
        )
        self.graph_context_menu.add_command(
            label="Экспорт узла", 
            command=self._export_node
        )
        
        # Привязываем контекстное меню
        self.graph_canvas.mpl_connect('button_press_event', self._show_graph_context_menu)

    def _show_graph_context_menu(self, event):
        """Показывает контекстное меню для графа."""
        if event.button == 3:  # Правая кнопка мыши
            self.graph_context_menu.tk_popup(event.x * 64, self.gui.root.winfo_height() - event.y * 64)

    def _create_info_panel(self, parent):
        """Создает информационную панель с текущей информацией."""
        info_frame = ttk.LabelFrame(parent, text="Информация о выбранном элементе")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Создаем фрейм для информации
        self.info_content = ttk.Frame(info_frame)
        self.info_content.pack(fill=tk.X, padx=10, pady=10)
        
        # Начальный текст
        ttk.Label(
            self.info_content, 
            text="Выберите узел или связь на графике для просмотра подробной информации",
            font=('Segoe UI', 9, 'italic'),
            wraplength=800
        ).pack(fill=tk.X, padx=5, pady=5)

    def _create_search_tab(self, parent):
        """Создает вкладку поиска."""
        # Создаем фрейм для поиска
        search_frame = ttk.Frame(parent)
        search_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Поле поиска
        search_container = ttk.Frame(search_frame)
        search_container.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(search_container, text="Поиск:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.search_entry = ttk.Entry(search_container, width=50)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.search_entry.bind("<Return>", self._perform_search)
        
        ttk.Button(
            search_container, 
            text="Найти", 
            command=self._perform_search
        ).pack(side=tk.LEFT)
        
        # Результаты поиска
        results_frame = ttk.LabelFrame(search_frame, text="Результаты поиска")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем Treeview для результатов
        columns = ("name", "type", "domain", "strength", "description")
        self.search_tree = ttk.Treeview(
            results_frame,
            columns=columns,
            show="headings",
            selectmode="browse"
        )
        
        # Настройка колонок
        self.search_tree.heading("name", text="Название")
        self.search_tree.heading("type", text="Тип")
        self.search_tree.heading("domain", text="Домен")
        self.search_tree.heading("strength", text="Сила")
        self.search_tree.heading("description", text="Описание")
        
        self.search_tree.column("name", width=150, anchor=tk.W)
        self.search_tree.column("type", width=100, anchor=tk.W)
        self.search_tree.column("domain", width=120, anchor=tk.W)
        self.search_tree.column("strength", width=80, anchor=tk.CENTER)
        self.search_tree.column("description", width=300, anchor=tk.W)
        
        # Добавляем прокрутку
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.search_tree.yview)
        self.search_tree.configure(yscrollcommand=scrollbar.set)
        
        # Размещаем Treeview и скроллбар
        self.search_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Добавляем контекстное меню
        self._create_search_context_menu()
        
        # Привязываем события
        self.search_tree.bind("<Double-1>", self._on_search_result_double_click)

    def _create_search_context_menu(self):
        """Создает контекстное меню для результатов поиска."""
        self.search_context_menu = Menu(self.search_tree, tearoff=0)
        self.search_context_menu.add_command(
            label="Показать на графике", 
            command=self._highlight_search_result
        )
        self.search_context_menu.add_command(
            label="Показать детали", 
            command=self._show_search_result_details
        )
        self.search_context_menu.add_separator()
        self.search_context_menu.add_command(
            label="Экспорт", 
            command=self._export_search_result
        )
        
        # Привязываем контекстное меню
        self.search_tree.bind("<Button-3>", self._show_search_context_menu)

    def _show_search_context_menu(self, event):
        """Показывает контекстное меню для результатов поиска."""
        item = self.search_tree.identify_row(event.y)
        if item:
            self.search_tree.selection_set(item)
            self.search_context_menu.tk_popup(event.x_root, event.y_root)

    def _create_domains_tab(self, parent):
        """Создает вкладку доменов."""
        # Создаем фрейм для доменов
        domains_frame = ttk.Frame(parent)
        domains_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем Treeview для доменов
        columns = ("domain", "nodes", "edges", "last_updated")
        self.domains_tree = ttk.Treeview(
            domains_frame,
            columns=columns,
            show="headings",
            selectmode="browse"
        )
        
        # Настройка колонок
        self.domains_tree.heading("domain", text="Домен")
        self.domains_tree.heading("nodes", text="Узлы")
        self.domains_tree.heading("edges", text="Связи")
        self.domains_tree.heading("last_updated", text="Последнее обновление")
        
        self.domains_tree.column("domain", width=200, anchor=tk.W)
        self.domains_tree.column("nodes", width=100, anchor=tk.CENTER)
        self.domains_tree.column("edges", width=100, anchor=tk.CENTER)
        self.domains_tree.column("last_updated", width=150, anchor=tk.W)
        
        # Добавляем прокрутку
        scrollbar = ttk.Scrollbar(domains_frame, orient=tk.VERTICAL, command=self.domains_tree.yview)
        self.domains_tree.configure(yscrollcommand=scrollbar.set)
        
        # Размещаем Treeview и скроллбар
        self.domains_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Добавляем контекстное меню
        self._create_domains_context_menu()
        
        # Заполняем данными
        self._load_domains_data()
        
        # Привязываем события
        self.domains_tree.bind("<Double-1>", self._on_domain_double_click)

    def _create_domains_context_menu(self):
        """Создает контекстное меню для доменов."""
        self.domains_context_menu = Menu(self.domains_tree, tearoff=0)
        self.domains_context_menu.add_command(
            label="Показать на графике", 
            command=self._show_domain_on_graph
        )
        self.domains_context_menu.add_command(
            label="Показать детали", 
            command=self._show_domain_details
        )
        self.domains_context_menu.add_separator()
        self.domains_context_menu.add_command(
            label="Экспорт домена", 
            command=self._export_domain
        )
        
        # Привязываем контекстное меню
        self.domains_tree.bind("<Button-3>", self._show_domains_context_menu)

    def _show_domains_context_menu(self, event):
        """Показывает контекстное меню для доменов."""
        item = self.domains_tree.identify_row(event.y)
        if item:
            self.domains_tree.selection_set(item)
            self.domains_context_menu.tk_popup(event.x_root, event.y_root)

    def _load_domains_data(self):
        """Загружает данные о доменах."""
        try:
            # Очищаем таблицу
            for item in self.domains_tree.get_children():
                self.domains_tree.delete(item)
            
            # Получаем данные из системы
            if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
                logger.warning("Граф знаний недоступен для загрузки доменов")
                return
            
            kg = self.gui.brain.knowledge_graph
            domain_stats = {}
            try:
                # Пытаемся получить из ядра
                domain_stats = kg.get_domain_statistics() or {}
            except Exception as e:
                logger.warning(f"Не удалось получить статистику доменов из ядра: {e}")
                domain_stats = {}

            # Если ядро вернуло в неожиданном формате или пусто — считаем самостоятельно
            if not isinstance(domain_stats, dict) or not domain_stats:
                domain_stats = {}
                nodes = kg.get_all_nodes()
                edges = kg.get_all_edges() if hasattr(kg, 'get_all_edges') else []
                # Подсчет узлов и типов
                for node in nodes:
                    if isinstance(node, dict):
                        domain = node.get("domain", "general")
                        ntype = node.get("node_type") or node.get("type") or "other"
                    else:
                        domain = getattr(node, "domain", "general")
                        ntype = getattr(node, "node_type", "other")
                    d = domain_stats.setdefault(domain, {"nodes": 0, "edges": 0, "node_types": {}})
                    d["nodes"] += 1
                    d["node_types"][ntype] = d["node_types"].get(ntype, 0) + 1
                # Подсчет связей по домену источника
                for edge in edges:
                    if isinstance(edge, dict):
                        src_id = edge.get("source_id") or edge.get("source")
                    else:
                        src_id = getattr(edge, "source_id", None) or getattr(edge, "source", None)
                    if not src_id:
                        continue
                    src_node = kg.get_node(src_id)
                    if src_node is None:
                        continue
                    domain = src_node.get("domain") if isinstance(src_node, dict) else getattr(src_node, "domain", "general")
                    d = domain_stats.setdefault(domain, {"nodes": 0, "edges": 0, "node_types": {}})
                    d["edges"] += 1

            # Общая метка времени
            overall_ts = None
            try:
                overall = kg.get_statistics() if hasattr(kg, 'get_statistics') else None
                if isinstance(overall, dict):
                    overall_ts = overall.get("last_update") or overall.get("last_updated")
            except Exception:
                pass

            # Заполняем таблицу
            for domain, stats in domain_stats.items():
                last_ts = stats.get("last_updated") if isinstance(stats, dict) else None
                last_ts = last_ts or overall_ts or time.time()
                last_updated = datetime.fromtimestamp(last_ts).strftime("%Y-%m-%d %H:%M:%S") if last_ts else "N/A"
                nodes_cnt = stats.get("nodes", 0) if isinstance(stats, dict) else 0
                edges_cnt = stats.get("edges", 0) if isinstance(stats, dict) else 0
                self.domains_tree.insert("", tk.END, values=(
                    domain,
                    nodes_cnt,
                    edges_cnt,
                    last_updated
                ))
        except Exception as e:
            logger.error(f"Ошибка загрузки данных о доменах: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось загрузить данные о доменах: {str(e)}"
            )

    def _create_stats_tab(self, parent):
        """Создает вкладку статистики."""
        # Создаем фрейм для статистики
        stats_frame = ttk.Frame(parent)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем фрейм для общей статистики
        general_frame = ttk.LabelFrame(stats_frame, text="Общая статистика")
        general_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Создаем фрейм для метрик
        metrics_frame = ttk.Frame(general_frame)
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Узлы
        nodes_frame = ttk.Frame(metrics_frame)
        nodes_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(nodes_frame, text="Узлы:", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)
        self.nodes_value = ttk.Label(nodes_frame, text="0", font=('Segoe UI', 9))
        self.nodes_value.pack(anchor=tk.W)
        
        # Связи
        edges_frame = ttk.Frame(metrics_frame)
        edges_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(edges_frame, text="Связи:", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)
        self.edges_value = ttk.Label(edges_frame, text="0", font=('Segoe UI', 9))
        self.edges_value.pack(anchor=tk.W)
        
        # Домены
        domains_frame = ttk.Frame(metrics_frame)
        domains_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(domains_frame, text="Домены:", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)
        self.domains_value = ttk.Label(domains_frame, text="0", font=('Segoe UI', 9))
        self.domains_value.pack(anchor=tk.W)
        
        # Противоречия
        contradictions_frame = ttk.Frame(metrics_frame)
        contradictions_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(contradictions_frame, text="Противоречия:", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)
        self.contradictions_value = ttk.Label(contradictions_frame, text="0", font=('Segoe UI', 9))
        self.contradictions_value.pack(anchor=tk.W)
        
        # Графики статистики
        charts_frame = ttk.LabelFrame(stats_frame, text="Графики статистики")
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем фрейм для двух графиков
        charts_container = ttk.Frame(charts_frame)
        charts_container.pack(fill=tk.BOTH, expand=True)
        
        # График типов узлов
        node_types_frame = ttk.Frame(charts_container, style='Card.TFrame')
        node_types_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._create_node_types_chart(node_types_frame)
        
        # График доменов
        domains_chart_frame = ttk.Frame(charts_container, style='Card.TFrame')
        domains_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._create_domains_chart(domains_chart_frame)
        
        # Загружаем статистику
        self._load_statistics()

    def _create_node_types_chart(self, parent):
        """Создает график типов узлов."""
        # Заголовок
        ttk.Label(parent, text="Типы узлов", font=('Segoe UI', 10, 'bold')).pack(pady=(5, 0))
        
        # Создаем область для графика
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.set_title('Распределение типов узлов', fontsize=10)
        
        # Создаем пустой график
        self.node_types_pie = ax.pie([1], labels=["Загрузка..."], startangle=90)
        
        # Добавляем график в интерфейс
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Сохраняем объекты для обновления
        self.canvas_objects = {
            'node_types': {
                'fig': fig,
                'ax': ax,
                'canvas': canvas,
                'pie': self.node_types_pie
            }
        }

    def _create_domains_chart(self, parent):
        """Создает график доменов."""
        # Заголовок
        ttk.Label(parent, text="Домены", font=('Segoe UI', 10, 'bold')).pack(pady=(5, 0))
        
        # Создаем область для графика
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.set_title('Распределение по доменам', fontsize=10)
        
        # Создаем пустой график
        self.domains_bar = ax.bar(["Загрузка..."], [1])
        
        # Добавляем график в интерфейс
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Сохраняем объекты для обновления
        self.canvas_objects['domains'] = {
            'fig': fig,
            'ax': ax,
            'canvas': canvas,
            'bar': self.domains_bar
        }

    def _load_statistics(self):
        """Загружает статистику о графе знаний."""
        try:
            # Получаем данные из системы
            if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
                logger.warning("Граф знаний недоступен для загрузки статистики")
                return
            
            # Получаем статистику
            kg = self.gui.brain.knowledge_graph
            stats = {}
            try:
                raw_stats = kg.get_statistics()
                if isinstance(raw_stats, dict):
                    stats = raw_stats
            except Exception as e:
                logger.warning(f"Не удалось получить общую статистику из ядра: {e}")
                stats = {}
            # Безопасные значения по умолчанию
            if not stats:
                try:
                    total_nodes = len(kg.get_all_nodes()) if hasattr(kg, 'get_all_nodes') else 0
                    total_edges = len(kg.get_all_edges()) if hasattr(kg, 'get_all_edges') else 0
                except Exception:
                    total_nodes = 0
                    total_edges = 0
                stats = {
                    "total_nodes": total_nodes,
                    "total_edges": total_edges,
                    "node_types": {},
                    "domains": [],
                    "last_updated": time.time(),
                }
            
            # Приводим статистику доменов к формату dict[domain] -> count для графиков
            try:
                domain_stats = kg.get_domain_statistics()
                domain_counts = {}
                for domain, dstats in (domain_stats or {}).items():
                    if isinstance(dstats, dict):
                        count = dstats.get("nodes") or dstats.get("total_nodes") or dstats.get("count") or 0
                    else:
                        # Если вернулся числовой формат
                        count = int(dstats) if isinstance(dstats, (int, float)) else 0
                    domain_counts[domain] = count
                stats["domains"] = domain_counts
            except Exception as e2:
                logger.warning(f"Не удалось получить детальную статистику доменов: {e2}")
                # Fallback: посчитать домены по узлам
                try:
                    domain_counts = {}
                    for node in kg.get_all_nodes():
                        domain = node.get("domain") if isinstance(node, dict) else getattr(node, "domain", "general")
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    stats["domains"] = domain_counts
                except Exception:
                    stats["domains"] = {}
            
            # Обновляем метрики
            self.nodes_value.config(text=str(stats.get("total_nodes", 0)))
            self.edges_value.config(text=str(stats.get("total_edges", 0)))
            domains_obj = stats.get("domains", {})
            domains_count = len(domains_obj) if isinstance(domains_obj, dict) else len(domains_obj) if isinstance(domains_obj, list) else 0
            self.domains_value.config(text=str(domains_count))
            self.contradictions_value.config(text=str(stats.get("contradictions", 0)))
            
            # Обновляем графики
            self._update_statistics_charts(stats)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки статистики: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось загрузить статистику: {str(e)}"
            )

    def _update_statistics_charts(self, stats):
        """Обновляет графики статистики."""
        try:
            # Обновляем график типов узлов
            if 'node_types' in self.canvas_objects:
                ax = self.canvas_objects['node_types']['ax']
                canvas = self.canvas_objects['node_types']['canvas']
                
                # Очищаем график
                ax.clear()
                
                # Подготовка данных
                node_types_dict = stats.get("node_types") or {}
                node_types = list(node_types_dict.keys())
                counts = list(node_types_dict.values())
                
                # Создаем круговую диаграмму
                if sum(counts) > 0:
                    wedges, texts, autotexts = ax.pie(
                        counts, 
                        labels=node_types, 
                        autopct='%1.1f%%',
                        startangle=90
                    )
                    ax.axis('equal')
                    plt.setp(autotexts, size=8, weight="bold")
                else:
                    ax.text(0.5, 0.5, "Нет данных", 
                           horizontalalignment='center',
                           verticalalignment='center')
                
                # Обновляем график
                canvas.draw()
            
            # Обновляем график доменов
            if 'domains' in self.canvas_objects:
                ax = self.canvas_objects['domains']['ax']
                canvas = self.canvas_objects['domains']['canvas']
                
                # Очищаем график
                ax.clear()
                
                # Подготовка данных
                domains_obj = stats.get("domains") or {}
                if isinstance(domains_obj, dict):
                    domains = list(domains_obj.keys())
                    counts = list(domains_obj.values())
                elif isinstance(domains_obj, list):
                    # Если это список доменов без счетчиков — считаем по 1
                    domains = domains_obj
                    counts = [1 for _ in domains_obj]
                else:
                    domains = []
                    counts = []
                
                # Создаем гистограмму
                if len(domains) > 0:
                    ax.bar(range(len(domains)), counts)
                    ax.set_xticks(range(len(domains)))
                    ax.set_xticklabels(domains, rotation=45, ha='right')
                    ax.set_ylabel('Количество')
                else:
                    ax.text(0.5, 0.5, "Нет данных", 
                           horizontalalignment='center',
                           verticalalignment='center')
                
                # Обновляем график
                canvas.draw()
                
        except Exception as e:
            logger.error(f"Ошибка обновления графиков статистики: {e}", exc_info=True)

    def _initialize_graph(self):
        """Инициализирует граф знаний."""
        try:
            # Создаем пустой граф
            self.graph = nx.DiGraph()
            
            # Добавляем тестовые узлы (временно, пока нет данных)
            self.graph.add_node("node1", label="Когнитивные системы", type="concept", domain="AI", strength=0.8)
            self.graph.add_node("node2", label="Этический анализ", type="fact", domain="Ethics", strength=0.7)
            self.graph.add_node("node3", label="Адаптация", type="concept", domain="AI", strength=0.6)
            
            # Добавляем тестовые связи
            self.graph.add_edge("node1", "node2", label="включает", strength=0.9)
            self.graph.add_edge("node1", "node3", label="требует", strength=0.8)
            
            # Позиционируем узлы
            self.pos = nx.spring_layout(self.graph)
            
            # Отображаем граф
            self._update_graph_visualization()
            
        except Exception as e:
            logger.error(f"Ошибка инициализации графа: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось инициализировать граф знаний: {str(e)}"
            )

    def _update_graph_visualization(self):
        """Обновляет визуализацию графа знаний."""
        if not self.graph or not self.graph_canvas:
            return
            
        try:
            # Очищаем текущий график безопасно
            fig = self.graph_canvas.figure
            
            # Проверяем, что фигура и ее атрибуты существуют
            if hasattr(fig, 'clear') and hasattr(self, 'graph_canvas'):
                try:
                    # Сохраняем ссылку на toolbar перед очисткой
                    toolbar = None
                    if hasattr(self.graph_canvas, 'get_tk_widget'):
                        toolbar = self.graph_canvas.manager.toolbar if hasattr(self.graph_canvas.manager, 'toolbar') else None
                    
                    # Очищаем фигуру
                    fig.clear()
                    
                    # Восстанавливаем toolbar если он был уничтожен
                    if toolbar is not None and hasattr(toolbar, 'update'):
                        try:
                            toolbar.update()
                        except:
                            pass  # Игнорируем ошибки toolbar
                            
                except Exception as clear_error:
                    # Если очистка не удалась, создаем новую фигуру
                    import matplotlib.pyplot as plt
                    fig = plt.Figure(figsize=(8, 6), dpi=100)
                    self.graph_canvas.figure = fig
            else:
                # Создаем новую фигуру если старая недоступна
                import matplotlib.pyplot as plt
                fig = plt.Figure(figsize=(8, 6), dpi=100)
                self.graph_canvas.figure = fig
            
            # Создаем новую ось
            ax = fig.add_subplot(111)
            ax.set_title('Граф знаний', fontsize=12)
            ax.axis('off')
            
            # Определяем цвета в зависимости от темы
            bg_color = self.color_scheme["background"]["dark"] if self.gui.theme == "dark" else self.color_scheme["background"]["light"]
            text_color = "white" if self.gui.theme == "dark" else "black"
            edge_color = "#a0a0a0" if self.gui.theme == "dark" else "#666666"
            
            # Устанавливаем цвет фона
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            
            # Всегда пересчитываем позиции узлов для текущего графа,
            # чтобы избежать отсутствующих позиций у новых узлов
            self.pos = nx.spring_layout(self.graph, k=0.5, iterations=50, seed=42)
            
            # Определяем цвета узлов
            node_colors = []
            node_sizes = []
            
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                node_type = node_data.get("type", "fact")
                strength = node_data.get("strength", 0.5)
                
                # Определяем цвет узла
                if node_type in self.color_scheme and "node" in self.color_scheme[node_type]:
                    node_color = self.color_scheme[node_type]["node"]
                else:
                    node_color = self.color_scheme["fact"]["node"]
                
                # Определяем размер узла в зависимости от силы
                node_size = self.node_size * (0.5 + strength)
                
                node_colors.append(node_color)
                node_sizes.append(node_size)
            
            # Определяем цвета и толщину связей
            edge_colors = []
            edge_widths = []
            
            for edge in self.graph.edges():
                edge_data = self.graph.edges[edge]
                strength = edge_data.get("strength", 0.5)
                relation_type = edge_data.get("label", "relation")
                
                # Определяем цвет связи
                if relation_type == "contradicts":
                    edge_color = self.color_scheme["contradiction"]["edge"]
                    edge_width = self.color_scheme["contradiction"].get("width", self.edge_width * 1.5)
                elif relation_type in self.color_scheme and "edge" in self.color_scheme[relation_type]:
                    edge_color = self.color_scheme[relation_type]["edge"]
                    edge_width = self.edge_width
                else:
                    edge_color = self.color_scheme["relation"]["edge"]
                    edge_width = self.edge_width
                
                # Определяем толщину связи в зависимости от силы
                edge_width = edge_width * (0.5 + strength)
                
                edge_colors.append(edge_color)
                edge_widths.append(edge_width)
            
            # Рисуем узлы
            nx.draw_networkx_nodes(
                self.graph, 
                self.pos, 
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.9,
                ax=ax
            )
            
            # Рисуем связи
            nx.draw_networkx_edges(
                self.graph, 
                self.pos, 
                edge_color=edge_colors,
                width=edge_widths,
                alpha=0.7,
                arrows=True,
                arrowsize=15,
                ax=ax
            )
            
            # Рисуем метки узлов
            labels = {node: self.graph.nodes[node].get("label", node) for node in self.graph.nodes()}
            nx.draw_networkx_labels(
                self.graph, 
                self.pos, 
                labels=labels,
                font_size=self.font_size,
                font_color=text_color,
                ax=ax
            )
            
            # Рисуем метки связей
            edge_labels = {edge: self.graph.edges[edge].get("label", "") for edge in self.graph.edges()}
            nx.draw_networkx_edge_labels(
                self.graph, 
                self.pos, 
                edge_labels=edge_labels,
                font_size=max(6, self.font_size - 2),
                font_color=edge_color,
                ax=ax
            )
            
            # Обновляем холст безопасно
            try:
                if hasattr(self, 'graph_canvas') and self.graph_canvas is not None:
                    # Проверяем, что виджет все еще существует
                    widget = self.graph_canvas.get_tk_widget()
                    if widget.winfo_exists():
                        self.graph_canvas.draw()
                    else:
                        logger.debug("Canvas widget больше не существует, пропускаем отрисовку")
                else:
                    logger.debug("Graph canvas недоступен, пропускаем отрисовку")
            except Exception as draw_error:
                logger.debug(f"Ошибка отрисовки canvas: {draw_error}")
            
        except Exception as e:
            logger.error(f"Ошибка обновления визуализации графа: {e}", exc_info=True)

    def _on_graph_click(self, event):
        """Обрабатывает клик по графу."""
        if not self.graph or not self.pos:
            return
            
        # Проверяем, кликнули ли мы на узел
        for node, (x, y) in self.pos.items():
            # Преобразуем координаты в систему координат графика
            x_fig, y_fig = self.graph_canvas.figure.transFigure.inverted().transform((event.x, event.y))
            x_node, y_node = self.graph_canvas.figure.transFigure.transform((x, y))
            
            # Проверяем расстояние
            distance = np.sqrt((x_fig - x_node)**2 + (y_fig - y_node)**2)
            if distance < 0.02:  # Пороговое значение
                self.selected_node = node
                self.selected_edge = None
                self._show_node_info(node)
                return
        
        # Проверяем, кликнули ли мы на связь
        for edge in self.graph.edges():
            source, target = edge
            x1, y1 = self.pos[source]
            x2, y2 = self.pos[target]
            
            # Преобразуем координаты
            x1_fig, y1_fig = self.graph_canvas.figure.transFigure.transform((x1, y1))
            x2_fig, y2_fig = self.graph_canvas.figure.transFigure.transform((x2, y2))
            x_fig, y_fig = self.graph_canvas.figure.transFigure.inverted().transform((event.x, event.y))
            
            # Вычисляем расстояние до линии
            A = y2_fig - y1_fig
            B = x1_fig - x2_fig
            C = x2_fig*y1_fig - x1_fig*y2_fig
            distance = abs(A*x_fig + B*y_fig + C) / np.sqrt(A**2 + B**2)
            
            if distance < 0.02:
                self.selected_edge = edge
                self.selected_node = None
                self._show_edge_info(edge)
                return
        
        # Сброс выделения
        self.selected_node = None
        self.selected_edge = None
        self._clear_info_panel()

    def _on_graph_hover(self, event):
        """Обрабатывает наведение курсора на граф."""
        if not self.graph or not self.pos or event.inaxes is None:
            return
            
        # Проверяем, навели ли мы на узел
        for node, (x, y) in self.pos.items():
            # Преобразуем координаты в систему координат графика
            x_fig, y_fig = self.graph_canvas.figure.transFigure.inverted().transform((event.x, event.y))
            x_node, y_node = self.graph_canvas.figure.transFigure.transform((x, y))
            
            # Проверяем расстояние
            distance = np.sqrt((x_fig - x_node)**2 + (y_fig - y_node)**2)
            if distance < 0.02:  # Пороговое значение
                self.gui.root.config(cursor="hand2")
                return
        
        # Проверяем, навели ли мы на связь
        for edge in self.graph.edges():
            source, target = edge
            x1, y1 = self.pos[source]
            x2, y2 = self.pos[target]
            
            # Преобразуем координаты
            x1_fig, y1_fig = self.graph_canvas.figure.transFigure.transform((x1, y1))
            x2_fig, y2_fig = self.graph_canvas.figure.transFigure.transform((x2, y2))
            x_fig, y_fig = self.graph_canvas.figure.transFigure.inverted().transform((event.x, event.y))
            
            # Вычисляем расстояние до линии
            A = y2_fig - y1_fig
            B = x1_fig - x2_fig
            C = x2_fig*y1_fig - x1_fig*y2_fig
            distance = abs(A*x_fig + B*y_fig + C) / np.sqrt(A**2 + B**2)
            
            if distance < 0.02:
                self.gui.root.config(cursor="hand2")
                return
        
        # Возвращаем обычный курсор
        self.gui.root.config(cursor="")

    def _on_graph_scroll(self, event):
        """Обрабатывает прокрутку колесика мыши на графе."""
        if not self.graph or not self.pos:
            return
            
        # Масштабирование
        scale_factor = 1.1 if event.button == 'up' else 0.9
        
        # Обновляем позиции
        for node in self.pos:
            x, y = self.pos[node]
            self.pos[node] = (x * scale_factor, y * scale_factor)
        
        # Обновляем визуализацию
        self._update_graph_visualization()

    def _show_node_info(self, node_id):
        """Показывает информацию об узле."""
        # Очищаем панель
        self._clear_info_panel()
        
        try:
            # Получаем данные узла
            node_data = self.graph.nodes[node_id]
            
            # Создаем фрейм для информации
            info_frame = ttk.Frame(self.info_content)
            info_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Название
            ttk.Label(
                info_frame, 
                text="Название:", 
                font=('Segoe UI', 9, 'bold')
            ).grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(
                info_frame, 
                text=node_data.get("label", node_id), 
                font=('Segoe UI', 9)
            ).grid(row=0, column=1, sticky=tk.W, pady=2)
            
            # Тип
            ttk.Label(
                info_frame, 
                text="Тип:", 
                font=('Segoe UI', 9, 'bold')
            ).grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(
                info_frame, 
                text=node_data.get("type", "N/A"), 
                font=('Segoe UI', 9)
            ).grid(row=1, column=1, sticky=tk.W, pady=2)
            
            # Домен
            ttk.Label(
                info_frame, 
                text="Домен:", 
                font=('Segoe UI', 9, 'bold')
            ).grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(
                info_frame, 
                text=node_data.get("domain", "N/A"), 
                font=('Segoe UI', 9)
            ).grid(row=2, column=1, sticky=tk.W, pady=2)
            
            # Сила
            ttk.Label(
                info_frame, 
                text="Сила:", 
                font=('Segoe UI', 9, 'bold')
            ).grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(
                info_frame, 
                text=f"{node_data.get('strength', 0.0):.2f}", 
                font=('Segoe UI', 9)
            ).grid(row=3, column=1, sticky=tk.W, pady=2)
            
            # Описание
            ttk.Label(
                info_frame, 
                text="Описание:", 
                font=('Segoe UI', 9, 'bold')
            ).grid(row=4, column=0, sticky=tk.NW, padx=(0, 10), pady=(2, 0))
            
            description = node_data.get("description", "Описание отсутствует")
            description_label = ttk.Label(
                info_frame, 
                text=description, 
                font=('Segoe UI', 9),
                wraplength=600
            )
            description_label.grid(row=4, column=1, sticky=tk.W, pady=(2, 0))
            
            # Кнопки действий
            btn_frame = ttk.Frame(info_frame)
            btn_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
            
            ttk.Button(
                btn_frame,
                text="Показать связанные узлы",
                command=lambda: self._show_related_nodes(node_id)
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                btn_frame,
                text="Экспорт узла",
                command=lambda: self._export_node(node_id)
            ).pack(side=tk.LEFT, padx=5)
            
            # Дополнительные действия в зависимости от типа узла
            node_type = node_data.get("type", "")
            if node_type == "contradiction":
                ttk.Button(
                    btn_frame,
                    text="Разрешить противоречие",
                    command=lambda: self._resolve_contradiction(node_id)
                ).pack(side=tk.LEFT, padx=5)
                
        except Exception as e:
            logger.error(f"Ошибка отображения информации об узле: {e}", exc_info=True)
            self._show_error_in_info_panel(f"Ошибка при отображении информации: {str(e)}")

    def _show_edge_info(self, edge):
        """Показывает информацию о связи."""
        # Очищаем панель
        self._clear_info_panel()
        
        try:
            # Получаем данные связи
            edge_data = self.graph.edges[edge]
            source, target = edge
            
            # Создаем фрейм для информации
            info_frame = ttk.Frame(self.info_content)
            info_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Тип связи
            ttk.Label(
                info_frame, 
                text="Тип связи:", 
                font=('Segoe UI', 9, 'bold')
            ).grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(
                info_frame, 
                text=edge_data.get("label", "N/A"), 
                font=('Segoe UI', 9)
            ).grid(row=0, column=1, sticky=tk.W, pady=2)
            
            # Сила связи
            ttk.Label(
                info_frame, 
                text="Сила связи:", 
                font=('Segoe UI', 9, 'bold')
            ).grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(
                info_frame, 
                text=f"{edge_data.get('strength', 0.0):.2f}", 
                font=('Segoe UI', 9)
            ).grid(row=1, column=1, sticky=tk.W, pady=2)
            
            # Узлы
            source_data = self.graph.nodes[source]
            target_data = self.graph.nodes[target]
            
            ttk.Label(
                info_frame, 
                text="Источник:", 
                font=('Segoe UI', 9, 'bold')
            ).grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(
                info_frame, 
                text=f"{source_data.get('label', source)} ({source_data.get('type', 'N/A')})", 
                font=('Segoe UI', 9)
            ).grid(row=2, column=1, sticky=tk.W, pady=2)
            
            ttk.Label(
                info_frame, 
                text="Цель:", 
                font=('Segoe UI', 9, 'bold')
            ).grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(
                info_frame, 
                text=f"{target_data.get('label', target)} ({target_data.get('type', 'N/A')})", 
                font=('Segoe UI', 9)
            ).grid(row=3, column=1, sticky=tk.W, pady=2)
            
        except Exception as e:
            logger.error(f"Ошибка отображения информации о связи: {e}", exc_info=True)
            self._show_error_in_info_panel(f"Ошибка при отображении информации: {str(e)}")

    def _clear_info_panel(self):
        """Очищает информационную панель."""
        for widget in self.info_content.winfo_children():
            widget.destroy()

    def _show_error_in_info_panel(self, message):
        """Показывает сообщение об ошибке в информационной панели."""
        self._clear_info_panel()
        
        ttk.Label(
            self.info_content, 
            text=message,
            foreground="red",
            font=('Segoe UI', 9, 'italic'),
            wraplength=800
        ).pack(fill=tk.X, padx=5, pady=5)

    def _refresh_graph(self):
        """Обновляет граф знаний."""
        self.current_view = "all"
        self._update_graph_data()

    def _show_search_dialog(self):
        """Показывает диалог поиска."""
        search_window = tk.Toplevel(self.gui.root)
        search_window.title("Поиск в графе знаний")
        search_window.geometry("400x150")
        search_window.transient(self.gui.root)
        search_window.grab_set()
        
        # Фрейм для ввода
        input_frame = ttk.Frame(search_window, padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(input_frame, text="Введите запрос:").pack(anchor=tk.W, pady=(0, 5))
        
        search_entry = ttk.Entry(input_frame, width=50)
        search_entry.pack(fill=tk.X, pady=(0, 10))
        search_entry.focus()
        
        # Кнопки
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def perform_search():
            query = search_entry.get().strip()
            if query:
                self._perform_search(query)
                search_window.destroy()
        
        ttk.Button(
            button_frame,
            text="Поиск",
            command=perform_search
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="Отмена",
            command=search_window.destroy
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))
        
        # Обработка Enter
        search_entry.bind("<Return>", lambda event: perform_search())

    def _perform_search(self, query=None):
        """Выполняет поиск в графе знаний."""
        if not query:
            query = self.search_entry.get().strip()
            if not query:
                return
        
        try:
            # Получаем доступ к графу знаний
            if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
                raise ValueError("Граф знаний недоступен")
            
            # Выполняем поиск
            results = self.gui.brain.knowledge_graph.search_nodes(query)
            
            # Сохраняем результаты
            self.search_query = query
            self.search_results = results
            self.search_index = 0
            
            # Обновляем текущий вид
            self.current_view = "search"
            
            # Обновляем граф
            self._update_graph_data()
            
            # Обновляем результаты поиска
            self._update_search_results()
            
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось выполнить поиск: {str(e)}"
            )

    def _update_search_results(self):
        """Обновляет результаты поиска в таблице."""
        # Очищаем таблицу
        for item in self.search_tree.get_children():
            self.search_tree.delete(item)
        
        # Заполняем таблицу результатами
        for result in self.search_results:
            if isinstance(result, dict):
                name = result.get("name") or result.get("label") or "N/A"
                node_type = result.get("node_type") or result.get("type") or "N/A"
                domain = result.get("domain", "N/A")
                strength_val = result.get("strength", 0)
                description = result.get("description") or result.get("content") or "N/A"
            else:
                name = getattr(result, "name", None) or getattr(result, "content", None) or "N/A"
                node_type = getattr(result, "node_type", "N/A")
                domain = getattr(result, "domain", "N/A")
                strength_val = getattr(result, "strength", 0)
                description = getattr(result, "description", None) or getattr(result, "content", None) or "N/A"
            try:
                strength = f"{float(strength_val):.2f}"
            except Exception:
                strength = "0.00"
            self.search_tree.insert("", tk.END, values=(
                name,
                node_type,
                domain,
                strength,
                description
            ))

    def _on_search_result_double_click(self, event):
        """Обрабатывает двойной клик по результату поиска."""
        item = self.search_tree.selection()
        if item:
            self._highlight_search_result()

    def _highlight_search_result(self):
        """Выделяет результат поиска на графике."""
        item = self.search_tree.selection()
        if not item:
            return
        
        # Получаем данные
        values = self.search_tree.item(item, "values")
        if not values:
            return
        
        # Ищем узел по названию
        node_id = None
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if node_data.get("label", "") == values[0]:
                node_id = node
                break
        
        if node_id:
            # Выделяем узел
            self.selected_node = node_id
            self.selected_edge = None
            self._show_node_info(node_id)
            
            # Центрируем граф на узле
            self._center_graph_on_node(node_id)

    def _show_search_result_details(self):
        """Показывает детали результата поиска."""
        item = self.search_tree.selection()
        if not item:
            return
        
        # Получаем данные
        values = self.search_tree.item(item, "values")
        if not values:
            return
        
        # Ищем узел по названию
        node_id = None
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if node_data.get("label", "") == values[0]:
                node_id = node
                break
        
        if node_id:
            self._show_node_details(node_id)

    def _export_search_result(self):
        """Экспортирует результат поиска."""
        item = self.search_tree.selection()
        if not item:
            return
        
        # Получаем данные
        values = self.search_tree.item(item, "values")
        if not values:
            return
        
        try:
            # Диалог выбора файла
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")],
                title="Экспорт результата поиска"
            )
            
            if not file_path:
                return  # Пользователь отменил операцию
            
            # Находим узел
            node_id = None
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                if node_data.get("label", "") == values[0]:
                    node_id = node
                    break
            
            if not node_id:
                raise ValueError("Узел не найден")
            
            # Получаем данные узла
            node_data = self.graph.nodes[node_id]
            
            # Создаем данные для экспорта
            export_data = {
                "node": {
                    "id": node_id,
                    "name": node_data.get("label", "N/A"),
                    "type": node_data.get("type", "N/A"),
                    "domain": node_data.get("domain", "N/A"),
                    "strength": node_data.get("strength", 0.0),
                    "description": node_data.get("description", "N/A")
                },
                "related_nodes": [],
                "related_edges": [],
                "export_time": time.time(),
                "format_version": "1.0"
            }
            
            # Добавляем связанные узлы и связи
            for edge in self.graph.edges(node_id):
                # Добавляем связь
                edge_data = self.graph.edges[edge]
                export_data["related_edges"].append({
                    "id": edge_data.get("id", ""),
                    "source_id": edge[0],
                    "target_id": edge[1],
                    "relation_type": edge_data.get("label", ""),
                    "strength": edge_data.get("strength", 0.0)
                })
                
                # Добавляем связанный узел
                other_node = edge[1] if edge[0] == node_id else edge[0]
                other_node_data = self.graph.nodes[other_node]
                export_data["related_nodes"].append({
                    "id": other_node,
                    "name": other_node_data.get("label", "N/A"),
                    "type": other_node_data.get("type", "N/A"),
                    "domain": other_node_data.get("domain", "N/A"),
                    "strength": other_node_data.get("strength", 0.0)
                })
            
            # Сохраняем в JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo(
                "Успех", 
                f"Результат поиска успешно экспортирован в {os.path.basename(file_path)}"
            )
            
        except Exception as e:
            logger.error(f"Ошибка экспорта результата поиска: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось экспортировать результат: {str(e)}"
            )

    def _show_node_details(self, node_id=None):
        """Показывает детали узла."""
        if not node_id:
            if not self.selected_node:
                return
            node_id = self.selected_node
        
        try:
            # Получаем данные узла
            if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
                raise ValueError("Граф знаний недоступен")
            
            node = self.gui.brain.knowledge_graph.get_node(node_id)
            if not node:
                raise ValueError("Узел не найден")
            
            # Создаем окно с деталями
            details_window = tk.Toplevel(self.gui.root)
            details_window.title(f"Детали: {node['name']}")
            details_window.geometry("600x500")
            details_window.transient(self.gui.root)
            details_window.grab_set()
            
            # Создаем вкладки
            notebook = ttk.Notebook(details_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Вкладка основной информации
            info_frame = ttk.Frame(notebook)
            notebook.add(info_frame, text="Основная информация")
            
            # Формируем информацию
            info_text = (
                f"Название: {node.get('name', 'N/A')}\n"
                f"Тип: {node.get('node_type', 'N/A')}\n"
                f"Домен: {node.get('domain', 'N/A')}\n"
                f"Сила: {node.get('strength', 0.0):.2f}\n\n"
                f"Описание: {node.get('description', 'N/A')}\n\n"
            )
            
            # Отображаем информацию
            info_label = ttk.Label(
                info_frame, 
                text=info_text,
                justify=tk.LEFT,
                wraplength=550
            )
            info_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Вкладка связей
            edges_frame = ttk.Frame(notebook)
            notebook.add(edges_frame, text="Связи")
            
            # Создаем Treeview для связей
            columns = ("source", "relation", "target", "strength")
            edges_tree = ttk.Treeview(
                edges_frame,
                columns=columns,
                show="headings",
                selectmode="browse"
            )
            
            # Настройка колонок
            edges_tree.heading("source", text="Источник")
            edges_tree.heading("relation", text="Связь")
            edges_tree.heading("target", text="Цель")
            edges_tree.heading("strength", text="Сила")
            
            edges_tree.column("source", width=150, anchor=tk.W)
            edges_tree.column("relation", width=100, anchor=tk.W)
            edges_tree.column("target", width=150, anchor=tk.W)
            edges_tree.column("strength", width=80, anchor=tk.CENTER)
            
            # Добавляем прокрутку
            scrollbar = ttk.Scrollbar(edges_frame, orient=tk.VERTICAL, command=edges_tree.yview)
            edges_tree.configure(yscrollcommand=scrollbar.set)
            
            # Размещаем Treeview и скроллбар
            edges_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Заполняем данными
            if self.gui.brain and hasattr(self.gui.brain, 'knowledge_graph'):
                edges = self.gui.brain.knowledge_graph.get_edges(node_id)
                for edge in edges:
                    source = edge["source_id"]
                    target = edge["target_id"]
                    
                    # Получаем названия узлов
                    source_node = self.gui.brain.knowledge_graph.get_node(source)
                    target_node = self.gui.brain.knowledge_graph.get_node(target)
                    
                    source_name = source_node["name"] if source_node else "N/A"
                    target_name = target_node["name"] if target_node else "N/A"
                    
                    # Добавляем запись в таблицу
                    edges_tree.insert("", tk.END, values=(
                        source_name,
                        edge["relation_type"],
                        target_name,
                        f"{edge['strength']:.2f}"
                    ))
            
            # Вкладка истории
            history_frame = ttk.Frame(notebook)
            notebook.add(history_frame, text="История")
            
            history_text = scrolledtext.ScrolledText(
                history_frame,
                wrap=tk.WORD,
                font=('Segoe UI', 9),
                state=tk.DISABLED
            )
            history_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Добавляем историю
            if self.gui.brain and hasattr(self.gui.brain, 'knowledge_graph'):
                history = self.gui.brain.knowledge_graph.get_node_history(node_id)
                for entry in history:
                    timestamp = datetime.fromtimestamp(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    history_text.config(state=tk.NORMAL)
                    history_text.insert(tk.END, f"[{timestamp}] {entry['action']}\n")
                    history_text.insert(tk.END, f"   {entry['details']}\n\n")
                    history_text.config(state=tk.DISABLED)
            
            # Кнопка закрытия
            ttk.Button(
                details_window,
                text="Закрыть",
                command=details_window.destroy
            ).pack(pady=5)
            
        except Exception as e:
            logger.error(f"Ошибка отображения деталей узла: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось отобразить детали: {str(e)}"
            )

    def _show_related_nodes(self, node_id=None):
        """Показывает связанные узлы."""
        if not node_id:
            if not self.selected_node:
                return
            node_id = self.selected_node
        
        try:
            # Получаем данные узла
            if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
                raise ValueError("Граф знаний недоступен")
            
            # Получаем связанные узлы
            edges = self.gui.brain.knowledge_graph.get_edges(node_id)
            
            # Формируем подграф
            node_ids = [node_id]
            for edge in edges:
                node_ids.append(edge["source_id"])
                node_ids.append(edge["target_id"])
            
            # Создаем окно с подграфом
            subgraph_window = tk.Toplevel(self.gui.root)
            subgraph_window.title(f"Связанные узлы: {node_id}")
            subgraph_window.geometry("800x600")
            subgraph_window.transient(self.gui.root)
            subgraph_window.grab_set()
            
            # Создаем фрейм для графа
            graph_frame = ttk.Frame(subgraph_window)
            graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Создаем граф
            G = nx.DiGraph()
            
            # Добавляем узлы
            for node_id in set(node_ids):
                node = self.gui.brain.knowledge_graph.get_node(node_id)
                if node:
                    G.add_node(
                        node_id,
                        label=node["name"],
                        type=node["node_type"],
                        domain=node["domain"],
                        strength=node["strength"]
                    )
            
            # Добавляем связи
            for edge in edges:
                G.add_edge(
                    edge["source_id"],
                    edge["target_id"],
                    label=edge["relation_type"],
                    strength=edge["strength"]
                )
            
            # Создаем область для графа
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title('Связанные узлы', fontsize=12)
            ax.axis('off')
            
            # Позиционируем узлы
            pos = nx.spring_layout(G)
            
            # Определяем цвета в зависимости от темы
            bg_color = self.color_scheme["background"]["dark"] if self.gui.theme == "dark" else self.color_scheme["background"]["light"]
            text_color = "white" if self.gui.theme == "dark" else "black"
            edge_color = "#a0a0a0" if self.gui.theme == "dark" else "#666666"
            
            # Устанавливаем цвет фона
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            
            # Определяем цвета узлов
            node_colors = []
            node_sizes = []
            
            for node in G.nodes():
                node_data = G.nodes[node]
                node_type = node_data.get("type", "fact")
                strength = node_data.get("strength", 0.5)
                
                # Определяем цвет узла
                if node_type in self.color_scheme and "node" in self.color_scheme[node_type]:
                    node_color = self.color_scheme[node_type]["node"]
                else:
                    node_color = self.color_scheme["fact"]["node"]
                
                # Определяем размер узла в зависимости от силы
                node_size = self.node_size * (0.5 + strength)
                
                node_colors.append(node_color)
                node_sizes.append(node_size)
            
            # Определяем цвета и толщину связей
            edge_colors = []
            edge_widths = []
            
            for edge in G.edges():
                edge_data = G.edges[edge]
                strength = edge_data.get("strength", 0.5)
                
                # Определяем цвет связи
                edge_color = self.color_scheme["relation"]["edge"]
                edge_width = self.edge_width
                
                # Определяем толщину связи в зависимости от силы
                edge_width = edge_width * (0.5 + strength)
                
                edge_colors.append(edge_color)
                edge_widths.append(edge_width)
            
            # Рисуем узлы
            nx.draw_networkx_nodes(
                G, 
                pos, 
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.9,
                ax=ax
            )
            
            # Рисуем связи
            nx.draw_networkx_edges(
                G, 
                pos, 
                edge_color=edge_colors,
                width=edge_widths,
                alpha=0.7,
                arrows=True,
                arrowsize=15,
                ax=ax
            )
            
            # Рисуем метки узлов
            labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
            nx.draw_networkx_labels(
                G, 
                pos, 
                labels=labels,
                font_size=self.font_size,
                font_color=text_color,
                ax=ax
            )
            
            # Рисуем метки связей
            edge_labels = {edge: G.edges[edge].get("label", "") for edge in G.edges()}
            nx.draw_networkx_edge_labels(
                G, 
                pos, 
                edge_labels=edge_labels,
                font_size=max(6, self.font_size - 2),
                font_color=edge_color,
                ax=ax
            )
            
            # Создаем холст для графа
            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Добавляем навигационную панель
            toolbar = NavigationToolbar2Tk(canvas, graph_frame)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Кнопка закрытия
            ttk.Button(
                subgraph_window,
                text="Закрыть",
                command=subgraph_window.destroy
            ).pack(pady=5)
            
        except Exception as e:
            logger.error(f"Ошибка отображения связанных узлов: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось отобразить связанные узлы: {str(e)}"
            )

    def _search_by_domain(self):
        """Ищет узлы по домену."""
        if not self.selected_node:
            return
        
        try:
            # Получаем данные узла
            node_data = self.graph.nodes[self.selected_node]
            domain = node_data.get("domain", "")
            
            if not domain:
                messagebox.showinfo("Информация", "У этого узла не указан домен")
                return
            
            # Обновляем текущий вид
            self.current_view = f"domain:{domain}"
            
            # Обновляем граф
            self._update_graph_data()
            
            # Переключаемся на вкладку доменов
            for i in range(self.notebook.index("end")):
                if self.notebook.tab(i, "text") == "Домены":
                    self.notebook.select(i)
                    break
            
        except Exception as e:
            logger.error(f"Ошибка поиска по домену: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось выполнить поиск по домену: {str(e)}"
            )

    def _export_node(self, node_id=None):
        """Экспортирует узел в файл."""
        if not node_id:
            if not self.selected_node:
                return
            node_id = self.selected_node
        
        try:
            # Получаем данные узла
            if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
                raise ValueError("Граф знаний недоступен")
            
            node = self.gui.brain.knowledge_graph.get_node(node_id)
            if not node:
                raise ValueError("Узел не найден")
            
            # Диалог выбора файла
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")],
                title="Экспорт узла"
            )
            
            if not file_path:
                return  # Пользователь отменил операцию
            
            # Получаем связанные узлы и связи
            edges = self.gui.brain.knowledge_graph.get_edges(node_id)
            related_nodes = []
            for edge in edges:
                source_id = edge["source_id"]
                target_id = edge["target_id"]
                if source_id != node_id:
                    related_nodes.append(source_id)
                if target_id != node_id:
                    related_nodes.append(target_id)
            
            # Создаем данные для экспорта
            export_data = {
                "node": {
                    "id": node_id,
                    "name": node["name"],
                    "description": node["description"],
                    "node_type": node["node_type"],
                    "domain": node["domain"],
                    "strength": node["strength"],
                    "timestamp": node["timestamp"],
                    "meta": node["meta"]
                },
                "related_nodes": [],
                "related_edges": [],
                "export_time": time.time(),
                "format_version": "1.0"
            }
            
            # Добавляем связанные узлы
            for related_id in set(related_nodes):
                related_node = self.gui.brain.knowledge_graph.get_node(related_id)
                if related_node:
                    export_data["related_nodes"].append({
                        "id": related_id,
                        "name": related_node["name"],
                        "description": related_node["description"],
                        "node_type": related_node["node_type"],
                        "domain": related_node["domain"],
                        "strength": related_node["strength"]
                    })
            
            # Добавляем связанные связи
            for edge in edges:
                export_data["related_edges"].append({
                    "id": edge["id"],
                    "source_id": edge["source_id"],
                    "target_id": edge["target_id"],
                    "relation_type": edge["relation_type"],
                    "strength": edge["strength"],
                    "timestamp": edge["timestamp"],
                    "meta": edge["meta"]
                })
            
            # Сохраняем в JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo(
                "Успех", 
                f"Узел успешно экспортирован в {os.path.basename(file_path)}"
            )
            
        except Exception as e:
            logger.error(f"Ошибка экспорта узла: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось экспортировать узел: {str(e)}"
            )

    def _export_graph(self):
        """Экспортирует граф знаний в файл."""
        try:
            # Диалог выбора файла
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")],
                title="Экспорт графа знаний"
            )
            
            if not file_path:
                return  # Пользователь отменил операцию
            
            # Получаем данные из системы
            if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
                raise ValueError("Граф знаний недоступен")
            
            # Получаем все узлы и связи
            nodes = self.gui.brain.knowledge_graph.get_all_nodes()
            edges = self.gui.brain.knowledge_graph.get_all_edges()
            
            # Создаем данные для экспорта
            export_data = {
                "metadata": {
                    "export_time": time.time(),
                    "format_version": "1.0",
                    "system_name": "CogniFlex"
                },
                "nodes": nodes,
                "edges": edges
            }
            
            # Сохраняем в JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo(
                "Успех", 
                f"Граф знаний успешно экспортирован в {os.path.basename(file_path)}"
            )
            
        except Exception as e:
            logger.error(f"Ошибка экспорта графа: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось экспортировать граф: {str(e)}"
            )

    def _show_settings_dialog(self):
        """Показывает диалог настроек графа."""
        settings_window = tk.Toplevel(self.gui.root)
        settings_window.title("Настройки графа знаний")
        settings_window.geometry("400x300")
        settings_window.transient(self.gui.root)
        settings_window.grab_set()
        
        # Фрейм для настроек
        settings_frame = ttk.Frame(settings_window, padding=10)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Размер узлов
        ttk.Label(settings_frame, text="Размер узлов:").pack(anchor=tk.W, pady=(0, 5))
        node_size_var = tk.IntVar(value=int(self.node_size / 10))
        node_size_slider = ttk.Scale(
            settings_frame,
            from_=5,
            to=50,
            orient=tk.HORIZONTAL,
            variable=node_size_var
        )
        node_size_slider.pack(fill=tk.X, pady=(0, 15))
        
        # Толщина связей
        ttk.Label(settings_frame, text="Толщина связей:").pack(anchor=tk.W, pady=(0, 5))
        edge_width_var = tk.DoubleVar(value=self.edge_width)
        edge_width_slider = tk.Scale(
            settings_frame,
            from_=0.5,
            to=5.0,
            orient=tk.HORIZONTAL,
            variable=edge_width_var,
            resolution=0.1
        )
        edge_width_slider.pack(fill=tk.X, pady=(0, 15))
        
        # Размер шрифта
        ttk.Label(settings_frame, text="Размер шрифта:").pack(anchor=tk.W, pady=(0, 5))
        font_size_var = tk.IntVar(value=self.font_size)
        font_size_slider = ttk.Scale(
            settings_frame,
            from_=6,
            to=16,
            orient=tk.HORIZONTAL,
            variable=font_size_var
        )
        font_size_slider.pack(fill=tk.X, pady=(0, 15))
        
        # Кнопки
        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def apply_settings():
            # Применяем настройки
            self.node_size = node_size_var.get() * 10
            self.edge_width = edge_width_var.get()
            self.font_size = font_size_var.get()
            
            # Обновляем граф
            self._update_graph_visualization()
            
            settings_window.destroy()
        
        ttk.Button(
            button_frame,
            text="Применить",
            command=apply_settings
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="Отмена",
            command=settings_window.destroy
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

    def _on_tab_changed(self, event):
        """Обрабатывает смену вкладки."""
        # Получаем текущую вкладку
        current_tab = self.notebook.select()
        tab_name = self.notebook.tab(current_tab, "text")
        
        # Обновляем данные в зависимости от вкладки
        if tab_name == "Статистика":
            self._load_statistics()
        elif tab_name == "Домены":
            self._load_domains_data()

    def _on_domain_double_click(self, event):
        """Обрабатывает двойной клик по домену."""
        item = self.domains_tree.selection()
        if item:
            self._show_domain_on_graph()

    def _show_domain_on_graph(self):
        """Показывает домен на графике."""
        item = self.domains_tree.selection()
        if not item:
            return
        
        # Получаем данные
        values = self.domains_tree.item(item, "values")
        if not values:
            return
        
        domain = values[0]
        
        # Обновляем текущий вид
        self.current_view = f"domain:{domain}"
        
        # Обновляем граф
        self._update_graph_data()

    def _show_domain_details(self):
        """Показывает детали домена."""
        item = self.domains_tree.selection()
        if not item:
            return
        
        # Получаем данные
        values = self.domains_tree.item(item, "values")
        if not values:
            return
        
        domain = values[0]
        
        try:
            # Создаем окно с деталями
            details_window = tk.Toplevel(self.gui.root)
            details_window.title(f"Домен: {domain}")
            details_window.geometry("600x400")
            details_window.transient(self.gui.root)
            details_window.grab_set()
            
            # Создаем вкладки
            notebook = ttk.Notebook(details_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Вкладка информации
            info_frame = ttk.Frame(notebook)
            notebook.add(info_frame, text="Информация")
            
            # Формируем информацию
            info_text = (
                f"Домен: {domain}\n"
                f"Узлы: {values[1]}\n"
                f"Связи: {values[2]}\n"
                f"Последнее обновление: {values[3]}\n\n"
                f"Домены представляют собой категории знаний в системе.\n"
                f"Каждый узел графа знаний относится к определенному домену."
            )
            
            # Отображаем информацию
            info_label = ttk.Label(
                info_frame, 
                text=info_text,
                justify=tk.LEFT,
                wraplength=550
            )
            info_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Вкладка узлов
            nodes_frame = ttk.Frame(notebook)
            notebook.add(nodes_frame, text="Узлы")
            
            # Создаем Treeview для узлов
            columns = ("name", "type", "strength", "description")
            nodes_tree = ttk.Treeview(
                nodes_frame,
                columns=columns,
                show="headings",
                selectmode="browse"
            )
            
            # Настройка колонок
            nodes_tree.heading("name", text="Название")
            nodes_tree.heading("type", text="Тип")
            nodes_tree.heading("strength", text="Сила")
            nodes_tree.heading("description", text="Описание")
            
            nodes_tree.column("name", width=150, anchor=tk.W)
            nodes_tree.column("type", width=100, anchor=tk.W)
            nodes_tree.column("strength", width=80, anchor=tk.CENTER)
            nodes_tree.column("description", width=200, anchor=tk.W)
            
            # Добавляем прокрутку
            scrollbar = ttk.Scrollbar(nodes_frame, orient=tk.VERTICAL, command=nodes_tree.yview)
            nodes_tree.configure(yscrollcommand=scrollbar.set)
            
            # Размещаем Treeview и скроллбар
            nodes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Заполняем данными
            if self.gui.brain and hasattr(self.gui.brain, 'knowledge_graph'):
                nodes = self.gui.brain.knowledge_graph.get_nodes_by_domain(domain)
                for node in nodes:
                    strength = f"{node.get('strength', 0):.2f}"
                    nodes_tree.insert("", tk.END, values=(
                        node.get("name", "N/A"),
                        node.get("node_type", "N/A"),
                        strength,
                        node.get("description", "N/A")
                    ))
            
            # Кнопка закрытия
            ttk.Button(
                details_window,
                text="Закрыть",
                command=details_window.destroy
            ).pack(pady=5)
            
        except Exception as e:
            logger.error(f"Ошибка отображения деталей домена: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось отобразить детали: {str(e)}"
            )

    def _export_domain(self):
        """Экспортирует домен в файл."""
        item = self.domains_tree.selection()
        if not item:
            return
        
        # Получаем данные
        values = self.domains_tree.item(item, "values")
        if not values:
            return
        
        domain = values[0]
        
        try:
            # Диалог выбора файла
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")],
                title=f"Экспорт домена {domain}"
            )
            
            if not file_path:
                return  # Пользователь отменил операцию
            
            # Получаем данные из системы
            if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
                raise ValueError("Граф знаний недоступен")
            
            # Получаем узлы и связи домена
            nodes = self.gui.brain.knowledge_graph.get_nodes_by_domain(domain)
            node_ids = [node["id"] for node in nodes]
            
            # Получаем все связи для этих узлов
            edges = []
            for node_id in node_ids:
                node_edges = self.gui.brain.knowledge_graph.get_edges(node_id)
                edges.extend(node_edges)
            
            # Создаем данные для экспорта
            export_data = {
                "metadata": {
                    "domain": domain,
                    "export_time": time.time(),
                    "format_version": "1.0",
                    "system_name": "CogniFlex"
                },
                "nodes": nodes,
                "edges": edges
            }
            
            # Сохраняем в JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo(
                "Успех", 
                f"Домен '{domain}' успешно экспортирован в {os.path.basename(file_path)}"
            )
            
        except Exception as e:
            logger.error(f"Ошибка экспорта домена: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось экспортировать домен: {str(e)}"
            )

    def _center_graph_on_node(self, node_id):
        """Центрирует граф на указанном узле."""
        if not self.graph or not self.pos or node_id not in self.pos:
            return
        
        # Получаем позицию узла
        x, y = self.pos[node_id]
        
        # Центрируем граф
        for node in self.pos:
            self.pos[node] = (self.pos[node][0] - x, self.pos[node][1] - y)
        
        # Обновляем визуализацию
        self._update_graph_visualization()

    def _resolve_contradiction(self, node_id):
        """Запускает процесс разрешения противоречия."""
        try:
            # Проверяем, что это противоречие
            node_data = self.graph.nodes[node_id]
            if node_data.get("type", "") != "contradiction":
                messagebox.showinfo("Информация", "Этот узел не является противоречием")
                return
            
            # Создаем окно разрешения
            resolve_window = tk.Toplevel(self.gui.root)
            resolve_window.title(f"Разрешение противоречия: {node_data.get('label', 'N/A')}")
            resolve_window.geometry("500x400")
            resolve_window.transient(self.gui.root)
            resolve_window.grab_set()
            
            # Информационная панель
            info_frame = ttk.LabelFrame(resolve_window, text="Информация о противоречии")
            info_frame.pack(fill=tk.X, padx=10, pady=10)
            
            ttk.Label(
                info_frame,
                text=f"Концепт 1: {node_data.get('concept1', 'N/A')}\n"
                     f"Концепт 2: {node_data.get('concept2', 'N/A')}\n"
                     f"Тип: {node_data.get('type', 'N/A')}\n"
                     f"Приоритет: {node_data.get('priority', 0):.0%}",
                justify=tk.LEFT
            ).pack(padx=10, pady=10)
            
            # Варианты разрешения
            options_frame = ttk.LabelFrame(resolve_window, text="Варианты разрешения")
            options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Переменная для хранения выбранного варианта
            resolution_var = tk.StringVar(value="accept_concept1")
            
            ttk.Radiobutton(
                options_frame,
                text=f"Принять концепт '{node_data.get('concept1', 'N/A')}'",
                variable=resolution_var,
                value="accept_concept1"
            ).pack(anchor=tk.W, padx=10, pady=5)
            
            ttk.Radiobutton(
                options_frame,
                text=f"Принять концепт '{node_data.get('concept2', 'N/A')}'",
                variable=resolution_var,
                value="accept_concept2"
            ).pack(anchor=tk.W, padx=10, pady=5)
            
            ttk.Radiobutton(
                options_frame,
                text="Создать новый концепт, объединяющий оба",
                variable=resolution_var,
                value="create_new_concept"
            ).pack(anchor=tk.W, padx=10, pady=5)
            
            ttk.Radiobutton(
                options_frame,
                text="Оставить как есть (контекстно-зависимое)",
                variable=resolution_var,
                value="context_dependent"
            ).pack(anchor=tk.W, padx=10, pady=5)
            
            # Обоснование
            ttk.Label(
                resolve_window,
                text="Обоснование:",
                font=('Segoe UI', 9, 'bold')
            ).pack(anchor=tk.W, padx=10)
            
            reason_text = scrolledtext.ScrolledText(
                resolve_window,
                height=5,
                wrap=tk.WORD,
                font=('Segoe UI', 9)
            )
            reason_text.pack(fill=tk.X, padx=10, pady=5)
            
            # Кнопки действий
            btn_frame = ttk.Frame(resolve_window)
            btn_frame.pack(fill=tk.X, padx=10, pady=10)
            
            def apply_resolution():
                resolution_type = resolution_var.get()
                reasoning = reason_text.get("1.0", tk.END).strip()
                
                if not reasoning:
                    messagebox.showwarning("Предупреждение", "Пожалуйста, укажите обоснование разрешения.")
                    return
                
                try:
                    # Вызываем метод разрешения из ядра
                    if self.gui.brain and hasattr(self.gui.brain, 'contradiction_resolver'):
                        self.gui.brain.contradiction_resolver.resolve_contradiction(
                            node_id,
                            resolution_type,
                            reasoning
                        )
                        
                        # Обновляем граф
                        self._refresh_graph()
                        
                        # Закрываем окно
                        resolve_window.destroy()
                        
                        messagebox.showinfo(
                            "Успех", 
                            "Противоречие успешно разрешено"
                        )
                    else:
                        raise ValueError("Модуль разрешения противоречий недоступен")
                except Exception as e:
                    logger.error(f"Ошибка разрешения противоречия: {e}", exc_info=True)
                    messagebox.showerror(
                        "Ошибка", 
                        f"Не удалось разрешить противоречие: {str(e)}"
                    )
            
            ttk.Button(
                btn_frame,
                text="Применить",
                command=apply_resolution
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                btn_frame,
                text="Отмена",
                command=resolve_window.destroy
            ).pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            logger.error(f"Ошибка разрешения противоречия: {e}", exc_info=True)
            messagebox.showerror(
                "Ошибка", 
                f"Не удалось запустить разрешение: {str(e)}"
            )
    
    def update_theme(self):
        """Обновляет тему графа знаний при смене темы интерфейса."""
        if not hasattr(self, 'graph_canvas') or not self.graph_canvas:
            return
            
        # Перерисовываем граф с учетом новой темы
        self._update_graph_visualization()
