# gui/knowledge_graph_module.py
"""Модуль графа знаний для ЕВА GUI - полнофункциональная реализация"""
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
from eva.knowledge.knowledge_graph import KnowledgeGraph

from eva.gui.kg_visualization import (
    create_graph_tab,
    _create_node_types_chart,
    _create_domains_chart,
    initialize_graph,
    update_graph_visualization,
    on_graph_click,
    on_graph_hover,
    on_graph_scroll,
    center_graph_on_node,
    update_theme,
)
from eva.gui.kg_search import (
    create_search_tab,
    show_search_dialog,
    perform_search,
    update_search_results,
    on_search_result_double_click,
    highlight_search_result,
    show_search_result_details,
    export_search_result,
)
from eva.gui.kg_stats import (
    create_domains_tab,
    load_domains_data,
    create_stats_tab,
    load_statistics,
    update_statistics_charts,
)
from eva.gui.kg_nodes import (
    show_node_info,
    show_edge_info,
    clear_info_panel,
    show_error_in_info_panel,
    show_node_details,
    show_related_nodes,
    resolve_contradiction,
)
from eva.gui.kg_actions import (
    refresh_graph,
    search_by_domain,
    export_node,
    export_graph,
    show_settings_dialog,
    on_tab_changed,
    on_domain_double_click,
    show_domain_on_graph,
    show_domain_details,
    export_domain,
)

logger = logging.getLogger("eva.gui.knowledge")

class KnowledgeGraphModule:
    """Модуль для визуализации и работы с графом знаний."""
    
    def __init__(self, gui):
        self.gui = gui
        
        self.knowledge_frame = None
        self.notebook = None
        self.graph_canvas = None
        self.graph = None
        self.pos = None
        self.selected_node = None
        self.selected_edge = None
        self.current_view = "all"
        self.search_query = ""
        
        self.update_thread = None
        self.update_running = False
        
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
        
        self.pending_after_events = []
        
        self.search_results = []
        self.search_index = 0
        self.update_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.update_interval = 5000
        self.pending_after_ids = []
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
        
        if not self._log_brain_access("knowledge_graph"):
            logger.error("Компонент knowledge_graph недоступен")
            return
        
        for widget in self.gui.content_area.winfo_children():
            widget.destroy()
            
        self._create_knowledge_interface()
        
        self._start_update_thread()
        
        logger.info("Модуль графа знаний активирован")

    def deactivate(self):
        """Деактивирует модуль графа знаний."""
        self._stop_update_thread()
        
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
                    pass
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
            return None

    def _update_data_loop(self):
        """Цикл обновления данных графа знаний."""
        while not self.stop_event.is_set():
            try:
                if self.current_view == "all" or self.current_view == "search":
                    self._update_graph_data()
                
                self._safe_after(100, self._process_update_queue)
                
                time.sleep(self.update_interval / 1000.0)
                
            except Exception as e:
                logger.error(f"Ошибка в цикле обновления данных графа: {e}", exc_info=True)
                time.sleep(5.0)

    def _update_graph_data(self):
        """Обновляет данные графа знаний из системы."""
        try:
            if not self._log_brain_access('knowledge_graph'):
                logger.warning("Граф знаний недоступен для загрузки доменов")
                return
            
            knowledge_graph = self.gui.brain.knowledge_graph
            
            if self.current_view == "all":
                nodes = knowledge_graph.get_all_nodes()
                edges = knowledge_graph.get_all_edges()
            elif self.current_view == "search" and self.search_query:
                nodes = knowledge_graph.search_nodes(self.search_query)
                node_ids = [getattr(node, "id", None) if not isinstance(node, dict) else node.get("id") for node in nodes]
                node_ids = [nid for nid in node_ids if nid]
                edges = []
                for node_id in node_ids:
                    node_edges = knowledge_graph.get_edges(node_id)
                    edges.extend(node_edges)
                
                related_node_ids = set()
                for edge in edges:
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
                
                related_nodes = []
                for node_id in related_node_ids:
                    if node_id not in node_ids:
                        node = knowledge_graph.get_node(node_id)
                        if node:
                            related_nodes.append(node)
                
                nodes.extend(related_nodes)
            elif self.current_view.startswith("domain:"):
                domain = self.current_view.split(":", 1)[1]
                nodes = knowledge_graph.get_nodes_by_domain(domain)
                node_ids = [getattr(node, "id", None) if not isinstance(node, dict) else node.get("id") for node in nodes]
                node_ids = [nid for nid in node_ids if nid]
                edges = []
                for node_id in node_ids:
                    node_edges = knowledge_graph.get_edges(node_id)
                    edges.extend(node_edges)
            else:
                return
            
            G = nx.DiGraph()
            
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
            while not self.update_queue.empty():
                data = self.update_queue.get_nowait()
                
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
        
        header_frame = ttk.Frame(self.knowledge_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            header_frame, 
            text="Граф знаний", 
            font=('Segoe UI', 16, 'bold')
        ).pack(side=tk.LEFT)
        
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
        
        self.notebook = ttk.Notebook(self.knowledge_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        graph_tab = ttk.Frame(self.notebook)
        self.notebook.add(graph_tab, text="Визуализация")
        self._create_graph_tab(graph_tab)
        
        search_tab = ttk.Frame(self.notebook)
        self.notebook.add(search_tab, text="Поиск")
        self._create_search_tab(search_tab)
        
        domains_tab = ttk.Frame(self.notebook)
        self.notebook.add(domains_tab, text="Домены")
        self._create_domains_tab(domains_tab)
        
        stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(stats_tab, text="Статистика")
        self._create_stats_tab(stats_tab)
        
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        
        self._initialize_graph()

    _create_graph_tab = create_graph_tab
    _create_search_tab = create_search_tab
    _create_domains_tab = create_domains_tab
    _create_stats_tab = create_stats_tab
    _create_node_types_chart = _create_node_types_chart
    _create_domains_chart = _create_domains_chart
    _initialize_graph = initialize_graph
    _update_graph_visualization = update_graph_visualization
    _on_graph_click = on_graph_click
    _on_graph_hover = on_graph_hover
    _on_graph_scroll = on_graph_scroll
    _show_node_info = show_node_info
    _show_edge_info = show_edge_info
    _clear_info_panel = clear_info_panel
    _show_error_in_info_panel = show_error_in_info_panel
    _refresh_graph = refresh_graph
    _show_search_dialog = show_search_dialog
    _perform_search = perform_search
    _update_search_results = update_search_results
    _on_search_result_double_click = on_search_result_double_click
    _highlight_search_result = highlight_search_result
    _show_search_result_details = show_search_result_details
    _export_search_result = export_search_result
    _load_domains_data = load_domains_data
    _load_statistics = load_statistics
    _update_statistics_charts = update_statistics_charts
    _show_node_details = show_node_details
    _show_related_nodes = show_related_nodes
    _search_by_domain = search_by_domain
    _export_node = export_node
    _export_graph = export_graph
    _show_settings_dialog = show_settings_dialog
    _on_tab_changed = on_tab_changed
    _on_domain_double_click = on_domain_double_click
    _show_domain_on_graph = show_domain_on_graph
    _show_domain_details = show_domain_details
    _export_domain = export_domain
    _center_graph_on_node = center_graph_on_node
    _resolve_contradiction = resolve_contradiction
    update_theme = update_theme
