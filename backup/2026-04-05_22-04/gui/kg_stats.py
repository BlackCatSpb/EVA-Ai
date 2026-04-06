# gui/kg_stats.py
"""Statistics, analytics, metrics for KnowledgeGraphModule."""
import tkinter as tk
from tkinter import ttk, messagebox
import logging
import time
from datetime import datetime

logger = logging.getLogger("eva.gui.knowledge")


def create_domains_tab(self, parent):
    """Creates the domains tab."""
    domains_frame = ttk.Frame(parent)
    domains_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    columns = ("domain", "nodes", "edges", "last_updated")
    self.domains_tree = ttk.Treeview(
        domains_frame,
        columns=columns,
        show="headings",
        selectmode="browse"
    )

    self.domains_tree.heading("domain", text="Домен")
    self.domains_tree.heading("nodes", text="Узлы")
    self.domains_tree.heading("edges", text="Связи")
    self.domains_tree.heading("last_updated", text="Последнее обновление")

    self.domains_tree.column("domain", width=200, anchor=tk.W)
    self.domains_tree.column("nodes", width=100, anchor=tk.CENTER)
    self.domains_tree.column("edges", width=100, anchor=tk.CENTER)
    self.domains_tree.column("last_updated", width=150, anchor=tk.W)

    scrollbar = ttk.Scrollbar(domains_frame, orient=tk.VERTICAL, command=self.domains_tree.yview)
    self.domains_tree.configure(yscrollcommand=scrollbar.set)

    self.domains_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    _create_domains_context_menu(self)

    self._load_domains_data()

    self.domains_tree.bind("<Double-1>", self._on_domain_double_click)


def _create_domains_context_menu(self):
    """Creates context menu for domains."""
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
    self.domains_tree.bind("<Button-3>", self._show_domains_context_menu)


def _show_domains_context_menu(self, event):
    """Shows context menu for domains."""
    item = self.domains_tree.identify_row(event.y)
    if item:
        self.domains_tree.selection_set(item)
        self.domains_context_menu.tk_popup(event.x_root, event.y_root)


def load_domains_data(self):
    """Loads domain data."""
    try:
        for item in self.domains_tree.get_children():
            self.domains_tree.delete(item)

        if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
            logger.warning("Граф знаний недоступен для загрузки доменов")
            return

        kg = self.gui.brain.knowledge_graph
        domain_stats = {}
        try:
            domain_stats = kg.get_domain_statistics() or {}
        except Exception as e:
            logger.warning(f"Не удалось получить статистику доменов из ядра: {e}")
            domain_stats = {}

        if not isinstance(domain_stats, dict) or not domain_stats:
            domain_stats = {}
            nodes = kg.get_all_nodes()
            edges = kg.get_all_edges() if hasattr(kg, 'get_all_edges') else []
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

        overall_ts = None
        try:
            overall = kg.get_statistics() if hasattr(kg, 'get_statistics') else None
            if isinstance(overall, dict):
                overall_ts = overall.get("last_update") or overall.get("last_updated")
        except Exception:
            pass

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


def create_stats_tab(self, parent):
    """Creates the statistics tab."""
    stats_frame = ttk.Frame(parent)
    stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    general_frame = ttk.LabelFrame(stats_frame, text="Общая статистика")
    general_frame.pack(fill=tk.X, padx=5, pady=5)

    metrics_frame = ttk.Frame(general_frame)
    metrics_frame.pack(fill=tk.X, padx=10, pady=10)

    nodes_frame = ttk.Frame(metrics_frame)
    nodes_frame.pack(side=tk.LEFT, padx=20)
    ttk.Label(nodes_frame, text="Узлы:", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)
    self.nodes_value = ttk.Label(nodes_frame, text="0", font=('Segoe UI', 9))
    self.nodes_value.pack(anchor=tk.W)

    edges_frame = ttk.Frame(metrics_frame)
    edges_frame.pack(side=tk.LEFT, padx=20)
    ttk.Label(edges_frame, text="Связи:", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)
    self.edges_value = ttk.Label(edges_frame, text="0", font=('Segoe UI', 9))
    self.edges_value.pack(anchor=tk.W)

    domains_frame = ttk.Frame(metrics_frame)
    domains_frame.pack(side=tk.LEFT, padx=20)
    ttk.Label(domains_frame, text="Домены:", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)
    self.domains_value = ttk.Label(domains_frame, text="0", font=('Segoe UI', 9))
    self.domains_value.pack(anchor=tk.W)

    contradictions_frame = ttk.Frame(metrics_frame)
    contradictions_frame.pack(side=tk.LEFT, padx=20)
    ttk.Label(contradictions_frame, text="Противоречия:", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)
    self.contradictions_value = ttk.Label(contradictions_frame, text="0", font=('Segoe UI', 9))
    self.contradictions_value.pack(anchor=tk.W)

    charts_frame = ttk.LabelFrame(stats_frame, text="Графики статистики")
    charts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    charts_container = ttk.Frame(charts_frame)
    charts_container.pack(fill=tk.BOTH, expand=True)

    node_types_frame = ttk.Frame(charts_container, style='Card.TFrame')
    node_types_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    self._create_node_types_chart(node_types_frame)

    domains_chart_frame = ttk.Frame(charts_container, style='Card.TFrame')
    domains_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    self._create_domains_chart(domains_chart_frame)

    self._load_statistics()


def load_statistics(self):
    """Loads knowledge graph statistics."""
    try:
        if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
            logger.warning("Граф знаний недоступен для загрузки статистики")
            return

        kg = self.gui.brain.knowledge_graph
        stats = {}
        try:
            raw_stats = kg.get_statistics()
            if isinstance(raw_stats, dict):
                stats = raw_stats
        except Exception as e:
            logger.warning(f"Не удалось получить общую статистику из ядра: {e}")
            stats = {}
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

        try:
            domain_stats = kg.get_domain_statistics()
            domain_counts = {}
            for domain, dstats in (domain_stats or {}).items():
                if isinstance(dstats, dict):
                    count = dstats.get("nodes") or dstats.get("total_nodes") or dstats.get("count") or 0
                else:
                    count = int(dstats) if isinstance(dstats, (int, float)) else 0
                domain_counts[domain] = count
            stats["domains"] = domain_counts
        except Exception as e2:
            logger.warning(f"Не удалось получить детальную статистику доменов: {e2}")
            try:
                domain_counts = {}
                for node in kg.get_all_nodes():
                    domain = node.get("domain") if isinstance(node, dict) else getattr(node, "domain", "general")
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                stats["domains"] = domain_counts
            except Exception:
                stats["domains"] = {}

        self.nodes_value.config(text=str(stats.get("total_nodes", 0)))
        self.edges_value.config(text=str(stats.get("total_edges", 0)))
        domains_obj = stats.get("domains", {})
        domains_count = len(domains_obj) if isinstance(domains_obj, dict) else len(domains_obj) if isinstance(domains_obj, list) else 0
        self.domains_value.config(text=str(domains_count))
        self.contradictions_value.config(text=str(stats.get("contradictions", 0)))

        self._update_statistics_charts(stats)

    except Exception as e:
        logger.error(f"Ошибка загрузки статистики: {e}", exc_info=True)
        messagebox.showerror(
            "Ошибка",
            f"Не удалось загрузить статистику: {str(e)}"
        )


def update_statistics_charts(self, stats):
    """Updates statistics charts."""
    try:
        if 'node_types' in self.canvas_objects:
            ax = self.canvas_objects['node_types']['ax']
            canvas = self.canvas_objects['node_types']['canvas']

            ax.clear()

            node_types_dict = stats.get("node_types") or {}
            node_types = list(node_types_dict.keys())
            counts = list(node_types_dict.values())

            if sum(counts) > 0:
                wedges, texts, autotexts = ax.pie(
                    counts,
                    labels=node_types,
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax.axis('equal')
                import matplotlib.pyplot as plt
                plt.setp(autotexts, size=8, weight="bold")
            else:
                ax.text(0.5, 0.5, "Нет данных",
                       horizontalalignment='center',
                       verticalalignment='center')

            canvas.draw()

        if 'domains' in self.canvas_objects:
            ax = self.canvas_objects['domains']['ax']
            canvas = self.canvas_objects['domains']['canvas']

            ax.clear()

            domains_obj = stats.get("domains") or {}
            if isinstance(domains_obj, dict):
                domains = list(domains_obj.keys())
                counts = list(domains_obj.values())
            elif isinstance(domains_obj, list):
                domains = domains_obj
                counts = [1 for _ in domains_obj]
            else:
                domains = []
                counts = []

            if len(domains) > 0:
                ax.bar(range(len(domains)), counts)
                ax.set_xticks(range(len(domains)))
                ax.set_xticklabels(domains, rotation=45, ha='right')
                ax.set_ylabel('Количество')
            else:
                ax.text(0.5, 0.5, "Нет данных",
                       horizontalalignment='center',
                       verticalalignment='center')

            canvas.draw()

    except Exception as e:
        logger.error(f"Ошибка обновления графиков статистики: {e}", exc_info=True)
