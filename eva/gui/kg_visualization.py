# gui/kg_visualization.py
"""Graph visualization, D3 rendering, node/edge display for KnowledgeGraphModule."""
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import Menu
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

logger = logging.getLogger("eva.gui.knowledge")


def create_graph_tab(self, parent):
    """Creates the graph visualization tab."""
    graph_frame = ttk.Frame(parent)
    graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Граф знаний', fontsize=12)
    ax.axis('off')

    self.graph_canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    self.graph_canvas.draw()
    self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(self.graph_canvas, graph_frame)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    self.graph_canvas.mpl_connect('button_press_event', self._on_graph_click)
    self.graph_canvas.mpl_connect('motion_notify_event', self._on_graph_hover)
    self.graph_canvas.mpl_connect('scroll_event', self._on_graph_scroll)

    _create_graph_context_menu(self)
    _create_info_panel(self, parent)


def _create_graph_context_menu(self):
    """Creates context menu for the graph."""
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
    self.graph_canvas.mpl_connect('button_press_event', self._show_graph_context_menu)


def _show_graph_context_menu(self, event):
    """Shows context menu for the graph."""
    if event.button == 3:
        self.graph_context_menu.tk_popup(event.x * 64, self.gui.root.winfo_height() - event.y * 64)


def _create_info_panel(self, parent):
    """Creates info panel with current item information."""
    info_frame = ttk.LabelFrame(parent, text="Информация о выбранном элементе")
    info_frame.pack(fill=tk.X, padx=5, pady=5)

    self.info_content = ttk.Frame(info_frame)
    self.info_content.pack(fill=tk.X, padx=10, pady=10)

    ttk.Label(
        self.info_content,
        text="Выберите узел или связь на графике для просмотра подробной информации",
        font=('Segoe UI', 9, 'italic'),
        wraplength=800
    ).pack(fill=tk.X, padx=5, pady=5)


def _create_node_types_chart(self, parent):
    """Creates node types pie chart."""
    ttk.Label(parent, text="Типы узлов", font=('Segoe UI', 10, 'bold')).pack(pady=(5, 0))

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.set_title('Распределение типов узлов', fontsize=10)

    self.node_types_pie = ax.pie([1], labels=["Загрузка..."], startangle=90)

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    self.canvas_objects = {
        'node_types': {
            'fig': fig,
            'ax': ax,
            'canvas': canvas,
            'pie': self.node_types_pie
        }
    }


def _create_domains_chart(self, parent):
    """Creates domains bar chart."""
    ttk.Label(parent, text="Домены", font=('Segoe UI', 10, 'bold')).pack(pady=(5, 0))

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.set_title('Распределение по доменам', fontsize=10)

    self.domains_bar = ax.bar(["Загрузка..."], [1])

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    self.canvas_objects['domains'] = {
        'fig': fig,
        'ax': ax,
        'canvas': canvas,
        'bar': self.domains_bar
    }


def initialize_graph(self):
    """Initializes the knowledge graph with sample data."""
    try:
        self.graph = nx.DiGraph()

        self.graph.add_node("node1", label="Когнитивные системы", type="concept", domain="AI", strength=0.8)
        self.graph.add_node("node2", label="Этический анализ", type="fact", domain="Ethics", strength=0.7)
        self.graph.add_node("node3", label="Адаптация", type="concept", domain="AI", strength=0.6)

        self.graph.add_edge("node1", "node2", label="включает", strength=0.9)
        self.graph.add_edge("node1", "node3", label="требует", strength=0.8)

        self.pos = nx.spring_layout(self.graph)

        self._update_graph_visualization()

    except Exception as e:
        logger.error(f"Ошибка инициализации графа: {e}", exc_info=True)
        messagebox.showerror(
            "Ошибка",
            f"Не удалось инициализировать граф знаний: {str(e)}"
        )


def update_graph_visualization(self):
    """Updates the knowledge graph visualization."""
    if not self.graph or not self.graph_canvas:
        return

    try:
        fig = self.graph_canvas.figure

        if hasattr(fig, 'clear') and hasattr(self, 'graph_canvas'):
            try:
                toolbar = None
                if hasattr(self.graph_canvas, 'get_tk_widget'):
                    toolbar = self.graph_canvas.manager.toolbar if hasattr(self.graph_canvas.manager, 'toolbar') else None

                fig.clear()

                if toolbar is not None and hasattr(toolbar, 'update'):
                    try:
                        toolbar.update()
                    except Exception:
                        pass

            except Exception as clear_error:
                fig = plt.Figure(figsize=(8, 6), dpi=100)
                self.graph_canvas.figure = fig
        else:
            fig = plt.Figure(figsize=(8, 6), dpi=100)
            self.graph_canvas.figure = fig

        ax = fig.add_subplot(111)
        ax.set_title('Граф знаний', fontsize=12)
        ax.axis('off')

        theme = getattr(self.gui, 'theme', 'light') if hasattr(self, 'gui') and self.gui else 'light'
        bg_color = self.color_scheme["background"]["dark"] if theme == "dark" else self.color_scheme["background"]["light"]
        text_color = "white" if theme == "dark" else "black"
        edge_color = "#a0a0a0" if theme == "dark" else "#666666"

        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        self.pos = nx.spring_layout(self.graph, k=0.5, iterations=50, seed=42)

        node_colors = []
        node_sizes = []

        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_type = node_data.get("type", "fact")
            strength = node_data.get("strength", 0.5)

            if node_type in self.color_scheme and "node" in self.color_scheme[node_type]:
                node_color = self.color_scheme[node_type]["node"]
            else:
                node_color = self.color_scheme["fact"]["node"]

            node_size = self.node_size * (0.5 + strength)

            node_colors.append(node_color)
            node_sizes.append(node_size)

        edge_colors = []
        edge_widths = []

        for edge in self.graph.edges():
            edge_data = self.graph.edges[edge]
            strength = edge_data.get("strength", 0.5)
            relation_type = edge_data.get("label", "relation")

            if relation_type == "contradicts":
                edge_color = self.color_scheme["contradiction"]["edge"]
                edge_width = self.color_scheme["contradiction"].get("width", self.edge_width * 1.5)
            elif relation_type in self.color_scheme and "edge" in self.color_scheme[relation_type]:
                edge_color = self.color_scheme[relation_type]["edge"]
                edge_width = self.edge_width
            else:
                edge_color = self.color_scheme["relation"]["edge"]
                edge_width = self.edge_width

            edge_width = edge_width * (0.5 + strength)

            edge_colors.append(edge_color)
            edge_widths.append(edge_width)

        nx.draw_networkx_nodes(
            self.graph,
            self.pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )

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

        labels = {node: self.graph.nodes[node].get("label", node) for node in self.graph.nodes()}
        nx.draw_networkx_labels(
            self.graph,
            self.pos,
            labels=labels,
            font_size=self.font_size,
            font_color=text_color,
            ax=ax
        )

        edge_labels = {edge: self.graph.edges[edge].get("label", "") for edge in self.graph.edges()}
        nx.draw_networkx_edge_labels(
            self.graph,
            self.pos,
            edge_labels=edge_labels,
            font_size=max(6, self.font_size - 2),
            font_color=edge_color,
            ax=ax
        )

        try:
            if hasattr(self, 'graph_canvas') and self.graph_canvas is not None:
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


def on_graph_click(self, event):
    """Handles click on the graph."""
    if not self.graph or not self.pos:
        return

    for node, (x, y) in self.pos.items():
        x_fig, y_fig = self.graph_canvas.figure.transFigure.inverted().transform((event.x, event.y))
        x_node, y_node = self.graph_canvas.figure.transFigure.transform((x, y))

        distance = np.sqrt((x_fig - x_node)**2 + (y_fig - y_node)**2)
        if distance < 0.02:
            self.selected_node = node
            self.selected_edge = None
            self._show_node_info(node)
            return

    for edge in self.graph.edges():
        source, target = edge
        x1, y1 = self.pos[source]
        x2, y2 = self.pos[target]

        x1_fig, y1_fig = self.graph_canvas.figure.transFigure.transform((x1, y1))
        x2_fig, y2_fig = self.graph_canvas.figure.transFigure.transform((x2, y2))
        x_fig, y_fig = self.graph_canvas.figure.transFigure.inverted().transform((event.x, event.y))

        A = y2_fig - y1_fig
        B = x1_fig - x2_fig
        C = x2_fig*y1_fig - x1_fig*y2_fig
        distance = abs(A*x_fig + B*y_fig + C) / np.sqrt(A**2 + B**2)

        if distance < 0.02:
            self.selected_edge = edge
            self.selected_node = None
            self._show_edge_info(edge)
            return

    self.selected_node = None
    self.selected_edge = None
    self._clear_info_panel()


def on_graph_hover(self, event):
    """Handles hover over the graph."""
    if not self.graph or not self.pos or event.inaxes is None:
        return

    for node, (x, y) in self.pos.items():
        x_fig, y_fig = self.graph_canvas.figure.transFigure.inverted().transform((event.x, event.y))
        x_node, y_node = self.graph_canvas.figure.transFigure.transform((x, y))

        distance = np.sqrt((x_fig - x_node)**2 + (y_fig - y_node)**2)
        if distance < 0.02:
            self.gui.root.config(cursor="hand2")
            return

    for edge in self.graph.edges():
        source, target = edge
        x1, y1 = self.pos[source]
        x2, y2 = self.pos[target]

        x1_fig, y1_fig = self.graph_canvas.figure.transFigure.transform((x1, y1))
        x2_fig, y2_fig = self.graph_canvas.figure.transFigure.transform((x2, y2))
        x_fig, y_fig = self.graph_canvas.figure.transFigure.inverted().transform((event.x, event.y))

        A = y2_fig - y1_fig
        B = x1_fig - x2_fig
        C = x2_fig*y1_fig - x1_fig*y2_fig
        distance = abs(A*x_fig + B*y_fig + C) / np.sqrt(A**2 + B**2)

        if distance < 0.02:
            self.gui.root.config(cursor="hand2")
            return

    self.gui.root.config(cursor="")


def on_graph_scroll(self, event):
    """Handles mouse wheel scroll on the graph."""
    if not self.graph or not self.pos:
        return

    scale_factor = 1.1 if event.button == 'up' else 0.9

    for node in self.pos:
        x, y = self.pos[node]
        self.pos[node] = (x * scale_factor, y * scale_factor)

    self._update_graph_visualization()


def center_graph_on_node(self, node_id):
    """Centers the graph on the specified node."""
    if not self.graph or not self.pos or node_id not in self.pos:
        return

    x, y = self.pos[node_id]

    for node in self.pos:
        self.pos[node] = (self.pos[node][0] - x, self.pos[node][1] - y)

    self._update_graph_visualization()


def update_theme(self):
    """Updates graph theme when interface theme changes."""
    if not hasattr(self, 'graph_canvas') or not self.graph_canvas:
        return

    self._update_graph_visualization()
