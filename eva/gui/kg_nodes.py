# gui/kg_nodes.py
"""Node creation, editing, deletion, properties for KnowledgeGraphModule."""
import tkinter as tk
from tkinter import ttk, messagebox
import logging

logger = logging.getLogger("eva.gui.knowledge")


def show_node_info(self, node_id):
    """Shows node information from the real knowledge_graph."""
    self._clear_info_panel()

    try:
        node_data = None
        source = "local"

        if hasattr(self.gui, 'brain') and self.gui.brain:
            brain = self.gui.brain
            if hasattr(brain, 'knowledge_graph') and brain.knowledge_graph:
                kg = brain.knowledge_graph
                try:
                    node = kg.get_node(node_id)
                    if node:
                        node_data = {
                            'label': node.name,
                            'type': node.node_type,
                            'domain': node.domain,
                            'strength': node.strength,
                            'description': node.description,
                            'timestamp': getattr(node, 'timestamp', 0),
                            'last_updated': getattr(node, 'last_updated', 0)
                        }
                        source = "knowledge_graph"
                except Exception as e:
                    logger.debug(f"Не удалось получить узел из knowledge_graph: {e}")

        if not node_data and self.graph and node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]

        if not node_data:
            self._show_error_in_info_panel(f"Узел {node_id} не найден")
            return

        info_frame = ttk.Frame(self.info_content)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(
            info_frame,
            text=f"Источник: {source}",
            font=('Segoe UI', 8, 'italic'),
            foreground="gray"
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        row = 1

        if node_data.get('label'):
            ttk.Label(info_frame, text="Название:", font=('Segoe UI', 9, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(info_frame, text=node_data.get('label', 'N/A'), font=('Segoe UI', 9)).grid(row=row, column=1, sticky=tk.W, pady=2)
            row += 1

        if node_data.get('type'):
            ttk.Label(info_frame, text="Тип:", font=('Segoe UI', 9, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(info_frame, text=node_data.get('type', 'N/A'), font=('Segoe UI', 9)).grid(row=row, column=1, sticky=tk.W, pady=2)
            row += 1

        if node_data.get('domain'):
            ttk.Label(info_frame, text="Домен:", font=('Segoe UI', 9, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(info_frame, text=node_data.get('domain', 'N/A'), font=('Segoe UI', 9)).grid(row=row, column=1, sticky=tk.W, pady=2)
            row += 1

        if node_data.get('strength') is not None:
            ttk.Label(info_frame, text="Сила:", font=('Segoe UI', 9, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(info_frame, text=f"{node_data.get('strength', 0):.2f}", font=('Segoe UI', 9)).grid(row=row, column=1, sticky=tk.W, pady=2)
            row += 1

        if node_data.get('description'):
            ttk.Label(info_frame, text="Описание:", font=('Segoe UI', 9, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            row += 1
            desc_label = ttk.Label(info_frame, text=node_data.get('description', 'N/A')[:200], font=('Segoe UI', 9), wraplength=400)
            desc_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)

        logger.debug(f"Показана информация об узле {node_id} из {source}")

    except Exception as e:
        logger.error(f"Ошибка отображения информации об узле: {e}")
        self._show_error_in_info_panel(f"Ошибка: {str(e)}")


def show_edge_info(self, edge):
    """Shows edge information."""
    self._clear_info_panel()

    try:
        edge_data = self.graph.edges[edge]
        source, target = edge

        info_frame = ttk.Frame(self.info_content)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

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


def clear_info_panel(self):
    """Clears the info panel."""
    for widget in self.info_content.winfo_children():
        widget.destroy()


def show_error_in_info_panel(self, message):
    """Shows error message in info panel."""
    self._clear_info_panel()

    ttk.Label(
        self.info_content,
        text=message,
        foreground="red",
        font=('Segoe UI', 9, 'italic'),
        wraplength=800
    ).pack(fill=tk.X, padx=5, pady=5)


def show_node_details(self, node_id=None):
    """Shows node details."""
    if not node_id:
        if not self.selected_node:
            return
        node_id = self.selected_node

    try:
        if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
            raise ValueError("Граф знаний недоступен")

        node = self.gui.brain.knowledge_graph.get_node(node_id)
        if not node:
            raise ValueError("Узел не найден")

        details_window = tk.Toplevel(self.gui.root)
        details_window.title(f"Детали: {node['name']}")
        details_window.geometry("600x500")
        details_window.transient(self.gui.root)
        details_window.grab_set()

        notebook = ttk.Notebook(details_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        info_frame = ttk.Frame(notebook)
        notebook.add(info_frame, text="Основная информация")

        info_text = (
            f"Название: {node.get('name', 'N/A')}\n"
            f"Тип: {node.get('node_type', 'N/A')}\n"
            f"Домен: {node.get('domain', 'N/A')}\n"
            f"Сила: {node.get('strength', 0.0):.2f}\n\n"
            f"Описание: {node.get('description', 'N/A')}\n\n"
        )

        info_label = ttk.Label(
            info_frame,
            text=info_text,
            justify=tk.LEFT,
            wraplength=550
        )
        info_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        edges_frame = ttk.Frame(notebook)
        notebook.add(edges_frame, text="Связи")

        columns = ("source", "relation", "target", "strength")
        edges_tree = ttk.Treeview(
            edges_frame,
            columns=columns,
            show="headings",
            selectmode="browse"
        )

        edges_tree.heading("source", text="Источник")
        edges_tree.heading("relation", text="Связь")
        edges_tree.heading("target", text="Цель")
        edges_tree.heading("strength", text="Сила")

        edges_tree.column("source", width=150, anchor=tk.W)
        edges_tree.column("relation", width=100, anchor=tk.W)
        edges_tree.column("target", width=150, anchor=tk.W)
        edges_tree.column("strength", width=80, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(edges_frame, orient=tk.VERTICAL, command=edges_tree.yview)
        edges_tree.configure(yscrollcommand=scrollbar.set)

        edges_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        if self.gui.brain and hasattr(self.gui.brain, 'knowledge_graph'):
            edges = self.gui.brain.knowledge_graph.get_edges(node_id)
            for edge in edges:
                source = edge["source_id"]
                target = edge["target_id"]

                source_node = self.gui.brain.knowledge_graph.get_node(source)
                target_node = self.gui.brain.knowledge_graph.get_node(target)

                source_name = source_node["name"] if source_node else "N/A"
                target_name = target_node["name"] if target_node else "N/A"

                edges_tree.insert("", tk.END, values=(
                    source_name,
                    edge["relation_type"],
                    target_name,
                    f"{edge['strength']:.2f}"
                ))

        history_frame = ttk.Frame(notebook)
        notebook.add(history_frame, text="История")

        from tkinter import scrolledtext
        history_text = scrolledtext.ScrolledText(
            history_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 9),
            state=tk.DISABLED
        )
        history_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        if self.gui.brain and hasattr(self.gui.brain, 'knowledge_graph'):
            node = self.gui.brain.knowledge_graph.get_node(node_id)
            if node:
                history = node.history if hasattr(node, 'history') else []
            else:
                history = []
            for entry in history:
                from datetime import datetime
                timestamp = datetime.fromtimestamp(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                history_text.config(state=tk.NORMAL)
                history_text.insert(tk.END, f"[{timestamp}] {entry['action']}\n")
                history_text.insert(tk.END, f"   {entry['details']}\n\n")
                history_text.config(state=tk.DISABLED)

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


def show_related_nodes(self, node_id=None):
    """Shows related nodes."""
    if not node_id:
        if not self.selected_node:
            return
        node_id = self.selected_node

    try:
        if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
            raise ValueError("Граф знаний недоступен")

        edges = self.gui.brain.knowledge_graph.get_edges(node_id)

        node_ids = [node_id]
        for edge in edges:
            node_ids.append(edge["source_id"])
            node_ids.append(edge["target_id"])

        subgraph_window = tk.Toplevel(self.gui.root)
        subgraph_window.title(f"Связанные узлы: {node_id}")
        subgraph_window.geometry("800x600")
        subgraph_window.transient(self.gui.root)
        subgraph_window.grab_set()

        graph_frame = ttk.Frame(subgraph_window)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        import networkx as nx
        G = nx.DiGraph()

        for nid in set(node_ids):
            node = self.gui.brain.knowledge_graph.get_node(nid)
            if node:
                G.add_node(
                    nid,
                    label=node["name"],
                    type=node["node_type"],
                    domain=node["domain"],
                    strength=node["strength"]
                )

        for edge in edges:
            G.add_edge(
                edge["source_id"],
                edge["target_id"],
                label=edge["relation_type"],
                strength=edge["strength"]
            )

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Связанные узлы', fontsize=12)
        ax.axis('off')

        pos = nx.spring_layout(G)

        theme = getattr(self.gui, 'theme', 'light') if hasattr(self, 'gui') and self.gui else 'light'
        bg_color = self.color_scheme["background"]["dark"] if theme == "dark" else self.color_scheme["background"]["light"]
        text_color = "white" if theme == "dark" else "black"
        edge_color = "#a0a0a0" if theme == "dark" else "#666666"

        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        node_colors = []
        node_sizes = []

        for node in G.nodes():
            node_data = G.nodes[node]
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

        for edge in G.edges():
            edge_data = G.edges[edge]
            strength = edge_data.get("strength", 0.5)

            edge_color = self.color_scheme["relation"]["edge"]
            edge_width = self.edge_width

            edge_width = edge_width * (0.5 + strength)

            edge_colors.append(edge_color)
            edge_widths.append(edge_width)

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )

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

        labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            font_size=self.font_size,
            font_color=text_color,
            ax=ax
        )

        edge_labels = {edge: G.edges[edge].get("label", "") for edge in G.edges()}
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=max(6, self.font_size - 2),
            font_color=edge_color,
            ax=ax
        )

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, graph_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

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


def resolve_contradiction(self, node_id):
    """Launches contradiction resolution process."""
    try:
        node_data = self.graph.nodes[node_id]
        if node_data.get("type", "") != "contradiction":
            messagebox.showinfo("Информация", "Этот узел не является противоречием")
            return

        resolve_window = tk.Toplevel(self.gui.root)
        resolve_window.title(f"Разрешение противоречия: {node_data.get('label', 'N/A')}")
        resolve_window.geometry("500x400")
        resolve_window.transient(self.gui.root)
        resolve_window.grab_set()

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

        options_frame = ttk.LabelFrame(resolve_window, text="Варианты разрешения")
        options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

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

        ttk.Label(
            resolve_window,
            text="Обоснование:",
            font=('Segoe UI', 9, 'bold')
        ).pack(anchor=tk.W, padx=10)

        from tkinter import scrolledtext
        reason_text = scrolledtext.ScrolledText(
            resolve_window,
            height=5,
            wrap=tk.WORD,
            font=('Segoe UI', 9)
        )
        reason_text.pack(fill=tk.X, padx=10, pady=5)

        btn_frame = ttk.Frame(resolve_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        def apply_resolution():
            resolution_type = resolution_var.get()
            reasoning = reason_text.get("1.0", tk.END).strip()

            if not reasoning:
                messagebox.showwarning("Предупреждение", "Пожалуйста, укажите обоснование разрешения.")
                return

            try:
                if self.gui.brain and hasattr(self.gui.brain, 'contradiction_resolver'):
                    self.gui.brain.contradiction_resolver.resolve_contradiction(
                        node_id,
                        resolution_type,
                        reasoning
                    )

                    self._refresh_graph()

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
