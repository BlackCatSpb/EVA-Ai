# gui/kg_actions.py
"""Actions, event handlers, UI interactions for KnowledgeGraphModule."""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import time
import json
import os

logger = logging.getLogger("eva_ai.gui.knowledge")


def refresh_graph(self):
    """Refreshes the knowledge graph."""
    self.current_view = "all"
    self._update_graph_data()


def search_by_domain(self):
    """Searches nodes by domain."""
    if not self.selected_node:
        return

    try:
        node_data = self.graph.nodes[self.selected_node]
        domain = node_data.get("domain", "")

        if not domain:
            messagebox.showinfo("Информация", "У этого узла не указан домен")
            return

        self.current_view = f"domain:{domain}"

        self._update_graph_data()

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


def export_node(self, node_id=None):
    """Exports node to file."""
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

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")],
            title="Экспорт узла"
        )

        if not file_path:
            return

        edges = self.gui.brain.knowledge_graph.get_edges(node_id)
        related_nodes = []
        for edge in edges:
            source_id = edge["source_id"]
            target_id = edge["target_id"]
            if source_id != node_id:
                related_nodes.append(source_id)
            if target_id != node_id:
                related_nodes.append(target_id)

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


def export_graph(self):
    """Exports knowledge graph to file."""
    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")],
            title="Экспорт графа знаний"
        )

        if not file_path:
            return

        if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
            raise ValueError("Граф знаний недоступен")

        nodes = self.gui.brain.knowledge_graph.get_all_nodes()
        edges = self.gui.brain.knowledge_graph.get_all_edges()

        export_data = {
            "metadata": {
                "export_time": time.time(),
                "format_version": "1.0",
                "system_name": "ЕВА"
            },
            "nodes": nodes,
            "edges": edges
        }

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


def show_settings_dialog(self):
    """Shows graph settings dialog."""
    settings_window = tk.Toplevel(self.gui.root)
    settings_window.title("Настройки графа знаний")
    settings_window.geometry("400x300")
    settings_window.transient(self.gui.root)
    settings_window.grab_set()

    settings_frame = ttk.Frame(settings_window, padding=10)
    settings_frame.pack(fill=tk.BOTH, expand=True)

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

    button_frame = ttk.Frame(settings_frame)
    button_frame.pack(fill=tk.X, pady=(10, 0))

    def apply_settings():
        self.node_size = node_size_var.get() * 10
        self.edge_width = edge_width_var.get()
        self.font_size = font_size_var.get()

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


def on_tab_changed(self, event):
    """Handles tab change."""
    current_tab = self.notebook.select()
    tab_name = self.notebook.tab(current_tab, "text")

    if tab_name == "Статистика":
        self._load_statistics()
    elif tab_name == "Домены":
        self._load_domains_data()


def on_domain_double_click(self, event):
    """Handles double click on domain."""
    item = self.domains_tree.selection()
    if item:
        self._show_domain_on_graph()


def show_domain_on_graph(self):
    """Shows domain on the graph."""
    item = self.domains_tree.selection()
    if not item:
        return

    values = self.domains_tree.item(item, "values")
    if not values:
        return

    domain = values[0]

    self.current_view = f"domain:{domain}"

    self._update_graph_data()


def show_domain_details(self):
    """Shows domain details."""
    item = self.domains_tree.selection()
    if not item:
        return

    values = self.domains_tree.item(item, "values")
    if not values:
        return

    domain = values[0]

    try:
        details_window = tk.Toplevel(self.gui.root)
        details_window.title(f"Домен: {domain}")
        details_window.geometry("600x400")
        details_window.transient(self.gui.root)
        details_window.grab_set()

        notebook = ttk.Notebook(details_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        info_frame = ttk.Frame(notebook)
        notebook.add(info_frame, text="Информация")

        info_text = (
            f"Домен: {domain}\n"
            f"Узлы: {values[1]}\n"
            f"Связи: {values[2]}\n"
            f"Последнее обновление: {values[3]}\n\n"
            f"Домены представляют собой категории знаний в системе.\n"
            f"Каждый узел графа знаний относится к определенному домену."
        )

        info_label = ttk.Label(
            info_frame,
            text=info_text,
            justify=tk.LEFT,
            wraplength=550
        )
        info_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        nodes_frame = ttk.Frame(notebook)
        notebook.add(nodes_frame, text="Узлы")

        columns = ("name", "type", "strength", "description")
        nodes_tree = ttk.Treeview(
            nodes_frame,
            columns=columns,
            show="headings",
            selectmode="browse"
        )

        nodes_tree.heading("name", text="Название")
        nodes_tree.heading("type", text="Тип")
        nodes_tree.heading("strength", text="Сила")
        nodes_tree.heading("description", text="Описание")

        nodes_tree.column("name", width=150, anchor=tk.W)
        nodes_tree.column("type", width=100, anchor=tk.W)
        nodes_tree.column("strength", width=80, anchor=tk.CENTER)
        nodes_tree.column("description", width=200, anchor=tk.W)

        scrollbar = ttk.Scrollbar(nodes_frame, orient=tk.VERTICAL, command=nodes_tree.yview)
        nodes_tree.configure(yscrollcommand=scrollbar.set)

        nodes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

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


def export_domain(self):
    """Exports domain to file."""
    item = self.domains_tree.selection()
    if not item:
        return

    values = self.domains_tree.item(item, "values")
    if not values:
        return

    domain = values[0]

    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")],
            title=f"Экспорт домена {domain}"
        )

        if not file_path:
            return

        if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
            raise ValueError("Граф знаний недоступен")

        nodes = self.gui.brain.knowledge_graph.get_nodes_by_domain(domain)
        node_ids = [node["id"] for node in nodes]

        edges = []
        for node_id in node_ids:
            node_edges = self.gui.brain.knowledge_graph.get_edges(node_id)
            edges.extend(node_edges)

        export_data = {
            "metadata": {
                "domain": domain,
                "export_time": time.time(),
                "format_version": "1.0",
                "system_name": "ЕВА"
            },
            "nodes": nodes,
            "edges": edges
        }

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
