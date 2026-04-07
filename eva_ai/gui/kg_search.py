# gui/kg_search.py
"""Search functionality, filtering, query handling for KnowledgeGraphModule."""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import Menu
import logging
import time
import json
import os

logger = logging.getLogger("eva_ai.gui.knowledge")


def create_search_tab(self, parent):
    """Creates the search tab."""
    search_frame = ttk.Frame(parent)
    search_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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

    results_frame = ttk.LabelFrame(search_frame, text="Результаты поиска")
    results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    columns = ("name", "type", "domain", "strength", "description")
    self.search_tree = ttk.Treeview(
        results_frame,
        columns=columns,
        show="headings",
        selectmode="browse"
    )

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

    scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.search_tree.yview)
    self.search_tree.configure(yscrollcommand=scrollbar.set)

    self.search_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    _create_search_context_menu(self)

    self.search_tree.bind("<Double-1>", self._on_search_result_double_click)


def _create_search_context_menu(self):
    """Creates context menu for search results."""
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
    self.search_tree.bind("<Button-3>", self._show_search_context_menu)


def _show_search_context_menu(self, event):
    """Shows context menu for search results."""
    item = self.search_tree.identify_row(event.y)
    if item:
        self.search_tree.selection_set(item)
        self.search_context_menu.tk_popup(event.x_root, event.y_root)


def show_search_dialog(self):
    """Shows search dialog."""
    search_window = tk.Toplevel(self.gui.root)
    search_window.title("Поиск в графе знаний")
    search_window.geometry("400x150")
    search_window.transient(self.gui.root)
    search_window.grab_set()

    input_frame = ttk.Frame(search_window, padding=10)
    input_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(input_frame, text="Введите запрос:").pack(anchor=tk.W, pady=(0, 5))

    search_entry = ttk.Entry(input_frame, width=50)
    search_entry.pack(fill=tk.X, pady=(0, 10))
    search_entry.focus()

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

    search_entry.bind("<Return>", lambda event: perform_search())


def perform_search(self, query=None):
    """Performs search in the knowledge graph."""
    if not query:
        query = self.search_entry.get().strip()
        if not query:
            return

    try:
        if not self.gui.brain or not hasattr(self.gui.brain, 'knowledge_graph'):
            raise ValueError("Граф знаний недоступен")

        results = self.gui.brain.knowledge_graph.search_nodes(query)

        self.search_query = query
        self.search_results = results
        self.search_index = 0

        self.current_view = "search"

        self._update_graph_data()

        self._update_search_results()

    except Exception as e:
        logger.error(f"Ошибка поиска: {e}", exc_info=True)
        messagebox.showerror(
            "Ошибка",
            f"Не удалось выполнить поиск: {str(e)}"
        )


def update_search_results(self):
    """Updates search results in the table."""
    for item in self.search_tree.get_children():
        self.search_tree.delete(item)

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


def on_search_result_double_click(self, event):
    """Handles double click on search result."""
    item = self.search_tree.selection()
    if item:
        self._highlight_search_result()


def highlight_search_result(self):
    """Highlights search result on the graph."""
    item = self.search_tree.selection()
    if not item:
        return

    values = self.search_tree.item(item, "values")
    if not values:
        return

    node_id = None
    for node in self.graph.nodes():
        node_data = self.graph.nodes[node]
        if node_data.get("label", "") == values[0]:
            node_id = node
            break

    if node_id:
        self.selected_node = node_id
        self.selected_edge = None
        self._show_node_info(node_id)

        self._center_graph_on_node(node_id)


def show_search_result_details(self):
    """Shows details of search result."""
    item = self.search_tree.selection()
    if not item:
        return

    values = self.search_tree.item(item, "values")
    if not values:
        return

    node_id = None
    for node in self.graph.nodes():
        node_data = self.graph.nodes[node]
        if node_data.get("label", "") == values[0]:
            node_id = node
            break

    if node_id:
        self._show_node_details(node_id)


def export_search_result(self):
    """Exports search result."""
    item = self.search_tree.selection()
    if not item:
        return

    values = self.search_tree.item(item, "values")
    if not values:
        return

    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")],
            title="Экспорт результата поиска"
        )

        if not file_path:
            return

        node_id = None
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if node_data.get("label", "") == values[0]:
                node_id = node
                break

        if not node_id:
            raise ValueError("Узел не найден")

        node_data = self.graph.nodes[node_id]

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

        for edge in self.graph.edges(node_id):
            edge_data = self.graph.edges[edge]
            export_data["related_edges"].append({
                "id": edge_data.get("id", ""),
                "source_id": edge[0],
                "target_id": edge[1],
                "relation_type": edge_data.get("label", ""),
                "strength": edge_data.get("strength", 0.0)
            })

            other_node = edge[1] if edge[0] == node_id else edge[0]
            other_node_data = self.graph.nodes[other_node]
            export_data["related_nodes"].append({
                "id": other_node,
                "name": other_node_data.get("label", "N/A"),
                "type": other_node_data.get("type", "N/A"),
                "domain": other_node_data.get("domain", "N/A"),
                "strength": other_node_data.get("strength", 0.0)
            })

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
