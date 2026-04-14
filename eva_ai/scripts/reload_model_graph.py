"""
Скрипт очистки FractalGraph от устаревших данных qwen2.5 
и загрузки актуальных данных qwen3 4B
"""
import os
import sys

# Добавляем путь к EVA
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph
from eva_ai.memory.fractal_graph_v2.gguf_parser import clear_and_reload_model_graph


def main():
    print("=" * 60)
    print("Очистка FractalGraph от устаревших данных qwen2.5")
    print("=" * 60)
    
    # Инициализируем граф
    graph_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "memory", "fractal_graph_v2", "fractal_graph_v2_data"
    )
    
    print(f"\nГраф: {graph_dir}")
    
    graph = FractalMemoryGraph(storage_dir=graph_dir)
    
    # Проверяем текущее состояние
    total_nodes = len(graph.storage.nodes)
    print(f"Всего узлов до очистки: {total_nodes}")
    
    # Пути к актуальным моделям
    project_root = r"C:\Users\black\OneDrive\Desktop\CogniFlex"
    model_paths = [
        os.path.join(project_root, "eva_pie_architecture", "models", "gguf_models", "ruadapt_qwen3_4b_q4_k_m.gguf"),
        os.path.join(project_root, "eva_pie_architecture", "models", "gguf_models", "qwen2.5-coder-1.5b-instruct", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
    ]
    
    # Очищаем и перезагружаем
    print("\nОчистка старых MODEL_ узлов...")
    result = clear_and_reload_model_graph(graph, model_paths)
    
    print(f"\nРезультат:")
    print(f"  Удалено: {result['nodes_removed']}")
    print(f"  Добавлено: {result['nodes_added']}")
    
    # Проверяем итоговое состояние
    total_nodes = len(graph.storage.nodes)
    print(f"\nВсего узлов после: {total_nodes}")
    
    # Выводим MODEL_ узлы
    print("\nАктуальные MODEL_ узлы:")
    for node_id, node in graph.storage.nodes.items():
        if node.node_type.startswith('MODEL_'):
            print(f"  [{node.node_type}] {node.content[:80]}")


if __name__ == "__main__":
    main()
