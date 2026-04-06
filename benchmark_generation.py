"""
Benchmark: Сравнение производительности old vs new системы генерации

Тестирует:
1. Скорость токенизации
2. Скорость поиска контекста
3. Скорость семантического поиска
4. Точность генерации
"""

import os
import sys
import time
import json
import shutil

sys.stdout.reconfigure(encoding='utf-8')


def benchmark_old_system():
    """Старая система - direct JSON loading."""
    print("\n=== OLD SYSTEM (JSON files) ===")
    
    # Загрузка данных из JSON
    start = time.perf_counter()
    
    nodes_path = 'eva/memory/fractal_torch_storage/unified_memory/nodes.json'
    edges_path = 'eva/memory/fractal_torch_storage/unified_memory/edges.json'
    
    with open(nodes_path, 'r', encoding='utf-8') as f:
        old_nodes = json.load(f)
    
    with open(edges_path, 'r', encoding='utf-8') as f:
        old_edges = json.load(f)
    
    load_time = time.perf_counter() - start
    print(f"JSON load time: {load_time:.3f}s")
    print(f"  Nodes: {len(old_nodes)}")
    print(f"  Edges: {len(old_edges)}")
    
    # Тест простого поиска (линейный)
    queries = ['логика', 'python', 'машинное обучение', 'код', 'развитие']
    
    search_times = []
    for query in queries:
        query_lower = query.lower()
        
        start = time.perf_counter()
        results = []
        for node_id, node in old_nodes.items():
            content = node.get('content', '').lower()
            if query_lower in content:
                results.append((node_id, content[:50]))
        search_time = time.perf_counter() - start
        search_times.append(search_time)
    
    avg_search = sum(search_times) / len(search_times)
    print(f"Linear search avg: {avg_search*1000:.2f}ms")
    
    return {
        'load_time': load_time,
        'nodes': len(old_nodes),
        'edges': len(old_edges),
        'search_time': avg_search
    }


def benchmark_new_system():
    """Новая система - fractal_graph_v2."""
    print("\n=== NEW SYSTEM (fractal_graph_v2) ===")
    
    from eva.memory.fractal_graph_v2 import FractalMemoryGraph
    
    # Инициализация
    start = time.perf_counter()
    graph = FractalMemoryGraph()
    init_time = time.perf_counter() - start
    print(f"Init time: {init_time:.3f}s")
    print(f"  Nodes: {len(graph.storage.nodes)}")
    print(f"  Edges: {len(graph.storage.edges)}")
    
    # Тест токенизации
    from eva.memory.fractal_graph_v2 import GraphTokenizer
    tokenizer = GraphTokenizer(graph=graph)
    
    test_texts = [
        "что такое машинное обучение",
        "как работает python",
        "объясни концепцию логики",
        "напиши код на python",
        "расскажи про развитие"
    ]
    
    tokenize_times = []
    for text in test_texts:
        start = time.perf_counter()
        tokens = tokenizer.tokenize(text, add_special_tokens=True)
        t_time = time.perf_counter() - start
        tokenize_times.append(t_time)
    
    avg_tokenize = sum(tokenize_times) / len(tokenize_times)
    print(f"Tokenize avg: {avg_tokenize*1000:.2f}ms")
    print(f"  Token cache size: {len(tokenizer._tokenize_cache)}")
    
    # Тест семантического поиска
    queries = ['логика', 'python', 'машинное обучение', 'код', 'развитие']
    
    search_times = []
    for query in queries:
        start = time.perf_counter()
        results = graph.semantic_search(query, top_k=5, min_level=1)
        search_time = time.perf_counter() - start
        search_times.append(search_time)
        print(f"  Search '{query}': {search_time*1000:.2f}ms, {len(results)} results")
    
    avg_search = sum(search_times) / len(search_times)
    print(f"Semantic search avg: {avg_search*1000:.2f}ms")
    
    # Тест получения контекста
    context_times = []
    for query in queries:
        start = time.perf_counter()
        context, node_ids = tokenizer.get_context_for_generation(query, max_context_length=256)
        c_time = time.perf_counter() - start
        context_times.append(c_time)
    
    avg_context = sum(context_times) / len(context_times)
    print(f"Context generation avg: {avg_context*1000:.2f}ms")
    print(f"  Context cache size: {len(tokenizer._context_cache)}")
    
    return {
        'init_time': init_time,
        'nodes': len(graph.storage.nodes),
        'edges': len(graph.storage.edges),
        'tokenize_time': avg_tokenize,
        'search_time': avg_search,
        'context_time': avg_context
    }


def benchmark_full_generation():
    """Тест полной генерации ответа (без LLM - только из графа)."""
    print("\n=== FULL GENERATION TEST ===")
    
    from eva.memory.fractal_graph_v2 import FractalMemoryGraph, GraphTokenizer
    
    graph = FractalMemoryGraph()
    tokenizer = GraphTokenizer(graph=graph)
    
    queries = [
        "что такое python",
        "объясни машинное обучение",
        "что такое логика",
    ]
    
    for query in queries:
        # Получение контекста
        start = time.perf_counter()
        context, node_ids = tokenizer.get_context_for_generation(query, max_context_length=512)
        context_time = time.perf_counter() - start
        
        # Семантический поиск
        start = time.perf_counter()
        results = graph.semantic_search(query, top_k=10, min_level=1)
        search_time = time.perf_counter() - start
        
        # Формирование ответа
        start = time.perf_counter()
        if results:
            response_parts = []
            for r in results[:5]:
                content = r.get('content', '')
                if content:
                    response_parts.append(content)
            response = "; ".join(response_parts[:3])
        else:
            response = "Контекст не найден"
        response_time = time.perf_counter() - start
        
        total = context_time + search_time + response_time
        
        print(f"\nQuery: {query}")
        print(f"  Context: {context_time*1000:.1f}ms")
        print(f"  Search: {search_time*1000:.1f}ms")
        print(f"  Response: {response_time*1000:.1f}ms")
        print(f"  TOTAL: {total*1000:.1f}ms")
        print(f"  Response: {response[:80]}..." if len(response) > 80 else f"  Response: {response}")


def main():
    print("=" * 60)
    print("BENCHMARK: Old vs New Graph System")
    print("=" * 60)
    
    # Старая система
    old_stats = benchmark_old_system()
    
    # Новая система
    new_stats = benchmark_new_system()
    
    # Полная генерация
    benchmark_full_generation()
    
    # Итоговое сравнение
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Old system:")
    print(f"  Load: {old_stats['load_time']:.3f}s")
    print(f"  Linear search: {old_stats['search_time']*1000:.1f}ms")
    print()
    print(f"New system:")
    print(f"  Init: {new_stats['init_time']:.3f}s")
    print(f"  Tokenize: {new_stats['tokenize_time']*1000:.2f}ms")
    print(f"  Semantic search: {new_stats['search_time']*1000:.1f}ms")
    print(f"  Context gen: {new_stats['context_time']*1000:.1f}ms")
    print()
    
    speedup = old_stats['search_time'] / new_stats['search_time']
    print(f"Search speedup: {speedup:.1f}x faster")


if __name__ == '__main__':
    main()