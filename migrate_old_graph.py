"""
Migration script: Old Graph (JSON) -> New Graph (fractal_graph_v2)

Переносит:
- Модельные узлы (A, B, C) как статические
- Компоненты моделей (L0-L4)
- Опыт пользователей (experiences/)
"""

import json
import logging
import os
import sys
import shutil
import glob

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("migration")

# Paths
OLD_NODES = 'eva/memory/fractal_torch_storage/unified_memory/nodes.json'
OLD_EDGES = 'eva/memory/fractal_torch_storage/unified_memory/edges.json'
OLD_EXPERIENCES = 'eva/memory/fractal_torch_storage/unified_memory/experiences/*.json'
NEW_DATA_DIR = 'eva/memory/fractal_graph_v2/fractal_graph_v2_data'


def load_old_graph():
    """Загрузить старый граф из JSON."""
    logger.info("Загрузка старого графа...")
    
    with open(OLD_NODES, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    with open(OLD_EDGES, 'r', encoding='utf-8') as f:
        edges = json.load(f)
    
    logger.info(f"  Загружено узлов: {len(nodes)}")
    logger.info(f"  Загружено связей: {len(edges)}")
    
    return nodes, edges


def load_experiences():
    """Загрузить опыт пользователей."""
    logger.info("Загрузка опыта пользователей...")
    
    exp_files = glob.glob(OLD_EXPERIENCES)
    logger.info(f"  Найдено файлов опыта: {len(exp_files)}")
    
    experiences = []
    for exp_file in exp_files:
        with open(exp_file, 'r', encoding='utf-8') as f:
            exp = json.load(f)
            experiences.append(exp)
    
    logger.info(f"  Загружено опыта: {len(experiences)}")
    return experiences


def categorize_nodes(nodes):
    """Категоризировать узлы по типу."""
    model_nodes = {}       # model_a, model_b, model_c
    component_nodes = {}   # L0-L4 компоненты моделей
    knowledge_nodes = {}   # Опыт пользователей
    
    for node_id, node in nodes.items():
        node_type = node.get('node_type', '')
        
        if node_type in ['model_a', 'model_b', 'model_c']:
            model_nodes[node_id] = node
        elif node_id.startswith('model::model_'):
            component_nodes[node_id] = node
        else:
            knowledge_nodes[node_id] = node
    
    return model_nodes, component_nodes, knowledge_nodes


def migrate_to_new_graph(model_nodes, component_nodes, knowledge_nodes, edges, experiences):
    """Перенести данные в новый граф."""
    from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph
    
    # Очистка нового графа
    if os.path.exists(NEW_DATA_DIR):
        shutil.rmtree(NEW_DATA_DIR)
    
    graph = FractalMemoryGraph()
    
    logger.info("Начало миграции...")
    
    # 1. Добавляем модели A, B, C как статические узлы
    model_mapping = {}
    for node_id, node in model_nodes.items():
        content = node.get('content', '')
        node_type = node.get('node_type', 'concept')
        
        if 'model_a' in node_type:
            model_name = 'Qwen 2.5 3B (Model A) - логика'
            level = 0
        elif 'model_b' in node_type:
            model_name = 'Qwen 2.5 3B (Model B) - развитие'
            level = 0
        elif 'model_c' in node_type:
            model_name = 'Qwen 2.5 Coder 1.5B (Model C) - код'
            level = 0
        else:
            model_name = content[:50]
            level = 0
        
        new_node = graph.add_node(
            content=model_name,
            node_type=node_type,
            level=level,
            confidence=1.0,
            metadata={'source': 'migration', 'is_static': True},
            auto_vectorize=False
        )
        model_mapping[node_id] = new_node.id
        logger.info(f"  Добавлена модель: {node_type}")
    
    # 2. Добавляем компоненты моделей (только основные, не все L4 тензоры)
    component_count = 0
    component_mapping = {}
    
    for node_id, node in component_nodes.items():
        if 'L4::' in node_id:
            continue
        
        content = node.get('content', '')[:100]
        node_type = node.get('node_type', 'component')
        level = node.get('level', 1)
        
        new_node = graph.add_node(
            content=content,
            node_type=node_type,
            level=level,
            confidence=1.0,
            metadata={'source': 'migration', 'is_static': True},
            auto_vectorize=False
        )
        component_mapping[node_id] = new_node.id
        component_count += 1
    
    logger.info(f"  Добавлено компонентов: {component_count}")
    
    # 3. Добавляем знания из nodes.json
    knowledge_count = 0
    for node_id, node in knowledge_nodes.items():
        content = node.get('content', '')
        if not content or len(content) < 3:
            continue
        
        node_type = node.get('node_type', 'concept')
        level = node.get('level', 1)
        
        if len(content) > 500:
            content = content[:500]
        
        new_node = graph.add_node(
            content=content,
            node_type=node_type,
            level=level,
            confidence=0.7,
            metadata={'source': 'migration'},
            auto_vectorize=True
        )
        knowledge_count += 1
    
    logger.info(f"  Добавлено знаний из nodes: {knowledge_count}")
    
    # 4. Добавляем опыт пользователей
    exp_count = 0
    exp_query_nodes = []
    
    for exp in experiences:
        query = exp.get('query', '')
        response = exp.get('response', '')
        model_used = exp.get('model_used', 'unknown')
        
        if not query or len(query) < 3:
            continue
        
        # Добавляем query как узел
        query_node = graph.add_node(
            content=query[:200],
            node_type='query',
            level=2,
            confidence=exp.get('quality_score', 0.5),
            metadata={'source': 'experience', 'model': model_used},
            auto_vectorize=True
        )
        exp_query_nodes.append(query_node.id)
        
        # Добавляем response как связанный узел
        if response and len(response) > 3:
            resp_node = graph.add_node(
                content=response[:300],
                node_type='response',
                level=2,
                confidence=exp.get('quality_score', 0.5),
                metadata={'source': 'experience', 'model': model_used},
                auto_vectorize=True
            )
            
            # Связываем query -> response
            graph.storage.add_edge(
                source_id=query_node.id,
                target_id=resp_node.id,
                relation_type='generated_by',
                weight=exp.get('quality_score', 0.5)
            )
        
        exp_count += 1
        if exp_count % 100 == 0:
            logger.info(f"    Обработано опыта: {exp_count}")
    
    logger.info(f"  Добавлено опыта: {exp_count}")
    
    # 5. Переносим связи (только основные)
    edge_count = 0
    for edge_id, edge in edges.items():
        source_old = edge.get('source_id', '')
        target_old = edge.get('target_id', '')
        relation = edge.get('relation_type', 'related_to')
        
        if 'L4::' in source_old or 'L4::' in target_old:
            continue
        
        source_new = model_mapping.get(source_old) or component_mapping.get(source_old)
        target_new = model_mapping.get(target_old) or component_mapping.get(target_old)
        
        if source_new and target_new:
            graph.storage.add_edge(
                source_id=source_new,
                target_id=target_new,
                relation_type=relation,
                weight=edge.get('strength', 0.5)
            )
            edge_count += 1
    
    logger.info(f"  Добавлено связей: {edge_count}")
    
    return graph


def main():
    """Главная функция миграции."""
    logger.info("=" * 60)
    logger.info("МИГРАЦИЯ: Старый граф -> fractal_graph_v2")
    logger.info("=" * 60)
    
    # Загружаем старый граф
    nodes, edges = load_old_graph()
    
    # Загружаем опыт
    experiences = load_experiences()
    
    # Категоризируем
    model_nodes, component_nodes, knowledge_nodes = categorize_nodes(nodes)
    logger.info(f"  Модели: {len(model_nodes)}")
    logger.info(f"  Компоненты: {len(component_nodes)}")
    logger.info(f"  Знания: {len(knowledge_nodes)}")
    
    # Миграция
    graph = migrate_to_new_graph(model_nodes, component_nodes, knowledge_nodes, edges, experiences)
    
    logger.info("=" * 60)
    logger.info("МИГРАЦИЯ ЗАВЕРШЕНА")
    logger.info(f"Узлов в новом графе: {len(graph.storage.nodes)}")
    logger.info(f"Связей в новом графе: {len(graph.storage.edges)}")
    logger.info("=" * 60)
    
    # Тест поиска
    results = graph.semantic_search('логика', top_k=3, min_level=0)
    logger.info(f"\nТест поиска 'логика': {len(results)} результатов")
    for r in results:
        logger.info(f"  - {r.get('content', 'N/A')[:50]}")


if __name__ == '__main__':
    main()