"""
Model status и Fractal Graph API маршруты для Web GUI ЕВА
"""
import os
import logging
from flask import jsonify, request

logger = logging.getLogger("eva.webgui")


def register_routes(app, web_gui_instance):

    @app.route('/api/model-status')
    def api_model_status():
        """Получить статус моделей пайплайна."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        model_status = {
            'pipeline_ready': False,
            'models': {
                'model_a': {
                    'name': 'Qwen 2.5 3B Instruct',
                    'role': 'Логика и факты',
                    'loaded': False,
                    'n_ctx': 2048,
                    'temperature': 0.3
                },
                'model_b': {
                    'name': 'Qwen 2.5 3B Instruct',
                    'role': 'Развитие мысли',
                    'loaded': False,
                    'n_ctx': 2048,
                    'temperature': 0.3
                },
                'model_c': {
                    'name': 'Qwen 2.5 Coder 1.5B Instruct',
                    'role': 'Генерация кода',
                    'loaded': False,
                    'n_ctx': 2048,
                    'temperature': 0.1,
                    'lazy_load': True
                }
            },
            'fractal_memory': {
                'enabled': False,
                'nodes': 0,
                'edges': 0,
                'experiences': 0,
                'concepts': 0
            }
        }

        try:
            brain = web_gui_instance.brain
            if brain:
                if hasattr(brain, 'two_model_pipeline') and brain.two_model_pipeline:
                    pipeline = brain.two_model_pipeline
                    model_status['pipeline_ready'] = True
                    model_status['models']['model_a']['loaded'] = pipeline.model_a is not None
                    model_status['models']['model_b']['loaded'] = pipeline.model_b is not None
                    model_status['models']['model_c']['loaded'] = pipeline.model_c is not None

                if hasattr(brain, 'fractal_memory') and brain.fractal_memory:
                    fm = brain.fractal_memory
                    model_status['fractal_memory']['enabled'] = True
                    stats = fm.get_stats()
                    model_status['fractal_memory']['nodes'] = stats.get('total_nodes', 0)
                    model_status['fractal_memory']['edges'] = stats.get('total_edges', 0)

                    exp_dir = os.path.join(fm.storage_dir, 'experiences')
                    concept_dir = os.path.join(fm.storage_dir, 'concepts')
                    if os.path.exists(exp_dir):
                        model_status['fractal_memory']['experiences'] = len([f for f in os.listdir(exp_dir) if f.endswith('.json')])
                    if os.path.exists(concept_dir):
                        model_status['fractal_memory']['concepts'] = len([f for f in os.listdir(concept_dir) if f.endswith('.json')])
        except Exception as e:
            logger.error(f"Error getting model status: {e}")

        return jsonify(model_status)

    @app.route('/api/fractal-graph')
    def api_fractal_graph():
        """Получить данные фрактального графа памяти для визуализации."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        graph_data = {
            'nodes': [],
            'edges': [],
            'stats': {}
        }

        try:
            brain = web_gui_instance.brain
            if brain and hasattr(brain, 'fractal_memory') and brain.fractal_memory:
                fm = brain.fractal_memory

                for node_id, node in fm.nodes.items():
                    if 'model::' in node_id:
                        ctx = getattr(node, 'context', {})
                        node_type = ctx.get('node_type', 'unknown')
                        level = getattr(node, 'level', 0)

                        graph_data['nodes'].append({
                            'id': node_id,
                            'label': node_type,
                            'level': level,
                            'group': node_type,
                            'size': max(5, 20 - level * 3),
                            'content': getattr(node, 'content', '')[:100]
                        })

                for edge_id, edge in fm.edges.items():
                    graph_data['edges'].append({
                        'from': edge.source_id,
                        'to': edge.target_id,
                        'relation': edge.relation_type
                    })

                graph_data['stats'] = {
                    'total_nodes': len(fm.nodes),
                    'model_nodes': len([n for n in graph_data['nodes'] if 'model::' in n.get('id', '')]),
                    'total_edges': len(fm.edges)
                }
        except Exception as e:
            logger.error(f"Error getting fractal graph: {e}")

        return jsonify(graph_data)
