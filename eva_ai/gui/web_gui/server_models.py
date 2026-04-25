"""
Model status и Fractal Graph API маршруты для Web GUI ЕВА
"""
import os
import logging
from flask import jsonify, request

logger = logging.getLogger("eva_ai.webgui")


def register_routes(app, web_gui_instance):

    @app.route('/api/model-status')
    def api_model_status():
        """Получить статус моделей пайплайна."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        model_status = {
            'pipeline_ready': False,
            'models': {
                'model_a': {'loaded': False},
                'model_b': {'loaded': False},
                'model_c': {'loaded': False},
                'fcp': {'loaded': False}
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
                if hasattr(brain, 'fcp_pipeline') and brain.fcp_pipeline and brain.fcp_pipeline.pipeline:
                    model_status['pipeline_ready'] = True
                    model_status['models']['fcp']['loaded'] = True
                elif hasattr(brain, 'two_model_pipeline') and brain.two_model_pipeline:
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
                graph_data['nodes'] = fm.get_all_nodes()
                graph_data['edges'] = fm.get_all_edges()
                stats = fm.get_stats()
                graph_data['stats'] = stats
        except Exception as e:
            logger.error(f"Error getting fractal graph: {e}")

        return jsonify(graph_data)

    @app.route('/api/model-load', methods=['POST'])
    def api_model_load():
        """Загрузить модель по требованию."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        data = request.get_json() or {}
        model_id = data.get('model_id', '')

        try:
            brain = web_gui_instance.brain
            if not brain:
                return jsonify({'error': 'Brain не инициализирован'}), 500

            if model_id == 'model_a':
                if hasattr(brain, 'two_model_pipeline') and brain.two_model_pipeline:
                    brain.two_model_pipeline._ensure_model_a_loaded()
                    return jsonify({'success': True, 'model': 'model_a'})
            elif model_id == 'model_b':
                if hasattr(brain, 'two_model_pipeline') and brain.two_model_pipeline:
                    brain.two_model_pipeline._ensure_model_b_loaded()
                    return jsonify({'success': True, 'model': 'model_b'})
            elif model_id == 'fcp':
                if hasattr(brain, 'fcp_pipeline') and brain.fcp_pipeline:
                    return jsonify({'success': True, 'model': 'fcp'})

            return jsonify({'error': 'Модель не найдена'}), 404
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return jsonify({'error': str(e)}), 500