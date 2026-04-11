"""
Graph routes - endpoints для работы с графом знаний (FractalGraph v2)
"""
import logging
from flask import jsonify, request

logger = logging.getLogger("eva_ai.webgui.routes_graph")


def register_graph_routes(app, web_gui_instance):
    """Регистрация routes для графа знаний."""
    
    @app.route('/api/contradictions')
    def api_contradictions():
        """Получить список противоречий."""
        result = {'total': 0, 'active': 0, 'resolved': 0, 'items': []}
        try:
            if web_gui_instance and web_gui_instance.brain:
                cm = getattr(web_gui_instance.brain, 'contradiction_manager', None)
                if cm is None:
                    cm = web_gui_instance.brain.components.get('contradiction_manager')
                if cm and hasattr(cm, 'get_contradictions'):
                    result['items'] = cm.get_contradictions()[:50]
                    result['total'] = len(result['items'])
                    result['active'] = len([c for c in result['items'] if not getattr(c, 'resolved', False)])
                    result['resolved'] = len([c for c in result['items'] if getattr(c, 'resolved', False)])
        except Exception as e:
            logger.error(f'Contradictions error: {e}')
        return jsonify(result)

    @app.route('/api/concepts')
    def api_concepts():
        """Получить список концепций из графа."""
        result = {'total': 0, 'items': []}
        try:
            if web_gui_instance and web_gui_instance.brain:
                fg = getattr(web_gui_instance.brain, 'fractal_graph_v2', None)
                if fg is None:
                    fg = web_gui_instance.brain.components.get('fractal_graph_v2')
                if fg and hasattr(fg, 'get_nodes_list'):
                    nodes = [n for n in fg.get_nodes_list() if getattr(n, 'node_type', '') == 'concept']
                    result['total'] = len(nodes)
                    result['items'] = [
                        {
                            'id': getattr(n, 'id', str(i)),
                            'content': getattr(n, 'content', '')[:100],
                            'metadata': getattr(n, 'metadata', {})
                        }
                        for i, n in enumerate(nodes[:50])
                    ]
        except Exception as e:
            logger.error(f'Concepts error: {e}')
        return jsonify(result)

    @app.route('/api/graph/stats')
    def api_graph_stats():
        """Получить статистику графа."""
        result = {'total_nodes': 0, 'total_edges': 0, 'groups': 0}
        try:
            if web_gui_instance and web_gui_instance.brain:
                fg = getattr(web_gui_instance.brain, 'fractal_graph_v2', None)
                if fg is None:
                    fg = web_gui_instance.brain.components.get('fractal_graph_v2')
                if fg:
                    if hasattr(fg, 'get_stats'):
                        result.update(fg.get_stats())
                    else:
                        result['total_nodes'] = len(fg.storage.nodes)
                        result['total_edges'] = len(fg.storage.edges)
        except Exception as e:
            logger.error(f'Graph stats error: {e}')
        return jsonify(result)

    @app.route('/api/nodes')
    def api_nodes():
        """Получить узлы графа."""
        limit = request.args.get('limit', 200, type=int)
        nodes = []
        try:
            if web_gui_instance and web_gui_instance.brain:
                fg = getattr(web_gui_instance.brain, 'fractal_graph_v2', None)
                if fg is None:
                    fg = web_gui_instance.brain.components.get('fractal_graph_v2')
                if fg and hasattr(fg, 'get_nodes_list'):
                    for n in fg.get_nodes_list(limit=limit):
                        nodes.append({
                            'id': getattr(n, 'id', ''),
                            'content': getattr(n, 'content', '')[:200],
                            'type': getattr(n, 'node_type', 'unknown'),
                            'metadata': getattr(n, 'metadata', {})
                        })
        except Exception as e:
            logger.error(f'Nodes error: {e}')
        return jsonify({'nodes': nodes, 'count': len(nodes)})

    @app.route('/api/edges')
    def api_edges():
        """Получить связи графа."""
        limit = request.args.get('limit', 500, type=int)
        edges = []
        try:
            if web_gui_instance and web_gui_instance.brain:
                fg = getattr(web_gui_instance.brain, 'fractal_graph_v2', None)
                if fg is None:
                    fg = web_gui_instance.brain.components.get('fractal_graph_v2')
                if fg and hasattr(fg, 'get_edges_list'):
                    for e in fg.get_edges_list(limit=limit):
                        edges.append({
                            'id': getattr(e, 'id', ''),
                            'source': getattr(e, 'source', ''),
                            'target': getattr(e, 'target', ''),
                            'type': getattr(e, 'edge_type', 'related')
                        })
        except Exception as e:
            logger.error(f'Edges error: {e}')
        return jsonify({'edges': edges, 'count': len(edges)})

    logger.info("Graph routes registered")
