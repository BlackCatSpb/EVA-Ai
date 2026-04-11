"""
Analytics routes: memory-graph, analytics, learning, events
"""
import os
import logging
import time
import json
from datetime import datetime

from flask import jsonify, request, Response, stream_with_context
import psutil

logger = logging.getLogger("eva_ai.webgui.routes_analytics")


def register_analytics_routes(app, web_gui_instance):
    """Register analytics routes."""
    logger.info("Registering analytics routes...")

    @app.route('/api/memory-graph')
    def api_memory_graph():
        """Get memory graph data."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        graph_data = {'nodes': [], 'edges': [], 'stats': {}}

        try:
            if hasattr(web_gui_instance, 'bridge') and web_gui_instance.bridge:
                cached = web_gui_instance.bridge.get_cached_memory_graph()
                if cached and cached.get('nodes'):
                    return jsonify(cached)
            
            fg = getattr(web_gui_instance.brain, 'fractal_graph_v2', None)
            if fg is None:
                fg = getattr(web_gui_instance.brain, 'components', {}).get('fractal_graph_v2')
            if fg is None and hasattr(web_gui_instance.brain, 'memory_manager'):
                fg = getattr(web_gui_instance.brain.memory_manager, 'fractal_graph_v2', None)
            
            if fg:
                if hasattr(fg, 'get_nodes_list'):
                    nodes_list = fg.get_nodes_list()
                    graph_data['nodes'] = [
                        {
                            'id': n.id if hasattr(n, 'id') else str(i),
                            'label': n.content[:100] if hasattr(n, 'content') else str(n),
                            'type': n.node_type if hasattr(n, 'node_type') else 'concept',
                            'level': n.level if hasattr(n, 'level') else 1
                        }
                        for i, n in enumerate(nodes_list[:200])
                    ]
                    graph_data['stats']['total_nodes'] = len(nodes_list)
                
                if hasattr(fg, 'get_edges_list'):
                    edges_list = fg.get_edges_list()
                    graph_data['edges'] = [
                        {
                            'id': e.id if hasattr(e, 'id') else str(i),
                            'source': e.source if hasattr(e, 'source') else '',
                            'target': e.target if hasattr(e, 'target') else ''
                        }
                        for i, e in enumerate(edges_list[:500])
                    ]
                    graph_data['stats']['total_edges'] = len(edges_list)
                
                if hasattr(fg, 'get_stats'):
                    stats = fg.get_stats()
                    if isinstance(stats, dict):
                        graph_data['stats'].update(stats)
            
            elif web_gui_instance.brain and hasattr(web_gui_instance.brain, 'memory_manager'):
                mm = web_gui_instance.brain.memory_manager
                if hasattr(mm, 'get_graph_data'):
                    graph_data = mm.get_graph_data()
                elif hasattr(mm, 'nodes') and mm.nodes:
                    graph_data['nodes'] = [
                        {'id': n.get('id', i), 'label': n.get('content', '')[:50], 'type': 'memory'}
                        for i, n in enumerate(mm.nodes[:100])
                    ]
                    graph_data['stats']['total_nodes'] = len(mm.nodes)
        except Exception as e:
            logger.error(f"Error getting memory graph: {e}")

        return jsonify(graph_data)

    @app.route('/api/analytics')
    def api_analytics():
        """Get analytics data for dashboard."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        analytics = {
            'queries': 0,
            'avg_time': 0,
            'success_rate': 0,
            'cpu': 0,
            'memory': 0,
            'vram': 0,
            'dialogs': 0,
            'gaps': 0,
            'learned': 0,
            'cache_hit_rate': 0,
            'cache_utilization': 0,
            'activities': []
        }

        try:
            brain = web_gui_instance.brain
            
            rm = getattr(brain, 'resource_manager', None)
            if rm:
                try:
                    analytics['cpu'] = rm.get_cpu_usage() * 100
                    analytics['memory'] = rm.get_memory_usage() * 100
                    current = rm.get_current_metrics()
                    if isinstance(current, dict):
                        analytics['cpu'] = current.get('cpu_percent', analytics['cpu'])
                        analytics['memory'] = current.get('memory_percent', analytics['memory'])
                        analytics['vram'] = current.get('gpu_memory', 0)
                except Exception as e:
                    logger.debug(f"resource_manager error: {e}")
            
            if hasattr(brain, 'get_cache_stats'):
                try:
                    cache_stats = brain.get_cache_stats()
                    if cache_stats:
                        analytics['cache_hit_rate'] = cache_stats.get('hit_rate', 0.0)
                        analytics['cache_utilization'] = cache_stats.get('cache_utilization_percent', 0.0)
                except Exception as e:
                    logger.debug(f"get_cache_stats error: {e}")

            if hasattr(brain, 'self_dialog_learning') and brain.self_dialog_learning:
                try:
                    sdl = brain.self_dialog_learning
                    if hasattr(sdl, 'get_stats'):
                        stats = sdl.get_stats()
                        analytics['dialogs'] = stats.get('total_dialogs', 0)
                        analytics['gaps'] = stats.get('knowledge_gaps_identified', 0)
                        analytics['learned'] = stats.get('successful_learning', 0)
                except Exception as e:
                    logger.debug(f"self_dialog_learning error: {e}")

            fg = getattr(brain, 'fractal_graph_v2', None)
            if fg is None:
                fg = getattr(brain, 'components', {}).get('fractal_graph_v2')
            if fg is None and hasattr(brain, 'memory_manager'):
                fg = getattr(brain.memory_manager, 'fractal_graph_v2', None)
            
            if fg and hasattr(fg, 'get_stats'):
                try:
                    fg_stats = fg.get_stats()
                    analytics['fractal_nodes'] = fg_stats.get('total_nodes', 0)
                    analytics['fractal_edges'] = fg_stats.get('total_edges', 0)
                    analytics['fractal_groups'] = fg_stats.get('total_groups', 0)
                except Exception as e:
                    logger.debug(f"FractalGraphV2 stats error: {e}")

            if hasattr(brain, 'graph_curator') and brain.graph_curator:
                try:
                    curator = brain.graph_curator
                    if hasattr(curator, 'get_metrics'):
                        cur_metrics = curator.get_metrics()
                        analytics['curator_cycles'] = cur_metrics.get('cycles_completed', 0)
                        analytics['curator_state'] = cur_metrics.get('state', 'idle')
                        analytics['curator_next_run'] = cur_metrics.get('next_run', 0)
                except Exception as e:
                    logger.debug(f"GraphCurator metrics error: {e}")

            analytics['cpu'] = psutil.cpu_percent(interval=None) or analytics['cpu']
            analytics['memory'] = psutil.virtual_memory().percent or analytics['memory']
            
        except Exception as e:
            logger.error(f"Analytics error: {e}")

        return jsonify(analytics)

    @app.route('/api/learning')
    def api_learning():
        """Get learning data."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        brain = web_gui_instance.brain
        sdl = getattr(brain, 'self_dialog_learning', None)
        
        result = {
            'total': 0,
            'success': 0,
            'pending': 0,
            'dialogs': [],
            'opportunities': [],
            'recent_dialogs': []
        }
        
        if sdl:
            try:
                if hasattr(sdl, 'get_opportunities'):
                    result['opportunities'] = sdl.get_opportunities()
                if hasattr(sdl, 'get_recent_dialogs'):
                    result['recent_dialogs'] = sdl.get_recent_dialogs()
                if hasattr(sdl, 'get_stats'):
                    stats = sdl.get_stats()
                    result.update(stats)
            except Exception as e:
                logger.error(f"Learning error: {e}")
        
        return jsonify(result)

    @app.route('/api/self-dialog', methods=['GET', 'POST'])
    def api_self_dialog():
        """Get or create self-dialog."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        brain = web_gui_instance.brain
        sdl = getattr(brain, 'self_dialog_learning', None)
        
        if request.method == 'GET':
            if sdl and hasattr(sdl, 'get_current_dialog'):
                return jsonify(sdl.get_current_dialog())
            return jsonify({'error': 'SelfDialogLearning not available'}), 404
        
        elif request.method == 'POST':
            data = request.get_json() or {}
            query = data.get('query', '')
            response = data.get('response', '')
            
            if sdl and hasattr(sdl, 'create_dialog'):
                try:
                    dialog = sdl.create_dialog(query, response)
                    return jsonify({'success': True, 'dialog': dialog})
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
            
            return jsonify({'error': 'SelfDialogLearning not available'}), 404

    @app.route('/api/events/stream')
    def api_events_stream():
        """SSE stream for system events."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        def generate():
            while True:
                try:
                    events = web_gui_instance.get_pending_events()
                    for event in events:
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    time.sleep(0.5)
                except GeneratorExit:
                    break
                except Exception as e:
                    logger.error(f"Events stream error: {e}")
                    break

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    @app.route('/api/dashboard')
    def api_dashboard():
        """Get dashboard data."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        try:
            brain = web_gui_instance.brain
            
            dashboard = {
                'timestamp': time.time(),
                'system': {
                    'cpu': psutil.cpu_percent(interval=0.1),
                    'memory': psutil.virtual_memory().percent,
                    'disk': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
                },
                'brain': {
                    'running': getattr(brain, 'running', False),
                    'initialized': getattr(brain, 'initialized', False)
                },
                'sessions': len(getattr(web_gui_instance, 'sessions', {}))
            }
            
            fg = getattr(brain, 'fractal_graph_v2', None)
            if fg is None:
                fg = getattr(brain, 'components', {}).get('fractal_graph_v2')
            if fg and hasattr(fg, 'get_stats'):
                try:
                    stats = fg.get_stats()
                    dashboard['graph'] = {
                        'nodes': stats.get('total_nodes', 0),
                        'edges': stats.get('total_edges', 0)
                    }
                except:
                    pass
            
            return jsonify(dashboard)
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/generation-status', defaults={'command_id': None})
    @app.route('/api/generation-status/<command_id>')
    def api_generation_status(command_id):
        """Get generation status."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        brain = web_gui_instance.brain
        tracker = getattr(brain, 'generation_tracker', None)
        
        if not command_id:
            return jsonify({'error': 'command_id required'}), 400
        
        if tracker and hasattr(tracker, 'get_status'):
            status = tracker.get_status(command_id)
            return jsonify(status)
        
        return jsonify({'error': 'Tracker not available', 'command_id': command_id})

    logger.info("Analytics routes registered")
