"""Flask сервер для Web GUI ЕВА — расширенные обработчики (analytics, learning, settings, etc.)"""
import logging
from datetime import datetime

from flask import jsonify, request

logger = logging.getLogger("eva.webgui")

from .server_main import web_gui_instance, app


# ========================================================================
# Analytics
# ========================================================================

@app.route('/api/analytics')
def api_analytics():
    if not web_gui_instance:
        logger.warning("api_analytics: web_gui_instance is None")
        return jsonify({'error': 'not_initialized'})

    analytics = {
        'queries': 0, 'avg_time': 0, 'success_rate': 0,
        'cpu': 0, 'memory': 0, 'vram': 0,
        'dialogs': 0, 'gaps': 0, 'learned': 0,
        'cache_hit_rate': 0, 'cache_utilization': 0,
        'activities': []
    }

    try:
        if web_gui_instance.brain:
            if hasattr(web_gui_instance.brain, 'get_resource_snapshot'):
                snapshot = web_gui_instance.brain.get_resource_snapshot()
                analytics['cpu'] = snapshot.get('cpu_usage', snapshot.get('cpu_percent', 0))
                analytics['memory'] = snapshot.get('memory_usage', snapshot.get('memory_percent', 0))
                analytics['vram'] = snapshot.get('gpu_memory', snapshot.get('gpu_memory_percent', 0))

            if hasattr(web_gui_instance.brain, 'get_cache_stats'):
                cache_stats = web_gui_instance.brain.get_cache_stats()
                analytics['cache_hit_rate'] = cache_stats.get('hit_rate', 0.0)
                analytics['cache_utilization'] = cache_stats.get('cache_utilization_percent', 0.0)

            if hasattr(web_gui_instance.brain, 'self_dialog_learning'):
                sdl = web_gui_instance.brain.self_dialog_learning
                if hasattr(sdl, 'get_stats'):
                    stats = sdl.get_stats()
                    analytics['dialogs'] = stats.get('total_dialogs', 0)
                    analytics['gaps'] = stats.get('knowledge_gaps_identified', 0)
                    analytics['learned'] = stats.get('successful_learning', 0)

            if hasattr(web_gui_instance.brain, 'performance_analyzer'):
                pa = web_gui_instance.brain.performance_analyzer
                if hasattr(pa, 'analyze_performance'):
                    try:
                        perf_data = pa.analyze_performance()
                        analytics['queries'] = perf_data.get('total_queries', 0)
                        analytics['avg_time'] = perf_data.get('avg_query_time_ms', 0)
                        analytics['success_rate'] = perf_data.get('success_rate', 0)
                    except Exception as e:
                        logger.debug(f"PerformanceAnalyzer error: {e}")

            try:
                import psutil
                analytics['cpu'] = psutil.cpu_percent(interval=None)
                analytics['memory'] = psutil.virtual_memory().percent
            except ImportError:
                pass

            activities = []
            if hasattr(web_gui_instance.brain, 'memory_manager') and web_gui_instance.brain.memory_manager:
                mm = web_gui_instance.brain.memory_manager
                try:
                    if hasattr(mm, 'get_recent_interactions'):
                        interactions = mm.get_recent_interactions(limit=100)
                        count = len(interactions) if interactions else 0
                    elif hasattr(mm, 'nodes') and mm.nodes:
                        count = len(mm.nodes)
                    elif hasattr(mm, 'get_stats'):
                        stats = mm.get_stats()
                        count = stats.get('total_nodes', 0)
                    else:
                        count = 0

                    if count > 0:
                        activities.append({
                            'icon': 'memory',
                            'title': f'Память: {count} записей',
                            'time': 'Сейчас'
                        })
                except Exception as e:
                    logger.debug(f"Memory activity error: {e}")

            if hasattr(web_gui_instance.brain, 'self_dialog_learning'):
                sdl = web_gui_instance.brain.self_dialog_learning
                if hasattr(sdl, 'stats'):
                    stats = sdl.stats
                    if stats.get('total_dialogs', 0) > 0:
                        activities.append({
                            'icon': 'learn',
                            'title': f'Диалогов обучения: {stats["total_dialogs"]}',
                            'time': 'Сегодня'
                        })

            analytics['activities'] = activities[:10]

    except Exception as e:
        logger.error(f"Error getting analytics: {e}")

    if not web_gui_instance or not web_gui_instance.brain:
        analytics['dialogs'] = 0
        analytics['gaps'] = 0
        analytics['learned'] = 0

    return jsonify(analytics)


# ========================================================================
# Learning
# ========================================================================

@app.route('/api/learning')
def api_learning():
    if not web_gui_instance:
        logger.warning("api_learning: web_gui_instance is None")
        return jsonify({'error': 'not_initialized'})

    learning = {
        'opportunities': [], 'total': 0, 'success': 0,
        'pending': 0, 'dialogs': [], 'recent_dialogs': []
    }

    try:
        if web_gui_instance.brain:
            if hasattr(web_gui_instance.brain, 'self_dialog_learning'):
                sdl = web_gui_instance.brain.self_dialog_learning

                if hasattr(sdl, '_get_learning_opportunities'):
                    try:
                        opportunities = sdl._get_learning_opportunities()
                        for op in opportunities[:10]:
                            priority = op.get('priority', 0.5)
                            priority_level = 'high' if priority >= 0.7 else 'medium' if priority >= 0.4 else 'low'
                            learning['opportunities'].append({
                                'concept': op.get('concept', 'Unknown'),
                                'type': op.get('opportunity_type', 'expansion'),
                                'priority': priority,
                                'priority_level': priority_level,
                                'domain': op.get('domain', 'general')
                            })
                    except Exception as e:
                        logger.debug(f"Error getting opportunities: {e}")

                if hasattr(sdl, 'get_stats'):
                    stats = sdl.get_stats()
                    learning['total'] = stats.get('total_dialogs', 0)
                    learning['success'] = stats.get('successful_learning', 0)

                if hasattr(sdl, 'get_recent_learning'):
                    try:
                        recent = sdl.get_recent_learning(limit=5)
                        for d in recent:
                            learning['recent_dialogs'].append({
                                'topic': d.get('topic', '')[:50],
                                'outcome': d.get('outcome', 'unknown'),
                                'gaps': d.get('gaps', [])
                            })
                    except Exception as e:
                        logger.debug(f"Error getting recent dialogs: {e}")

            learning['pending'] = len(learning['opportunities'])

    except Exception as e:
        logger.error(f"Error getting learning data: {e}")

    if not web_gui_instance or not web_gui_instance.brain:
        learning['total'] = 0
        learning['success'] = 0
        learning['pending'] = 0

    return jsonify(learning)


# ========================================================================
# Settings
# ========================================================================

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})

    if request.method == 'GET':
        settings = {
            'auto_learning': True, 'sre_enabled': True, 'memory_enabled': True,
            'dark_theme': True, 'sound_enabled': False,
            'model_name': 'Qwen2.5-0.5B GGUF', 'version': '1.0.0'
        }

        try:
            if web_gui_instance.brain:
                if hasattr(web_gui_instance.brain, 'self_dialog_learning'):
                    sdl = web_gui_instance.brain.self_dialog_learning
                    settings['auto_learning'] = getattr(sdl, 'auto_execute_enabled', True)

                if hasattr(web_gui_instance.brain, 'llama_cpp_deployment'):
                    settings['model_name'] = 'Qwen2.5-0.5B GGUF'
                elif hasattr(web_gui_instance.brain, 'qwen_model_manager'):
                    settings['model_name'] = 'Qwen3.5-0.8B'
        except Exception as e:
            logger.debug(f"Error getting settings: {e}")

        return jsonify(settings)

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    try:
        if web_gui_instance.brain:
            if 'auto_learning' in data and hasattr(web_gui_instance.brain, 'self_dialog_learning'):
                sdl = web_gui_instance.brain.self_dialog_learning
                if hasattr(sdl, 'auto_execute_enabled'):
                    sdl.auto_execute_enabled = data['auto_learning']

        return jsonify({'status': 'ok', 'updated': list(data.keys())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================================================================
# Documents
# ========================================================================

@app.route('/api/documents', methods=['GET'])
def api_documents():
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})

    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'session_id required'}), 400

    documents = []

    try:
        if hasattr(web_gui_instance.brain, 'hybrid_cache') and web_gui_instance.brain.hybrid_cache:
            docs = web_gui_instance.brain.hybrid_cache.get_session_documents(session_id)
            documents = docs

        session = web_gui_instance.session_manager.get_session(session_id)
        if session:
            context_nodes = session.get('context_nodes', [])
            file_docs = []
            for node in context_nodes:
                if isinstance(node, dict) and node.get('file_data'):
                    file_docs.append({
                        'file_id': node['file_data'].get('file_id', ''),
                        'filename': node['file_data'].get('filename', ''),
                        'timestamp': node.get('timestamp', '')
                    })
            if file_docs and not documents:
                documents = file_docs
    except Exception as e:
        logger.debug(f"Error getting documents: {e}")

    return jsonify({'documents': documents})


# ========================================================================
# Knowledge Graph
# ========================================================================

@app.route('/api/knowledge-graph', methods=['GET', 'POST'])
def api_knowledge_graph():
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})

    if request.method == 'GET':
        action = request.args.get('action', 'get')

        try:
            kg = getattr(web_gui_instance.brain, 'knowledge_graph', None)
            if kg:
                if action == 'get':
                    nodes = []
                    if hasattr(kg, 'nodes'):
                        for n in kg.nodes[:50]:
                            nodes.append({
                                'id': getattr(n, 'id', ''),
                                'name': getattr(n, 'name', '')[:50],
                                'content': getattr(n, 'content', '')[:100]
                            })
                    return jsonify({'nodes': nodes, 'total': len(nodes)})

                elif action == 'search':
                    query = request.args.get('query', '')
                    if hasattr(kg, 'search_nodes'):
                        results = kg.search_nodes(query, limit=10)
                        return jsonify({'results': results})
            return jsonify({'nodes': [], 'total': 0})
        except Exception as e:
            logger.debug(f"Error accessing knowledge graph: {e}")
            return jsonify({'error': str(e)}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    try:
        kg = getattr(web_gui_instance.brain, 'knowledge_graph', None)
        if kg and hasattr(kg, 'add_node'):
            name = data.get('name', '')
            content = data.get('content', '')
            node_type = data.get('type', 'concept')

            if name or content:
                kg.add_node(name=name, content=content, node_type=node_type)
                return jsonify({'status': 'ok', 'message': 'Node added'})

        return jsonify({'error': 'Knowledge graph not available'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================================================================
# Cache Stats
# ========================================================================

@app.route('/api/cache-stats')
def api_cache_stats():
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})

    stats = {'hybrid_cache': {}, 'search_cache': {}, 'memory': {}}

    try:
        if hasattr(web_gui_instance.brain, 'hybrid_cache') and web_gui_instance.brain.hybrid_cache:
            hc = web_gui_instance.brain.hybrid_cache
            if hasattr(hc, 'get_cache_stats'):
                stats['hybrid_cache'] = hc.get_cache_stats()
            if hasattr(hc, 'get_search_cache_stats'):
                stats['search_cache'] = hc.get_search_cache_stats()

        if hasattr(web_gui_instance.brain, 'memory_manager'):
            mm = web_gui_instance.brain.memory_manager
            if hasattr(mm, 'nodes'):
                stats['memory']['total_nodes'] = len(mm.nodes)
    except Exception as e:
        logger.debug(f"Error getting cache stats: {e}")

    return jsonify(stats)


# ========================================================================
# System
# ========================================================================

@app.route('/api/system')
def api_system():
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})

    system_info = {
        'version': '1.0.0',
        'model': 'Qwen2.5-0.5B GGUF',
        'qwen_ready': False, 'llama_cpp_ready': False,
        'modules': {
            'contradiction': False, 'ethics': False,
            'web_search': False, 'knowledge_graph': False
        },
        'features': {
            'self_learning': False, 'knowledge_graph': False, 'web_search': False
        }
    }

    try:
        if web_gui_instance.brain:
            brain = web_gui_instance.brain

            system_info['qwen_ready'] = getattr(brain, 'qwen_ready', False)
            system_info['llama_cpp_ready'] = getattr(brain, 'llama_cpp_ready', False)

            if hasattr(brain, 'llama_cpp_deployment'):
                system_info['model'] = 'Qwen2.5-0.5B GGUF'
            elif hasattr(brain, 'qwen_model_manager'):
                system_info['model'] = 'Qwen3.5-0.8B'

            system_info['modules']['contradiction'] = hasattr(brain, 'contradiction_manager')
            system_info['modules']['ethics'] = hasattr(brain, 'ethics_framework')
            system_info['modules']['web_search'] = hasattr(brain, 'web_search_engine')
            system_info['modules']['knowledge_graph'] = hasattr(brain, 'knowledge_graph')

            system_info['features']['self_learning'] = hasattr(brain, 'self_dialog_learning')
            system_info['features']['knowledge_graph'] = hasattr(brain, 'knowledge_graph')
            system_info['features']['web_search'] = hasattr(brain, 'web_search_engine')
    except Exception as e:
        logger.debug(f"Error getting system info: {e}")

    return jsonify(system_info)
