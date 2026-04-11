"""
Analytics routes: memory-graph, analytics, learning, events, dashboard
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
            logger.warning("api_analytics: web_gui_instance is None")
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
            'activities': [],
            # Additional fields expected by frontend
            'fractal_nodes': 0,
            'fractal_edges': 0,
            'fractal_groups': 0,
            'curator_cycles': 0,
            'curator_state': 'idle',
            'curator_next_run': 0,
            'tavily_requests': 0,
            'tavily_responses': 0,
            'web_searches': 0,
            'web_cache_hits': 0,
            'wiki_queries': 0,
            'wiki_articles': 0,
            'wiki_cached': 0
        }

        try:
            logger.debug(f"api_handler: brain = {web_gui_instance.brain is not None}")

            if web_gui_instance.brain:
                brain = web_gui_instance.brain
                
                # Resource Manager metrics
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
                elif hasattr(brain, 'get_resource_snapshot'):
                    try:
                        snapshot = brain.get_resource_snapshot()
                        if snapshot:
                            analytics['cpu'] = snapshot.get('cpu_usage', snapshot.get('cpu_percent', 0))
                            analytics['memory'] = snapshot.get('memory_usage', snapshot.get('memory_percent', 0))
                            analytics['vram'] = snapshot.get('gpu_memory', snapshot.get('gpu_memory_percent', 0))
                    except Exception as e:
                        logger.debug(f"get_resource_snapshot error: {e}")

                # Cache stats
                if hasattr(brain, 'get_cache_stats'):
                    try:
                        cache_stats = brain.get_cache_stats()
                        if cache_stats:
                            analytics['cache_hit_rate'] = cache_stats.get('hit_rate', 0.0)
                            analytics['cache_utilization'] = cache_stats.get('cache_utilization_percent', 0.0)
                    except Exception as e:
                        logger.debug(f"get_cache_stats error: {e}")

                # Self Dialog Learning stats
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

                # FractalGraphV2 metrics
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

                # GraphCurator metrics
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

                # Process metrics
                if hasattr(brain, '_process_metrics') and brain._process_metrics:
                    try:
                        pm = brain._process_metrics
                        analytics['queries'] = pm.get('total_queries', 0)
                        total = pm.get('total_queries', 1)
                        success = pm.get('successful_queries', 0)
                        analytics['success_rate'] = success / total if total > 0 else 0
                        analytics['avg_time'] = pm.get('avg_generation_time', 0) * 1000  # ms
                    except Exception as e:
                        logger.debug(f"ProcessTrackerMixin error: {e}")

                # Web Search / Tavily metrics
                try:
                    web_search = getattr(brain, 'web_search_engine', None)
                    if web_search and hasattr(web_search, 'stats'):
                        analytics['tavily_requests'] = web_search.stats.get('tavily_requests', 0)
                        analytics['tavily_responses'] = web_search.stats.get('tavily_responses', 0)
                        analytics['web_searches'] = web_search.stats.get('searches_performed', 0)
                        analytics['web_cache_hits'] = web_search.stats.get('cache_hits', 0)
                except Exception as e:
                    logger.debug(f"WebSearch stats error: {e}")
                
                # Wikipedia metrics
                try:
                    analytics['wiki_queries'] = analytics.get('web_searches', 0)
                    analytics['wiki_articles'] = analytics.get('fractal_nodes', 0)
                    analytics['wiki_cached'] = analytics.get('web_cache_hits', 0)
                except Exception as e:
                    logger.debug(f"Wiki stats error: {e}")

                # Activities list
                activities = []
                
                if hasattr(brain, 'memory_manager') and brain.memory_manager:
                    try:
                        mm = brain.memory_manager
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

                if hasattr(brain, 'self_dialog_learning') and brain.self_dialog_learning:
                    try:
                        sdl = brain.self_dialog_learning
                        if hasattr(sdl, 'stats'):
                            stats = sdl.stats
                            if stats.get('total_dialogs', 0) > 0:
                                activities.append({
                                    'icon': 'learn',
                                    'title': f'Диалогов обучения: {stats["total_dialogs"]}',
                                    'time': 'Сегодня'
                                })
                    except Exception as e:
                        logger.debug(f"self_dialog_learning stats error: {e}")

                analytics['activities'] = activities[:10]

            # System metrics via psutil
            try:
                analytics['cpu'] = psutil.cpu_percent(interval=None) or analytics['cpu']
                analytics['memory'] = psutil.virtual_memory().percent or analytics['memory']
            except Exception:
                pass
                
        except Exception as e:
            logger.error(f"Analytics error: {e}")

        return jsonify(analytics)

    @app.route('/api/learning')
    def api_learning():
        """Get learning data."""
        if not web_gui_instance:
            logger.warning("api_learning: web_gui_instance is None")
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        learning = {
            'opportunities': [],
            'total': 0,
            'success': 0,
            'pending': 0,
            'dialogs': [],
            'recent_dialogs': []
        }

        try:
            # Сначала пробуем получить из bridge кэша
            if hasattr(web_gui_instance, 'bridge') and web_gui_instance.bridge:
                cached = web_gui_instance.bridge.get_cached_learning_stats()
                if cached and cached.get('total', 0) > 0:
                    logger.info("api_learning: returning cached data")
                    return jsonify(cached)
            
            brain = web_gui_instance.brain
            
            if not brain:
                logger.warning("api_learning: brain is None")
                return jsonify(learning)
            
            logger.info(f"api_learning: brain type = {type(brain).__name__}")
            logger.info(f"api_learning: has self_dialog_learning = {hasattr(brain, 'self_dialog_learning')}")
            
            sdl = None
            
            # 1. Прямой атрибут brain
            if hasattr(brain, 'self_dialog_learning'):
                sdl = brain.self_dialog_learning
            
            # 2. Через components
            if not sdl and hasattr(brain, 'components') and brain.components:
                sdl = brain.components.get('self_dialog_learning')
                if sdl:
                    logger.info("api_learning: SDL from components")
            
            if sdl:
                logger.info(f"api_learning: SDL found, type = {type(sdl).__name__}")
                
                # Stats
                if hasattr(sdl, 'get_stats'):
                    stats = sdl.get_stats()
                    learning['total'] = stats.get('total_dialogs', 0)
                    learning['success'] = stats.get('successful_learning', 0)
                    learning['pending'] = stats.get('opportunities_executed', 0)
                    logger.info(f"api_learning: stats = {stats}")
                
                # Opportunities
                if hasattr(sdl, '_get_learning_opportunities'):
                    try:
                        opportunities = sdl._get_learning_opportunities()
                        logger.info(f"api_learning: {len(opportunities)} opportunities")
                        for op in opportunities[:10]:
                            learning['opportunities'].append({
                                'concept': op.get('concept', 'Unknown'),
                                'type': op.get('opportunity_type', 'expansion'),
                                'priority': op.get('priority', 0.5),
                                'priority_level': 'high' if op.get('priority', 0) >= 0.7 else 'medium' if op.get('priority', 0) >= 0.4 else 'low',
                                'domain': op.get('domain', 'general')
                            })
                    except Exception as e:
                        logger.error(f"api_learning: opportunities error: {e}")

            logger.info(f"api_learning: returning {learning}")
            return jsonify(learning)
            
        except Exception as e:
            logger.error(f"Error getting learning data: {e}")
            return jsonify({
                'opportunities': [],
                'total': 0,
                'success': 0,
                'pending': 0,
                'dialogs': [],
                'recent_dialogs': []
            })

    @app.route('/api/self-dialog', methods=['GET', 'POST'])
    def api_self_dialog():
        """Get or create self-dialog."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        brain = web_gui_instance.brain
        sdl = getattr(brain, 'self_dialog_learning', None)
        
        if request.method == 'GET':
            status = {
                'enabled': False,
                'running': False,
                'total_dialogs': 0,
                'successful': 0,
                'failed': 0,
                'recent_topics': []
            }

            try:
                if brain and hasattr(brain, 'self_dialog_learning'):
                    sdl = brain.self_dialog_learning
                    status['enabled'] = getattr(sdl, 'enabled', False)
                    status['running'] = getattr(sdl, 'running', False)

                    if hasattr(sdl, 'get_stats'):
                        stats = sdl.get_stats()
                        status['total_dialogs'] = stats.get('total_dialogs', 0)
                        status['successful'] = stats.get('successful', 0)
                        status['failed'] = stats.get('failed', 0)

                    if hasattr(sdl, 'get_recent_topics'):
                        status['recent_topics'] = sdl.get_recent_topics(5)
            except Exception as e:
                logger.error(f"Error getting self-dialog status: {e}")

            return jsonify(status)
        
        elif request.method == 'POST':
            data = request.json or {}
            topic = data.get('topic', '')

            try:
                if brain and hasattr(brain, 'self_dialog_learning'):
                    sdl = brain.self_dialog_learning
                    if hasattr(sdl, 'create_dialog'):
                        result = sdl.create_dialog(topic=topic if topic else None)
                        return jsonify({
                            'status': 'success',
                            'dialog_id': result.get('dialog_id', ''),
                            'topic': result.get('topic', topic)
                        })
                    else:
                        return jsonify({'error': 'create_dialog not available'}), 500
                else:
                    return jsonify({'error': 'Self-dialog learning not available'}), 500
            except Exception as e:
                logger.error(f"Error triggering self-dialog: {e}")
                return jsonify({'error': str(e)}), 500

    @app.route('/api/events/stream')
    def api_events_stream():
        """SSE stream for system events."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        def event_stream():
            """Генератор событий с подпиской на EventBus."""
            if not web_gui_instance or not web_gui_instance.brain:
                yield 'event: error\ndata: {"error": "Brain not available"}\n\n'
                return

            brain = web_gui_instance.brain
            event_bus = getattr(brain, 'event_bus', None) or getattr(brain, '_new_event_bus', None)

            if not event_bus:
                yield 'event: error\ndata: {"error": "EventBus not available"}\n\n'
                return

            import queue
            msg_queue = queue.Queue()

            def handler(event):
                try:
                    data = event.data if hasattr(event, 'data') else {}
                    event_type = event.event_type if hasattr(event, 'event_type') else ''
                    msg_queue.put({'event_type': event_type, 'data': data})
                except Exception:
                    pass

            # Список событий для подписки
            pipeline_events = [
                'pipeline.start', 'pipeline.model_a.start', 'pipeline.model_a.complete',
                'pipeline.model_b.start', 'pipeline.model_b.complete',
                'pipeline.model_c.start', 'pipeline.model_c.complete',
                'pipeline.complete', 'pipeline.failed',
                'generation.progress', 'generation.started', 'generation.completed',
                'generation.failed', 'generation.timeout',
                'curator.started', 'curator.completed', 'curator.error',
                'curator.graph_optimized', 'curator.knowledge_extracted', 'curator.cleanup_done',
                'curator.metrics_updated',
                'self_dialog.started', 'self_dialog.completed', 'self_dialog.learning',
            ]

            subscriptions = {}
            try:
                for evt_name in pipeline_events:
                    try:
                        sub_id = event_bus.subscribe(evt_name, handler)
                        subscriptions[evt_name] = sub_id
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                yield ': connected\n\n'
                while True:
                    try:
                        msg = msg_queue.get(timeout=1)
                        event_name = msg['event_type']
                        data = msg['data']
                        yield f'event: {event_name}\ndata: {json.dumps(data, default=str)}\n\n'
                    except queue.Empty:
                        yield ': heartbeat\n\n'
            except GeneratorExit:
                pass
            finally:
                for sub_id in subscriptions.values():
                    try:
                        event_bus.unsubscribe(sub_id)
                    except Exception:
                        pass

        return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

    @app.route('/api/generation-status', defaults={'command_id': None})
    @app.route('/api/generation-status/<command_id>')
    def api_generation_status(command_id):
        """Get generation status."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        try:
            brain = web_gui_instance.brain
            if not brain:
                return jsonify({'error': 'Brain not available'}), 500

            if hasattr(brain, '_handle_generation_status'):
                result = brain._handle_generation_status(command_id)
                return jsonify(result)

            tracker = getattr(brain, 'generation_tracker', None)
            if not tracker:
                return jsonify({'error': 'GenerationTracker not initialized'}), 500

            if command_id:
                status = tracker.get_status(command_id)
                if status:
                    return jsonify(status)
                return jsonify({'error': f'Command {command_id} not found'}), 404
            return jsonify({'active_generations': tracker.get_all_active()})
        except Exception as e:
            logger.error(f"Error getting generation status: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/dashboard')
    def api_dashboard():
        """Dashboard data endpoint."""
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
                'sessions': len(getattr(web_gui_instance.session_manager, 'sessions', {}))
            }
            
            # Metrics from registry
            try:
                from eva_ai.core.metrics import get_metrics_registry, get_eva_metrics
                registry = get_metrics_registry()
                eva_metrics = get_eva_metrics()
                
                dashboard['summary'] = {
                    'cache_hit_rate': eva_metrics.get_cache_hit_rate(),
                    'generation_stats': eva_metrics.get_generation_stats(),
                }
                dashboard['charts'] = {
                    'request_duration': registry.histogram('eva_request_duration_seconds').get_stats() if hasattr(registry, 'histogram') else {},
                    'generation_duration': registry.histogram('eva_generation_duration_seconds').get_stats() if hasattr(registry, 'histogram') else {},
                    'search_duration': registry.histogram('eva_search_duration_seconds').get_stats() if hasattr(registry, 'histogram') else {},
                }
                dashboard['counters'] = {
                    'requests_total': registry.counter('eva_requests_total').get() if hasattr(registry, 'counter') else 0,
                    'generations_total': registry.counter('eva_generations_total').get() if hasattr(registry, 'counter') else 0,
                    'errors_total': registry.counter('eva_errors_total').get() if hasattr(registry, 'counter') else 0,
                    'cache_hits': registry.counter('eva_cache_hits_total').get() if hasattr(registry, 'counter') else 0,
                    'cache_misses': registry.counter('eva_cache_misses_total').get() if hasattr(registry, 'counter') else 0,
                }
                dashboard['gauges'] = {
                    'cpu_percent': registry.gauge('system_cpu_percent').get() if hasattr(registry, 'gauge') else 0,
                    'memory_percent': registry.gauge('system_memory_percent').get() if hasattr(registry, 'gauge') else 0,
                }
            except Exception as e:
                logger.debug(f"Dashboard metrics error: {e}")
            
            # Graph stats
            fg = getattr(brain, 'fractal_graph_v2', None)
            if fg is None:
                fg = getattr(brain, 'components', {}).get('fractal_graph_v2')
            if fg and hasattr(fg, 'get_stats'):
                try:
                    stats = fg.get_stats()
                    dashboard['graph'] = {
                        'nodes': stats.get('total_nodes', 0),
                        'edges': stats.get('total_edges', 0),
                        'groups': stats.get('total_groups', 0)
                    }
                except:
                    pass
            
            return jsonify(dashboard)
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            return jsonify({'error': str(e)}), 500

    logger.info("Analytics routes registered")
