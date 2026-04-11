"""
Core routes для Web GUI
System, status, debug, health, metrics, dashboard
"""
import logging
import time
import json
from datetime import datetime
from flask import render_template, jsonify, request, Response
from eva_ai.core.api_compat import API_VERSION, API_PREFIX, api_version

from .server_routes_utils import check_brain_initialized, get_brain_components

logger = logging.getLogger("eva_ai.webgui")


def register_core_routes(app, web_gui_instance):
    """Регистрирует core роуты"""
    
    @app.route('/favicon.ico')
    def favicon():
        return '', 204

    @app.route('/api/system', methods=['GET'])
    def api_system():
        """System information endpoint."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        system_info = {
            'version': '3.1',
            'model': 'CogniFlex (EVA-Ai)',
            'modules': [],
            'features': {
                'memory': True,
                'learning': True,
                'ethics': True,
                'websearch': True,
                'contradiction': True,
                'concepts': True,
                'self_dialog': True
            }
        }
        
        if web_gui_instance.brain and hasattr(web_gui_instance.brain, 'components'):
            system_info['modules'] = list(web_gui_instance.brain.components.keys())
        
        return jsonify(system_info)
    
    @app.route('/')
    def index():
        # Add cache-busting for JS/CSS
        return render_template('index.html', _v=time.time())
    
    @app.route('/api/debug/test')
    def api_debug_test():
        """Simple test endpoint."""
        return jsonify({
            'status': 'ok',
            'time': datetime.now().isoformat(),
            'web_gui_instance': str(web_gui_instance is not None),
            'users': list(web_gui_instance.auth_manager.users.keys()) if web_gui_instance else []
        })
    
    @app.route('/api/status')
    def api_status():
        """System status endpoint for frontend."""
        if not web_gui_instance:
            return jsonify({'status': 'not_initialized'})
        
        status = {
            'status': 'active',
            'sessions_count': len(web_gui_instance.session_manager.sessions) if web_gui_instance.session_manager else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        if web_gui_instance.brain:
            status['brain_connected'] = True
            if hasattr(web_gui_instance.brain, 'running'):
                status['brain_running'] = web_gui_instance.brain.running
            if hasattr(web_gui_instance.brain, 'components'):
                status['components'] = len(web_gui_instance.brain.components)
            if hasattr(web_gui_instance.brain, 'get_state'):
                status['brain_state'] = str(web_gui_instance.brain.get_state())
        else:
            status['brain_connected'] = False
            status['brain_running'] = False
        
        return jsonify(status)
    
    @app.route('/api/debug/deferred')
    def api_debug_deferred():
        """Debug endpoint - получить данные из системы отложенных команд."""
        if not web_gui_instance or not web_gui_instance.brain:
            return jsonify({'error': 'Brain не инициализирован'}), 500
        
        brain = web_gui_instance.brain
        
        result = {
            'available': False,
            'deferred_system': None,
            'event_bus': None,
            'event_system': None,
            'brain_components': get_brain_components(web_gui_instance)
        }
        
        # DeferredCommandSystem
        if hasattr(brain, 'deferred_system') and brain.deferred_system:
            deferred = brain.deferred_system
            result['available'] = True
            
            commands_data = {}
            try:
                if hasattr(deferred, 'commands') and hasattr(deferred, 'commands_lock'):
                    with deferred.commands_lock:
                        for cmd_id, cmd in deferred.commands.items():
                            commands_data[cmd_id] = {
                                'id': cmd.id,
                                'status': cmd.status.value if hasattr(cmd, 'status') else 'unknown',
                                'priority': cmd.priority.name if hasattr(cmd, 'priority') else 'unknown',
                                'attempts': cmd.attempts if hasattr(cmd, 'attempts') else 0,
                                'created_at': cmd.created_at if hasattr(cmd, 'created_at') else 0
                            }
            except Exception as e:
                logger.error("Error reading deferred commands: {}".format(e))
                commands_data = {'error': str(e)}
            
            result['deferred_system'] = {
                'type': type(deferred).__name__,
                'commands': commands_data,
                'commands_count': len(commands_data),
                'recovery_strategies': list(getattr(deferred, 'recovery_strategies', {}).keys()),
                'health_checks': list(getattr(deferred, 'module_health_checks', {}).keys()),
                'stats': getattr(deferred, 'stats', {}),
                'running': getattr(deferred, 'running', False),
                'shutting_down': getattr(deferred, '_shutting_down', False)
            }
        
        # EventBus
        if hasattr(brain, 'event_bus') and brain.event_bus:
            eb = brain.event_bus
            result['event_bus'] = {
                'type': type(eb).__name__,
                'running': getattr(eb, '_running', False),
                'stats': getattr(eb, '_stats', {}),
                'subscribers': {}
            }
            
            try:
                if hasattr(eb, '_subscribers') and hasattr(eb, '_lock'):
                    with eb._lock:
                        for event_type, subs in eb._subscribers.items():
                            result['event_bus']['subscribers'][event_type] = len(subs)
            except Exception as e:
                logger.error("Error reading event bus subscribers: {}".format(e))
        
        # Old EventSystem
        if hasattr(brain, 'events') and brain.events:
            result['event_system'] = {
                'type': type(brain.events).__name__,
                'available': True
            }
        
        return jsonify(result)
    
    @app.route('/api/debug/events')
    def api_debug_events():
        """Debug endpoint - получить историю событий из EventBus."""
        if not web_gui_instance or not web_gui_instance.brain:
            return jsonify({'error': 'Brain не инициализирован'}), 500
        
        brain = web_gui_instance.brain
        result = {'events': [], 'event_bus_stats': {}}
        
        if hasattr(brain, 'event_bus') and brain.event_bus:
            eb = brain.event_bus
            
            try:
                if hasattr(eb, '_event_history') and hasattr(eb, '_lock'):
                    with eb._lock:
                        history = eb._event_history[-50:]
                        for event in history:
                            result['events'].append({
                                'event_type': event.event_type,
                                'source': event.source,
                                'timestamp': event.timestamp if hasattr(event, 'timestamp') else 0,
                                'data': event.data
                            })
            except Exception as e:
                logger.error("Error reading event history: {}".format(e))
            
            if hasattr(eb, '_stats'):
                result['event_bus_stats'] = eb._stats
        
        return jsonify(result)
    
    @app.route('/api/health')
    def api_health():
        """Health check endpoint."""
        if not web_gui_instance:
            return jsonify({'status': 'not_initialized'}), 503
        
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Check brain
        if web_gui_instance.brain:
            health['checks']['brain'] = 'ok'
            if hasattr(web_gui_instance.brain, 'running'):
                health['checks']['brain_running'] = web_gui_instance.brain.running
        else:
            health['checks']['brain'] = 'not_available'
        
        # Check session manager
        if web_gui_instance.session_manager:
            health['checks']['sessions'] = 'ok'
        else:
            health['checks']['sessions'] = 'not_available'
        
        # Check auth manager
        if web_gui_instance.auth_manager:
            health['checks']['auth'] = 'ok'
        else:
            health['checks']['auth'] = 'not_available'
        
        # Overall status
        if all(v == 'ok' or v == True for v in health['checks'].values() if isinstance(v, (str, bool))):
            health['status'] = 'healthy'
        else:
            health['status'] = 'degraded'
        
        return jsonify(health)
    
    @app.route('/api/health/detailed')
    def api_health_detailed():
        """Detailed health check with component statuses."""
        if not web_gui_instance or not web_gui_instance.brain:
            return jsonify({'status': 'not_initialized', 'checks': {}}), 503
        
        brain = web_gui_instance.brain
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'components': {}
        }
        
        # Check all brain components
        components_to_check = [
            'memory_manager', 'self_dialog_learning', 'hybrid_cache',
            'knowledge_graph', 'web_search_engine', 'contradiction_manager',
            'concept_extractor', 'concept_miner', 'graph_curator',
            'two_model_pipeline', 'deferred_system', 'event_bus'
        ]
        
        for comp in components_to_check:
            if hasattr(brain, comp):
                obj = getattr(brain, comp)
                health['components'][comp] = {
                    'available': obj is not None,
                    'type': type(obj).__name__ if obj else None
                }
                if obj is not None:
                    # Check if component has running/is_running attribute
                    if hasattr(obj, 'running'):
                        health['components'][comp]['running'] = obj.running
                    elif hasattr(obj, 'is_running'):
                        health['components'][comp]['running'] = obj.is_running
            else:
                health['components'][comp] = {'available': False}
        
        # Overall status based on critical components
        critical_components = ['memory_manager', 'knowledge_graph']
        critical_ok = all(
            health['components'].get(comp, {}).get('available', False)
            for comp in critical_components
        )
        
        if critical_ok:
            health['status'] = 'healthy'
        else:
            health['status'] = 'degraded'
        
        return jsonify(health)
    
    @app.route('/api/metrics')
    def api_metrics():
        """Metrics endpoint for monitoring."""
        if not web_gui_instance or not web_gui_instance.brain:
            return jsonify({'error': 'Brain не инициализирован'}), 500
        
        brain = web_gui_instance.brain
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': {},
            'components': {},
            'graph': {},
            'learning': {}
        }
        
        # System metrics
        try:
            import psutil
            metrics['system'] = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else None
            }
        except Exception as e:
            logger.debug(f"Error getting system metrics: {e}")
        
        # Component metrics
        if hasattr(brain, 'metrics_manager') and brain.metrics_manager:
            try:
                mm = brain.metrics_manager
                metrics['components'] = {
                    'total_generations': getattr(mm, 'total_generations', 0),
                    'failed_generations': getattr(mm, 'failed_generations', 0),
                    'avg_response_time': getattr(mm, 'avg_response_time', 0)
                }
            except Exception as e:
                logger.debug(f"Error getting metrics manager data: {e}")
        
        # Graph metrics
        if hasattr(brain, 'fractal_graph_v2') and brain.fractal_graph_v2:
            try:
                fg = brain.fractal_graph_v2
                if hasattr(fg, 'storage'):
                    storage = fg.storage
                    metrics['graph'] = {
                        'total_nodes': len(storage.nodes) if hasattr(storage, 'nodes') else 0,
                        'total_edges': len(storage.edges) if hasattr(storage, 'edges') else 0,
                        'total_groups': len(storage.semantic_groups) if hasattr(storage, 'semantic_groups') else 0
                    }
            except Exception as e:
                logger.debug(f"Error getting graph metrics: {e}")
        
        # Learning metrics
        if hasattr(brain, 'self_dialog_learning') and brain.self_dialog_learning:
            try:
                sdl = brain.self_dialog_learning
                metrics['learning'] = {
                    'total_dialogs': getattr(sdl, 'stats', {}).get('total_dialogs', 0),
                    'queue_size': len(getattr(sdl, '_concept_queue', [])),
                    'contradictions_queue': len(getattr(sdl, '_contradiction_topics', []))
                }
            except Exception as e:
                logger.debug(f"Error getting learning metrics: {e}")
        
        return jsonify(metrics)
    
    @app.route('/api/dashboard')
    def api_dashboard():
        """Dashboard data endpoint."""
        if not web_gui_instance or not web_gui_instance.brain:
            return jsonify({'error': 'Brain не инициализирован'}), 500
        
        try:
            brain = web_gui_instance.brain
            
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'active',
                'components': {},
                'stats': {
                    'sessions': len(web_gui_instance.session_manager.sessions) if web_gui_instance.session_manager else 0,
                    'users': len(web_gui_instance.auth_manager.users) if web_gui_instance.auth_manager else 0
                }
            }
            
            # Component statuses
            for comp_name in ['memory_manager', 'knowledge_graph', 'self_dialog_learning', 
                             'contradiction_manager', 'concept_extractor', 'concept_miner']:
                if hasattr(brain, comp_name):
                    comp = getattr(brain, comp_name)
                    dashboard_data['components'][comp_name] = {
                        'available': comp is not None,
                        'running': getattr(comp, 'running', False) or getattr(comp, 'is_running', False)
                    }
            
            # Graph stats
            if hasattr(brain, 'fractal_graph_v2') and brain.fractal_graph_v2:
                fg = brain.fractal_graph_v2
                if hasattr(fg, 'storage'):
                    dashboard_data['graph'] = {
                        'nodes': len(fg.storage.nodes),
                        'edges': len(fg.storage.edges),
                        'groups': len(fg.storage.semantic_groups)
                    }
            
            return jsonify(dashboard_data)
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return jsonify({'error': str(e)}), 500
    
    logger.info("Core routes registered")
