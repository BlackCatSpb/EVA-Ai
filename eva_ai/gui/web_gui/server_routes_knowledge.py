"""
Knowledge routes: documents, knowledge-graph, settings, snapshots, cache-stats
"""
import os
import logging
import time
import json
from datetime import datetime

from flask import jsonify, request
from eva_ai.core.api_compat import API_VERSION, API_PREFIX, api_version

logger = logging.getLogger("eva_ai.webgui.routes_knowledge")


def register_knowledge_routes(app, web_gui_instance):
    """Register knowledge routes."""
    logger.info("Registering knowledge routes...")

    @app.route('/api/documents', methods=['GET'])
    def api_documents():
        """Получить список документов для текущей сессии."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        user_id = request.headers.get('X-User-ID')
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

    @app.route('/api/documents/<file_id>', methods=['DELETE'])
    def api_delete_document(file_id):
        """Удалить документ из сессии."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'error': 'session_id required'}), 400

        session = web_gui_instance.session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Сессия не найдена'}), 404

        context_nodes = session.get('context_nodes', [])
        original_count = len(context_nodes)
        import glob as glob_mod
        filtered_nodes = [n for n in context_nodes if not (n.get('file_data') and n['file_data'].get('file_id') == file_id)]
        removed = original_count - len(filtered_nodes)

        if removed > 0:
            web_gui_instance.session_manager.update_session(session_id, {'context_nodes': filtered_nodes})

        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        for ext in ['']:
            for filepath in glob_mod.glob(os.path.join(upload_dir, f'{file_id}*')):
                try:
                    os.remove(filepath)
                except Exception as e:
                    logger.warning(f"Failed to delete file {filepath}: {e}")

        return jsonify({
            'status': 'ok',
            'removed': removed,
            'message': f'Удалено {removed} документов' if removed > 0 else 'Документ не найден'
        })

    @app.route('/api/documents/memory', methods=['GET'])
    def api_documents_memory():
        """Получить список документов из DocumentVirtualMemory для сессии."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'error': 'session_id required'}), 400

        try:
            documents = web_gui_instance.get_session_documents(session_id)
            
            # Добавляем статистику для каждого документа
            enriched_docs = {}
            for doc_id, doc_meta in documents.items():
                stats = web_gui_instance.get_document_stats(session_id, doc_id)
                enriched_docs[doc_id] = {
                    **doc_meta,
                    'stats': stats
                }
            
            return jsonify({
                'documents': enriched_docs,
                'count': len(enriched_docs)
            })
        except Exception as e:
            logger.error(f"Error getting documents from memory: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/documents/memory/<document_id>', methods=['GET', 'DELETE'])
    def api_document_memory_detail(document_id):
        """Получить детали документа или удалить его из памяти."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'error': 'session_id required'}), 400

        if request.method == 'GET':
            try:
                stats = web_gui_instance.get_document_stats(session_id, document_id)
                if stats:
                    return jsonify({
                        'document_id': document_id,
                        'stats': stats
                    })
                else:
                    return jsonify({'error': 'Документ не найден'}), 404
            except Exception as e:
                logger.error(f"Error getting document stats: {e}")
                return jsonify({'error': str(e)}), 500
        
        elif request.method == 'DELETE':
            # Очистка документов сессии
            try:
                web_gui_instance.clear_session_documents(session_id)
                return jsonify({
                    'status': 'ok',
                    'message': 'Документы сессии очищены'
                })
            except Exception as e:
                logger.error(f"Error clearing documents: {e}")
                return jsonify({'error': str(e)}), 500

    @app.route('/api/knowledge-graph', methods=['GET', 'POST'])
    def api_knowledge_graph():
        """Операции с графом знаний."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован', 'nodes': [], 'total': 0}), 500

        if request.method == 'GET':
            action = request.args.get('action', 'get')

            # Пробуем получить из bridge кэша
            if hasattr(web_gui_instance, 'bridge') and web_gui_instance.bridge:
                cached = web_gui_instance.bridge.get_cached_knowledge_graph()
                if cached and cached.get('nodes'):
                    logger.info("api_knowledge_graph: returning cached data")
                    return jsonify(cached)
            
            try:
                kg = getattr(web_gui_instance.brain, 'knowledge_graph', None)
                
                if action == 'get':
                    nodes = []
                    
                    if kg:
                        try:
                            if hasattr(kg, 'nodes') and kg.nodes:
                                node_list = kg.nodes if isinstance(kg.nodes, list) else list(kg.nodes)
                                for n in node_list[:50]:
                                    try:
                                        nodes.append({
                                            'id': getattr(n, 'id', '') or getattr(n, 'name', ''),
                                            'name': getattr(n, 'name', '')[:50] if hasattr(n, 'name') else '',
                                            'content': getattr(n, 'content', '')[:100] if hasattr(n, 'content') else ''
                                        })
                                    except Exception:
                                        continue
                            elif hasattr(kg, 'get_nodes'):
                                node_list = kg.get_nodes()[:50]
                                for n in node_list:
                                    try:
                                        nodes.append({
                                            'id': getattr(n, 'id', '') or getattr(n, 'name', ''),
                                            'name': getattr(n, 'name', '')[:50] if hasattr(n, 'name') else '',
                                            'content': getattr(n, 'content', '')[:100] if hasattr(n, 'content') else ''
                                        })
                                    except Exception:
                                        continue
                        except Exception as e:
                            logger.debug(f"Error accessing nodes: {e}")
                    
                    if not nodes:
                        nodes = [
                            {'id': 'system', 'name': 'EVA System', 'content': 'Когнитивная система CogniFlex'},
                            {'id': 'memory', 'name': 'Память', 'content': 'Фрактальная память системы'},
                            {'id': 'learning', 'name': 'Обучение', 'content': 'Система самодиалога'}
                        ]
                    
                    return jsonify({'nodes': nodes, 'total': len(nodes)})
                
                elif action == 'search':
                    if kg and hasattr(kg, 'search_nodes'):
                        try:
                            query = request.args.get('query', '')
                            results = kg.search_nodes(query, limit=10)
                            return jsonify({'results': results})
                        except Exception as e:
                            logger.debug(f"Search error: {e}")
                    
                    return jsonify({'results': []})
            except Exception as e:
                logger.debug(f"Error accessing knowledge graph: {e}")
                return jsonify({'nodes': [
                    {'id': 'system', 'name': 'EVA System', 'content': 'Когнитивная система CogniFlex'}
                ], 'total': 1})

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

    @app.route('/api/cache-stats')
    def api_cache_stats():
        """Получить статистику кэша."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        stats = {
            'hybrid_cache': {},
            'search_cache': {},
            'memory': {}
        }

        try:
            if hasattr(web_gui_instance.brain, 'hybrid_cache') and web_gui_instance.brain.hybrid_cache:
                hc = web_gui_instance.brain.hybrid_cache
                if hc and hasattr(hc, 'get_cache_stats'):
                    stats['hybrid_cache'] = hc.get_cache_stats()
                if hc and hasattr(hc, 'get_search_cache_stats'):
                    stats['search_cache'] = hc.get_search_cache_stats()

            if hasattr(web_gui_instance.brain, 'memory_manager') and web_gui_instance.brain.memory_manager:
                mm = web_gui_instance.brain.memory_manager
                if hasattr(mm, 'nodes'):
                    stats['memory']['total_nodes'] = len(mm.nodes)
        except Exception as e:
            logger.debug(f"Error getting cache stats: {e}")

        return jsonify(stats)

    @app.route('/api/settings', methods=['GET', 'POST'])
    @app.route(f'{API_PREFIX}/settings', methods=['GET', 'POST'])
    def api_settings():
        """Получить или обновить настройки системы."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        if request.method == 'GET':
            settings = {
                'auto_learning': True,
                'sre_enabled': True,
                'memory_enabled': True,
                'dark_theme': True,
                'sound_enabled': False,
                'model_name': 'Qwen2.5-0.5B GGUF',
                'version': API_VERSION,
                'api_prefix': API_PREFIX,
                'available_versions': ['1.0.0'],
                'language_mode': 'russian_only',
                'quantization_mode': 'q4_k_m',
                'available_language_modes': ['russian_only', 'no_chinese', 'no_foreign', 'full'],
                'available_quantization_modes': ['q2_k', 'q4_k_m', 'q5_k_m', 'q8_0']
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
                    
                    if hasattr(web_gui_instance.brain, 'mode_controller') and web_gui_instance.brain.mode_controller:
                        mc = web_gui_instance.brain.mode_controller
                        status = mc.get_status()
                        settings['language_mode'] = status.get('language_mode', 'russian_only')
                        settings['quantization_mode'] = status.get('quantization', {}).get('mode', 'q4_k_m')
            except Exception as e:
                logger.debug(f"Error getting settings: {e}")

            return jsonify(settings)

        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400

        try:
            logger.debug(f"api_handler: brain = {web_gui_instance.brain is not None}")

            if web_gui_instance.brain:
                if 'auto_learning' in data and hasattr(web_gui_instance.brain, 'self_dialog_learning'):
                    sdl = web_gui_instance.brain.self_dialog_learning
                    if hasattr(sdl, 'auto_execute_enabled'):
                        sdl.auto_execute_enabled = data['auto_learning']

                if 'memory_enabled' in data and hasattr(web_gui_instance.brain, 'memory_manager') and web_gui_instance.brain.memory_manager:
                    mm = web_gui_instance.brain.memory_manager
                    if hasattr(mm, 'enabled'):
                        mm.enabled = data['memory_enabled']

                if 'sre_enabled' in data:
                    if hasattr(web_gui_instance.brain, 'self_reasoning_engine'):
                        sre = web_gui_instance.brain.self_reasoning_engine
                        if hasattr(sre, 'enabled'):
                            sre.enabled = data['sre_enabled']
                
                if 'language_mode' in data and hasattr(web_gui_instance.brain, 'mode_controller') and web_gui_instance.brain.mode_controller:
                    web_gui_instance.brain.mode_controller.set_language_mode(data['language_mode'])
                    logger.info(f"Language mode changed to: {data['language_mode']}")
                
                if 'quantization_mode' in data and hasattr(web_gui_instance.brain, 'mode_controller') and web_gui_instance.brain.mode_controller:
                    web_gui_instance.brain.mode_controller.set_quantization_mode(data['quantization_mode'])
                    logger.info(f"Quantization mode changed to: {data['quantization_mode']}")
                
            return jsonify({'status': 'ok', 'updated': list(data.keys())})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/snapshots', methods=['GET', 'POST'])
    def api_snapshots():
        """Управление слепками знаний."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        if request.method == 'GET':
            try:
                brain = web_gui_instance.brain
                if brain and hasattr(brain, 'fractal_memory') and brain.fractal_memory:
                    fm = brain.fractal_memory
                    if fm.snapshot_manager:
                        snapshots = fm.snapshot_manager.list_snapshots()
                        return jsonify({'snapshots': snapshots})
            except Exception as e:
                logger.error(f"Error listing snapshots: {e}")

            return jsonify({'snapshots': []})

        elif request.method == 'POST':
            data = request.json or {}
            name = data.get('name', '')

            try:
                brain = web_gui_instance.brain
                if brain and hasattr(brain, 'fractal_memory') and brain.fractal_memory:
                    fm = brain.fractal_memory
                    if fm.snapshot_manager:
                        path = fm.snapshot_manager.export_snapshot(name if name else None)
                        return jsonify({
                            'status': 'success',
                            'path': path
                        })
            except Exception as e:
                logger.error(f"Error creating snapshot: {e}")

            return jsonify({'error': 'Failed to create snapshot'}), 500

    @app.route('/api/stats')
    def api_stats():
        """Get general statistics (alias для analytics)."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        stats = {
            'fractal_nodes': 0,
            'fractal_edges': 0,
            'fractal_groups': 0,
            'queries': 0,
            'dialogs': 0,
            'cache_hit_rate': 0,
            'cpu': 0,
            'memory': 0,
            'web_searches': 0,
            'wiki_articles': 0,
        }
        
        try:
            if web_gui_instance.brain:
                # Graph stats
                fg = getattr(web_gui_instance.brain, 'fractal_graph_v2', None)
                if fg is None:
                    fg = getattr(web_gui_instance.brain, 'components', {}).get('fractal_graph_v2')
                if fg is None and hasattr(web_gui_instance.brain, 'memory_manager'):
                    fg = getattr(web_gui_instance.brain.memory_manager, 'fractal_graph_v2', None)
                
                if fg and hasattr(fg, 'get_stats'):
                    fg_stats = fg.get_stats()
                    if isinstance(fg_stats, dict):
                        stats['fractal_nodes'] = fg_stats.get('total_nodes', 0)
                        stats['fractal_edges'] = fg_stats.get('total_edges', 0)
                        stats['fractal_groups'] = fg_stats.get('total_groups', 0)
                
                # Cache stats
                if hasattr(web_gui_instance.brain, 'get_cache_stats'):
                    cache = web_gui_instance.brain.get_cache_stats()
                    if cache:
                        stats['cache_hit_rate'] = cache.get('hit_rate', 0)
                
                # Web search
                web_search = getattr(web_gui_instance.brain, 'web_search_engine', None)
                if web_search and hasattr(web_search, 'stats'):
                    stats['web_searches'] = web_search.stats.get('searches_performed', 0)
                
                # System
                import psutil
                stats['cpu'] = psutil.cpu_percent(interval=0.1)
                stats['memory'] = psutil.virtual_memory().percent
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
        
        return jsonify(stats)

    @app.route('/api/websearch_stats')
    @app.route('/api/websearch/stats')
    def api_websearch_stats():
        """Get web search statistics including Tavily metrics."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован', 'stats': {}}), 500

        stats = {
            'searches_performed': 0,
            'results_found': 0,
            'cache_hits': 0,
            'errors': 0,
            'tavily_requests': 0,
            'tavily_responses': 0,
            'tavily_errors': 0,
            'active_requests': 0
        }

        try:
            if web_gui_instance.brain:
                web_search = getattr(web_gui_instance.brain, 'web_search_engine', None)
                if web_search:
                    if hasattr(web_search, 'stats'):
                        stats.update(web_search.stats)
                    elif hasattr(web_search, 'get_search_statistics'):
                        search_stats = web_search.get_search_statistics()
                        if search_stats:
                            stats.update(search_stats)
        except Exception as e:
            logger.debug(f"WebSearch stats error: {e}")

        return jsonify({'stats': stats})

    @app.route('/api/knowledge', methods=['GET', 'POST'])
    def api_knowledge():
        """Knowledge base endpoint for frontend."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        if request.method == 'GET':
            # Get knowledge statistics
            result = {
                'total_entities': 0,
                'total_relations': 0,
                'entities': [],
                'relations': [],
                'session_entities': 0
            }
            
            try:
                # Get from fractal graph
                fg = getattr(web_gui_instance.brain, 'fractal_graph_v2', None)
                if fg:
                    if hasattr(fg, 'storage') and hasattr(fg.storage, 'nodes'):
                        result['total_entities'] = len(fg.storage.nodes)
                    if hasattr(fg, 'storage') and hasattr(fg.storage, 'edges'):
                        result['total_relations'] = len(fg.storage.edges)
                    
                    # Get sample entities
                    if hasattr(fg, 'get_nodes_list'):
                        nodes = fg.get_nodes_list()[:50]
                        result['entities'] = [
                            {
                                'id': n.id if hasattr(n, 'id') else str(i),
                                'name': n.content[:50] if hasattr(n, 'content') else str(n)[:50],
                                'type': n.node_type if hasattr(n, 'node_type') else 'concept'
                            }
                            for i, n in enumerate(nodes)
                        ]
                
                # Get session entities
                user_id = request.headers.get('X-User-ID')
                if user_id and hasattr(web_gui_instance.session_manager, 'get_user_sessions'):
                    sessions = web_gui_instance.session_manager.get_user_sessions(user_id)
                    total_session_entities = 0
                    for session in sessions:
                        if isinstance(session, dict):
                            total_session_entities += len(session.get('entities', []))
                    result['session_entities'] = total_session_entities
                    
            except Exception as e:
                logger.debug(f"Knowledge GET error: {e}")
            
            return jsonify(result)
        
        elif request.method == 'POST':
            # Search knowledge
            data = request.get_json() or {}
            action = data.get('action', 'search')
            query = data.get('query', '')
            
            if action == 'search':
                results = []
                try:
                    fg = getattr(web_gui_instance.brain, 'fractal_graph_v2', None)
                    if fg and hasattr(fg, 'search_nodes') and query:
                        matches = fg.search_nodes(query, limit=20)
                        results = [
                            {
                                'name': n.content[:100] if hasattr(n, 'content') else str(n)[:100],
                                'type': n.node_type if hasattr(n, 'node_type') else 'concept',
                                'content': n.content[:200] if hasattr(n, 'content') else ''
                            }
                            for n in matches
                        ]
                except Exception as e:
                    logger.debug(f"Knowledge search error: {e}")
                
                return jsonify({'results': results, 'matches': results})
            
            return jsonify({'error': 'Unknown action'}), 400

    logger.info("Knowledge routes registered")
