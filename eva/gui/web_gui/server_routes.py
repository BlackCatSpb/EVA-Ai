"""
Основные маршруты Flask для Web GUI ЕВА
"""
import os
import logging
import json
import uuid
from datetime import datetime

from flask import render_template, jsonify, request

logger = logging.getLogger("eva.webgui")

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    logger.info(f"Tesseract configured at: {TESSERACT_PATH}")
except Exception as e:
    logger.warning(f"Failed to configure Tesseract: {e}")


def register_routes(app, web_gui_instance):

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/status')
    def api_status():
        if not web_gui_instance:
            return jsonify({'status': 'not_initialized'})

        status = {
            'status': 'active',
            'sessions_count': len(web_gui_instance.session_manager.sessions),
            'timestamp': datetime.now().isoformat()
        }

        if web_gui_instance.brain:
            status['brain_connected'] = True
            if hasattr(web_gui_instance.brain, 'running'):
                status['brain_running'] = web_gui_instance.brain.running
            if hasattr(web_gui_instance.brain, 'components'):
                status['components'] = len(web_gui_instance.brain.components)
        else:
            status['brain_connected'] = False

        return jsonify(status)

    @app.route('/api/system')
    def api_system():
        """Получить общую информацию о системе."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        system_info = {
            'version': '1.0.0',
            'model': 'Qwen2.5-0.5B GGUF',
            'qwen_ready': False,
            'llama_cpp_ready': False,
            'modules': {
                'contradiction': False,
                'ethics': False,
                'web_search': False,
                'knowledge_graph': False
            },
            'features': {
                'self_learning': False,
                'knowledge_graph': False,
                'web_search': False
            }
        }

        try:
            logger.debug(f"api_handler: brain = {web_gui_instance.brain is not None}")

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

    @app.route('/api/login', methods=['POST'])
    def api_login():
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        username = data.get('username', '')
        password = data.get('password', '')

        if web_gui_instance:
            user = web_gui_instance.auth_manager.authenticate(username, password)
            if user:
                existing_sessions = web_gui_instance.session_manager.get_user_sessions(user['user_id'])

                session_id = web_gui_instance.session_manager.create_session(
                    user['user_id'],
                    f"Сессия {username}"
                )

                sessions = web_gui_instance.session_manager.get_user_sessions(user['user_id'])

                return jsonify({
                    'user': user['username'],
                    'session_id': session_id,
                    'sessions': sessions
                })

        return jsonify({'error': 'Неверные учетные данные'}), 401

    @app.route('/api/chat', methods=['POST'])
    def api_chat():
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        try:
            data = request.get_json(force=True)
            if not data:
                return jsonify({'error': 'Invalid JSON'}), 400
            message = data.get('message', '')
            session_id = data.get('session_id')
            user_id = data.get('user_id')
            file_data = data.get('file_data')

            result = web_gui_instance.process_message(message, session_id, user_id, file_data)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Ошибка в api_chat: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/sessions', methods=['GET', 'POST', 'DELETE'])
    def api_sessions():
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        user_id = request.headers.get('X-User-ID')

        if request.method == 'GET':
            sessions = web_gui_instance.session_manager.get_user_sessions(user_id)
            return jsonify({'sessions': sessions})

        if request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON'}), 400
            name = data.get('name')
            session_id = web_gui_instance.session_manager.create_session(user_id, name)
            sessions = web_gui_instance.session_manager.get_user_sessions(user_id)
            return jsonify({'session_id': session_id, 'sessions': sessions})

        if request.method == 'DELETE':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON'}), 400
            session_id = data.get('session_id')
            if not session_id:
                return jsonify({'error': 'session_id is required'}), 400
            session = web_gui_instance.session_manager.get_session(session_id)
            if not session:
                return jsonify({'error': 'Сессия не найдена'}), 404
            if user_id and session.get('user_id') != user_id:
                return jsonify({'error': 'Доступ запрещён'}), 403
            web_gui_instance.session_manager.delete_session(session_id)
            sessions = web_gui_instance.session_manager.get_user_sessions(user_id)
            return jsonify({'sessions': sessions, 'message': 'Сессия удалена'})

    @app.route('/api/session/<session_id>', methods=['GET', 'DELETE', 'PUT'])
    def api_session(session_id):
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        if request.method == 'GET':
            session = web_gui_instance.session_manager.get_session(session_id)
            if session:
                return jsonify({
                    'session': session,
                    'context': session.get('context_nodes', []),
                    'entities': session.get('entities', [])
                })
            return jsonify({'error': 'Сессия не найдена'}), 404

        if request.method == 'DELETE':
            session = web_gui_instance.session_manager.get_session(session_id)
            if not session:
                return jsonify({'error': 'Сессия не найдена'}), 404
            web_gui_instance.session_manager.delete_session(session_id)
            return jsonify({'message': 'Сессия удалена'})

        if request.method == 'PUT':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON'}), 400
            session = web_gui_instance.session_manager.get_session(session_id)
            if not session:
                return jsonify({'error': 'Сессия не найдена'}), 404
            allowed_fields = {'name', 'chat_history', 'context_nodes', 'entities'}
            update_data = {k: v for k, v in data.items() if k in allowed_fields}
            if not update_data:
                return jsonify({'error': 'Нет допустимых полей для обновления'}), 400
            web_gui_instance.session_manager.update_session(session_id, update_data)
            updated = web_gui_instance.session_manager.get_session(session_id)
            return jsonify({'session': updated, 'message': 'Сессия обновлена'})

    @app.route('/api/upload', methods=['POST'])
    def api_upload():
        """Загрузка файла с извлечением текста"""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'Файл не прикреплён'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400

        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        file_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1].lower()
        safe_filename = f"{file_id}{ext}"
        filepath = os.path.join(upload_dir, safe_filename)
        file.save(filepath)

        extracted_text = extract_text_from_file(filepath, ext)

        if extracted_text:
            logger.info(f"Текст извлечён из файла {file.filename} (метод: {ext})")

        return jsonify({
            'file_id': file_id,
            'filename': file.filename,
            'size': os.path.getsize(filepath),
            'extracted_text': extracted_text[:5000] if extracted_text else '',
            'status': 'ok'
        })

    @app.route('/api/entities/<session_id>', methods=['GET', 'DELETE'])
    def api_entities(session_id):
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        if request.method == 'GET':
            session = web_gui_instance.session_manager.get_session(session_id)
            if session:
                return jsonify({
                    'entities': session.get('entities', []),
                    'context_count': len(session.get('context_nodes', []))
                })
            return jsonify({'error': 'Сессия не найдена'}), 404

        if request.method == 'DELETE':
            session = web_gui_instance.session_manager.get_session(session_id)
            if not session:
                return jsonify({'error': 'Сессия не найдена'}), 404
            entity_type = request.args.get('type')
            entity_id = request.args.get('entity_id')
            entities = session.get('entities', [])
            original_count = len(entities)
            if entity_id:
                entities = [e for e in entities if e.get('id') != entity_id]
            elif entity_type:
                entities = [e for e in entities if e.get('type') != entity_type]
            else:
                entities = []
            web_gui_instance.session_manager.update_session(session_id, {'entities': entities})
            removed = original_count - len(entities)
            return jsonify({'entities': entities, 'removed': removed, 'message': f'Удалено {removed} сущностей'})

    @app.route('/api/feedback', methods=['POST'])
    def api_feedback():
        """Receive user feedback (like/dislike) for messages"""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON'}), 400

            rating = data.get('rating', 0)
            message_text = data.get('message_text', '')
            message_index = data.get('message_index', 0)

            logger.info(f"User feedback: rating={rating}, index={message_index}")

            if web_gui_instance.brain:
                try:
                    if hasattr(web_gui_instance.brain, 'trigger_subjective_correctness'):
                        web_gui_instance.brain.trigger_subjective_correctness(
                            message_text=message_text,
                            rating=rating
                        )
                except Exception as e:
                    logger.debug(f"Feedback brain trigger error: {e}")

            return jsonify({'success': True, 'rating': rating})

        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/metrics')
    def api_metrics():
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'cache_hit_rate': 0.0,
            'timestamp': datetime.now().isoformat()
        }

        try:
            logger.debug(f"api_handler: brain = {web_gui_instance.brain is not None}")

            if web_gui_instance.brain:
                if hasattr(web_gui_instance.brain, 'get_resource_snapshot'):
                    snapshot = web_gui_instance.brain.get_resource_snapshot()
                    metrics.update(snapshot)
                if hasattr(web_gui_instance.brain, 'get_cache_stats'):
                    cache = web_gui_instance.brain.get_cache_stats()
                    metrics['cache_hit_rate'] = cache.get('hit_rate', 0.0)
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")

        return jsonify(metrics)

    @app.route('/api/memory-graph')
    def api_memory_graph():
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        graph_data = {'nodes': [], 'edges': [], 'stats': {}}

        try:
            if web_gui_instance.brain and hasattr(web_gui_instance.brain, 'memory_manager'):
                mm = web_gui_instance.brain.memory_manager
                if hasattr(mm, 'get_graph_data'):
                    graph_data = mm.get_graph_data()
                elif hasattr(mm, 'nodes'):
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
            'activities': []
        }

        try:
            logger.debug(f"api_handler: brain = {web_gui_instance.brain is not None}")

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
                            analytics['queries'] = 0
                            analytics['avg_time'] = 0
                            analytics['success_rate'] = 0

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

    @app.route('/api/learning')
    def api_learning():
        """Get learning opportunities and stats."""
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
            logger.debug(f"api_handler: brain = {web_gui_instance.brain is not None}")

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

    @app.route('/api/settings', methods=['GET', 'POST'])
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
                'version': '1.0.0'
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
            logger.debug(f"api_handler: brain = {web_gui_instance.brain is not None}")

            if web_gui_instance.brain:
                if 'auto_learning' in data and hasattr(web_gui_instance.brain, 'self_dialog_learning'):
                    sdl = web_gui_instance.brain.self_dialog_learning
                    if hasattr(sdl, 'auto_execute_enabled'):
                        sdl.auto_execute_enabled = data['auto_learning']

                if 'memory_enabled' in data and hasattr(web_gui_instance.brain, 'memory_manager'):
                    mm = web_gui_instance.brain.memory_manager
                    if hasattr(mm, 'enabled'):
                        mm.enabled = data['memory_enabled']

                if 'sre_enabled' in data:
                    if hasattr(web_gui_instance.brain, 'self_reasoning_engine'):
                        sre = web_gui_instance.brain.self_reasoning_engine
                        if hasattr(sre, 'enabled'):
                            sre.enabled = data['sre_enabled']

            return jsonify({'status': 'ok', 'updated': list(data.keys())})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

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

    @app.route('/api/knowledge-graph', methods=['GET', 'POST'])
    def api_knowledge_graph():
        """Операции с графом знаний."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

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

    @app.route('/api/self-dialog', methods=['GET', 'POST'])
    def api_self_dialog():
        """Управление самодиалогом обучения."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

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
                brain = web_gui_instance.brain
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
                brain = web_gui_instance.brain
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


def extract_text_from_file(filepath, ext):
    """Извлекает текст из файла в зависимости от типа. Всегда возвращает строку."""
    try:
        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        elif ext == '.pdf':
            text = ''

            try:
                import fitz
                doc = fitz.open(filepath)
                for page in doc:
                    text += page.get_text() + '\n'
                doc.close()
                if text.strip():
                    logger.info(f"PDF прочитан через pymupdf: {len(text)} символов")
                    return text
            except Exception as e:
                logger.warning(f"pymupdf failed: {e}")

            try:
                import pdfplumber
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n'
                if text.strip():
                    logger.info(f"PDF прочитан через pdfplumber: {len(text)} символов")
                    return text
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")

            try:
                import PyPDF2
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n'
                if text.strip():
                    logger.info(f"PDF прочитан через PyPDF2: {len(text)} символов")
                    return text
            except Exception as e:
                logger.warning(f"PyPDF2 failed: {e}")

            try:
                import pytesseract
                from PIL import Image
                import fitz
                doc = fitz.open(filepath)
                ocr_text = ''
                for page_num in range(min(3, len(doc))):
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img, lang='rus+eng')
                    if page_text:
                        ocr_text += page_text + '\n'
                doc.close()
                if ocr_text.strip():
                    logger.info(f"PDF OCR (Tesseract): {len(ocr_text)} символов")
                    return ocr_text
            except Exception as e:
                logger.warning(f"Tesseract OCR failed: {e}")

            return "[PDF отсканирован - установите Tesseract OCR для распознавания текста]"

        elif ext == '.docx':
            try:
                from docx import Document
                doc = Document(filepath)
                text = '\n'.join([p.text for p in doc.paragraphs])
                return text
            except ImportError:
                logger.warning("python-docx не установлен")
                return f"[DOCX: python-docx не установлен для чтения {os.path.basename(filepath)}]"

        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            try:
                import pytesseract
                from PIL import Image
                img = Image.open(filepath)
                text = pytesseract.image_to_string(img, lang='rus+eng')
                return text
            except ImportError:
                logger.warning("pytesseract/Pillow не установлены")
                return f"[Изображение: {os.path.basename(filepath)} - OCR недоступен]"

        elif ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.md', '.rst', '.csv', '.log', '.ini', '.cfg', '.conf']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        else:
            return f"[Формат файла не поддерживается: {ext}]"
    except Exception as e:
        logger.error(f"Ошибка извлечения текста: {e}")
        return f"[Ошибка чтения файла: {str(e)}]"
