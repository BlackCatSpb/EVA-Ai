"""
Основные маршруты Flask для Web GUI ЕВА
"""
import os
import logging
import json
import uuid
import time
import threading
from datetime import datetime

from flask import render_template, jsonify, request, abort, Response, stream_with_context
from eva_ai.core.api_compat import API_VERSION, API_PREFIX, api_version

logger = logging.getLogger("eva_ai.webgui")

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    logger.info("Tesseract configured at: {}".format(TESSERACT_PATH))
except Exception as e:
    logger.warning("Failed to configure Tesseract: {}".format(e))


def register_routes(app, web_gui_instance):
    logger.info("=== REGISTERING ROUTES ===")
    logger.info("  web_gui_instance type: {}".format(type(web_gui_instance).__name__ if web_gui_instance else "None"))

    @app.route('/favicon.ico')
    def favicon():
        return '', 204

    @app.route('/api/system', methods=['GET'])
    def api_system():
        """System information endpoint."""
        logger.debug("=== /api/system CALLED ===")
        logger.debug("  web_gui_instance is None: {}".format(web_gui_instance is None))
        if not web_gui_instance:
            logger.error("web_gui_instance is None!")
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
                'contradiction': True
            }
        }
        
        if web_gui_instance.brain and hasattr(web_gui_instance.brain, 'components'):
            system_info['modules'] = list(web_gui_instance.brain.components.keys())
        
        logger.debug("  Returning: {}".format(system_info))
        return jsonify(system_info)
    
    logger.info("=== ROUTES REGISTERED ===")

    @app.before_request
    def check_request_timeout():
        if request.path == '/api/chat':
            request._start_time = time.time()

    @app.route('/')
    def index():
        # Add cache-busting for JS/CSS
        import time
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
            'brain_components': {}
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
        
        # Brain components
        component_types = [
            'memory_manager', 'self_dialog_learning', 'hybrid_cache',
            'knowledge_graph', 'web_search_engine', 'self_reasoning_engine',
            'two_model_pipeline', 'llama_cpp_deployment', 'qwen_model_manager'
        ]
        for comp in component_types:
            if hasattr(brain, comp):
                result['brain_components'][comp] = {
                    'available': True,
                    'type': type(getattr(brain, comp)).__name__ if getattr(brain, comp) else None
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
    
    @app.route('/api/debug/auth')
    def api_debug_login():
        """Debug login - shows detailed auth process."""
        logger.info("=== DEBUG LOGIN REQUEST ===")
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        username = data.get('username', '')
        password = data.get('password', '')
        
        result = {
            'success': False,
            'step': 'start',
            'details': {},
            'error': None
        }
        
        try:
            result['step'] = 'check_instance'
            if not web_gui_instance:
                result['error'] = 'web_gui_instance is None'
                return jsonify(result), 401
            
            result['step'] = 'get_auth_manager'
            auth_manager = web_gui_instance.auth_manager
            
            result['step'] = 'check_user_exists'
            result['details']['users_in_db'] = list(auth_manager.users.keys())
            
            if username not in auth_manager.users:
                result['step'] = 'user_not_found'
                result['error'] = 'User not found in database'
                logger.error("DEBUG LOGIN: {} - {}".format(result['step'], result['error']))
                return jsonify(result), 401
            
            user_data = auth_manager.users[username]
            result['step'] = 'user_found'
            result['details']['stored_user'] = {
                'username': user_data.get('username'),
                'salt': user_data.get('salt', 'EMPTY'),
                'hash_prefix': user_data.get('password_hash', 'EMPTY')[:30] if user_data.get('password_hash') else 'EMPTY'
            }
            
            # Manual password verification
            result['step'] = 'verify_password'
            import hashlib
            
            salt = user_data.get('salt', '')
            stored_hash = user_data.get('password_hash', '')
            
            if not salt:
                result['step'] = 'no_salt'
                result['error'] = 'No salt in stored user data'
                computed_hash = hashlib.sha256(password.encode()).hexdigest()
            else:
                computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
            
            result['details']['computed_hash_prefix'] = computed_hash[:30]
            result['details']['stored_hash_prefix'] = stored_hash[:30]
            result['details']['hash_match'] = computed_hash == stored_hash
            
            if computed_hash != stored_hash:
                result['step'] = 'hash_mismatch'
                result['error'] = 'Password hash does not match'
                logger.error("DEBUG LOGIN: {} - computed: {} vs stored: {}".format(
                    result['step'], computed_hash[:30], stored_hash[:30]))
                return jsonify(result), 401
            
            # Authentication successful
            result['step'] = 'authenticate_user'
            user = auth_manager.authenticate(username, password)
            
            if not user:
                result['step'] = 'auth_failed'
                result['error'] = 'authenticate() returned None'
                return jsonify(result), 401
            
            result['step'] = 'create_session'
            result['success'] = True
            result['details']['user_id'] = user.get('user_id')
            
            session_id = web_gui_instance.session_manager.create_session(
                user['user_id'],
                "Сессия {}".format(username)
            )
            
            sessions = web_gui_instance.session_manager.get_user_sessions(user['user_id'])
            
            result['details']['session_id'] = session_id
            result['details']['sessions_count'] = len(sessions)
            
            logger.info("DEBUG LOGIN: SUCCESS for user {}".format(username))
            
            return jsonify({
                'user': user['username'],
                'session_id': session_id,
                'sessions': sessions
            })
            
        except Exception as e:
            result['step'] = 'exception'
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
            logger.error("DEBUG LOGIN EXCEPTION: {}".format(e))
            return jsonify(result), 500
    
    @app.route('/api/login', methods=['POST'])
    def api_login():
        logger.info("=== LOGIN REQUEST ===")
        
        data = request.get_json()
        if not data:
            logger.error("LOGIN: Invalid JSON received")
            return jsonify({'error': 'Invalid JSON'}), 400
        username = data.get('username', '')
        password = data.get('password', '')
        
        logger.info("LOGIN attempt: username='{}', password_len={}".format(username, len(password)))
        logger.info("  web_gui_instance: {}".format(web_gui_instance))

        if web_gui_instance:
            logger.info("  auth_manager: {}".format(web_gui_instance.auth_manager))
            logger.info("  auth_manager.users: {}".format(list(web_gui_instance.auth_manager.users.keys())))
            
            # Check if user exists
            if username in web_gui_instance.auth_manager.users:
                stored_user = web_gui_instance.auth_manager.users[username]
                logger.info("  User found in DB:")
                logger.info("    - username: {}".format(stored_user.get('username')))
                logger.info("    - salt: {}".format(stored_user.get('salt', 'NOT SET')))
                logger.info("    - hash (first 20): {}".format(stored_user.get('password_hash', 'NOT SET')[:20] if stored_user.get('password_hash') else 'NOT SET'))
            else:
                logger.warning("  User '{}' NOT found in database".format(username))
            
            # Perform authentication
            logger.info("  Calling authenticate('{}', '***')...".format(username))
            user = web_gui_instance.auth_manager.authenticate(username, password)
            logger.info("  authenticate result: {}".format(user))
            
            if user:
                logger.info("=== LOGIN SUCCESS ===")
                logger.info("  user_id: {}".format(user.get('user_id')))
                
                # Create session
                session_id = web_gui_instance.session_manager.create_session(
                    user['user_id'],
                    "Сессия {}".format(username)
                )
                logger.info("  session_id created: {}".format(session_id))

                sessions = web_gui_instance.session_manager.get_user_sessions(user['user_id'])
                logger.info("  total sessions: {}".format(len(sessions)))

                return jsonify({
                    'user': user['username'],
                    'session_id': session_id,
                    'sessions': sessions
                })
            else:
                logger.warning("=== LOGIN FAILED: Invalid credentials ===")
        else:
            logger.error("=== LOGIN FAILED: web_gui_instance is None ===")

        return jsonify({'error': 'Неверные учетные данные'}), 401

    @app.route('/api/chat', methods=['POST'])
    def api_chat():
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        try:
            # Получаем raw data для отладки
            raw_data = request.get_data(as_text=True)
            logger.debug(f"Raw request data: {raw_data[:200]}")
            
            # Пробуем распарсить JSON
            try:
                data = request.get_json(force=True)
            except Exception as json_err:
                logger.error(f"JSON parse error: {json_err}, raw data: {raw_data[:200]}")
                # Пробуем исправить JSON с одинарными кавычками
                try:
                    import json
                    import re
                    # Заменяем одинарные кавычки на двойные (грубый фикс)
                    fixed_data = raw_data.replace("'", '"')
                    data = json.loads(fixed_data)
                except Exception as fix_err:
                    logger.error(f"Failed to fix JSON: {fix_err}")
                    return jsonify({'error': 'Invalid JSON format'}), 400
            
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

    @app.route(f'{API_PREFIX}/chat', methods=['POST'])
    @api_version("1.0.0")
    def api_chat_v1():
        """v1 API endpoint for chat"""
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
            
            base_timeout = 120
            extra_timeout = (len(message) // 100) * 30
            total_timeout = min(base_timeout + extra_timeout, 360)
            
            result_holder = {'done': False, 'result': None, 'error': None}
            
            def _process():
                try:
                    result_holder['result'] = web_gui_instance.process_message(message, session_id, user_id, file_data)
                except Exception as e:
                    result_holder['error'] = str(e)
                result_holder['done'] = True
            
            worker = threading.Thread(target=_process)
            worker.daemon = True
            worker.start()
            worker.join(timeout=total_timeout)
            
            if not result_holder['done']:
                return jsonify({
                    'version': API_VERSION,
                    'response': f'Таймаут генерации',
                    'status': 'timeout'
                }), 504
            
            if result_holder['error']:
                return jsonify({'version': API_VERSION, 'error': result_holder['error']}), 500
            
            response_data = result_holder['result']
            response_data['version'] = API_VERSION
            return jsonify(response_data)
        except Exception as e:
            logger.error(f"Ошибка в api_chat_v1: {e}", exc_info=True)
            return jsonify({'version': API_VERSION, 'error': str(e)}), 500

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
                    'entities': session.get('entities', []),
                    'chat_history': session.get('chat_history', [])
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
            'timestamp': datetime.now().isoformat(),
            'graph': {},
            'contradictions': {},
            'concepts': {},
            'health': {}
        }

        try:
            logger.debug(f"api_handler: brain = {web_gui_instance.brain is not None}")

            if web_gui_instance.brain:
                rm = getattr(web_gui_instance.brain, 'resource_manager', None)
                if rm:
                    try:
                        metrics['cpu_usage'] = rm.get_cpu_usage() * 100
                        metrics['memory_usage'] = rm.get_memory_usage() * 100
                        current = rm.get_current_metrics()
                        if isinstance(current, dict):
                            metrics['cpu_usage'] = current.get('cpu_percent', metrics['cpu_usage'])
                            metrics['memory_usage'] = current.get('memory_percent', metrics['memory_usage'])
                            metrics['disk_usage'] = current.get('disk_usage_percent', 0)
                            metrics['io_tokens'] = current.get('io_tokens', 0)
                    except Exception as e:
                        logger.debug(f"resource_manager error: {e}")
                elif hasattr(web_gui_instance.brain, 'get_resource_snapshot'):
                    snapshot = web_gui_instance.brain.get_resource_snapshot()
                    if snapshot:
                        metrics.update(snapshot)
                if hasattr(web_gui_instance.brain, 'get_cache_stats'):
                    cache = web_gui_instance.brain.get_cache_stats()
                    metrics['cache_hit_rate'] = cache.get('hit_rate', 0.0)
                
                # === Graph Metrics (FGv2 only) ===
                fg = getattr(web_gui_instance.brain, 'fractal_graph_v2', None)
                if fg is None:
                    fg = getattr(web_gui_instance.brain, 'components', {}).get('fractal_graph_v2')
                if fg is None and hasattr(web_gui_instance.brain, 'memory_manager'):
                    fg = getattr(web_gui_instance.brain.memory_manager, 'fractal_graph_v2', None)
                
                if fg and hasattr(fg, 'get_stats'):
                    try:
                        fg_stats = fg.get_stats()
                        metrics['graph']['fractal_graph_v2'] = fg_stats if isinstance(fg_stats, dict) else {}
                    except Exception as e:
                        logger.debug(f"FractalGraph stats error: {e}")
                
                # === Graph Curator Metrics ===
                gc = getattr(web_gui_instance.brain, 'graph_curator', None)
                if gc and hasattr(gc, 'get_metrics'):
                    try:
                        gc_metrics = gc.get_metrics()
                        metrics['graph']['curator'] = gc_metrics
                    except Exception as e:
                        logger.debug(f"GraphCurator metrics error: {e}")
                
                # === Contradictions ===
                cm = getattr(web_gui_instance.brain, 'contradiction_manager', None)
                if cm:
                    try:
                        if hasattr(cm, 'get_stats'):
                            metrics['contradictions'] = cm.get_stats()
                        elif hasattr(cm, 'contradictions'):
                            metrics['contradictions'] = {
                                'total': len(getattr(cm, 'contradictions', [])),
                                'active': sum(1 for c in getattr(cm, 'contradictions', []) if c.get('status') != 'resolved')
                            }
                    except Exception as e:
                        logger.debug(f"Contradictions error: {e}")
                
                # === Concepts (from FractalGraphV2) ===
                if metrics.get('graph', {}).get('fractal_graph_v2', {}):
                    fg_stats = metrics['graph']['fractal_graph_v2']
                    nodes_by_type = fg_stats.get('nodes_by_type', {})
                    concept_miner = getattr(web_gui_instance.brain, 'concept_miner', None)
                    if concept_miner:
                        try:
                            if hasattr(concept_miner, 'get_metrics'):
                                cm_metrics = concept_miner.get_metrics()
                                candidates = concept_miner.get_candidates() if hasattr(concept_miner, 'get_candidates') else []
                                provisional = sum(1 for c in candidates if c.get('status') == 'provisional')
                                confirmed = sum(1 for c in candidates if c.get('status') == 'confirmed')
                                archived = sum(1 for c in candidates if c.get('status') == 'archived')
                                metrics['concepts'] = {
                                    'provisional': provisional,
                                    'confirmed': confirmed,
                                    'archived': archived,
                                    'total': len(candidates),
                                    'hypothesis_confirmation_ratio': cm_metrics.get('hypothesis_confirmation_ratio', 0),
                                    'status': cm_metrics.get('status', 'unknown')
                                }
                        except Exception as e:
                            logger.debug(f"ConceptMiner metrics error: {e}")
                    else:
                        # Новый формат концептов для отображения
                        fg_v2 = metrics.get('graph', {}).get('fractal_graph_v2', {})
                        nodes_by_type = fg_v2.get('nodes_by_type', {})
                        metrics['concepts'] = {
                            'concept_nodes': nodes_by_type.get('concept', 0),  # Существующие
                            'aci_concepts': nodes_by_type.get('aci_concept', 0),  # В процессе
                            'response_nodes': nodes_by_type.get('response', 0),  # Завершенные (ответы)
                            'total': nodes_by_type.get('concept', 0) + nodes_by_type.get('aci_concept', 0),
                            'status': 'active'
                        }
                
                # === Graph Metrics (только FractalGraphV2) ===
                fg = getattr(web_gui_instance.brain, 'fractal_graph_v2', None)
                if fg is None:
                    fg = getattr(web_gui_instance.brain, 'components', {}).get('fractal_graph_v2')
                if fg is None and hasattr(web_gui_instance.brain, 'memory_manager'):
                    fg = getattr(web_gui_instance.brain.memory_manager, 'fractal_graph_v2', None)
                
                if fg and hasattr(fg, 'get_stats'):
                    try:
                        fg_stats = fg.get_stats()
                        if isinstance(fg_stats, dict):
                            metrics['graph'].update(fg_stats)
                    except Exception as e:
                        logger.debug(f"FractalGraph stats error: {e}")
                
                # === Graph Curator Metrics ===
                gc = getattr(web_gui_instance.brain, 'graph_curator', None)
                if gc and hasattr(gc, 'get_metrics'):
                    try:
                        gc_metrics = gc.get_metrics()
                        if 'graph' not in metrics:
                            metrics['graph'] = {}
                        metrics['graph']['curator'] = gc_metrics
                    except Exception as e:
                        logger.debug(f"GraphCurator metrics error: {e}")
                
                # === Health Check (только FractalGraphV2) ===
                try:
                    health = {
                        'status': 'healthy',
                        'issues': []
                    }
                    
                    total_nodes = 0
                    
                    # Только FractalGraphV2
                    if fg and hasattr(fg, 'get_stats'):
                        try:
                            fg_stats = fg.get_stats()
                            total_nodes = fg_stats.get('total_nodes', 0)
                        except Exception as e:
                            logger.debug(f"FractalGraph stats error: {e}")
                    
                    if total_nodes == 0:
                        health['issues'].append('Фрактальная память пуста')
                    
                    # Проверка куратора
                    if gc and hasattr(gc, 'get_state'):
                        gc_state = gc.get_state()
                        if isinstance(gc_state, str) and gc_state == 'error':
                            health['issues'].append('GraphCurator в состоянии ошибки')
                        elif isinstance(gc_state, dict) and gc_state.get('state') == 'error':
                            health['issues'].append('GraphCurator в состоянии ошибки')
                    
                    if health['issues']:
                        health['status'] = 'degraded'
                    
                    metrics['health'] = health
                except Exception as e:
                    logger.debug(f"Health check error: {e}")
                    metrics['health'] = {'status': 'error', 'error': str(e)}
                    
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
        
        if metrics.get('cpu_usage', 0) == 0 or metrics.get('memory_usage', 0) == 0:
            try:
                import psutil
                metrics['cpu_usage'] = psutil.cpu_percent(interval=0.1)
                metrics['memory_usage'] = psutil.virtual_memory().percent
                metrics['disk_usage'] = psutil.disk_usage('C:\\' if os.name == 'nt' else '/').percent
            except Exception:
                pass

        return jsonify(metrics)

    @app.route('/api/memory-graph')
    def api_memory_graph():
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        graph_data = {'nodes': [], 'edges': [], 'stats': {}}

        try:
            # Сначала пробуем получить из bridge кэша
            if hasattr(web_gui_instance, 'bridge') and web_gui_instance.bridge:
                cached = web_gui_instance.bridge.get_cached_memory_graph()
                if cached and cached.get('nodes'):
                    logger.info("api_memory_graph: returning cached data")
                    return jsonify(cached)
            
            if web_gui_instance.brain and hasattr(web_gui_instance.brain, 'memory_manager') and web_gui_instance.brain.memory_manager:
                mm = web_gui_instance.brain.memory_manager
                if hasattr(mm, 'get_graph_data'):
                    graph_data = mm.get_graph_data()
                elif hasattr(mm, 'nodes') and mm.nodes:
                    graph_data['nodes'] = [
                        {'id': n.get('id', i), 'label': n.get('content', '')[:50], 'type': 'memory'}
                        for i, n in enumerate(mm.nodes[:100])
                    ]
                    graph_data['stats']['total_nodes'] = len(mm.nodes)
            else:
                logger.debug("memory_manager not available")
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
                rm = getattr(web_gui_instance.brain, 'resource_manager', None)
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
                elif hasattr(web_gui_instance.brain, 'get_resource_snapshot'):
                    try:
                        snapshot = web_gui_instance.brain.get_resource_snapshot()
                        if snapshot:
                            analytics['cpu'] = snapshot.get('cpu_usage', snapshot.get('cpu_percent', 0))
                            analytics['memory'] = snapshot.get('memory_usage', snapshot.get('memory_percent', 0))
                            analytics['vram'] = snapshot.get('gpu_memory', snapshot.get('gpu_memory_percent', 0))
                    except Exception as e:
                        logger.debug(f"get_resource_snapshot error: {e}")

                if hasattr(web_gui_instance.brain, 'get_cache_stats'):
                    try:
                        cache_stats = web_gui_instance.brain.get_cache_stats()
                        if cache_stats:
                            analytics['cache_hit_rate'] = cache_stats.get('hit_rate', 0.0)
                            analytics['cache_utilization'] = cache_stats.get('cache_utilization_percent', 0.0)
                    except Exception as e:
                        logger.debug(f"get_cache_stats error: {e}")

                if hasattr(web_gui_instance.brain, 'self_dialog_learning') and web_gui_instance.brain.self_dialog_learning:
                    try:
                        sdl = web_gui_instance.brain.self_dialog_learning
                        if hasattr(sdl, 'get_stats'):
                            stats = sdl.get_stats()
                            analytics['dialogs'] = stats.get('total_dialogs', 0)
                            analytics['gaps'] = stats.get('knowledge_gaps_identified', 0)
                            analytics['learned'] = stats.get('successful_learning', 0)
                    except Exception as e:
                        logger.debug(f"self_dialog_learning error: {e}")

                # FractalGraphV2 метрики
                fg = getattr(web_gui_instance.brain, 'fractal_graph_v2', None)
                if fg is None:
                    fg = getattr(web_gui_instance.brain, 'components', {}).get('fractal_graph_v2')
                if fg is None and hasattr(web_gui_instance.brain, 'memory_manager'):
                    fg = getattr(web_gui_instance.brain.memory_manager, 'fractal_graph_v2', None)
                
                if fg and hasattr(fg, 'get_stats'):
                    try:
                        fg_stats = fg.get_stats()
                        analytics['fractal_nodes'] = fg_stats.get('total_nodes', 0)
                        analytics['fractal_edges'] = fg_stats.get('total_edges', 0)
                        analytics['fractal_groups'] = fg_stats.get('total_groups', 0)
                    except Exception as e:
                        logger.debug(f"FractalGraphV2 stats error: {e}")

                # GraphCurator метрики
                if hasattr(web_gui_instance.brain, 'graph_curator') and web_gui_instance.brain.graph_curator:
                    try:
                        curator = web_gui_instance.brain.graph_curator
                        if hasattr(curator, 'get_metrics'):
                            cur_metrics = curator.get_metrics()
                            analytics['curator_cycles'] = cur_metrics.get('cycles_completed', 0)
                            analytics['curator_state'] = cur_metrics.get('state', 'idle')
                            analytics['curator_next_run'] = cur_metrics.get('next_run', 0)
                    except Exception as e:
                        logger.debug(f"GraphCurator metrics error: {e}")

                # ProcessTrackerMixin - получить напрямую из _process_metrics
                if hasattr(web_gui_instance.brain, '_process_metrics') and web_gui_instance.brain._process_metrics:
                    try:
                        pm = web_gui_instance.brain._process_metrics
                        analytics['queries'] = pm.get('total_queries', 0)
                        total = pm.get('total_queries', 1)
                        success = pm.get('successful_queries', 0)
                        analytics['success_rate'] = success / total if total > 0 else 0
                        analytics['avg_time'] = pm.get('avg_generation_time', 0) * 1000  # ms
                    except Exception as e:
                        logger.debug(f"ProcessTrackerMixin error: {e}")
                    
                    # GPU метрики
                    try:
                        if hasattr(rm, 'get_gpu_metrics'):
                            gpu_metrics = rm.get_gpu_metrics()
                            analytics['vram'] = gpu_metrics.get('vram_percent', 0.0)
                    except Exception as gpu_e:
                        logger.debug(f"GPU metrics error: {gpu_e}")

                if hasattr(web_gui_instance.brain, 'performance_analyzer') and web_gui_instance.brain.performance_analyzer:
                    try:
                        pa = web_gui_instance.brain.performance_analyzer
                        if hasattr(pa, 'analyze_performance'):
                            perf_data = pa.analyze_performance()
                            if analytics['queries'] == 0:
                                analytics['queries'] = perf_data.get('total_queries', 0)
                            if analytics['avg_time'] == 0:
                                analytics['avg_time'] = perf_data.get('avg_query_time_ms', 0)
                            if analytics['success_rate'] == 0:
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

                if hasattr(web_gui_instance.brain, 'self_dialog_learning') and web_gui_instance.brain.self_dialog_learning:
                    try:
                        sdl = web_gui_instance.brain.self_dialog_learning
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
                
                # Web Search / Tavily метрики
                try:
                    web_search = getattr(web_gui_instance.brain, 'web_search_engine', None)
                    if web_search and hasattr(web_search, 'stats'):
                        analytics['tavily_requests'] = web_search.stats.get('tavily_requests', 0)
                        analytics['tavily_responses'] = web_search.stats.get('tavily_responses', 0)
                        analytics['web_searches'] = web_search.stats.get('searches_performed', 0)
                        analytics['web_cache_hits'] = web_search.stats.get('cache_hits', 0)
                except Exception as e:
                    logger.debug(f"WebSearch stats error: {e}")
                
                # Wikipedia метрики
                try:
                    analytics['wiki_queries'] = analytics.get('web_searches', 0)
                    analytics['wiki_articles'] = analytics.get('fractal_nodes', 0)
                    analytics['wiki_cached'] = analytics.get('web_cache_hits', 0)
                except Exception as e:
                    logger.debug(f"Wiki stats error: {e}")

        except Exception as e:
            logger.error(f"Error getting analytics: {e}")

        if not web_gui_instance or not web_gui_instance.brain:
            analytics['dialogs'] = 0
            analytics['gaps'] = 0
            analytics['learned'] = 0
        
        if analytics.get('cpu', 0) == 0 or analytics.get('memory', 0) == 0:
            try:
                import psutil
                analytics['cpu'] = psutil.cpu_percent(interval=0.1)
                analytics['memory'] = psutil.virtual_memory().percent
            except Exception:
                pass

        return jsonify(analytics)

    @app.route('/api/learning')
    def api_learning():
        """Get learning opportunities and stats."""
        import traceback
        
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

        if not web_gui_instance or not web_gui_instance.brain:
            learning['total'] = 0
            learning['success'] = 0
            learning['pending'] = 0

        return jsonify(learning)

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

    # @app.route('/api/knowledge')  # MOVED to server_api_knowledge.py

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

    @app.route('/api/generation-status', defaults={'command_id': None})
    @app.route('/api/generation-status/<command_id>')
    def api_generation_status(command_id):
        """Get status of active generation(s)."""
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

    @app.route('/api/events/stream')
    def api_events_stream():
        """Server-Sent Events endpoint for real-time generation progress."""
        def event_stream():
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
                    event_type = event.type if hasattr(event, 'type') else ''
                    msg_queue.put({'event_type': event_type, 'data': data})
                except Exception:
                    pass

            pipeline_events = [
                'pipeline.start', 'pipeline.model_a.start', 'pipeline.model_a.complete',
                'pipeline.model_b.start', 'pipeline.model_b.complete',
                'pipeline.model_c.start', 'pipeline.model_c.complete',
                'pipeline.complete', 'pipeline.failed',
                'generation.progress', 'generation.started', 'generation.completed',
                'generation.failed', 'generation.timeout',
                # События куратора
                'curator.started', 'curator.completed', 'curator.error',
                'curator.graph_optimized', 'curator.knowledge_extracted', 'curator.cleanup_done',
                'curator.metrics_updated',
                # События self-dialog
                'self_dialog.started', 'self_dialog.completed', 'self_dialog.learning',
            ]

            subscriptions = {}
            try:
                from eva_ai.core.event_bus import EventTypes
                event_type_map = {
                    'pipeline.start': EventTypes.PIPELINE_START,
                    'pipeline.model_a.start': EventTypes.PIPELINE_MODEL_A_START,
                    'pipeline.model_a.complete': EventTypes.PIPELINE_MODEL_A_COMPLETE,
                    'pipeline.model_b.start': EventTypes.PIPELINE_MODEL_B_START,
                    'pipeline.model_b.complete': EventTypes.PIPELINE_MODEL_B_COMPLETE,
                    'pipeline.complete': EventTypes.PIPELINE_COMPLETE,
                    'pipeline.failed': EventTypes.PIPELINE_FAILED,
                }
                for evt_name, evt_type in event_type_map.items():
                    sub_id = event_bus.subscribe(evt_type, handler)
                    subscriptions[evt_name] = sub_id
            except Exception:
                for evt_name in pipeline_events:
                    try:
                        sub_id = event_bus.subscribe(evt_name, handler)
                        subscriptions[evt_name] = sub_id
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

    @app.route('/api/websearch_stats')
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
        return f"[Ошибка чтения файла: {str(e)}"
