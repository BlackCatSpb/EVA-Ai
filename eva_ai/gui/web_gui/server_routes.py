"""
Основные маршруты Flask для Web GUI ЕВА

NOTE: Многие routes были перенесены в отдельные модули:
- server_routes_core.py: system, health, metrics, stats
- server_routes_chat.py: login, chat, sessions
- server_routes_analytics.py: memory-graph, analytics, dashboard
- server_routes_knowledge.py: documents, knowledge-graph, settings, snapshots

Этот файл содержит оставшиеся routes и служит агрегатором.
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
TESSDATA_PREFIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core', 'tessdata')
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    os.environ['TESSDATA_PREFIX'] = TESSDATA_PREFIX
    logger.info("Tesseract configured at: {} with tessdata: {}".format(TESSERACT_PATH, TESSDATA_PREFIX))
except Exception as e:
    logger.warning("Failed to configure Tesseract: {}".format(e))


def register_routes(app, web_gui_instance):
    logger.info("=== REGISTERING ROUTES (self-dialog only) ===")
    logger.info("  web_gui_instance type: {}".format(type(web_gui_instance).__name__ if web_gui_instance else "None"))

    # NOTE: Most routes moved to separate modules (server_routes_core.py, etc.)
    # Only self-dialog endpoints kept here to avoid breaking changes
    
    @app.route('/api/self-dialog/monitor', methods=['GET'])
    def api_self_dialog_monitor():
        """Monitor endpoint."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        try:
            brain = web_gui_instance.brain
            if brain and hasattr(brain, 'self_dialog_learning'):
                sdl = brain.self_dialog_learning
                return jsonify({
                    'status': 'ok',
                    'monitor': {
                        'concepts_in_queue': len(getattr(sdl, '_concept_queue', [])),
                        'contradictions_in_queue': len(getattr(sdl, '_contradiction_topics', []))
                    }
                })
            return jsonify({'error': 'Not available'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
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

    @app.route('/api/chat/stream', methods=['POST'])
    def api_chat_stream():
        """Streaming chat endpoint с Server-Sent Events."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        try:
            data = request.get_json(force=True)
            if not data:
                return jsonify({'error': 'Invalid JSON'}), 400
            
            message = data.get('message', '')
            session_id = data.get('session_id')
            user_id = data.get('user_id')
            mode = data.get('mode', 'extended')
            
            def generate_stream():
                """Генератор для SSE."""
                try:
                    # Получаем dual_generator
                    dg = None
                    if (web_gui_instance.brain and 
                        hasattr(web_gui_instance.brain, 'two_model_pipeline') and
                        web_gui_instance.brain.two_model_pipeline and
                        hasattr(web_gui_instance.brain.two_model_pipeline, 'dual_generator')):
                        dg = web_gui_instance.brain.two_model_pipeline.dual_generator
                    
                    if not dg:
                        yield f"data: {json.dumps({'type': 'error', 'text': 'Generator not available'})}\n\n"
                        return
                    
                    # Отправляем начало
                    yield f"data: {json.dumps({'type': 'start', 'timestamp': time.time()})}\n\n"
                    
                    # Генерируем поток токенов
                    full_text = ""
                    for chunk in dg.generate_streaming(message, mode=mode):
                        chunk_data = {
                            'type': chunk.get('type', 'chunk'),
                            'text': chunk.get('text', ''),
                            'tokens_count': chunk.get('tokens_count', 0),
                            'elapsed_ms': chunk.get('elapsed_ms', 0)
                        }
                        full_text += chunk.get('text', '')
                        
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        
                        if chunk.get('type') == 'complete':
                            break
                    
                    # Сохраняем в историю
                    if session_id:
                        web_gui_instance.session_manager.add_chat_message(session_id, 'assistant', full_text)
                    
                    # Отправляем завершение
                    yield f"data: {json.dumps({'type': 'done', 'full_text': full_text})}\n\n"
                    
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
            
            return Response(
                generate_stream(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
            
        except Exception as e:
            logger.error(f"Ошибка в api_chat_stream: {e}", exc_info=True)
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
            
            # Получаем FractalGraphV2
            fg = getattr(web_gui_instance.brain, 'fractal_graph_v2', None)
            if fg is None:
                fg = getattr(web_gui_instance.brain, 'components', {}).get('fractal_graph_v2')
            if fg is None and hasattr(web_gui_instance.brain, 'memory_manager'):
                fg = getattr(web_gui_instance.brain.memory_manager, 'fractal_graph_v2', None)
            
            # Если FractalGraphV2 найден
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
            
            # Fallback на memory_manager
            elif web_gui_instance.brain and hasattr(web_gui_instance.brain, 'memory_manager') and web_gui_instance.brain.memory_manager:
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
                @app.route('/api/eva/introspection', methods=['GET'])
    def api_eva_introspection():
        """Self-awareness endpoint - EVA может узнать о своём состоянии."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        try:
            brain = web_gui_instance.brain
            if not brain:
                return jsonify({'error': 'Brain не доступен'}), 500
            
            # Собираем информацию о системе
            introspection = {
                'components': {},
                'memory': {},
                'learning': {},
                'status': {}
            }
            
            # Компоненты
            if hasattr(brain, 'components'):
                for name, comp in brain.components.items():
                    status = 'unknown'
                    if hasattr(comp, 'initialized'):
                        status = 'active' if getattr(comp, 'initialized', False) else 'inactive'
                    elif hasattr(comp, 'is_running'):
                        status = 'running' if comp.is_running() else 'stopped'
                    introspection['components'][name] = {'status': status}
            
            # Память
            if hasattr(brain, 'fractal_graph_v2'):
                try:
                    fgv2 = brain.fractal_graph_v2
                    nodes = getattr(fgv2, 'nodes', {})
                    introspection['memory']['nodes_count'] = len(nodes)
                    introspection['memory']['types'] = {}
                    for node in nodes.values():
                        t = getattr(node, 'node_type', 'unknown')
                        introspection['memory']['types'][t] = introspection['memory']['types'].get(t, 0) + 1
                except Exception as e:
                    introspection['memory']['error'] = str(e)
            
            # Обучение
            if hasattr(brain, 'self_dialog_learning'):
                sdl = brain.self_dialog_learning
                introspection['learning'] = {
                    'enabled': getattr(sdl, 'enabled', False),
                    'running': getattr(sdl, 'running', False),
                    'concepts_queue': len(getattr(sdl, '_concept_queue', [])),
                    'contradictions_queue': len(getattr(sdl, '_contradiction_topics', []))
                }
            
            # Общий статус
            introspection['status'] = {
                'connected': True,
                'sessions': getattr(web_gui_instance, '_sessions', {}).get('count', 0)
            }
            
            return jsonify({'status': 'ok', 'introspection': introspection})
            
        except Exception as e:
            logger.error(f"Introspection error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/self-dialog/monitor', methods=['GET'])
    def api_self_dialog_monitor():
        """Получить монитор внутренних рассуждений."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        try:
            brain = web_gui_instance.brain
            if brain and hasattr(brain, 'self_dialog_learning'):
                sdl = brain.self_dialog_learning
                
                # Получаем статистику dual circuit
                dual_stats = {}
                if hasattr(sdl, '_get_dual_generator'):
                    dual_gen = sdl._get_dual_generator()
                    if dual_gen and hasattr(dual_gen, 'get_stats'):
                        stats = dual_gen.get_stats()
                        dual_stats = stats.get('dual_circuit', {})
                
                # Статистика очередей
                queue_info = {
                    'concepts_in_queue': len(getattr(sdl, '_concept_queue', [])),
                    'contradictions_in_queue': len(getattr(sdl, '_contradiction_topics', [])),
                    'dual_circuit_calls': dual_stats.get('calls', 0),
                    'concepts_extracted': dual_stats.get('concepts_extracted', 0),
                    'knowledge_saved': dual_stats.get('knowledge_saved', 0),
                }
                
                return jsonify({
                    'status': 'ok',
                    'monitor': queue_info
                })
            else:
                return jsonify({'error': 'Self-dialog learning not available'}), 500
        except Exception as e:
            logger.error(f"Error getting monitor: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/self-dialog/trigger', methods=['POST'])
    def api_self_dialog_trigger():
        """Триггер для запуска самообучения по требованию."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        data = request.json or {}
        reason = data.get('reason', 'manual')
        
        try:
            brain = web_gui_instance.brain
            if brain and hasattr(brain, 'self_dialog_learning'):
                sdl = brain.self_dialog_learning
                
                # Вызываем trigger_self_dialog если доступен
                if hasattr(sdl, 'trigger_self_dialog'):
                    result = sdl.trigger_self_dialog(reason=reason)
                    return jsonify({
                        'status': 'success' if result else 'no_work',
                        'triggered': result,
                        'reason': reason,
                        'queue_size': len(getattr(sdl, '_concept_queue', [])),
                        'contradictions_size': len(getattr(sdl, '_contradiction_topics', []))
                    })
                else:
                    return jsonify({'error': 'trigger_self_dialog not available'}), 500
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

    @app.route('/api/metrics')
    def api_metrics():
        """Получить все метрики производительности."""
        metrics = {
            'counters': {},
            'gauges': {},
            'histograms': {},
            'system': {},
            'graph': {},
            'timestamp': time.time()
        }
        
        try:
            # Метрики из реестра
            from eva_ai.core.metrics import get_metrics_registry
            registry = get_metrics_registry()
            
            # Prometheus format по запросу
            if request.args.get('format') == 'prometheus':
                prometheus_data = registry.export_prometheus()
                return Response(prometheus_data, mimetype='text/plain')
            
            # JSON format - собираем все метрики
            all_metrics = registry.get_all_metrics()
            metrics.update(all_metrics)
            
            # Системные метрики
            try:
                import psutil
                metrics['system'] = {
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
                    'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
                }
            except Exception:
                pass
            
            # Метрики из brain
            if web_gui_instance and web_gui_instance.brain:
                brain = web_gui_instance.brain
                
                # FractalGraphV2 stats
                fg = getattr(brain, 'fractal_graph_v2', None)
                if fg is None:
                    fg = getattr(brain, 'components', {}).get('fractal_graph_v2')
                if fg is None and hasattr(brain, 'memory_manager'):
                    fg = getattr(brain.memory_manager, 'fractal_graph_v2', None)
                
                if fg and hasattr(fg, 'get_stats'):
                    try:
                        fg_stats = fg.get_stats()
                        if isinstance(fg_stats, dict):
                            metrics['graph'] = fg_stats
                    except Exception:
                        pass
                
                # Cache stats
                if hasattr(brain, 'get_cache_stats'):
                    try:
                        cache = brain.get_cache_stats()
                        if cache:
                            metrics['gauges']['cache_hit_rate'] = cache.get('hit_rate', 0)
                    except Exception:
                        pass
                
                # Web search stats
                web_search = getattr(brain, 'web_search_engine', None)
                if web_search and hasattr(web_search, 'stats'):
                    try:
                        metrics['counters']['web_searches'] = web_search.stats.get('searches_performed', 0)
                    except Exception:
                        pass
            
            return jsonify(metrics)
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return jsonify({'error': str(e), 'metrics': metrics}), 500

    @app.route('/api/health')
    def api_health():
        """Health check endpoint."""
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'components': {}
        }
        
        # Проверяем компоненты
        if web_gui_instance:
            health_status['components']['web_gui'] = 'ok'
            
            if web_gui_instance.brain:
                health_status['components']['brain'] = 'ok'
                
                # Проверяем pipeline
                if hasattr(web_gui_instance.brain, 'two_model_pipeline'):
                    health_status['components']['pipeline'] = 'ok'
                
                # Проверяем graph
                if hasattr(web_gui_instance.brain, 'fractal_graph_v2'):
                    fg = web_gui_instance.brain.fractal_graph_v2
                    cache_stats = fg.get_search_cache_stats()
                    health_status['components']['fractal_graph'] = {
                        'status': 'ok',
                        'nodes_count': len(fg.storage.nodes),
                        'cache_hit_rate': cache_stats.get('hit_rate', 0)
                    }
        else:
            health_status['status'] = 'unhealthy'
            health_status['components']['web_gui'] = 'error'
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code

    @app.route('/api/health/detailed')
    def api_health_detailed():
        """Детальная проверка здоровья системы с метриками."""
        try:
            from eva_ai.core.metrics import get_metrics_registry, get_eva_metrics
            import psutil
            
            registry = get_metrics_registry()
            eva_metrics = get_eva_metrics()
            
            # Системные метрики
            system_metrics = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory': {
                    'percent': psutil.virtual_memory().percent,
                    'used_gb': psutil.virtual_memory().used / (1024**3),
                    'available_gb': psutil.virtual_memory().available / (1024**3)
                },
                'disk': {
                    'percent': psutil.disk_usage('/').percent,
                    'used_gb': psutil.disk_usage('/').used / (1024**3)
                }
            }
            
            # Метрики EVA
            eva_stats = {
                'cache_hit_rate': eva_metrics.get_cache_hit_rate(),
                'generation_stats': eva_metrics.get_generation_stats(),
                'all_metrics': registry.get_all_metrics()
            }
            
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time(),
                'system': system_metrics,
                'eva': eva_stats
            })
            
        except Exception as e:
            logger.error(f"Error in detailed health check: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @app.route('/api/dashboard')
    def api_dashboard():
        """Performance dashboard data."""
        try:
            from eva_ai.core.metrics import get_eva_metrics, get_metrics_registry
            
            eva_metrics = get_eva_metrics()
            registry = get_metrics_registry()
            
            # Ключевые метрики для дашборда
            dashboard_data = {
                'summary': {
                    'cache_hit_rate': eva_metrics.get_cache_hit_rate(),
                    'generation_stats': eva_metrics.get_generation_stats(),
                },
                'charts': {
                    'request_duration': registry.histogram('eva_request_duration_seconds').get_stats(),
                    'generation_duration': registry.histogram('eva_generation_duration_seconds').get_stats(),
                    'search_duration': registry.histogram('eva_search_duration_seconds').get_stats(),
                },
                'counters': {
                    'requests_total': registry.counter('eva_requests_total').get(),
                    'generations_total': registry.counter('eva_generations_total').get(),
                    'errors_total': registry.counter('eva_errors_total').get(),
                    'cache_hits': registry.counter('eva_cache_hits_total').get(),
                    'cache_misses': registry.counter('eva_cache_misses_total').get(),
                },
                'gauges': {
                    'cpu_percent': registry.gauge('system_cpu_percent').get(),
                    'memory_percent': registry.gauge('system_memory_percent').get(),
                }
            }
            
            return jsonify(dashboard_data)
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return jsonify({'error': str(e)}), 500


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
