"""
Chat routes для Web GUI
Chat endpoints, streaming, sessions
"""
import json
import logging
import threading
import time
from flask import jsonify, request, Response
from eva_ai.core.api_compat import API_VERSION, API_PREFIX, api_version

logger = logging.getLogger("eva_ai.webgui")


def register_chat_routes(app, web_gui_instance):
    """Регистрирует chat и session роуты"""
    
    @app.route('/api/chat', methods=['POST'])
    def api_chat():
        """Main chat endpoint."""
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
            
            # Calculate timeout based on message length
            base_timeout = 120
            extra_timeout = (len(message) // 100) * 30
            total_timeout = min(base_timeout + extra_timeout, 360)
            
            result_holder = {'done': False, 'result': None, 'error': None}
            
            def _process():
                try:
                    result_holder['result'] = web_gui_instance.process_message(
                        message, session_id, user_id, file_data
                    )
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
                return jsonify({
                    'version': API_VERSION, 
                    'error': result_holder['error']
                }), 500
            
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
                """Генератор для SSE со стримингом чанков."""
                try:
                    # Получаем HybridKnowledgeDialogManager (приоритет) или PipelineAdapter
                    dialog_manager = None
                    pipeline = None
                    
                    logger.info(f"web_gui_instance: {web_gui_instance}")
                    logger.info(f"web_gui_instance.brain: {getattr(web_gui_instance, 'brain', 'NO ATTR')}")
                    
                    # 1. Проверяем HybridKnowledgeDialogManager
                    if (web_gui_instance and 
                        hasattr(web_gui_instance, 'brain') and
                        web_gui_instance.brain and
                        hasattr(web_gui_instance.brain, 'hybrid_dialog_manager') and
                        web_gui_instance.brain.hybrid_dialog_manager and
                        web_gui_instance.brain.hybrid_dialog_manager.initialized):
                        dialog_manager = web_gui_instance.brain.hybrid_dialog_manager
                        logger.info(f"Using HybridKnowledgeDialogManager: {type(dialog_manager)}")
                    
                    # 2. Fallback на two_model_pipeline
                    elif (web_gui_instance and 
                        hasattr(web_gui_instance, 'brain') and
                        web_gui_instance.brain and
                        hasattr(web_gui_instance.brain, 'two_model_pipeline') and
                        web_gui_instance.brain.two_model_pipeline):
                        pipeline = web_gui_instance.brain.two_model_pipeline
                        logger.info(f"Using PipelineAdapter: {type(pipeline)}")
                    
                    if not dialog_manager and not pipeline:
                        error_msg = 'Pipeline не найден - GUI brain не инициализирован'
                        logger.error(error_msg)
                        yield f"data: {json.dumps({'type': 'error', 'text': error_msg})}\n\n"
                        return
                    
                    # Используем HybridKnowledgeDialogManager если доступен
                    if dialog_manager:
                        logger.info("Processing with HybridKnowledgeDialogManager streaming...")
                        try:
                            # Используем streaming метод
                            for chunk_data in dialog_manager.generate_streaming(
                                user_input=message,
                                max_tokens=4096,
                                temperature=0.6,
                                chunk_size=25
                            ):
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                            return
                            
                        except Exception as e:
                            logger.error(f"HybridKnowledgeDialogManager error: {e}")
                            yield f"data: {json.dumps({'type': 'error', 'text': f'Ошибка: {str(e)[:100]}'})}\n\n"
                            return
                    
                    # Fallback на streaming
                    if not hasattr(pipeline, 'generate_streaming'):
                        error_msg = 'Pipeline не поддерживает streaming'
                        logger.error(error_msg)
                        yield f"data: {json.dumps({'type': 'error', 'text': error_msg})}\n\n"
                        return
                    
                    # Отправляем начало
                    yield f"data: {json.dumps({'type': 'start', 'timestamp': time.time()})}\n\n"
                    
                    # Проверяем нужен ли веб-поиск
                    search_results = []
                    enhanced_message = message
                    
                    try:
                        from eva_ai.core.brain_query import needs_web_search
                        web_search = None
                        
                        if web_gui_instance and web_gui_instance.brain:
                            web_search = getattr(web_gui_instance.brain, 'web_search_engine', None)
                        
                        need_search, search_reason = needs_web_search(message)
                        logger.info(f"[STREAM] Web search check: need_search={need_search}, reason={search_reason}")
                        
                        if web_search and hasattr(web_search, 'search') and need_search:
                            try:
                                search_result = web_search.search(message[:200], max_results=3)
                                if search_result:
                                    raw_results = search_result.get('results', []) if isinstance(search_result, dict) else []
                                    for sr in raw_results:
                                        if hasattr(sr, 'title'):
                                            search_results.append({
                                                'title': str(sr.title) if sr.title else '',
                                                'url': str(sr.url) if sr.url else '',
                                                'snippet': str(sr.snippet) if sr.snippet else ''
                                            })
                                        elif isinstance(sr, dict):
                                            search_results.append(sr)
                                    
                                    if search_results:
                                        web_context = "\n\nИнформация из интернета:\n"
                                        for i, sr in enumerate(search_results[:3]):
                                            title = sr.get('title', 'No title')[:100]
                                            snippet = sr.get('snippet', '')[:300]
                                            web_context += f"\n{i+1}. {title}: {snippet}..."
                                        enhanced_message = f"{message}\n\n{web_context}\n\nДай ответ используя эту информацию."
                                        logger.info(f"[STREAM] Web search found {len(search_results)} results, enhanced prompt")
                            except Exception as e:
                                logger.error(f"[STREAM] Web search error: {e}")
                    except ImportError:
                        logger.warning("[STREAM] Could not import needs_web_search")
                    
                    # Генерируем со стримингом чанков
                    full_text = ""
                    chunk_count = 0
                    
                    for chunk_data in pipeline.generate_streaming(
                        prompt=enhanced_message,
                        max_tokens=4096,
                        temperature=0.6,
                        chunk_size=20,
                        task_type="context"
                    ):
                        chunk_type = chunk_data.get('type', 'chunk')
                        chunk_text = chunk_data.get('text', '')
                        
                        if chunk_text:
                            full_text += chunk_text
                            chunk_count += 1
                        
                        # Отправляем чанк клиенту
                        yield f"data: {json.dumps({
                            'type': chunk_type,
                            'text': chunk_text,
                            'is_final': chunk_data.get('is_final', False),
                            'tokens_count': chunk_data.get('tokens_count', 0),
                            'elapsed_ms': chunk_data.get('elapsed_ms', 0),
                            'chunk_index': chunk_data.get('chunk_index', 0),
                            'total_chunks': chunk_data.get('total_chunks', 0),
                            'progress': chunk_data.get('progress', 0)
                        })}\n\n"
                        
                        # Если это финальный чанк - выходим
                        if chunk_data.get('is_final', False):
                            break
                    
                    # Сохраняем в историю
                    if session_id and full_text:
                        web_gui_instance.session_manager.add_chat_message(
                            session_id, 'assistant', full_text
                        )
                    
                    # Отправляем завершение
                    yield f"data: {json.dumps({'type': 'done', 'full_text': full_text, 'chunks_sent': chunk_count})}\n\n"
                    
                except Exception as e:
                    logger.error(f"Streaming error: {e}", exc_info=True)
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

    # ===== SESSION ROUTES =====
    
    @app.route('/api/sessions', methods=['GET', 'POST', 'DELETE'])
    def api_sessions():
        """Manage user sessions."""
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
        """Get, update or delete a specific session."""
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
    
    logger.info("Chat routes registered")
