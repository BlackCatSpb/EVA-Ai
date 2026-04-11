"""
Chat routes для Web GUI
Chat endpoints, streaming
"""
import json
import logging
import threading
import time
from flask import jsonify, request, Response
from eva_ai.core.api_compat import API_VERSION, API_PREFIX, api_version

logger = logging.getLogger("eva_ai.webgui")


def register_chat_routes(app, web_gui_instance):
    """Регистрирует chat роуты"""
    
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
                        web_gui_instance.session_manager.add_chat_message(
                            session_id, 'assistant', full_text
                        )
                    
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
    
    logger.info("Chat routes registered")
