"""Flask сервер для Web GUI ЕВА — маршруты и обработчики API."""
import os
import uuid
import logging
from datetime import datetime

from flask import jsonify, request

logger = logging.getLogger("eva_ai.webgui")

from .server_main import web_gui_instance, app, extract_text_from_file


# ========================================================================
# Index
# ========================================================================

@app.route('/')
def index():
    return app.send_static_file('index.html') if os.path.exists(
        os.path.join(app.static_folder or '', 'index.html')
    ) else app.response_class("Template not found", status=404)


# ========================================================================
# Auth
# ========================================================================

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
            if existing_sessions:
                session_id = existing_sessions[0]['id']
                web_gui_instance.session_manager.update_session(session_id, {})
            else:
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


# ========================================================================
# Sessions
# ========================================================================

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
        web_gui_instance.session_manager.delete_session(session_id)
        sessions = web_gui_instance.session_manager.get_user_sessions(user_id)
        return jsonify({'sessions': sessions})


@app.route('/api/session/<session_id>', methods=['GET'])
def api_session(session_id):
    if not web_gui_instance:
        return jsonify({'error': 'Сервер не инициализирован'}), 500

    session = web_gui_instance.session_manager.get_session(session_id)
    if session:
        return jsonify({
            'session': session,
            'context': session.get('context_nodes', []),
            'entities': session.get('entities', [])
        })
    return jsonify({'error': 'Сессия не найдена'}), 404


# ========================================================================
# Upload
# ========================================================================

@app.route('/api/upload', methods=['POST'])
def api_upload():
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


# ========================================================================
# Chat - MOVED to gui/web_gui/server_routes_chat.py
# ========================================================================

# ========================================================================
# Entities
# ========================================================================

@app.route('/api/entities/<session_id>', methods=['GET'])
def api_entities(session_id):
    if not web_gui_instance:
        return jsonify({'error': 'Сервер не инициализирован'}), 500

    session = web_gui_instance.session_manager.get_session(session_id)
    if session:
        return jsonify({
            'entities': session.get('entities', []),
            'context_count': len(session.get('context_nodes', []))
        })
    return jsonify({'error': 'Сессия не найдена'}), 404


# ========================================================================
# Feedback
# ========================================================================

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400

        rating = data.get('rating', 0)
        message_text = data.get('message_text', '')
        message_index = data.get('message_index', 0)
        
        explicit_accuracy = data.get('explicit_accuracy')
        coherence_score = data.get('coherence_score')
        helpfulness = data.get('helpfulness')
        toxicity = data.get('toxicity')
        corrected_answer = data.get('corrected_answer')
        preferred_response = data.get('preferred_response')
        reasoning_quality = data.get('reasoning_quality')

        logger.info(f"User feedback: rating={rating}, index={message_index}, accuracy={explicit_accuracy}")

        if web_gui_instance.brain:
            try:
                if hasattr(web_gui_instance.brain, 'trigger_subjective_correctness'):
                    web_gui_instance.brain.trigger_subjective_correctness(
                        message_text=message_text,
                        rating=rating
                    )
            except Exception as e:
                logger.debug(f"Feedback brain trigger error: {e}")
            
            try:
                if hasattr(web_gui_instance.brain, 'feedback_processor'):
                    feedback_data = {
                        'rating': rating,
                        'message_index': message_index,
                        'explicit_accuracy': explicit_accuracy if explicit_accuracy is not None else 0.5,
                        'coherence_score': coherence_score if coherence_score is not None else 0.5,
                        'helpfulness': helpfulness if helpfulness is not None else 0.5,
                        'toxicity': toxicity if toxicity is not None else 0.0,
                        'corrected_answer': corrected_answer,
                        'preferred_response': preferred_response,
                        'reasoning_quality': reasoning_quality if reasoning_quality is not None else 0.5
                    }
                    web_gui_instance.brain.feedback_processor.process_feedback(feedback_data)
            except Exception as e:
                logger.debug(f"Feedback processor error: {e}")

        return jsonify({'success': True, 'rating': rating})

    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({'error': str(e)}), 500


# ========================================================================
# Status
# ========================================================================

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


# ========================================================================
# Metrics
# ========================================================================

@app.route('/api/metrics')
def api_metrics():
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})

    metrics = {
        'cpu_usage': 0.0,
        'memory_usage': 0.0,
        'cache_hit_rate': 0.0,
        'timestamp': datetime.now().isoformat()
    }

    try:
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


# ========================================================================
# Memory Graph
# ========================================================================

@app.route('/api/memory-graph')
def api_memory_graph():
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})

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
