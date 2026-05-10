"""
Upload and file processing routes
Handles file uploads, text extraction, entities, feedback
"""
import os
import uuid
import logging
from flask import jsonify, request

logger = logging.getLogger("eva_ai.webgui.routes_upload")

from eva_ai.gui.web_gui.server_routes_utils import extract_text_from_file, setup_tesseract

setup_tesseract()


def register_upload_routes(app, web_gui_instance):
    """Register upload and entity routes."""
    logger.info("Registering upload routes...")

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
        """Get or delete entities for a session."""
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

    logger.info("Upload routes registered")
