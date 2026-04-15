"""
Upload and file processing routes
Handles file uploads, text extraction, entities, feedback
"""
import os
import uuid
import logging
from flask import jsonify, request

logger = logging.getLogger("eva_ai.webgui.routes_upload")

# Tesseract path for OCR
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_PREFIX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'core', 'tessdata')
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    os.environ['TESSDATA_PREFIX'] = TESSDATA_PREFIX
    logger.info(f"Tesseract configured at: {TESSERACT_PATH} with tessdata: {TESSDATA_PREFIX}")
except Exception as e:
    logger.warning(f"Failed to configure Tesseract: {e}")


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
        return f"[Ошибка извлечения текста: {e}]"


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
