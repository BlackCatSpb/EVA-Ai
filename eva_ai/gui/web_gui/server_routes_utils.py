"""
Утилиты для Web GUI routes
"""
import os
import logging
import json
import uuid
import time
from datetime import datetime

from flask import request, jsonify

try:
    from eva_ai.core.api_compat import API_VERSION, API_PREFIX, api_version
except ImportError:
    API_VERSION = "v1"
    API_PREFIX = f"/api/{API_VERSION}"
    def api_version(func):
        return func

logger = logging.getLogger("eva_ai.webgui")

# Tesseract configuration
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_PREFIX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'core', 'tessdata')
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    os.environ['TESSDATA_PREFIX'] = TESSDATA_PREFIX
    logger.info("Tesseract configured at: {} with tessdata: {}".format(TESSERACT_PATH, TESSDATA_PREFIX))
except Exception as e:
    logger.warning("Failed to configure Tesseract: {}".format(e))


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


def validate_json_request():
    """Валидирует JSON в запросе."""
    data = request.get_json()
    if not data:
        return None, jsonify({'error': 'Invalid JSON'}), 400
    return data, None, None


def check_brain_initialized(web_gui_instance):
    """Проверяет инициализацию brain."""
    if not web_gui_instance or not web_gui_instance.brain:
        return False, jsonify({'error': 'Brain не инициализирован'}), 500
    return True, None, None


def get_brain_components(web_gui_instance):
    """Получает словарь компонентов brain."""
    if not web_gui_instance or not web_gui_instance.brain:
        return {}
    
    brain = web_gui_instance.brain
    components = {}
    
    component_list = [
        'memory_manager', 'self_dialog_learning', 'hybrid_cache',
        'knowledge_graph', 'web_search_engine', 'self_reasoning_engine',
        'two_model_pipeline', 'llama_cpp_deployment', 'qwen_model_manager',
        'contradiction_manager', 'concept_extractor', 'concept_miner',
        'graph_curator', 'deferred_system', 'event_bus', 'metrics_manager'
    ]
    
    for comp in component_list:
        if hasattr(brain, comp):
            obj = getattr(brain, comp)
            components[comp] = {
                'available': obj is not None,
                'type': type(obj).__name__ if obj else None
            }
    
    return components
