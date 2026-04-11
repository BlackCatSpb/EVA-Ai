"""
Knowledge routes: documents, knowledge-graph, settings, snapshots
"""
import os
import logging
import time
import json
from datetime import datetime

from flask import jsonify, request, send_file

logger = logging.getLogger("eva_ai.webgui.routes_knowledge")

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def register_knowledge_routes(app, web_gui_instance):
    """Register knowledge routes."""
    logger.info("Registering knowledge routes...")

    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    except Exception as e:
        logger.warning(f"Failed to configure Tesseract: {e}")

    @app.route('/api/documents', methods=['GET'])
    def api_documents():
        """Get list of documents."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        docs = []
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'documents')
        
        if os.path.exists(docs_dir):
            for f in os.listdir(docs_dir):
                filepath = os.path.join(docs_dir, f)
                if os.path.isfile(filepath):
                    docs.append({
                        'id': f,
                        'name': f,
                        'size': os.path.getsize(filepath),
                        'modified': os.path.getmtime(filepath)
                    })
        
        return jsonify({'documents': docs})

    @app.route('/api/documents/<file_id>', methods=['DELETE'])
    def api_document_delete(file_id):
        """Delete a document."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'documents', file_id)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True})
        
        return jsonify({'error': 'Document not found'}), 404

    @app.route('/api/documents/memory', methods=['GET'])
    def api_documents_memory():
        """Get documents from memory."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        brain = web_gui_instance.brain
        docs = []
        
        mm = getattr(brain, 'memory_manager', None)
        if mm and hasattr(mm, 'get_documents'):
            try:
                docs = mm.get_documents()
            except Exception as e:
                logger.error(f"get_documents error: {e}")
        
        return jsonify({'documents': docs})

    @app.route('/api/documents/memory/<document_id>', methods=['GET', 'DELETE'])
    def api_document_memory(document_id):
        """Get or delete a document from memory."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        brain = web_gui_instance.brain
        mm = getattr(brain, 'memory_manager', None)
        
        if request.method == 'GET':
            if mm and hasattr(mm, 'get_document'):
                try:
                    doc = mm.get_document(document_id)
                    if doc:
                        return jsonify(doc)
                except Exception as e:
                    logger.error(f"get_document error: {e}")
            return jsonify({'error': 'Document not found'}), 404
        
        elif request.method == 'DELETE':
            if mm and hasattr(mm, 'delete_document'):
                try:
                    mm.delete_document(document_id)
                    return jsonify({'success': True})
                except Exception as e:
                    logger.error(f"delete_document error: {e}")
            return jsonify({'error': 'Delete not available'}), 500

    @app.route('/api/knowledge-graph', methods=['GET', 'POST'])
    def api_knowledge_graph():
        """Get or update knowledge graph (FGv2)."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        brain = web_gui_instance.brain
        
        if request.method == 'GET':
            fg = getattr(brain, 'fractal_graph_v2', None) or getattr(brain, 'knowledge_graph', None)
            if fg:
                try:
                    if hasattr(fg, 'get_nodes_list') and hasattr(fg, 'get_edges_list'):
                        return jsonify({
                            'nodes': fg.get_nodes_list(),
                            'edges': fg.get_edges_list()
                        })
                    elif hasattr(fg, 'get_all'):
                        return jsonify(fg.get_all())
                except Exception as e:
                    logger.error(f"knowledge_graph.get_all error: {e}")
            
            return jsonify({'nodes': [], 'edges': []})
        
        elif request.method == 'POST':
            data = request.get_json() or {}
            node_type = data.get('type', 'concept')
            content = data.get('content', '')
            
            if not content:
                return jsonify({'error': 'Content required'}), 400
            
            if hasattr(brain, 'add_knowledge'):
                try:
                    node_id = brain.add_knowledge(node_type, content)
                    return jsonify({'success': True, 'id': node_id})
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
            
            return jsonify({'error': 'add_knowledge not available'}), 500

    @app.route('/api/cache-stats')
    def api_cache_stats():
        """Get cache statistics."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        stats = {'hit_rate': 0, 'miss_rate': 0, 'size': 0}
        
        brain = web_gui_instance.brain
        if hasattr(brain, 'get_cache_stats'):
            try:
                stats = brain.get_cache_stats()
            except Exception as e:
                logger.error(f"get_cache_stats error: {e}")
        
        return jsonify(stats)

    @app.route('/api/settings', methods=['GET', 'POST'])
    def api_settings():
        """Get or update settings."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        if request.method == 'GET':
            settings = web_gui_instance.get_settings()
            return jsonify(settings)
        
        elif request.method == 'POST':
            data = request.get_json() or {}
            web_gui_instance.update_settings(data)
            return jsonify({'success': True})

    @app.route('/api/snapshots', methods=['GET', 'POST'])
    def api_snapshots():
        """Get or create memory snapshots."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        brain = web_gui_instance.brain
        
        if request.method == 'GET':
            snapshots = []
            
            if hasattr(brain, 'get_snapshots'):
                try:
                    snapshots = brain.get_snapshots()
                except Exception as e:
                    logger.error(f"get_snapshots error: {e}")
            
            return jsonify({'snapshots': snapshots})
        
        elif request.method == 'POST':
            data = request.get_json() or {}
            name = data.get('name', f'snapshot_{int(time.time())}')
            
            if hasattr(brain, 'create_snapshot'):
                try:
                    snapshot_id = brain.create_snapshot(name)
                    return jsonify({'success': True, 'id': snapshot_id})
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
            
            return jsonify({'error': 'create_snapshot not available'}), 500

    @app.route('/api/file-content', methods=['POST'])
    def api_file_content():
        """Extract text content from a file."""
        data = request.get_json() or {}
        filepath = data.get('path', '')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        elif ext == '.pdf':
            try:
                import pytesseract
                from pdf2image import convert_from_path
                pages = convert_from_path(filepath)
                ocr_text = ''
                for page in pages:
                    page_text = pytesseract.image_to_string(page, lang='rus+eng')
                    ocr_text += page_text + '\n'
                if ocr_text.strip():
                    return ocr_text
            except Exception as e:
                logger.warning(f"Tesseract OCR failed: {e}")
            return "[PDF OCR failed]"
        
        elif ext == '.docx':
            try:
                from docx import Document
                doc = Document(filepath)
                text = '\n'.join([p.text for p in doc.paragraphs])
                return text
            except ImportError:
                return "[DOCX: python-docx not installed]"
        
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            try:
                import pytesseract
                from PIL import Image
                img = Image.open(filepath)
                text = pytesseract.image_to_string(img, lang='rus+eng')
                return text
            except ImportError:
                return "[Image OCR not available]"
        
        elif ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.md', '.rst', '.csv', '.log', '.ini', '.cfg', '.conf']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        else:
            return f"[Unsupported format: {ext}]"

    logger.info("Knowledge routes registered")
