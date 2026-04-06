"""Flask сервер для Web GUI ЕВА — ядро, инициализация, жизненный цикл."""
import os
import logging
import threading
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, render_template, jsonify, request

logger = logging.getLogger("eva.webgui")

app = Flask(__name__,
            template_folder='eva/templates',
            static_folder='eva/static',
            static_url_path='/static')


def _get_secret_key():
    env_key = os.environ.get('COGNIFLEX_SECRET_KEY')
    if env_key:
        return env_key
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eva_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if config.get('web_gui', {}).get('secret_key'):
                    return config['web_gui']['secret_key']
        except Exception:
            pass
    return os.urandom(32).hex()


app.config['SECRET_KEY'] = _get_secret_key()
app.config['JSON_AS_ASCII'] = False

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    logger.info(f"Tesseract configured at: {TESSERACT_PATH}")
except Exception as e:
    logger.warning(f"Failed to configure Tesseract: {e}")


# ========================================================================
# SessionManager
# ========================================================================

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self._lock = threading.Lock()
        gui_dir = os.path.dirname(os.path.dirname(__file__))
        self._storage_file = os.path.join(gui_dir, 'eva_gui_cache', 'sessions.json')
        self._ensure_storage_dir()
        self._load_sessions()

    def _ensure_storage_dir(self):
        os.makedirs(os.path.dirname(self._storage_file), exist_ok=True)

    def _load_sessions(self):
        try:
            if os.path.exists(self._storage_file):
                with open(self._storage_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Ошибка парсинга JSON сессий: {e}")
                        self.sessions = {}
                        return
                    self.sessions = data.get('sessions', {})
                    logger.info(f"Загружено {len(self.sessions)} сессий из хранилища")
        except Exception as e:
            logger.warning(f"Не удалось загрузить сессии: {e}")

    def _save_sessions(self):
        try:
            with open(self._storage_file, 'w', encoding='utf-8') as f:
                json.dump({'sessions': self.sessions}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Не удалось сохранить сессии: {e}")

    def create_session(self, user_id: str, session_name: str = None) -> str:
        with self._lock:
            session_id = str(uuid.uuid4())
            if session_name is None:
                session_name = f"Сессия {len(self.sessions) + 1}"
            self.sessions[session_id] = {
                'id': session_id, 'user_id': user_id, 'name': session_name,
                'created_at': datetime.now().isoformat(),
                'last_active': datetime.now().isoformat(),
                'context_nodes': [], 'entities': []
            }
            self._save_sessions()
            return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        with self._lock:
            return self.sessions.get(session_id)

    def get_user_sessions(self, user_id: str) -> list:
        with self._lock:
            return [s for s in self.sessions.values() if s['user_id'] == user_id]

    def update_session(self, session_id: str, data: Dict):
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(data)
                self.sessions[session_id]['last_active'] = datetime.now().isoformat()
                self._save_sessions()

    def delete_session(self, session_id: str):
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                self._save_sessions()

    def add_context_node(self, session_id: str, node: Dict):
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]['context_nodes'].append(node)
                self._save_sessions()

    def add_entity(self, session_id: str, entity: Dict):
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]['entities'].append(entity)
                self._save_sessions()


# ========================================================================
# AuthManager
# ========================================================================

class AuthManager:
    def __init__(self):
        self.users = {}
        self._lock = threading.Lock()

    def set_default_credentials(self, username: str, password: str):
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        with self._lock:
            self.users[username] = {
                'username': username,
                'password_hash': password_hash,
                'created_at': datetime.now().isoformat()
            }

    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        with self._lock:
            user = self.users.get(username)
            if user:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                if password_hash == user['password_hash']:
                    if 'user_id' not in user:
                        user['user_id'] = str(uuid.uuid4())
                    return {'username': username, 'user_id': user['user_id']}
        return None


# ========================================================================
# EntityExtractor
# ========================================================================

class EntityExtractor:
    def __init__(self):
        self.entity_keywords = {
            'project': ['проект', 'задача', 'цель'],
            'person': ['человек', 'люди', 'коллега'],
            'location': ['место', 'город', 'страна'],
            'organization': ['компания', 'организация', 'команда'],
            'event': ['событие', 'встреча', 'конференция'],
            'concept': ['идея', 'концепция', 'теория'],
            'fact': ['факт', 'данные', 'информация'],
            'preference': ['нравится', 'предпочитаю'],
            'skill': ['умею', 'навык', 'опыт'],
        }

    def extract_entities(self, text: str) -> list:
        entities = []
        text_lower = text.lower()
        for entity_type, keywords in self.entity_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    entities.append({
                        'type': entity_type, 'keyword': keyword,
                        'extracted_at': datetime.now().isoformat(),
                        'context': text[:200]
                    })
                    break
        return entities

    def is_personal_info(self, text: str) -> bool:
        personal_keywords = ['пароль', 'номер карты', 'адрес', 'телефон', 'email', 'день рождения']
        text_lower = text.lower()
        return any(kw in text_lower for kw in personal_keywords)


# ========================================================================
# EthicsChecker
# ========================================================================

class EthicsChecker:
    def __init__(self):
        self.blocked_patterns = ['взлом', 'хакер', 'вирус', 'наркотик', 'оружие', 'убийство']

    def check_message(self, text: str) -> Dict[str, Any]:
        result = {'allowed': True, 'warnings': [], 'blocked': False, 'reason': None}
        text_lower = text.lower()
        for pattern in self.blocked_patterns:
            if pattern in text_lower:
                result['allowed'] = False
                result['blocked'] = True
                result['reason'] = f'Заблокировано: {pattern}'
                return result
        return result

    def sanitize_entity(self, entity: Dict) -> Dict:
        if entity.get('type') in ['person', 'preference']:
            entity['sanitized'] = True
            entity['contains_pii'] = True
            entity['display_name'] = 'Пользователь'
        return entity


# ========================================================================
# WebGUI
# ========================================================================

class WebGUI:
    def __init__(self, brain=None, integrator=None, host='127.0.0.1', port=5555):
        self.brain = brain
        self.integrator = integrator
        self.host = host
        self.port = port
        self.running = False
        self.thread = None

        self.auth_manager = AuthManager()
        self.session_manager = SessionManager()
        self.entity_extractor = EntityExtractor()
        self.ethics_checker = EthicsChecker()

        admin_user = os.environ.get('COGNIFLEX_ADMIN_USER', 'admin')
        admin_pass = os.environ.get('COGNIFLEX_ADMIN_PASS')
        if not admin_pass:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eva_config.json')
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        admin_pass = config.get('web_gui', {}).get('admin_password')
                except Exception:
                    pass
        if not admin_pass:
            admin_pass = 'admin'
        self.auth_manager.set_default_credentials(admin_user, admin_pass)

        logger.info(f"WebGUI инициализирован на {host}:{port}")

    def process_message(self, query: str, session_id: str, user_id: str, file_data: Dict = None) -> Dict[str, Any]:
        if file_data and file_data.get('extracted_text'):
            filename = file_data.get('filename', 'file')
            extracted = file_data['extracted_text']
            query = f"""Пользователь прикрепил файл "{filename}".
Содержимое файла:
---
{extracted}
---

Запрос пользователя: {query}"""

        ethics_result = self.ethics_checker.check_message(query)
        if not ethics_result['allowed']:
            return {
                'response': 'Извините, это сообщение заблокировано.',
                'status': 'blocked',
                'reason': ethics_result.get('reason')
            }

        entities = self.entity_extractor.extract_entities(query)

        if session_id:
            context_node = {
                'id': str(uuid.uuid4()), 'user_message': query,
                'timestamp': datetime.now().isoformat(),
                'entities': entities, 'file_data': file_data
            }
            self.session_manager.add_context_node(session_id, context_node)
            for entity in entities:
                if not self.entity_extractor.is_personal_info(entity.get('context', '')):
                    sanitized = self.ethics_checker.sanitize_entity(entity)
                    self.session_manager.add_entity(session_id, sanitized)

        response_text = "Система обрабатывает запрос..."
        result = None
        debug_info = {"brain": self.brain is not None, "has_process_query": False}

        conversation_history = []
        if session_id and isinstance(session_id, str) and session_id.strip():
            session = self.session_manager.get_session(session_id)
            if session and isinstance(session, dict) and 'context_nodes' in session:
                context_nodes = session.get('context_nodes')
                if context_nodes and isinstance(context_nodes, list):
                    user_msg = None
                    for node in context_nodes[-20:]:
                        if isinstance(node, dict):
                            if 'user_message' in node and node['user_message']:
                                user_msg = node['user_message']
                            elif 'assistant_message' in node and node['assistant_message']:
                                if user_msg:
                                    conversation_history.append({"role": "user", "content": user_msg})
                                    conversation_history.append({"role": "assistant", "content": node['assistant_message']})
                                    user_msg = None
                    logger.debug(f"Loaded {len(conversation_history)} messages for context")

        user_context = {
            'session_id': session_id, 'user_id': user_id,
            'conversation_history': conversation_history
        }

        result = None
        if self.integrator:
            result = self.integrator.process_query(query, user_context)
            if result and isinstance(result, dict):
                response_text = result.get('response', response_text)
        elif self.brain and hasattr(self.brain, 'process_query'):
            debug_info["has_process_query"] = True
            debug_info["brain_loaded"] = self.brain is not None and hasattr(self.brain, 'self_reasoning_engine') and self.brain.self_reasoning_engine is not None
            debug_info["enhanced_reasoning_loaded"] = self.brain is not None and hasattr(self.brain, 'enhanced_reasoning_engine') and self.brain.enhanced_reasoning_engine is not None
            result = self.brain.process_query(query, user_context)
            debug_info["result_keys"] = list(result.keys()) if result and isinstance(result, dict) else []
            debug_info["result_reasoning"] = str(result.get('reasoning'))[:100] if result and isinstance(result, dict) else None
            debug_info["result_source"] = result.get('source') if result and isinstance(result, dict) else None
            logger.debug(f"brain result: source={result.get('source') if result and isinstance(result, dict) else 'None'}")
            if result and isinstance(result, dict):
                response_text = result.get('response', result.get('text', response_text))
        else:
            debug_info["reason"] = "no brain or no process_query"

        if query.strip().lower() == "debug123":
            return {
                'response': 'Debug info: ' + str(debug_info),
                'status': 'ok',
                'reasoning': str(debug_info)
            }

        brain_reasoning = None
        reasoning_data = None
        reasoning_steps = []

        if result and isinstance(result, dict):
            brain_reasoning = result.get('reasoning')
            brain_reasoning_raw = result.get('reasoning_raw')
            source = result.get('source', '')
            confidence = result.get('confidence', 0)
            search_results = result.get('search_results', [])
            if ethics_result is None and result.get('ethics_result'):
                ethics_result = result.get('ethics_result')

            web_search_info = None
            if search_results and len(search_results) > 0:
                web_search_info = f"Найдено {len(search_results)} результатов:"
                for i, sr in enumerate(search_results[:3]):
                    title = sr.get('title', 'No title')[:60]
                    url = sr.get('url', '')[:50]
                    web_search_info += f"\n{i+1}. {title}... ({url})"

            if source == 'llama_cpp_with_modules' or (file_data and file_data.get('extracted_text')):
                reasoning_steps = []
                if file_data and file_data.get('extracted_text'):
                    filename = file_data.get('filename', 'file')
                    text_len = len(file_data.get('extracted_text', ''))
                    reasoning_steps.append({
                        'step': 0, 'phase': 'document_analysis',
                        'thought': f'Анализ документа "{filename}" - извлечено {text_len} символов',
                        'confidence': 0.9
                    })
                reasoning_steps.append({
                    'step': len(reasoning_steps) + 1, 'phase': 'generation',
                    'thought': 'Первичная генерация ответа через LlamaCpp (GGUF)',
                    'confidence': 0.5
                })
                contr_result = result.get('contradiction_result')
                contr_count = 0
                if contr_result:
                    contr_count = contr_result.get('significant_count', 0)
                    contr_conf = 1.0 - contr_result.get('contradiction_level', 0.0)
                    reasoning_steps.append({
                        'step': 2, 'phase': 'contradiction_check',
                        'thought': f'Проверка противоречий: {contr_count} найдено, уровень={contr_result.get("contradiction_level", 0):.2f}',
                        'confidence': contr_conf
                    })
                if ethics_result is None:
                    ethics_result = result.get('ethics_result')
                has_violations = False
                if ethics_result:
                    has_violations = ethics_result.get('has_violations', False)
                    ethics_conf = ethics_result.get('is_ethical', 1.0)
                    reasoning_steps.append({
                        'step': 3, 'phase': 'ethics_check',
                        'thought': f'Проверка этики: violations={has_violations}, score={ethics_conf:.2f}',
                        'confidence': ethics_conf
                    })
                if search_results and len(search_results) > 0:
                    reasoning_steps.append({
                        'step': 4, 'phase': 'web_search',
                        'thought': f'Веб-поиск: найдено {len(search_results)} результатов',
                        'confidence': 0.8
                    })
                    reasoning_steps.append({
                        'step': 5, 'phase': 'refinement',
                        'thought': 'Перегенерация с контекстом из веб-поиска',
                        'confidence': 0.9
                    })
                elif contr_count > 0 or (ethics_result and has_violations):
                    reasoning_steps.append({
                        'step': 4, 'phase': 'refinement',
                        'thought': 'Перегенерация после исправления модулей',
                        'confidence': 0.7
                    })
                reasoning_steps.append({
                    'step': len(reasoning_steps) + 1, 'phase': 'final_synthesis',
                    'thought': 'Финальный ответ с учетом всех проверок',
                    'confidence': result.get('confidence', 0.9)
                })
                reasoning_data = "Рассуждения системы (qwen_only_mode):\n\n" + "\n".join([
                    f"{s['step']}. [{s['phase']}] {s['thought']} (conf: {s['confidence']:.2f})"
                    for s in reasoning_steps
                ])

            if source == 'self_reasoning_engine':
                if brain_reasoning_raw and isinstance(brain_reasoning_raw, dict):
                    steps = brain_reasoning_raw.get('steps', [])
                    if steps:
                        reasoning_text = "Рассуждения системы:\n\n"
                        for i, step in enumerate(steps):
                            phase = step.get('phase', 'unknown')
                            thought = step.get('thought', '')
                            conf = step.get('confidence', 0)
                            reasoning_steps.append({
                                'step': i + 1, 'phase': phase,
                                'thought': thought, 'confidence': conf
                            })
                            if i < 10:
                                reasoning_text += f"{i+1}. [{phase}] {thought} (conf: {conf:.2f})\n"
                        reasoning_data = reasoning_text
                    elif brain_reasoning:
                        reasoning_data = str(brain_reasoning)
                elif brain_reasoning:
                    reasoning_data = str(brain_reasoning)

            elif source == 'enhanced_reasoning_engine':
                if brain_reasoning_raw and isinstance(brain_reasoning_raw, dict):
                    chain = brain_reasoning_raw.get('reasoning_chain', [])
                    if chain:
                        reasoning_text = "Регенерация ответа:\n\n"
                        for i, iteration in enumerate(chain):
                            resp_preview = iteration.get('response', '')[:80]
                            conf = iteration.get('confidence', 0)
                            has_contr = iteration.get('has_contradictions', False)
                            has_ethics = iteration.get('has_ethics_issues', False)
                            module_prompts = iteration.get('module_prompts', {})
                            prompts_text = ""
                            if module_prompts:
                                prompts_text = "\nПромты модулей:"
                                for mod, prompt in module_prompts.items():
                                    prompts_text += f"\n  [{mod.upper()}]: {prompt[:100]}..."
                            reasoning_steps.append({
                                'step': i + 1, 'phase': 'regeneration',
                                'thought': resp_preview, 'confidence': conf,
                                'has_contradictions': has_contr,
                                'has_ethics_issues': has_ethics,
                                'module_prompts': module_prompts
                            })
                            status = ""
                            if has_contr:
                                status += " [противоречия]"
                            if has_ethics:
                                status += " [этика]"
                            reasoning_text += f"Итерация {i+1}: {resp_preview}...{status}\n"
                            reasoning_text += f"  Уверенность: {conf:.2f}\n"
                            if prompts_text:
                                reasoning_text += f"  {prompts_text}\n"
                            reasoning_text += "\n"
                        reasoning_data = reasoning_text
                    elif brain_reasoning:
                        reasoning_data = str(brain_reasoning)
                elif brain_reasoning:
                    reasoning_data = str(brain_reasoning)

            elif source == 'qwen_model' and confidence >= 0.85:
                reasoning_data = f"Qwen Model обработал запрос (уверенность: {confidence:.2f})"

        if session_id and response_text:
            response_node = {
                'id': str(uuid.uuid4()), 'assistant_message': response_text,
                'timestamp': datetime.now().isoformat(), 'reasoning': reasoning_data
            }
            self.session_manager.add_context_node(session_id, response_node)

        self_dialog_result = None
        if self.brain and hasattr(self.brain, 'self_dialog_learning') and self.brain.self_dialog_learning:
            try:
                self.brain.self_dialog_learning.create_dialog(
                    topic=query[:100],
                    context={
                        "user_query": query,
                        "system_response": response_text[:200] if response_text else "",
                        "source": "web_gui_chat"
                    }
                )
                logger.info(f"Запущен самодиалог для темы: {query[:50]}...")
                if hasattr(self.brain.self_dialog_learning, 'get_recent_learning'):
                    recent_dialogs = self.brain.self_dialog_learning.get_recent_learning(limit=1)
                    if recent_dialogs:
                        self_dialog_result = recent_dialogs[0]
            except Exception as e:
                logger.debug(f"Error triggering self-dialog: {e}")

        return_data = {
            'response': response_text, 'status': 'ok',
            'warnings': ethics_result.get('warnings', []) if ethics_result else [],
            'reasoning': reasoning_data, 'reasoning_steps': reasoning_steps,
            'self_dialog': self_dialog_result,
            'search_results': search_results if search_results else None,
            'web_search_info': web_search_info
        }
        return return_data

    def start(self):
        if self.running:
            return
        self.running = True

        def run():
            import click
            click.echo = lambda *args, **kwargs: None
            app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        logger.info(f"WebGUI сервер запущен на http://{self.host}:{self.port}")

    def stop(self):
        self.running = False
        logger.info("WebGUI сервер остановлен")


# ========================================================================
# Global instance & factory
# ========================================================================

web_gui_instance: Optional[WebGUI] = None


def create_app(brain=None, integrator=None, host='127.0.0.1', port=5555):
    global web_gui_instance
    web_gui_instance = WebGUI(brain=brain, integrator=integrator, host=host, port=port)
    web_gui_instance.start()
    return web_gui_instance


def get_app() -> WebGUI:
    return web_gui_instance


# ========================================================================
# File text extraction
# ========================================================================

def extract_text_from_file(filepath, ext):
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
                return '\n'.join([p.text for p in doc.paragraphs])
            except ImportError:
                return f"[DOCX: python-docx не установлен для чтения {os.path.basename(filepath)}]"

        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            try:
                import pytesseract
                from PIL import Image
                img = Image.open(filepath)
                return pytesseract.image_to_string(img, lang='rus+eng')
            except ImportError:
                return f"[Изображение: {os.path.basename(filepath)} - OCR недоступен]"

        elif ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.md', '.rst', '.csv', '.log', '.ini', '.cfg', '.conf']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        else:
            return f"[Формат файла не поддерживается: {ext}]"
    except Exception as e:
        logger.error(f"Ошибка извлечения текста: {e}")
        return f"[Ошибка чтения файла: {str(e)}]"


# ========================================================================
# Import routes & handlers (registers them with app)
# ========================================================================

from . import server_routes  # noqa: F401
from . import server_handlers  # noqa: F401

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='127.0.0.1', port=5555, debug=False, use_reloader=False)
