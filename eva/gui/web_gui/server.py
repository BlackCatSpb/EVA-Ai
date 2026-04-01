"""
Flask сервер для Web GUI ЕВА с аутентификацией и сессиями
"""
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
            template_folder='templates',
            static_folder='static',
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

# Configure Tesseract path for OCR
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    logger.info(f"Tesseract configured at: {TESSERACT_PATH}")
except Exception as e:
    logger.warning(f"Failed to configure Tesseract: {e}")


class SessionManager:
    def __init__(self):
        self.sessions = {}
        self._lock = threading.Lock()
        gui_dir = os.path.dirname(os.path.dirname(__file__))
        self._storage_file = os.path.join(
            gui_dir, 
            'eva_gui_cache', 'sessions.json'
        )
        self._ensure_storage_dir()
        self._load_sessions()
    
    def _ensure_storage_dir(self):
        storage_dir = os.path.dirname(self._storage_file)
        os.makedirs(storage_dir, exist_ok=True)
    
    def _load_sessions(self):
        """Загрузка сессий из файла"""
        try:
            if os.path.exists(self._storage_file):
                with open(self._storage_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Ошибка парсинга JSON сессий: {e}, файл будет перезаписан")
                        self.sessions = {}
                        return
                    self.sessions = data.get('sessions', {})
                    logger.info(f"Загружено {len(self.sessions)} сессий из хранилища")
        except Exception as e:
            logger.warning(f"Не удалось загрузить сессии: {e}")
    
    def _save_sessions(self):
        """Сохранение сессий в файл"""
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
                'id': session_id,
                'user_id': user_id,
                'name': session_name,
                'created_at': datetime.now().isoformat(),
                'last_active': datetime.now().isoformat(),
                'context_nodes': [],
                'entities': []
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
                    user_id = user['user_id']
                    return {'username': username, 'user_id': user_id}
        return None


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
                        'type': entity_type,
                        'keyword': keyword,
                        'extracted_at': datetime.now().isoformat(),
                        'context': text[:200]
                    })
                    break
        
        return entities
    
    def is_personal_info(self, text: str) -> bool:
        personal_keywords = ['пароль', 'номер карты', 'адрес', 'телефон', 'email', 'день рождения']
        text_lower = text.lower()
        return any(kw in text_lower for kw in personal_keywords)


class EthicsChecker:
    def __init__(self):
        self.blocked_patterns = ['взлом', 'хакер', 'вирус', 'наркотик', 'оружие', 'убийство']
    
    def check_message(self, text: str) -> Dict[str, Any]:
        result = {'allowed': True, 'warnings': [], 'blocked': False, 'reason': None}
        text_lower = text.lower()
        
        blocked = ['взлом', 'хакер', 'вирус', 'наркотик', 'оружие', 'убийство']
        
        for pattern in blocked:
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
        
        # If file attached, enhance query with file context
        if file_data and file_data.get('extracted_text'):
            filename = file_data.get('filename', 'file')
            extracted = file_data['extracted_text']
            enhanced_query = f"""Пользователь прикрепил файл "{filename}".
Содержимое файла:
---
{extracted}
---

Запрос пользователя: {query}"""
            query = enhanced_query
        
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
                'id': str(uuid.uuid4()),
                'user_message': query,
                'timestamp': datetime.now().isoformat(),
                'entities': entities,
                'file_data': file_data  # Save file info in context
            }
            self.session_manager.add_context_node(session_id, context_node)
            
            for entity in entities:
                if not self.entity_extractor.is_personal_info(entity.get('context', '')):
                    sanitized = self.ethics_checker.sanitize_entity(entity)
                    self.session_manager.add_entity(session_id, sanitized)
        
        response_text = "Система обрабатывает запрос..."
        
        result = None
        debug_info = {"brain": self.brain is not None, "has_process_query": False}
        
        # Get conversation history for context
        conversation_history = []
        if session_id and isinstance(session_id, str) and session_id.strip():
            session = self.session_manager.get_session(session_id)
            if session and isinstance(session, dict) and 'context_nodes' in session:
                context_nodes = session.get('context_nodes')
                if context_nodes and isinstance(context_nodes, list):
                    for node in context_nodes[-10:]:
                        if isinstance(node, dict):
                            if 'user_message' in node and isinstance(node['user_message'], str):
                                conversation_history.append({"role": "user", "content": node['user_message']})
                            if 'assistant_message' in node and isinstance(node['assistant_message'], str):
                                conversation_history.append({"role": "assistant", "content": node['assistant_message']})
        
        user_context = {
            'session_id': session_id,
            'user_id': user_id,
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
        
        # DEBUG - special query to get debug info
        if query.strip().lower() == "debug123":
            return {
                'response': 'Debug info: ' + str(debug_info),
                'status': 'ok',
                'reasoning': str(debug_info)
            }
        
        # Get reasoning from brain result if available
        brain_reasoning = None
        reasoning_data = None
        reasoning_steps = []  # Для live display
        
        if result and isinstance(result, dict):
            brain_reasoning = result.get('reasoning')
            brain_reasoning_raw = result.get('reasoning_raw')
            source = result.get('source', '')
            confidence = result.get('confidence', 0)
            search_results = result.get('search_results', [])
            # Get ethics result early so it's available for return_data (but keep local result if brain returns None)
            if ethics_result is None and result.get('ethics_result'):
                ethics_result = result.get('ethics_result')
            
            # Extract web search info if available
            web_search_info = None
            if search_results and len(search_results) > 0:
                web_search_info = f"Найдено {len(search_results)} результатов:"
                for i, sr in enumerate(search_results[:3]):
                    title = sr.get('title', 'No title')[:60]
                    url = sr.get('url', '')[:50]
                    web_search_info += f"\n{i+1}. {title}... ({url})"
            
            # For qwen_only_mode with modules - show module interactions
            if source == 'llama_cpp_with_modules' or (file_data and file_data.get('extracted_text')):
                # Build reasoning steps from module results
                reasoning_steps = []
                
                # Step 0: Document analysis (if file attached)
                if file_data and file_data.get('extracted_text'):
                    filename = file_data.get('filename', 'file')
                    text_len = len(file_data.get('extracted_text', ''))
                    reasoning_steps.append({
                        'step': 0,
                        'phase': 'document_analysis',
                        'thought': f'Анализ документа "{filename}" - извлечено {text_len} символов',
                        'confidence': 0.9
                    })
                
                # Step 1: Initial generation
                reasoning_steps.append({
                    'step': len(reasoning_steps) + 1,
                    'phase': 'generation',
                    'thought': 'Первичная генерация ответа через LlamaCpp (GGUF)',
                    'confidence': 0.5
                })
                
                # Step 2: Contradiction check
                contr_result = result.get('contradiction_result')
                contr_count = 0
                if contr_result:
                    contr_count = contr_result.get('significant_count', 0)
                    contr_conf = 1.0 - contr_result.get('contradiction_level', 0.0)
                    reasoning_steps.append({
                        'step': 2,
                        'phase': 'contradiction_check',
                        'thought': f'Проверка противоречий: {contr_count} найдено, уровень={contr_result.get("contradiction_level", 0):.2f}',
                        'confidence': contr_conf
                    })
                
                # Step 3: Ethics check (keep local result if brain returns None)
                if ethics_result is None:
                    ethics_result = result.get('ethics_result')
                has_violations = False
                if ethics_result:
                    has_violations = ethics_result.get('has_violations', False)
                    ethics_conf = ethics_result.get('is_ethical', 1.0)
                    reasoning_steps.append({
                        'step': 3,
                        'phase': 'ethics_check',
                        'thought': f'Проверка этики: violations={has_violations}, score={ethics_conf:.2f}',
                        'confidence': ethics_conf
                    })
                
                # Step 4: Web search
                if search_results and len(search_results) > 0:
                    reasoning_steps.append({
                        'step': 4,
                        'phase': 'web_search',
                        'thought': f'Веб-поиск: найдено {len(search_results)} результатов',
                        'confidence': 0.8
                    })
                    
                    # Step 5: Refinement with web context
                    reasoning_steps.append({
                        'step': 5,
                        'phase': 'refinement',
                        'thought': 'Перегенерация с контекстом из веб-поиска',
                        'confidence': 0.9
                    })
                elif contr_count > 0 or (ethics_result and has_violations):
                    reasoning_steps.append({
                        'step': 4,
                        'phase': 'refinement',
                        'thought': 'Перегенерация после исправления модулей',
                        'confidence': 0.7
                    })
                
                # Final step
                reasoning_steps.append({
                    'step': len(reasoning_steps) + 1,
                    'phase': 'final_synthesis',
                    'thought': 'Финальный ответ с учетом всех проверок',
                    'confidence': result.get('confidence', 0.9)
                })
                
                reasoning_data = "Рассуждения системы (qwen_only_mode):\n\n" + "\n".join([
                    f"{s['step']}. [{s['phase']}] {s['thought']} (conf: {s['confidence']:.2f})"
                    for s in reasoning_steps
                ])
            
            # For SelfReasoningEngine - show reasoning steps if available
            if source == 'self_reasoning_engine':
                if brain_reasoning_raw and isinstance(brain_reasoning_raw, dict):
                    # Extract reasoning steps
                    steps = brain_reasoning_raw.get('steps', [])
                    if steps:
                        reasoning_text = "Рассуждения системы:\n\n"
                        for i, step in enumerate(steps):
                            phase = step.get('phase', 'unknown')
                            thought = step.get('thought', '')
                            conf = step.get('confidence', 0)
                            reasoning_steps.append({
                                'step': i + 1,
                                'phase': phase,
                                'thought': thought,
                                'confidence': conf
                            })
                            if i < 10:  # Show first 10 steps
                                reasoning_text += f"{i+1}. [{phase}] {thought} (conf: {conf:.2f})\n"
                        reasoning_data = reasoning_text
                    elif brain_reasoning:
                        # Fallback to simple reasoning
                        reasoning_data = str(brain_reasoning)
                elif brain_reasoning:
                    reasoning_data = str(brain_reasoning)
            
            # For EnhancedReasoningEngine - show iteration chain with module prompts
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
                            
                            # Добавляем промты модулей
                            prompts_text = ""
                            if module_prompts:
                                prompts_text = "\nПромты модулей:"
                                for mod, prompt in module_prompts.items():
                                    prompts_text += f"\n  [{mod.upper()}]: {prompt[:100]}..."
                            
                            reasoning_steps.append({
                                'step': i + 1,
                                'phase': 'regeneration',
                                'thought': resp_preview,
                                'confidence': conf,
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
            # For Qwen - show simple indicator only if high confidence
            elif source == 'qwen_model' and confidence >= 0.85:
                reasoning_data = f"🤖 Qwen Model обработал запрос (уверенность: {confidence:.2f})"
            # reasoning_data stays None
        
        if session_id and response_text:
            response_node = {
                'id': str(uuid.uuid4()),
                'assistant_message': response_text,
                'timestamp': datetime.now().isoformat(),
                'reasoning': reasoning_data
            }
            self.session_manager.add_context_node(session_id, response_node)
        
        # Запускаем самодиалог после каждого запроса
        self_dialog_result = None
        if self.brain and hasattr(self.brain, 'self_dialog_learning') and self.brain.self_dialog_learning:
            try:
                # Создаем самодиалог на основе текущего запроса
                self.brain.self_dialog_learning.create_dialog(
                    topic=query[:100],
                    context={
                        "user_query": query,
                        "system_response": response_text[:200] if response_text else "",
                        "source": "web_gui_chat"
                    }
                )
                logger.info(f"Запущен самодиалог для темы: {query[:50]}...")
                
                # Получаем результат последнего диалога
                if hasattr(self.brain.self_dialog_learning, 'get_recent_learning'):
                    recent_dialogs = self.brain.self_dialog_learning.get_recent_learning(limit=1)
                    if recent_dialogs:
                        self_dialog_result = recent_dialogs[0]
            except Exception as e:
                logger.debug(f"Error triggering self-dialog: {e}")
        
        return_data = {
            'response': response_text,
            'status': 'ok',
            'warnings': ethics_result.get('warnings', []) if ethics_result else [],
            'reasoning': reasoning_data,
            'reasoning_steps': reasoning_steps,
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


web_gui_instance: Optional[WebGUI] = None


def create_app(brain=None, integrator=None, host='127.0.0.1', port=5555):
    global web_gui_instance
    web_gui_instance = WebGUI(brain=brain, integrator=integrator, host=host, port=port)
    web_gui_instance.start()
    return web_gui_instance


def get_app() -> WebGUI:
    return web_gui_instance


@app.route('/')
def index():
    return render_template('index.html')


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
            # Получаем существующие сессии пользователя
            existing_sessions = web_gui_instance.session_manager.get_user_sessions(user['user_id'])
            
            # Если есть сессии - используем первую, иначе создаём новую
            if existing_sessions:
                session_id = existing_sessions[0]['id']
                # Обновляем last_active
                web_gui_instance.session_manager.update_session(session_id, {})
            else:
                session_id = web_gui_instance.session_manager.create_session(
                    user['user_id'], 
                    f"Сессия {username}"
                )
            
            # Обновляем список сессий
            sessions = web_gui_instance.session_manager.get_user_sessions(user['user_id'])
            
            return jsonify({
                'user': user['username'],
                'session_id': session_id,
                'sessions': sessions
            })
    
    return jsonify({'error': 'Неверные учетные данные'}), 401


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
    
    # Create uploads directory
    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save file with unique name
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    safe_filename = f"{file_id}{ext}"
    filepath = os.path.join(upload_dir, safe_filename)
    file.save(filepath)
    
    # Extract text from file
    extracted_text = extract_text_from_file(filepath, ext)
    
    if extracted_text:
        logger.info(f"Текст извлечён из файла {file.filename} (метод: {ext})")

    return jsonify({
        'file_id': file_id,
        'filename': file.filename,
        'size': os.path.getsize(filepath),
        'extracted_text': extracted_text[:5000] if extracted_text else '',  # Limit to 5000 chars
        'status': 'ok'
    })


def extract_text_from_file(filepath, ext):
    """Извлекает текст из файла в зависимости от типа. Всегда возвращает строку."""
    try:
        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        elif ext == '.pdf':
            text = ''
            
            # Try pymupdf first (best for complex PDFs)
            try:
                import fitz  # pymupdf
                doc = fitz.open(filepath)
                for page in doc:
                    text += page.get_text() + '\n'
                doc.close()
                if text.strip():
                    logger.info(f"PDF прочитан через pymupdf: {len(text)} символов")
                    return text
            except Exception as e:
                logger.warning(f"pymupdf failed: {e}")
            
            # Fallback to pdfplumber
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
            
            # Last resort: PyPDF2
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
            
            # Note: OCR for scanned PDFs requires Tesseract to be installed
            # Install from: https://github.com/UB-Mannheim/tesseract/wiki
            
            # Fallback: Tesseract if available
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
            # For images, try OCR if available
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


@app.route('/api/chat', methods=['POST'])
def api_chat():
    if not web_gui_instance:
        return jsonify({'error': 'Сервер не инициализирован'}), 500
    
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        message = data.get('message', '')
        session_id = data.get('session_id')
        user_id = data.get('user_id')
        file_data = data.get('file_data')  # {file_id, filename, extracted_text}
        
        result = web_gui_instance.process_message(message, session_id, user_id, file_data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Ошибка в api_chat: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


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


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """Receive user feedback (like/dislike) for messages"""
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        rating = data.get('rating', 0)  # 1 = like, -1 = dislike
        message_text = data.get('message_text', '')
        message_index = data.get('message_index', 0)
        
        # Log feedback for learning system
        logger.info(f"User feedback: rating={rating}, index={message_index}")
        
        # Send to brain if available
        if web_gui_instance.brain:
            try:
                # Use subjective correctness trigger
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
        logger.debug(f"api_analytics: brain = {web_gui_instance.brain}")
        
        if web_gui_instance.brain:
            if hasattr(web_gui_instance.brain, 'get_resource_snapshot'):
                snapshot = web_gui_instance.brain.get_resource_snapshot()
                metrics.update(snapshot)
            if hasattr(web_gui_instance.brain, 'get_cache_stats'):
                cache = web_gui_instance.brain.get_cache_stats()
                analytics['cache_hit_rate'] = cache.get('hit_rate', 0.0)
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
    
    return jsonify(metrics)


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


@app.route('/api/analytics')
def api_analytics():
    """Get analytics data for dashboard."""
    if not web_gui_instance:
        logger.warning("api_analytics: web_gui_instance is None")
        return jsonify({'error': 'not_initialized'})
    
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
        logger.debug(f"api_analytics: brain = {web_gui_instance.brain}")
        
        if web_gui_instance.brain:
            # Get system metrics
            if hasattr(web_gui_instance.brain, 'get_resource_snapshot'):
                snapshot = web_gui_instance.brain.get_resource_snapshot()
                # Handle different key names from get_resource_snapshot
                analytics['cpu'] = snapshot.get('cpu_usage', snapshot.get('cpu_percent', 0))
                analytics['memory'] = snapshot.get('memory_usage', snapshot.get('memory_percent', 0))
                analytics['vram'] = snapshot.get('gpu_memory', snapshot.get('gpu_memory_percent', 0))
            
            # Get cache stats for metrics view
            if hasattr(web_gui_instance.brain, 'get_cache_stats'):
                cache_stats = web_gui_instance.brain.get_cache_stats()
                # Add to metrics API
                analytics['cache_hit_rate'] = cache_stats.get('hit_rate', 0.0)
                analytics['cache_utilization'] = cache_stats.get('cache_utilization_percent', 0.0)
            
            # Get learning stats
            if hasattr(web_gui_instance.brain, 'self_dialog_learning'):
                sdl = web_gui_instance.brain.self_dialog_learning
                if hasattr(sdl, 'get_stats'):
                    stats = sdl.get_stats()
                    analytics['dialogs'] = stats.get('total_dialogs', 0)
                    analytics['gaps'] = stats.get('knowledge_gaps_identified', 0)
                    analytics['learned'] = stats.get('successful_learning', 0)
            
            # Get performance stats
            if hasattr(web_gui_instance.brain, 'performance_analyzer'):
                pa = web_gui_instance.brain.performance_analyzer
                if hasattr(pa, 'analyze_performance'):
                    try:
                        perf_data = pa.analyze_performance()
                        analytics['queries'] = perf_data.get('total_queries', 0)
                        analytics['avg_time'] = perf_data.get('avg_query_time_ms', 0)
                        analytics['success_rate'] = perf_data.get('success_rate', 0)
                    except Exception as e:
                        logger.debug(f"PerformanceAnalyzer error: {e}")
                        # Fallback - count from session data
                        analytics['queries'] = 0
                        analytics['avg_time'] = 0
                        analytics['success_rate'] = 0
            
            # Get real system metrics if available
            try:
                import psutil
                analytics['cpu'] = psutil.cpu_percent(interval=None)
                analytics['memory'] = psutil.virtual_memory().percent
            except ImportError:
                pass  # Use defaults
            
            # Build activity list
            activities = []
            
            # Memory activity - check if memory_manager exists
            if hasattr(web_gui_instance.brain, 'memory_manager') and web_gui_instance.brain.memory_manager:
                mm = web_gui_instance.brain.memory_manager
                try:
                    # Try to get actual memory data
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
            
            # Learning activity
            if hasattr(web_gui_instance.brain, 'self_dialog_learning'):
                sdl = web_gui_instance.brain.self_dialog_learning
                if hasattr(sdl, 'stats'):
                    stats = sdl.stats
                    if stats.get('total_dialogs', 0) > 0:
                        activities.append({
                            'icon': 'learn',
                            'title': f'Диалогов обучения: {stats["total_dialogs"]}',
                            'time': 'Сегодня'
                        })
            
            analytics['activities'] = activities[:10]
            
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
    
    # Add fallback for demo mode if no brain
    if not web_gui_instance or not web_gui_instance.brain:
        analytics['dialogs'] = 0
        analytics['gaps'] = 0
        analytics['learned'] = 0
    
    return jsonify(analytics)


@app.route('/api/learning')
def api_learning():
    """Get learning opportunities and stats."""
    if not web_gui_instance:
        logger.warning("api_learning: web_gui_instance is None")
        return jsonify({'error': 'not_initialized'})
    
    learning = {
        'opportunities': [],
        'total': 0,
        'success': 0,
        'pending': 0,
        'dialogs': [],
        'recent_dialogs': []
    }
    
    try:
        logger.debug(f"api_analytics: brain = {web_gui_instance.brain}")
        
        if web_gui_instance.brain:
            # Get learning opportunities
            if hasattr(web_gui_instance.brain, 'self_dialog_learning'):
                sdl = web_gui_instance.brain.self_dialog_learning
                
                # Get opportunities from analyzer
                if hasattr(sdl, '_get_learning_opportunities'):
                    try:
                        opportunities = sdl._get_learning_opportunities()
                        for op in opportunities[:10]:
                            priority = op.get('priority', 0.5)
                            priority_level = 'high' if priority >= 0.7 else 'medium' if priority >= 0.4 else 'low'
                            learning['opportunities'].append({
                                'concept': op.get('concept', 'Unknown'),
                                'type': op.get('opportunity_type', 'expansion'),
                                'priority': priority,
                                'priority_level': priority_level,
                                'domain': op.get('domain', 'general')
                            })
                    except Exception as e:
                        logger.debug(f"Error getting opportunities: {e}")
                
                # Get stats
                if hasattr(sdl, 'get_stats'):
                    stats = sdl.get_stats()
                    learning['total'] = stats.get('total_dialogs', 0)
                    learning['success'] = stats.get('successful_learning', 0)
                
                # Get recent dialogs
                if hasattr(sdl, 'get_recent_learning'):
                    try:
                        recent = sdl.get_recent_learning(limit=5)
                        for d in recent:
                            learning['recent_dialogs'].append({
                                'topic': d.get('topic', '')[:50],
                                'outcome': d.get('outcome', 'unknown'),
                                'gaps': d.get('gaps', [])
                            })
                    except Exception as e:
                        logger.debug(f"Error getting recent dialogs: {e}")
            
            learning['pending'] = len(learning['opportunities'])
            
    except Exception as e:
        logger.error(f"Error getting learning data: {e}")
    
    # Add fallback for demo mode if no brain
    if not web_gui_instance or not web_gui_instance.brain:
        learning['total'] = 0
        learning['success'] = 0
        learning['pending'] = 0
    
    return jsonify(learning)


# ========================================================================
# Settings API - Управление настройками системы
# ========================================================================

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    """Получить или обновить настройки системы."""
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})
    
    if request.method == 'GET':
        settings = {
            'auto_learning': True,
            'sre_enabled': True,
            'memory_enabled': True,
            'dark_theme': True,
            'sound_enabled': False,
            'model_name': 'Qwen2.5-0.5B GGUF',
            'version': '1.0.0'
        }
        
        try:
            if web_gui_instance.brain:
                if hasattr(web_gui_instance.brain, 'self_dialog_learning'):
                    sdl = web_gui_instance.brain.self_dialog_learning
                    settings['auto_learning'] = getattr(sdl, 'auto_execute_enabled', True)
                
                # Get model info
                if hasattr(web_gui_instance.brain, 'llama_cpp_deployment'):
                    settings['model_name'] = 'Qwen2.5-0.5B GGUF'
                elif hasattr(web_gui_instance.brain, 'qwen_model_manager'):
                    settings['model_name'] = 'Qwen3.5-0.8B'
        except Exception as e:
            logger.debug(f"Error getting settings: {e}")
        
        return jsonify(settings)
    
    # POST - update settings
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400
    
    try:
        logger.debug(f"api_analytics: brain = {web_gui_instance.brain}")
        
        if web_gui_instance.brain:
            if 'auto_learning' in data and hasattr(web_gui_instance.brain, 'self_dialog_learning'):
                sdl = web_gui_instance.brain.self_dialog_learning
                if hasattr(sdl, 'auto_execute_enabled'):
                    sdl.auto_execute_enabled = data['auto_learning']
        
        return jsonify({'status': 'ok', 'updated': list(data.keys())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================================================================
# Documents API - Работа с документами
# ========================================================================

@app.route('/api/documents', methods=['GET'])
def api_documents():
    """Получить список документов для текущей сессии."""
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})
    
    user_id = request.headers.get('X-User-ID')
    session_id = request.args.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'session_id required'}), 400
    
    documents = []
    
    try:
        # Try to get from hybrid cache first
        if hasattr(web_gui_instance.brain, 'hybrid_cache') and web_gui_instance.brain.hybrid_cache:
            docs = web_gui_instance.brain.hybrid_cache.get_session_documents(session_id)
            documents = docs
        
        # Also check session manager
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


# ========================================================================
# Knowledge Graph API - Операции с графом знаний
# ========================================================================

@app.route('/api/knowledge-graph', methods=['GET', 'POST'])
def api_knowledge_graph():
    """Операции с графом знаний."""
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})
    
    if request.method == 'GET':
        # Get graph data
        action = request.args.get('action', 'get')
        
        try:
            kg = getattr(web_gui_instance.brain, 'knowledge_graph', None)
            if kg:
                if action == 'get':
                    nodes = []
                    if hasattr(kg, 'nodes'):
                        for n in kg.nodes[:50]:
                            nodes.append({
                                'id': getattr(n, 'id', ''),
                                'name': getattr(n, 'name', '')[:50],
                                'content': getattr(n, 'content', '')[:100]
                            })
                    return jsonify({'nodes': nodes, 'total': len(nodes)})
                
                elif action == 'search':
                    query = request.args.get('query', '')
                    if hasattr(kg, 'search_nodes'):
                        results = kg.search_nodes(query, limit=10)
                        return jsonify({'results': results})
            return jsonify({'nodes': [], 'total': 0})
        except Exception as e:
            logger.debug(f"Error accessing knowledge graph: {e}")
            return jsonify({'error': str(e)}), 500
    
    # POST - add node
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


# ========================================================================
# Cache API - Статистика кэша
# ========================================================================

@app.route('/api/cache-stats')
def api_cache_stats():
    """Получить статистику кэша."""
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})
    
    stats = {
        'hybrid_cache': {},
        'search_cache': {},
        'memory': {}
    }
    
    try:
        # Hybrid cache stats
        if hasattr(web_gui_instance.brain, 'hybrid_cache') and web_gui_instance.brain.hybrid_cache:
            hc = web_gui_instance.brain.hybrid_cache
            if hasattr(hc, 'get_cache_stats'):
                stats['hybrid_cache'] = hc.get_cache_stats()
            if hasattr(hc, 'get_search_cache_stats'):
                stats['search_cache'] = hc.get_search_cache_stats()
        
        # Memory stats
        if hasattr(web_gui_instance.brain, 'memory_manager'):
            mm = web_gui_instance.brain.memory_manager
            if hasattr(mm, 'nodes'):
                stats['memory']['total_nodes'] = len(mm.nodes)
    except Exception as e:
        logger.debug(f"Error getting cache stats: {e}")
    
    return jsonify(stats)


# ========================================================================
# System API - Общая информация о системе
# ========================================================================

@app.route('/api/system')
def api_system():
    """Получить общую информацию о системе."""
    if not web_gui_instance:
        return jsonify({'error': 'not_initialized'})
    
    system_info = {
        'version': '1.0.0',
        'model': 'Qwen2.5-0.5B GGUF',
        'qwen_ready': False,
        'llama_cpp_ready': False,
        'modules': {
            'contradiction': False,
            'ethics': False,
            'web_search': False,
            'knowledge_graph': False
        },
        'features': {
            'self_learning': False,
            'knowledge_graph': False,
            'web_search': False
        }
    }
    
    try:
        logger.debug(f"api_analytics: brain = {web_gui_instance.brain}")
        
        if web_gui_instance.brain:
            brain = web_gui_instance.brain
            
            # Model status
            system_info['qwen_ready'] = getattr(brain, 'qwen_ready', False)
            system_info['llama_cpp_ready'] = getattr(brain, 'llama_cpp_ready', False)
            
            if hasattr(brain, 'llama_cpp_deployment'):
                system_info['model'] = 'Qwen2.5-0.5B GGUF'
            elif hasattr(brain, 'qwen_model_manager'):
                system_info['model'] = 'Qwen3.5-0.8B'
            
            # Module status
            system_info['modules']['contradiction'] = hasattr(brain, 'contradiction_manager')
            system_info['modules']['ethics'] = hasattr(brain, 'ethics_framework')
            system_info['modules']['web_search'] = hasattr(brain, 'web_search_engine')
            system_info['modules']['knowledge_graph'] = hasattr(brain, 'knowledge_graph')
            
            # Feature status
            system_info['features']['self_learning'] = hasattr(brain, 'self_dialog_learning')
            system_info['features']['knowledge_graph'] = hasattr(brain, 'knowledge_graph')
            system_info['features']['web_search'] = hasattr(brain, 'web_search_engine')
    except Exception as e:
        logger.debug(f"Error getting system info: {e}")
    
    return jsonify(system_info)




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='127.0.0.1', port=5555, debug=False, use_reloader=False)
