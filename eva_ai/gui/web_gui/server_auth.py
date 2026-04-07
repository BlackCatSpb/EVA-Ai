"""
Аутентификация и управление сессиями для Web GUI ЕВА
"""
import os
import logging
import threading
import json
import uuid
import hashlib
import secrets
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("eva_ai.webgui")


class SessionManager:
    """
    Управление сессиями пользователей.

    NOTE: sessions.json содержит конфиденциальные данные (идентификаторы сессий,
    историю чатов, контекстные узлы). В production этот файл должен быть зашифрован.
    На Windows файл не является world-readable, но всё равно требует защиты.
    """
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
                'chat_history': [],
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
            user_sessions = [s for s in self.sessions.values() if s['user_id'] == user_id]
            # Sort by last_active descending (most recent first)
            user_sessions.sort(key=lambda s: s.get('last_active', ''), reverse=True)
            return user_sessions

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
                if len(self.sessions[session_id]['context_nodes']) > 20:
                    self.sessions[session_id]['context_nodes'] = self.sessions[session_id]['context_nodes'][-20:]
                    logger.debug(f"Ограничены context_nodes до 20 для сессии {session_id}")
                self._save_sessions()

    def add_chat_message(self, session_id: str, role: str, content: str):
        """Добавляет сообщение в историю чата (сырой текст)."""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]['chat_history'].append({
                    'role': role,
                    'content': content,
                    'timestamp': datetime.now().isoformat()
                })
                if len(self.sessions[session_id]['chat_history']) > 50:
                    self.sessions[session_id]['chat_history'] = self.sessions[session_id]['chat_history'][-50:]
                    logger.debug(f"Ограничен chat_history до 50 для сессии {session_id}")
                self._save_sessions()

    def convert_chat_to_knowledge(self, session_id: str, fractal_memory) -> bool:
        """Преобразует chat_history в knowledge nodes через fractal_memory."""
        if not fractal_memory or not hasattr(fractal_memory, 'save_experience'):
            return False
        
        with self._lock:
            if session_id not in self.sessions:
                return False
            
            chat_history = self.sessions[session_id].get('chat_history', [])
            if len(chat_history) < 2:
                return False
            
            pairs = []
            for i in range(0, len(chat_history) - 1, 2):
                if i + 1 < len(chat_history):
                    user_msg = chat_history[i].get('content', '') if chat_history[i].get('role') == 'user' else ''
                    assistant_msg = chat_history[i + 1].get('content', '') if i + 1 < len(chat_history) and chat_history[i + 1].get('role') == 'assistant' else ''
                    if user_msg and assistant_msg:
                        pairs.append((user_msg, assistant_msg))
            
            saved_count = 0
            for query, response in pairs:
                try:
                    fractal_memory.save_experience(
                        query=query,
                        response=response,
                        model_used='web_ui',
                        quality_score=0.5
                    )
                    saved_count += 1
                except Exception as e:
                    logger.warning(f"Не удалось сохранить опыт: {e}")
            
            logger.info(f"Сохранено {saved_count} опытов в fractal_memory для сессии {session_id}")
            return saved_count > 0

    def get_chat_history(self, session_id: str, limit: int = 20) -> list:
        """Возвращает историю чата для контекста генерации."""
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                history = session.get('chat_history', [])
                return history[-limit:]
            return []

    def add_entity(self, session_id: str, entity: Dict):
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]['entities'].append(entity)
                if len(self.sessions[session_id]['entities']) > 30:
                    self.sessions[session_id]['entities'] = self.sessions[session_id]['entities'][-30:]
                    logger.debug(f"Ограничены entities до 30 для сессии {session_id}")
                self._save_sessions()


class AuthManager:
    def __init__(self):
        self.users = {}
        self._lock = threading.Lock()

    def _hash_password(self, password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()

    def set_default_credentials(self, username: str, password: str):
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        with self._lock:
            self.users[username] = {
                'username': username,
                'password_hash': password_hash,
                'salt': salt,
                'created_at': datetime.now().isoformat()
            }

    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        with self._lock:
            user = self.users.get(username)
            if user:
                salt = user.get('salt', '')
                if salt:
                    password_hash = self._hash_password(password, salt)
                else:
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
