# gui/auth_module.py
"""Модуль аутентификации для CogniFlex - полнофункциональная реализация"""
import os
import time
import hashlib
import base64
import logging
import sqlite3
import json
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger("cogniflex.auth")

class AuthModule:
    """Модуль аутентификации для CogniFlex - обеспечивает безопасный доступ к системе."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует модуль аутентификации.
        
        Args:
            brain: Ссылка на ядро CogniFlex (опционально)
            cache_dir: Путь к директории кэша
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "cogniflex_auth_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Путь к базе данных
        self.db_path = os.path.join(self.cache_dir, "auth.db")
        
        # Текущая сессия
        self.current_user_id = None
        self.session_token = None
        self.session_expiry = None
        
        # Инициализируем базу данных
        self._init_db()
        
        # Загружаем пользователей
        self._load_users()
        
        # Инициализируем системные пользователи
        self._init_system_users()
        
        logger.info("Модуль аутентификации инициализирован")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Возвращает соединение с базой данных для текущего потока."""
        if not hasattr(threading.current_thread(), "auth_module_connection"):
            # Создаем новое соединение для этого потока
            threading.current_thread().auth_module_connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False  # Разрешаем использование в разных потоках
            )
        return threading.current_thread().auth_module_connection
    
    def _init_db(self):
        """Инициализирует базу данных для аутентификации."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Таблица пользователей
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at REAL NOT NULL,
                last_login REAL,
                is_active BOOLEAN DEFAULT TRUE
            )
            """)
            
            # Таблица сессий
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """)
            
            # Таблица разрешений
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                permission TEXT NOT NULL,
                UNIQUE(role, permission)
            )
            """)
            
            # Таблица настроек пользователей
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences TEXT DEFAULT '{}',
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """)
            
            conn.commit()
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных аутентификации: {e}", exc_info=True)
    
    def _load_users(self):
        """Загружает пользователей из базы данных."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Загружаем пользователей
            cursor.execute("SELECT id, name, email, role, last_login, is_active FROM users")
            self.users = {}
            for row in cursor.fetchall():
                self.users[row[0]] = {
                    "id": row[0],
                    "name": row[1],
                    "email": row[2],
                    "role": row[3],
                    "last_login": row[4],
                    "is_active": row[5]
                }
            
            logger.info(f"Загружено {len(self.users)} пользователей")
        except Exception as e:
            logger.error(f"Ошибка загрузки пользователей: {e}", exc_info=True)
            self.users = {}
    
    def _init_system_users(self):
        """Инициализирует системных пользователей (админ и гость)."""
        # Проверяем наличие гостевого пользователя
        if "guest" not in self.users:
            self._create_guest_user()
        
        # Проверяем наличие администратора
        admin_exists = False
        for user_id, user in self.users.items():
            if user["role"] == "admin":
                admin_exists = True
                break
        
        if not admin_exists:
            self._create_default_admin()
    
    def _create_guest_user(self):
        """Создает гостевого пользователя."""
        guest_id = "guest"
        guest_data = {
            "id": guest_id,
            "name": "Гость",
            "email": "guest@cogniflex.local",
            "role": "guest",
            "created_at": time.time(),
            "is_active": True
        }
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Хэшируем пароль для гостя (хотя он не будет использоваться)
            password_hash = self._hash_password("guest")
            
            cursor.execute("""
            INSERT INTO users (id, name, email, password_hash, role, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                guest_id,
                guest_data["name"],
                guest_data["email"],
                password_hash,
                guest_data["role"],
                guest_data["created_at"],
                guest_data["is_active"]
            ))
            
            # Создаем настройки для гостя
            cursor.execute("""
            INSERT INTO user_preferences (user_id, preferences)
            VALUES (?, ?)
            """, (
                guest_id,
                json.dumps({
                    "theme": "light",
                    "language": "ru",
                    "notifications": False
                })
            ))
            
            conn.commit()
            
            # Добавляем базовые разрешения для гостя
            self._add_permissions_for_role("guest", ["read"])
            
            self.users[guest_id] = guest_data
            logger.info("Создан гостевой пользователь")
            
        except sqlite3.IntegrityError:
            # Пользователь уже существует
            pass
        except Exception as e:
            logger.error(f"Ошибка создания гостевого пользователя: {e}", exc_info=True)
    
    def _create_default_admin(self):
        """Создает администратора по умолчанию."""
        admin_id = "admin"
        admin_data = {
            "id": admin_id,
            "name": "Администратор",
            "email": "admin@cogniflex.local",
            "role": "admin",
            "created_at": time.time(),
            "is_active": True
        }
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Хэшируем пароль для администратора
            password_hash = self._hash_password("admin123")
            
            cursor.execute("""
            INSERT INTO users (id, name, email, password_hash, role, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                admin_id,
                admin_data["name"],
                admin_data["email"],
                password_hash,
                admin_data["role"],
                admin_data["created_at"],
                admin_data["is_active"]
            ))
            
            # Создаем настройки для администратора
            cursor.execute("""
            INSERT INTO user_preferences (user_id, preferences)
            VALUES (?, ?)
            """, (
                admin_id,
                json.dumps({
                    "theme": "dark",
                    "language": "ru",
                    "notifications": True
                })
            ))
            
            conn.commit()
            
            # Добавляем разрешения для администратора
            self._add_permissions_for_role("admin", [
                "read", "write", "delete", "manage_users", "manage_system"
            ])
            
            self.users[admin_id] = admin_data
            logger.info("Создан администратор по умолчанию (id: admin, пароль: admin123)")
            
        except sqlite3.IntegrityError:
            # Пользователь уже существует
            pass
        except Exception as e:
            logger.error(f"Ошибка создания администратора: {e}", exc_info=True)
    
    def _hash_password(self, password: str) -> str:
        """
        Хэширует пароль с использованием соли.
        
        Args:
            password: Пароль для хэширования
            
        Returns:
            str: Хэшированный пароль
        """
        # Генерируем соль
        salt = os.urandom(16)
        # Хэшируем пароль с солью
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        # Соединяем соль и хэш в один объект
        return base64.b64encode(salt + password_hash).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """
        Проверяет пароль на соответствие хэшу.
        
        Args:
            password: Пароль для проверки
            password_hash: Хэшированный пароль
            
        Returns:
            bool: Соответствует ли пароль хэшу
        """
        try:
            # Декодируем хэш
            decoded = base64.b64decode(password_hash.encode('utf-8'))
            # Извлекаем соль (первые 16 байт)
            salt = decoded[:16]
            # Извлекаем хэш (оставшиеся байты)
            stored_hash = decoded[16:]
            # Хэшируем введенный пароль с той же солью
            new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            # Сравниваем хэши
            return new_hash == stored_hash
        except Exception as e:
            logger.error(f"Ошибка проверки пароля: {e}", exc_info=True)
            return False
    
    def _create_session(self, user_id: str, ip_address: Optional[str] = None, 
                       user_agent: Optional[str] = None) -> str:
        """
        Создает новую сессию для пользователя.
        
        Args:
            user_id: ID пользователя
            ip_address: IP-адрес пользователя
            user_agent: User-Agent пользователя
            
        Returns:
            str: Токен сессии
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Генерируем уникальный токен
            token = base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
            # Время создания сессии
            created_at = time.time()
            # Время окончания сессии (2 часа)
            expires_at = created_at + (2 * 3600)
            
            # Сохраняем сессию в базу данных
            cursor.execute("""
            INSERT INTO sessions (token, user_id, created_at, expires_at, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                token,
                user_id,
                created_at,
                expires_at,
                ip_address,
                user_agent
            ))
            
            conn.commit()
            
            # Обновляем время последнего входа
            cursor.execute("""
            UPDATE users SET last_login = ? WHERE id = ?
            """, (created_at, user_id))
            
            conn.commit()
            
            # Обновляем внутреннее состояние
            self.current_user_id = user_id
            self.session_token = token
            self.session_expiry = expires_at
            
            # Обновляем локальный кэш пользователей
            if user_id in self.users:
                self.users[user_id]["last_login"] = created_at
            
            logger.info(f"Создана сессия для пользователя {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Ошибка создания сессии: {e}", exc_info=True)
            return ""
    
    def _clear_expired_sessions(self):
        """Очищает просроченные сессии."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            current_time = time.time()
            
            cursor.execute("DELETE FROM sessions WHERE expires_at < ?", (current_time,))
            conn.commit()
            
            logger.debug(f"Очищено {cursor.rowcount} просроченных сессий")
        except Exception as e:
            logger.error(f"Ошибка очистки сессий: {e}", exc_info=True)
    
    def _add_permissions_for_role(self, role: str, permissions: List[str]):
        """
        Добавляет разрешения для роли.
        
        Args:
            role: Роль
            permissions: Список разрешений
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            for permission in permissions:
                try:
                    cursor.execute("""
                    INSERT INTO permissions (role, permission)
                    VALUES (?, ?)
                    """, (role, permission))
                except sqlite3.IntegrityError:
                    # Разрешение уже существует
                    pass
            
            conn.commit()
        except Exception as e:
            logger.error(f"Ошибка добавления разрешений: {e}", exc_info=True)
    
    def get_current_user(self) -> Dict[str, Any]:
        """
        Возвращает текущего пользователя.
        
        Returns:
            Dict: Информация о пользователе
        """
        # Проверяем активную сессию
        if self.session_token and self._validate_session(self.session_token):
            return self.get_user_by_id(self.current_user_id)
        
        # Возвращаем гостевого пользователя
        return self.get_user_by_id("guest")
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: Optional[str] = None, 
                         user_agent: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Аутентифицирует пользователя.
        
        Args:
            username: Имя пользователя или email
            password: Пароль
            ip_address: IP-адрес пользователя
            user_agent: User-Agent пользователя
            
        Returns:
            Optional[Dict]: Информация о пользователе или None
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Ищем пользователя по email или name
            cursor.execute("SELECT id, password_hash FROM users WHERE email = ? OR name = ?", 
                          (username, username))
            user = cursor.fetchone()
            
            if not user:
                logger.warning(f"Попытка входа с несуществующим пользователем: {username}")
                return None
            
            user_id, password_hash = user
            
            # Проверяем пароль
            if not self._verify_password(password, password_hash):
                logger.warning(f"Неверный пароль для пользователя: {username}")
                return None
            
            # Проверяем активность пользователя
            cursor.execute("SELECT is_active FROM users WHERE id = ?", (user_id,))
            is_active = cursor.fetchone()[0]
            
            if not is_active:
                logger.warning(f"Попытка входа заблокированного пользователя: {username}")
                return None
            
            # Создаем сессию
            session_token = self._create_session(user_id, ip_address, user_agent)
            
            if not session_token:
                return None
            
            # Возвращаем информацию о пользователе
            return self.get_user_by_id(user_id)
            
        except Exception as e:
            logger.error(f"Ошибка аутентификации пользователя: {e}", exc_info=True)
            return None
    
    def logout(self) -> bool:
        """
        Выполняет выход текущего пользователя.
        
        Returns:
            bool: Успешно ли выполнен выход
        """
        if not self.current_user_id or not self.session_token:
            return False
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Удаляем сессию
            cursor.execute("DELETE FROM sessions WHERE token = ?", (self.session_token,))
            conn.commit()
            
            # Сбрасываем состояние
            self.current_user_id = None
            self.session_token = None
            self.session_expiry = None
            
            logger.info(f"Пользователь вышел из системы")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка выхода из системы: {e}", exc_info=True)
            return False
    
    def register_user(self, username: str, email: str, password: str, 
                     role: str = "user") -> Dict[str, Any]:
        """
        Регистрирует нового пользователя.
        
        Args:
            username: Имя пользователя
            email: Email пользователя
            password: Пароль
            role: Роль пользователя
            
        Returns:
            Dict: Результат регистрации
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Проверяем, существует ли пользователь
            cursor.execute("SELECT id FROM users WHERE email = ? OR name = ?", (email, username))
            if cursor.fetchone():
                return {
                    "success": False,
                    "error": "Пользователь с таким email или именем уже существует"
                }
            
            # Генерируем ID пользователя
            user_id = f"user_{hashlib.md5(email.encode()).hexdigest()[:8]}"
            
            # Хэшируем пароль
            password_hash = self._hash_password(password)
            created_at = time.time()
            
            # Добавляем пользователя в базу данных
            cursor.execute("""
            INSERT INTO users (id, name, email, password_hash, role, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                username,
                email,
                password_hash,
                role,
                created_at,
                True
            ))
            
            # Создаем настройки для пользователя
            cursor.execute("""
            INSERT INTO user_preferences (user_id, preferences)
            VALUES (?, ?)
            """, (
                user_id,
                json.dumps({
                    "theme": "light",
                    "language": "ru",
                    "notifications": True
                })
            ))
            
            conn.commit()
            
            # Добавляем пользователя в локальный кэш
            self.users[user_id] = {
                "id": user_id,
                "name": username,
                "email": email,
                "role": role,
                "created_at": created_at,
                "last_login": None,
                "is_active": True
            }
            
            logger.info(f"Пользователь {username} зарегистрирован")
            return {
                "success": True,
                "user_id": user_id,
                "message": "Пользователь успешно зарегистрирован"
            }
            
        except Exception as e:
            logger.error(f"Ошибка регистрации пользователя: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Ошибка регистрации: {str(e)}"
            }
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any], 
                          current_user_id: Optional[str] = None) -> bool:
        """
        Обновляет профиль пользователя.
        
        Args:
            user_id: ID пользователя для обновления
            profile_data: Данные профиля
            current_user_id: ID текущего пользователя (для проверки прав)
            
        Returns:
            bool: Успешно ли обновлено
        """
        try:
            # Проверяем права
            if current_user_id and current_user_id != user_id:
                if not self.check_permission(current_user_id, "manage_users"):
                    logger.warning(f"Попытка обновления профиля без прав: {current_user_id} -> {user_id}")
                    return False
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Проверяем существование пользователя
            cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
            if not cursor.fetchone():
                logger.warning(f"Попытка обновления несуществующего пользователя: {user_id}")
                return False
            
            # Подготавливаем данные для обновления
            update_fields = []
            params = []
            
            if "name" in profile_data:
                update_fields.append("name = ?")
                params.append(profile_data["name"])
            
            if "email" in profile_data:
                update_fields.append("email = ?")
                params.append(profile_data["email"])
            
            if "role" in profile_data and current_user_id and self.check_permission(current_user_id, "manage_users"):
                update_fields.append("role = ?")
                params.append(profile_data["role"])
            
            if "is_active" in profile_data and current_user_id and self.check_permission(current_user_id, "manage_users"):
                update_fields.append("is_active = ?")
                params.append(profile_data["is_active"])
            
            # Выполняем обновление
            if update_fields:
                params.append(user_id)
                query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
                cursor.execute(query, params)
                conn.commit()
            
            # Обновляем настройки, если они есть
            if "preferences" in profile_data:
                cursor.execute("SELECT preferences FROM user_preferences WHERE user_id = ?", (user_id,))
                result = cursor.fetchone()
                
                if result:
                    current_prefs = json.loads(result[0])
                    current_prefs.update(profile_data["preferences"])
                    new_prefs = json.dumps(current_prefs)
                    
                    cursor.execute("""
                    UPDATE user_preferences SET preferences = ? WHERE user_id = ?
                    """, (new_prefs, user_id))
                else:
                    cursor.execute("""
                    INSERT INTO user_preferences (user_id, preferences)
                    VALUES (?, ?)
                    """, (user_id, json.dumps(profile_data["preferences"])))
                
                conn.commit()
            
            # Обновляем локальный кэш
            if user_id in self.users:
                if "name" in profile_data:
                    self.users[user_id]["name"] = profile_data["name"]
                if "email" in profile_data:
                    self.users[user_id]["email"] = profile_data["email"]
                if "role" in profile_data and current_user_id and self.check_permission(current_user_id, "manage_users"):
                    self.users[user_id]["role"] = profile_data["role"]
                if "is_active" in profile_data and current_user_id and self.check_permission(current_user_id, "manage_users"):
                    self.users[user_id]["is_active"] = profile_data["is_active"]
            
            logger.info(f"Профиль пользователя {user_id} обновлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления профиля: {e}", exc_info=True)
            return False
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Возвращает настройки пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Dict: Настройки пользователя
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT preferences FROM user_preferences WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            
            if result:
                return json.loads(result[0])
            
            # Возвращаем настройки по умолчанию
            default_prefs = {
                "theme": "light",
                "language": "ru",
                "notifications": True
            }
            
            # Сохраняем настройки по умолчанию
            cursor.execute("""
            INSERT INTO user_preferences (user_id, preferences)
            VALUES (?, ?)
            """, (user_id, json.dumps(default_prefs)))
            
            conn.commit()
            
            return default_prefs
            
        except Exception as e:
            logger.error(f"Ошибка получения настроек пользователя: {e}", exc_info=True)
            return {
                "theme": "light",
                "language": "ru",
                "notifications": True
            }
    
    def set_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """
        Устанавливает настройку пользователя.
        
        Args:
            user_id: ID пользователя
            key: Ключ настройки
            value: Значение настройки
            
        Returns:
            bool: Успешно ли установлено
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Получаем текущие настройки
            cursor.execute("SELECT preferences FROM user_preferences WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            
            if result:
                prefs = json.loads(result[0])
                prefs[key] = value
                new_prefs = json.dumps(prefs)
                
                cursor.execute("""
                UPDATE user_preferences SET preferences = ? WHERE user_id = ?
                """, (new_prefs, user_id))
            else:
                # Создаем новые настройки
                prefs = {key: value}
                cursor.execute("""
                INSERT INTO user_preferences (user_id, preferences)
                VALUES (?, ?)
                """, (user_id, json.dumps(prefs)))
            
            conn.commit()
            logger.debug(f"Настройка {key} установлена для пользователя {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка установки настройки: {e}", exc_info=True)
            return False
    
    def validate_session(self, session_token: str) -> bool:
        """
        Проверяет валидность сессии.
        
        Args:
            session_token: Токен сессии
            
        Returns:
            bool: Валидна ли сессия
        """
        try:
            # Очищаем просроченные сессии
            self._clear_expired_sessions()
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Проверяем существование и срок действия сессии
            current_time = time.time()
            cursor.execute("""
            SELECT user_id, expires_at FROM sessions WHERE token = ? AND expires_at > ?
            """, (session_token, current_time))
            
            session = cursor.fetchone()
            
            if session:
                user_id, expires_at = session
                self.current_user_id = user_id
                self.session_token = session_token
                self.session_expiry = expires_at
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка проверки сессии: {e}", exc_info=True)
            return False
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает пользователя по ID.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Optional[Dict]: Информация о пользователе или None
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Получаем информацию о пользователе
            cursor.execute("""
            SELECT id, name, email, role, last_login, is_active 
            FROM users WHERE id = ?
            """, (user_id,))
            
            user = cursor.fetchone()
            
            if not user:
                return None
            
            # Получаем настройки пользователя
            cursor.execute("SELECT preferences FROM user_preferences WHERE user_id = ?", (user_id,))
            prefs = cursor.fetchone()
            
            user_data = {
                "id": user[0],
                "name": user[1],
                "email": user[2],
                "role": user[3],
                "last_login": user[4],
                "is_active": user[5],
                "preferences": json.loads(prefs[0]) if prefs else {}
            }
            
            # Добавляем информацию о сессии, если это текущий пользователь
            if user_id == self.current_user_id and self.session_token:
                user_data["session_token"] = self.session_token
                user_data["session_expiry"] = self.session_expiry
            
            return user_data
            
        except Exception as e:
            logger.error(f"Ошибка получения пользователя: {e}", exc_info=True)
            return None
    
    def get_all_users(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        Возвращает список всех пользователей.
        
        Args:
            include_inactive: Включать ли неактивных пользователей
            
        Returns:
            List[Dict]: Список пользователей
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Формируем запрос в зависимости от параметра include_inactive
            if include_inactive:
                cursor.execute("""
                SELECT id, name, email, role, last_login, is_active 
                FROM users
                """)
            else:
                cursor.execute("""
                SELECT id, name, email, role, last_login, is_active 
                FROM users WHERE is_active = 1
                """)
            
            users = []
            for row in cursor.fetchall():
                # Получаем настройки пользователя
                cursor.execute("SELECT preferences FROM user_preferences WHERE user_id = ?", (row[0],))
                prefs = cursor.fetchone()
                
                users.append({
                    "id": row[0],
                    "name": row[1],
                    "email": row[2],
                    "role": row[3],
                    "last_login": row[4],
                    "is_active": row[5],
                    "preferences": json.loads(prefs[0]) if prefs else {}
                })
            
            return users
            
        except Exception as e:
            logger.error(f"Ошибка получения списка пользователей: {e}", exc_info=True)
            return []
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """
        Проверяет наличие у пользователя определенного разрешения.
        
        Args:
            user_id: ID пользователя
            permission: Разрешение
            
        Returns:
            bool: Есть ли разрешение
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Получаем роль пользователя
            cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
            role_result = cursor.fetchone()
            
            if not role_result:
                return False
            
            role = role_result[0]
            
            # Проверяем наличие разрешения
            cursor.execute("""
            SELECT 1 FROM permissions WHERE role = ? AND permission = ?
            """, (role, permission))
            
            return cursor.fetchone() is not None
            
        except Exception as e:
            logger.error(f"Ошибка проверки разрешения: {e}", exc_info=True)
            return False
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """
        Изменяет пароль пользователя.
        
        Args:
            user_id: ID пользователя
            old_password: Старый пароль
            new_password: Новый пароль
            
        Returns:
            bool: Успешно ли изменено
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Получаем хэш текущего пароля
            cursor.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Попытка изменения пароля несуществующего пользователя: {user_id}")
                return False
            
            password_hash = result[0]
            
            # Проверяем старый пароль
            if not self._verify_password(old_password, password_hash):
                logger.warning(f"Неверный старый пароль при изменении для пользователя: {user_id}")
                return False
            
            # Хэшируем новый пароль
            new_password_hash = self._hash_password(new_password)
            
            # Обновляем пароль
            cursor.execute("""
            UPDATE users SET password_hash = ? WHERE id = ?
            """, (new_password_hash, user_id))
            
            conn.commit()
            logger.info(f"Пароль пользователя {user_id} успешно изменен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка изменения пароля: {e}", exc_info=True)
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает отчет о здоровье системы аутентификации.
        
        Returns:
            Dict: Отчет о здоровье
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Получаем статистику
            cursor.execute("SELECT COUNT(*) FROM users")
            total_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
            active_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM sessions WHERE expires_at > ?", (time.time(),))
            active_sessions = cursor.fetchone()[0]
            
            # Рассчитываем общий показатель здоровья
            health_score = 100.0
            
            # Учитываем количество пользователей
            if total_users < 5:
                health_score -= 25
            elif total_users < 10:
                health_score -= 10
            
            # Учитываем активность
            active_ratio = active_users / total_users if total_users > 0 else 0
            if active_ratio < 0.5:
                health_score -= 15
            
            # Проверяем наличие администратора
            cursor.execute("SELECT 1 FROM users WHERE role = 'admin' LIMIT 1")
            admin_exists = cursor.fetchone() is not None
            
            if not admin_exists:
                health_score -= 30
            
            # Анализируем проблемы
            problem_areas = []
            if total_users < 5:
                problem_areas.append("Мало зарегистрированных пользователей")
            
            if active_ratio < 0.5:
                problem_areas.append("Низкая активность пользователей")
            
            if not admin_exists:
                problem_areas.append("Отсутствует администратор системы")
            
            # Формируем рекомендации
            recommendations = []
            if total_users < 5:
                recommendations.append("Добавьте больше пользователей для тестирования системы")
            
            if active_ratio < 0.5:
                recommendations.append("Реализуйте функции для увеличения вовлеченности пользователей")
            
            if not admin_exists:
                recommendations.append("Создайте учетную запись администратора для управления системой")
            
            if not recommendations:
                recommendations.append("Система аутентификации работает стабильно")
            
            return {
                "health_score": max(0, min(100, health_score)),
                "total_users": total_users,
                "active_users": active_users,
                "active_sessions": active_sessions,
                "admin_exists": admin_exists,
                "problem_areas": problem_areas,
                "recommendations": recommendations,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о здоровье системы аутентификации: {e}", exc_info=True)
            return {
                "health_score": 0,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def close(self):
        """Закрывает соединение с базой данных."""
        if hasattr(threading.current_thread(), "auth_module_connection"):
            try:
                threading.current_thread().auth_module_connection.close()
                delattr(threading.current_thread(), "auth_module_connection")
            except Exception as e:
                logger.error(f"Ошибка закрытия соединения с БД: {e}")
        
        logger.info("Модуль аутентификации закрыт")