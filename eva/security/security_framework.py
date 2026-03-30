#!/usr/bin/env python3
"""
ЕВА Security Framework
Базовая система безопасности для защиты системы от злоупотреблений.
"""

import os
import time
import hashlib
import logging
import threading
import secrets
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger("eva.security")

@dataclass
class User:
    """Пользователь системы."""
    id: str
    username: str
    role: str  # 'admin', 'user', 'guest'
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class SecurityEvent:
    """Событие безопасности."""
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime
    details: Dict[str, Any]

class RateLimiter:
    """Ограничитель частоты запросов."""

    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, identifier: str) -> bool:
        """Проверяет, разрешен ли запрос."""
        current_time = time.time()
        window_start = current_time - 60  # 1 минута

        with self.lock:
            # Очистка старых запросов
            self.requests[identifier] = [
                timestamp for timestamp in self.requests[identifier]
                if timestamp > window_start
            ]

            # Проверяем burst limit (запросы в последнюю секунду)
            recent_requests = [
                timestamp for timestamp in self.requests[identifier]
                if timestamp > current_time - 1  # Последняя секунда
            ]

            if len(recent_requests) >= self.burst_limit:
                return False

            # Проверяем общий лимит на минуту
            if len(self.requests[identifier]) >= self.requests_per_minute:
                return False

            # Добавляем новый запрос
            self.requests[identifier].append(current_time)
            return True

    def get_remaining_requests(self, identifier: str) -> int:
        """Возвращает количество оставшихся запросов."""
        current_time = time.time()
        window_start = current_time - 60

        with self.lock:
            recent_requests = [
                timestamp for timestamp in self.requests[identifier]
                if timestamp > window_start
            ]
            return max(0, self.requests_per_minute - len(recent_requests))

class AuthenticationManager:
    """Менеджер аутентификации."""

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

        # Создание дефолтного администратора
        self._create_default_admin()

    def _create_default_admin(self):
        """Создает дефолтного администратора."""
        admin_user = User(
            id="admin",
            username="admin",
            role="admin"
        )
        self.users["admin"] = admin_user
        logger.info("Создан дефолтный администратор")

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Аутентифицирует пользователя."""
        with self.lock:
            user = self.users.get(username)
            if not user or not user.is_active:
                return None

            # Простая проверка пароля (в продакшене использовать bcrypt)
            if self._verify_password(password, user):
                session_token = self._create_session(user)
                user.last_login = datetime.now()
                logger.info(f"Пользователь {username} аутентифицирован")
                return session_token

            return None

    def _verify_password(self, password: str, user: User) -> bool:
        """Проверяет пароль пользователя."""
        # Для демо используем простой хэш
        # В продакшене использовать bcrypt или argon2
        expected_hash = hashlib.sha256(f"{user.username}:{password}".encode()).hexdigest()
        stored_hash = getattr(user, 'password_hash', '')
        return expected_hash == stored_hash

    def _create_session(self, user: User) -> str:
        """Создает сессию для пользователя."""
        session_token = secrets.token_hex(32)
        self.sessions[session_token] = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=24)
        }
        return session_token

    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Валидирует сессию."""
        with self.lock:
            session = self.sessions.get(session_token)
            if not session:
                return None

            if datetime.now() > session['expires_at']:
                del self.sessions[session_token]
                return None

            return session

    def logout(self, session_token: str) -> bool:
        """Завершает сессию."""
        with self.lock:
            if session_token in self.sessions:
                del self.sessions[session_token]
                return True
            return False

    def create_user(self, username: str, password: str, role: str = "user") -> Optional[User]:
        """Создает нового пользователя."""
        with self.lock:
            if username in self.users:
                return None

            user = User(
                id=username,
                username=username,
                role=role
            )
            # Установка пароля (демо версия)
            user.password_hash = hashlib.sha256(f"{username}:{password}".encode()).hexdigest()

            self.users[username] = user
            logger.info(f"Создан пользователь {username} с ролью {role}")
            return user

class AuthorizationManager:
    """Менеджер авторизации."""

    def __init__(self):
        self.permissions = {
            'admin': ['read', 'write', 'delete', 'admin', 'system'],
            'user': ['read', 'write', 'chat'],
            'guest': ['read', 'chat']
        }

    def check_permission(self, user_role: str, permission: str) -> bool:
        """Проверяет разрешение пользователя."""
        if user_role not in self.permissions:
            return False
        return permission in self.permissions[user_role]

    def get_user_permissions(self, user_role: str) -> List[str]:
        """Возвращает все разрешения пользователя."""
        return self.permissions.get(user_role, [])

class SecurityManager:
    """Основной менеджер безопасности."""

    def __init__(self):
        self.auth_manager = AuthenticationManager()
        self.authz_manager = AuthorizationManager()
        self.rate_limiter = RateLimiter()
        self.event_log: List[SecurityEvent] = []
        self.lock = threading.Lock()

    def authenticate_request(self, username: str, password: str, ip_address: str,
                           user_agent: str) -> Tuple[Optional[str], str]:
        """Аутентифицирует запрос пользователя."""
        # Проверка rate limit
        if not self.rate_limiter.is_allowed(ip_address):
            self._log_event("rate_limit_exceeded", None, ip_address, user_agent,
                          {"reason": "too_many_requests"})
            return None, "rate_limit_exceeded"

        # Аутентификация
        session_token = self.auth_manager.authenticate(username, password)

        if session_token:
            self._log_event("authentication_success", username, ip_address, user_agent, {})
            return session_token, "success"
        else:
            self._log_event("authentication_failed", username, ip_address, user_agent,
                          {"reason": "invalid_credentials"})
            return None, "authentication_failed"

    def authorize_request(self, session_token: str, required_permission: str,
                         ip_address: str, user_agent: str) -> Tuple[bool, str]:
        """Авторизует запрос."""
        # Валидация сессии
        session = self.auth_manager.validate_session(session_token)
        if not session:
            self._log_event("authorization_failed", None, ip_address, user_agent,
                          {"reason": "invalid_session"})
            return False, "invalid_session"

        user_role = session['role']
        username = session['username']

        # Проверка разрешения
        if self.authz_manager.check_permission(user_role, required_permission):
            self._log_event("authorization_success", username, ip_address, user_agent,
                          {"permission": required_permission})
            return True, "success"
        else:
            self._log_event("authorization_failed", username, ip_address, user_agent,
                          {"reason": "insufficient_permissions", "permission": required_permission})
            return False, "insufficient_permissions"

    def validate_request(self, session_token: Optional[str], required_permission: str,
                        ip_address: str, user_agent: str) -> Tuple[bool, str]:
        """Комплексная валидация запроса."""
        # Проверка rate limit
        if not self.rate_limiter.is_allowed(ip_address):
            return False, "rate_limit_exceeded"

        # Если требуется аутентификация
        if required_permission != "guest":
            if not session_token:
                return False, "authentication_required"

            authorized, reason = self.authorize_request(
                session_token, required_permission, ip_address, user_agent
            )
            return authorized, reason

        # Для гостевых запросов только rate limit
        return True, "success"

    def _log_event(self, event_type: str, user_id: Optional[str],
                   ip_address: str, user_agent: str, details: Dict[str, Any]):
        """Логирует событие безопасности."""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(),
            details=details
        )

        with self.lock:
            self.event_log.append(event)

        logger.warning(f"Security event: {event_type} for user {user_id} from {ip_address}")

    def get_security_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Возвращает последние события безопасности."""
        with self.lock:
            return self.event_log[-limit:]

    def get_rate_limit_status(self, identifier: str) -> Dict[str, Any]:
        """Возвращает статус rate limit для идентификатора."""
        remaining = self.rate_limiter.get_remaining_requests(identifier)
        return {
            "remaining_requests": remaining,
            "total_limit": self.rate_limiter.requests_per_minute,
            "is_allowed": remaining > 0
        }

# Глобальный экземпляр менеджера безопасности
security_manager = SecurityManager()

def get_security_manager() -> SecurityManager:
    """Возвращает глобальный менеджер безопасности."""
    return security_manager

# Вспомогательные функции для интеграции
def require_authentication(permission: str = "user"):
    """Декоратор для защиты функций аутентификацией."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Получение session_token из аргументов (зависит от реализации)
            session_token = kwargs.get('session_token')
            ip_address = kwargs.get('ip_address', '127.0.0.1')
            user_agent = kwargs.get('user_agent', 'unknown')

            authorized, reason = security_manager.validate_request(
                session_token, permission, ip_address, user_agent
            )

            if not authorized:
                raise PermissionError(f"Access denied: {reason}")

            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_security_event(event_type: str, user_id: Optional[str] = None,
                      ip_address: str = "127.0.0.1", user_agent: str = "unknown",
                      details: Optional[Dict[str, Any]] = None):
    """Логирует событие безопасности."""
    if details is None:
        details = {}
    security_manager._log_event(event_type, user_id, ip_address, user_agent, details)
