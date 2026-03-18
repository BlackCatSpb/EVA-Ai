"""
Модуль безопасности CogniFlex.

Содержит классы для управления аутентификацией, авторизацией и безопасностью.
"""

from .security_framework import (
    SecurityManager,
    AuthenticationManager,
    AuthorizationManager,
    RateLimiter,
    User,
    SecurityEvent,
    security_manager,
    get_security_manager,
    log_security_event,
)

__all__ = [
    'SecurityManager',
    'AuthenticationManager',
    'AuthorizationManager',
    'RateLimiter',
    'User',
    'SecurityEvent',
    'security_manager',
    'get_security_manager',
    'log_security_event',
]
