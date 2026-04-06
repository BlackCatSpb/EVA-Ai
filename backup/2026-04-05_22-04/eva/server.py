"""Flask сервер для Web GUI ЕВА с аутентификацией и сессиями.

Модуль был разделён на:
- server_main.py — Main server class, initialization, lifecycle
- server_routes.py — API routes, handlers
- server_handlers.py — Request handlers, response formatting
"""
from .server_main import (
    app,
    web_gui_instance,
    SessionManager,
    AuthManager,
    EntityExtractor,
    EthicsChecker,
    WebGUI,
    create_app,
    get_app,
    extract_text_from_file,
)

__all__ = [
    'app',
    'web_gui_instance',
    'SessionManager',
    'AuthManager',
    'EntityExtractor',
    'EthicsChecker',
    'WebGUI',
    'create_app',
    'get_app',
    'extract_text_from_file',
]
