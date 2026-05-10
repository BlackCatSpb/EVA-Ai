"""
Flask сервер для Web GUI ЕВА с аутентификацией и сессиями

Этот модуль является точкой входа для обратной совместимости.
Вся логика разбита на модули:
  - server_auth.py       : SessionManager, AuthManager, EntityExtractor, EthicsChecker
  - server_routes.py     : Основные маршруты (/, /api/status, /api/system, /api/login, /api/chat, /api/sessions, ...)
  - server_api_wikipedia.py : /api/wikipedia
  - server_api_knowledge.py : /api/knowledge
  - server_api_export.py    : /api/export, /api/import
  - server_models.py     : /api/model-status, /api/fractal-graph
  - server_main.py       : Flask app, WebGUI, create_app, get_app
"""

from eva_ai.gui.web_gui.server_main import (
    app,
    WebGUI,
    web_gui_instance,
    create_app,
    get_app,
)

from eva_ai.gui.web_gui.server_auth import (
    SessionManager,
    AuthManager,
    EntityExtractor,
    EthicsChecker,
)

from eva_ai.gui.web_gui.server_routes_utils import (
    extract_text_from_file,
)

__all__ = [
    'app',
    'WebGUI',
    'web_gui_instance',
    'create_app',
    'get_app',
    'SessionManager',
    'AuthManager',
    'EntityExtractor',
    'EthicsChecker',
    'extract_text_from_file',
]
