"""
Auth routes для Web GUI
Login, authentication
"""
import logging
from flask import jsonify, request
from eva_ai.core.api_compat import API_VERSION, API_PREFIX

logger = logging.getLogger("eva_ai.webgui")


def register_auth_routes(app, web_gui_instance):
    """Регистрирует auth роуты"""
    
    @app.route('/api/login', methods=['POST'])
    def api_login():
        """Login endpoint."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        username = data.get('username', '')
        password = data.get('password', '')
        
        logger.info(f"Login attempt for user: {username}")
        
        try:
            # Check credentials
            if web_gui_instance.auth_manager:
                success = web_gui_instance.auth_manager.authenticate(username, password)
                if success:
                    # Create session
                    session_id = web_gui_instance.session_manager.create_session(username)
                    return jsonify({
                        'success': True,
                        'session_id': session_id,
                        'username': username
                    })
            
            return jsonify({'error': 'Invalid credentials'}), 401
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/debug/auth', methods=['GET', 'POST'])
    def api_debug_auth():
        """Debug login - shows detailed auth process."""
        logger.info("=== DEBUG LOGIN REQUEST ===")
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        username = data.get('username', '')
        password = data.get('password', '')
        
        result = {
            'success': False,
            'step': 'start',
            'details': {},
            'username': username
        }
        
        try:
            # Step 1: Check web_gui_instance
            result['step'] = 'check_instance'
            result['details']['instance_exists'] = web_gui_instance is not None
            
            if not web_gui_instance:
                result['details']['error'] = 'Web GUI instance is None'
                return jsonify(result)
            
            # Step 2: Check auth_manager
            result['step'] = 'check_auth_manager'
            result['details']['auth_manager_exists'] = hasattr(web_gui_instance, 'auth_manager')
            
            if not hasattr(web_gui_instance, 'auth_manager'):
                result['details']['error'] = 'Auth manager not found'
                return jsonify(result)
            
            auth_manager = web_gui_instance.auth_manager
            result['details']['auth_manager_type'] = type(auth_manager).__name__
            
            # Step 3: Check users
            result['step'] = 'check_users'
            result['details']['users_count'] = len(auth_manager.users) if hasattr(auth_manager, 'users') else 0
            result['details']['users'] = list(auth_manager.users.keys()) if hasattr(auth_manager, 'users') else []
            
            # Step 4: Authenticate
            result['step'] = 'authenticate'
            success = auth_manager.authenticate(username, password)
            result['details']['auth_result'] = success
            result['success'] = success
            
            if success:
                result['step'] = 'complete'
            else:
                result['details']['error'] = 'Authentication failed'
            
        except Exception as e:
            result['details']['error'] = str(e)
            result['details']['exception_type'] = type(e).__name__
            logger.error(f"Debug login error: {e}", exc_info=True)
        
        return jsonify(result)
    
    logger.info("Auth routes registered")
