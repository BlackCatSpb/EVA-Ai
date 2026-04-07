from functools import wraps
from flask import jsonify, request

API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"

def api_version(version: str):
    """Decorator for API endpoint versioning."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            client_version = request.headers.get('X-API-Version') or request.args.get('version', '1.0.0')
            
            if _is_compatible(client_version, version):
                return func(*args, **kwargs)
            else:
                return jsonify({
                    'error': 'version_incompatible',
                    'message': f'API version {client_version} not compatible with {version}',
                    'supported_version': version
                }), 426
        return wrapper
    return decorator

def _is_compatible(client: str, server: str) -> bool:
    """Check version compatibility."""
    try:
        client_major = int(client.split('.')[0])
        server_major = int(server.split('.')[0])
        return client_major >= server_major
    except:
        return True
