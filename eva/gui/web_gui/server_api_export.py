"""
Export/Import API маршруты для Web GUI ЕВА
"""
import logging
import json
import csv
import io
from datetime import datetime
from flask import jsonify, request

logger = logging.getLogger("eva.webgui")


def register_routes(app, web_gui_instance):

    @app.route('/api/export', methods=['POST'])
    def api_export():
        """Экспорт данных сессий, сущностей и контекста."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        try:
            data = request.get_json() or {}
            session_id = data.get('session_id')
            export_format = data.get('format', 'json')
            user_id = request.headers.get('X-User-ID')

            export_data = {
                'exported_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }

            if session_id:
                session = web_gui_instance.session_manager.get_session(session_id)
                if not session:
                    return jsonify({'error': 'Сессия не найдена'}), 404
                if user_id and session.get('user_id') != user_id:
                    return jsonify({'error': 'Доступ запрещён'}), 403
                export_data['session'] = {
                    'id': session['id'],
                    'name': session['name'],
                    'created_at': session['created_at'],
                    'chat_history': session.get('chat_history', []),
                    'context_nodes': session.get('context_nodes', []),
                    'entities': session.get('entities', [])
                }
            else:
                sessions = web_gui_instance.session_manager.get_user_sessions(user_id) if user_id else []
                export_data['sessions'] = sessions
                export_data['total_sessions'] = len(sessions)

            if export_format == 'json':
                response = app.response_class(
                    response=json.dumps(export_data, ensure_ascii=False, indent=2),
                    status=200,
                    mimetype='application/json'
                )
                response.headers['Content-Disposition'] = f'attachment; filename="eva_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"'
                return response

            elif export_format == 'csv':
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(['session_id', 'name', 'created_at', 'role', 'content', 'timestamp'])

                sessions_to_export = []
                if session_id:
                    session = web_gui_instance.session_manager.get_session(session_id)
                    if session:
                        sessions_to_export = [session]
                else:
                    sessions_to_export = web_gui_instance.session_manager.get_user_sessions(user_id) if user_id else []

                for sess in sessions_to_export:
                    for msg in sess.get('chat_history', []):
                        writer.writerow([
                            sess['id'],
                            sess['name'],
                            sess['created_at'],
                            msg.get('role', ''),
                            msg.get('content', ''),
                            msg.get('timestamp', '')
                        ])

                response = app.response_class(
                    response=output.getvalue(),
                    status=200,
                    mimetype='text/csv'
                )
                response.headers['Content-Disposition'] = f'attachment; filename="eva_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
                return response

            return jsonify({'error': f'Неподдерживаемый формат: {export_format}'}), 400

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/import', methods=['POST'])
    def api_import():
        """Импорт данных сессий из JSON файла."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        try:
            user_id = request.headers.get('X-User-ID')
            imported_count = 0
            errors = []

            if 'file' in request.files:
                file = request.files['file']
                if file.filename and file.filename.endswith('.json'):
                    try:
                        import_data = json.loads(file.read().decode('utf-8'))
                    except Exception as e:
                        return jsonify({'error': f'Ошибка чтения JSON: {e}'}), 400
                else:
                    return jsonify({'error': 'Требуется JSON файл'}), 400
            else:
                import_data = request.get_json()
                if not import_data:
                    return jsonify({'error': 'Нет данных для импорта'}), 400

            if 'session' in import_data:
                session_data = import_data['session']
                new_session_id = web_gui_instance.session_manager.create_session(
                    user_id or session_data.get('user_id', 'imported'),
                    session_data.get('name', 'Импортированная сессия')
                )
                update_data = {}
                if 'chat_history' in session_data:
                    update_data['chat_history'] = session_data['chat_history']
                if 'context_nodes' in session_data:
                    update_data['context_nodes'] = session_data['context_nodes']
                if 'entities' in session_data:
                    update_data['entities'] = session_data['entities']
                if update_data:
                    web_gui_instance.session_manager.update_session(new_session_id, update_data)
                imported_count += 1

            elif 'sessions' in import_data:
                for sess in import_data['sessions']:
                    new_id = web_gui_instance.session_manager.create_session(
                        user_id or sess.get('user_id', 'imported'),
                        sess.get('name', 'Импортированная сессия')
                    )
                    update_data = {}
                    for field in ['chat_history', 'context_nodes', 'entities']:
                        if field in sess:
                            update_data[field] = sess[field]
                    if update_data:
                        web_gui_instance.session_manager.update_session(new_id, update_data)
                    imported_count += 1
            else:
                return jsonify({'error': 'Неизвестный формат импорта. Ожидается "session" или "sessions"'}), 400

            return jsonify({
                'status': 'ok',
                'imported_count': imported_count,
                'errors': errors,
                'message': f'Импортировано {imported_count} сессий'
            })

        except Exception as e:
            logger.error(f"Error importing data: {e}")
            return jsonify({'error': str(e)}), 500
