"""
Knowledge API маршрут для Web GUI ЕВА
"""
import logging
import uuid
from datetime import datetime
from flask import jsonify, request

logger = logging.getLogger("eva.webgui")


def register_routes(app, web_gui_instance):

    @app.route('/api/knowledge', methods=['GET', 'POST', 'DELETE'])
    def api_knowledge():
        """Управление базой знаний."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        if request.method == 'GET':
            action = request.args.get('action', 'list')
            query = request.args.get('q', '')
            limit = int(request.args.get('limit', 50))

            knowledge_items = []

            try:
                brain = web_gui_instance.brain

                if action == 'list':
                    if brain and hasattr(brain, 'knowledge_graph') and brain.knowledge_graph:
                        kg = brain.knowledge_graph
                        if hasattr(kg, 'nodes'):
                            for node in list(kg.nodes)[:limit]:
                                knowledge_items.append({
                                    'id': getattr(node, 'id', str(uuid.uuid4())),
                                    'name': getattr(node, 'name', '')[:100],
                                    'content': getattr(node, 'content', '')[:500],
                                    'type': getattr(node, 'node_type', getattr(node, 'type', 'concept')),
                                    'created_at': getattr(node, 'created_at', '')
                                })

                    if brain and hasattr(brain, 'memory_manager') and brain.memory_manager:
                        mm = brain.memory_manager
                        if hasattr(mm, 'get_recent_interactions'):
                            try:
                                interactions = mm.get_recent_interactions(limit=limit)
                                if interactions:
                                    for interaction in interactions:
                                        knowledge_items.append({
                                            'id': interaction.get('id', str(uuid.uuid4())),
                                            'name': 'Interaction',
                                            'content': str(interaction.get('content', ''))[:500],
                                            'type': 'interaction',
                                            'created_at': interaction.get('timestamp', '')
                                        })
                            except Exception:
                                pass

                elif action == 'search' and query:
                    if brain and hasattr(brain, 'knowledge_graph') and brain.knowledge_graph:
                        kg = brain.knowledge_graph
                        if hasattr(kg, 'search_nodes'):
                            results = kg.search_nodes(query, limit=limit)
                            if isinstance(results, list):
                                knowledge_items = results
                            elif isinstance(results, dict):
                                knowledge_items = results.get('results', results.get('nodes', []))
                    else:
                        for session in web_gui_instance.session_manager.sessions.values():
                            for node in session.get('context_nodes', []):
                                text = node.get('user_message', '') + node.get('assistant_message', '')
                                if query.lower() in text.lower():
                                    knowledge_items.append({
                                        'id': node.get('id', str(uuid.uuid4())),
                                        'name': 'Context Node',
                                        'content': text[:500],
                                        'type': 'context',
                                        'created_at': node.get('timestamp', '')
                                    })
                                    if len(knowledge_items) >= limit:
                                        break

            except Exception as e:
                logger.error(f"Error getting knowledge: {e}")

            return jsonify({'knowledge': knowledge_items, 'total': len(knowledge_items)})

        if request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON'}), 400

            name = data.get('name', '')
            content = data.get('content', '')
            node_type = data.get('type', 'concept')
            session_id = data.get('session_id')

            if not name and not content:
                return jsonify({'error': 'name или content обязательны'}), 400

            try:
                brain = web_gui_instance.brain

                if brain and hasattr(brain, 'knowledge_graph') and brain.knowledge_graph:
                    kg = brain.knowledge_graph
                    if hasattr(kg, 'add_node'):
                        kg.add_node(name=name, content=content, node_type=node_type)
                        return jsonify({'status': 'ok', 'message': 'Запись добавлена в граф знаний'})

                if session_id:
                    session = web_gui_instance.session_manager.get_session(session_id)
                    if session:
                        knowledge_node = {
                            'id': str(uuid.uuid4()),
                            'name': name,
                            'content': content,
                            'type': node_type,
                            'timestamp': datetime.now().isoformat()
                        }
                        web_gui_instance.session_manager.add_context_node(session_id, knowledge_node)
                        return jsonify({'status': 'ok', 'message': 'Запись добавлена в сессию', 'node_id': knowledge_node['id']})
                    else:
                        return jsonify({'error': 'Сессия не найдена'}), 404

                return jsonify({'error': 'Граф знаний недоступен и session_id не указан'}), 400

            except Exception as e:
                logger.error(f"Error adding knowledge: {e}")
                return jsonify({'error': str(e)}), 500

        if request.method == 'DELETE':
            data = request.get_json() or {}
            knowledge_id = data.get('id')
            node_type = data.get('type')
            session_id = data.get('session_id')

            if not knowledge_id and not node_type and not session_id:
                return jsonify({'error': 'Укажите id, type или session_id для удаления'}), 400

            try:
                removed = 0

                if session_id:
                    session = web_gui_instance.session_manager.get_session(session_id)
                    if session:
                        nodes = session.get('context_nodes', [])
                        if knowledge_id:
                            nodes = [n for n in nodes if n.get('id') != knowledge_id]
                            removed = 1
                        elif node_type:
                            original = len(nodes)
                            nodes = [n for n in nodes if n.get('type') != node_type]
                            removed = original - len(nodes)
                        else:
                            removed = len(nodes)
                            nodes = []
                        web_gui_instance.session_manager.update_session(session_id, {'context_nodes': nodes})
                        return jsonify({'status': 'ok', 'removed': removed, 'message': f'Удалено {removed} записей'})
                    else:
                        return jsonify({'error': 'Сессия не найдена'}), 404

                if knowledge_id and web_gui_instance.brain:
                    kg = getattr(web_gui_instance.brain, 'knowledge_graph', None)
                    if kg and hasattr(kg, 'remove_node'):
                        kg.remove_node(knowledge_id)
                        return jsonify({'status': 'ok', 'message': 'Запись удалена из графа знаний'})

                return jsonify({'error': 'Не удалось удалить запись'}), 400

            except Exception as e:
                logger.error(f"Error deleting knowledge: {e}")
                return jsonify({'error': str(e)}), 500
