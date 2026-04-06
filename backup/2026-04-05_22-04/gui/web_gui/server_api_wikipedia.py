"""
Wikipedia Knowledge Base API маршрут для Web GUI ЕВА
"""
import logging
from flask import jsonify, request

logger = logging.getLogger("eva.webgui")


def register_routes(app, web_gui_instance):

    @app.route('/api/wikipedia', methods=['GET', 'POST'])
    def api_wikipedia():
        """Wikipedia Knowledge Base API."""
        if not web_gui_instance:
            return jsonify({'error': 'Сервер не инициализирован'}), 500

        if not web_gui_instance.brain:
            return jsonify({'enabled': False, 'articles': 0, 'categories': 0, 'message': 'Brain not connected'})

        brain = web_gui_instance.brain
        wiki_kb = getattr(brain, 'wikipedia_kb', None)
        wiki_loader = getattr(brain, 'wikipedia_loader', None)

        if request.method == 'GET':
            action = request.args.get('action', 'stats')

            if action == 'stats':
                stats = {'enabled': wiki_kb is not None}
                if wiki_kb:
                    stats.update(wiki_kb.get_stats())
                if wiki_loader:
                    stats['loader'] = wiki_loader.get_stats()
                return jsonify(stats)

            elif action == 'search':
                if not wiki_kb:
                    return jsonify({'error': 'Wikipedia KB не инициализирована'}), 400
                query = request.args.get('query', '')
                limit = int(request.args.get('limit', 5))
                results = wiki_kb.search(query, limit=limit)
                return jsonify({'query': query, 'results': results, 'count': len(results)})

            elif action == 'article':
                if not wiki_kb:
                    return jsonify({'error': 'Wikipedia KB не инициализирована'}), 400
                article_id = request.args.get('id', '')
                article = wiki_kb.get_article(article_id)
                if article:
                    return jsonify(article)
                return jsonify({'error': 'Статья не найдена'}), 404

        elif request.method == 'POST':
            if not wiki_loader:
                return jsonify({'error': 'Wikipedia Loader не инициализирован'}), 400

            data = request.get_json() or {}
            action = data.get('action', 'load_topic')

            if action == 'load_topic':
                topic = data.get('topic', '')
                limit = int(data.get('limit', 10))
                if not topic:
                    return jsonify({'error': 'Укажите тему'}), 400
                result = wiki_loader.load_topic(topic, limit)
                return jsonify({'action': 'load_topic', 'topic': topic, 'result': result})

            elif action == 'load_category':
                category = data.get('category', '')
                limit = int(data.get('limit', 20))
                if not category:
                    return jsonify({'error': 'Укажите категорию'}), 400
                result = wiki_loader.load_category(category, limit)
                return jsonify({'action': 'load_category', 'category': category, 'result': result})

            elif action == 'load_random':
                limit = int(data.get('limit', 10))
                result = wiki_loader.load_random(limit)
                return jsonify({'action': 'load_random', 'result': result})

            elif action == 'start_auto_learn':
                if wiki_loader and not wiki_loader._running:
                    wiki_config = brain.config.get('wikipedia', {})
                    wiki_loader.start_auto_learning(
                        categories=wiki_config.get('categories', ['Наука']),
                        articles_per_category=wiki_config.get('articles_per_category', 10),
                        interval_hours=wiki_config.get('interval_hours', 24),
                        include_random=wiki_config.get('random_per_cycle', 5),
                    )
                    return jsonify({'status': 'started'})
                return jsonify({'status': 'already_running'})

            elif action == 'stop_auto_learn':
                if wiki_loader:
                    wiki_loader.stop_auto_learning()
                    return jsonify({'status': 'stopped'})
                return jsonify({'status': 'not_running'})

            return jsonify({'error': 'Неизвестное действие'}), 400
