"""
Wikipedia Knowledge Base API маршрут для Web GUI ЕВА
"""
import logging
import os
import json
import time
import requests
from datetime import datetime, date
from flask import jsonify, request

logger = logging.getLogger("eva.webgui")

TAVILY_API_URL = "https://api.tavily.com/search"
MONTHLY_SEARCH_LIMIT = 1000
QUOTA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'cache', 'search_quota.json')
MIN_SEARCH_INTERVAL = 5  # seconds between searches
_last_search_time = 0


def load_brain_config():
    config_path = os.path.join(os.getcwd(), 'brain_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_quota():
    today = date.today().strftime("%Y-%m")
    if os.path.exists(QUOTA_FILE):
        try:
            with open(QUOTA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get("month") != today:
                data = {"month": today, "used": 0}
                save_quota(data)
            return data
        except Exception:
            pass
    data = {"month": today, "used": 0}
    save_quota(data)
    return data


def save_quota(data):
    try:
        os.makedirs(os.path.dirname(QUOTA_FILE), exist_ok=True)
        with open(QUOTA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения квоты: {e}")


def check_quota():
    quota = load_quota()
    remaining = MONTHLY_SEARCH_LIMIT - quota["used"]
    return remaining > 0, remaining, quota["used"]


def increment_quota():
    quota = load_quota()
    quota["used"] += 1
    save_quota(quota)


def tavily_search(query: str, max_results: int = 5):
    config = load_brain_config()
    api_key = config.get('tavily_api_key') or os.environ.get('TAVILY_API_KEY')
    if not api_key:
        return {"error": "Tavily API key не найден", "results": []}

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {"query": query, "max_results": max_results, "search_depth": "basic"}

    try:
        response = requests.post(TAVILY_API_URL, headers=headers, json=data, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Tavily API error: {response.status_code} - {response.text}")
            return {"error": f"API error: {response.status_code}", "results": []}
    except requests.exceptions.Timeout:
        return {"error": "API timeout", "results": []}
    except Exception as e:
        logger.error(f"Tavily API exception: {e}")
        return {"error": str(e), "results": []}


def register_routes(app, web_gui_instance):

    @app.route('/api/wikipedia', methods=['GET', 'POST'])
    def api_wikipedia():
        """Wikipedia Knowledge Base API — только поиск."""
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
            data = request.get_json() or {}
            action = data.get('action', 'search')

            if action == 'search':
                global _last_search_time
                now = time.time()
                if now - _last_search_time < MIN_SEARCH_INTERVAL:
                    wait = int(MIN_SEARCH_INTERVAL - (now - _last_search_time))
                    return jsonify({'error': f'Слишком частые запросы. Подождите {wait} сек.'}), 429

                query = data.get('query', '').strip()
                limit = int(data.get('limit', 5))
                if not query:
                    return jsonify({'error': 'Укажите запрос'}), 400
                if len(query) < 3:
                    return jsonify({'error': 'Запрос слишком короткий. Сформулируйте точнее.'}), 400
                words = query.split()
                if len(words) > 15:
                    return jsonify({'error': 'Запрос слишком длинный. Максимум 15 слов.'}), 400

                available, remaining, used = check_quota()
                if not available:
                    return jsonify({
                        'error': f'Лимит поиска исчерпан ({MONTHLY_SEARCH_LIMIT}/месяц). Использовано: {used}',
                        'quota_exhausted': True
                    }), 429

                wiki_results = []
                tavily_results = []
                tavily_success = False

                if wiki_kb:
                    try:
                        wiki_results = wiki_kb.search(query, limit=limit)
                    except Exception as e:
                        logger.warning(f"Ошибка поиска в Wikipedia KB: {e}")

                try:
                    tavily_result = tavily_search(query, max_results=limit)
                    if "error" not in tavily_result:
                        tavily_success = True
                        for r in tavily_result.get("results", []):
                            tavily_results.append({
                                "title": r.get("title", ""),
                                "url": r.get("url", ""),
                                "text": r.get("content", r.get("snippet", "")),
                                "score": r.get("score", 0.8),
                                "source": "tavily"
                            })
                except Exception as e:
                    logger.warning(f"Ошибка Tavily поиска: {e}")

                if tavily_success:
                    increment_quota()
                    _last_search_time = time.time()
                
                new_quota = load_quota()

                return jsonify({
                    'query': query,
                    'wikipedia_results': wiki_results,
                    'tavily_results': tavily_results,
                    'wiki_count': len(wiki_results),
                    'tavily_count': len(tavily_results),
                    'quota_remaining': MONTHLY_SEARCH_LIMIT - new_quota["used"],
                    'quota_used': new_quota["used"]
                })

            return jsonify({'error': 'Неизвестное действие. Доступно только: search'}), 400
